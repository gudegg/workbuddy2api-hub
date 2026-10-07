"""Exercise Anthropic HTTP routes with real upstream handling and translation.

Only the external transport is stubbed. These tests catch shared upstream
contract changes even when the protocol conversion helpers still pass.
All credentials, settings and usage records belong to temporary directories.
"""
import io
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
from contextlib import ExitStack
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wb_accounts
import wb_proxy as P


def upstream_sse(finish_reason="stop", tool=False):
    deltas = [{"reasoning_content": "Let me think."}]
    if tool:
        deltas.extend([
            {"tool_calls": [{"index": 0, "id": "call_weather", "type": "function",
                             "function": {"name": "weather", "arguments": '{"city":'}}]},
            {"tool_calls": [{"index": 0, "function": {"arguments": '"Shanghai"}'}}]},
        ])
    else:
        deltas.extend([{"content": "Hello "}, {"content": "world!"}])
    chunks = [{"choices": [{"index": 0, "delta": delta}]} for delta in deltas]
    chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                   "usage": {"prompt_tokens": 12, "completion_tokens": 8,
                             "total_tokens": 20}})
    return ("".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
            + "data: [DONE]\n\n").encode("utf-8")


class AnthropicHTTPTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="wb-anthropic-http-")
        self.addCleanup(temporary.cleanup)
        self.usage_log = os.path.join(temporary.name, "usage.jsonl")
        accounts_dir = os.path.join(temporary.name, "accounts")
        os.makedirs(accounts_dir)
        self.account = wb_accounts.Account({
            "uid": "test-account", "realm": "intl", "accessToken": "test-token",
            "expiresAt": 4102444800,
        })
        self.pool = wb_accounts.AccountPool(accounts_dir)
        self.pool.accounts = [self.account]
        self.upstream_data = upstream_sse()
        self.failure_status = None
        self.forwarded = []
        self.responses = []
        patches = ExitStack()
        self.addCleanup(patches.close)
        patches.enter_context(mock.patch.multiple(
            P, ACCOUNTS_DIR=accounts_dir, USAGE_DIR=temporary.name,
            USAGE_LOG=self.usage_log, POOL=self.pool, CURRENT_REALM="intl",
            API_KEY="test-api-key", SYSTEM_PROMPT=""))
        patches.enter_context(mock.patch.object(P, "auto_switch_product_enabled", return_value=False))
        # Pricing is outside this adapter's contract; keep accounting local.
        patches.enter_context(mock.patch.object(P.wb_pricing, "current_policy_id", return_value=None))
        patches.enter_context(mock.patch.object(wb_accounts, "urlopen", side_effect=self.transport))
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), P.Handler)
        self.addCleanup(self.server.server_close)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.thread.join(timeout=5)

    def transport(self, request, **kwargs):
        self.forwarded.append(json.loads(request.data.decode("utf-8")))
        if self.failure_status is not None:
            raise urllib.error.HTTPError(request.full_url, self.failure_status,
                                         "upstream rejected", {}, io.BytesIO(b"fixture rejection"))
        response = io.BytesIO(self.upstream_data)
        self.responses.append(response)
        return response

    def post(self, path="/v1/messages?beta=true", stream=False, headers=None, raw=None):
        payload = {
            "model": "deepseek-v4.1-flash", "max_tokens": 32, "stream": stream,
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"name": "weather", "input_schema": {"type": "object"}}],
        }
        body = json.dumps(payload).encode("utf-8") if raw is None else raw
        request_headers = {"Content-Type": "application/json", "x-api-key": "test-api-key"}
        if headers is not None:
            request_headers = dict(headers, **{"Content-Type": "application/json"})
        connection = HTTPConnection(*self.server.server_address, timeout=5)
        try:
            connection.request("POST", path, body=body, headers=request_headers)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read().decode("utf-8")
        finally:
            connection.close()

    def usage_rows(self):
        if not os.path.exists(self.usage_log):
            return []
        with open(self.usage_log, encoding="utf-8") as source:
            return [json.loads(line) for line in source if line.strip()]

    def events(self, body):
        events = []
        for frame in body.split("\n\n"):
            lines = frame.splitlines()
            name = next((line[7:] for line in lines if line.startswith("event: ")), None)
            data = next((line[6:] for line in lines if line.startswith("data: ")), None)
            if name and data:
                item = json.loads(data)
                self.assertEqual(item["type"], name)
                events.append(item)
        return events

    def assert_block_lifecycle(self, events):
        active = None
        next_index = 0
        for event in events:
            kind = event["type"]
            if kind == "content_block_start":
                self.assertIsNone(active, "close the previous block before starting another")
                self.assertEqual(event["index"], next_index)
                active = next_index
                next_index += 1
            elif kind in ("content_block_delta", "content_block_stop"):
                self.assertEqual(event["index"], active)
                if kind == "content_block_stop":
                    active = None
            elif kind in ("message_delta", "message_stop"):
                self.assertIsNone(active)
        self.assertIsNone(active)

    def assert_completed_usage(self, stream):
        rows = self.usage_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["outcome"], "completed")
        self.assertEqual(rows[0]["account"], self.account.uid)
        self.assertEqual(rows[0]["total_tokens"], 20)
        self.assertEqual(rows[0]["stream"], stream)
        self.assertTrue(self.responses[0].closed)

    def test_nonstreaming_response_uses_real_upstream_contract(self):
        status, headers, body = self.post(headers={"Authorization": "Bearer test-api-key"})
        self.assertEqual(status, 200, body)
        self.assertIn("application/json", headers["Content-Type"])
        message = json.loads(body)
        self.assertEqual(message["type"], "message")
        self.assertEqual(message["stop_reason"], "end_turn")
        self.assertEqual(message["content"][-1], {"type": "text", "text": "Hello world!"})
        self.assertEqual(message["usage"]["output_tokens"], 8)
        self.assertTrue(self.forwarded[0]["stream"])
        self.assertEqual(self.forwarded[0]["tools"][0]["function"]["name"], "weather")
        self.assert_completed_usage(False)

    def test_streaming_emits_real_anthropic_events(self):
        self.upstream_data = upstream_sse(finish_reason="length")
        status, headers, body = self.post(stream=True)
        self.assertEqual(status, 200, body)
        self.assertIn("text/event-stream", headers["Content-Type"])
        events = self.events(body)
        self.assert_block_lifecycle(events)
        self.assertEqual(events[0]["type"], "message_start")
        self.assertEqual(events[-1]["type"], "message_stop")
        self.assertFalse(any(event["type"] == "error" for event in events))
        deltas = [event["delta"] for event in events if event["type"] == "content_block_delta"]
        self.assertEqual("".join(delta.get("text", "") for delta in deltas), "Hello world!")
        self.assertEqual("".join(delta.get("thinking", "") for delta in deltas), "Let me think.")
        self.assertTrue(any(delta["type"] == "signature_delta" for delta in deltas))
        starts = [event["index"] for event in events if event["type"] == "content_block_start"]
        stops = [event["index"] for event in events if event["type"] == "content_block_stop"]
        self.assertEqual(starts, stops)
        final_delta = next(event for event in events if event["type"] == "message_delta")
        self.assertEqual(final_delta["delta"]["stop_reason"], "max_tokens")
        self.assertEqual(final_delta["usage"]["output_tokens"], 8)
        self.assertEqual(final_delta["usage"]["input_tokens"], 12)
        self.assert_completed_usage(True)

    def test_fragmented_tool_arguments_survive_both_response_modes(self):
        self.upstream_data = upstream_sse(finish_reason="tool_calls", tool=True)
        for stream in (False, True):
            with self.subTest(stream=stream):
                status, _, body = self.post(stream=stream)
                self.assertEqual(status, 200, body)
                if stream:
                    events = self.events(body)
                    self.assert_block_lifecycle(events)
                    tool = next(event["content_block"] for event in events
                                if event["type"] == "content_block_start"
                                and event["content_block"]["type"] == "tool_use")
                    arguments = "".join(event["delta"]["partial_json"] for event in events
                                        if event["type"] == "content_block_delta"
                                        and event["delta"]["type"] == "input_json_delta")
                    self.assertEqual(json.loads(arguments), {"city": "Shanghai"})
                    final = next(event for event in events if event["type"] == "message_delta")
                    self.assertEqual(final["delta"]["stop_reason"], "tool_use")
                else:
                    message = json.loads(body)
                    tool = next(block for block in message["content"] if block["type"] == "tool_use")
                    self.assertEqual(tool["input"], {"city": "Shanghai"})
                    self.assertEqual(message["stop_reason"], "tool_use")
                self.assertEqual(tool["name"], "weather")
                self.assertEqual(tool["id"], "call_weather")

    def test_empty_or_truncated_upstream_is_not_a_success(self):
        for stream in (False, True):
            for upstream_data in (b"", b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'):
                with self.subTest(stream=stream, upstream_data=upstream_data):
                    self.upstream_data = upstream_data
                    status, _, body = self.post(stream=stream)
                    if stream:
                        self.assertEqual(status, 200, body)
                        events = self.events(body)
                        self.assertTrue(any(event["type"] == "error" for event in events))
                        self.assertFalse(any(event["type"] == "message_stop" for event in events))
                        self.assertEqual(self.usage_rows()[-1]["outcome"], "upstream_aborted")
                    else:
                        self.assertEqual(status, 502, body)
                        self.assertEqual(json.loads(body)["error"]["type"], "api_error")
                    self.assertTrue(self.usage_rows()[-1]["error"])

    def test_finished_upstream_without_done_sentinel_is_accepted(self):
        self.upstream_data = upstream_sse().replace(b"data: [DONE]\n\n", b"")
        for stream in (False, True):
            with self.subTest(stream=stream):
                status, _, body = self.post(stream=stream)
                self.assertEqual(status, 200, body)
                if stream:
                    self.assertEqual(self.events(body)[-1]["type"], "message_stop")
                else:
                    self.assertEqual(json.loads(body)["stop_reason"], "end_turn")

    def test_parallel_tools_and_fragmented_names_emit_consecutive_blocks(self):
        chunks = [
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "call_weather", "function": {"name": "wea", "arguments": '{"city":'}},
                {"index": 1, "id": "call_time", "function": {"name": "time", "arguments": '{"zone":'}},
            ]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 1, "function": {"arguments": '"Asia/Shanghai"}'}},
                {"index": 0, "function": {"name": "ther", "arguments": '"Shanghai"}'}},
            ]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
             "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}},
        ]
        self.upstream_data = ("".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
                              + "data: [DONE]\n\n").encode()
        status, _, body = self.post(stream=True)
        self.assertEqual(status, 200, body)
        events = self.events(body)
        self.assert_block_lifecycle(events)
        starts = [event for event in events if event["type"] == "content_block_start"]
        self.assertEqual([event["content_block"]["name"] for event in starts], ["weather", "time"])
        self.assertEqual([event["content_block"]["id"] for event in starts], ["call_weather", "call_time"])
        for start, expected in zip(starts, ({"city": "Shanghai"}, {"zone": "Asia/Shanghai"})):
            arguments = "".join(event["delta"]["partial_json"] for event in events
                                if event["type"] == "content_block_delta" and event["index"] == start["index"])
            self.assertEqual(json.loads(arguments), expected)

    def test_stream_and_nonstream_report_the_same_actual_cached_usage(self):
        self.upstream_data = upstream_sse().replace(
            b'"total_tokens": 20',
            b'"total_tokens": 20, "prompt_tokens_details": {"cached_tokens": 8}')
        for stream in (False, True):
            with self.subTest(stream=stream):
                status, _, body = self.post(stream=stream)
                self.assertEqual(status, 200, body)
                if stream:
                    events = self.events(body)
                    usage = next(event["usage"] for event in events if event["type"] == "message_delta")
                else:
                    usage = json.loads(body)["usage"]
                self.assertEqual(usage["input_tokens"], 4)
                self.assertEqual(usage["cache_read_input_tokens"], 8)
                self.assertEqual(usage["cache_creation_input_tokens"], 0)
                self.assertEqual(usage["output_tokens"], 8)

    def test_upstream_error_frame_is_not_a_success(self):
        self.upstream_data = b'data: {"error":{"message":"fixture upstream error"}}\n\n'
        for stream in (False, True):
            with self.subTest(stream=stream):
                status, _, body = self.post(stream=stream)
                self.assertIn("fixture upstream error", body)
                if stream:
                    self.assertEqual(status, 200, body)
                    self.assertEqual(self.events(body)[-1]["type"], "error")
                else:
                    self.assertEqual(status, 502, body)
                self.assertTrue(self.usage_rows()[-1]["error"])

    def test_empty_tool_placeholders_do_not_request_tool_execution(self):
        self.upstream_data = (
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"","arguments":""}}]},'
            b'"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":12,"completion_tokens":8,"total_tokens":20}}\n\n'
            b'data: [DONE]\n\n'
        )
        for stream in (False, True):
            with self.subTest(stream=stream):
                status, _, body = self.post(stream=stream)
                self.assertEqual(status, 200, body)
                if stream:
                    events = self.events(body)
                    final = next(event for event in events if event["type"] == "message_delta")
                    self.assertEqual(final["delta"]["stop_reason"], "end_turn")
                    self.assertFalse(any(event["type"] == "content_block_start"
                                         and event["content_block"]["type"] == "tool_use" for event in events))
                else:
                    self.assertEqual(json.loads(body)["stop_reason"], "end_turn")

    def test_reported_zero_token_counts_are_preserved_in_both_modes(self):
        for prompt_tokens, completion_tokens in ((12, 0), (0, 8)):
            self.upstream_data = upstream_sse().replace(
                b'"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20',
                ('"prompt_tokens": %d, "completion_tokens": %d, "total_tokens": %d' % (
                    prompt_tokens, completion_tokens, prompt_tokens + completion_tokens)).encode())
            for stream in (False, True):
                with self.subTest(stream=stream, prompt=prompt_tokens, completion=completion_tokens):
                    status, _, body = self.post(stream=stream)
                    self.assertEqual(status, 200, body)
                    if stream:
                        usage = next(event["usage"] for event in self.events(body)
                                     if event["type"] == "message_delta")
                    else:
                        usage = json.loads(body)["usage"]
                    self.assertEqual(usage["input_tokens"], prompt_tokens)
                    self.assertEqual(usage["output_tokens"], completion_tokens)
                    self.assertEqual(self.usage_rows()[-1]["completion_tokens"], completion_tokens)

    def test_wrong_or_missing_key_is_rejected_before_upstream(self):
        for headers in ({}, {"x-api-key": "wrong"}):
            with self.subTest(headers=headers):
                status, _, body = self.post(headers=headers)
                self.assertEqual(status, 401, body)
                error = json.loads(body)
                self.assertEqual(error["type"], "error")
                self.assertEqual(error["error"]["type"], "authentication_error")
        self.assertEqual(self.forwarded, [])
        self.assertEqual(self.usage_rows(), [])

    def test_invalid_payload_has_anthropic_error_shape(self):
        for raw in (b"not json", b"[]", b'{"model":"deepseek-v4.1-flash","messages":[]}'):
            with self.subTest(raw=raw):
                status, _, body = self.post(raw=raw)
                self.assertEqual(status, 400, body)
                self.assertEqual(json.loads(body)["error"]["type"], "invalid_request_error")
        self.assertEqual(self.forwarded, [])

    def test_count_tokens_does_not_call_upstream(self):
        payload = json.dumps({"model": "deepseek-v4.1-flash",
                              "messages": [{"role": "user", "content": "hello"}]}).encode()
        for path in ("/v1/messages/count_tokens", "/messages/count_tokens"):
            with self.subTest(path=path):
                status, _, body = self.post(path=path, raw=payload)
                self.assertEqual(status, 200, body)
                tokens = json.loads(body)["input_tokens"]
                self.assertIsInstance(tokens, int)
                self.assertGreater(tokens, 0)
        self.assertEqual(self.forwarded, [])
        self.assertEqual(self.usage_rows(), [])

    def test_upstream_rate_limit_preserves_anthropic_error_and_retry_header(self):
        self.failure_status = 429
        status, headers, body = self.post(stream=True)
        self.assertEqual(status, 429, body)
        self.assertEqual(json.loads(body)["error"]["type"], "rate_limit_error")
        self.assertGreaterEqual(int(headers["Retry-After"]), 1)
        self.assertEqual(self.usage_rows()[0]["status"], 429)

    def test_upstream_rejection_preserves_anthropic_error(self):
        self.failure_status = 403
        status, _, body = self.post(path="/messages", stream=True)
        self.assertEqual(status, 403, body)
        self.assertEqual(json.loads(body)["type"], "error")
        self.assertIn("fixture rejection", json.loads(body)["error"]["message"])
        self.assertEqual(self.usage_rows()[0]["status"], 403)


if __name__ == "__main__":
    unittest.main()
