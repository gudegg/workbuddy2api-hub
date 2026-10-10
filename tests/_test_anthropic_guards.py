"""Anthropic routes must share chat guards and request accounting.

Reuse the loopback HTTP fixture by composition: its existing tests are not
inherited or discovered here. Stores and credentials remain temporary, and
only upstream transport and deterministic client-disconnect output are fake.
"""
import io
import json
import os
import sys
import threading
import unittest
from email.message import Message
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _test_anthropic_fork_http as http_fixture

P = http_fixture.P
wb_accounts = http_fixture.wb_accounts


def partial_sse():
    chunk = {
        "choices": [{"index": 0, "delta": {"content": "Partial reply"}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13},
    }
    return ("data: " + json.dumps(chunk) + "\n\n").encode("utf-8")


class InterruptedResponse(io.BytesIO):
    def __iter__(self):
        while True:
            line = self.readline()
            if not line:
                break
            yield line
        raise OSError("fixture upstream disconnected")


class DisconnectedWriter(io.BytesIO):
    def write(self, data):
        if data.startswith(b"event: content_block_delta\n"):
            raise BrokenPipeError("fixture client disconnected")
        return super().write(data)


class HeaderDisconnectedWriter(io.BytesIO):
    def write(self, data):
        raise BrokenPipeError("fixture client disconnected during headers")


class AnthropicGuardTests(unittest.TestCase):
    def setUp(self):
        self.http = http_fixture.AnthropicHTTPTests()
        self.addCleanup(self.http.doCleanups)
        self.http.setUp()
        self.configure_key()

    def configure_key(self, models=None):
        P.wb_settings.set_api_keys(self.http.pool.dir, [{
            "id": "guard-key", "name": "guard fixture", "key": "test-api-key",
            "enabled": True, "models": models or [],
        }])

    def assert_single_row(self, outcome, tokens=None):
        rows = self.http.usage_rows()
        self.assertEqual(len(rows), 1, rows)
        self.assertEqual(rows[0]["outcome"], outcome)
        self.assertEqual(rows[0]["account"], self.http.account.uid)
        if tokens is not None:
            self.assertEqual(rows[0]["total_tokens"], tokens)
        return rows[0]

    def direct_handler(self, writer):
        payload = json.dumps({
            "model": "deepseek-v4.1-flash", "max_tokens": 32, "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        }).encode("utf-8")
        handler = object.__new__(P.Handler)
        handler.path = "/v1/messages"
        handler.headers = Message()
        handler.headers["x-api-key"] = "test-api-key"
        handler.headers["Content-Length"] = str(len(payload))
        handler.rfile = io.BytesIO(payload)
        handler.wfile = writer
        handler.requestline = "POST /v1/messages HTTP/1.1"
        handler.request_version = "HTTP/1.1"
        handler.command = "POST"
        return handler

    def test_key_model_restrictions_block_both_response_modes(self):
        self.configure_key(models=["gpt-6-astra"])
        for stream in (False, True):
            with self.subTest(stream=stream):
                status, _, body = self.http.post(stream=stream)
                self.assertEqual(status, 400, body)
                error = json.loads(body)
                self.assertEqual(error["type"], "error")
                self.assertEqual(error["error"]["type"], "invalid_request_error")
        self.assertEqual(self.http.forwarded, [])
        for row in self.http.usage_rows():
            self.assertEqual(row["outcome"], "failed")
            self.assertEqual(row["key"], "guard-key")
            self.assertEqual(row.get("total_tokens", 0), 0)

    def test_background_requests_are_blocked_on_both_anthropic_routes(self):
        payload = json.dumps({
            "model": "deepseek-v4.1-flash", "max_tokens": 32,
            "messages": [{"role": "user", "content": "hello"}],
            "client_metadata": {"request_kind": "auto_review", "thread_source": "user"},
        }).encode("utf-8")
        with mock.patch.object(P, "BLOCK_BACKGROUND_REQUESTS", True):
            for path in ("/v1/messages", "/v1/messages/count_tokens"):
                status, _, body = self.http.post(path=path, raw=payload)
                self.assertEqual(status, 400, body)
                self.assertEqual(json.loads(body)["error"]["type"], "invalid_request_error")
        self.assertEqual(self.http.forwarded, [])

    def test_allowed_model_records_request_key_in_both_response_modes(self):
        self.configure_key(models=["deepseek*"])
        for stream in (False, True):
            with self.subTest(stream=stream):
                status, _, body = self.http.post(stream=stream)
                self.assertEqual(status, 200, body)
                row = self.http.usage_rows()[-1]
                self.assertEqual(row["outcome"], "completed")
                self.assertEqual(row["key"], "guard-key")
                self.assertEqual(row["stream"], stream)
                self.assertEqual(row["total_tokens"], 20)

    def test_completed_requests_record_resolved_effort_in_both_modes(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                status, _, body = self.http.post(stream=stream)
                self.assertEqual(status, 200, body)
                self.assertEqual(self.http.forwarded[-1]["reasoning_effort"], "high")
                row = self.http.usage_rows()[-1]
                self.assertEqual(row.get("reasoning_effort"), "high")

    def test_disabled_thinking_effort_is_preserved_in_both_modes(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                raw = json.dumps({
                    "model": "deepseek-v4.1-flash", "max_tokens": 32,
                    "messages": [{"role": "user", "content": "hello"}],
                    "thinking": {"type": "disabled"}, "stream": stream,
                }).encode("utf-8")
                status, _, body = self.http.post(raw=raw)
                self.assertEqual(status, 200, body)
                self.assertEqual(self.http.usage_rows()[-1].get("reasoning_effort"), "none")

    def test_upstream_rejection_records_request_key(self):
        self.http.failure_status = 403
        status, _, body = self.http.post(stream=True)
        self.assertEqual(status, 403, body)
        self.assertEqual(json.loads(body)["type"], "error")
        row = self.assert_single_row("failed")
        self.assertEqual(row["status"], 403)
        self.assertEqual(row["key"], "guard-key")

    def test_upstream_stream_abort_records_key_and_one_terminal_outcome(self):
        def interrupted_transport(request, **kwargs):
            self.http.forwarded.append(json.loads(request.data.decode("utf-8")))
            response = InterruptedResponse(partial_sse())
            self.http.responses.append(response)
            return response

        with mock.patch.object(wb_accounts, "urlopen", side_effect=interrupted_transport):
            status, _, body = self.http.post(stream=True)
        self.assertEqual(status, 200, body)
        events = self.http.events(body)
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["error"]["type"], "api_error")
        self.assertNotIn("message_stop", [event["type"] for event in events])
        row = self.assert_single_row("upstream_aborted", tokens=13)
        self.assertEqual(row["status"], 502)
        self.assertEqual(row["key"], "guard-key")
        self.assertTrue(self.http.responses[0].closed)

    def test_client_abort_records_request_key_and_resolved_effort(self):
        # A failing writer makes disconnection deterministic on Windows too;
        # auth, dispatch, upstream handling and SSE translation still run.
        self.http.upstream_data = partial_sse()
        handler = self.direct_handler(DisconnectedWriter())
        handler.do_POST()
        row = self.assert_single_row("client_aborted", tokens=13)
        self.assertEqual(row["key"], "guard-key")
        self.assertEqual(row.get("reasoning_effort"), "high")
        self.assertTrue(self.http.responses[0].closed)

    def test_client_abort_keeps_received_text_in_fallback_accounting(self):
        text = "x" * 100
        self.http.upstream_data = ("data: " + json.dumps({
            "choices": [{"delta": {"content": text}}],
        }) + "\n\n").encode()
        self.direct_handler(DisconnectedWriter()).do_POST()
        row = self.assert_single_row("client_aborted")
        self.assertEqual(row["completion_tokens"], P.estimate_tokens(text))
        self.assertEqual(row["key"], "guard-key")

    def test_client_abort_during_headers_is_recorded(self):
        self.direct_handler(HeaderDisconnectedWriter()).do_POST()
        row = self.assert_single_row("client_aborted")
        self.assertEqual(row["key"], "guard-key")
        self.assertEqual(row.get("reasoning_effort"), "high")
        self.assertTrue(self.http.responses[0].closed)

    def test_concurrent_overload_uses_anthropic_error_envelope(self):
        semaphore = threading.BoundedSemaphore(1)
        semaphore.acquire()
        with mock.patch.object(P, "_chat_slots", semaphore), \
                mock.patch.object(P, "CHAT_SLOT_WAIT_SECONDS", 0):
            for path in ("/v1/messages", "/messages"):
                with self.subTest(path=path):
                    status, _, body = self.http.post(path=path, stream=True)
                    self.assertEqual(status, 503, body)
                    error = json.loads(body)
                    self.assertEqual(error.get("type"), "error")
                    self.assertEqual(error["error"]["type"], "overloaded_error")
        self.assertEqual(self.http.forwarded, [])
        rows = self.http.usage_rows()
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["outcome"], "failed")
            self.assertEqual(row["status"], 503)
            self.assertEqual(row["key"], "guard-key")
            self.assertEqual(row.get("total_tokens", 0), 0)
        self.assertFalse(semaphore.acquire(blocking=False))
        semaphore.release()

    def test_count_tokens_bypasses_exhausted_chat_semaphore(self):
        semaphore = threading.BoundedSemaphore(1)
        semaphore.acquire()
        payload = json.dumps({
            "model": "deepseek-v4.1-flash",
            "messages": [{"role": "user", "content": "hello"}],
        }).encode("utf-8")
        with mock.patch.object(P, "_chat_slots", semaphore), \
                mock.patch.object(P, "CHAT_SLOT_WAIT_SECONDS", 0):
            for path in ("/v1/messages/count_tokens", "/messages/count_tokens"):
                with self.subTest(path=path):
                    status, _, body = self.http.post(path=path, raw=payload)
                    self.assertEqual(status, 200, body)
                    tokens = json.loads(body)["input_tokens"]
                    self.assertIsInstance(tokens, int)
                    self.assertGreater(tokens, 0)
        self.assertEqual(self.http.forwarded, [])
        self.assertEqual(self.http.usage_rows(), [])
        self.assertFalse(semaphore.acquire(blocking=False))
        semaphore.release()

    def test_count_tokens_keeps_auth_when_chat_semaphore_is_exhausted(self):
        semaphore = threading.BoundedSemaphore(1)
        semaphore.acquire()
        with mock.patch.object(P, "_chat_slots", semaphore), \
                mock.patch.object(P, "CHAT_SLOT_WAIT_SECONDS", 0):
            status, _, body = self.http.post(path="/v1/messages/count_tokens", headers={})
        self.assertEqual(status, 401, body)
        self.assertEqual(json.loads(body)["error"]["type"], "authentication_error")
        self.assertEqual(self.http.forwarded, [])
        self.assertEqual(self.http.usage_rows(), [])
        semaphore.release()


if __name__ == "__main__":
    unittest.main()
