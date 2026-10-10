"""Anthropic conversion regressions; standard library only, no upstream I/O."""

import copy
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wb_proxy as P


def request(**extra):
    payload = {
        "model": "synthetic-model", "max_tokens": 32,
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(extra)
    return payload


def tool_request(content, is_error=False):
    return request(messages=[
        {"role": "assistant", "content": [{
            "type": "tool_use", "id": "toolu_image", "name": "screenshot", "input": {},
        }]},
        {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "toolu_image",
            "content": content, "is_error": is_error,
        }]},
    ])


def image(data="YWJj"):
    return {"type": "image", "source": {
        "type": "base64", "media_type": "image/png", "data": data,
    }}


class AnthropicConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="wb-anthropic-conversion-")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.accounts = patch.object(P, "ACCOUNTS_DIR", os.path.join(self.temp.name, "accounts"))
        self.usage = patch.object(P, "USAGE_DIR", os.path.join(self.temp.name, "usage"))
        self.network = patch.object(P.urllib.request, "urlopen", side_effect=AssertionError("network is forbidden"))
        for mocked in (self.accounts, self.usage, self.network):
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_output_effort_is_forwarded_without_clamping_when_metadata_absent(self):
        for effort in ("low", "medium", "high", "xhigh", "max"):
            with self.subTest(effort=effort):
                body = P.messages_to_chat(request(output_config={"effort": effort}))
                self.assertEqual(body.get("reasoning_effort"), effort)
                self.assertEqual(P.build_upstream_body(body).get("reasoning_effort"), effort)

    def test_zero_token_budget_survives_conversion_and_model_defaults(self):
        payload = request(model="deepseek-v4.1-flash", max_tokens=0)
        original = copy.deepcopy(payload)
        for required in (True, False):
            with self.subTest(required=required):
                chat = P.messages_to_chat(payload, require_max_tokens=required)
                self.assertEqual(chat["max_tokens"], 0)
                self.assertEqual(P.build_upstream_body(chat)["max_tokens"], 0)
        self.assertEqual(payload, original)

    def test_token_budget_rejects_negative_and_noninteger_values(self):
        for value in (-1, False, True, 0.5, "0", [], {}):
            for required in (True, False):
                with self.subTest(value=value, required=required):
                    with self.assertRaisesRegex(P.AnthropicRequestError, "max_tokens"):
                        P.messages_to_chat(request(max_tokens=value), require_max_tokens=required)
        with self.assertRaisesRegex(P.AnthropicRequestError, "max_tokens"):
            P.messages_to_chat(request(max_tokens=None))

    def test_zero_token_budget_rejects_options_that_require_output(self):
        for extra in (
            {"stream": True},
            {"thinking": {"type": "enabled", "budget_tokens": 1024}},
            {"tool_choice": {"type": "any"}},
            {"tool_choice": {"type": "tool", "name": "lookup"}},
            {"output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}}},
        ):
            with self.subTest(extra=extra):
                with self.assertRaises(P.AnthropicRequestError):
                    P.messages_to_chat(request(max_tokens=0, **extra))

    def test_zero_token_budget_accepts_nonforcing_tool_choices(self):
        for choice in (None, {"type": "auto"}, {"type": "none"}):
            with self.subTest(choice=choice):
                chat = P.messages_to_chat(request(max_tokens=0, tool_choice=choice))
                self.assertEqual(chat["max_tokens"], 0)

    def test_zero_token_response_uses_the_native_empty_response_shape(self):
        obj = {
            "model": "synthetic-model",
            "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 0, "total_tokens": 12},
        }
        message = P.chat_to_messages(obj, request(max_tokens=0))
        self.assertEqual(message["content"], [])
        self.assertEqual(message["stop_reason"], "max_tokens")
        self.assertEqual(message["usage"]["input_tokens"], 12)
        self.assertEqual(message["usage"]["output_tokens"], 0)

    def test_explicit_effort_respects_supported_and_fixed_catalog_values(self):
        for effort in ("low", "high", "max"):
            body = P.messages_to_chat(request(
                model="deepseek-v4.1-flash", output_config={"effort": effort},
            ))
            self.assertEqual(body.get("reasoning_effort"), effort)
        with self.assertRaisesRegex(P.AnthropicRequestError, "effort.*medium"):
            P.messages_to_chat(request(
                model="deepseek-v4.1-flash", output_config={"effort": "medium"},
            ))
        body = P.messages_to_chat(request(
            model="gemini-3.5-flash", output_config={"effort": "medium"},
        ))
        self.assertEqual(body.get("reasoning_effort"), "medium")
        with self.assertRaisesRegex(P.AnthropicRequestError, "effort.*max"):
            P.messages_to_chat(request(
                model="gemini-3.5-flash", output_config={"effort": "max"},
            ))

    def test_invalid_output_effort_and_configuration_are_rejected(self):
        for effort in (1, True, "", "turbo", "none"):
            with self.subTest(effort=effort):
                with self.assertRaisesRegex(P.AnthropicRequestError, "output_config.effort"):
                    P.messages_to_chat(request(output_config={"effort": effort}))
        with self.assertRaisesRegex(P.AnthropicRequestError, "output_config.*object"):
            P.messages_to_chat(request(output_config=[]))

    def test_nullable_effort_fields_are_treated_as_omitted(self):
        for extra, expected in (
            ({"output_config": {"effort": None}}, None),
            ({"reasoning_effort": None, "reasoningEffort": None}, None),
            ({"output_config": {"effort": None}, "reasoningEffort": "low"}, "low"),
            ({"reasoning_effort": None, "reasoningEffort": "medium"}, "medium"),
            ({"output_config": {"effort": "high"}, "reasoning_effort": None,
              "reasoningEffort": None}, "high"),
        ):
            with self.subTest(extra=extra):
                body = P.messages_to_chat(request(**extra))
                self.assertEqual(body.get("reasoning_effort"), expected)
                if expected is None:
                    self.assertNotIn("reasoning_effort", body)

    def test_compatibility_effort_aliases_agree_or_report_conflicts(self):
        body = P.messages_to_chat(request(
            output_config={"effort": "low"}, reasoning_effort="low", reasoningEffort="low",
        ))
        self.assertEqual(body.get("reasoning_effort"), "low")
        self.assertNotIn("reasoningEffort", body)
        body = P.messages_to_chat(request(reasoningEffort="none"))
        self.assertEqual(body.get("reasoning_effort"), "none")
        for extra in (
            {"output_config": {"effort": "high"}, "reasoning_effort": "low"},
            {"reasoning_effort": "high", "reasoningEffort": "low"},
            {"output_config": {"effort": "high"}, "thinking": {"type": "disabled"}},
        ):
            with self.subTest(extra=extra):
                with self.assertRaisesRegex(P.AnthropicRequestError, "conflict"):
                    P.messages_to_chat(request(**extra))

    def test_json_schema_output_fails_explicitly_instead_of_being_ignored(self):
        with self.assertRaisesRegex(P.AnthropicRequestError, "output_config.format.*not supported"):
            P.messages_to_chat(request(output_config={"format": {
                "type": "json_schema", "schema": {"type": "object"},
            }}))

    def test_legacy_json_schema_output_is_also_rejected(self):
        with self.assertRaisesRegex(P.AnthropicRequestError, "output_format.*not supported"):
            P.messages_to_chat(request(output_format={
                "type": "json_schema", "schema": {"type": "object"},
            }))

    def test_harmless_metadata_and_empty_output_config_remain_compatible(self):
        body = P.messages_to_chat(request(
            metadata={"user_id": "synthetic"}, output_config={}, service_tier="auto",
        ))
        self.assertEqual(body["messages"][0]["content"], "hello")

    def test_text_tool_results_keep_the_existing_string_shape(self):
        body = P.messages_to_chat(tool_request([
            {"type": "text", "text": "first"}, {"type": "text", "text": "second"},
        ], is_error=True))
        self.assertEqual(body["messages"][-1]["content"], "Tool error: firstsecond")

    def test_tool_result_images_survive_conversion_and_upstream_sanitization(self):
        payload = tool_request([
            {"type": "text", "text": "screen"}, image(),
            {"type": "image", "source": {"type": "url", "url": "https://example.test/screen.png"}},
        ])
        original = copy.deepcopy(payload)
        body = P.messages_to_chat(payload)
        expected = [
            {"type": "text", "text": "screen"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,YWJj"}},
            {"type": "image_url", "image_url": {"url": "https://example.test/screen.png"}},
        ]
        self.assertEqual(body["messages"][-1]["content"], expected)
        upstream = P.build_upstream_body(body)
        tool_message = next(message for message in upstream["messages"] if message["role"] == "tool")
        self.assertEqual(tool_message["content"], expected)
        self.assertEqual(tool_message["tool_call_id"], "toolu_image")
        self.assertEqual(payload, original)

    def test_image_tool_error_keeps_the_image_and_adds_a_text_prefix(self):
        body = P.messages_to_chat(tool_request([image()], is_error=True))
        content = body["messages"][-1]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Tool error: "})
        self.assertEqual(content[1]["type"], "image_url")

    def test_tool_result_documents_are_rejected_as_unsupported(self):
        with self.assertRaisesRegex(P.AnthropicRequestError, "unsupported.*document"):
            P.messages_to_chat(tool_request([{
                "type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "YWJj"},
            }]))

    def test_input_estimate_includes_tool_names_descriptions_and_schema(self):
        plain = P.messages_to_chat(request())
        with_tools = P.messages_to_chat(request(tools=[{
            "name": "lookup", "description": "Detailed tool instructions. " * 40,
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
        }]))
        self.assertGreater(P._anthropic_estimate_chat_tokens(with_tools), P._anthropic_estimate_chat_tokens(plain) + 100)

    def test_image_estimate_uses_fixed_cost_not_encoded_data_length(self):
        for as_tool in (False, True):
            estimates = []
            for data in ("YWJj", "a" * 200000):
                payload = tool_request([image(data)]) if as_tool else request(messages=[{
                    "role": "user", "content": [image(data)],
                }])
                body = P.messages_to_chat(payload)
                original = copy.deepcopy(body)
                estimates.append(P._anthropic_estimate_chat_tokens(body))
                self.assertEqual(body, original)
            with self.subTest(as_tool=as_tool):
                self.assertEqual(estimates[0], estimates[1])
                self.assertGreaterEqual(estimates[0], 1024)

    def test_cached_tokens_are_subtracted_from_inclusive_prompt_usage(self):
        usage = P._anthropic_usage({
            "prompt_tokens": 200, "completion_tokens": 9,
            "prompt_tokens_details": {"cached_tokens": 120},
        })
        self.assertEqual(usage, {
            "input_tokens": 80, "output_tokens": 9,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 120,
        })
        self.assertEqual(sum(usage[key] for key in (
            "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
        )), 200)

    def test_uncached_usage_always_has_zero_cache_counters(self):
        usage = P._anthropic_usage({"prompt_tokens": 12, "completion_tokens": 8})
        self.assertEqual(usage.get("cache_creation_input_tokens"), 0)
        self.assertEqual(usage.get("cache_read_input_tokens"), 0)
        self.assertEqual(usage["input_tokens"], 12)

    def test_cache_usage_aliases_follow_the_canonical_field_precedence(self):
        for extra, expected in (
            ({"prompt_cache_hit_tokens": 5}, 5),
            ({"completion_tokens_details": {"cached_tokens": 4}}, 4),
            ({"prompt_tokens_details": {"cached_tokens": 3}}, 3),
            ({"prompt_cache_hit_tokens": 5, "completion_tokens_details": {"cached_tokens": 4},
              "prompt_tokens_details": {"cached_tokens": 3}}, 3),
            ({"prompt_cache_hit_tokens": 0, "completion_tokens_details": {"cached_tokens": 4},
              "prompt_tokens_details": {"cached_tokens": 3}}, 3),
            ({"prompt_cache_hit_tokens": 0, "completion_tokens_details": {"cached_tokens": 0},
              "prompt_tokens_details": {"cached_tokens": 3}}, 3),
        ):
            with self.subTest(extra=extra):
                upstream = {"prompt_tokens": 12, "completion_tokens": 8}
                upstream.update(extra)
                usage = P._anthropic_usage(upstream)
                self.assertEqual(usage["cache_read_input_tokens"], expected)
                self.assertEqual(usage["input_tokens"], 12 - expected)
                self.assertEqual(usage["cache_read_input_tokens"], P._extract_usage(upstream)["cached_tokens"])

    def test_malformed_cache_alias_values_remain_robust(self):
        for extra in (
            {"prompt_cache_hit_tokens": "bad"},
            {"completion_tokens_details": {"cached_tokens": "bad"}},
            {"prompt_tokens_details": {"cached_tokens": "bad"}},
            {"completion_tokens_details": [], "prompt_tokens_details": "bad"},
            {"completion_tokens_details": "bad", "prompt_tokens_details": {"cached_tokens": 0}},
        ):
            with self.subTest(extra=extra):
                upstream = {"prompt_tokens": 12, "completion_tokens": 8}
                upstream.update(extra)
                original = copy.deepcopy(upstream)
                usage = P._anthropic_usage(upstream)
                self.assertEqual(usage["cache_read_input_tokens"], 0)
                self.assertEqual(usage["input_tokens"], 12)
                self.assertEqual(upstream, original)

    def test_cache_usage_is_nonnegative_and_never_exceeds_prompt_total(self):
        for cached, expected_read in ((20, 20), (30, 20), (-4, 0), ("bad", 0)):
            with self.subTest(cached=cached):
                usage = P._anthropic_usage({
                    "prompt_tokens": 20, "completion_tokens": 1,
                    "prompt_tokens_details": {"cached_tokens": cached},
                })
                self.assertEqual(usage.get("cache_read_input_tokens"), expected_read)
                self.assertEqual(usage["input_tokens"], 20 - expected_read)

    def test_missing_upstream_usage_uses_the_local_estimate(self):
        usage = P._anthropic_usage(None, input_tokens=37, output_text="hello")
        self.assertEqual(usage["input_tokens"], 37)
        self.assertGreater(usage["output_tokens"], 0)
        self.assertEqual(usage.get("cache_creation_input_tokens"), 0)
        self.assertEqual(usage.get("cache_read_input_tokens"), 0)

    def test_explicit_zero_input_and_output_usage_is_preserved(self):
        for prompt, completion in ((0, 8), (12, 0), (0, 0)):
            with self.subTest(prompt=prompt, completion=completion):
                usage = P._anthropic_usage({
                    "prompt_tokens": prompt, "completion_tokens": completion,
                    "total_tokens": prompt + completion,
                }, input_tokens=37, output_text="hello")
                self.assertEqual(usage["input_tokens"], prompt)
                self.assertEqual(usage["output_tokens"], completion)

    def test_only_missing_invalid_or_negative_usage_uses_estimates(self):
        for upstream in ({}, {"prompt_tokens": None, "completion_tokens": None},
                         {"prompt_tokens": "bad", "completion_tokens": "bad"},
                         {"prompt_tokens": -1, "completion_tokens": -1}):
            with self.subTest(upstream=upstream):
                usage = P._anthropic_usage(upstream, input_tokens=37, output_text="hello")
                self.assertEqual(usage["input_tokens"], 37)
                self.assertEqual(usage["output_tokens"], P.estimate_tokens("hello"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
