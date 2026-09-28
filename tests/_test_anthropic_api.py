"""Pure protocol tests for the Anthropic Messages adapter."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ACCOUNTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_acc_anthropic"))
os.environ.setdefault("USAGE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_use_anthropic"))

import wb_proxy as P

PASS = FAIL = 0


def check(label, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  [PASS] " + label)
    else:
        FAIL += 1
        print("  [FAIL] " + label + ("  " + str(extra) if extra else ""))


payload = {
    "model": "deepseek-v4.1-flash",
    "system": [{"type": "text", "text": "You are helpful."}],
    "max_tokens": 300,
    "messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "Find the weather."},
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": "abc"
            }},
        ]},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "I should call the weather tool."},
            {"type": "tool_use", "id": "toolu_1", "name": "weather",
             "input": {"city": "Shanghai"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1",
             "content": [{"type": "text", "text": "Sunny, 25C"}]},
        ]},
    ],
    "tools": [{
        "name": "weather",
        "description": "Get weather",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    }],
    "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
}

print("[1] Anthropic request converts to the existing Chat shape")
chat = P._anthropic_messages_to_chat(payload)
check("system becomes a system message", chat["messages"][0] == {
    "role": "system", "content": "You are helpful."
}, chat["messages"][0])
check("image becomes an OpenAI data URL",
      chat["messages"][1]["content"][1]["image_url"]["url"] == "data:image/png;base64,abc")
assistant = chat["messages"][2]
check("assistant thinking is preserved as reasoning_content",
      assistant.get("reasoning_content") == "I should call the weather tool.")
check("tool_use becomes an OpenAI tool call",
      assistant["tool_calls"][0]["function"]["name"] == "weather")
check("tool_result becomes a tool message",
      chat["messages"][3] == {
          "role": "tool", "tool_call_id": "toolu_1", "content": "Sunny, 25C"
      }, chat["messages"][3])
check("tool schema and choice are translated",
      chat["tools"][0]["function"]["parameters"]["type"] == "object" and
      chat["tool_choice"] == "auto" and chat["parallel_tool_calls"] is False)

print()
print("[2] Anthropic response and SSE envelope are valid")
chat_response = {
    "model": payload["model"],
    "choices": [{"message": {
        "role": "assistant",
        "content": "The weather is sunny.",
        "reasoning_content": "I checked the tool result.",
        "tool_calls": [{"id": "call_1", "function": {
            "name": "weather", "arguments": '{"city":"Shanghai"}'
        }}],
    }, "finish_reason": "tool_calls"}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 8},
}
message = P._anthropic_response_from_chat(chat_response, payload, input_tokens=10)
check("response has Anthropic message envelope",
      message["type"] == "message" and message["role"] == "assistant")
check("response blocks include thinking, text and tool_use",
      [block["type"] for block in message["content"]] == ["thinking", "text", "tool_use"],
      message["content"])
check("tool_use input is decoded JSON",
      message["content"][-1]["input"] == {"city": "Shanghai"})
check("finish reason maps to tool_use",
      message["stop_reason"] == "tool_use")
event = P._anthropic_sse_event("message_start", {"message": message})
lines = event.decode("utf-8").splitlines()
event_data = json.loads(lines[1][len("data: "):])
check("SSE has named event and matching data type",
      lines[0] == "event: message_start" and event_data["type"] == "message_start")

print()
print("[3] Validation and count-token conversion")
try:
    P._anthropic_messages_to_chat({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    valid_max_tokens = False
except P.AnthropicRequestError:
    valid_max_tokens = True
check("messages requires positive max_tokens", valid_max_tokens)
count_body = P._anthropic_messages_to_chat({
    "model": "m", "messages": [{"role": "user", "content": "hello"}]
}, require_max_tokens=False)
check("count_tokens accepts a request without max_tokens",
      "max_tokens" not in count_body and P._anthropic_input_tokens(count_body) > 0)

print()
print("%d passed, %d failed" % (PASS, FAIL))
raise SystemExit(1 if FAIL else 0)
