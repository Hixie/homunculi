"""Test SSE parsing logic extracted from AnthropicBackend."""
import json
import unittest


# We test the SSE parsing state machine in isolation by simulating the
# iterator that the _stream_response method processes.

def parse_sse_events(lines):
    """Simulate the SSE parsing logic and return (type, data) pairs."""
    results = []
    input_tokens = 0
    output_tokens = 0
    tool_calls = {}
    current_block = None

    for line in lines:
        line = line.rstrip("\n")
        if line.startswith(":") or not line:
            continue
        if line.startswith("data:"):
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            etype = data.get("type", "")
            if etype == "message_start":
                usage = data.get("message", {}).get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                results.append(("message_start", input_tokens))
            elif etype == "content_block_start":
                block = data.get("content_block", {})
                current_block = block
                if block.get("type") == "tool_use":
                    tool_calls[block["id"]] = {"name": block["name"], "args_str": ""}
            elif etype == "content_block_delta":
                delta = data.get("delta", {})
                dtype = delta.get("type", "")
                if dtype == "text_delta":
                    results.append(("text_delta", delta.get("text", "")))
                elif dtype == "input_json_delta" and current_block:
                    bid = current_block.get("id", "")
                    if bid in tool_calls:
                        tool_calls[bid]["args_str"] += delta.get("partial_json", "")
            elif etype == "content_block_stop" and current_block:
                if current_block.get("type") == "tool_use":
                    bid = current_block["id"]
                    tc = tool_calls.get(bid, {})
                    try:
                        args = json.loads(tc.get("args_str", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    results.append(("tool_call", {"name": tc["name"], "args": args}))
            elif etype == "message_delta":
                usage = data.get("usage", {})
                output_tokens = usage.get("output_tokens", 0)
                results.append(("message_delta", output_tokens))
            elif etype == "message_stop":
                results.append(("message_stop", (input_tokens, output_tokens)))
    return results


class TestAnthropicSSEParsing(unittest.TestCase):
    def test_message_start_input_tokens(self):
        lines = [
            'data: {"type":"message_start","message":{"usage":{"input_tokens":42}}}\n',
        ]
        r = parse_sse_events(lines)
        self.assertEqual(r[0], ("message_start", 42))

    def test_text_delta_carries_text(self):
        lines = [
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n',
        ]
        r = parse_sse_events(lines)
        self.assertIn(("text_delta", "Hello"), r)

    def test_tool_use_accumulates(self):
        lines = [
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"c1","name":"stat"}}\n',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\\"res"}}\n',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"ource\\\":\\\"f\\\"}"}}\n',
        ]
        r = parse_sse_events(lines)
        # No tool_call yet (no stop)
        self.assertFalse(any(t == "tool_call" for t, _ in r))

    def test_tool_use_stop(self):
        lines = [
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"c1","name":"stat"}}\n',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\\"resource\\\":\\\"f\\\"}"}}\n',
            'data: {"type":"content_block_stop","index":0}\n',
        ]
        r = parse_sse_events(lines)
        tc = [d for t, d in r if t == "tool_call"]
        self.assertEqual(len(tc), 1)
        self.assertEqual(tc[0]["name"], "stat")

    def test_message_delta_output_tokens(self):
        lines = [
            'data: {"type":"message_delta","delta":{},"usage":{"output_tokens":77}}\n',
        ]
        r = parse_sse_events(lines)
        self.assertIn(("message_delta", 77), r)

    def test_message_stop(self):
        lines = [
            'data: {"type":"message_start","message":{"usage":{"input_tokens":5}}}\n',
            'data: {"type":"message_delta","delta":{},"usage":{"output_tokens":3}}\n',
            'data: {"type":"message_stop"}\n',
        ]
        r = parse_sse_events(lines)
        stop = [d for t, d in r if t == "message_stop"]
        self.assertEqual(len(stop), 1)

    def test_sse_comment_ignored(self):
        lines = [
            ': this is a comment\n',
            'data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}\n',
        ]
        r = parse_sse_events(lines)
        self.assertEqual(r[0][0], "message_start")

    def test_end_of_stream_clean(self):
        lines = [
            'data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}\n',
            'data: [DONE]\n',
        ]
        r = parse_sse_events(lines)
        # Should not raise
        self.assertEqual(r[0][0], "message_start")
