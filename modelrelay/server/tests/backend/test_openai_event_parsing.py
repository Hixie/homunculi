"""Tests for OpenAI Realtime event parsing (simulated)."""
import json
import unittest


def parse_openai_events(frames):
    """Simulate OpenAI Realtime WS event parsing."""
    results = []
    pending_tool_args = {}

    for frame in frames:
        etype = frame.get("type", "")
        if etype == "response.output_text.delta":
            results.append(("TEXT_DELTA", frame.get("delta", "")))
        elif etype == "response.function_call_arguments.delta":
            cid = frame.get("call_id", "")
            if cid not in pending_tool_args:
                pending_tool_args[cid] = {"name": frame.get("name",""), "args_str": ""}
            pending_tool_args[cid]["args_str"] += frame.get("delta", "")
        elif etype == "response.function_call_arguments.done":
            cid = frame.get("call_id", "")
            tc = pending_tool_args.pop(cid, {"name":"","args_str":"{}"})
            try:
                args = json.loads(tc["args_str"])
            except json.JSONDecodeError:
                args = {"_error": "invalid json"}
            results.append(("TOOL_CALL", {"call_id": cid, "name": tc["name"], "args": args}))
        elif etype == "response.done":
            usage = frame.get("response", {}).get("usage", {})
            results.append(("TURN_DONE", usage))
        elif etype == "rate_limits.updated":
            results.append(("RATE_LIMITED", frame))
        elif etype == "error":
            results.append(("ERROR", frame))
    return results


class TestOpenAIEventParsing(unittest.TestCase):
    def test_text_delta_carries_text(self):
        frames = [{"type": "response.output_text.delta", "delta": "Hello"}]
        r = parse_openai_events(frames)
        self.assertEqual(r[0], ("TEXT_DELTA", "Hello"))

    def test_function_call_accumulates(self):
        frames = [
            {"type":"response.function_call_arguments.delta","call_id":"c1","name":"stat","delta":"{\"res"},
            {"type":"response.function_call_arguments.delta","call_id":"c1","delta":"ource\":\"f\"}"},
        ]
        r = parse_openai_events(frames)
        self.assertEqual(r, [])  # no done yet

    def test_function_call_done(self):
        frames = [
            {"type":"response.function_call_arguments.delta","call_id":"c1","name":"stat","delta":"{\"resource\":\"f.py\"}"},
            {"type":"response.function_call_arguments.done","call_id":"c1"},
        ]
        r = parse_openai_events(frames)
        self.assertEqual(r[0][0], "TOOL_CALL")
        self.assertEqual(r[0][1]["args"]["resource"], "f.py")

    def test_response_done_usage(self):
        frames = [{"type":"response.done","response":{"usage":{"input_tokens":10,"output_tokens":5}}}]
        r = parse_openai_events(frames)
        self.assertEqual(r[0][0], "TURN_DONE")

    def test_rate_limit_event(self):
        frames = [{"type":"rate_limits.updated","rate_limits":[]}]
        r = parse_openai_events(frames)
        self.assertEqual(r[0][0], "RATE_LIMITED")

    def test_api_error(self):
        frames = [{"type":"error","error":{"message":"bad request"}}]
        r = parse_openai_events(frames)
        self.assertEqual(r[0][0], "ERROR")

    def test_malformed_args(self):
        frames = [
            {"type":"response.function_call_arguments.delta","call_id":"c1","name":"stat","delta":"not-valid-json"},
            {"type":"response.function_call_arguments.done","call_id":"c1"},
        ]
        r = parse_openai_events(frames)
        self.assertEqual(r[0][0], "TOOL_CALL")
        self.assertIn("_error", r[0][1]["args"])
