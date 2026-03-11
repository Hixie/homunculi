import json
import unittest
import uuid
from ...stdio.protocol import (
    decode, encode, make_envelope, content_request, model_text_delta,
    activity, replace_request, insert_request, stat_request, error_msg,
    session_ended, usage_msg,
)


class TestProtocol(unittest.TestCase):
    def test_encode_decode_roundtrip(self):
        env = make_envelope("cmd.test", {"key": "value"})
        self.assertEqual(decode(encode(env)), env)

    def test_encode_produces_single_line(self):
        env = make_envelope("test", {"x": 1})
        b = encode(env)
        self.assertEqual(b.count(b"\n"), 1)
        self.assertTrue(b.endswith(b"\n"))

    def test_newlines_in_payload_are_escaped(self):
        env = make_envelope("test", {"text": "line1\nline2"})
        b = encode(env)
        self.assertEqual(b.count(b"\n"), 1)

    def test_decode_raises_on_malformed(self):
        with self.assertRaises(Exception):
            decode(b"not json\n")

    def test_make_envelope_fields(self):
        env = make_envelope("my.type", {"a": 1})
        self.assertEqual(env["lojp"], "1.0")
        self.assertEqual(env["type"], "my.type")
        self.assertIsInstance(env["id"], str)
        uuid.UUID(env["id"])  # valid UUID
        self.assertIn("T", env["ts"])  # ISO timestamp

    def test_make_envelope_reuses_provided_id(self):
        my_id = str(uuid.uuid4())
        env = make_envelope("test", {}, id_=my_id)
        self.assertEqual(env["id"], my_id)

    def test_content_request_read_form(self):
        env = content_request("src/auth.py", ranges=[{"start_line": 1, "num_lines": 10}])
        self.assertEqual(env["type"], "tool.content_request")
        p = env["payload"]
        self.assertEqual(p["resource"], "src/auth.py")
        self.assertIn("ranges", p)
        self.assertNotIn("pattern", p)

    def test_content_request_search_form(self):
        env = content_request("src/auth.py", pattern="foo")
        self.assertEqual(env["type"], "tool.content_request")
        p = env["payload"]
        self.assertIn("pattern", p)
        self.assertNotIn("ranges", p)

    def test_model_text_delta_factory(self):
        env = model_text_delta("hello")
        self.assertEqual(env["type"], "model.text_delta")
        self.assertEqual(env["payload"]["text"], "hello")

    def test_all_convenience_factories(self):
        factories = [
            replace_request("f", 1, 2, ["x\n"], 10),
            insert_request("f", 0, ["x\n"], 10),
            stat_request("f"),
            model_text_delta("hi"),
            activity("idle", "ok"),
            usage_msg({"total_tokens": 0}),
            session_ended("/log", {"turns": 1}),
            error_msg("E", "msg"),
        ]
        for env in factories:
            self.assertEqual(env["lojp"], "1.0")
            self.assertIn("type", env)
            self.assertIn("payload", env)
