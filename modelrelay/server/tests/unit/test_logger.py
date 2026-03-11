"""Tests for auditlog.Logger — verifies every record shape and field invariant
described in auditlog/logger.py."""
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from ...auditlog.logger import Logger


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def make_logger(self):
        start = datetime(2024, 3, 15, 9, 22, 11, tzinfo=timezone.utc)
        return Logger(self.tmpdir, session_start=start)

    def records(self, lg):
        with open(lg.path) as f:
            return [json.loads(l) for l in f]

    # ── file / format basics ──────────────────────────────────────────────────

    def test_filename_format(self):
        lg = self.make_logger()
        lg.close()
        self.assertRegex(os.path.basename(lg.path), r"^\d{8}T\d{6}\.jsonl$")

    def test_seq_monotonic(self):
        lg = self.make_logger()
        for _ in range(5):
            lg.log_protocol(
                {"id": "x", "ts": "t", "type": "cmd.prompt", "payload": {}},
                "inbound")
        lg.close()
        seqs = [r["seq"] for r in self.records(lg)]
        self.assertEqual(seqs, list(range(1, 6)))

    def test_each_line_valid_json(self):
        lg = self.make_logger()
        lg.log_protocol({"id": "a", "ts": "t", "type": "cmd.prompt", "payload": {}},
                        "inbound")
        lg.log_wire_rx("anthropic", {"type": "message_start"}, "processed")
        lg.log_model_tx("send_turn", text="hello")
        lg.close()
        recs = self.records(lg)
        self.assertEqual(len(recs), 3)

    def test_no_literal_newlines(self):
        lg = self.make_logger()
        lg.log_protocol(
            {"id": "x", "ts": "t", "type": "cmd.prompt",
             "payload": {"text": "line1\nline2"}},
            "outbound")
        lg.close()
        with open(lg.path, "rb") as f:
            self.assertEqual(f.read().count(b"\n"), 1)

    def test_close_then_write_silent(self):
        lg = self.make_logger()
        lg.close()
        lg.log_protocol(
            {"id": "x", "ts": "t", "type": "cmd.prompt", "payload": {}},
            "outbound")  # must not raise

    # ── common envelope: seq, ts, category must be FIRST three fields ─────────

    def test_seq_ts_category_come_first(self):
        """seq, ts, category must be the first three keys in every serialised record."""
        lg = self.make_logger()
        lg.log_protocol({"id": "x", "ts": "t", "type": "cmd.prompt", "payload": {}},
                        "inbound")
        lg.log_wire_rx("anthropic", {"type": "message_start"}, "processed")
        lg.log_model_rx("TEXT_DELTA", {"text": "hi"})
        lg.log_model_tx("send_turn", text="hi")
        lg.write_event("session_start", {})
        lg.close()
        with open(lg.path) as f:
            for raw in f:
                keys = list(json.loads(raw).keys())
                self.assertEqual(keys[:3], ["seq", "ts", "category"],
                                 f"wrong key order in: {raw.rstrip()}")

    def test_ts_is_iso8601_utc(self):
        """ts must match ISO-8601 with explicit +00:00 offset."""
        import re
        lg = self.make_logger()
        lg.log_wire_rx("anthropic", {"type": "message_start"}, "processed")
        lg.close()
        rec = self.records(lg)[0]
        self.assertRegex(rec["ts"],
                         r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\+00:00$")

    def test_no_log_meta_wrapper(self):
        """Legacy _log wrapper must not appear in any record."""
        lg = self.make_logger()
        lg.log_wire_rx("anthropic", {"type": "message_start"}, "processed")
        lg.log_model_rx("TEXT_DELTA", {"text": "hi"})
        lg.log_model_tx("send_turn", text="hi")
        lg.close()
        for rec in self.records(lg):
            self.assertNotIn("_log", rec, f"stale _log found in {rec}")

    def test_every_record_has_seq_ts_category(self):
        lg = self.make_logger()
        lg.log_protocol({"id": "x", "ts": "t", "type": "p", "payload": {}}, "inbound")
        lg.log_wire_rx("anthropic", {"type": "message_start"}, "processed")
        lg.log_model_rx("TEXT_DELTA", {"text": "hi"})
        lg.log_model_tx("send_turn", text="hi")
        lg.write_event("session_start", {})
        lg.close()
        for rec in self.records(lg):
            self.assertIn("seq",      rec, f"missing seq in {rec}")
            self.assertIn("ts",       rec, f"missing ts in {rec}")
            self.assertIn("category", rec, f"missing category in {rec}")

    # ── category: protocol ────────────────────────────────────────────────────

    def test_protocol_inbound_fields(self):
        lg = self.make_logger()
        lg.log_protocol(
            {"id": "abc-123", "ts": "2026-01-01T00:00:00.000Z",
             "type": "cmd.prompt", "payload": {"text": "go"}},
            "inbound")
        lg.close()
        rec = self.records(lg)[0]
        self.assertEqual(rec["category"],    "protocol")
        self.assertEqual(rec["direction"],   "inbound")
        self.assertEqual(rec["disposition"], "processed")
        self.assertEqual(rec["msg_id"],      "abc-123")
        self.assertEqual(rec["msg_ts"],      "2026-01-01T00:00:00.000Z")
        self.assertEqual(rec["msg_type"],    "cmd.prompt")
        self.assertEqual(rec["payload"],     {"text": "go"})

    def test_protocol_outbound_fields(self):
        lg = self.make_logger()
        lg.log_protocol(
            {"id": "def-456", "ts": "2026-01-01T00:00:01.000Z",
             "type": "status.activity",
             "payload": {"state": "idle", "description": "Turn complete"}},
            "outbound")
        lg.close()
        rec = self.records(lg)[0]
        self.assertEqual(rec["direction"],   "outbound")
        self.assertEqual(rec["disposition"], "processed")
        self.assertEqual(rec["msg_type"],    "status.activity")
        self.assertEqual(rec["payload"]["state"], "idle")

    def test_protocol_no_lojp_field(self):
        """The lojp version string must not appear in log records."""
        lg = self.make_logger()
        lg.log_protocol(
            {"lojp": "1.0", "id": "x", "ts": "t",
             "type": "cmd.prompt", "payload": {}},
            "inbound")
        lg.close()
        rec = self.records(lg)[0]
        self.assertNotIn("lojp", rec)

    def test_protocol_inbound_error_fields(self):
        """log_protocol_inbound_error must write direction=inbound_error,
        disposition=erroneous, and the raw/reason/detail fields."""
        lg = self.make_logger()
        lg.log_protocol_inbound_error("bad json{", "PARSE_ERROR", "Expecting value")
        lg.close()
        rec = self.records(lg)[0]
        self.assertEqual(rec["category"],    "protocol")
        self.assertEqual(rec["direction"],   "inbound_error")
        self.assertEqual(rec["disposition"], "erroneous")
        self.assertEqual(rec["raw"],         "bad json{")
        self.assertEqual(rec["reason"],      "PARSE_ERROR")
        self.assertIn("Expecting value",     rec["detail"])

    def test_protocol_unknown_type_error_fields(self):
        lg = self.make_logger()
        lg.log_protocol_inbound_error(
            '{"type":"bogus.msg"}', "UNKNOWN_TYPE", "bogus.msg")
        lg.close()
        rec = self.records(lg)[0]
        self.assertEqual(rec["direction"],   "inbound_error")
        self.assertEqual(rec["disposition"], "erroneous")
        self.assertEqual(rec["reason"],      "UNKNOWN_TYPE")
        self.assertEqual(rec["detail"],      "bogus.msg")

    def test_protocol_disposition_before_msg_id(self):
        """disposition must appear before msg_id in the serialised record
        (seq, ts, category, direction, disposition, msg_id, …)."""
        lg = self.make_logger()
        lg.log_protocol(
            {"id": "x", "ts": "t", "type": "cmd.prompt", "payload": {}}, "inbound")
        lg.close()
        with open(lg.path) as f:
            keys = list(__import__("json").loads(f.read()).keys())
        self.assertLess(keys.index("disposition"), keys.index("msg_id"))

    def test_protocol_no_ignored_disposition(self):
        """The protocol category never uses disposition='ignored': every accepted
        message causes an action and every unknown message is erroneous."""
        lg = self.make_logger()
        lg.log_protocol(
            {"id": "x", "ts": "t", "type": "cmd.prompt", "payload": {}}, "inbound")
        lg.log_protocol(
            {"id": "y", "ts": "t", "type": "status.activity",
             "payload": {"state": "idle", "description": ""}}, "outbound")
        lg.close()
        for rec in self.records(lg):
            self.assertNotEqual(rec.get("disposition"), "ignored",
                f"protocol record must never be 'ignored': {rec}")

    # ── category: wire_rx ─────────────────────────────────────────────────────

    def test_wire_rx_fields(self):
        lg = self.make_logger()
        lg.log_wire_rx("anthropic",
                       {"type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": "hi"}},
                       "processed")
        lg.close()
        rec = self.records(lg)[0]
        self.assertEqual(rec["category"],    "wire_rx")
        self.assertEqual(rec["backend"],     "anthropic")
        self.assertEqual(rec["disposition"], "processed")
        self.assertEqual(rec["frame"]["type"], "content_block_delta")
        self.assertEqual(rec["frame"]["delta"]["text"], "hi")

    def test_wire_rx_openai_backend(self):
        lg = self.make_logger()
        lg.log_wire_rx("openai-realtime",
                       {"type": "response.output_item.added",
                        "item": {"type": "function_call",
                                 "call_id": "c1", "name": "read"}},
                       "processed")
        lg.close()
        rec = self.records(lg)[0]
        self.assertEqual(rec["backend"],     "openai-realtime")
        self.assertEqual(rec["disposition"], "processed")
        self.assertEqual(rec["frame"]["item"]["name"], "read")

    def test_wire_rx_ignored_frame_still_logged(self):
        """Frames that produce no model_rx are still written to wire_rx,
        and their disposition is "ignored"."""
        lg = self.make_logger()
        lg.log_wire_rx("openai-realtime",
                       {"type": "session.created", "session": {"id": "s1"}},
                       "ignored")
        lg.close()
        rec = self.records(lg)[0]
        self.assertEqual(rec["frame"]["type"], "session.created")
        self.assertEqual(rec["disposition"],   "ignored")

    def test_wire_rx_disposition_values(self):
        """All three disposition values must be accepted and written verbatim."""
        lg = self.make_logger()
        frame = {"type": "x"}
        lg.log_wire_rx("anthropic", frame, "processed")
        lg.log_wire_rx("anthropic", frame, "ignored")
        lg.close()
        recs = self.records(lg)
        self.assertEqual(recs[0]["disposition"], "processed")
        self.assertEqual(recs[1]["disposition"], "ignored")

    def test_wire_rx_error_disposition_always_erroneous(self):
        """log_wire_rx_error must always write disposition='erroneous'."""
        lg = self.make_logger()
        lg.log_wire_rx_error("openai-realtime", "bad{{{", "Expecting value")
        lg.close()
        rec = self.records(lg)[0]
        self.assertEqual(rec["category"],    "wire_rx")
        self.assertEqual(rec["disposition"], "erroneous")
        self.assertEqual(rec["frame_raw"],   "bad{{{")
        self.assertIn("Expecting value", rec["error"])
        self.assertNotIn("frame", rec)  # no "frame" key on error records

    def test_wire_rx_disposition_before_frame(self):
        """disposition must appear before frame in the serialised record
        (seq, ts, category, backend, disposition, frame)."""
        lg = self.make_logger()
        lg.log_wire_rx("anthropic", {"type": "message_start"}, "processed")
        lg.close()
        with open(lg.path) as f:
            keys = list(__import__("json").loads(f.read()).keys())
        self.assertLess(keys.index("disposition"), keys.index("frame"))

    # ── category: model_rx ───────────────────────────────────────────────────

    def test_model_rx_text_delta(self):
        lg = self.make_logger()
        lg.log_model_rx("TEXT_DELTA", {"text": "hello"})
        lg.close()
        rec = self.records(lg)[0]
        self.assertEqual(rec["category"], "model_rx")
        self.assertEqual(rec["event"],    "TEXT_DELTA")
        self.assertEqual(rec["data"]["text"], "hello")

    def test_model_rx_tool_call(self):
        lg = self.make_logger()
        lg.log_model_rx("TOOL_CALL",
                        {"call_id": "c1", "name": "read", "args": {}})
        lg.close()
        rec = self.records(lg)[0]
        self.assertEqual(rec["event"], "TOOL_CALL")
        self.assertEqual(rec["data"]["name"], "read")

    def test_model_rx_turn_done(self):
        lg = self.make_logger()
        lg.log_model_rx("TURN_DONE",
                        {"input_tokens": 10, "output_tokens": 5})
        lg.close()
        rec = self.records(lg)[0]
        self.assertEqual(rec["event"], "TURN_DONE")
        self.assertEqual(rec["data"]["input_tokens"], 10)

    # ── category: model_tx ───────────────────────────────────────────────────

    def test_model_tx_send_turn(self):
        lg = self.make_logger()
        lg.log_model_tx("send_turn", text="fix the bug")
        lg.close()
        rec = self.records(lg)[0]
        self.assertEqual(rec["category"], "model_tx")
        self.assertEqual(rec["kind"],     "send_turn")
        self.assertEqual(rec["text"],     "fix the bug")

    def test_model_tx_tool_result(self):
        lg = self.make_logger()
        lg.log_model_tx("tool_result",
                        call_id="c1", name="read",
                        args={"resource": "f.py"},
                        result="line 1\nline 2")
        lg.close()
        rec = self.records(lg)[0]
        self.assertEqual(rec["kind"],    "tool_result")
        self.assertEqual(rec["call_id"], "c1")
        self.assertEqual(rec["name"],    "read")
        self.assertEqual(rec["result"],  "line 1\nline 2")

    def test_model_tx_inject_note(self):
        lg = self.make_logger()
        lg.log_model_tx("inject_note", text="file changed")
        lg.close()
        rec = self.records(lg)[0]
        self.assertEqual(rec["kind"], "inject_note")
        self.assertEqual(rec["text"], "file changed")

    def test_model_tx_fields_inlined(self):
        """kind-specific fields are inlined, not nested under a sub-key."""
        lg = self.make_logger()
        lg.log_model_tx("send_turn", text="go")
        lg.close()
        rec = self.records(lg)[0]
        self.assertIn("text", rec)         # inlined at top level
        self.assertNotIn("fields", rec)    # no wrapper

    # ── category: lifecycle ───────────────────────────────────────────────────

    def test_lifecycle_session_start(self):
        lg = self.make_logger()
        lg.write_event("session_start",
                       {"backend": "anthropic", "model": "claude-opus-4-5",
                        "log_file": "20260309T213300.jsonl"})
        lg.close()
        rec = self.records(lg)[0]
        self.assertEqual(rec["category"], "lifecycle")
        self.assertEqual(rec["event"],    "session_start")
        self.assertEqual(rec["data"]["backend"], "anthropic")
        self.assertEqual(rec["data"]["model"],   "claude-opus-4-5")

    def test_lifecycle_session_end(self):
        lg = self.make_logger()
        lg.write_event("session_end",
                       {"total_tokens": 4864, "cost_usd": 0.0282})
        lg.close()
        rec = self.records(lg)[0]
        self.assertEqual(rec["category"], "lifecycle")
        self.assertEqual(rec["event"],    "session_end")
        self.assertEqual(rec["data"]["total_tokens"], 4864)

    # ── cross-category ordering ───────────────────────────────────────────────

    def test_seq_spans_all_categories(self):
        lg = self.make_logger()
        lg.write_event("session_start",
                       {"backend": "mock", "model": "mock", "log_file": "x.jsonl"})
        lg.log_protocol({"id":"x","ts":"t","type":"cmd.prompt","payload":{}},
                        "inbound")
        lg.log_model_tx("send_turn", text="go")
        lg.log_wire_rx("anthropic", {"type": "message_start"}, "processed")
        lg.log_model_rx("TEXT_DELTA", {"text": "hi"})
        lg.log_model_rx("TURN_DONE", {"input_tokens": 1, "output_tokens": 1})
        lg.log_protocol({"id":"y","ts":"t","type":"status.activity",
                         "payload":{"state":"idle","description":"Turn complete"}},
                        "outbound")
        lg.write_event("session_end", {"total_tokens": 2, "cost_usd": 0.0})
        lg.close()
        seqs = [r["seq"] for r in self.records(lg)]
        self.assertEqual(seqs, list(range(1, 9)))

    def test_all_five_categories_representable(self):
        lg = self.make_logger()
        lg.log_protocol({"id":"x","ts":"t","type":"p","payload":{}}, "inbound")
        lg.log_wire_rx("anthropic", {"type": "t"}, "processed")
        lg.log_model_rx("TEXT_DELTA", {"text": "hi"})
        lg.log_model_tx("send_turn", text="hi")
        lg.write_event("session_start", {})
        lg.close()
        cats = {r["category"] for r in self.records(lg)}
        self.assertEqual(cats,
                         {"protocol", "wire_rx", "model_rx", "model_tx", "lifecycle"})
