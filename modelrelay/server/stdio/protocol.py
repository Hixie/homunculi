"""LOJP envelope helpers and factory functions."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def make_envelope(type_: str, payload: dict, id_: str | None = None) -> dict:
    return {
        "lojp": "1.0",
        "id": id_ or str(uuid.uuid4()),
        "ts": _now_iso(),
        "type": type_,
        "payload": payload,
    }


def encode(envelope: dict) -> bytes:
    return (json.dumps(envelope, ensure_ascii=False) + "\n").encode("utf-8")


def decode(line: bytes) -> dict:
    return json.loads(line.decode("utf-8").rstrip("\n"))


def content_request(resource: str, *, ranges=None, pattern=None,
                    max_results=20, context_lines=2, id_=None) -> dict:
    if ranges is not None:
        payload = {"resource": resource, "ranges": ranges}
    elif pattern is not None:
        payload = {"resource": resource, "pattern": pattern,
                   "max_results": max_results, "context_lines": context_lines}
    else:
        raise ValueError("Either ranges or pattern must be provided")
    return make_envelope("tool.content_request", payload, id_=id_)


def replace_request(resource, start_line, end_line, new_lines, total_lines, id_=None):
    return make_envelope("tool.replace_request", {
        "resource": resource, "start_line": start_line, "end_line": end_line,
        "new_lines": new_lines, "total_lines": total_lines,
    }, id_=id_)


def insert_request(resource, after_line, new_lines, total_lines, id_=None):
    return make_envelope("tool.insert_request", {
        "resource": resource, "after_line": after_line,
        "new_lines": new_lines, "total_lines": total_lines,
    }, id_=id_)


def stat_request(resource, id_=None):
    return make_envelope("tool.stat_request", {"resource": resource}, id_=id_)


def model_text_delta(text, id_=None):
    return make_envelope("model.text_delta", {"text": text}, id_=id_)


def activity(state, description, id_=None):
    return make_envelope("status.activity",
                         {"state": state, "description": description}, id_=id_)


def usage_msg(data, id_=None):
    return make_envelope("status.usage", data, id_=id_)


def session_ended(logfile, summary, id_=None):
    return make_envelope("session.ended",
                         {"logfile": logfile, "summary": summary}, id_=id_)


def error_msg(code, message, fatal=False, id_=None):
    return make_envelope("error",
                         {"code": code, "message": message, "fatal": fatal}, id_=id_)
