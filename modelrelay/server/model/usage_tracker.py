"""Token/cost accumulation."""
from __future__ import annotations
import time
from ..backends.base import NormalizedUsage

PRICING = {
    "claude-opus":    (15.0, 75.0),   # per 1M tokens
    "claude-sonnet":  (3.0,  15.0),
    "claude-haiku":   (0.25, 1.25),
    "gpt-4o":         (5.0,  15.0),
    "gpt-4":          (30.0, 60.0),
    "mock":           (0.0,  0.0),
    "default":        (5.0,  15.0),
}


def _get_pricing(model: str):
    for prefix, prices in PRICING.items():
        if prefix != "default" and prefix in model.lower():
            return prices
    return PRICING["default"]


class UsageTracker:
    def __init__(self, model: str = ""):
        self._model = model
        self._total_input  = 0
        self._total_output = 0
        self._cost         = 0.0
        self._last_usage:  NormalizedUsage | None = None
        self._start_time   = time.time()
        self._turns        = 0
        self._on_update    = None  # async callable(data)

    def set_on_update(self, fn):
        self._on_update = fn

    def update(self, usage: NormalizedUsage):
        self._total_input  += usage.input_tokens
        self._total_output += usage.output_tokens
        self._last_usage    = usage
        self._turns        += 1
        inp_price, out_price = _get_pricing(self._model)
        self._cost = (self._total_input  * inp_price / 1_000_000 +
                      self._total_output * out_price / 1_000_000)

    def as_dict(self) -> dict:
        u = self._last_usage
        return {
            "prompt_tokens":            self._total_input,
            "completion_tokens":        self._total_output,
            "total_tokens":             self._total_input + self._total_output,
            "cost_usd":                 round(self._cost, 6),
            "requests_remaining":       u.requests_remaining if u else None,
            "tokens_remaining_per_min": u.tokens_remaining_per_min if u else None,
            "reset_at":                 u.reset_at if u else None,
        }

    def session_summary(self) -> dict:
        return {
            "total_prompt_tokens":     self._total_input,
            "total_completion_tokens": self._total_output,
            "total_cost_usd":          round(self._cost, 6),
            "duration_seconds":        round(time.time() - self._start_time, 1),
            "turns":                   self._turns,
        }
