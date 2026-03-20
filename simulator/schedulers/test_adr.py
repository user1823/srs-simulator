from __future__ import annotations

import math
from typing import Sequence

from simulator.core import CardView
from simulator.schedulers.fsrs import FSRS6Scheduler
from simulator.math.fsrs import (
    fsrs6_forgetting_curve,
    fsrs6_init_state,
    fsrs6_next_d,
    fsrs6_next_interval,
    fsrs6_stability_after_failure,
    fsrs6_stability_after_success,
    fsrs6_stability_short_term,
    _clamp_d,
    _clamp_s,
)

# ---------------- ADR parameters ----------------
ADR_FLAT = 2.15
ADR_S_MULTI = 0.135
ADR_D_MULTI = -0.085


def _sigmoid(x: float) -> float:
    x = max(-10.0, min(10.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


class TestADRScheduler(FSRS6Scheduler):
    """
    Simple FSRS-6 + linear ADR test scheduler.

    DR is computed from the post-update state:
        DR = sigmoid(ADR_FLAT + ADR_S_MULTI * ln(s) + ADR_D_MULTI * d)

    This intentionally skips the Rust annealer and uses only the three
    editable constants above.
    """

    def __init__(
        self,
        weights: Sequence[float] | None = None,
        priority_mode: str = "low_retrievability",
    ):
        # desired_retention is unused here, but FSRS6Scheduler sets up params,
        # bounds, and review priority helpers for us.
        super().__init__(
            weights=weights,
            desired_retention=0.9,
            priority_mode=priority_mode,
        )

    def _target_dr(self, s: float, d: float) -> float:
        s = max(self.params.bounds.s_min, float(s))
        logit = ADR_FLAT + ADR_S_MULTI * math.log(s) + ADR_D_MULTI * float(d)
        return min(0.995, _sigmoid(logit))

    def init_card(self, card_view: CardView, rating: int, day: float):
        s, d = fsrs6_init_state(self.params, rating)
        state = {
            "s": _clamp_s(self.params.bounds, s),
            "d": _clamp_d(self.params.bounds, d),
        }
        dr = self._target_dr(state["s"], state["d"])
        interval = fsrs6_next_interval(self.params, state["s"], dr)
        return interval, state

    def schedule(self, card_view: CardView, rating: int, elapsed: float, day: float):
        state = card_view.scheduler_state or {
            "s": fsrs6_init_state(self.params, 3)[0],
            "d": fsrs6_init_state(self.params, 3)[1],
        }

        s = max(self.params.bounds.s_min, float(state["s"]))
        d = max(
            self.params.bounds.d_min,
            min(float(state["d"]), self.params.bounds.d_max),
        )

        r = fsrs6_forgetting_curve(self.params, elapsed, s)

        if elapsed < 1.0:
            s = fsrs6_stability_short_term(self.params, s, rating)
        else:
            if rating > 1:
                s = fsrs6_stability_after_success(self.params, s, r, d, rating)
            else:
                s = fsrs6_stability_after_failure(self.params, s, r, d)

        d = fsrs6_next_d(self.params, d, rating)

        state = {
            "s": _clamp_s(self.params.bounds, s),
            "d": _clamp_d(self.params.bounds, d),
        }

        dr = self._target_dr(state["s"], state["d"])
        interval = fsrs6_next_interval(self.params, state["s"], dr)
        return interval, state
