from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

from simulator.core import CardView
from simulator.math.fsrs import (
    _clamp_d,
    _clamp_s,
    fsrs6_forgetting_curve,
    fsrs6_init_state,
    fsrs6_next_d,
    fsrs6_next_interval,
    fsrs6_stability_after_failure,
    fsrs6_stability_after_success,
    fsrs6_stability_short_term,
)
from simulator.schedulers.fsrs import FSRS6Scheduler, FSRS6VectorizedSchedulerOps

if TYPE_CHECKING:
    import torch

# ---------------- ADR parameters ----------------
ADR_FLAT = 2.15
ADR_S_MULTI = 0.135
ADR_D_MULTI = -0.085

MAX_TARGET_DR = 0.94


def adr_target_dr(s: float, d: float) -> float:
    """Compute the ADR target retrieval rate from stability and difficulty.

    Implements:
        logit = ADR_FLAT + ADR_S_MULTI * ln(s) + ADR_D_MULTI * d
        DR    = clip(sigmoid(logit), 0, MAX_TARGET_DR)

    s must be positive; caller is responsible for clamping to s_min first.
    """
    logit = ADR_FLAT + ADR_S_MULTI * math.log(max(s, 1e-12)) + ADR_D_MULTI * d
    logit = max(-10.0, min(10.0, logit))
    return min(1.0 / (1.0 + math.exp(-logit)), MAX_TARGET_DR)


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
        return adr_target_dr(max(self.params.bounds.s_min, float(s)), float(d))

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


class TestADRVectorizedSchedulerOps(FSRS6VectorizedSchedulerOps):
    """
    Vectorized ops for TestADRScheduler.

    The parent class handles all state updates (s, d); we only replace the
    final interval calculation with a torch port of adr_target_dr:

        logit = ADR_FLAT + ADR_S_MULTI * ln(s) + ADR_D_MULTI * d
        dr    = clip(sigmoid(logit), 0, MAX_TARGET_DR)
        ivl   = max(1, s / factor * (dr^(1/decay) - 1))
    """

    def _adr_interval(self, s: "torch.Tensor", d: "torch.Tensor") -> "torch.Tensor":
        logit = ADR_FLAT + ADR_S_MULTI * s.clamp(min=1e-12).log() + ADR_D_MULTI * d
        xc = logit.clamp(-10.0, 10.0)
        dr = (1.0 / (1.0 + self._torch.exp(-xc))).clamp(max=MAX_TARGET_DR)
        retention_factor = dr.pow(1.0 / self._decay) - 1.0
        return (s / self._factor * retention_factor).clamp(min=1.0)

    def update_review(
        self,
        state,
        idx: "torch.Tensor",
        elapsed: "torch.Tensor",
        rating: "torch.Tensor",
        prev_interval: "torch.Tensor",
    ) -> "torch.Tensor":
        # Let the parent update state.s[idx] and state.d[idx] in place.
        super().update_review(state, idx, elapsed, rating, prev_interval)
        if idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)
        return self._adr_interval(state.s[idx], state.d[idx])

    def update_learn(
        self,
        state,
        idx: "torch.Tensor",
        rating: "torch.Tensor",
    ) -> "torch.Tensor":
        # Let the parent update state.s[idx] and state.d[idx] in place.
        super().update_learn(state, idx, rating)
        if idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)
        return self._adr_interval(state.s[idx], state.d[idx])
