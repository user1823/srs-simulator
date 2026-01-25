from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from simulator.core import CardView, Scheduler
from simulator.schedulers.fsrs import FSRS6Scheduler
import srs_simulator_rs

if TYPE_CHECKING:
    import torch


class FSRS6ADRScheduler(Scheduler):

    def __init__(self, weights: Sequence[float], dr_equivalent, initial_rating_prob, initial_cost, review_rating_prob_given_success, review_cost) -> None:
        self.lib = srs_simulator_rs.Lib(weights, dr_equivalent, initial_rating_prob, initial_cost, review_rating_prob_given_success, review_cost)
        self.truth = FSRS6Scheduler(weights)
        exit()
        pass

    def init_card(self, card_view: CardView, rating: int, day: float):
        ret = self.lib.init(rating)
        # true_ret = self.truth.init_card(card_view, rating, day)
        return ret

    def schedule(self, card_view: CardView, rating: int, elapsed: float, day: float):
        s, d = card_view.scheduler_state
        # print(s, d, rating, elapsed)
        ret = self.lib.schedule(s, d, rating, elapsed)
        # card_view.scheduler_state = { "s": s, "d": d}
        # true_ret = self.truth.schedule(card_view, rating, elapsed, day)
        # print("start:", s, d)
        # print(ret, true_ret)
        # print(ret)
        return ret


@dataclass
class FixedVectorizedState:
    interval: float


class FixedVectorizedSchedulerOps:
    def __init__(
        self,
        scheduler: FixedIntervalScheduler,
        *,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> None:
        raise ValueError("not imp")
        import torch

        self._torch = torch
        self.device = device
        self.dtype = dtype
        self._interval = float(scheduler.interval)
        self._interval_days = max(1, int(round(self._interval)))

    def init_state(self, deck_size: int) -> FixedVectorizedState:
        return FixedVectorizedState(interval=self._interval)

    def review_priority(
        self, state: FixedVectorizedState, idx: "torch.Tensor", elapsed: "torch.Tensor"
    ) -> "torch.Tensor":
        return self._torch.zeros(idx.numel(), device=self.device, dtype=self.dtype)

    def update_review(
        self,
        state: FixedVectorizedState,
        idx: "torch.Tensor",
        elapsed: "torch.Tensor",
        rating: "torch.Tensor",
        prev_interval: "torch.Tensor",
    ) -> "torch.Tensor":
        return self._torch.full(
            (idx.numel(),),
            float(self._interval_days),
            device=self.device,
            dtype=self.dtype,
        )

    def update_learn(
        self,
        state: FixedVectorizedState,
        idx: "torch.Tensor",
        rating: "torch.Tensor",
    ) -> "torch.Tensor":
        return self._torch.full(
            (idx.numel(),),
            float(self._interval_days),
            device=self.device,
            dtype=self.dtype,
        )
