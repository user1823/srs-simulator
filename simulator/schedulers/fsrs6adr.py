from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from simulator.core import CardView, Scheduler
from simulator.schedulers.fsrs import FSRS6Scheduler
import srs_simulator_rs
import torch

if TYPE_CHECKING:
    import torch


class FSRS6ADRScheduler(Scheduler):

    def __init__(self, weights: Sequence[float], dr_equivalent, days, deck_size, new_cards_per_day, initial_rating_prob, initial_cost, review_rating_prob_given_success, review_cost) -> None:
        self.lib = srs_simulator_rs.Lib(weights, dr_equivalent, days, max(1, deck_size), max(1, new_cards_per_day), initial_rating_prob, initial_cost, review_rating_prob_given_success, review_cost)
        exit(2)
        pass

    def init_card(self, card_view: CardView, rating: int, day: float):
        ret = self.lib.init_single(rating)
        return ret

    def schedule(self, card_view: CardView, rating: int, elapsed: float, day: float):
        s, d = card_view.scheduler_state
        ret = self.lib.schedule_single(s, d, rating, elapsed)
        return ret


@dataclass
class FSRS6ADRVectorizedState:
    s: "torch.Tensor"
    d: "torch.Tensor"


class FSRS6ADRVectorizedSchedulerOps:
    def __init__(
        self,
        scheduler: FSRS6ADRScheduler,
        *,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> None:
        self.scheduler = scheduler
        self._torch = torch
        self.device = device
        self.dtype = dtype

    def init_state(self, deck_size: int) -> FSRS6ADRVectorizedState:
        s = self._torch.full(
            (deck_size,), 0, dtype=self.dtype, device=self.device
        )
        d = self._torch.full(
            (deck_size,), 0, dtype=self.dtype, device=self.device
        )
        return FSRS6ADRVectorizedState(s=s, d=d)

    def review_priority(
        self, state: FSRS6ADRVectorizedState, idx: "torch.Tensor", elapsed: "torch.Tensor"
    ) -> "torch.Tensor":
        return self._torch.zeros(idx.numel(), device=self.device, dtype=self.dtype)

    def update_review(
        self,
        state: FSRS6ADRVectorizedState,
        idx: "torch.Tensor",
        elapsed: "torch.Tensor",
        rating: "torch.Tensor",
        prev_interval: "torch.Tensor",
    ) -> "torch.Tensor":
        if idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)
        sched_s = state.s[idx].float().cpu().numpy()
        sched_d = state.d[idx].float().cpu().numpy()
        intervals, s_init, d_init = self.scheduler.lib.review(sched_s, sched_d, rating.int().cpu().numpy(), elapsed.float().cpu().numpy())
        state.s[idx] = torch.tensor(s_init, device=self.device, dtype=self.dtype)
        state.d[idx] = torch.tensor(d_init, device=self.device, dtype=self.dtype)
        return torch.tensor(intervals, device=self.device, dtype=self.dtype)

    def update_learn(
        self,
        state: FSRS6ADRVectorizedState,
        idx: "torch.Tensor",
        rating: "torch.Tensor",
    ) -> "torch.Tensor":
        if idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)
        intervals, s_init, d_init = self.scheduler.lib.init_state(rating.int().cpu().numpy())
        state.s[idx] = torch.tensor(s_init, device=self.device, dtype=self.dtype)
        state.d[idx] = torch.tensor(d_init, device=self.device, dtype=self.dtype)
        return torch.tensor(intervals, device=self.device, dtype=self.dtype)