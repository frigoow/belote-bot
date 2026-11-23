"""Belot's batched C++ environments with cardbot-framework's env protocol, so
cardbot.train's trainers drive them:

- PlayEnv: bg_cpp.BgBatchPlayEnv (card play; auctions by naturalBid or a bid
  net) with observe() / step(); `belief_targets` and `belief_marginals` ask it
  for the belief net's data on the next observations.
- BidEnv: bg_cpp.BgBatchBidEnv (counterfactual bid targets) with use_bid_model().
"""

from __future__ import annotations

import numpy as np

import bg_cpp as bgc


class PlayEnv:
    STATE_DIM = bgc.STATE_DIM
    PLY_DIM = bgc.PLY_DIM
    MAX_PLIES = bgc.MAX_PLIES
    ACTION_DIM = bgc.ACTION_DIM
    MAX_LEGAL = bgc.MAX_LEGAL
    POLICY_DIM = 32  # the fast policy scores canonical cards

    def __init__(self, **kwargs):
        self.env = bgc.BgBatchPlayEnv(**kwargs)
        self.belief_targets = False
        self.belief_marginals = False

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    def reset_all(self, seed: int) -> None:
        self.env.reset_all(seed)

    def observe(self) -> dict:
        b = self.env.observe_batch(include_history=True, include_belief_targets=self.belief_targets,
                                   include_belief_marginals=self.belief_marginals)
        b["team"] = b["current_team"]
        # each legal action's features start with the one-hot canonical card: its policy slot
        k = np.arange(self.MAX_LEGAL)[None, :] < b["n_legal"][:, None]
        b["legal_slots"] = np.where(k, b["legal_actions"][:, :, :32].argmax(axis=2), -1).astype(np.int32)
        return b

    def step(self, action_idx, auto_reset: bool = True):
        done, r0 = self.env.step_batch(np.asarray(action_idx, dtype=np.int32), auto_reset)
        return done, np.stack([r0, -r0], axis=1)


class BidEnv:
    """The bid env for cardbot.train.bidding: generate rows with generate_batch() (the
    trainer's generate_fn), and the self-play bidder through use_bid_model()."""

    def __init__(self, env: "bgc.BgBatchBidEnv"):
        self.env = env

    def use_bid_model(self, path: str, tag: str = "") -> None:
        if not self.env.load_bid_model(str(path)):
            raise RuntimeError(f"Failed to reload {path}")

    def __getattr__(self, name):
        return getattr(self.env, name)
