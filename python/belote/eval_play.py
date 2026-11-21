"""Paired duplicate-deal evaluation for PyTorch card-play & belief models."""

from __future__ import annotations

import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

import belote_cpp as bc
from belote.models import BeliefNet, FastPolicy


@torch.no_grad()
def choose_cards_batch(
    model: torch.nn.Module,
    deals: list[bc.Deal],
    device: torch.device,
) -> list[int]:
    """Selects greedy best legal card for a batch of Deals in kPlay."""
    n = len(deals)
    if n == 0:
        return []
    use_hist = getattr(model, "arch", None) == "gru"
    is_fast = isinstance(model, FastPolicy)

    states = np.empty((n, bc.STATE_DIM), dtype=np.float32)
    hists = np.empty((n, bc.MAX_PLIES, bc.PLY_DIM), dtype=np.float32) if use_hist else None
    n_plies = np.empty(n, dtype=np.int32) if use_hist else None
    legal_lists: list[list[int]] = []
    max_k = 1
    obs_list: list[bc.Observation] = []

    for i, d in enumerate(deals):
        me = d.current_player
        obs = bc.observe(d, me)
        obs_list.append(obs)
        if use_hist:
            s, h, np_i = bc.encode_observation(obs)
            states[i] = s
            hists[i] = h
            n_plies[i] = np_i
        else:
            states[i] = bc.encode_state(obs)
        lc = d.legal_cards()
        legal_lists.append(lc)
        if len(lc) > max_k:
            max_k = len(lc)

    s_t = torch.from_numpy(states).to(device)
    if is_fast:
        logits = model(s_t).cpu().numpy()  # (n, 32) canonical cards
        chosen = []
        for i, d in enumerate(deals):
            trump = d.trump
            lc = legal_lists[i]
            if len(lc) == 1:
                chosen.append(lc[0])
                continue
            best_c = max(lc, key=lambda c: logits[i, bc.canon_card(c, trump)])
            chosen.append(best_c)
        return chosen

    actions = np.zeros((n, max_k, bc.ACTION_DIM), dtype=np.float32)
    valid = np.zeros((n, max_k), dtype=bool)
    for i, lc in enumerate(legal_lists):
        for k, card in enumerate(lc):
            actions[i, k] = bc.encode_action(obs_list[i], card)
            valid[i, k] = True

    a_t = torch.from_numpy(actions).to(device)
    v_t = torch.from_numpy(valid).to(device)
    if use_hist:
        h_t = torch.from_numpy(hists).to(device)
        np_t = torch.from_numpy(n_plies).to(device)
        q = model(s_t, a_t, h_t, np_t, valid_mask=v_t)
    else:
        q = model(s_t, a_t, valid_mask=v_t)
    q = q.masked_fill(~v_t, -1e9)
    best_idx = q.argmax(dim=-1).cpu().tolist()
    return [legal_lists[i][best_idx[i]] for i in range(n)]


def eval_vs_cpp_agent(
    model: torch.nn.Module,
    opponent_spec: str = "rule",
    decks: int = 200,
    seed: int = 12345,
    device: torch.device | None = None,
    config: bc.Config | None = None,
) -> dict[str, float]:
    """Paired duplicate card-play eval: PyTorch `model` vs C++ `opponent_spec`."""
    device = device or next(model.parameters()).device
    config = config or bc.Config.belgian()
    model.eval()
    t0 = time.perf_counter()

    # Pre-generate all decks & run fixed rule bidder so both mirrored runs share contracts.
    rng = random.Random(seed)
    active_deals: list[tuple[int, int, bc.Deal]] = []  # (deck_idx, model_team, deal)
    opp_agents: list[bc.Agent] = []
    skipped = 0
    for i in range(decks):
        deck = list(range(32))
        rng.shuffle(deck)
        dealer = i % 4
        d0 = bc.Deal(config, dealer, deck)
        bids = []
        while d0.phase in (bc.Phase.BID1, bc.Phase.BID2):
            b = bc.rule_bid(d0)
            bids.append(b)
            d0.bid(b)
        if d0.phase != bc.Phase.PLAY:
            skipped += 1
            continue
        d1 = bc.Deal(config, dealer, deck)
        for b in bids:
            d1.bid(b)
        active_deals.append((i, 0, d0))
        opp_agents.append(bc.make_agent(opponent_spec, seed * 1000 + 2 * i))
        active_deals.append((i, 1, d1))
        opp_agents.append(bc.make_agent(opponent_spec, seed * 1000 + 2 * i + 1))

    # Step all active deals in batched lockstep (32 plies max).
    for _ in range(32):
        model_indices = []
        model_deals = []
        for idx, (_, m_team, d) in enumerate(active_deals):
            if d.phase != bc.Phase.PLAY:
                continue
            p = d.current_player
            if (p & 1) == m_team:
                model_indices.append(idx)
                model_deals.append(d)
            else:
                d.play(opp_agents[idx].choose_card(d))
        if model_deals:
            cards = choose_cards_batch(model, model_deals, device)
            for idx, card in zip(model_indices, cards):
                active_deals[idx][2].play(card)

    # Aggregate paired duplicate margins.
    units_pairs = []
    raw_pairs = []
    for k in range(0, len(active_deals), 2):
        _, _, d0 = active_deals[k]
        _, _, d1 = active_deals[k + 1]
        r0 = bc.record_deal(d0.result, config)
        r1 = bc.record_deal(d1.result, config)
        u0 = r0.scores[0] - r0.scores[1]
        u1 = r1.scores[1] - r1.scores[0]
        raw0 = d0.result.credited[0] - d0.result.credited[1]
        raw1 = d1.result.credited[1] - d1.result.credited[0]
        units_pairs.append(0.5 * (u0 + u1))
        raw_pairs.append(0.5 * (raw0 + raw1))

    n = len(units_pairs)
    u_arr = np.asarray(units_pairs, dtype=np.float64)
    r_arr = np.asarray(raw_pairs, dtype=np.float64)
    u_mean = float(u_arr.mean()) if n else 0.0
    u_se = float(u_arr.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    r_mean = float(r_arr.mean()) if n else 0.0
    r_se = float(r_arr.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return {
        "decks": n,
        "skipped": skipped,
        "units": u_mean,
        "units_ci95": 1.96 * u_se,
        "raw": r_mean,
        "raw_ci95": 1.96 * r_se,
        "seconds": time.perf_counter() - t0,
    }


@torch.no_grad()
def eval_belief_per_trick(
    belief_net: BeliefNet,
    device: torch.device,
    num_deals: int = 256,
    seed: int = 99999,
) -> dict[str, object]:
    """Compare BeliefNet per-trick cross-entropy log-loss vs constrained-uniform."""
    belief_net.eval()
    env = bc.BatchPlayEnv(num_envs=num_deals, config=bc.Config.belgian(), seed=seed, bid_jitter=0.0)
    rng = np.random.default_rng(seed)

    model_loss_by_trick = np.zeros(8, dtype=np.float64)
    unif_loss_by_trick = np.zeros(8, dtype=np.float64)
    count_by_trick = np.zeros(8, dtype=np.int64)

    for ply in range(32):
        trick = ply // 4
        batch = env.observe_batch(include_history=True, include_beliefs=True)
        s_t = torch.from_numpy(batch["state"]).to(device)
        h_t = torch.from_numpy(batch["history"]).to(device)
        np_t = torch.from_numpy(batch["n_plies"]).to(device)
        al_t = torch.from_numpy(batch["belief_allowed"]).to(device)
        bt = batch["belief_targets"]  # (B, 32) int8
        ba = batch["belief_allowed"]  # (B, 32, 3) uint8

        logits = belief_net(s_t, h_t, np_t, al_t)  # (B, 32, 3)
        log_probs = F.log_softmax(logits, dim=-1).cpu().numpy()

        mask = bt >= 0
        if np.any(mask):
            rows, cols = np.nonzero(mask)
            targets = bt[rows, cols]
            n_allowed = ba[rows, cols].sum(axis=-1).astype(np.float64)
            # Only evaluate cards with >1 allowed seat (pinned cards have 0 entropy for both).
            amb = n_allowed > 1
            if np.any(amb):
                rows_a, cols_a, tgt_a = rows[amb], cols[amb], targets[amb]
                model_loss_by_trick[trick] += float(-log_probs[rows_a, cols_a, tgt_a].sum())
                unif_loss_by_trick[trick] += float(np.log(n_allowed[amb]).sum())
                count_by_trick[trick] += int(amb.sum())

        # Play rule cards on validation deals to get realistic trick progression.
        acts = np.empty(num_deals, dtype=np.int32)
        lc = batch["legal_cards"]
        for b in range(num_deals):
            c = bc.rule_choose_card(env.deal(b))
            acts[b] = int(np.where(lc[b] == c)[0][0])
        env.step_batch(acts)

    per_trick = {}
    for t in range(8):
        if count_by_trick[t] > 0:
            per_trick[t] = (
                model_loss_by_trick[t] / count_by_trick[t],
                unif_loss_by_trick[t] / count_by_trick[t],
            )
    tot = count_by_trick.sum()
    return {
        "model_loss": float(model_loss_by_trick.sum() / max(1, tot)),
        "uniform_loss": float(unif_loss_by_trick.sum() / max(1, tot)),
        "per_trick": per_trick,
    }
