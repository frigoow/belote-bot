"""Evaluation utilities for Bulgarian Bridge-Belot neural models:
- eval_vs_cpp: mirrored deals against C++ rulePolicy or searchCard.
- eval_belief_per_trick: cross-entropy of BgBeliefNet B_phi against the REAL
  exact DealSampler::exactMarginals() baseline on every trick 1..7.
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import bg_cpp as bgc
from bg.models import BgBeliefNet, BgCardQGRU, BgFastPolicy


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def eval_vs_cpp(
    model: nn.Module,
    device: torch.device,
    opp_spec: str = "policy",
    num_deals: int = 400,
    seed: int = 12345,
    threads: int = 12,
    is_fast_policy: bool = False,
    mode: str = "all",
    allow_double: bool = False,
) -> dict[str, float]:
    """Evaluate a Python Q-GRU or FastPolicy in mirrored deals vs a C++ opponent."""
    model.eval()
    sess = bgc.BgEvalSession(
        num_deals=num_deals, seed=seed, opp=opp_spec, threads=threads, mode=mode, allow_double=allow_double
    )
    idx_range = torch.arange(bgc.MAX_LEGAL, device=device).unsqueeze(0)

    with _autocast(device):
        while not sess.done:
            obs = sess.observe_active(include_history=not is_fast_policy)
            s_t = torch.from_numpy(obs["state"]).to(device)
            nl_t = torch.from_numpy(obs["n_legal"]).to(device)
            if is_fast_policy:
                logits32 = model(s_t).float()  # (N, 32)
                # Pick highest-logit legal card among obs["legal_cards"]? Wait: legal_actions[i,k,:32] has canonCard one-hot!
                a_t = torch.from_numpy(obs["legal_actions"][:, :, :32]).to(device)  # (N, K, 32)
                scores = torch.bmm(a_t.float(), logits32.unsqueeze(-1)).squeeze(-1)
                valid = idx_range < nl_t.unsqueeze(1)
                scores = scores.masked_fill(~valid, -1e9)
                act = scores.argmax(dim=-1).cpu().numpy().astype(np.int32)
            else:
                a_t = torch.from_numpy(obs["legal_actions"]).to(device)
                h_t = torch.from_numpy(obs["history"]).to(device)
                np_t = torch.from_numpy(obs["n_plies"]).to(device)
                valid = idx_range < nl_t.unsqueeze(1)
                q = model(s_t, a_t, h_t, np_t, valid_mask=valid).float()
                q = q.masked_fill(~valid, -1e9)
                act = q.argmax(dim=-1).cpu().numpy().astype(np.int32)
            sess.step_active(act)

    return sess.summary()


@torch.no_grad()
def eval_belief_per_trick(
    q_model: nn.Module,
    b_model: BgBeliefNet,
    device: torch.device,
    num_deals: int = 512,
    seed: int = 54321,
    threads: int = 8,
    mode: str = "all",
    allow_double: bool = False,
) -> dict[str, dict[int, float]]:
    """Compare BgBeliefNet CE directly against exact DealSampler::exactMarginals() per trick."""
    q_model.eval()
    b_model.eval()
    env = bgc.BgBatchPlayEnv(
        num_envs=num_deals,
        seed=seed,
        bid_noise=0.0,
        hanging_prob=0.0,
        threads=threads,
        mode=mode,
        allow_double=allow_double,
    )
    idx_range = torch.arange(bgc.MAX_LEGAL, device=device).unsqueeze(0)

    ce_model = {t: 0.0 for t in range(1, 9)}
    ce_sampler = {t: 0.0 for t in range(1, 9)}
    counts = {t: 0 for t in range(1, 9)}
    finished = np.zeros(num_deals, dtype=bool)

    with _autocast(device):
        for _step in range(28):
            if np.all(finished):
                break
            batch = env.observe_batch(include_history=True, include_beliefs=True)
            s_t = torch.from_numpy(batch["state"]).to(device)
            a_t = torch.from_numpy(batch["legal_actions"]).to(device)
            h_t = torch.from_numpy(batch["history"]).to(device)
            np_t = torch.from_numpy(batch["n_plies"]).to(device)
            nl_t = torch.from_numpy(batch["n_legal"]).to(device)
            bm_t = torch.from_numpy(batch["belief_marginals"]).to(device)
            bt_t = torch.from_numpy(batch["belief_targets"]).to(device).long()

            active_np = ~finished
            active_idx = torch.from_numpy(np.nonzero(active_np)[0]).to(device)
            if active_idx.numel() > 0:
                logits = b_model(s_t[active_idx], h_t[active_idx], np_t[active_idx], bm_t[active_idx])
                log_p_model = F.log_softmax(logits.float(), dim=-1)
                log_p_samp = torch.log(bm_t[active_idx].float().clamp(min=1e-6))

                bt_a = bt_t[active_idx]  # (A, 32)
                tricks = (np_t[active_idx] // 4 + 1).clamp(1, 8).cpu().numpy()
                valid_c = bt_a >= 0
                safe_t = bt_a.clamp(min=0).unsqueeze(-1)
                nll_m = -log_p_model.gather(-1, safe_t).squeeze(-1)  # (A, 32)
                nll_s = -log_p_samp.gather(-1, safe_t).squeeze(-1)

                nll_m_cpu = (nll_m * valid_c).sum(dim=-1).cpu().numpy()
                nll_s_cpu = (nll_s * valid_c).sum(dim=-1).cpu().numpy()
                cnt_cpu = valid_c.sum(dim=-1).cpu().numpy()
                for i, tr in enumerate(tricks):
                    t_int = int(tr)
                    ce_model[t_int] += float(nll_m_cpu[i])
                    ce_sampler[t_int] += float(nll_s_cpu[i])
                    counts[t_int] += int(cnt_cpu[i])

            valid = idx_range < nl_t.unsqueeze(1)
            q = q_model(s_t, a_t, h_t, np_t, valid_mask=valid).float().masked_fill(~valid, -1e9)
            act = q.argmax(dim=-1).cpu().numpy().astype(np.int32)
            done, _ = env.step_batch(act)
            finished |= done != 0

    tot_c = max(1, sum(counts.values()))
    res_m = {t: ce_model[t] / counts[t] for t in range(1, 9) if counts[t] > 0}
    res_s = {t: ce_sampler[t] / counts[t] for t in range(1, 9) if counts[t] > 0}
    res_m[0] = sum(ce_model.values()) / tot_c
    res_s[0] = sum(ce_sampler.values()) / tot_c
    return {"model": res_m, "sampler": res_s, "counts": counts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--opp", type=str, default="policy")
    ap.add_argument("--deals", type=int, default=500)
    ap.add_argument("--seed", type=int, default=54321)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--eval-belief", action="store_true")
    ap.add_argument("--eval-fast", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    q_model = BgCardQGRU().to(device)
    q_model.load_state_dict(ckpt["q_model"])

    if args.eval_fast:
        assert "fast_policy" in ckpt, "Checkpoint has no fast_policy"
        fp = BgFastPolicy().to(device)
        fp.load_state_dict(ckpt["fast_policy"])
        s = eval_vs_cpp(fp, device, opp_spec=args.opp, num_deals=args.deals, seed=args.seed, threads=args.threads, is_fast_policy=True)
        print(f"FastPolicy vs [{args.opp}]: {s['mean']:+.3f} +- {s['ci']:.3f} game pts/deal ({s['pairs']} mirrored pairs, seed={args.seed})")
    else:
        s = eval_vs_cpp(q_model, device, opp_spec=args.opp, num_deals=args.deals, seed=args.seed, threads=args.threads)
        print(f"Q-GRU vs [{args.opp}]: {s['mean']:+.3f} +- {s['ci']:.3f} game pts/deal ({s['pairs']} mirrored pairs, seed={args.seed})")

    if args.eval_belief and "b_model" in ckpt:
        b_model = BgBeliefNet().to(device)
        b_model.load_state_dict(ckpt["b_model"])
        br = eval_belief_per_trick(q_model, b_model, device, num_deals=512, seed=args.seed, threads=args.threads)
        print("Belief CE (lower is better) — B_phi vs exact DealSampler::exactMarginals():")
        for t in sorted(br["model"].keys()):
            label = "All" if t == 0 else f"T{t}"
            m_ce, s_ce = br["model"][t], br["sampler"][t]
            print(f"  {label:>3}: B_phi={m_ce:.4f} vs DealSampler={s_ce:.4f} (delta={m_ce - s_ce:+.4f})")


if __name__ == "__main__":
    main()
