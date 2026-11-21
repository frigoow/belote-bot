"""Manille v1.2.3 pure-DMC GPU-resident BF16 trainer for Bulgarian Bridge-Belot:
- Converged Dueling BgCardQGRU (2-block ResNet trunk + chronological GRU +
  training-only auxiliary 32x3 ownership head, discarded on BGQ1 export).
- Residual BgBeliefNet B_phi trained directly on top of exact
  DealSampler::exactMarginals() (exported as BGB1 -> wired into DealSampler).
- Compact 256x128 BgFastPolicy distilled from converged Q-GRU (exported as BGP1).
- Saves & restores FULL optimizer + scheduler state so multi-million-deal
  training runs resume smoothly without cold-AdamW collapse.

Usage:
  PYTHONPATH=build:python .venv/bin/python -m bg.train_dmc \
      --episodes 5000000 --envs 2048 --batch-size 2048 --train-belief \
      --save checkpoints/bg_q_gru_5m.pt --export-dir checkpoints/bg_models
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import bg_cpp as bgc
from bg.eval_bg import eval_belief_per_trick, eval_vs_cpp
from bg.models import BgBeliefNet, BgCardQGRU, BgFastPolicy

MAX_DECISIONS = 28  # Trick 8 (4 cards) is always auto-played by bg::Round.


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class BgEpisodeBuffers:
    """Pre-allocated GPU-resident BF16 buffers for one full B-deal batch."""

    def __init__(self, bsz: int, device: torch.device, include_beliefs: bool):
        self.bsz = bsz
        self.device = device
        self.include_beliefs = include_beliefs
        dt = torch.bfloat16 if device.type == "cuda" else torch.float32

        self.states = torch.empty((MAX_DECISIONS, bsz, bgc.STATE_DIM), dtype=dt, device=device)
        self.hists = torch.zeros((MAX_DECISIONS, bsz, bgc.MAX_PLIES, bgc.PLY_DIM), dtype=dt, device=device)
        self.n_plies = torch.zeros((MAX_DECISIONS, bsz), dtype=torch.int32, device=device)
        self.legal_actions = torch.empty((MAX_DECISIONS, bsz, bgc.MAX_LEGAL, bgc.ACTION_DIM), dtype=dt, device=device)
        self.n_legal = torch.empty((MAX_DECISIONS, bsz), dtype=torch.int8, device=device)
        self.chosen_idx = torch.empty((MAX_DECISIONS, bsz), dtype=torch.int8, device=device)
        self.teams = torch.empty((MAX_DECISIONS, bsz), dtype=torch.int8, device=device)
        self.valid_step = torch.zeros((MAX_DECISIONS, bsz), dtype=torch.bool, device=device)
        self.returns = torch.zeros((MAX_DECISIONS, bsz), dtype=torch.float32, device=device)

        if include_beliefs:
            self.bt = torch.empty((MAX_DECISIONS, bsz, 32), dtype=torch.int8, device=device)
            self.bm = torch.empty((MAX_DECISIONS, bsz, 32, 3), dtype=dt, device=device)
        else:
            self.bt = None
            self.bm = None


def collect_episodes_gpu(
    env: bgc.BgBatchPlayEnv,
    q_model: BgCardQGRU,
    buf: BgEpisodeBuffers,
    eps: float,
    round_seed: int,
    rng: np.random.Generator,
    collect_targets: bool = True,
    collect_marginals: bool = False,
) -> int:
    """Reset all B envs and play every deal cleanly to completion in <= 28 steps."""
    bsz = env.num_envs
    device = buf.device
    idx_range = torch.arange(bgc.MAX_LEGAL, device=device).unsqueeze(0)

    env.reset_all(round_seed)
    buf.valid_step.zero_()
    finished_np = np.zeros(bsz, dtype=bool)
    final_rew_t0 = np.zeros(bsz, dtype=np.float32)

    q_model.eval()
    steps_run = 0
    with torch.no_grad(), _autocast(device):
        for step in range(MAX_DECISIONS):
            if np.all(finished_np):
                break
            steps_run = step + 1
            active_np = ~finished_np
            batch = env.observe_batch(
                include_history=True,
                include_belief_targets=collect_targets,
                include_belief_marginals=collect_marginals,
            )

            s_t = torch.from_numpy(batch["state"]).to(device, non_blocking=True)
            a_t = torch.from_numpy(batch["legal_actions"]).to(device, non_blocking=True)
            np_np = batch["n_plies"]
            np_t = torch.from_numpy(np_np).to(device, non_blocking=True)
            nl_np = batch["n_legal"]
            nl_t = torch.from_numpy(nl_np).to(device, non_blocking=True)
            ct_t = torch.from_numpy(batch["current_team"]).to(device, non_blocking=True)
            max_p = int(np_np.max()) if bsz > 0 else 0
            if max_p > 0:
                h_slice = torch.from_numpy(batch["history"][:, :max_p, :]).to(device, non_blocking=True)
                buf.hists[step, :, :max_p, :].copy_(h_slice)
                h_in = h_slice
            else:
                h_in = buf.hists[0, :, :1, :]

            buf.states[step].copy_(s_t)
            buf.n_plies[step].copy_(np_t)
            buf.legal_actions[step].copy_(a_t)
            buf.n_legal[step].copy_(nl_t)
            buf.teams[step].copy_(ct_t)
            buf.valid_step[step].copy_(torch.from_numpy(active_np).to(device, non_blocking=True))

            if collect_targets and buf.bt is not None:
                buf.bt[step].copy_(torch.from_numpy(batch["belief_targets"]).to(device, non_blocking=True))
            if collect_marginals and buf.bm is not None:
                buf.bm[step].copy_(torch.from_numpy(batch["belief_marginals"]).to(device, non_blocking=True))

            valid = idx_range < nl_t.unsqueeze(1)
            q_vals = q_model(s_t, a_t, h_in, np_t, valid_mask=valid).float()
            q_vals = q_vals.masked_fill(~valid, -1e9)
            greedy = q_vals.argmax(dim=-1).cpu().numpy().astype(np.int32)

            if eps > 0.0:
                explore = (rng.random(bsz) < eps) & (nl_np > 1)
                rand_idx = (rng.integers(0, 10000, size=bsz) % nl_np).astype(np.int32)
                action_idx = np.where(explore, rand_idx, greedy)
            else:
                action_idx = greedy

            buf.chosen_idx[step].copy_(torch.from_numpy(action_idx.astype(np.int8)).to(device, non_blocking=True))
            done, rew_t0 = env.step_batch(action_idx)
            newly_done = active_np & (done != 0)
            if np.any(newly_done):
                final_rew_t0[newly_done] = rew_t0[newly_done]
                finished_np |= newly_done

    # Safety: in case any deal had 28+ decisions (impossible since trick 8 is auto-played), exclude unfinished.
    if not np.all(finished_np):
        unfin_t = torch.from_numpy(~finished_np).to(device)
        buf.valid_step[:, unfin_t] = False

    r0_t = torch.from_numpy(final_rew_t0).to(device, non_blocking=True)
    signs = torch.where(buf.teams[:steps_run] == 0, 1.0, -1.0).to(torch.float32)
    buf.returns[:steps_run].copy_(signs * r0_t.unsqueeze(0))
    return steps_run


def distill_fast_policy(
    q_model: BgCardQGRU,
    device: torch.device,
    num_deals: int = 60000,
    batch_envs: int = 1024,
    epochs: int = 8,
    temperature: float = 0.04,
    save_bin: Path | None = None,
    mode: str = "all",
    allow_double: bool = False,
) -> BgFastPolicy:
    """Distill compact 256x128 BgFastPolicy from converged BgCardQGRU."""
    print(f"\n--- Distilling 256x128 BgFastPolicy (mode={mode}) over {num_deals:,} deals (T={temperature}) ---", flush=True)
    q_model.eval()
    fast = BgFastPolicy().to(device)
    opt = torch.optim.AdamW(fast.parameters(), lr=1.5e-3, weight_decay=1e-4)

    env = bgc.BgBatchPlayEnv(
        num_envs=batch_envs,
        seed=777,
        bid_noise=0.0,
        hanging_prob=0.0,
        threads=12,
        mode=mode,
        allow_double=allow_double,
    )
    n_rounds = max(1, num_deals // batch_envs)
    idx_range = torch.arange(bgc.MAX_LEGAL, device=device).unsqueeze(0)
    empty_h = torch.zeros((batch_envs, 1, bgc.PLY_DIM), dtype=torch.float32, device=device)

    states_all, legal_masks_all, target_probs_all = [], [], []
    t0 = time.perf_counter()
    with torch.no_grad(), _autocast(device):
        for r_idx in range(n_rounds):
            env.reset_all(777000 + r_idx)
            finished = np.zeros(batch_envs, dtype=bool)
            for _step in range(MAX_DECISIONS):
                if np.all(finished):
                    break
                batch = env.observe_batch(include_history=True, include_beliefs=False)
                s_np = batch["state"]
                la_np = batch["legal_actions"]
                nl_np = batch["n_legal"]
                np_np = batch["n_plies"]
                max_p = int(np_np.max())

                s_t = torch.from_numpy(s_np).to(device)
                a_t = torch.from_numpy(la_np).to(device)
                h_t = torch.from_numpy(batch["history"][:, :max_p, :]).to(device) if max_p > 0 else empty_h
                np_t = torch.from_numpy(np_np).to(device)
                valid = idx_range < torch.from_numpy(nl_np).to(device).unsqueeze(1)

                q = q_model(s_t, a_t, h_t, np_t, valid_mask=valid).float().masked_fill(~valid, -1e9)
                probs = F.softmax(q / temperature, dim=-1).cpu().numpy()
                greedy = q.argmax(dim=-1).cpu().numpy().astype(np.int32)

                keep = np.nonzero((~finished) & (nl_np > 1))[0]
                if len(keep) > 0:
                    # legal_actions[b, k, :32] is the one-hot canonical card vector!
                    oh = la_np[keep, :, :32]  # (M, K, 32)
                    p_keep = probs[keep]  # (M, K)
                    # Mask out padded action slots cleanly.
                    valid_k = np.arange(bgc.MAX_LEGAL)[None, :] < nl_np[keep, None]
                    p_keep = p_keep * valid_k
                    tgt32 = np.einsum("mk,mkc->mc", p_keep, oh).astype(np.float32)
                    lmask32 = np.einsum("mk,mkc->mc", valid_k.astype(np.float32), oh).astype(np.float32)
                    states_all.append(s_np[keep])
                    legal_masks_all.append(lmask32)
                    target_probs_all.append(tgt32)

                done, _ = env.step_batch(greedy)
                finished |= done != 0

    X = np.concatenate(states_all, axis=0)
    M = np.concatenate(legal_masks_all, axis=0)
    Y = np.concatenate(target_probs_all, axis=0)
    print(f"Collected {len(X):,} teacher decisions in {time.perf_counter() - t0:.1f}s", flush=True)

    ds_n = len(X)
    mb = 2048
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * math.ceil(ds_n / mb))
    rng = np.random.default_rng(1)
    fast.train()
    for ep in range(epochs):
        perm = rng.permutation(ds_n)
        tot_loss = tot_acc = 0.0
        steps = 0
        for start in range(0, ds_n, mb):
            idx = perm[start : start + mb]
            xb = torch.from_numpy(X[idx]).to(device)
            mb_t = torch.from_numpy(M[idx]).to(device)
            yb = torch.from_numpy(Y[idx]).to(device)
            logits = fast(xb).masked_fill(mb_t == 0, -1e9)
            log_p = F.log_softmax(logits, dim=-1)
            loss = -(yb * log_p).sum(dim=-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += float(loss.item())
            tot_acc += float((logits.argmax(dim=-1) == yb.argmax(dim=-1)).float().mean().item())
            steps += 1
        print(
            f"  BgFastPolicy epoch {ep + 1}/{epochs}: kl_ce={tot_loss / steps:.4f}, top1={100 * tot_acc / steps:.1f}%",
            flush=True,
        )

    if save_bin is not None:
        fast.export_bin(save_bin)
        print(f"Exported BgFastPolicy binary weights to {save_bin}", flush=True)
    return fast


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("all", "suit", "nt", "at"), default="all")
    ap.add_argument("--allow-double", action="store_true", help="Allow Double/Redouble during training (off for clean single-hand Manille mode)")
    ap.add_argument("--hanging-prob", type=float, default=0.0)
    ap.add_argument("--episodes", type=int, default=5_000_000)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--env-threads", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--replay-rounds", type=int, default=4, help="GPU BF16 rolling FIFO replay window in rounds (eliminates single-batch oscillation)")
    ap.add_argument("--mb-per-round", type=int, default=32, help="Minibatch gradient steps per round sampled across the rolling replay window")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--eps-start", type=float, default=0.08)
    ap.add_argument("--eps-end", type=float, default=0.01)
    ap.add_argument("--bid-noise", type=float, default=0.0)
    ap.add_argument("--aux-own-coef", type=float, default=0.02)
    ap.add_argument("--train-belief", action="store_true")
    ap.add_argument("--belief-every", type=int, default=4)
    ap.add_argument("--distill-fast", action="store_true")
    ap.add_argument("--distill-only", action="store_true", help="Skip DMC training and distill BgFastPolicy from --resume")
    ap.add_argument("--distill-deals", type=int, default=80000)
    ap.add_argument("--eval-every", type=int, default=250_000)
    ap.add_argument("--eval-decks", type=int, default=800)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--save", type=Path, default=Path("checkpoints/bg_q_gru.pt"))
    ap.add_argument("--export-dir", type=Path, default=Path("checkpoints/bg_models"))
    ap.add_argument("--log-file", type=Path, default=None, help="Log file path (defaults to runs/<save_stem>_<ts>.log)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    log_path = args.log_file
    if log_path is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path("runs") / f"{args.save.stem}_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("a", buffering=1)

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams
        def write(self, data):
            for s in self._streams:
                s.write(data)
            return len(data)
        def flush(self):
            for s in self._streams:
                s.flush()

    sys.stdout = _Tee(sys.stdout, log_fh)
    sys.stderr = _Tee(sys.stderr, log_fh)
    print(f"Logging training run to {log_path}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | STATE_DIM={bgc.STATE_DIM} PLY_DIM={bgc.PLY_DIM} ACTION_DIM={bgc.ACTION_DIM}", flush=True)

    q_model = BgCardQGRU().to(device)
    b_model = BgBeliefNet().to(device) if args.train_belief else None

    total_rounds = max(1, math.ceil(args.episodes / args.envs))
    q_opt = torch.optim.AdamW(q_model.parameters(), lr=args.lr, weight_decay=1e-4)
    q_sched = torch.optim.lr_scheduler.CosineAnnealingLR(q_opt, T_max=total_rounds, eta_min=args.min_lr)
    b_opt = torch.optim.AdamW(b_model.parameters(), lr=args.lr, weight_decay=1e-4) if b_model else None
    b_rounds = max(1, math.ceil(total_rounds / max(1, args.belief_every)))
    b_sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(b_opt, T_max=b_rounds, eta_min=args.min_lr)
        if b_opt
        else None
    )

    episodes_done = 0
    best_vs_policy = -1e9
    best_belief_ce = 1e9
    start_round = 0

    if args.resume is not None and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        q_model.load_state_dict(ckpt["q_model"])
        if "q_opt" in ckpt:
            q_opt.load_state_dict(ckpt["q_opt"])
        if "q_sched" in ckpt:
            q_sched.load_state_dict(ckpt["q_sched"])
        if b_model is not None and "b_model" in ckpt:
            b_model.load_state_dict(ckpt["b_model"])
            if "b_opt" in ckpt:
                b_opt.load_state_dict(ckpt["b_opt"])
            if "b_sched" in ckpt:
                b_sched.load_state_dict(ckpt["b_sched"])
        episodes_done = int(ckpt.get("episodes_done", 0))
        best_vs_policy = float(ckpt.get("best_vs_policy", -1e9))
        best_belief_ce = float(ckpt.get("best_belief_ce", 1e9))
        start_round = episodes_done // args.envs
        print(
            f"Resumed {args.resume} at {episodes_done:,} deals "
            f"(best Q {best_vs_policy:+.3f}, best B {best_belief_ce:.4f})",
            flush=True,
        )

    if args.distill_only:
        assert args.resume is not None and args.resume.exists(), "--distill-only requires an existing --resume checkpoint"
        args.export_dir.mkdir(parents=True, exist_ok=True)
        fast_policy = distill_fast_policy(
            q_model,
            device,
            num_deals=args.distill_deals,
            save_bin=args.export_dir / "bg_fast_policy.bin",
            mode=args.mode,
            allow_double=args.allow_double,
        )
        ev_fp = eval_vs_cpp(
            fast_policy,
            device,
            opp_spec="policy",
            num_deals=args.eval_decks,
            seed=54321,
            threads=args.env_threads,
            is_fast_policy=True,
            mode=args.mode,
            allow_double=args.allow_double,
        )
        print(
            f"  >>> Distilled BgFastPolicy (mode={args.mode}) vs rulePolicy (seed 54321): "
            f"{ev_fp['mean']:+.3f} +- {ev_fp['ci']:.3f} game pts/deal ({ev_fp['pairs']} pairs)",
            flush=True,
        )
        return

    need_belief_data = args.train_belief or args.aux_own_coef > 0.0
    env = bgc.BgBatchPlayEnv(
        num_envs=args.envs,
        seed=args.seed,
        bid_noise=args.bid_noise,
        hanging_prob=args.hanging_prob,
        threads=args.env_threads,
        mode=args.mode,
        allow_double=args.allow_double,
    )
    print(
        f"Training mode={args.mode} allow_double={args.allow_double} hanging_prob={args.hanging_prob} "
        f"replay_rounds={args.replay_rounds} mb_per_round={args.mb_per_round}",
        flush=True,
    )
    buf = BgEpisodeBuffers(args.envs, device, include_beliefs=need_belief_data)
    rng = np.random.default_rng(args.seed + 999)
    idx_range = torch.arange(bgc.MAX_LEGAL, device=device).unsqueeze(0)

    # GPU BF16 rolling FIFO replay ring of the last `replay_rounds` self-play rounds.
    ring_cap = max(1, args.replay_rounds)
    ring_q: list[tuple] = []
    ring_ptr = 0

    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.export_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    next_eval = ((episodes_done // args.eval_every) + 1) * args.eval_every
    round_idx = start_round

    def save_ckpt(path: Path, fast_policy: BgFastPolicy | None = None) -> None:
        payload = {
            "q_model": q_model.state_dict(),
            "q_opt": q_opt.state_dict(),
            "q_sched": q_sched.state_dict(),
            "episodes_done": episodes_done,
            "best_vs_policy": best_vs_policy,
            "best_belief_ce": best_belief_ce,
            "encoder_version": bgc.ENCODER_VERSION,
        }
        if b_model is not None:
            payload["b_model"] = b_model.state_dict()
            payload["b_opt"] = b_opt.state_dict()
            payload["b_sched"] = b_sched.state_dict()
        if fast_policy is not None:
            payload["fast_policy"] = fast_policy.state_dict()
        torch.save(payload, path)

    while episodes_done < args.episodes:
        frac = min(1.0, episodes_done / max(1, args.episodes))
        eps = args.eps_start + (args.eps_end - args.eps_start) * frac

        train_belief_now = (b_model is not None) and (round_idx % max(1, args.belief_every) == 0)
        steps_run = collect_episodes_gpu(
            env,
            q_model,
            buf,
            eps=eps,
            round_seed=args.seed * 1000000 + round_idx,
            rng=rng,
            collect_targets=need_belief_data,
            collect_marginals=train_belief_now,
        )
        episodes_done += args.envs
        round_idx += 1

        # Flatten valid completed transitions for ply-bucketed training.
        v_flat = buf.valid_step[:steps_run].reshape(-1)
        valid_idx = torch.nonzero(v_flat, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue

        states_new = buf.states[:steps_run].reshape(-1, bgc.STATE_DIM)[valid_idx]
        hists_new = buf.hists[:steps_run].reshape(-1, bgc.MAX_PLIES, bgc.PLY_DIM)[valid_idx]
        nplies_new = buf.n_plies[:steps_run].reshape(-1)[valid_idx]
        acts_new = buf.legal_actions[:steps_run].reshape(-1, bgc.MAX_LEGAL, bgc.ACTION_DIM)[valid_idx]
        nl_new = buf.n_legal[:steps_run].reshape(-1)[valid_idx].long()
        chosen_new = buf.chosen_idx[:steps_run].reshape(-1)[valid_idx].long()
        ret_new = buf.returns[:steps_run].reshape(-1)[valid_idx]
        bt_new = buf.bt[:steps_run].reshape(-1, 32)[valid_idx].long() if need_belief_data else None
        bm_new = buf.bm[:steps_run].reshape(-1, 32, 3)[valid_idx] if train_belief_now else None

        # Push this round into the rolling GPU FIFO replay window.
        entry = (states_new, hists_new, nplies_new, acts_new, nl_new, chosen_new, ret_new, bt_new)
        if len(ring_q) < ring_cap:
            ring_q.append(entry)
        else:
            ring_q[ring_ptr] = entry
        ring_ptr = (ring_ptr + 1) % ring_cap

        if len(ring_q) == 1:
            states_f, hists_f, nplies_f, acts_f, nl_f, chosen_f, ret_f, bt_f = ring_q[0]
        else:
            states_f = torch.cat([e[0] for e in ring_q], dim=0)
            hists_f = torch.cat([e[1] for e in ring_q], dim=0)
            nplies_f = torch.cat([e[2] for e in ring_q], dim=0)
            acts_f = torch.cat([e[3] for e in ring_q], dim=0)
            nl_f = torch.cat([e[4] for e in ring_q], dim=0)
            chosen_f = torch.cat([e[5] for e in ring_q], dim=0)
            ret_f = torch.cat([e[6] for e in ring_q], dim=0)
            bt_f = torch.cat([e[7] for e in ring_q], dim=0) if need_belief_data else None

        q_model.train()
        if b_model is not None:
            b_model.train()

        q_loss_sum = own_loss_sum = b_loss_sum = 0.0
        n_mb = n_b_mb = 0

        # 4 ply-length buckets sampled across the rolling replay window, shuffled so updates
        # are i.i.d. across tricks AND smooth across the last `replay_rounds` snapshots!
        buckets = (nplies_f // 8).clamp(0, 3)
        mb_schedule: list[tuple[int, torch.Tensor]] = []
        for bkt in range(4):
            b_indices = torch.nonzero(buckets == bkt, as_tuple=False).squeeze(-1)
            nb = b_indices.numel()
            if nb == 0:
                continue
            perm = b_indices[torch.randperm(nb, device=device)]
            h_len = max(1, int(nplies_f[b_indices].max().item()))
            for start in range(0, nb, args.batch_size):
                mb_schedule.append((h_len, perm[start : start + args.batch_size]))

        if mb_schedule:
            order = rng.permutation(len(mb_schedule))
            max_steps = min(len(order), max(1, args.mb_per_round))
            for oi in order[:max_steps]:
                h_len, mb_idx = mb_schedule[int(oi)]
                s_mb = states_f[mb_idx]
                h_mb = hists_f[mb_idx, :h_len, :]
                np_mb = nplies_f[mb_idx]
                a_mb = acts_f[mb_idx]
                nl_mb = nl_f[mb_idx]
                ch_mb = chosen_f[mb_idx]
                ret_mb = ret_f[mb_idx]
                valid_a = idx_range < nl_mb.unsqueeze(1)

                with _autocast(device):
                    if args.aux_own_coef > 0.0 and bt_f is not None:
                        q_all, own_logits = q_model(s_mb, a_mb, h_mb, np_mb, valid_mask=valid_a, return_aux_own=True)
                    else:
                        q_all = q_model(s_mb, a_mb, h_mb, np_mb, valid_mask=valid_a)
                        own_logits = None

                q_taken = q_all.float().gather(1, ch_mb.unsqueeze(1)).squeeze(1)
                q_mse = F.mse_loss(q_taken, ret_mb)
                loss = q_mse

                if own_logits is not None and bt_f is not None:
                    bt_mb = bt_f[mb_idx]
                    own_ce = F.cross_entropy(own_logits.float().reshape(-1, 3), bt_mb.reshape(-1), ignore_index=-1)
                    loss = loss + args.aux_own_coef * own_ce
                    own_loss_sum += float(own_ce.item())

                q_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_model.parameters(), 2.0)
                q_opt.step()
                q_loss_sum += float(q_mse.item())
                n_mb += 1

        # Train residual BeliefNet B_phi on the fresh round's exact DealSampler marginals.
        if train_belief_now and b_model is not None and b_opt is not None and bt_new is not None and bm_new is not None:
            b_buckets = (nplies_new // 8).clamp(0, 3)
            b_sched_list: list[tuple[int, torch.Tensor]] = []
            for bkt in range(4):
                b_indices = torch.nonzero(b_buckets == bkt, as_tuple=False).squeeze(-1)
                nb = b_indices.numel()
                if nb == 0:
                    continue
                perm = b_indices[torch.randperm(nb, device=device)]
                h_len = max(1, int(nplies_new[b_indices].max().item()))
                for start in range(0, nb, args.batch_size):
                    b_sched_list.append((h_len, perm[start : start + args.batch_size]))
            for oi in rng.permutation(len(b_sched_list)):
                h_len, mb_idx = b_sched_list[int(oi)]
                with _autocast(device):
                    b_logits = b_model(states_new[mb_idx], hists_new[mb_idx, :h_len, :], nplies_new[mb_idx], bm_new[mb_idx])
                b_ce = F.cross_entropy(b_logits.float().reshape(-1, 3), bt_new[mb_idx].reshape(-1), ignore_index=-1)
                b_opt.zero_grad(set_to_none=True)
                b_ce.backward()
                torch.nn.utils.clip_grad_norm_(b_model.parameters(), 2.0)
                b_opt.step()
                b_loss_sum += float(b_ce.item())
                n_b_mb += 1

        q_sched.step()
        if b_sched is not None and train_belief_now:
            b_sched.step()

        if round_idx % 25 == 0 or episodes_done >= next_eval or episodes_done >= args.episodes:
            elapsed = time.perf_counter() - t_start
            dps = (episodes_done - start_round * args.envs) / max(elapsed, 1e-6)
            lr_now = q_opt.param_groups[0]["lr"]
            msg = (
                f"[{episodes_done:>9,}/{args.episodes:,}] "
                f"{dps:6.0f} d/s | eps={eps:.3f} lr={lr_now:.2e} | "
                f"q_mse={q_loss_sum / max(1, n_mb):.4f}"
            )
            if args.aux_own_coef > 0.0:
                msg += f" own_ce={own_loss_sum / max(1, n_mb):.4f}"
            if train_belief_now:
                msg += f" b_ce={b_loss_sum / max(1, n_b_mb):.4f}"
            print(msg, flush=True)

        if episodes_done >= next_eval or episodes_done >= args.episodes:
            next_eval += args.eval_every
            # Selection eval on fixed seed 12345 (held-out reporting uses seed 54321).
            ev = eval_vs_cpp(
                q_model,
                device,
                opp_spec="policy",
                num_deals=args.eval_decks,
                seed=12345,
                threads=args.env_threads,
                mode=args.mode,
                allow_double=args.allow_double,
            )
            print(
                f"  >>> EVAL ({episodes_done:,}, mode={args.mode}) Q-GRU vs rulePolicy: "
                f"{ev['mean']:+.3f} +- {ev['ci']:.3f} game pts/deal ({ev['pairs']} mirrored pairs)",
                flush=True,
            )
            if b_model is not None:
                br = eval_belief_per_trick(
                    q_model,
                    b_model,
                    device,
                    num_deals=256,
                    seed=12345,
                    threads=args.env_threads,
                    mode=args.mode,
                    allow_double=args.allow_double,
                )
                print(
                    f"  >>> BELIEF All={br['model'][0]:.4f} vs DealSampler={br['sampler'][0]:.4f} "
                    f"(T1 {br['model'].get(1,0):.4f}/{br['sampler'].get(1,0):.4f}, "
                    f"T4 {br['model'].get(4,0):.4f}/{br['sampler'].get(4,0):.4f}, "
                    f"T7 {br['model'].get(7,0):.4f}/{br['sampler'].get(7,0):.4f})",
                    flush=True,
                )
                if br["model"][0] < best_belief_ce:
                    best_belief_ce = br["model"][0]
                    b_model.export_bin(args.export_dir / "bg_belief.bin")
                    torch.save({"b_model": b_model.state_dict(), "best_belief_ce": best_belief_ce, "episodes_done": episodes_done},
                               args.save.with_name(args.save.stem + "_best_belief.pt"))
                    print(f"  >>> New best B_phi ({best_belief_ce:.4f})! Exported bg_belief.bin.", flush=True)
            save_ckpt(args.save)
            if ev["mean"] > best_vs_policy:
                best_vs_policy = ev["mean"]
                best_path = args.save.with_name(args.save.stem + "_best.pt")
                save_ckpt(best_path)
                q_model.export_bin(args.export_dir / "bg_q_gru.bin")
                print(f"  >>> New best Q-GRU ({best_vs_policy:+.3f})! Saved {best_path} & exported bg_q_gru.bin.", flush=True)

    # Load best checkpoint for final export + FastPolicy distillation.
    best_path = args.save.with_name(args.save.stem + "_best.pt")
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        q_model.load_state_dict(best_ckpt["q_model"])
        if b_model is not None and "b_model" in best_ckpt:
            b_model.load_state_dict(best_ckpt["b_model"])

    q_model.export_bin(args.export_dir / "bg_q_gru.bin")
    if b_model is not None:
        b_model.export_bin(args.export_dir / "bg_belief.bin")

    fast_policy = None
    if args.distill_fast:
        fast_policy = distill_fast_policy(
            q_model,
            device,
            num_deals=args.distill_deals,
            save_bin=args.export_dir / "bg_fast_policy.bin",
            mode=args.mode,
            allow_double=args.allow_double,
        )
        ev_fp = eval_vs_cpp(
            fast_policy,
            device,
            opp_spec="policy",
            num_deals=args.eval_decks,
            seed=12345,
            threads=args.env_threads,
            is_fast_policy=True,
            mode=args.mode,
            allow_double=args.allow_double,
        )
        print(
            f"  >>> FastPolicy vs rulePolicy: {ev_fp['mean']:+.3f} +- {ev_fp['ci']:.3f} game pts/deal",
            flush=True,
        )
        save_ckpt(args.save, fast_policy=fast_policy)


if __name__ == "__main__":
    main()
