"""M2 GPU-resident BF16 Dueling DMC trainer for CardQMLP / CardQGRU, BeliefNet B_phi, and FastPolicy.

Key optimizations for RTX 3060 (12 GB VRAM) + 16 CPU cores:
- GPU-resident BF16 episode buffers eliminate per-minibatch CPU->GPU copies.
- Ply-bucketed GRU training (4 trick-pair buckets) halves average GRU sequence
  length while keeping exact chronological hidden states.
- Active-action trimming in DuelingQHead skips padded legal-move slots.
- Dueling Q(s, a) = V(s) + A(s, a) - mean_{a' in legal} A(s, a') with exact MSE
  loss and DouZero-style low epsilon (0.08 -> 0.015).

Usage examples:
  # Full Dueling Q-GRU + BeliefNet B_phi + distilled 256x128 FastPolicy
  PYTHONPATH=build:python .venv/bin/python -m belote.train_dmc \
      --arch gru --episodes 1000000 --envs 1024 --train-belief --distill-fast \
      --save checkpoints/q_gru_1m.pt
"""

from __future__ import annotations

import argparse
import contextlib
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import belote_cpp as bc
from belote.eval_play import eval_belief_per_trick, eval_vs_cpp_agent
from belote.models import BeliefNet, FastPolicy, build_q_model


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class EpisodeBuffers:
    """Pre-allocated GPU-resident buffers for one synchronized B-deal rollout."""

    def __init__(self, bsz: int, device: torch.device, use_hist: bool, include_beliefs: bool):
        self.bsz = bsz
        self.device = device
        self.use_hist = use_hist
        self.include_beliefs = include_beliefs
        dt = torch.bfloat16 if device.type == "cuda" else torch.float32

        self.states = torch.empty((32, bsz, bc.STATE_DIM), dtype=dt, device=device)
        self.legal_actions = torch.empty((32, bsz, bc.MAX_LEGAL, bc.ACTION_DIM), dtype=dt, device=device)
        self.n_legal = torch.empty((32, bsz), dtype=torch.int8, device=device)
        self.chosen_idx = torch.empty((32, bsz), dtype=torch.int8, device=device)
        self.teams = torch.empty((32, bsz), dtype=torch.int8, device=device)
        self.returns = torch.empty((32, bsz), dtype=torch.float32, device=device)
        self.n_plies = torch.arange(32, dtype=torch.int32, device=device).unsqueeze(1).expand(32, bsz).contiguous()
        if use_hist:
            self.hists = torch.zeros((32, bsz, bc.MAX_PLIES, bc.PLY_DIM), dtype=dt, device=device)
        else:
            self.hists = None
        if include_beliefs:
            self.bt = torch.empty((32, bsz, 32), dtype=torch.int8, device=device)
            self.ba = torch.empty((32, bsz, 32, 3), dtype=torch.uint8, device=device)
        else:
            self.bt = None
            self.ba = None


def collect_episodes_gpu(
    env: bc.BatchPlayEnv,
    q_model: nn.Module,
    buf: EpisodeBuffers,
    eps: float,
    collect_beliefs: bool,
    rng: np.random.Generator,
) -> None:
    """Runs all B envs through one synchronized 32-ply deal directly into GPU buffers."""
    bsz = env.num_envs
    device = buf.device
    q_uses_hist = q_model.arch == "gru"
    need_hist = q_uses_hist or collect_beliefs
    idx_range = torch.arange(bc.MAX_LEGAL, device=device).unsqueeze(0)

    q_model.eval()
    final_rew_t0 = np.zeros(bsz, dtype=np.float32)
    with torch.no_grad(), _autocast(device):
        for ply in range(32):
            batch = env.observe_batch(include_history=need_hist, include_beliefs=collect_beliefs)
            s_t = torch.from_numpy(batch["state"]).to(device, non_blocking=True)
            a_t = torch.from_numpy(batch["legal_actions"]).to(device, non_blocking=True)
            nl_np = batch["n_legal"]
            nl_t = torch.from_numpy(nl_np).to(device, non_blocking=True)
            ct_t = torch.from_numpy(batch["current_team"]).to(device, non_blocking=True)

            buf.states[ply].copy_(s_t)
            buf.legal_actions[ply].copy_(a_t)
            buf.n_legal[ply].copy_(nl_t)
            buf.teams[ply].copy_(ct_t)

            if need_hist and ply > 0:
                h_slice = torch.from_numpy(batch["history"][:, :ply, :]).to(device, non_blocking=True)
                buf.hists[ply, :, :ply, :].copy_(h_slice)
            else:
                h_slice = None

            if collect_beliefs:
                buf.bt[ply].copy_(torch.from_numpy(batch["belief_targets"]).to(device, non_blocking=True))
                buf.ba[ply].copy_(torch.from_numpy(batch["belief_allowed"]).to(device, non_blocking=True))

            valid = idx_range < nl_t.unsqueeze(1)
            if q_uses_hist:
                np_t = buf.n_plies[ply]
                h_in = h_slice if h_slice is not None else buf.hists[0, :, :1, :]
                q_vals = q_model(s_t, a_t, h_in, np_t, valid_mask=valid)
            else:
                q_vals = q_model(s_t, a_t, valid_mask=valid)

            q_vals = q_vals.masked_fill(~valid, -1e9)
            greedy = q_vals.argmax(dim=-1).cpu().numpy().astype(np.int32)

            if eps > 0.0:
                explore = (rng.random(bsz) < eps) & (nl_np > 1)
                rand_idx = (rng.integers(0, 10000, size=bsz) % nl_np).astype(np.int32)
                action_idx = np.where(explore, rand_idx, greedy)
            else:
                action_idx = greedy

            buf.chosen_idx[ply].copy_(torch.from_numpy(action_idx.astype(np.int8)).to(device, non_blocking=True))
            done, rew_t0 = env.step_batch(action_idx)
            if ply == 31:
                assert np.all(done == 1), "Expected all synchronized deals to finish on ply 31"
                final_rew_t0 = rew_t0

    r0_t = torch.from_numpy(final_rew_t0).to(device, non_blocking=True)  # (B,)
    signs = torch.where(buf.teams == 0, 1.0, -1.0).to(torch.float32)
    buf.returns.copy_(signs * r0_t.unsqueeze(0))


def distill_fast_policy(
    q_model: nn.Module,
    device: torch.device,
    num_deals: int = 50000,
    batch_envs: int = 512,
    epochs: int = 6,
    temperature: float = 0.04,
    save_bin: Path | None = None,
) -> FastPolicy:
    """Distill a compact 256x128 FastPolicy from a frozen Dueling Q model."""
    print(f"\n--- Distilling 256x128 FastPolicy from {q_model.arch.upper()} over {num_deals:,} deals ---", flush=True)
    q_model.eval()
    use_hist = q_model.arch == "gru"
    fast = FastPolicy().to(device)
    opt = torch.optim.AdamW(fast.parameters(), lr=1.5e-3, weight_decay=1e-4)

    env = bc.BatchPlayEnv(num_envs=batch_envs, config=bc.Config.belgian(), seed=777, bid_jitter=0.6, threads=8)
    n_rounds = max(1, num_deals // batch_envs)
    idx_range = torch.arange(bc.MAX_LEGAL, device=device).unsqueeze(0)
    empty_h = torch.zeros((batch_envs, 1, bc.PLY_DIM), dtype=torch.float32, device=device)

    states_all = []
    legal_masks_all = []
    target_probs_all = []

    t0 = time.perf_counter()
    with torch.no_grad(), _autocast(device):
        for _ in range(n_rounds):
            for ply in range(32):
                batch = env.observe_batch(include_history=use_hist, include_beliefs=False)
                s_np = batch["state"]
                la_np = batch["legal_actions"]
                lc_np = batch["legal_cards"]
                nl_np = batch["n_legal"]

                s_t = torch.from_numpy(s_np).to(device)
                a_t = torch.from_numpy(la_np).to(device)
                valid = idx_range < torch.from_numpy(nl_np).to(device).unsqueeze(1)
                if use_hist:
                    h_t = torch.from_numpy(batch["history"][:, :ply, :]).to(device) if ply > 0 else empty_h
                    np_t = torch.full((batch_envs,), ply, dtype=torch.int32, device=device)
                    q = q_model(s_t, a_t, h_t, np_t, valid_mask=valid)
                else:
                    q = q_model(s_t, a_t, valid_mask=valid)

                q = q.float().masked_fill(~valid, -1e9)
                probs = F.softmax(q / temperature, dim=-1).cpu().numpy()
                greedy = q.argmax(dim=-1).cpu().numpy().astype(np.int32)

                keep = np.nonzero(nl_np > 1)[0]
                if len(keep) > 0:
                    tgt32 = np.zeros((len(keep), 32), dtype=np.float32)
                    lmask32 = np.zeros((len(keep), 32), dtype=np.float32)
                    for row_i, b in enumerate(keep):
                        trump = env.deal(int(b)).trump
                        for k in range(int(nl_np[b])):
                            cc = bc.canon_card(int(lc_np[b, k]), trump)
                            tgt32[row_i, cc] = probs[b, k]
                            lmask32[row_i, cc] = 1.0
                    states_all.append(s_np[keep])
                    legal_masks_all.append(lmask32)
                    target_probs_all.append(tgt32)

                env.step_batch(greedy)

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
        tot_loss = 0.0
        tot_acc = 0.0
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
            f"  FastPolicy epoch {ep + 1}/{epochs}: kl_ce={tot_loss / steps:.4f}, top1={100 * tot_acc / steps:.1f}%",
            flush=True,
        )

    if save_bin is not None:
        fast.export_bin(save_bin)
        print(f"Exported FastPolicy binary weights to {save_bin}", flush=True)
    return fast


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=("mlp", "gru"), default="gru")
    ap.add_argument("--episodes", type=int, default=1000000)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--env-threads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--epochs-per-round", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--eps-start", type=float, default=0.08)
    ap.add_argument("--eps-end", type=float, default=0.015)
    ap.add_argument("--bid-jitter", type=float, default=0.8)
    ap.add_argument("--eval-every", type=int, default=200000)
    ap.add_argument("--eval-decks", type=int, default=250)
    ap.add_argument("--train-belief", action="store_true")
    ap.add_argument("--belief-every", type=int, default=4, help="Train BeliefNet every N rounds")
    ap.add_argument("--distill-fast", action="store_true")
    ap.add_argument("--distill-only", action="store_true", help="Skip Q training and only distill FastPolicy from --resume")
    ap.add_argument("--distill-deals", type=int, default=60000)
    ap.add_argument("--distill-temp", type=float, default=0.04, help="Softmax temperature for Q->FastPolicy distillation")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save", type=Path, default=Path("checkpoints/q_gru.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    use_hist = args.arch == "gru" or args.train_belief
    q_model = build_q_model(args.arch).to(device)
    belief_net = BeliefNet().to(device) if args.train_belief else None

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        q_model.load_state_dict(ckpt["q_state_dict"])
        if belief_net is not None and "belief_state_dict" in ckpt:
            belief_net.load_state_dict(ckpt["belief_state_dict"])
        print(f"Resumed weights from {args.resume}", flush=True)

    q_opt = torch.optim.AdamW(q_model.parameters(), lr=args.lr, weight_decay=1e-4)
    b_opt = torch.optim.AdamW(belief_net.parameters(), lr=1e-3, weight_decay=1e-4) if belief_net else None

    total_rounds = max(1, args.episodes // args.envs)
    q_sched = torch.optim.lr_scheduler.CosineAnnealingLR(q_opt, T_max=total_rounds, eta_min=args.lr * 0.05)
    b_sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(b_opt, T_max=max(1, total_rounds // args.belief_every), eta_min=5e-5)
        if b_opt
        else None
    )

    if args.distill_only:
        fast_bin = args.save.with_suffix(".fast.bin")
        fast = distill_fast_policy(
            q_model,
            device=device,
            num_deals=args.distill_deals,
            batch_envs=min(512, args.envs),
            temperature=args.distill_temp,
            save_bin=fast_bin,
        )
        ev_fast = eval_vs_cpp_agent(fast, "rule", decks=args.eval_decks, device=device)
        print(
            f"FastPolicy (256x128, T={args.distill_temp}) vs rule: {ev_fast['units']:+.2f}±{ev_fast['units_ci95']:.2f}u "
            f"({ev_fast['raw']:+.1f}±{ev_fast['raw_ci95']:.1f} raw pts)",
            flush=True,
        )
        torch.save(
            {"encoder_version": bc.ENCODER_VERSION, "fast_state_dict": fast.state_dict()},
            args.save.with_suffix(".fast.pt"),
        )
        return

    env = bc.BatchPlayEnv(
        num_envs=args.envs,
        config=bc.Config.belgian(),
        seed=args.seed,
        bid_threshold=5.0,
        bid_jitter=args.bid_jitter,
        threads=args.env_threads,
    )
    buf = EpisodeBuffers(args.envs, device, use_hist=use_hist, include_beliefs=args.train_belief)
    rng = np.random.default_rng(args.seed)
    idx_range = torch.arange(bc.MAX_LEGAL, device=device).unsqueeze(0)

    print(
        f"Training GPU-BF16 Dueling {args.arch.upper()} on {device} | envs={args.envs} | "
        f"target={args.episodes:,} deals | eps={args.eps_start}->{args.eps_end} | lr={args.lr} | "
        f"ep/rnd={args.epochs_per_round} | params={sum(p.numel() for p in q_model.parameters()):,} | "
        f"belief={args.train_belief}(1/{args.belief_every})",
        flush=True,
    )
    t_start = time.perf_counter()
    episodes_done = 0
    next_eval = args.eval_every
    ema_q_loss = 0.0
    best_score = -1e9
    args.save.parent.mkdir(parents=True, exist_ok=True)

    ply_buckets = ((0, 8), (8, 16), (16, 24), (24, 32)) if args.arch == "gru" else ((0, 32),)

    for rnd in range(1, total_rounds + 1):
        frac = rnd / total_rounds
        eps = args.eps_start + (args.eps_end - args.eps_start) * min(1.0, frac / 0.90)
        do_belief = args.train_belief and (rnd % args.belief_every == 0)
        collect_episodes_gpu(env, q_model, buf, eps, do_belief, rng)
        episodes_done += args.envs

        q_model.train()
        q_losses = []
        for _ in range(args.epochs_per_round):
            for p_lo, p_hi in ply_buckets:
                mask = (buf.n_legal[p_lo:p_hi] > 1).reshape(-1)
                valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                if len(valid_idx) == 0:
                    continue
                perm = valid_idx[torch.randperm(len(valid_idx), device=device)]
                sb_all = buf.states[p_lo:p_hi].reshape(-1, bc.STATE_DIM)
                lab_all = buf.legal_actions[p_lo:p_hi].reshape(-1, bc.MAX_LEGAL, bc.ACTION_DIM)
                nlb_all = buf.n_legal[p_lo:p_hi].reshape(-1)
                cib_all = buf.chosen_idx[p_lo:p_hi].reshape(-1)
                rb_all = buf.returns[p_lo:p_hi].reshape(-1)
                if args.arch == "gru":
                    t_max = max(1, p_hi - 1)
                    hb_all = buf.hists[p_lo:p_hi, :, :t_max, :].reshape(-1, t_max, bc.PLY_DIM)
                    npb_all = buf.n_plies[p_lo:p_hi].reshape(-1)

                for s in range(0, len(perm), args.batch_size):
                    mb = perm[s : s + args.batch_size]
                    sb = sb_all[mb]
                    lab = lab_all[mb]
                    nlb = nlb_all[mb]
                    cib = cib_all[mb].long()
                    rb = rb_all[mb]
                    valid = idx_range < nlb.unsqueeze(1)
                    with _autocast(device):
                        if args.arch == "gru":
                            q_all = q_model(sb, lab, hb_all[mb], npb_all[mb], valid_mask=valid)
                        else:
                            q_all = q_model(sb, lab, valid_mask=valid)
                        pred = q_all[torch.arange(len(mb), device=device), cib]
                        loss = F.mse_loss(pred.float(), rb)
                    q_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(q_model.parameters(), 2.0)
                    q_opt.step()
                    q_losses.append(float(loss.item()))
        q_sched.step()
        cur_ql = float(np.mean(q_losses))
        ema_q_loss = cur_ql if ema_q_loss == 0.0 else 0.95 * ema_q_loss + 0.05 * cur_ql

        if do_belief and belief_net is not None:
            belief_net.train()
            for p_lo, p_hi in ((0, 8), (8, 16), (16, 24), (24, 32)):
                t_max = max(1, p_hi - 1)
                sb_all = buf.states[p_lo:p_hi].reshape(-1, bc.STATE_DIM)
                hb_all = buf.hists[p_lo:p_hi, :, :t_max, :].reshape(-1, t_max, bc.PLY_DIM)
                npb_all = buf.n_plies[p_lo:p_hi].reshape(-1)
                bt_all = buf.bt[p_lo:p_hi].reshape(-1, 32)
                ba_all = buf.ba[p_lo:p_hi].reshape(-1, 32, 3)
                perm = torch.randperm(len(sb_all), device=device)
                for s in range(0, len(perm), args.batch_size):
                    mb = perm[s : s + args.batch_size]
                    with _autocast(device):
                        logits = belief_net(sb_all[mb], hb_all[mb], npb_all[mb], ba_all[mb])
                        loss = F.cross_entropy(logits.float().view(-1, 3), bt_all[mb].long().view(-1), ignore_index=-1)
                    b_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(belief_net.parameters(), 2.0)
                    b_opt.step()
            if b_sched is not None:
                b_sched.step()

        if episodes_done >= next_eval or rnd == total_rounds:
            elapsed = time.perf_counter() - t_start
            ev_rule = eval_vs_cpp_agent(q_model, "rule", decks=args.eval_decks, device=device)
            ev_roll = eval_vs_cpp_agent(q_model, "rollout:32", decks=args.eval_decks, device=device)
            msg = (
                f"[{episodes_done:7d} deals | {elapsed:5.1f}s | {episodes_done / elapsed:5.0f} d/s | "
                f"eps={eps:.3f} | mse={ema_q_loss:.4f}] "
                f"vs rule: {ev_rule['units']:+.2f}±{ev_rule['units_ci95']:.2f}u | "
                f"vs rollout:32: {ev_roll['units']:+.2f}±{ev_roll['units_ci95']:.2f}u"
            )
            if belief_net is not None:
                bev_self = eval_belief_per_trick(belief_net, device, num_deals=64, policy_model=q_model)
                bev_rule = eval_belief_per_trick(belief_net, device, num_deals=64)
                msg += (
                    f" | B_phi self={bev_self['model_loss']:.4f}/samp={bev_self['sampler_loss']:.4f}"
                    f" rule={bev_rule['model_loss']:.4f}/samp={bev_rule['sampler_loss']:.4f}"
                )
            print(msg, flush=True)

            # Track best checkpoint by rollout:32 margin (tighter discrimination once above rule).
            score = ev_roll["units"] + 0.5 * ev_rule["units"]
            if score > best_score:
                best_score = score
                ckpt = {
                    "encoder_version": bc.ENCODER_VERSION,
                    "arch": args.arch,
                    "episodes": episodes_done,
                    "best_vs_rule": ev_rule["units"],
                    "best_vs_rollout32": ev_roll["units"],
                    "q_state_dict": q_model.state_dict(),
                }
                if belief_net is not None:
                    ckpt["belief_state_dict"] = belief_net.state_dict()
                torch.save(ckpt, args.save)
            next_eval += args.eval_every

    best_ckpt = torch.load(args.save, map_location=device, weights_only=True)
    q_model.load_state_dict(best_ckpt["q_state_dict"])
    # Held-out seed (54321 != selection seed 12345) for unbiased final gate reporting.
    ev_mcts1000 = eval_vs_cpp_agent(q_model, "ismcts:1000", decks=250, seed=54321, device=device)
    print(
        f"\nBest Dueling {args.arch.upper()} (ep={best_ckpt['episodes']:,}) vs ismcts:1000 (held-out seed 54321): "
        f"{ev_mcts1000['units']:+.2f}±{ev_mcts1000['units_ci95']:.2f}u "
        f"({ev_mcts1000['raw']:+.1f}±{ev_mcts1000['raw_ci95']:.1f} raw pts)",
        flush=True,
    )
    if belief_net is not None and "belief_state_dict" in best_ckpt:
        belief_net.load_state_dict(best_ckpt["belief_state_dict"])
        bev = eval_belief_per_trick(belief_net, device, num_deals=200, seed=54321)
        print(
            f"B_phi overall CE={bev['model_loss']:.4f} vs WorldSampler={bev['sampler_loss']:.4f} "
            f"(naive_unif={bev['uniform_loss']:.4f})",
            flush=True,
        )
        for t, (ml, sl, ul) in bev["per_trick"].items():
            print(f"  trick {t + 1}: B_phi={ml:.4f} vs sampler={sl:.4f} (delta={ml - sl:+.4f}, naive={ul:.4f})", flush=True)

    if args.distill_fast:
        fast_bin = args.save.with_suffix(".fast.bin")
        fast = distill_fast_policy(
            q_model,
            device=device,
            num_deals=args.distill_deals,
            batch_envs=min(512, args.envs),
            temperature=args.distill_temp,
            save_bin=fast_bin,
        )
        ev_fast = eval_vs_cpp_agent(fast, "rule", decks=args.eval_decks, device=device)
        print(
            f"FastPolicy (256x128) vs rule: {ev_fast['units']:+.2f}±{ev_fast['units_ci95']:.2f}u "
            f"({ev_fast['raw']:+.1f}±{ev_fast['raw_ci95']:.1f} raw pts)",
            flush=True,
        )
        torch.save(
            {"encoder_version": bc.ENCODER_VERSION, "fast_state_dict": fast.state_dict()},
            args.save.with_suffix(".fast.pt"),
        )


if __name__ == "__main__":
    main()
