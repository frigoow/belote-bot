"""Expert Iteration (ExIt) trainer for CardQGRU, BeliefNet B_phi, and FastPolicy.

Uses our superhuman C++ hybrid search (FastPolicy play-outs + play-history weighting
+ exact double-dummy AB endgame from trick 5) to generate low-variance multi-action
CRN Q-value labels across all legal moves, then trains CardQGRU with combined
Q-value MSE + soft-policy KL loss, updates BeliefNet B_phi on the search-played
trajectories, and distills an upgraded 256x128 FastPolicy binary.
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
from belote.models import BeliefNet, CardQGRU
from belote.train_dmc import distill_fast_policy


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=Path, required=True, help="Starting Q-GRU + B_phi checkpoint")
    ap.add_argument("--teacher-fast", type=Path, required=True, help="Teacher FastPolicy .bin for hybridSearch")
    ap.add_argument("--bid-bundle", type=Path, default=None, help="Optional M3 BBM1 BidModel for M4 deal distribution")
    ap.add_argument("--deals", type=int, default=16000, help="Total ExIt deals to collect")
    ap.add_argument("--chunk-deals", type=int, default=4000, help="Deals per C++ collection chunk")
    ap.add_argument("--worlds", type=int, default=48, help="Worlds per hybridSearch teacher call")
    ap.add_argument("--ab-trick", type=int, default=4, help="AB endgame handoff trick for teacher")
    ap.add_argument("--eps", type=float, default=0.04, help="Exploratory move rate during ExIt trajectories")
    ap.add_argument("--epochs", type=int, default=8, help="GPU training epochs over collected ExIt dataset")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--kl-weight", type=float, default=0.05)
    ap.add_argument("--kl-temp", type=float, default=0.04)
    ap.add_argument("--distill-deals", type=int, default=60000)
    ap.add_argument("--distill-temp", type=float, default=0.04)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--save", type=Path, default=Path("checkpoints/q_gru_exit1.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    config = bc.Config.belgian()
    teacher_fast = bc.FastPolicyModel.load(str(args.teacher_fast))
    bid_model = bc.BidModel.load(str(args.bid_bundle)) if args.bid_bundle else None

    q_model = CardQGRU().to(device)
    belief_net = BeliefNet().to(device)
    ckpt = torch.load(args.resume, map_location=device, weights_only=True)
    q_model.load_state_dict(ckpt["q_state_dict"])
    if "belief_state_dict" in ckpt:
        belief_net.load_state_dict(ckpt["belief_state_dict"])
    print(f"Loaded starting Q-GRU & BeliefNet from {args.resume}", flush=True)

    # Collect ExIt search-labeled states in memory-bounded chunks (stored in BF16 on CPU).
    chunks = []
    t0 = time.perf_counter()
    n_chunks = math.ceil(args.deals / args.chunk_deals)
    total_rows = 0
    for ci in range(n_chunks):
        cur_d = min(args.chunk_deals, args.deals - ci * args.chunk_deals)
        tc = time.perf_counter()
        raw = bc.collect_exit_dataset(
            num_deals=cur_d,
            config=config,
            fast_policy=teacher_fast,
            bid_model=bid_model,
            worlds=args.worlds,
            ab_handoff_trick=args.ab_trick,
            eps=args.eps,
            seed=args.seed + ci * 1000,
            threads=args.threads,
        )
        nrows = len(raw["n_legal"])
        total_rows += nrows
        chunks.append(
            {
                "state": torch.from_numpy(raw["state"]).to(torch.bfloat16),
                "history": torch.from_numpy(raw["history"]).to(torch.bfloat16),
                "n_plies": torch.from_numpy(raw["n_plies"]).to(torch.int32),
                "legal_actions": torch.from_numpy(raw["legal_actions"]).to(torch.bfloat16),
                "n_legal": torch.from_numpy(raw["n_legal"]).to(torch.int32),
                "q_targets": torch.from_numpy(raw["q_targets"]).to(torch.float32),
                "bt": torch.from_numpy(raw["belief_targets"]).to(torch.int8),
                "ba": torch.from_numpy(raw["belief_allowed"]).to(torch.uint8),
            }
        )
        print(
            f"  Chunk {ci + 1}/{n_chunks}: {cur_d:,} deals -> {nrows:,} search-labeled states "
            f"in {time.perf_counter() - tc:.1f}s ({cur_d / (time.perf_counter() - tc):.1f} d/s)",
            flush=True,
        )

    ds = {k: torch.cat([c[k] for c in chunks], dim=0) for k in chunks[0]}
    del chunks
    print(f"Total ExIt dataset: {total_rows:,} states collected in {time.perf_counter() - t0:.1f}s", flush=True)

    # Train Q-GRU and BeliefNet over the multi-action search Q-targets.
    q_opt = torch.optim.AdamW(q_model.parameters(), lr=args.lr, weight_decay=1e-4)
    b_opt = torch.optim.AdamW(belief_net.parameters(), lr=5e-4, weight_decay=1e-4)
    steps_per_ep = math.ceil(total_rows / args.batch_size)
    q_sched = torch.optim.lr_scheduler.CosineAnnealingLR(q_opt, T_max=args.epochs * steps_per_ep, eta_min=1e-5)
    b_sched = torch.optim.lr_scheduler.CosineAnnealingLR(b_opt, T_max=args.epochs * steps_per_ep, eta_min=5e-5)
    idx_range = torch.arange(bc.MAX_LEGAL, device=device).unsqueeze(0)

    best_score = -1e9
    args.save.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for ep in range(1, args.epochs + 1):
        te = time.perf_counter()
        perm = torch.from_numpy(rng.permutation(total_rows))
        q_model.train()
        belief_net.train()
        mse_sum, kl_sum, b_sum, top1_sum, n_steps = 0.0, 0.0, 0.0, 0.0, 0

        for start in range(0, total_rows, args.batch_size):
            mb = perm[start : start + args.batch_size]
            sb = ds["state"][mb].to(device, non_blocking=True)
            npb = ds["n_plies"][mb].to(device, non_blocking=True)
            max_t = max(1, int(npb.max().item()))
            hb = ds["history"][mb, :max_t, :].to(device, non_blocking=True)
            lab = ds["legal_actions"][mb].to(device, non_blocking=True)
            nlb = ds["n_legal"][mb].to(device, non_blocking=True)
            qtb = ds["q_targets"][mb].to(device, non_blocking=True)
            valid = idx_range < nlb.unsqueeze(1)

            with _autocast(device):
                q_pred = q_model(sb, lab, hb, npb, valid_mask=valid)
            qf = q_pred.float()
            vf = valid.float()
            mse = (((qf - qtb) ** 2) * vf).sum() / vf.sum().clamp(min=1.0)

            q_masked = qf.masked_fill(~valid, -1e9)
            t_masked = qtb.masked_fill(~valid, -1e9)
            t_probs = F.softmax(t_masked / args.kl_temp, dim=-1)
            p_logp = F.log_softmax(q_masked / args.kl_temp, dim=-1)
            kl = -(t_probs * p_logp).sum(dim=-1).mean()
            loss = mse + args.kl_weight * kl

            q_opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(q_model.parameters(), 2.0)
            q_opt.step()
            q_sched.step()

            # BeliefNet update on the same batch.
            btb = ds["bt"][mb].to(device, non_blocking=True)
            bab = ds["ba"][mb].to(device, non_blocking=True)
            with _autocast(device):
                blogits = belief_net(sb, hb, npb, bab)
                bloss = F.cross_entropy(blogits.float().view(-1, 3), btb.long().view(-1), ignore_index=-1)
            b_opt.zero_grad(set_to_none=True)
            bloss.backward()
            nn.utils.clip_grad_norm_(belief_net.parameters(), 2.0)
            b_opt.step()
            b_sched.step()

            mse_sum += float(mse.item())
            kl_sum += float(kl.item())
            b_sum += float(bloss.item())
            top1_sum += float((q_masked.argmax(dim=-1) == t_masked.argmax(dim=-1)).float().mean().item())
            n_steps += 1

        msg = (
            f"Epoch {ep}/{args.epochs} ({time.perf_counter() - te:.1f}s) | "
            f"q_mse={mse_sum/n_steps:.5f} | kl={kl_sum/n_steps:.4f} | "
            f"top1={100*top1_sum/n_steps:.1f}% | b_ce={b_sum/n_steps:.4f}"
        )
        if ep in (2, 4, 6, args.epochs):
            ev_r = eval_vs_cpp_agent(q_model, "rule", decks=250, device=device)
            ev_roll = eval_vs_cpp_agent(q_model, "rollout:32", decks=250, device=device)
            msg += (
                f" | vs rule: {ev_r['units']:+.2f}±{ev_r['units_ci95']:.2f}u"
                f" | vs rollout:32: {ev_roll['units']:+.2f}±{ev_roll['units_ci95']:.2f}u"
            )
            sc = ev_roll["units"] + 0.5 * ev_r["units"]
            if sc > best_score:
                best_score = sc
                torch.save(
                    {
                        "encoder_version": bc.ENCODER_VERSION,
                        "arch": "gru",
                        "exit_deals": args.deals,
                        "best_vs_rule": ev_r["units"],
                        "best_vs_rollout32": ev_roll["units"],
                        "q_state_dict": q_model.state_dict(),
                        "belief_state_dict": belief_net.state_dict(),
                    },
                    args.save,
                )
        print(msg, flush=True)

    # Load best ExIt Q-GRU & gate against ISMCTS-1000 on a held-out seed (54321 != 12345)!
    best = torch.load(args.save, map_location=device, weights_only=True)
    q_model.load_state_dict(best["q_state_dict"])
    belief_net.load_state_dict(best["belief_state_dict"])

    ev_m1000 = eval_vs_cpp_agent(q_model, "ismcts:1000", decks=400, seed=54321, device=device)
    bev_self = eval_belief_per_trick(belief_net, device, num_deals=128, seed=54321, policy_model=q_model)
    bev_rule = eval_belief_per_trick(belief_net, device, num_deals=128, seed=54321)
    print(
        f"\n=== ExIt Q-GRU GATE (400 decks, held-out seed 54321) ===\n"
        f"  Pure Q-GRU vs ismcts:1000: {ev_m1000['units']:+.2f}±{ev_m1000['units_ci95']:.2f}u "
        f"({ev_m1000['raw']:+.1f}±{ev_m1000['raw_ci95']:.1f} raw pts)\n"
        f"  B_phi CE (self-play): {bev_self['model_loss']:.4f} vs WorldSampler={bev_self['sampler_loss']:.4f}\n"
        f"  B_phi CE (rule):      {bev_rule['model_loss']:.4f} vs WorldSampler={bev_rule['sampler_loss']:.4f}",
        flush=True,
    )

    fast_bin = args.save.with_suffix(".fast.bin")
    fast = distill_fast_policy(
        q_model,
        device=device,
        num_deals=args.distill_deals,
        batch_envs=512,
        temperature=args.distill_temp,
        save_bin=fast_bin,
    )
    ev_fast = eval_vs_cpp_agent(fast, "rule", decks=300, device=device)
    print(
        f"Upgraded FastPolicy (256x128) vs rule: {ev_fast['units']:+.2f}±{ev_fast['units_ci95']:.2f}u "
        f"({ev_fast['raw']:+.1f}±{ev_fast['raw_ci95']:.1f} raw pts)",
        flush=True,
    )
    torch.save(
        {"encoder_version": bc.ENCODER_VERSION, "fast_state_dict": fast.state_dict()},
        args.save.with_suffix(".fast.pt"),
    )


if __name__ == "__main__":
    main()
