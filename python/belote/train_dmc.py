"""M2 DMC self-play trainer for CardQMLP / CardQGRU, BeliefNet B_phi, and FastPolicy.

Usage examples:
  # 1. Fast MLP sanity check (beats rule & approaches ISMCTS-1000 in ~1-2 min on RTX 3060)
  .venv/bin/python -m belote.train_dmc --arch mlp --episodes 100000 --envs 256 --save checkpoints/q_mlp.pt

  # 2. Full Q-GRU blueprint + BeliefNet B_phi + distilled 256x128 FastPolicy
  .venv/bin/python -m belote.train_dmc --arch gru --episodes 400000 --envs 256 \
      --train-belief --distill-fast --save checkpoints/q_gru.pt
"""

from __future__ import annotations

import argparse
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


def collect_episodes(
    env: bc.BatchPlayEnv,
    q_model: nn.Module,
    device: torch.device,
    eps: float,
    use_hist: bool,
    include_beliefs: bool,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Runs all B envs through exactly one complete 32-ply deal each."""
    bsz = env.num_envs
    states_buf = np.empty((32, bsz, bc.STATE_DIM), dtype=np.float32)
    hists_buf = np.empty((32, bsz, bc.MAX_PLIES, bc.PLY_DIM), dtype=np.float32) if use_hist else None
    nplies_buf = np.empty((32, bsz), dtype=np.int32) if use_hist else None
    chosen_act_buf = np.empty((32, bsz, bc.ACTION_DIM), dtype=np.float32)
    team_buf = np.empty((32, bsz), dtype=np.int8)
    nlegal_buf = np.empty((32, bsz), dtype=np.int8)
    bt_buf = np.empty((32, bsz, 32), dtype=np.int8) if include_beliefs else None
    ba_buf = np.empty((32, bsz, 32, 3), dtype=np.uint8) if include_beliefs else None

    q_model.eval()
    final_rew_t0 = np.zeros(bsz, dtype=np.float32)
    with torch.no_grad():
        for ply in range(32):
            batch = env.observe_batch(include_history=use_hist, include_beliefs=include_beliefs)
            states = batch["state"]
            legal_actions = batch["legal_actions"]
            n_legal = batch["n_legal"]
            cur_team = batch["current_team"]

            states_buf[ply] = states
            team_buf[ply] = cur_team.astype(np.int8)
            nlegal_buf[ply] = n_legal.astype(np.int8)
            if use_hist:
                hists_buf[ply] = batch["history"]
                nplies_buf[ply] = batch["n_plies"]
            if include_beliefs:
                bt_buf[ply] = batch["belief_targets"]
                ba_buf[ply] = batch["belief_allowed"]

            # Batched Q inference on GPU (only for envs with >1 legal move, or all at once).
            s_t = torch.from_numpy(states).to(device, non_blocking=True)
            a_t = torch.from_numpy(legal_actions).to(device, non_blocking=True)
            if use_hist:
                h_t = torch.from_numpy(batch["history"]).to(device, non_blocking=True)
                np_t = torch.from_numpy(batch["n_plies"]).to(device, non_blocking=True)
                q_vals = q_model(s_t, a_t, h_t, np_t)
            else:
                q_vals = q_model(s_t, a_t)

            idx_range = torch.arange(bc.MAX_LEGAL, device=device).unsqueeze(0)
            valid = idx_range < torch.from_numpy(n_legal).to(device, non_blocking=True).unsqueeze(1)
            q_vals = q_vals.masked_fill(~valid, -1e9)
            greedy = q_vals.argmax(dim=-1).cpu().numpy().astype(np.int32)

            if eps > 0.0:
                explore = (rng.random(bsz) < eps) & (n_legal > 1)
                rand_idx = (rng.integers(0, 10000, size=bsz) % n_legal).astype(np.int32)
                action_idx = np.where(explore, rand_idx, greedy)
            else:
                action_idx = greedy

            chosen_act_buf[ply] = legal_actions[np.arange(bsz), action_idx]
            done, rew_t0 = env.step_batch(action_idx)
            if ply == 31:
                assert np.all(done == 1), "Expected all synchronized deals to finish on ply 31"
                final_rew_t0 = rew_t0

    # Assign exact Monte Carlo returns signed by acting team (+r0 for team 0, -r0 for team 1).
    signs = np.where(team_buf == 0, 1.0, -1.0).astype(np.float32)
    returns = signs * final_rew_t0[None, :]  # (32, B)

    # Filter out trivial 1-legal-move forced plies for Q training (keep all plies for BeliefNet).
    decision_mask = (nlegal_buf > 1).reshape(-1)
    out = {
        "q_states": states_buf.reshape(-1, bc.STATE_DIM)[decision_mask],
        "q_actions": chosen_act_buf.reshape(-1, bc.ACTION_DIM)[decision_mask],
        "q_returns": returns.reshape(-1)[decision_mask],
    }
    if use_hist:
        out["q_hists"] = hists_buf.reshape(-1, bc.MAX_PLIES, bc.PLY_DIM)[decision_mask]
        out["q_nplies"] = nplies_buf.reshape(-1)[decision_mask]
    if include_beliefs:
        out["b_states"] = states_buf.reshape(-1, bc.STATE_DIM)
        out["b_hists"] = hists_buf.reshape(-1, bc.MAX_PLIES, bc.PLY_DIM)
        out["b_nplies"] = nplies_buf.reshape(-1)
        out["b_targets"] = bt_buf.reshape(-1, 32)
        out["b_allowed"] = ba_buf.reshape(-1, 32, 3)
    return out


def distill_fast_policy(
    q_model: nn.Module,
    device: torch.device,
    num_deals: int = 40000,
    batch_envs: int = 256,
    epochs: int = 4,
    temperature: float = 0.08,
    save_bin: Path | None = None,
) -> FastPolicy:
    """Distill a compact 256x128 FastPolicy from a frozen Q model (Q-MLP or Q-GRU)."""
    print(f"\n--- Distilling 256x128 FastPolicy from {q_model.arch.upper()} over {num_deals} deals ---")
    q_model.eval()
    use_hist = q_model.arch == "gru"
    fast = FastPolicy().to(device)
    opt = torch.optim.AdamW(fast.parameters(), lr=1e-3, weight_decay=1e-4)

    env = bc.BatchPlayEnv(num_envs=batch_envs, config=bc.Config.belgian(), seed=777, bid_jitter=0.6)
    n_rounds = max(1, num_deals // batch_envs)

    states_all = []
    legal_masks_all = []
    target_probs_all = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_rounds):
            for _ply in range(32):
                batch = env.observe_batch(include_history=use_hist, include_beliefs=False)
                s_np = batch["state"]
                la_np = batch["legal_actions"]
                lc_np = batch["legal_cards"]
                nl_np = batch["n_legal"]

                s_t = torch.from_numpy(s_np).to(device)
                a_t = torch.from_numpy(la_np).to(device)
                if use_hist:
                    h_t = torch.from_numpy(batch["history"]).to(device)
                    np_t = torch.from_numpy(batch["n_plies"]).to(device)
                    q = q_model(s_t, a_t, h_t, np_t)
                else:
                    q = q_model(s_t, a_t)

                idx_range = torch.arange(bc.MAX_LEGAL, device=device).unsqueeze(0)
                valid = idx_range < torch.from_numpy(nl_np).to(device).unsqueeze(1)
                q = q.masked_fill(~valid, -1e9)
                probs = F.softmax(q / temperature, dim=-1).cpu().numpy()
                greedy = q.argmax(dim=-1).cpu().numpy().astype(np.int32)

                # Keep only non-forced decisions (n_legal > 1).
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
    print(f"Collected {len(X):,} teacher decisions in {time.perf_counter() - t0:.1f}s")

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
        print(f"  FastPolicy epoch {ep + 1}/{epochs}: kl_ce={tot_loss / steps:.4f}, top1={100 * tot_acc / steps:.1f}%")

    if save_bin is not None:
        fast.export_bin(save_bin)
        print(f"Exported FastPolicy binary weights to {save_bin}")
    return fast


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=("mlp", "gru"), default="mlp")
    ap.add_argument("--episodes", type=int, default=100000)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eps-start", type=float, default=0.25)
    ap.add_argument("--eps-end", type=float, default=0.03)
    ap.add_argument("--bid-jitter", type=float, default=0.8)
    ap.add_argument("--eval-every", type=int, default=25000)
    ap.add_argument("--eval-decks", type=int, default=200)
    ap.add_argument("--train-belief", action="store_true")
    ap.add_argument("--distill-fast", action="store_true")
    ap.add_argument("--distill-deals", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save", type=Path, default=Path("checkpoints/q_model.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    use_hist = args.arch == "gru" or args.train_belief
    q_model = build_q_model(args.arch).to(device)
    q_opt = torch.optim.AdamW(q_model.parameters(), lr=args.lr, weight_decay=1e-4)

    belief_net = BeliefNet().to(device) if args.train_belief else None
    b_opt = torch.optim.AdamW(belief_net.parameters(), lr=1e-3, weight_decay=1e-4) if belief_net else None

    total_rounds = max(1, args.episodes // args.envs)
    q_sched = torch.optim.lr_scheduler.CosineAnnealingLR(q_opt, T_max=total_rounds, eta_min=args.lr * 0.05)

    env = bc.BatchPlayEnv(
        num_envs=args.envs,
        config=bc.Config.belgian(),
        seed=args.seed,
        bid_threshold=5.0,
        bid_jitter=args.bid_jitter,
    )
    rng = np.random.default_rng(args.seed)

    print(
        f"Training {args.arch.upper()} on {device} | envs={args.envs} | target={args.episodes:,} deals | "
        f"params={sum(p.numel() for p in q_model.parameters()):,} | belief={args.train_belief}"
    )
    t_start = time.perf_counter()
    episodes_done = 0
    next_eval = args.eval_every
    ema_q_loss = 0.0
    ema_b_loss = 0.0

    for rnd in range(1, total_rounds + 1):
        frac = rnd / total_rounds
        eps = args.eps_start + (args.eps_end - args.eps_start) * min(1.0, frac / 0.85)
        data = collect_episodes(env, q_model, device, eps, use_hist, args.train_belief, rng)
        episodes_done += args.envs

        # Q-network updates (2 epochs over fresh on-policy Monte Carlo batch).
        q_model.train()
        n_q = len(data["q_returns"])
        q_losses = []
        for _ in range(2):
            perm = rng.permutation(n_q)
            for s in range(0, n_q, args.batch_size):
                idx = perm[s : s + args.batch_size]
                sb = torch.from_numpy(data["q_states"][idx]).to(device, non_blocking=True)
                ab = torch.from_numpy(data["q_actions"][idx]).to(device, non_blocking=True)
                rb = torch.from_numpy(data["q_returns"][idx]).to(device, non_blocking=True)
                if args.arch == "gru":
                    hb = torch.from_numpy(data["q_hists"][idx]).to(device, non_blocking=True)
                    npb = torch.from_numpy(data["q_nplies"][idx]).to(device, non_blocking=True)
                    pred = q_model(sb, ab, hb, npb)
                else:
                    pred = q_model(sb, ab)
                loss = F.smooth_l1_loss(pred, rb, beta=0.5)
                q_opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(q_model.parameters(), 2.0)
                q_opt.step()
                q_losses.append(float(loss.item()))
        q_sched.step()
        cur_ql = float(np.mean(q_losses))
        ema_q_loss = cur_ql if rnd == 1 else 0.9 * ema_q_loss + 0.1 * cur_ql

        # BeliefNet update (1 pass over fresh self-play batch).
        if belief_net is not None:
            belief_net.train()
            n_b = len(data["b_states"])
            perm = rng.permutation(n_b)
            b_losses = []
            for s in range(0, n_b, args.batch_size):
                idx = perm[s : s + args.batch_size]
                sb = torch.from_numpy(data["b_states"][idx]).to(device, non_blocking=True)
                hb = torch.from_numpy(data["b_hists"][idx]).to(device, non_blocking=True)
                npb = torch.from_numpy(data["b_nplies"][idx]).to(device, non_blocking=True)
                bt = torch.from_numpy(data["b_targets"][idx]).long().to(device, non_blocking=True)
                ba = torch.from_numpy(data["b_allowed"][idx]).to(device, non_blocking=True)
                logits = belief_net(sb, hb, npb, ba)
                loss = F.cross_entropy(logits.view(-1, 3), bt.view(-1), ignore_index=-1)
                b_opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(belief_net.parameters(), 2.0)
                b_opt.step()
                b_losses.append(float(loss.item()))
            cur_bl = float(np.mean(b_losses))
            ema_b_loss = cur_bl if rnd == 1 else 0.9 * ema_b_loss + 0.1 * cur_bl

        if episodes_done >= next_eval or rnd == total_rounds:
            elapsed = time.perf_counter() - t_start
            ev_rule = eval_vs_cpp_agent(q_model, "rule", decks=args.eval_decks, device=device)
            ev_mcts = eval_vs_cpp_agent(q_model, "ismcts:100", decks=min(100, args.eval_decks), device=device)
            msg = (
                f"[{episodes_done:7d} deals | {elapsed:5.1f}s | {episodes_done / elapsed:5.0f} d/s | "
                f"eps={eps:.3f} | q_loss={ema_q_loss:.4f}] "
                f"vs rule: {ev_rule['units']:+.2f}±{ev_rule['units_ci95']:.2f}u | "
                f"vs ismcts:100: {ev_mcts['units']:+.2f}±{ev_mcts['units_ci95']:.2f}u"
            )
            if belief_net is not None:
                bev = eval_belief_per_trick(belief_net, device, num_deals=128)
                msg += f" | B_phi CE={bev['model_loss']:.4f} (unif={bev['uniform_loss']:.4f})"
            print(msg, flush=True)
            next_eval += args.eval_every

    # Save checkpoint.
    args.save.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "encoder_version": bc.ENCODER_VERSION,
        "arch": args.arch,
        "episodes": episodes_done,
        "q_state_dict": q_model.state_dict(),
    }
    if belief_net is not None:
        ckpt["belief_state_dict"] = belief_net.state_dict()
    torch.save(ckpt, args.save)
    print(f"Saved checkpoint to {args.save}")

    if args.distill_fast:
        fast_bin = args.save.with_suffix(".fast.bin")
        fast = distill_fast_policy(
            q_model,
            device=device,
            num_deals=args.distill_deals,
            batch_envs=args.envs,
            save_bin=fast_bin,
        )
        ev_fast = eval_vs_cpp_agent(fast, "rule", decks=args.eval_decks, device=device)
        print(
            f"FastPolicy (256x128) vs rule: {ev_fast['units']:+.2f}±{ev_fast['units_ci95']:.2f}u "
            f"({ev_fast['raw']:+.1f}±{ev_fast['raw_ci95']:.1f} raw pts)"
        )
        torch.save(
            {"encoder_version": bc.ENCODER_VERSION, "fast_state_dict": fast.state_dict()},
            args.save.with_suffix(".fast.pt"),
        )


if __name__ == "__main__":
    main()
