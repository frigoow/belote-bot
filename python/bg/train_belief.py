"""Dedicated high-capacity 384x192 GRU BeliefNet v2 trainer for Bulgarian Bridge-Belot.

Ported directly from our Bridge BeliefNet v2 breakthrough:
- Steps C++ BgBatchPlayEnv (1,024 parallel envs, neural bidder + doubles) using our frozen
  champion BgFastPolicy actor on GPU (70% greedy + 30% softmax-sampled for opponent robustness).
- Trains a wider BgBeliefNet(state_hidden=384, gru_hidden=192) on 100% of all states per round
  with exact S_3 (Suit) / S_4 (AT/NT) GPU suit-permutation augmentation and Cosine LR.
- Evaluates strictly unseen per-trick cross-entropy (Tricks 1..7) directly against both
  DealSampler::exactMarginals() and the existing baseline BGB1 checkpoint!

Usage:
  PYTHONPATH=build:python .venv/bin/python -u -m bg.train_belief \
      --mode suit \
      --actor-fast checkpoints/bg_models_suit_sp/bg_fast_policy.bin \
      --baseline-belief checkpoints/bg_models_suit_sp/bg_belief.bin \
      --episodes 1200000 --envs 1024 --threads 16 \
      --out checkpoints/bg_models_belief_v2/bg_suit_belief_v2.bin
"""

from __future__ import annotations

import argparse
import contextlib
import math
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import bg_cpp as bgc
from bg.models import BgBeliefNet, BgFastPolicy
from bg.suit_perm import SuitPermuter

MAX_DECISIONS = 28


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def load_bgp1(path: Path, device: torch.device) -> BgFastPolicy:
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != b"BGP1":
            raise ValueError(f"{path}: bad magic {magic!r}")
        ver, sdim, h1, h2, odim = struct.unpack("<5i", f.read(20))
        if (ver, sdim, odim) != (bgc.ENCODER_VERSION, bgc.STATE_DIM, 32):
            raise ValueError(f"{path}: incompatible BGP1 header {(ver, sdim, h1, h2, odim)}")
        model = BgFastPolicy(state_dim=sdim, h1=h1, h2=h2)
        with torch.no_grad():
            for layer in (model.fc1, model.fc2, model.fc3):
                w = np.frombuffer(f.read(layer.weight.numel() * 4), dtype=np.float32).reshape(layer.weight.shape)
                b = np.frombuffer(f.read(layer.bias.numel() * 4), dtype=np.float32).reshape(layer.bias.shape)
                layer.weight.copy_(torch.from_numpy(w.copy()))
                layer.bias.copy_(torch.from_numpy(b.copy()))
    return model.to(device).eval()


def load_bgb1(path: Path, device: torch.device) -> BgBeliefNet:
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != b"BGB1":
            raise ValueError(f"{path}: bad magic {magic!r}")
        ver, sdim, pdim, sh, gh = struct.unpack("<5i", f.read(20))
        if (ver, sdim, pdim) != (bgc.ENCODER_VERSION, bgc.STATE_DIM, bgc.PLY_DIM):
            raise ValueError(f"{path}: incompatible BGB1 header {(ver, sdim, pdim, sh, gh)}")
        model = BgBeliefNet(state_dim=sdim, ply_dim=pdim, state_hidden=sh, gru_hidden=gh)

        def read_lin(layer):
            w = np.frombuffer(f.read(layer.weight.numel() * 4), dtype=np.float32).reshape(layer.weight.shape)
            b = np.frombuffer(f.read(layer.bias.numel() * 4), dtype=np.float32).reshape(layer.bias.shape)
            layer.weight.copy_(torch.from_numpy(w.copy()))
            layer.bias.copy_(torch.from_numpy(b.copy()))

        def read_ln(layer):
            w = np.frombuffer(f.read(layer.weight.numel() * 4), dtype=np.float32).reshape(layer.weight.shape)
            b = np.frombuffer(f.read(layer.bias.numel() * 4), dtype=np.float32).reshape(layer.bias.shape)
            layer.weight.copy_(torch.from_numpy(w.copy()))
            layer.bias.copy_(torch.from_numpy(b.copy()))

        def read_res(rb):
            read_lin(rb.fc1)
            read_ln(rb.ln1)
            read_lin(rb.fc2)
            read_ln(rb.ln2)

        with torch.no_grad():
            read_lin(model.state_fc)
            read_ln(model.state_ln)
            for name in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
                t = getattr(model.hist_enc.gru, name)
                arr = np.frombuffer(f.read(t.numel() * 4), dtype=np.float32).reshape(t.shape)
                t.copy_(torch.from_numpy(arr.copy()))
            e_arr = np.frombuffer(f.read(model.hist_enc.empty_emb.numel() * 4), dtype=np.float32)
            model.hist_enc.empty_emb.copy_(torch.from_numpy(e_arr.copy()))
            read_res(model.trunk1)
            read_res(model.trunk2)
            read_lin(model.delta_head)
    return model.to(device).eval()


@torch.no_grad()
def eval_belief_unseen(
    b_model: BgBeliefNet,
    actor: BgFastPolicy,
    device: torch.device,
    mode: str,
    bid_model_path: str,
    num_deals: int = 1024,
    seed: int = 987654,
    threads: int = 16,
) -> tuple[float, float, dict[int, float]]:
    b_model.eval()
    actor.eval()
    env = bgc.BgBatchPlayEnv(
        num_envs=num_deals,
        seed=seed,
        bid_noise=0.0,
        hanging_prob=0.10,
        threads=threads,
        mode=mode,
        allow_double=True,
        bid_model=bid_model_path,
        bid_model_prob=0.75,
    )
    idx_range = torch.arange(bgc.MAX_LEGAL, device=device).unsqueeze(0)
    ce_m = {t: 0.0 for t in range(1, 8)}
    ce_s = {t: 0.0 for t in range(1, 8)}
    cnt = {t: 0 for t in range(1, 8)}
    finished = np.zeros(num_deals, dtype=bool)

    with _autocast(device):
        for _ in range(MAX_DECISIONS):
            if np.all(finished):
                break
            batch = env.observe_batch(include_history=True, include_beliefs=True)
            s_t = torch.from_numpy(batch["state"]).to(device)
            a_t = torch.from_numpy(batch["legal_actions"][:, :, :32]).to(device)
            h_t = torch.from_numpy(batch["history"]).to(device)
            np_t = torch.from_numpy(batch["n_plies"]).to(device)
            nl_t = torch.from_numpy(batch["n_legal"]).to(device)
            bm_t = torch.from_numpy(batch["belief_marginals"]).to(device)
            bt_t = torch.from_numpy(batch["belief_targets"]).to(device).long()

            active_idx = torch.from_numpy(np.nonzero(~finished)[0]).to(device)
            if active_idx.numel() > 0:
                logits = b_model(s_t[active_idx], h_t[active_idx], np_t[active_idx], bm_t[active_idx])
                lp_m = F.log_softmax(logits.float(), dim=-1)
                lp_s = torch.log(bm_t[active_idx].float().clamp(min=1e-6))
                bt_a = bt_t[active_idx]
                tricks = (np_t[active_idx] // 4 + 1).clamp(1, 7).cpu().numpy()
                vc = bt_a >= 0
                st = bt_a.clamp(min=0).unsqueeze(-1)
                nll_m = (-lp_m.gather(-1, st).squeeze(-1) * vc).sum(dim=-1).cpu().numpy()
                nll_s = (-lp_s.gather(-1, st).squeeze(-1) * vc).sum(dim=-1).cpu().numpy()
                cc = vc.sum(dim=-1).cpu().numpy()
                for i, tr in enumerate(tricks):
                    ti = int(tr)
                    ce_m[ti] += float(nll_m[i])
                    ce_s[ti] += float(nll_s[i])
                    cnt[ti] += int(cc[i])

            logits32 = actor(s_t).float()
            scores = torch.bmm(a_t.float(), logits32.unsqueeze(-1)).squeeze(-1)
            valid = idx_range < nl_t.unsqueeze(1)
            scores = scores.masked_fill(~valid, -1e9)
            act = scores.argmax(dim=-1).cpu().numpy().astype(np.int32)
            done, _ = env.step_batch(act, auto_reset=False)
            finished |= (done != 0)

    tot = max(1, sum(cnt.values()))
    per_t = {t: ce_m[t] / max(1, cnt[t]) for t in range(1, 8)}
    return sum(ce_m.values()) / tot, sum(ce_s.values()) / tot, per_t


def main():
    ap = argparse.ArgumentParser(description="Dedicated 384x192 GRU BeliefNet v2 trainer for Bulgarian Belot")
    ap.add_argument("--mode", choices=["suit", "at", "nt"], required=True)
    ap.add_argument("--actor-fast", type=Path, required=True)
    ap.add_argument("--baseline-belief", type=Path, default=None)
    ap.add_argument("--resume", type=Path, default=None, help="Optional .pt state_dict to warm-start BeliefNet")
    ap.add_argument("--bid-model", type=str, default="checkpoints/bg_models_bid_unmasked_dbl/bg_bid.bin")
    ap.add_argument("--episodes", type=int, default=1200000)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--state-hidden", type=int, default=384)
    ap.add_argument("--gru-hidden", type=int, default=192)
    ap.add_argument("--lr", type=float, default=6.0e-4)
    ap.add_argument("--min-lr", type=float, default=2.0e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--sample-temp", type=float, default=0.35)
    ap.add_argument("--eval-every", type=int, default=150)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.bfloat16 if dev.type == "cuda" else torch.float32

    actor = load_bgp1(args.actor_fast, dev)
    print(f"[{args.mode}] Loaded actor FastPolicy {args.actor_fast} ({actor.h1}x{actor.h2})", flush=True)

    if args.baseline_belief and args.baseline_belief.exists():
        base_b = load_bgb1(args.baseline_belief, dev)
        b_ce, s_ce, b_pt = eval_belief_unseen(base_b, actor, dev, args.mode, args.bid_model, threads=args.threads)
        t_str = " ".join(f"T{t}={b_pt[t]:.4f}" for t in range(1, 8))
        print(
            f"[{args.mode}] BASELINE ({base_b.state_hidden}x{base_b.gru_hidden}) {args.baseline_belief}: "
            f"CE={b_ce:.4f} (sampler={s_ce:.4f}, gain={s_ce - b_ce:+.4f}) | {t_str}",
            flush=True,
        )
        del base_b

    env = bgc.BgBatchPlayEnv(
        num_envs=args.envs,
        seed=args.seed,
        bid_noise=0.05,
        hanging_prob=0.10,
        threads=args.threads,
        mode=args.mode,
        allow_double=True,
        bid_model=args.bid_model,
        bid_model_prob=0.75,
    )
    model = BgBeliefNet(
        state_dim=bgc.STATE_DIM,
        ply_dim=bgc.PLY_DIM,
        state_hidden=args.state_hidden,
        gru_hidden=args.gru_hidden,
    ).to(dev)
    if args.resume and args.resume.exists():
        model.load_state_dict(torch.load(args.resume, map_location="cpu", weights_only=True))
        print(f"[{args.mode}] Warm-started BeliefNet ({args.state_hidden}x{args.gru_hidden}) from {args.resume}", flush=True)
    permuter = SuitPermuter(args.mode, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_rounds = max(1, math.ceil(args.episodes / args.envs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_rounds, eta_min=args.min_lr)

    bsz = args.envs
    buf_state = torch.empty((MAX_DECISIONS, bsz, bgc.STATE_DIM), dtype=dt, device=dev)
    buf_hist = torch.zeros((MAX_DECISIONS, bsz, bgc.MAX_PLIES, bgc.PLY_DIM), dtype=dt, device=dev)
    buf_np = torch.zeros((MAX_DECISIONS, bsz), dtype=torch.int32, device=dev)
    buf_bt = torch.empty((MAX_DECISIONS, bsz, 32), dtype=torch.int8, device=dev)
    buf_bm = torch.empty((MAX_DECISIONS, bsz, 32, 3), dtype=dt, device=dev)
    valid_step = torch.zeros((MAX_DECISIONS, bsz), dtype=torch.bool, device=dev)
    idx_range = torch.arange(bgc.MAX_LEGAL, device=dev).unsqueeze(0)
    stoch_env = torch.zeros(bsz, dtype=torch.bool, device=dev)
    stoch_env[: int(0.30 * bsz)] = True

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_bin = args.out.with_name(args.out.stem + "_best" + args.out.suffix)
    best_ce = 1e9
    t0 = time.perf_counter()
    episodes_done = 0

    for r in range(1, total_rounds + 1):
        env.reset_all(args.seed * 1000003 + r)
        valid_step.zero_()
        finished = np.zeros(bsz, dtype=bool)
        steps_run = 0

        with torch.no_grad(), _autocast(dev):
            for step in range(MAX_DECISIONS):
                if np.all(finished):
                    break
                steps_run = step + 1
                active = ~finished
                batch = env.observe_batch(include_history=True, include_beliefs=True)
                s_t = torch.from_numpy(batch["state"]).to(dev, non_blocking=True)
                a_t = torch.from_numpy(batch["legal_actions"][:, :, :32]).to(dev, non_blocking=True)
                np_np = batch["n_plies"]
                np_t = torch.from_numpy(np_np).to(dev, non_blocking=True)
                nl_t = torch.from_numpy(batch["n_legal"]).to(dev, non_blocking=True)
                max_p = int(np_np.max()) if bsz > 0 else 0
                if max_p > 0:
                    buf_hist[step, :, :max_p, :].copy_(
                        torch.from_numpy(batch["history"][:, :max_p, :]).to(dev, non_blocking=True)
                    )
                buf_state[step].copy_(s_t)
                buf_np[step].copy_(np_t)
                buf_bt[step].copy_(torch.from_numpy(batch["belief_targets"]).to(dev, non_blocking=True))
                buf_bm[step].copy_(torch.from_numpy(batch["belief_marginals"]).to(dev, non_blocking=True))
                valid_step[step].copy_(torch.from_numpy(active).to(dev, non_blocking=True))

                logits32 = actor(s_t).float()
                scores = torch.bmm(a_t.float(), logits32.unsqueeze(-1)).squeeze(-1)
                valid_a = idx_range < nl_t.unsqueeze(1)
                scores = scores.masked_fill(~valid_a, -1e9)
                greedy = scores.argmax(dim=-1)
                if args.sample_temp > 0.0:
                    probs = F.softmax(scores / args.sample_temp, dim=-1)
                    sampled = torch.multinomial(probs, 1).squeeze(-1)
                    act = torch.where(stoch_env & (nl_t > 1), sampled, greedy)
                else:
                    act = greedy
                done, _ = env.step_batch(act.cpu().numpy().astype(np.int32), auto_reset=False)
                finished |= (active & (done != 0))

        episodes_done += bsz
        v_idx = torch.nonzero(valid_step[:steps_run].reshape(-1), as_tuple=False).squeeze(-1)
        if v_idx.numel() == 0:
            continue

        states = buf_state[:steps_run].reshape(-1, bgc.STATE_DIM)[v_idx]
        hists = buf_hist[:steps_run].reshape(-1, bgc.MAX_PLIES, bgc.PLY_DIM)[v_idx]
        n_plies = buf_np[:steps_run].reshape(-1)[v_idx]
        bt_all = buf_bt[:steps_run].reshape(-1, 32)[v_idx]
        bm_all = buf_bm[:steps_run].reshape(-1, 32, 3)[v_idx]

        model.train()
        buckets = (n_plies // 4).clamp(0, 7)
        schedule = []
        for bkt in range(8):
            bidx = torch.nonzero(buckets == bkt, as_tuple=False).squeeze(-1)
            nb = bidx.numel()
            if nb == 0:
                continue
            perm = bidx[torch.randperm(nb, device=dev)]
            h_len = max(1, int(n_plies[bidx].max().item()))
            for start in range(0, nb, args.batch_size):
                schedule.append((h_len, perm[start : start + args.batch_size]))
        order = rng.permutation(len(schedule))

        tot_ce = 0.0
        n_steps = 0
        for oi in order:
            h_len, idx = schedule[int(oi)]
            sb = states[idx]
            hb = hists[idx, :h_len, :]
            npb = n_plies[idx]
            btb = bt_all[idx]
            bmb = bm_all[idx]

            pid = permuter.sample_perm_ids(sb.shape[0], dev)
            sb = permuter.permute_state(sb, pid)
            hb = permuter.permute_history(hb, pid)
            btb = permuter.permute_cards_1d(btb, pid)
            bmb = permuter.permute_cards_2d(bmb, pid)

            mask = btb >= 0
            if not mask.any():
                continue
            opt.zero_grad(set_to_none=True)
            with _autocast(dev):
                logits = model(sb, hb, npb, bmb)
                loss = F.cross_entropy(logits[mask].float(), btb[mask].long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tot_ce += float(loss.item())
            n_steps += 1
        sched.step()

        if r % 25 == 0 or r == total_rounds:
            dps = episodes_done / max(time.perf_counter() - t0, 1e-6)
            print(
                f"[{args.mode} {episodes_done:>9,}/{args.episodes:,}] {dps:6.0f} d/s | "
                f"lr={opt.param_groups[0]['lr']:.2e} | belief_ce={tot_ce/max(n_steps,1):.4f}",
                flush=True,
            )

        if r % args.eval_every == 0 or r == total_rounds:
            model.export_bin(args.out)
            m_ce, s_ce, per_t = eval_belief_unseen(model, actor, dev, args.mode, args.bid_model, threads=args.threads)
            tag = ""
            if m_ce < best_ce:
                best_ce = m_ce
                model.export_bin(best_bin)
                torch.save(model.state_dict(), best_bin.with_suffix(".pt"))
                tag = "  *NEW BEST*"
            t_str = " ".join(f"T{t}={per_t[t]:.4f}" for t in range(1, 8))
            print(
                f"  [eval @{episodes_done:,}] CE={m_ce:.4f} (sampler={s_ce:.4f}, gain={s_ce - m_ce:+.4f}) | {t_str}{tag}",
                flush=True,
            )

    print(f"[{args.mode}] Done! Best CE={best_ce:.4f} -> {best_bin}", flush=True)


if __name__ == "__main__":
    main()
