"""Phase B Stage B1 (v2 — Zero-Overfit Streaming + EMA + Paired Advantage):
Bulgarian Bridge-Belot Neural Bidder (BgBidNet).

Key fixes over v1 (which overfitted 18 passes on 6-iter replay: train RMSE 17.3 vs
val RMSE 21.5 -> +9.6 pt argmax winner's curse on NT/AT):
1. Pure 1-pass streaming on 100% FRESH deals every iter (zero multi-epoch memorization;
   train RMSE == val RMSE at all times!).
2. EMA weights (Polyak decay) exported to C++ for self-play and evaluation — eliminates
   single-batch SGD jitter and argmax overestimation bias.
3. Paired-advantage auxiliary MSE on (Q(s,a) - Q(s,Pass)) vs (y_a - y_Pass) on the exact
   same 32-card deal (same unbiased Bayes minimum Q*(s,a), lower CRN variance).
4. Warmup on naturalBid completions (first --warmup-iters iters) before entering EMA
   self-play, plus held-out val_rmse & chosen-bid calibration bias logged every eval!
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import struct
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build"))
import bg  # noqa: E402,F401  (puts cardbot-framework's package on the path)
import bg_cpp  # noqa: E402
from bg.envs import BidEnv  # noqa: E402
from cardbot.train import bidding  # noqa: E402


class TeeStream:
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        return len(data)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.ln1(self.fc1(x)))
        h = self.drop(h)
        h = self.ln2(self.fc2(h))
        return F.silu(x + h)


class BgBidNet(nn.Module):
    def __init__(
        self,
        suit_dim: int = bg_cpp.BID_SUIT_DIM,
        global_dim: int = bg_cpp.BID_GLOBAL_DIM,
        step_dim: int = bg_cpp.BID_STEP_DIM,
        suit_hidden: int = 32,
        global_hidden: int = 96,
        gru_hidden: int = 48,
        num_bids: int = bg_cpp.NUM_BIDS,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.suit_dim = suit_dim
        self.global_dim = global_dim
        self.step_dim = step_dim
        self.suit_hidden = suit_hidden
        self.global_hidden = global_hidden
        self.gru_hidden = gru_hidden
        self.num_bids = num_bids
        self.trunk_dim = 4 * suit_hidden + global_hidden + gru_hidden

        self.suit_fc1 = nn.Linear(suit_dim, suit_hidden)
        self.suit_ln1 = nn.LayerNorm(suit_hidden)
        self.suit_fc2 = nn.Linear(suit_hidden, suit_hidden)
        self.suit_ln2 = nn.LayerNorm(suit_hidden)

        self.glob_fc = nn.Linear(global_dim, global_hidden)
        self.glob_ln = nn.LayerNorm(global_hidden)

        self.hist_gru = nn.GRU(step_dim, gru_hidden, batch_first=True)
        self.empty_emb = nn.Parameter(torch.zeros(gru_hidden))

        self.trunk1 = ResBlock(self.trunk_dim, dropout=dropout)
        self.trunk2 = ResBlock(self.trunk_dim, dropout=dropout)
        self.head = nn.Linear(self.trunk_dim, num_bids)

    def encode_history(self, bid_hist: torch.Tensor, n_bids: torch.Tensor) -> torch.Tensor:
        B = bid_hist.shape[0]
        h_out = self.empty_emb.unsqueeze(0).expand(B, -1).clone()
        non_empty = n_bids > 0
        if non_empty.any():
            idx = torch.nonzero(non_empty, as_tuple=False).squeeze(1)
            lens = n_bids[idx].to("cpu", dtype=torch.int64)
            packed = nn.utils.rnn.pack_padded_sequence(
                bid_hist[idx], lens, batch_first=True, enforce_sorted=False
            )
            _, h_last = self.hist_gru(packed)
            h_out[idx] = h_last.squeeze(0).to(h_out.dtype)
        return h_out

    def forward(
        self,
        suit_feats: torch.Tensor,  # [B, 4, 16]
        global_feats: torch.Tensor,  # [B, 64]
        bid_hist: torch.Tensor,  # [B, 16, 13]
        n_bids: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        B = suit_feats.shape[0]
        s = F.silu(self.suit_ln1(self.suit_fc1(suit_feats)))
        s = F.silu(self.suit_ln2(self.suit_fc2(s)))
        s = s.reshape(B, 4 * self.suit_hidden)

        g = F.silu(self.glob_ln(self.glob_fc(global_feats)))
        h = self.encode_history(bid_hist, n_bids)

        x = torch.cat([s, g, h], dim=-1)
        x = self.trunk1(x)
        x = self.trunk2(x)
        return self.head(x)


def _write_f32(f, t: torch.Tensor) -> None:
    arr = t.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
    f.write(arr.tobytes())


def _write_linear(f, mod: nn.Linear) -> None:
    _write_f32(f, mod.weight)
    _write_f32(f, mod.bias)


def _write_ln(f, mod: nn.LayerNorm) -> None:
    _write_f32(f, mod.weight)
    _write_f32(f, mod.bias)


def _write_resblock(f, rb: ResBlock) -> None:
    _write_linear(f, rb.fc1)
    _write_ln(f, rb.ln1)
    _write_linear(f, rb.fc2)
    _write_ln(f, rb.ln2)


def export_bid_bin(model: BgBidNet, path: Path, use_natural_mask: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    ver = bg_cpp.BID_VERSION if use_natural_mask else bg_cpp.BID_VERSION_UNMASKED
    with open(tmp_path, "wb") as f:
        f.write(b"BGBD")
        f.write(
            struct.pack(
                "<8i",
                ver,
                model.suit_dim,
                model.global_dim,
                model.step_dim,
                model.suit_hidden,
                model.global_hidden,
                model.gru_hidden,
                model.num_bids,
            )
        )
        _write_linear(f, model.suit_fc1)
        _write_ln(f, model.suit_ln1)
        _write_linear(f, model.suit_fc2)
        _write_ln(f, model.suit_ln2)
        _write_linear(f, model.glob_fc)
        _write_ln(f, model.glob_ln)
        _write_f32(f, model.hist_gru.weight_ih_l0)
        _write_f32(f, model.hist_gru.weight_hh_l0)
        _write_f32(f, model.hist_gru.bias_ih_l0)
        _write_f32(f, model.hist_gru.bias_hh_l0)
        _write_f32(f, model.empty_emb)
        _write_resblock(f, model.trunk1)
        _write_resblock(f, model.trunk2)
        _write_linear(f, model.head)
    os.replace(tmp_path, path)


@torch.no_grad()
def verify_cpp_parity(
    model: BgBidNet, bin_path: Path, sample_batch: Dict[str, np.ndarray], use_natural_mask: bool = True
) -> float:
    export_bid_bin(model, bin_path, use_natural_mask=use_natural_mask)
    cpp_m = bg_cpp.BgBidModel()
    if not cpp_m.load(str(bin_path)):
        raise RuntimeError(f"Failed to load exported {bin_path} in C++ BgBidModel")

    model.eval()
    device = next(model.parameters()).device
    n_check = min(32, int(sample_batch["n_samples"]))
    sf = torch.from_numpy(sample_batch["suit_feats"][:n_check]).to(device)
    gf = torch.from_numpy(sample_batch["global_feats"][:n_check]).to(device)
    bh = torch.from_numpy(sample_batch["bid_hist"][:n_check]).to(device)
    nb = torch.from_numpy(sample_batch["n_bids"][:n_check]).to(device)
    py_q = model(sf, gf, bh, nb).cpu().numpy()

    max_diff = 0.0
    for i in range(n_check):
        cpp_q = cpp_m.forward_q(
            sample_batch["suit_feats"][i],
            sample_batch["global_feats"][i],
            sample_batch["bid_hist"][i],
            int(sample_batch["n_bids"][i]),
        )
        diff = float(np.max(np.abs(py_q[i] - cpp_q)))
        if diff > max_diff:
            max_diff = diff
    return max_diff


@torch.no_grad()
def eval_calibration(
    model: BgBidNet, env: bg_cpp.BgBatchBidEnv, device: torch.device, allow_double: bool, use_natural_mask: bool
) -> str:
    b = env.generate_batch(1000, seed=88888, explore_eps=0.0, allow_double=allow_double, use_natural_mask=use_natural_mask)
    N = int(b["n_samples"])
    if N == 0:
        return "val_rmse=n/a"
    sf = torch.from_numpy(b["suit_feats"]).to(device)
    gf = torch.from_numpy(b["global_feats"]).to(device)
    bh = torch.from_numpy(b["bid_hist"]).to(device)
    nb = torch.from_numpy(b["n_bids"]).to(device)
    lm = b["legal_mask"]
    tg = b["targets"]

    model.eval()
    preds = []
    for s in range(0, N, 2048):
        preds.append(model(sf[s : s + 2048], gf[s : s + 2048], bh[s : s + 2048], nb[s : s + 2048]).cpu().numpy())
    pred = np.concatenate(preds, axis=0)

    diff = (pred - tg) * lm
    val_rmse = math.sqrt(float((diff * diff).sum() / max(1.0, float(lm.sum())))) * 30.0
    masked_p = np.where(lm > 0.5, pred, -1e9)
    chosen = masked_p.argmax(axis=1)
    idx = np.arange(N)
    non_pass = chosen > 0
    if non_pass.any():
        bias_np = float((pred[idx[non_pass], chosen[non_pass]] - tg[idx[non_pass], chosen[non_pass]]).mean()) * 30.0
        adv_actual = float((tg[idx[non_pass], chosen[non_pass]] - tg[idx[non_pass], 0]).mean()) * 30.0
    else:
        bias_np = 0.0
        adv_actual = 0.0
    return f"val_rmse={val_rmse:.2f}pts nonpass_bias={bias_np:+.2f}pts actual_adv={adv_actual:+.2f}pts"


def run_eval_suite(
    model: BgBidNet,
    env: bg_cpp.BgBatchBidEnv,
    device: torch.device,
    eval_deals: int,
    eval_seed: int,
    allow_double: bool,
    use_natural_mask: bool,
) -> str:
    cal = eval_calibration(model, env, device, allow_double, use_natural_mask)
    r0 = env.eval_vs_natural(eval_deals, eval_seed, 0, False, allow_double)
    r10 = env.eval_vs_natural(eval_deals, eval_seed + 100, 10, False, allow_double)
    rmix = env.eval_vs_natural(eval_deals, eval_seed + 200, 0, True, allow_double)
    dbl_str = f" x2/x4={r0['pct_dbl']:.1f}/{r0['pct_rdbl']:.1f}%" if allow_double else ""
    return (
        f"{cal} | "
        f"vs nat(0): {r0['mean']:+.2f}±{r0['ci']:.2f} | "
        f"vs nat(+10AT): {r10['mean']:+.2f}±{r10['ci']:.2f} | "
        f"mixed(Bid+nat): {rmix['mean']:+.2f}±{rmix['ci']:.2f} | "
        f"mix S/NT/AT/P={r0['pct_suit']:.1f}/{r0['pct_nt']:.1f}/{r0['pct_at']:.1f}/{r0['pct_pass']:.1f}%{dbl_str} | "
        f"audit J9={r0['suit_has_j9']:.1f}% A={r0['nt_has_ace']:.1f}% J={r0['at_has_j']:.1f}%"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Bulgarian Bridge-Belot Neural Bidder (BgBidNet v2)")
    ap.add_argument("--fast-at", default="checkpoints/bg_models_at/bg_fast_policy.bin")
    ap.add_argument("--fast-suit", default="checkpoints/bg_models_suit/bg_fast_policy.bin")
    ap.add_argument("--fast-nt", default="checkpoints/bg_models_nt_latest/bg_fast_policy.bin")
    ap.add_argument("--out-dir", default="checkpoints/bg_models_bid")
    ap.add_argument("--ckpt", default="checkpoints/bg_bid_stage1_latest.pt")
    ap.add_argument("--log-file", default="runs/bg_bid_stage1_v2.log")
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--warmup-iters", type=int, default=8)
    ap.add_argument("--deals-per-iter", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--adv-weight", type=float, default=0.5)
    ap.add_argument("--ema-decay", type=float, default=0.992)
    ap.add_argument("--explore-start", type=float, default=0.10)
    ap.add_argument("--explore-end", type=float, default=0.02)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--eval-deals", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--suit-hidden", type=int, default=32)
    ap.add_argument("--global-hidden", type=int, default=96)
    ap.add_argument("--gru-hidden", type=int, default=48)
    ap.add_argument("--allow-double", action="store_true")
    ap.add_argument("--unmasked", action="store_true", help="Disable natural sanity mask; train on ALL legal Belot bids")
    ap.add_argument("--resume", default="")
    args = ap.parse_args()
    use_natural_mask = not args.unmasked

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "a", buffering=1)
    sys.stdout = TeeStream(sys.stdout, log_fh)
    sys.stderr = TeeStream(sys.stderr, log_fh)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / "bg_bid.bin"

    print(
        f"=== BgBidNet Stage B1 v2 (1-Pass Fresh Streaming + EMA + Paired Adv) === "
        f"device={device} bf16={use_bf16} iters={args.iters} warmup={args.warmup_iters} "
        f"deals/iter={args.deals_per_iter} lr={args.lr} wd={args.weight_decay} "
        f"adv_w={args.adv_weight} ema={args.ema_decay} allow_double={args.allow_double} "
        f"natural_mask={use_natural_mask}",
        flush=True,
    )

    env = bg_cpp.BgBatchBidEnv(args.fast_at, args.fast_suit, args.fast_nt, threads=args.threads)
    model = BgBidNet(
        suit_hidden=args.suit_hidden, global_hidden=args.global_hidden, gru_hidden=args.gru_hidden
    ).to(device)
    ema_model = copy.deepcopy(model).to(device)

    if args.resume:
        sd = torch.load(args.resume, map_location=device)
        model.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd)
        ema_model.load_state_dict(sd["ema_model"] if isinstance(sd, dict) and "ema_model" in sd else model.state_dict())
        export_bid_bin(ema_model, bin_path, use_natural_mask=use_natural_mask)
        env.load_bid_model(str(bin_path))
        print(f"Resumed from {args.resume}", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"BgBidNet params: {n_params:,} (trunk_dim={model.trunk_dim})", flush=True)

    probe = env.generate_batch(
        64, seed=args.seed + 999999, explore_eps=0.1, allow_double=args.allow_double, use_natural_mask=use_natural_mask
    )
    max_diff = verify_cpp_parity(ema_model, bin_path, probe, use_natural_mask=use_natural_mask)
    print(f"C++ <-> PyTorch parity check ({min(32, probe['n_samples'])} states): max |diff| = {max_diff:.2e}", flush=True)
    if max_diff > 1e-4:
        raise RuntimeError(f"Parity check failed: max_diff={max_diff}")
    if not args.resume:
        bin_path.unlink(missing_ok=True)

    # Training: cardbot-framework's counterfactual bid trainer (cardbot.train.bidding) with
    # Belot's rows, bid net and export plugged in.
    benv = BidEnv(env)
    totals = {"deals": 0, "nodes": 0, "branches": 0, "br": 0}
    t0 = time.time()

    def generate(_env, cfg, it, eps):
        batch = env.generate_batch(args.deals_per_iter, seed=args.seed * 10000 + it + 1, explore_eps=eps,
                                   allow_double=args.allow_double, use_natural_mask=use_natural_mask)
        totals["br"] = int(batch["legal_mask"].sum())
        totals["deals"] += args.deals_per_iter
        totals["nodes"] += int(batch["n_samples"])
        totals["branches"] += totals["br"]
        return {k: batch[k] for k in ("suit_feats", "global_feats", "bid_hist", "n_bids", "legal_mask", "targets")}

    def on_iter(i, ema, info):
        it = i + 1
        in_selfplay = (it >= args.warmup_iters) or bool(args.resume)
        if it == args.warmup_iters and not args.resume:
            wu_path = out_dir / "bg_bid_wu.bin"
            export_bid_bin(ema, wu_path, use_natural_mask=use_natural_mask)
            print(f"   Saved pure-anchored warmup model to {wu_path}", flush=True)
        n_samp = info["n"]
        rmse_pts = math.sqrt(max(0.0, info["mse"])) * 30.0
        phase = "SP" if in_selfplay else "WU"
        msg = (
            f"iter {it:3d}/{args.iters} [{phase}] | deals={totals['deals']:7d} nodes={n_samp:5d} "
            f"br/node={totals['br'] / max(1, n_samp):.2f} | trn_rmse={rmse_pts:.2f}pts lr={info['lr']:.1e} "
            f"eps={info['eps']:.2f} | gen={info['gen_s']:.1f}s trn={info['train_s']:.1f}s ({time.time() - t0:.0f}s)"
        )
        if it == args.warmup_iters or it % args.eval_every == 0 or it == args.iters:
            # The EMA weights are evaluated even during the warm-up.
            if not in_selfplay:
                export_bid_bin(ema, bin_path, use_natural_mask=use_natural_mask)
                env_eval = bg_cpp.BgBatchBidEnv(args.fast_at, args.fast_suit, args.fast_nt, threads=args.threads)
                env_eval.load_bid_model(str(bin_path))
                ev = run_eval_suite(ema, env_eval, device, args.eval_deals, eval_seed=54321,
                                    allow_double=args.allow_double, use_natural_mask=use_natural_mask)
            else:
                ev = run_eval_suite(ema, env, device, args.eval_deals, eval_seed=54321,
                                    allow_double=args.allow_double, use_natural_mask=use_natural_mask)
            msg += f"\n   EVAL: {ev}"
        print(msg, flush=True)

    cfg = bidding.BidConfig(
        iters=args.iters, deals_per_iter=args.deals_per_iter, batch_size=args.batch_size, lr=args.lr,
        min_lr=args.min_lr, weight_decay=args.weight_decay, eps_start=args.explore_start, eps_end=args.explore_end,
        adv_weight=args.adv_weight, ref_slot=0, ema_decay=args.ema_decay,
        warmup_iters=0 if args.resume else args.warmup_iters, out=bin_path, amp=use_bf16, seed=args.seed,
        device=str(device), log=None,
        predict=lambda m, b: m(b["suit_feats"], b["global_feats"], b["bid_hist"], b["n_bids"]),
        export=lambda net, path: export_bid_bin(net, Path(path), use_natural_mask=use_natural_mask))
    ema_model = bidding.train(benv, model, cfg, generate_fn=generate, on_iter=on_iter, ema=ema_model)
    total_deals, total_nodes, total_branches = totals["deals"], totals["nodes"], totals["branches"]

    # Save latest fully-annealed EMA checkpoint & binary
    ckpt_path = Path(args.ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "args": vars(args),
            "total_deals": total_deals,
            "total_nodes": total_nodes,
            "total_branches": total_branches,
        },
        ckpt_path,
    )
    export_bid_bin(ema_model, bin_path, use_natural_mask=use_natural_mask)
    final_diff = verify_cpp_parity(ema_model, bin_path, probe, use_natural_mask=use_natural_mask)
    print(
        f"Done! Saved latest EMA checkpoint {ckpt_path} and binary {bin_path} "
        f"(final parity max |diff| = {final_diff:.2e}, total branches = {total_branches:,})",
        flush=True,
    )


if __name__ == "__main__":
    main()
