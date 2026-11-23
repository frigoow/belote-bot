"""Bulgarian Belot's card-play trainer on cardbot-framework (cardbot.train):

- DMC for the dueling BgCardQGRU (cardbot.train.dmc.DMCTrainer: GPU-resident
  BF16 collection, a rolling replay window, history-length buckets, AdamW +
  cosine), with Belot's parts as overrides: the training-only card-ownership
  loss, the exact suit-permutation augmentation (--suit-perm), and the
  residual BgBeliefNet trained every --belief-every rounds on the round's
  exact DealSampler::exactMarginals();
- the 256x128 BgFastPolicy distilled from the converged Q-GRU
  (cardbot.train.distill);
- evaluation against the C++ rulePolicy, full checkpoints (optimizer and
  scheduler too) so long runs resume cleanly, and the C++ exports (BGQ1, BGB1,
  BGP1).

Usage:
  PYTHONPATH=build:python .venv/bin/python -m bg.train_dmc \
      --episodes 5000000 --envs 2048 --batch-size 2048 --train-belief \
      --save checkpoints/bg_q_gru_5m.pt --export-dir checkpoints/bg_models
"""

from __future__ import annotations

import argparse
import datetime
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import bg  # noqa: F401  (puts cardbot-framework's package on the path)
import bg_cpp as bgc
from bg.envs import PlayEnv
from bg.eval_bg import eval_belief_per_trick, eval_vs_cpp
from bg.models import BgBeliefNet, BgCardQGRU, BgFastPolicy
from bg.suit_perm import SuitPermuter
from cardbot.train import distill
from cardbot.train.dmc import DMCConfig, DMCTrainer, autocast

MAX_DECISIONS = 28  # Trick 8 (4 cards) is always auto-played by bg::Round.


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
    bid_model: str = "",
    bid_model_prob: float = 0.65,
    suit_perm: bool = False,
    threads: int = 12,
) -> BgFastPolicy:
    """Distill the compact 256x128 BgFastPolicy from the converged BgCardQGRU."""
    print(f"\n--- Distilling 256x128 BgFastPolicy (mode={mode}, dbl={allow_double}, suit_perm={suit_perm}, "
          f"bid={bid_model or 'nat'}) over {num_deals:,} deals (T={temperature}) ---", flush=True)
    q_model.eval()
    fast = BgFastPolicy().to(device)
    env = PlayEnv(num_envs=batch_envs, seed=777, bid_noise=0.0, hanging_prob=0.0, threads=threads, mode=mode,
                  allow_double=allow_double, bid_model=bid_model, bid_model_prob=bid_model_prob)
    permuter = SuitPermuter(mode, device) if suit_perm else None

    def augment(state, mask, target):
        pid = permuter.sample_perm_ids(state.shape[0], device)
        return (permuter.permute_state(state, pid), permuter.permute_cards_1d(mask, pid),
                permuter.permute_cards_1d(target, pid))

    cfg = distill.DistillConfig(rounds=max(1, num_deals // batch_envs), epochs=epochs, batch_size=2048, lr=1.5e-3,
                                min_lr=0.0, weight_decay=1e-4, temperature=temperature, eps=0.0, min_legal=2, seed=1,
                                collect_seed=lambda r: 777000 + r, device=str(device),
                                log=lambda m: print("  " + m, flush=True))
    distill.train(env, q_model, fast, cfg, augment=augment if permuter is not None else None)
    fast.train()
    if save_bin is not None:
        fast.export_bin(save_bin)
        print(f"Exported BgFastPolicy binary weights to {save_bin}", flush=True)
    return fast


class BelotDMC(DMCTrainer):
    """cardbot's DMC trainer with Belot's ownership loss, suit permutations and belief net."""

    def __init__(self, env, q_model, cfg: DMCConfig, args, b_model=None, permuter=None):
        super().__init__(env, q_model, cfg)
        self.args = args
        self.b_model = b_model
        self.permuter = permuter
        self.need_belief_data = args.train_belief or args.aux_own_coef > 0.0
        self.b_opt = torch.optim.AdamW(b_model.parameters(), lr=args.lr, weight_decay=1e-4) if b_model else None
        b_rounds = max(1, math.ceil(self.total_rounds / max(1, args.belief_every)))
        self.b_sched = (torch.optim.lr_scheduler.CosineAnnealingLR(self.b_opt, T_max=b_rounds, eta_min=args.min_lr)
                        if self.b_opt else None)
        self.train_belief_now = False
        self.b_loss = 0.0
        self.n_b_mb = 0
        self.next_eval = 0
        self.best_vs_policy = -1e9
        self.best_belief_ce = 1e9

    def env_flags(self) -> None:
        self.train_belief_now = self.b_model is not None and self.round_idx % max(1, self.args.belief_every) == 0
        self.env.belief_targets = self.need_belief_data
        self.env.belief_marginals = self.train_belief_now
        if self.b_model is not None:
            self.b_model.train()

    def augment(self, batch: dict) -> dict:
        if self.permuter is None:
            return batch
        p = self.permuter
        pid = p.sample_perm_ids(batch["state"].shape[0], self.device)
        batch["state"] = p.permute_state(batch["state"], pid)
        batch["history"] = p.permute_history(batch["history"], pid)
        batch["actions"] = p.permute_actions(batch["actions"], pid)
        if "belief_targets" in batch:
            batch["belief_targets"] = p.permute_cards_1d(batch["belief_targets"], pid)
        return batch

    def loss(self, batch: dict):
        valid = self.idx_range < batch["n_legal"].unsqueeze(1)
        bt = batch.get("belief_targets")
        with autocast(self.device, self.amp):
            if self.args.aux_own_coef > 0.0 and bt is not None:
                q_all, own = self.model(batch["state"], batch["actions"], batch["history"], batch["n_plies"],
                                        valid_mask=valid, return_aux_own=True)
            else:
                q_all = self.model(batch["state"], batch["actions"], batch["history"], batch["n_plies"],
                                   valid_mask=valid)
                own = None
        q_taken = q_all.float().gather(1, batch["chosen"].unsqueeze(1)).squeeze(1)
        q_mse = F.mse_loss(q_taken, batch["ret"])
        if own is None:
            return q_mse, {"q_mse": q_mse}
        own_ce = F.cross_entropy(own.float().reshape(-1, 3), bt.reshape(-1), ignore_index=-1)
        return q_mse + self.args.aux_own_coef * own_ce, {"q_mse": q_mse, "own_ce": own_ce}

    def after_round(self, new: dict) -> None:
        """The residual belief net on the fresh round's exact DealSampler marginals."""
        self.b_loss, self.n_b_mb = 0.0, 0
        if not (self.train_belief_now and self.b_model is not None and "belief_marginals" in new):
            return
        b = {k: new[k] for k in ("state", "history", "n_plies", "belief_marginals", "belief_targets")}
        n_plies = b["n_plies"]
        buckets = (n_plies // 8).clamp(0, 3)
        schedule = []
        for bkt in range(4):
            idx = torch.nonzero(buckets == bkt, as_tuple=False).squeeze(-1)
            nb = idx.numel()
            if nb == 0:
                continue
            perm = idx[torch.randperm(nb, device=self.device)]
            h_len = max(1, int(n_plies[idx].max().item()))
            for start in range(0, nb, self.cfg.batch_size):
                schedule.append((h_len, perm[start:start + self.cfg.batch_size]))
        for oi in self.rng.permutation(len(schedule)):
            h_len, idx = schedule[int(oi)]
            s, h, npl = b["state"][idx], b["history"][idx, :h_len, :], n_plies[idx]
            bm, bt = b["belief_marginals"][idx], b["belief_targets"][idx]
            if self.permuter is not None:
                p = self.permuter
                pid = p.sample_perm_ids(s.shape[0], self.device)
                s, h = p.permute_state(s, pid), p.permute_history(h, pid)
                bm, bt = p.permute_cards_2d(bm, pid), p.permute_cards_1d(bt, pid)
            with autocast(self.device, self.amp):
                logits = self.b_model(s, h, npl, bm)
            b_ce = F.cross_entropy(logits.float().reshape(-1, 3), bt.reshape(-1), ignore_index=-1)
            self.b_opt.zero_grad(set_to_none=True)
            b_ce.backward()
            torch.nn.utils.clip_grad_norm_(self.b_model.parameters(), 2.0)
            self.b_opt.step()
            self.b_loss += float(b_ce.item())
            self.n_b_mb += 1
        self.b_sched.step()

    def should_log(self) -> bool:
        return (self.round_idx % 25 == 0 or self.episodes_done >= self.next_eval
                or self.episodes_done >= self.cfg.episodes)

    def log_line(self, logs: dict, t_start: float, start_episodes: int) -> str:
        dps = (self.episodes_done - start_episodes) / max(time.perf_counter() - t_start, 1e-6)
        msg = (f"[{self.episodes_done:>9,}/{self.cfg.episodes:,}] {dps:6.0f} d/s | eps={logs['eps']:.3f} "
               f"lr={self.opt.param_groups[0]['lr']:.2e} | q_mse={logs.get('q_mse', 0.0):.4f}")
        if self.args.aux_own_coef > 0.0:
            msg += f" own_ce={logs.get('own_ce', 0.0):.4f}"
        if self.train_belief_now:
            msg += f" b_ce={self.b_loss / max(1, self.n_b_mb):.4f}"
        return msg

    def save_ckpt(self, path: Path, fast_policy: BgFastPolicy | None = None) -> None:
        payload = {
            "q_model": self.model.state_dict(),
            "q_opt": self.opt.state_dict(),
            "q_sched": self.sched.state_dict(),
            "episodes_done": self.episodes_done,
            "best_vs_policy": self.best_vs_policy,
            "best_belief_ce": self.best_belief_ce,
            "encoder_version": bgc.ENCODER_VERSION,
        }
        if self.b_model is not None:
            payload["b_model"] = self.b_model.state_dict()
            payload["b_opt"] = self.b_opt.state_dict()
            payload["b_sched"] = self.b_sched.state_dict()
        if fast_policy is not None:
            payload["fast_policy"] = fast_policy.state_dict()
        torch.save(payload, path)

    def on_round_end(self) -> None:
        a = self.args
        if not (self.episodes_done >= self.next_eval or self.episodes_done >= a.episodes):
            return
        self.next_eval += a.eval_every
        # Selection eval on fixed seed 12345 (held-out reporting uses seed 54321).
        ev = eval_vs_cpp(self.model, self.device, opp_spec="policy", num_deals=a.eval_decks, seed=12345,
                         threads=a.env_threads, mode=a.mode, allow_double=a.allow_double)
        print(f"  >>> EVAL ({self.episodes_done:,}, mode={a.mode}) Q-GRU vs rulePolicy: "
              f"{ev['mean']:+.3f} +- {ev['ci']:.3f} game pts/deal ({ev['pairs']} mirrored pairs)", flush=True)
        if self.b_model is not None:
            br = eval_belief_per_trick(self.model, self.b_model, self.device, num_deals=256, seed=12345,
                                       threads=a.env_threads, mode=a.mode, allow_double=a.allow_double)
            print(f"  >>> BELIEF All={br['model'][0]:.4f} vs DealSampler={br['sampler'][0]:.4f} "
                  f"(T1 {br['model'].get(1,0):.4f}/{br['sampler'].get(1,0):.4f}, "
                  f"T4 {br['model'].get(4,0):.4f}/{br['sampler'].get(4,0):.4f}, "
                  f"T7 {br['model'].get(7,0):.4f}/{br['sampler'].get(7,0):.4f})", flush=True)
            if br["model"][0] < self.best_belief_ce:
                self.best_belief_ce = br["model"][0]
                self.b_model.export_bin(a.export_dir / "bg_belief.bin")
                torch.save({"b_model": self.b_model.state_dict(), "best_belief_ce": self.best_belief_ce,
                            "episodes_done": self.episodes_done},
                           a.save.with_name(a.save.stem + "_best_belief.pt"))
                print(f"  >>> New best B_phi ({self.best_belief_ce:.4f})! Exported bg_belief.bin.", flush=True)
        self.save_ckpt(a.save)
        if ev["mean"] > self.best_vs_policy:
            self.best_vs_policy = ev["mean"]
            best_path = a.save.with_name(a.save.stem + "_best.pt")
            self.save_ckpt(best_path)
            self.model.export_bin(a.export_dir / "bg_q_gru.bin")
            print(f"  >>> New best Q-GRU ({self.best_vs_policy:+.3f})! Saved {best_path} & exported bg_q_gru.bin.",
                  flush=True)


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("all", "suit", "nt", "at"), default="all")
    ap.add_argument("--suit-perm", action="store_true", help="Enable exact S_3 (Suit, trump pinned at 0) / S_4 (AT/NT) suit-permutation augmentation")
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
    ap.add_argument("--bid-model", type=str, default="", help="Optional BgBidModel .bin for generating realistic auctions")
    ap.add_argument("--bid-model-prob", type=float, default=0.65, help="Fraction of deals bid by --bid-model (rest by naturalBidTuned)")
    ap.add_argument("--init-from", type=Path, default=None, help="Warm-start Q/B weights only (fresh optimizer + fresh cosine schedule)")
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

    def distill_now(fast_seed_eval: int):
        args.export_dir.mkdir(parents=True, exist_ok=True)
        fast_policy = distill_fast_policy(
            q_model, device, num_deals=args.distill_deals, save_bin=args.export_dir / "bg_fast_policy.bin",
            mode=args.mode, allow_double=args.allow_double, bid_model=args.bid_model,
            bid_model_prob=args.bid_model_prob, suit_perm=args.suit_perm, threads=args.env_threads)
        ev_fp = eval_vs_cpp(fast_policy, device, opp_spec="policy", num_deals=args.eval_decks, seed=fast_seed_eval,
                            threads=args.env_threads, is_fast_policy=True, mode=args.mode,
                            allow_double=args.allow_double)
        return fast_policy, ev_fp

    if args.distill_only:
        assert args.resume is not None and args.resume.exists(), "--distill-only requires an existing --resume checkpoint"
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        q_model.load_state_dict(ckpt["q_model"])
        _, ev_fp = distill_now(54321)
        print(f"  >>> Distilled BgFastPolicy (mode={args.mode}) vs rulePolicy (seed 54321): "
              f"{ev_fp['mean']:+.3f} +- {ev_fp['ci']:.3f} game pts/deal ({ev_fp['pairs']} pairs)", flush=True)
        return

    env = PlayEnv(num_envs=args.envs, seed=args.seed, bid_noise=args.bid_noise, hanging_prob=args.hanging_prob,
                  threads=args.env_threads, mode=args.mode, allow_double=args.allow_double, bid_model=args.bid_model,
                  bid_model_prob=args.bid_model_prob)
    permuter = SuitPermuter(args.mode, device) if args.suit_perm else None
    need_belief_data = args.train_belief or args.aux_own_coef > 0.0
    cfg = DMCConfig(episodes=args.episodes, batch_size=args.batch_size, replay_rounds=args.replay_rounds,
                    mb_per_round=args.mb_per_round, lr=args.lr, min_lr=args.min_lr, weight_decay=1e-4,
                    eps_start=args.eps_start, eps_end=args.eps_end, grad_clip=2.0, ply_bucket=8, n_buckets=4,
                    max_decisions=MAX_DECISIONS,
                    extra_keys=("belief_targets", "belief_marginals") if need_belief_data else (),
                    ring_keys=("belief_targets",) if need_belief_data else (), seed=args.seed, device=str(device),
                    log=lambda m: print(m, flush=True))
    trainer = BelotDMC(env, q_model, cfg, args, b_model=b_model, permuter=permuter)

    if args.init_from is not None and args.init_from.exists() and (args.resume is None or not args.resume.exists()):
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        q_model.load_state_dict(ckpt["q_model"])
        if b_model is not None and "b_model" in ckpt:
            b_model.load_state_dict(ckpt["b_model"])
        print(f"Warm-started Q/B weights from {args.init_from} (fresh optimizer + {args.episodes:,}-deal schedule)",
              flush=True)
    if args.resume is not None and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        q_model.load_state_dict(ckpt["q_model"])
        if "q_opt" in ckpt:
            trainer.opt.load_state_dict(ckpt["q_opt"])
        if "q_sched" in ckpt:
            trainer.sched.load_state_dict(ckpt["q_sched"])
        if b_model is not None and "b_model" in ckpt:
            b_model.load_state_dict(ckpt["b_model"])
            if "b_opt" in ckpt:
                trainer.b_opt.load_state_dict(ckpt["b_opt"])
            if "b_sched" in ckpt:
                trainer.b_sched.load_state_dict(ckpt["b_sched"])
        trainer.episodes_done = int(ckpt.get("episodes_done", 0))
        trainer.best_vs_policy = float(ckpt.get("best_vs_policy", -1e9))
        trainer.best_belief_ce = float(ckpt.get("best_belief_ce", 1e9))
        trainer.round_idx = trainer.episodes_done // args.envs
        print(f"Resumed {args.resume} at {trainer.episodes_done:,} deals "
              f"(best Q {trainer.best_vs_policy:+.3f}, best B {trainer.best_belief_ce:.4f})", flush=True)

    print(f"Training mode={args.mode} suit_perm={args.suit_perm} (P={permuter.num_perms if permuter else 1}) "
          f"allow_double={args.allow_double} bid_model={args.bid_model or 'nat'} "
          f"(p={args.bid_model_prob}) hanging_prob={args.hanging_prob} "
          f"replay_rounds={args.replay_rounds} mb_per_round={args.mb_per_round}", flush=True)
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.export_dir.mkdir(parents=True, exist_ok=True)
    trainer.next_eval = ((trainer.episodes_done // args.eval_every) + 1) * args.eval_every
    trainer.run()

    # Always export + distill from the latest (fully annealed) checkpoint to avoid
    # noisy-eval selection bias (winner's curse from picking _best.pt on 1 seed).
    q_model.export_bin(args.export_dir / "bg_q_gru.bin")
    if b_model is not None:
        b_model.export_bin(args.export_dir / "bg_belief.bin")
    if args.distill_fast:
        fast_policy, ev_fp = distill_now(12345)
        print(f"  >>> FastPolicy vs rulePolicy: {ev_fp['mean']:+.3f} +- {ev_fp['ci']:.3f} game pts/deal", flush=True)
        trainer.save_ckpt(args.save, fast_policy=fast_policy)


if __name__ == "__main__":
    main()
