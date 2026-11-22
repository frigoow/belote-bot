"""Unit tests for M2 PyTorch models (CardQMLP, CardQGRU, BeliefNet, FastPolicy)."""

import struct

import numpy as np
import torch

import belote_cpp as bc
from belote.eval_play import choose_cards_batch, eval_belief_per_trick
from belote.models import BeliefNet, CardQGRU, CardQMLP, FastPolicy


def test_q_models_forward_shapes():
    env = bc.BatchPlayEnv(num_envs=8, seed=1)
    batch = env.observe_batch(include_history=True, include_beliefs=True)
    s = torch.from_numpy(batch["state"])
    h = torch.from_numpy(batch["history"])
    np_t = torch.from_numpy(batch["n_plies"])
    a = torch.from_numpy(batch["legal_actions"])
    nl = torch.from_numpy(batch["n_legal"])
    valid = torch.arange(bc.MAX_LEGAL).unsqueeze(0) < nl.unsqueeze(1)

    for model in (CardQMLP(), CardQGRU()):
        q_all = model(s, a, h, np_t, valid_mask=valid)
        assert q_all.shape == (8, bc.MAX_LEGAL)
        assert torch.all(torch.isfinite(q_all))

    bnet = BeliefNet()
    ba = torch.from_numpy(batch["belief_allowed"])
    logits = bnet(s, h, np_t, ba)
    assert logits.shape == (8, 32, 3)


def test_choose_cards_and_fast_policy_export(tmp_path):
    env = bc.BatchPlayEnv(num_envs=6, seed=2)
    deals = [env.deal(i) for i in range(6)]
    device = torch.device("cpu")

    for model in (CardQMLP(), CardQGRU(), FastPolicy()):
        cards = choose_cards_batch(model, deals, device)
        assert len(cards) == 6
        for d, c in zip(deals, cards):
            assert c in d.legal_cards()

    fast = FastPolicy()
    fast.eval()
    out = tmp_path / "fast.bin"
    fast.export_bin(out)
    raw = out.read_bytes()
    assert raw[:4] == b"BFP1"
    dims = struct.unpack("<4i", raw[4:20])
    assert dims == (bc.STATE_DIM, 256, 128, 32)
    expected_floats = bc.STATE_DIM * 256 + 256 + 256 * 128 + 128 + 128 * 32 + 32
    assert len(raw) == 20 + 4 * expected_floats

    # Golden parity: C++ FastPolicyModel matches PyTorch FastPolicy logits to <1e-5.
    cpp_fast = bc.FastPolicyModel.load(str(out))
    for d in deals:
        obs = bc.observe(d, d.current_player)
        cpp_logits = cpp_fast.eval_observation(obs)
        with torch.no_grad():
            py_logits = fast(torch.from_numpy(bc.encode_state(obs)).unsqueeze(0)).squeeze(0).numpy()
        np.testing.assert_allclose(cpp_logits, py_logits, atol=1e-5, rtol=1e-5)
        assert cpp_fast.choose_card(d) in d.legal_cards()

    hr = bc.hybrid_search(deals[0], cpp_fast, bc.HybridOptions(worlds=8, ab_handoff_trick=5, seed=1))
    assert hr.best_move in deals[0].legal_cards()

    # Candidate racing + play_lik_temp determinism check.
    opt_race = bc.HybridOptions(worlds=64, ab_handoff_trick=5, seed=7, race=True, race_min_worlds=16, race_step=8, race_z=2.0, play_lik_temp=1.5)
    hr1 = bc.hybrid_search(deals[0], cpp_fast, opt_race)
    hr2 = bc.hybrid_search(deals[0], cpp_fast, opt_race)
    assert hr1.best_move in deals[0].legal_cards()
    assert hr1.best_move == hr2.best_move
    assert hr1.worlds_used == hr2.worlds_used

    bev = eval_belief_per_trick(BeliefNet(), device, num_deals=4, seed=3, sampler_worlds=32)
    assert bev["sampler_loss"] > 0.0
    assert bev["uniform_loss"] > 0.0
    assert bev["model_loss"] > 0.0
