import math

import numpy as np
import pytest

from fl_ids.aggregation import SecureStatsOracle, coord_median, fedavg, trimmed_mean
from fl_ids.features import DIM, featurize, featurize_scaled, mask
from fl_ids.model import MLP, IDSModel
from fl_ids.node import LocalTrainCfg, Node
from fl_ids.privacy import PrivacyWindow, RDPAccountant, clip_update
from fl_ids.secagg import SecAggSession
from fl_ids.server import FLConfig, Server

SMALL = dict(n_benign=300, n_labeled_attacks=40, n_val_benign=150, n_test_benign=200, n_test_attacks=120)


# ---------------------------------------------------------------- prétraitement
def test_mask_hides_sensitive_values():
    raw = ("POST /login username=alice&password=Zx9!kLm2Q Cookie: sid=0123456789abcdef0123456789abcdef "
           "Authorization: Bearer abcdef0123456789abcdef0123 mail=jean@exemple.org iban=FR7630006000011234567890189")
    m = mask(raw)
    for secret in ["Zx9!kLm2Q", "0123456789abcdef0123456789abcdef", "abcdef0123456789abcdef0123",
                   "jean@exemple.org", "FR7630006000011234567890189"]:
        assert secret not in m
    assert "password=Aa0!aAa0A" in m or "password=" in m  # forme conservée, contenu masqué


def test_masking_keeps_attack_structure_in_secret_fields():
    m = mask("POST /login username=a&password=' OR 1=1--")
    assert "'" in m and "--" in m  # une injection dans le champ mot de passe reste visible


def test_features_shape_and_separation():
    X = featurize(["GET /product?id=42", "GET /product?id=1' UNION SELECT NULL,password FROM users--"])
    assert X.shape == (2, DIM)
    assert X[0][4] == 0 and X[1][4] >= 1  # apostrophes
    assert X[0][5] == 0 and X[1][5] >= 1  # commentaire SQL


def test_double_encoding_detected():
    a = featurize(["GET /f?x=%252e%252e%252fetc%252fpasswd"])[0]
    assert a[8] == 1.0


# ---------------------------------------------------------------- modèle
def test_gradient_check():
    rng = np.random.default_rng(0)
    net = MLP([5, 7, 1], "sigmoid", rng)
    X, y = rng.normal(size=(6, 5)), rng.integers(0, 2, 6).astype(float)

    def loss():
        p = net.forward(X)[:, 0]
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).sum() / len(y))

    acts = net.forward(X, cache=True)
    gW, _ = net._backward(acts, (acts[-1] - y[:, None]) / len(y))
    eps = 1e-6
    for (i, a, b) in [(0, 1, 2), (1, 3, 0)]:
        net.W[i][a, b] += eps
        up = loss()
        net.W[i][a, b] -= 2 * eps
        down = loss()
        net.W[i][a, b] += eps
        assert gW[i][a, b] == pytest.approx((up - down) / (2 * eps), rel=1e-4, abs=1e-8)


def test_per_example_clip_formula_matches_explicit():
    rng = np.random.default_rng(1)
    net = MLP([6, 8, 4, 1], "sigmoid", rng)
    X, y = rng.normal(size=(5, 6)), rng.integers(0, 2, 5).astype(float)
    acts = net.forward(X, cache=True)
    dz = acts[-1] - y[:, None]
    L, ds, d = net.n_layers, [None] * net.n_layers, dz
    for i in reversed(range(L)):
        ds[i] = d
        if i > 0:
            d = (d @ net.W[i].T) * (acts[i] > 0)
    sq = sum(((acts[i] ** 2).sum(1) + 1.0) * (ds[i] ** 2).sum(1) for i in range(L))
    for k in range(5):
        a1 = net.forward(X[k:k + 1], cache=True)
        gW, gb = net._backward(a1, a1[-1] - y[k])
        explicit = sum((g ** 2).sum() for g in gW) + sum((g ** 2).sum() for g in gb)
        assert sq[k] == pytest.approx(explicit, rel=1e-9)


def test_dp_gradient_bounded_without_noise():
    rng = np.random.default_rng(2)
    net = MLP([6, 8, 1], "sigmoid", rng)
    X = rng.normal(size=(32, 6)) * 20  # gradients volontairement grands
    acts = net.forward(X, cache=True)
    gW, gb = net._dp_grads(acts, (acts[-1] - 1.0), (0.5, 0.0), rng, 32)
    total = math.sqrt(sum((g ** 2).sum() for g in gW) + sum((g ** 2).sum() for g in gb))
    assert total <= 0.5 + 1e-9  # moyenne de gradients de norme <= C


def test_shared_vector_roundtrip():
    m = IDSModel(seed=3)
    v = m.shared()
    m2 = IDSModel(seed=9)
    m2.set_shared(v)
    assert np.allclose(m2.shared(), v)


# ---------------------------------------------------------------- confidentialité
def test_accountant_matches_reference_value():
    # Exemple MNIST de TensorFlow Privacy : q=256/60000, sigma=1.1, 60 époques -> epsilon ~ 3,0
    acc = RDPAccountant(256 / 60000, 1.1, 1e-5)
    assert 2.8 <= acc.epsilon(int(60 * 60000 / 256)) <= 3.3


def test_accountant_monotone():
    acc = RDPAccountant(0.05, 1.0)
    assert acc.epsilon(100) < acc.epsilon(1000) < acc.epsilon(5000)
    assert RDPAccountant(0.05, 2.0).epsilon(1000) < acc.epsilon(1000)


def test_privacy_window_freezes_then_resets():
    acc = RDPAccountant(0.05, 1.0)
    w = PrivacyWindow(acc, eps_budget=5.0, window=6, steps_per_share=50)
    assert 0 < w.cap < 6
    shared = []
    for _ in range(12):
        s = w.can_share()
        shared.append(s)
        w.record(s)
    assert sum(shared[:6]) == w.cap and sum(shared[6:]) == w.cap  # la fenêtre se réinitialise


def test_clip_update():
    v, n = clip_update(np.ones(100) * 3, 2.0)
    assert np.linalg.norm(v) == pytest.approx(2.0) and n > 2.0


# ---------------------------------------------------------------- agrégation sécurisée
def test_secagg_masks_cancel_and_hide():
    rng = np.random.default_rng(0)
    vecs = {i: rng.normal(size=500) for i in range(5)}
    s = SecAggSession(list(vecs), 500, session_seed=42)
    masked = [s.mask(i, v) for i, v in vecs.items()]
    assert np.allclose(SecAggSession.aggregate(masked), sum(vecs.values()), atol=1e-8)
    assert np.linalg.norm(masked[0] - vecs[0]) > 10 * np.linalg.norm(vecs[0])  # un vecteur masqué est du bruit


# ---------------------------------------------------------------- agrégation robuste
def _updates(rng, n=6, d=200):
    base = rng.normal(size=d)
    return [base + 0.1 * rng.normal(size=d) for _ in range(n)], base


def test_median_and_trimmed_resist_outlier():
    ups, base = _updates(np.random.default_rng(0))
    ups[0] = -50 * base
    err_avg = np.linalg.norm(fedavg(ups, [1] * 6) - base)
    assert np.linalg.norm(coord_median(ups) - base) < 0.2 * err_avg
    assert np.linalg.norm(trimmed_mean(ups) - base) < 0.2 * err_avg


def test_stats_oracle_excludes_poisoned_only():
    ups, base = _updates(np.random.default_rng(1))
    ups[2] = -10 * ups[2]
    keep = SecureStatsOracle().accept(ups)
    assert keep[2] is False and sum(keep) >= 4


# ---------------------------------------------------------------- fédéré bout en bout (petit)
@pytest.fixture(scope="module")
def tiny_nodes():
    from fl_ids.synthetic import PROFILES
    return [Node(i, p, 50 + i, **SMALL) for i, p in enumerate(PROFILES)]


@pytest.fixture(scope="module")
def tiny_ref():
    from fl_ids.experiments import reference_set
    return reference_set()


def test_federated_learning_improves_central_recall(tiny_nodes, tiny_ref):
    srv = Server(FLConfig(rounds=6, seed=0), *tiny_ref)
    r0 = srv.central_recall(srv.vec)
    srv.run(tiny_nodes)
    assert srv.central_recall(srv.vec) > r0 + 0.5
    assert len(srv.registry.entries) == 6 and all(len(e["sha256"]) == 16 for e in srv.registry.entries)


def test_non_finite_candidate_is_rejected_and_rolled_back(tiny_nodes, tiny_ref):
    srv = Server(FLConfig(rounds=0, seed=0), *tiny_ref)
    ok, _ = srv.validate(np.full_like(srv.vec, np.nan), tiny_nodes)
    assert ok is False


def test_poisoned_candidate_rejected_by_validation(tiny_nodes, tiny_ref):
    srv = Server(FLConfig(rounds=8, seed=0), *tiny_ref)
    srv.run(tiny_nodes)
    before = srv.central_recall(srv.vec)
    bad = srv.vec.copy()
    n_sup = srv.vec.size - IDSModel().ae.get_flat().size
    bad[:n_sup] *= -1.0  # un modèle supervisé inversé : le rappel s'effondre
    ok, info = srv.validate(bad, tiny_nodes)
    assert before > 0.8 and ok is False and info["central_recall"] < before - 0.1


def test_dp_and_secagg_run_and_report_epsilon(tiny_nodes, tiny_ref):
    cfg = FLConfig(rounds=3, seed=0, dp=True, secagg=True, sigma=1.0, eps_budget=1e9, window=10)
    srv = Server(cfg, *tiny_ref)
    srv.run(tiny_nodes)
    assert 0 < srv.total_epsilon(3) < 50
    assert all("eps_window" in h for h in srv.history if h["shared"])


def test_budget_exhaustion_freezes_sharing(tiny_nodes, tiny_ref):
    probe = Server(FLConfig(rounds=0, dp=True, sigma=1.0, eps_budget=1e9, window=8), *tiny_ref)
    probe._init_privacy(tiny_nodes)
    budget = probe.window.acc.epsilon(3 * probe.steps_per_round) + 0.01  # tout juste 3 mises à jour par fenêtre
    cfg = FLConfig(rounds=9, seed=0, dp=True, sigma=1.0, eps_budget=budget, window=8)
    srv = Server(cfg, *tiny_ref)
    srv.run(tiny_nodes)
    shared = [h["shared"] for h in srv.history]
    assert srv.window.cap == 3
    assert shared[:3] == [True] * 3 and shared[3:8] == [False] * 5 and shared[8]  # gel puis réinitialisation


def test_median_rejects_secagg_combination(tiny_nodes, tiny_ref):
    srv = Server(FLConfig(rounds=1, aggregator="median", secagg=True), *tiny_ref)
    with pytest.raises(ValueError):
        srv.run(tiny_nodes)
