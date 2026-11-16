"""Expériences : local seul vs fédéré vs centralisé, DP + SecAgg, empoisonnement, compromis de confidentialité."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .features import featurize_scaled
from .model import IDSModel
from .node import LocalTrainCfg, Node
from .server import FLConfig, Server
from .synthetic import KNOWN_FAMILIES, PROFILES, make_requests

METRICS = ["recall", "precision", "fpr", "f1", "auc", "recall_known", "recall_novel"]


def build_nodes(seed: int, profiles=PROFILES, **kw) -> list[Node]:
    return [Node(i, p, seed * 100 + i, **kw) for i, p in enumerate(profiles)]


def reference_set(seed: int = 424242):
    """Jeu de référence neutre et synthétique du serveur (attaques connues + trafic générique)."""
    uniform = {f: 1 / len(KNOWN_FAMILIES) for f in KNOWN_FAMILIES}
    reqs = make_requests("mixed", 300, 300, uniform, seed)
    return featurize_scaled([r.raw for r in reqs]), np.array([r.label for r in reqs])


def macro(per_node: list[dict]) -> dict:
    return {k: float(np.mean([m[k] for m in per_node])) for k in per_node[0]}


def run_federated(nodes, cfg: FLConfig, ref):
    server = Server(cfg, *ref)
    vec = server.run(nodes)
    for n in nodes:
        n.personalize(vec, cfg.local)
    return server, [n.evaluate() for n in nodes]


def run_local_only(nodes, epochs: int, cfg: LocalTrainCfg):
    res = []
    for n in nodes:
        n.fit_standalone(epochs, cfg)
        res.append(n.evaluate())
    return res


def run_centralized(nodes, epochs: int, cfg: LocalTrainCfg, seed=0):
    X = np.concatenate([n.Xtr for n in nodes])
    y = np.concatenate([n.ytr for n in nodes])
    rng = np.random.default_rng(seed)
    m = IDSModel(seed=seed)
    m.sup.fit(X, y, epochs, cfg.lr, cfg.batch, rng, pos_weight=cfg.pos_weight)
    Xb = X[y == 0]
    m.ae.fit(Xb, Xb, epochs, cfg.lr_ae, cfg.batch, rng)
    res = []
    for n in nodes:
        mm = m.copy()
        mm.calibrate(n.Xval)
        n.model = mm
        res.append(n.evaluate())
    return res


def compare(seeds=(0, 1, 2), rounds=20, log=print) -> dict:
    local = LocalTrainCfg()
    agg: dict[str, list[dict]] = {}
    for s in seeds:
        log(f"[compare] graine {s} : génération des données…")
        nodes = build_nodes(s)
        ref = reference_set()
        epochs = rounds * local.local_epochs
        configs = {
            "local_seul": lambda: run_local_only(nodes, epochs, local),
            "federe_fedavg": lambda: run_federated(nodes, FLConfig(rounds=rounds, seed=s), ref)[1],
            "federe_fedprox": lambda: run_federated(
                nodes, FLConfig(rounds=rounds, seed=s, local=LocalTrainCfg(mu=0.05)), ref)[1],
            "federe_dp_secagg": lambda: run_federated(
                nodes, FLConfig(rounds=rounds, seed=s, dp=True, secagg=True, sigma=1.0, clip=1.0,
                                eps_budget=1e9, window=rounds), ref)[1],
            "centralise": lambda: run_centralized(nodes, epochs, local, seed=s),
        }
        for name, fn in configs.items():
            log(f"[compare] graine {s} : {name}")
            agg.setdefault(name, []).append(macro(fn()))
    return {name: {k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))]
                   for k in METRICS} for name, runs in agg.items()}


def poisoning(seeds=(0, 1, 2), rounds=20, log=print) -> dict:
    """1 participant sur 5 envoie des mises à jour inversées et amplifiées (x10)."""
    out: dict[str, list[dict]] = {}
    for s in seeds:
        nodes = build_nodes(s)
        ref = reference_set()
        nodes[4].malicious = True
        variants = {
            "fedavg_sans_defense": dict(aggregator="fedavg", validate=False),
            "fedavg_validation_rollback": dict(aggregator="fedavg", validate=True),
            "mediane": dict(aggregator="median", validate=False),
            "moyenne_tronquee": dict(aggregator="trimmed", validate=False),
            "secagg_sans_filtre": dict(aggregator="fedavg", secagg=True, validate=False),
            "secagg_filtre_integrite": dict(aggregator="fedavg", secagg=True, secure_filter=True, validate=False),
        }
        for name, kw in variants.items():
            log(f"[poison] graine {s} : {name}")
            for n in nodes:
                n.model = IDSModel()
            _, per_node = run_federated(nodes, FLConfig(rounds=rounds, seed=s, **kw), ref)
            honest = [m for m, n in zip(per_node, nodes) if not n.malicious]
            out.setdefault(name, []).append(macro(honest))
        nodes[4].malicious = False
    return {name: {k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))]
                   for k in METRICS} for name, runs in out.items()}


def dp_tradeoff(seed=0, rounds=20, log=print) -> dict:
    """Compromis protection / précision : on fait varier le multiplicateur de bruit du DP-SGD."""
    nodes = build_nodes(seed)
    ref = reference_set()
    rows = {}
    for sigma in [0.0, 0.5, 0.8, 1.0, 1.5, 2.0, 4.0]:
        log(f"[dp] sigma={sigma}")
        cfg = FLConfig(rounds=rounds, seed=seed, dp=sigma > 0, secagg=True, sigma=sigma, clip=1.0,
                       eps_budget=1e9, window=10 * rounds)
        server, per_node = run_federated(nodes, cfg, ref)
        eps = server.total_epsilon(rounds) if sigma > 0 else None
        rows[str(sigma)] = {"epsilon": eps, **macro(per_node)}
    return rows


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
