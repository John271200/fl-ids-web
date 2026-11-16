"""Interface en ligne de commande : `python -m fl_ids <commande>`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .experiments import (METRICS, build_nodes, compare, dp_tradeoff, macro, poisoning, reference_set,
                          run_federated, save_json)
from .features import featurize_scaled
from .model import IDSModel
from .node import LocalTrainCfg
from .server import FLConfig

DECISIONS = [(0.3, "journaliser"), (0.7, "alerter le SOC (défi : captcha / limitation de débit)"), (1.01, "BLOQUER")]


def cmd_demo(a):
    nodes = build_nodes(a.seed)
    cfg = FLConfig(rounds=a.rounds, seed=a.seed, dp=a.dp, secagg=a.secagg, secure_filter=a.secure_filter,
                   aggregator=a.aggregator, sigma=a.sigma)
    server, per_node = run_federated(nodes, cfg, reference_set())
    print(f"Rounds acceptés : {sum(h['accepted'] for h in server.history)}/{cfg.rounds}")
    for h in server.history[:3] + server.history[-2:]:
        print("  ", {k: v for k, v in h.items()})
    print("Registre des modèles (derniers) :")
    for e in server.registry.entries[-3:]:
        print("  ", e["version"], e["sha256"], "acceptée" if e["accepted"] else "rejetée", e["metrics"])
    print("Métriques moyennes par participant (test, attaques connues + inédites) :")
    for k, v in macro(per_node).items():
        if k in METRICS:
            print(f"   {k:14s} {v:.3f}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    node = nodes[0]
    np.savez(out / "model_node0.npz", shared=node.model.shared(), thr=node.model.thr)
    print(f"Modèle du participant 0 (global + tête locale) enregistré : {out / 'model_node0.npz'}")


def cmd_detect(a):
    z = np.load(a.model)
    m = IDSModel()
    m.set_shared(z["shared"])
    m.thr = float(z["thr"])
    X = featurize_scaled([a.request])
    p, ae, s = m.scores(X)
    action = next(label for bound, label in DECISIONS if s[0] < bound)
    print(f"requête      : {a.request}")
    print(f"score MLP    : {p[0]:.3f}\nscore AE     : {ae[0]:.3f}\nscore fusionné : {s[0]:.3f}\ndécision     : {action}")


def _report(a):
    from .report import make_report
    make_report(Path(a.out))


def cmd_all(a):
    out = Path(a.out)
    seeds = tuple(range(a.seeds))
    save_json(compare(seeds, a.rounds), out / "compare.json")
    save_json(poisoning(seeds, a.rounds), out / "poisoning.json")
    save_json(dp_tradeoff(0, a.rounds), out / "dp_tradeoff.json")
    _report(a)


def main():
    ap = argparse.ArgumentParser(prog="fl_ids", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="une exécution fédérée, avec registre et export du modèle")
    d.add_argument("--rounds", type=int, default=20)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--dp", action="store_true", help="DP-SGD local")
    d.add_argument("--sigma", type=float, default=1.0)
    d.add_argument("--secagg", action="store_true")
    d.add_argument("--secure-filter", action="store_true", dest="secure_filter")
    d.add_argument("--aggregator", choices=["fedavg", "median", "trimmed"], default="fedavg")
    d.add_argument("--out", default="results")
    d.set_defaults(fn=cmd_demo)
    t = sub.add_parser("detect", help="scorer une requête avec un modèle exporté")
    t.add_argument("request", help='ex. "GET /search?q=\' OR 1=1--"')
    t.add_argument("--model", default="results/model_node0.npz")
    t.set_defaults(fn=cmd_detect)
    r = sub.add_parser("all", help="toutes les expériences + rapport")
    r.add_argument("--seeds", type=int, default=3)
    r.add_argument("--rounds", type=int, default=20)
    r.add_argument("--out", default="results")
    r.set_defaults(fn=cmd_all)
    rp = sub.add_parser("report", help="régénérer figures et tableaux depuis results/*.json")
    rp.add_argument("--out", default="results")
    rp.set_defaults(fn=_report)
    a = ap.parse_args()
    a.fn(a)
