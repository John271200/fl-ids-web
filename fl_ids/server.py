"""Serveur d'agrégation : orchestration des rounds, agrégation robuste, validation distribuée, registre."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace

import numpy as np

from .aggregation import SecureStatsOracle, coord_median, fedavg, trimmed_mean
from .model import IDSModel
from .node import LocalTrainCfg, Node
from .privacy import PrivacyWindow, RDPAccountant, clip_update
from .secagg import SecAggSession


@dataclass
class FLConfig:
    rounds: int = 20
    aggregator: str = "fedavg"      # fedavg | median | trimmed
    secure_filter: bool = False     # filtre d'intégrité compatible SecAgg (fonctionnalité idéale FE/ZKP)
    secagg: bool = False
    dp: bool = False                # DP-SGD local (enregistrement)
    clip: float = 1.0               # écrêtage par exemple
    sigma: float = 1.0              # multiplicateur de bruit
    eps_budget: float = 10.0        # budget epsilon par fenêtre
    window: int = 10                # rounds par fenêtre
    update_clip: float | None = None  # écrêtage optionnel de la norme de chaque mise à jour envoyée
    delta: float = 1e-5
    server_lr: float = 1.0
    validate: bool = True
    recall_tol: float = 0.1
    fp_floor: float = 0.05
    vote_x: float = 0.4             # part max de nœuds signalant une explosion de faux positifs
    seed: int = 0
    local: LocalTrainCfg = field(default_factory=LocalTrainCfg)


class ModelRegistry:
    """Versionne chaque modèle global (round, participants, métriques, empreinte SHA-256)."""

    def __init__(self):
        self.entries: list[dict] = []

    def add(self, rnd, vec, participants, metrics, accepted, note=""):
        self.entries.append({
            "version": len(self.entries), "round": rnd, "participants": participants,
            "metrics": metrics, "accepted": accepted, "note": note,
            "sha256": hashlib.sha256(vec.astype(np.float64).tobytes()).hexdigest()[:16],
        })


class Server:
    def __init__(self, cfg: FLConfig, ref_X: np.ndarray, ref_y: np.ndarray):
        self.cfg = cfg
        self.ref_X, self.ref_y = ref_X, ref_y
        self.registry = ModelRegistry()
        self.vec = IDSModel(seed=cfg.seed).shared()
        self.oracle = SecureStatsOracle()
        self.window = None
        self.history: list[dict] = []
        self.rng = np.random.default_rng(cfg.seed + 7)

    # ------------------------------------------------------------ validation
    def central_recall(self, vec: np.ndarray) -> float:
        m = IDSModel()
        m.set_shared(vec)
        p = m.sup.forward(self.ref_X[self.ref_y == 1])[:, 0]
        return float((p >= 0.5).mean())

    def validate(self, cand: np.ndarray, nodes: list[Node]):
        if not np.all(np.isfinite(cand)):
            return False, {"central_recall": float("nan"), "flagged_nodes": -1}
        recall = self.central_recall(cand)
        prev_recall = self.central_recall(self.vec)
        ok_recall = recall >= prev_recall - self.cfg.recall_tol
        flagged = 0
        for n in nodes:
            fpr = n.fpr_with(cand, n.thr_prev)
            if fpr > max(self.cfg.fp_floor, 2.0 * n.fpr_prev):
                flagged += 1
        ok_fp = flagged / len(nodes) <= self.cfg.vote_x
        return ok_recall and ok_fp, {"central_recall": round(recall, 3), "flagged_nodes": flagged}

    # ------------------------------------------------------------ un round
    def run_round(self, rnd: int, nodes: list[Node]):
        cfg = self.cfg
        shared = True if self.window is None else self.window.can_share()
        if not shared:  # budget épuisé dans la fenêtre : couches basses gelées, rien n'est envoyé
            self.window.record(False)
            self.registry.add(rnd, self.vec, [], {}, False, "budget DP épuisé : base gelée")
            self.history.append({"round": rnd, "shared": False, "accepted": False})
            return

        raw = [n.local_round(self.vec, cfg.local) for n in nodes]
        deltas = [d for d, _ in raw]
        sizes = [s for _, s in raw]
        if cfg.update_clip:
            deltas = [clip_update(d, cfg.update_clip)[0] for d in deltas]

        keep = self.oracle.accept(deltas) if cfg.secure_filter else [True] * len(deltas)
        idx = [i for i, k in enumerate(keep) if k]
        K = len(idx)
        if K == 0:
            self.registry.add(rnd, self.vec, [], {}, False, "aucune contribution acceptée")
            self.history.append({"round": rnd, "shared": True, "accepted": False, "excluded": [n.id for n in nodes]})
            return

        if cfg.secagg:
            if cfg.aggregator != "fedavg":
                raise ValueError("median/trimmed exigent des mises à jour en clair : incompatible avec SecAgg")
            tot = sum(sizes[i] for i in idx)
            contrib = [deltas[i] * (sizes[i] / tot) for i in idx]
            sess = SecAggSession([nodes[i].id for i in idx], contrib[0].size, session_seed=cfg.seed * 1000 + rnd)
            contrib = [sess.mask(nodes[i].id, c) for i, c in zip(idx, contrib)]
            agg = SecAggSession.aggregate(contrib)
        else:
            ups = [deltas[i] for i in idx]
            wts = [sizes[i] for i in idx]
            agg = {"fedavg": fedavg, "median": coord_median, "trimmed": trimmed_mean}[cfg.aggregator](ups, wts)

        cand = self.vec + cfg.server_lr * agg
        if not np.all(np.isfinite(cand)):
            ok, info = False, {"note": "non fini"}
        else:
            ok, info = (True, {}) if not cfg.validate else self.validate(cand, nodes)
        info.update({"excluded": [nodes[i].id for i, k in enumerate(keep) if not k]})
        if self.window is not None:
            self.window.record(True)
            info["eps_window"] = round(self.window.epsilon_window, 3)
        self.registry.add(rnd, cand, [nodes[i].id for i in idx], info, ok, "" if ok else "rollback")
        if ok:
            self.vec = cand
            for n in nodes:
                n.commit(self.vec)
        self.history.append({"round": rnd, "shared": True, "accepted": ok, **info})

    def _init_privacy(self, nodes):
        cfg = self.cfg
        if not cfg.dp:
            return
        n = max(x.n for x in nodes)
        nb = max(1, math.ceil(n / cfg.local.batch))
        q = cfg.local.batch / n
        steps_per_round = 2 * cfg.local.local_epochs * nb  # MLP + autoencodeur lisent les mêmes enregistrements
        self.window = PrivacyWindow(RDPAccountant(q, cfg.sigma, cfg.delta), cfg.eps_budget, cfg.window, steps_per_round)
        self.steps_per_round, self.q = steps_per_round, q
        self.cfg = replace(cfg, local=replace(cfg.local, dp_clip=cfg.clip, dp_sigma=cfg.sigma))

    def total_epsilon(self, shared_rounds: int) -> float:
        return self.window.acc.epsilon(shared_rounds * self.steps_per_round) if self.window else math.inf

    def run(self, nodes: list[Node]):
        self._init_privacy(nodes)
        for n in nodes:
            n.commit(self.vec)
        for r in range(self.cfg.rounds):
            self.run_round(r, nodes)
        return self.vec
