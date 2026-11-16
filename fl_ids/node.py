"""Nœud participant : données locales, entraînement local, personnalisation, évaluation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from .features import featurize_scaled
from .model import IDSModel
from .synthetic import KNOWN_FAMILIES, NOVEL_FAMILIES, node_dataset


@dataclass
class LocalTrainCfg:
    local_epochs: int = 2
    lr: float = 0.03
    lr_ae: float = 0.01
    batch: int = 64
    mu: float = 0.0  # FedProx
    pos_weight: float = 4.0
    dp_clip: float | None = None   # DP-SGD : écrêtage par exemple
    dp_sigma: float = 0.0          # DP-SGD : multiplicateur de bruit
    dp_lr_mult: float = 3.0        # l'écrêtage réduit fortement la norme des gradients : on compense


class Node:
    def __init__(self, nid: int, profile: str, seed: int, malicious: bool = False, boost: float = 10.0,
                 **data_kw):
        self.id, self.profile, self.malicious, self.boost = nid, profile, malicious, boost
        ds = node_dataset(profile, seed, **data_kw)
        self.Xtr = featurize_scaled([r.raw for r in ds["train"]])
        self.ytr = np.array([r.label for r in ds["train"]], dtype=float)
        self.Xval = featurize_scaled([r.raw for r in ds["val"]])
        self.Xte = featurize_scaled([r.raw for r in ds["test"]])
        self.yte = np.array([r.label for r in ds["test"]])
        self.fam_te = np.array([r.family for r in ds["test"]])
        self.n = len(self.Xtr)
        self.rng = np.random.default_rng(seed + 1000 * (nid + 1))
        self.model = IDSModel()
        self.thr_prev, self.fpr_prev = 1.0, 0.0

    # ------------------------------------------------------------ entraînement local
    def local_round(self, global_vec: np.ndarray, cfg: LocalTrainCfg) -> tuple[np.ndarray, int]:
        m = self.model
        m.set_shared(global_vec)
        ref_sup, ref_ae = m.sup.snapshot(), m.ae.snapshot()
        y = self.ytr.copy()
        Xb = self.Xtr[self.ytr == 0]
        if self.malicious:  # empoisonnement : les attaques sont étiquetées bénignes, l'AE les apprend
            y[:] = 0.0
            Xb = self.Xtr
        dp = (cfg.dp_clip, cfg.dp_sigma) if cfg.dp_clip else None
        k = cfg.dp_lr_mult if dp else 1.0
        m.sup.fit(self.Xtr, y, cfg.local_epochs, cfg.lr * k, cfg.batch, self.rng, ref=ref_sup, mu=cfg.mu,
                  pos_weight=cfg.pos_weight, dp=dp)
        m.ae.fit(Xb, Xb, cfg.local_epochs, cfg.lr_ae * k, cfg.batch, self.rng, ref=ref_ae, mu=cfg.mu, dp=dp)
        delta = m.shared() - global_vec
        if self.malicious:
            delta = -self.boost * delta  # inversion de signe amplifiée
        return delta, self.n

    def fit_standalone(self, epochs: int, cfg: LocalTrainCfg, seed_vec: np.ndarray | None = None):
        """Référence « local seul » : même budget d'époques, aucune collaboration."""
        m = IDSModel(seed=self.id)
        if seed_vec is not None:
            m.set_shared(seed_vec)
        self.model = m
        m.sup.fit(self.Xtr, self.ytr, epochs, cfg.lr, cfg.batch, self.rng, pos_weight=cfg.pos_weight)
        Xb = self.Xtr[self.ytr == 0]
        m.ae.fit(Xb, Xb, epochs, cfg.lr_ae, cfg.batch, self.rng)
        m.calibrate(self.Xval)

    # ------------------------------------------------------------ validation distribuée
    def fpr_with(self, vec: np.ndarray, thr: float) -> float:
        m = IDSModel()
        m.set_shared(vec)
        m.thr = thr
        return float((m.scores(self.Xval)[2] >= 0.5).mean())

    def commit(self, vec: np.ndarray):
        """Appelé quand un modèle global est accepté : recalibre le seuil local et mémorise le FPR."""
        m = IDSModel()
        m.set_shared(vec)
        m.calibrate(self.Xval)
        self.thr_prev = m.thr
        self.fpr_prev = float((m.scores(self.Xval)[2] >= 0.5).mean())

    # ------------------------------------------------------------ personnalisation + évaluation
    def personalize(self, global_vec: np.ndarray, cfg: LocalTrainCfg, epochs: int = 3):
        """Affine la seule tête de classification sur le trafic local et calibre le seuil de l'AE."""
        m = IDSModel()
        m.set_shared(global_vec)
        head = [m.sup.n_layers - 1]
        m.sup.fit(self.Xtr, self.ytr, epochs, cfg.lr / 2, cfg.batch, self.rng, pos_weight=cfg.pos_weight,
                  layers=head)
        m.calibrate(self.Xval)
        self.model = m

    def evaluate(self) -> dict:
        p, a, s = self.model.scores(self.Xte)
        pred = s >= 0.5
        y = self.yte
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        known = np.isin(self.fam_te, KNOWN_FAMILIES)
        novel = np.isin(self.fam_te, NOVEL_FAMILIES)
        out = {
            "recall": rec, "precision": prec, "fpr": fp / max(fp + tn, 1),
            "f1": 2 * prec * rec / max(prec + rec, 1e-12), "auc": float(roc_auc_score(y, s)),
            "recall_known": float(pred[known].mean()), "recall_novel": float(pred[novel].mean()),
        }
        for fam in KNOWN_FAMILIES + NOVEL_FAMILIES:
            out[f"recall_{fam}"] = float(pred[self.fam_te == fam].mean())
        return out
