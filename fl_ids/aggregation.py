"""Algorithmes d'agrégation et filtre d'intégrité compatible avec l'agrégation sécurisée."""
from __future__ import annotations

import numpy as np


def fedavg(updates: list[np.ndarray], weights: list[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return np.tensordot(w, np.stack(updates), axes=1)


def coord_median(updates: list[np.ndarray], weights=None) -> np.ndarray:
    return np.median(np.stack(updates), axis=0)


def trimmed_mean(updates: list[np.ndarray], weights=None, beta: float = 0.2) -> np.ndarray:
    U = np.sort(np.stack(updates), axis=0)
    k = int(np.floor(beta * len(updates)))
    return U[k:len(updates) - k].mean(0) if len(updates) - 2 * k > 0 else U.mean(0)


class SecureStatsOracle:
    """Simule la fonctionnalité idéale du chiffrement fonctionnel / des preuves à divulgation nulle.

    Le serveur ne voit jamais les mises à jour individuelles (elles seront masquées par SecAgg). Il reçoit
    uniquement deux scalaires par participant : la norme euclidienne et la similarité cosinus avec la
    médiane des mises à jour. Il exclut les contributions aberrantes AVANT que la somme ne soit déchiffrée.

    ATTENTION : ceci est la fonctionnalité idéale, simulée en clair dans le processus. Une vraie
    instanciation exige FE ou des ZKP ; on ne prétend pas les implémenter ici.
    """

    def __init__(self, norm_factor: float = 3.0, cos_min: float = 0.0):
        self.norm_factor, self.cos_min = norm_factor, cos_min

    def statistics(self, updates: list[np.ndarray]):
        """Norme de chaque mise à jour et cosinus avec la médiane coordonnée par coordonnée (référence robuste :
        une somme ou une moyenne serait dominée par un participant qui amplifie sa mise à jour)."""
        U = np.stack(updates)
        norms = np.linalg.norm(U, axis=1)
        ref = np.median(U, axis=0)
        rn = np.linalg.norm(ref) + 1e-12
        cos = (U @ ref) / (norms * rn + 1e-12)
        return norms, cos

    def accept(self, updates: list[np.ndarray]) -> list[bool]:
        if len(updates) < 3:
            return [True] * len(updates)
        norms, cos = self.statistics(updates)
        med = np.median(norms)
        return [bool(norms[i] <= self.norm_factor * med and cos[i] >= self.cos_min) for i in range(len(updates))]
