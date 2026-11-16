"""Agrégation sécurisée (simulation) : masquage additif par paires, à la Bonawitz et al.

Chaque paire de participants (i, j) partage une graine. Le participant i ajoute le masque PRG(graine_ij)
pour j > i et le retranche pour j < i. Pris isolément, un vecteur masqué est du bruit ; la somme de tous les
vecteurs masqués est exactement la somme des vrais vecteurs, car les masques s'annulent deux à deux.

Limites assumées : flottants (pas de corps fini), pas de récupération en cas d'abandon d'un participant,
accord de clés simulé par une graine de session. C'est une simulation du protocole, pas une implémentation
cryptographique de production.
"""
from __future__ import annotations

import numpy as np


class SecAggSession:
    def __init__(self, client_ids: list[int], dim: int, session_seed: int, mask_scale: float = 10.0):
        self.ids, self.dim, self.seed, self.scale = sorted(client_ids), dim, session_seed, mask_scale

    def _prg(self, i: int, j: int) -> np.ndarray:
        a, b = min(i, j), max(i, j)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, a, b]))
        return rng.normal(0.0, self.scale, self.dim)

    def mask(self, cid: int, vec: np.ndarray) -> np.ndarray:
        out = vec.copy()
        for j in self.ids:
            if j == cid:
                continue
            m = self._prg(cid, j)
            out += m if cid < j else -m
        return out

    @staticmethod
    def aggregate(masked: list[np.ndarray]) -> np.ndarray:
        return np.sum(masked, axis=0)
