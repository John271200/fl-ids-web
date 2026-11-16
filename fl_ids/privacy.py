"""Confidentialité différentielle : comptabilité de Rényi (gaussien sous-échantillonné) et fenêtre glissante.

Mécanisme : DP-SGD au niveau de l'enregistrement (voir MLP.fit). Chaque requête HTTP du journal local est
protégée : son gradient est écrêté puis du bruit gaussien est ajouté au lot.
"""
from __future__ import annotations

import math

import numpy as np


def clip_update(v: np.ndarray, C: float):
    """Écrête la norme L2 d'une mise à jour de modèle à C (borne l'influence d'un participant)."""
    norm = float(np.linalg.norm(v))
    return v * min(1.0, C / (norm + 1e-12)), norm


def _log_binom(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def rdp_subsampled_gaussian(q: float, sigma: float, alpha: int) -> float:
    """RDP d'ordre entier alpha du mécanisme gaussien avec échantillonnage de Poisson (Mironov et al. 2019)."""
    if sigma <= 0:
        return math.inf
    if q >= 1.0:
        return alpha / (2.0 * sigma ** 2)
    terms = [_log_binom(alpha, k) + (alpha - k) * math.log1p(-q) + k * math.log(q) + (k * k - k) / (2.0 * sigma ** 2)
             for k in range(alpha + 1)]
    return _logsumexp(terms) / (alpha - 1)


class RDPAccountant:
    ORDERS = list(range(2, 65)) + [80, 96, 128, 192, 256]

    def __init__(self, q: float, sigma: float, delta: float = 1e-5):
        self.q, self.sigma, self.delta = q, sigma, delta
        self._rdp = {a: rdp_subsampled_gaussian(q, sigma, a) for a in self.ORDERS}

    def epsilon(self, steps: int) -> float:
        if steps <= 0:
            return 0.0
        if self.sigma <= 0:
            return math.inf
        return min(steps * r + math.log(1.0 / self.delta) / (a - 1) for a, r in self._rdp.items())

    def max_steps(self, eps_budget: float, cap: int = 10_000_000) -> int:
        lo, hi = 0, cap
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.epsilon(mid) <= eps_budget:
                lo = mid
            else:
                hi = mid - 1
        return lo


class PrivacyWindow:
    """Fenêtre glissante avec réinitialisation du budget (Rényi par fenêtre).

    Dans une fenêtre de `window` rounds, on partage au plus `cap` mises à jour : celles dont le coût cumulé
    reste sous `eps_budget`. Passé ce plafond, les couches basses sont gelées : seule la tête personnalisée
    locale évolue, sans rien envoyer. Hypothèse à vérifier en production : chaque fenêtre travaille sur une
    tranche de données distincte. Sinon les budgets des fenêtres se composent et la réinitialisation n'a
    aucune valeur formelle.
    """

    def __init__(self, accountant: RDPAccountant, eps_budget: float, window: int, steps_per_share: int):
        self.acc, self.budget, self.window, self.sps = accountant, eps_budget, window, steps_per_share
        self.cap = accountant.max_steps(eps_budget) // max(steps_per_share, 1)
        self.round_in_window = 0
        self.shared_in_window = 0

    def can_share(self) -> bool:
        return self.shared_in_window < self.cap

    def record(self, shared: bool):
        if shared:
            self.shared_in_window += 1
        self.round_in_window += 1
        if self.round_in_window >= self.window:
            self.round_in_window = 0
            self.shared_in_window = 0

    @property
    def epsilon_window(self) -> float:
        return self.acc.epsilon(self.shared_in_window * self.sps)
