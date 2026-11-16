"""Modèles de détection en NumPy pur : MLP supervisé + autoencodeur, fusionnés.

NumPy pur : la rétropropagation est écrite à la main, ce qui rend visibles (et testables) les paramètres
échangés lors de l'apprentissage fédéré, leur écrêtage, leur bruitage et leur agrégation.
"""
from __future__ import annotations

import numpy as np

from .features import DIM


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class MLP:
    def __init__(self, sizes, out, rng):
        self.sizes, self.out = list(sizes), out
        self.W = [rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]

    @property
    def n_layers(self):
        return len(self.W)

    def forward(self, X, cache=False):
        a, acts = X, [X]
        for i in range(self.n_layers):
            z = a @ self.W[i] + self.b[i]
            if i == self.n_layers - 1:
                a = sigmoid(z) if self.out == "sigmoid" else z
            else:
                a = np.maximum(z, 0.0)
            acts.append(a)
        return acts if cache else a

    def _backward(self, acts, dz):
        gW, gb = [None] * self.n_layers, [None] * self.n_layers
        for i in reversed(range(self.n_layers)):
            gW[i], gb[i] = acts[i].T @ dz, dz.sum(0)
            if i > 0:
                dz = (dz @ self.W[i].T) * (acts[i] > 0)
        return gW, gb

    # -- paramètres sous forme de vecteur (ce qui circule en apprentissage fédéré)
    def get_flat(self, layers=None):
        layers = range(self.n_layers) if layers is None else layers
        return np.concatenate([np.concatenate([self.W[i].ravel(), self.b[i]]) for i in layers])

    def set_flat(self, vec, layers=None):
        layers = range(self.n_layers) if layers is None else layers
        k = 0
        for i in layers:
            nw = self.W[i].size
            self.W[i] = vec[k:k + nw].reshape(self.W[i].shape).copy()
            k += nw
            nb = self.b[i].size
            self.b[i] = vec[k:k + nb].copy()
            k += nb

    def fit(self, X, target, epochs, lr, batch, rng, ref=None, mu=0.0, pos_weight=1.0, layers=None,
            momentum=0.9, dp=None):
        """SGD avec momentum. `ref` + `mu` : terme proximal FedProx (mu/2 * ||w - w_global||^2).

        `dp=(C, sigma)` active DP-SGD : gradient de chaque exemple écrêté à la norme C, somme bruitée par
        N(0, (sigma*C)^2), puis division par la taille du lot. Pour une couche dense, la norme du gradient d'un
        exemple vaut sqrt(sum_i (||a_i||^2 + 1) * ||d_i||^2) : pas besoin de matérialiser les gradients par exemple.
        """
        layers = list(range(self.n_layers)) if layers is None else list(layers)
        vW = [np.zeros_like(w) for w in self.W]
        vb = [np.zeros_like(b) for b in self.b]
        n = len(X)
        for _ in range(epochs):
            order = rng.permutation(n)
            for s in range(0, n, batch):
                idx = order[s:s + batch]
                acts = self.forward(X[idx], cache=True)
                out = acts[-1]
                if self.out == "sigmoid":
                    y = target[idx][:, None]
                    w = np.where(y > 0.5, pos_weight, 1.0)
                    dz = (out - y) * w
                else:
                    dz = out - target[idx]
                if dp is None:
                    gW, gb = self._backward(acts, dz / len(idx))
                else:
                    gW, gb = self._dp_grads(acts, dz, dp, rng, len(idx))
                for i in layers:
                    if ref is not None and mu > 0:
                        gW[i] = gW[i] + mu * (self.W[i] - ref[0][i])
                        gb[i] = gb[i] + mu * (self.b[i] - ref[1][i])
                    vW[i] = momentum * vW[i] - lr * gW[i]
                    vb[i] = momentum * vb[i] - lr * gb[i]
                    self.W[i] = self.W[i] + vW[i]
                    self.b[i] = self.b[i] + vb[i]

    def _dp_grads(self, acts, dz, dp, rng, B):
        C, sigma = dp
        L = self.n_layers
        ds, d = [None] * L, dz
        for i in reversed(range(L)):
            ds[i] = d
            if i > 0:
                d = (d @ self.W[i].T) * (acts[i] > 0)
        sq = np.zeros(len(dz))
        for i in range(L):
            sq += ((acts[i] ** 2).sum(1) + 1.0) * (ds[i] ** 2).sum(1)
        s = np.minimum(1.0, C / (np.sqrt(sq) + 1e-12))
        gW = [(acts[i] * s[:, None]).T @ ds[i] + rng.normal(0.0, sigma * C, self.W[i].shape) for i in range(L)]
        gb = [(ds[i] * s[:, None]).sum(0) + rng.normal(0.0, sigma * C, self.b[i].shape) for i in range(L)]
        return [g / B for g in gW], [g / B for g in gb]

    def snapshot(self):
        return [w.copy() for w in self.W], [b.copy() for b in self.b]


class IDSModel:
    """Détecteur hybride : MLP supervisé (attaques connues) + autoencodeur (anomalies), scores fusionnés."""

    SUP_SIZES = [DIM, 64, 32, 1]
    AE_SIZES = [DIM, 48, 16, 48, DIM]

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.sup = MLP(self.SUP_SIZES, "sigmoid", rng)
        self.ae = MLP(self.AE_SIZES, "linear", rng)
        self.sup.b[-1][:] = -3.0  # a priori : les attaques sont rares, donc le modèle initial ne bloque rien
        self.thr = 1.0  # seuil d'erreur de reconstruction (calibré localement)

    # vecteur partagé = MLP complet + autoencodeur
    def shared(self):
        return np.concatenate([self.sup.get_flat(), self.ae.get_flat()])

    def set_shared(self, vec):
        n = self.sup.get_flat().size
        self.sup.set_flat(vec[:n])
        self.ae.set_flat(vec[n:])

    def copy(self):
        m = IDSModel()
        m.set_shared(self.shared())
        m.thr = self.thr
        return m

    def recon_error(self, X):
        with np.errstate(all="ignore"):
            err = ((self.ae.forward(X) - X) ** 2).sum(1)
        return np.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0)  # modèle détruit : « ne détecte rien »

    def calibrate(self, X_benign, q=0.99):
        self.thr = float(np.quantile(self.recon_error(X_benign), q)) + 1e-9

    def scores(self, X):
        # un modèle détruit par un empoisonnement peut produire inf/NaN : on le traite comme « ne détecte rien »
        with np.errstate(all="ignore"):
            p = np.nan_to_num(self.sup.forward(X)[:, 0], nan=0.0, posinf=1.0, neginf=0.0)
        a = sigmoid(4.0 * (self.recon_error(X) / self.thr - 1.0))
        return p, a, 1.0 - (1.0 - p) * (1.0 - a)  # "ou bruité" : l'un ou l'autre suffit à alerter
