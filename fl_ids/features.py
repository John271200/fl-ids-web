"""Prétraitement local : masquage des valeurs sensibles puis extraction de caractéristiques.

Tout ce module s'exécute chez le participant. Les valeurs sensibles (jetons, mots de passe, IBAN, e-mails,
identifiants longs) sont remplacées par des marqueurs de type avant de produire les vecteurs : le modèle
apprend la forme des données, pas leur contenu.

La tokenisation repose sur un vocabulaire global fixe et non évolutif : un hachage de trigrammes de
caractères vers N_HASH cases. Le vocabulaire n'est donc jamais appris sur les données d'un participant.
"""
from __future__ import annotations

import math
import re
import urllib.parse as up
import zlib
from functools import lru_cache

import numpy as np

N_HASH = 128
N_SYN = 18
DIM = N_SYN + N_HASH

_SECRET = re.compile(r"(password|passwd|pwd|token|sid|secret)=([^&\s]*)", re.I)
_BEARER = re.compile(r"(Bearer)\s+(\S+)")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
_HEX = re.compile(r"\b[0-9a-fA-F]{16,}\b")
_NUM = re.compile(r"\d{6,}")


def _shape(value: str) -> str:
    """Remplace chaque caractère par sa classe (A, a, 0) en conservant les caractères spéciaux."""
    out = []
    for ch in value:
        c = "A" if ch.isupper() else "a" if ch.islower() else "0" if ch.isdigit() else ch
        if len(out) >= 3 and out[-1] == c and out[-2] == c and out[-3] == c and c in "Aa0":
            continue
        out.append(c)
    return "".join(out)


def mask(raw: str) -> str:
    s = _SECRET.sub(lambda m: f"{m.group(1)}={_shape(m.group(2))}", raw)
    s = _BEARER.sub(lambda m: f"{m.group(1)} <TOKEN>", s)
    s = _EMAIL.sub("<EMAIL>", s)
    s = _IBAN.sub("<IBAN>", s)
    s = _HEX.sub("<HEX>", s)
    return _NUM.sub("<NUM>", s)


def _decode(s: str, times: int) -> str:
    for _ in range(times):
        s = up.unquote(s.replace("+", " "))
    return s


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = np.bincount(np.frombuffer(s.encode("utf-8", "ignore"), dtype=np.uint8), minlength=256)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _max_special_run(s: str) -> int:
    best = cur = 0
    for ch in s:
        cur = cur + 1 if not ch.isalnum() and not ch.isspace() else 0
        best = max(best, cur)
    return best


def _one(raw: str) -> np.ndarray:
    m = mask(raw)
    d1, d2 = _decode(m, 1), _decode(m, 2)
    n = max(len(d2), 1)
    qpart = m.split("?", 1)[1] if "?" in m else m.split(" ", 2)[2] if m.count(" ") >= 2 else ""
    special = sum(1 for c in d2 if not c.isalnum() and not c.isspace())
    syn = np.array([
        math.log1p(len(m)),
        math.log1p(len(qpart)),
        qpart.count("=") + qpart.count("&"),
        special / n,
        d2.count("'") + d2.count('"'),
        d2.count("--") + d2.count("/*") + d2.count("#"),
        d2.count("<") + d2.count(">"),
        m.count("%") / n,
        float(d1 != d2),
        _entropy(d2) / 8.0,
        sum(c.isdigit() for c in d2) / n,
        sum(c.isupper() for c in d2) / n,
        d2.count(";") + d2.count("|") + d2.count("`") + d2.count("&&"),
        d2.count("../") + d2.count("..\\") + d2.count("....//"),
        d2.count("(") + d2.count(")"),
        _max_special_run(d2),
        m.split(" ")[1].count("/") if " " in m else 0,
        d2.count("{") + d2.count("$") + d2.count("["),
    ], dtype=np.float64)
    low = d2.lower()
    grams = np.zeros(N_HASH)
    for i in range(len(low) - 2):
        grams[zlib.crc32(low[i:i + 3].encode("utf-8", "ignore")) % N_HASH] += 1.0
    grams = np.log1p(grams)
    norm = np.linalg.norm(grams)
    if norm > 0:
        grams = grams / norm * 3.0
    return np.concatenate([syn, grams])


def featurize(raws: list[str]) -> np.ndarray:
    return np.stack([_one(r) for r in raws]) if raws else np.zeros((0, DIM))


@lru_cache(maxsize=1)
def genesis_scaler() -> tuple[np.ndarray, np.ndarray]:
    """Moyenne/écart-type fixés à la genèse du consortium sur un corpus public synthétique."""
    from .synthetic import make_requests, ATTACK_MIX
    reqs = make_requests("mixed", 1500, 300, ATTACK_MIX["mixed"], seed=987654)
    X = featurize([r.raw for r in reqs])
    return X.mean(0), X.std(0) + 1e-6


def featurize_scaled(raws: list[str]) -> np.ndarray:
    mu, sd = genesis_scaler()
    return (featurize(raws) - mu) / sd
