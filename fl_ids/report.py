"""Figures et tableau de résultats à partir des JSON produits par les expériences."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAMES = {
    "local_seul": "Local seul", "federe_fedavg": "Fédéré (FedAvg)", "federe_fedprox": "Fédéré (FedProx)",
    "federe_dp_secagg": "Fédéré + DP + SecAgg", "centralise": "Centralisé (borne haute)",
    "fedavg_sans_defense": "FedAvg sans défense", "fedavg_validation_rollback": "FedAvg + validation/rollback",
    "mediane": "Médiane", "moyenne_tronquee": "Moyenne tronquée", "secagg_sans_filtre": "SecAgg sans filtre",
    "secagg_filtre_integrite": "SecAgg + filtre d'intégrité",
}
NAVY, ACC, GREY = "#1F3A5F", "#B8862B", "#8A94A3"


def _bars(data, keys, title, path):
    names = list(data)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.8 / len(keys)
    palette = [NAVY, ACC, GREY, "#4F7CAC"]
    for j, k in enumerate(keys):
        xs = [i + j * w for i in range(len(names))]
        ax.bar(xs, [data[n][k][0] for n in names], w, yerr=[data[n][k][1] for n in names], label=k,
               color=palette[j % 4], capsize=2)
    ax.set_xticks([i + w * (len(keys) - 1) / 2 for i in range(len(names))])
    ax.set_xticklabels([NAMES.get(n, n) for n in names], rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, ncol=len(keys))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _table(data, cols):
    head = "| Configuration | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    rows = ""
    for n, v in data.items():
        rows += f"| {NAMES.get(n, n)} | " + " | ".join(f"{v[c][0]:.3f} ± {v[c][1]:.3f}" for c in cols) + " |\n"
    return head + rows


def make_report(out: Path):
    cols = ["recall", "precision", "fpr", "f1", "recall_known", "recall_novel"]
    md = ["# Résultats\n"]
    cmp_path, poi_path, dp_path = out / "compare.json", out / "poisoning.json", out / "dp_tradeoff.json"
    if cmp_path.exists():
        c = json.loads(cmp_path.read_text())
        _bars(c, ["f1", "recall_known", "recall_novel"], "Local seul, fédéré et centralisé (moyenne des 5 participants)",
              out / "compare.png")
        md.append("## Local seul, fédéré, centralisé\n\nMoyenne ± écart-type sur les graines, test sur attaques connues et inédites.\n\n"
                  + _table(c, cols) + "\n![comparaison](compare.png)\n")
    if poi_path.exists():
        p = json.loads(poi_path.read_text())
        _bars(p, ["f1", "recall_known"], "Un participant malveillant sur cinq (métriques des participants honnêtes)",
              out / "poisoning.png")
        md.append("## Empoisonnement (1 participant sur 5, mises à jour inversées et amplifiées x10)\n\n"
                  + _table(p, cols) + "\n![empoisonnement](poisoning.png)\n")
    if dp_path.exists():
        d = json.loads(dp_path.read_text())
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        xs = [k for k in d]
        ax.plot(xs, [d[k]["f1"] for k in xs], "-o", color=NAVY, label="F1")
        ax.plot(xs, [d[k]["recall_novel"] for k in xs], "-s", color=ACC, label="rappel (attaques inédites)")
        ax.set_xlabel("multiplicateur de bruit sigma (0 = sans DP)")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "dp_tradeoff.png", dpi=150)
        plt.close(fig)
        rows = "| sigma | epsilon (delta=1e-5, 20 rounds) | F1 | rappel | rappel inédites |\n|---|---|---|---|---|\n"
        for k, v in d.items():
            eps = "infini (pas de DP)" if v["epsilon"] is None else f"{v['epsilon']:.1f}"
            rows += f"| {k} | {eps} | {v['f1']:.3f} | {v['recall']:.3f} | {v['recall_novel']:.3f} |\n"
        md.append("## Compromis confidentialité / précision (DP-SGD + SecAgg)\n\n" + rows + "\n![dp](dp_tradeoff.png)\n")
    (out / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
