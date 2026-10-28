"""
Régénère les figures du rapport (matplotlib uniquement) à partir des résultats bruts de results/.

    python3 make_report_figures.py

  results/rep1.json : RMSE de chaque étape (validation V, test local T), temps, précision.
  results/rep2.json : sensibilité des hyperparamètres des 3 étages PD (RMSE de validation, une valeur changée à la fois).
  results/rep5_meta.json : même mesures pour l'étape 6 (métadonnées : genres + époque), module soumis.
  results/rep6_mf.json : réglage de la factorisation (tune_mf.py) ; fournit la ligne « tuned on V ».
Produit fig_ladder.pdf et fig_sens.pdf.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2de"
plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "savefig.dpi": 300, "font.family": "DejaVu Sans",
})


def load(name):
    with open(os.path.join(HERE, "results", name)) as f:
        return json.load(f)


# ---------------------------------------------------------------- figure : une ligne par étape, protocoles V et T
def fig_ladder(r1):
    r1 = dict(r1)
    rows = [("mf_starter", "Matrix factorization, defaults"), ("mf_tuned", "Matrix factorization, tuned"),
            ("baseline", "Bias baseline"),
            ("pd_paper", "PD of the paper"), ("r1_residuals", "(1) PD on residuals, paper weights"),
            ("r2_mean_gap", "(2) + mean gap, confidence, shrinkage"), ("r3_items", "(3) neighbours = movies"),
            ("r4_stages", "(4) + 3 chained stages"), ("r5_final", "(5) + best 100 neighbours")]
    if "r6_meta" in r1:
        rows.append(("r6_meta", "(6) + genres and era (final)"))
    rows = [r for r in rows if r[0] in r1]           # lignes absentes (pas de métadonnées, pas de réglage MF) sautées
    xmin, xmax = 0.810, 0.925
    fig, ax = plt.subplots(figsize=(5.9, 0.18 * len(rows) + 0.45))
    for k, (key, lab) in enumerate(rows):
        y = len(rows) - 1 - k
        d = r1[key]
        vals = [np.mean(d["val"]), d["test"]]
        for v, c, m in zip(vals, (BLUE, AQUA), ("o", "D")):
            if v <= xmax:
                ax.plot(v, y, m, color=c, ms=5.5, mec="white", mew=0.6, zorder=3)
        if max(vals) > xmax:
            ax.text(xmax - 0.002, y, "val. %.2f, test %.2f  (off scale)  →" % tuple(vals),
                    ha="right", va="center", fontsize=8.5, color=INK2)
        else:
            ax.plot([min(vals), max(vals)], [y, y], color=GRID, lw=1.6, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[1] for r in rows][::-1])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("RMSE (lower is better)")
    ax.grid(axis="x")
    ax.tick_params(axis="y", length=0)
    handles = [plt.Line2D([], [], marker=m, color=c, ls="", ms=5.5, mec="white", label=l)
               for m, c, l in zip(("o", "D"), (BLUE, AQUA), ("Validation", "Test"))]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5, handletextpad=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_ladder.pdf"))
    fig.savefig(os.path.join(HERE, "fig_ladder.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- figure 2 : sensibilité, un hyperparamètre à la fois
def fig_sens(r2):
    final = r2["final"][0]
    chosen = {1: {"sigma": 1.5, "kappa": 0.25, "overlap": 640, "min_common": 8, "topk": 100},
              2: {"sigma": 0.5, "kappa": 0.5, "overlap": 80, "min_common": 5},
              3: {"sigma": 0.7, "kappa": 0.25, "overlap": 80, "min_common": 3}}
    panels = [("sigma", r"$\sigma$", [0.3, 1, 3]), ("kappa", r"$\kappa$", [0.1, 0.5, 2]),
              ("overlap", r"$\lambda$ (confidence)", [20, 160, 1280]),
              ("min_common", r"$c_{\min}$", [2, 4, 8, 16]), ("topk", r"$K$ (stage 1 only)", None)]
    colors = {1: BLUE, 2: ORANGE, 3: AQUA}
    fig, axs = plt.subplots(1, 5, figsize=(6.5, 1.95), sharey=True)
    for ax, (p, title, ticks) in zip(axs, panels):
        for st in (1, 2, 3):
            if p == "topk" and st != 1:
                continue
            data = r2["stage1_topk"] if p == "topk" else r2["stage%d_%s" % (st, p)]
            xs = list(range(len(data))) if p == "topk" else [d[0] for d in data]
            ys = [(d[1] - final) * 1000 for d in data]
            ax.plot(xs, ys, color=colors[st], lw=1.3, marker="o", ms=3, label="Stage %d" % st, zorder=2)
            idx = [i for i, d in enumerate(data) if d[0] == chosen[st][p]][0]
            ax.plot(xs[idx], ys[idx], "o", ms=7, mfc="none", mec=INK, mew=1.0, zorder=3)
        ax.axhline(0, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
        if p == "topk":
            ax.set_xticks(range(6))
            ax.set_xticklabels([str(d[0]) for d in r2["stage1_topk"]], fontsize=8.5, rotation=45, ha="right")
        else:
            ax.set_xscale("log")
            ax.set_xticks(ticks)
            ax.set_xticklabels(["%g" % t for t in ticks], fontsize=8.5)
            ax.minorticks_off()
        ax.set_xlabel(title, fontsize=10)
        ax.grid(axis="y")
        ax.tick_params(axis="y", labelsize=8.5, length=2)
        ax.tick_params(axis="x", length=2)
        ax.set_ylim(-1.2, 5)
    axs[0].set_ylabel(r"$\Delta$ val. RMSE ($\times10^{-3}$)", fontsize=9.5)
    handles = [plt.Line2D([], [], color=colors[st], lw=1.3, marker="o", ms=3, label="Stage %d" % st) for st in (1, 2, 3)]
    axs[4].legend(handles=handles, frameon=False, fontsize=8.5, loc="upper center", handlelength=1.2,
                  handletextpad=0.3, borderaxespad=0.2)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(HERE, "fig_sens.pdf"))
    fig.savefig(os.path.join(HERE, "fig_sens.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    r1 = load("rep1.json")
    if os.path.exists(os.path.join(HERE, "results", "rep5_meta.json")):
        r1["r6_meta"] = load("rep5_meta.json")      # étape genres (absente si namesngenre2.npy manquait)
    if os.path.exists(os.path.join(HERE, "results", "rep6_mf.json")):
        r1["mf_tuned"] = load("rep6_mf.json")["selected"]   # configuration retenue sur V (tune_mf.py)
        r1["mf_tuned"]["val"] = list(r1["mf_tuned"]["val"])
    fig_ladder(r1)
    if os.path.exists(os.path.join(HERE, "results", "rep2.json")):
        fig_sens(load("rep2.json"))                  # absent si run_experiments.py --no-sensitivity
    print("fig_ladder.(pdf|png) et fig_sens.(pdf|png) écrits dans", HERE)
