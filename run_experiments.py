"""
Recalcule, à partir des données, tous les résultats qui alimentent les graphiques du rapport.

À lancer dans le dossier du dépôt (à côté de ratings_train.npy, ratings_test.npy, mf.py
et pd_ameliore.py ; namesngenre2.npy en option pour l'étape genres) :

    python run_experiments.py                  # tout : étapes + sensibilité (≈ 10 à 20 min)
    python run_experiments.py --no-sensitivity # seulement le graphique des étapes (≈ 2 min)

Écrit results/rep1.json (étapes, protocoles V et T), results/rep5_meta.json (étape genres) et
results/rep2.json (sensibilité). Puis :  python make_report_figures.py

Protocoles
  V : entrée = train + test moins 30 % des notes du TRAIN (cachées), graines 0, 1, 2 ; cible = notes cachées.
  T : entrée = train, cible = test.
"""
import argparse
import json
import os
import time

import numpy as np

import pd_ameliore as pa
from mf import matrix_factorization_gd

LO, HI = 0.5, 5.0
SEEDS = (0, 1, 2)
HIDE_FRAC = 0.30


def clip(x):
    return np.clip(x, LO, HI)


def rmse(actual, predicted):
    m = np.isfinite(actual)
    return float(np.sqrt(np.mean((actual[m] - predicted[m]) ** 2)))


def accuracy(actual, predicted):
    """Part des notes dont la prédiction arrondie au demi-point est exacte."""
    m = np.isfinite(actual)
    return float(np.mean(np.round(clip(predicted[m]) * 2) / 2 == actual[m]))


def load_data():
    train = np.load("ratings_train.npy").astype(float)
    test = np.load("ratings_test.npy").astype(float)
    train[train == 0] = np.nan
    test[test == 0] = np.nan
    both = np.where(np.isfinite(train), train, test)
    return train, test, both


def make_val_splits(train, both):
    """Protocole V : (entrée, cible) pour chaque graine."""
    out = []
    ii, jj = np.where(np.isfinite(train))
    for s in SEEDS:
        rng = np.random.default_rng(1000 + s)
        pick = rng.permutation(len(ii))[: int(HIDE_FRAC * len(ii))]
        sub = both.copy()
        sub[ii[pick], jj[pick]] = np.nan
        val = np.full_like(both, np.nan)
        val[ii[pick], jj[pick]] = train[ii[pick], jj[pick]]
        out.append((sub, val))
    return out


# ------------------------------------------------------------------ PD du papier (référence)
def pd_papier(R, sigma, min_common=2):
    """
    PD de Pennock et al. sur les notes brutes, voisins = utilisateurs : poids exp(-somme des écarts^2 / 2 sigma^2)
    sur les films notés en commun, prédiction = espérance (et non le mode, meilleure pour le RMSE).
    Repli sur une baseline à biais (régularisation 10) quand aucun voisin n'a noté le film.
    """
    mask = np.isfinite(R)
    M = mask.astype(np.float64)
    Rz = np.where(mask, R, 0.0)
    D = np.maximum(M @ (Rz ** 2).T + (Rz ** 2) @ M.T - 2.0 * (Rz @ Rz.T), 0.0)
    log_w = -D / (2.0 * sigma ** 2)
    invalide = (M @ M.T) < min_common
    np.fill_diagonal(invalide, True)
    log_w[invalide] = -np.inf
    mx = np.max(log_w, axis=1, keepdims=True)
    mx[~np.isfinite(mx)] = 0.0
    W = np.exp(log_w - mx)
    W[invalide] = 0.0
    total = W @ M
    pred = np.where(total > 0, (W @ Rz) / np.maximum(total, 1e-300), np.nan)
    sans_voisin = np.isnan(pred)
    if sans_voisin.any():
        pred[sans_voisin] = pa.baseline_biais(R, mask, 10.0, 10.0)[sans_voisin]
    return np.clip(pred, R[mask].min(), R[mask].max())


# ------------------------------------------------------------------ étapes (1) et (2), réimplémentées ici
def _cote(Z, M):
    """Distances au carré cumulées entre lignes sur les colonnes notées par les deux, et nombre de colonnes communes."""
    Z = Z.astype(np.float32)
    M = M.astype(np.float32)
    Z2 = Z ** 2
    D = np.maximum(M @ Z2.T + Z2 @ M.T - 2 * (Z @ Z.T), 0)
    return Z, M, D, M @ M.T


def pd_residus_poids_papier(R, sigma):
    """Étape (1) : voisins = utilisateurs, poids du papier exp(-somme des écarts^2 / 2 sigma^2), sur les résidus."""
    m = np.isfinite(R)
    B = pa.baseline_biais(R, m, 3.0, 5.0)
    Z, M, D, C = _cote(np.where(m, R - B, 0.0), m)
    logw = -D / (2 * sigma ** 2)
    invalide = C < 2
    np.fill_diagonal(invalide, True)
    logw[invalide] = -np.inf
    mx = np.max(logw, axis=1, keepdims=True)
    mx[~np.isfinite(mx)] = 0
    W = np.exp(logw - mx)
    W[invalide] = 0
    W = W.astype(np.float32)
    return B + (W @ Z) / ((W @ M) + 1e-9)


def pd_ecart_moyen(R):
    """Étape (2) : écart moyen, confiance c/(c+40), au moins 8 notes communes, rétrécissement kappa = 2."""
    m = np.isfinite(R)
    B = pa.baseline_biais(R, m, 3.0, 5.0)
    Z, M, D, C = _cote(np.where(m, R - B, 0.0), m)
    W = np.exp(-(D / np.maximum(C, 1)) / (2 * 0.7 ** 2)) * (C / (C + 40.0))
    W[C < 8] = 0
    np.fill_diagonal(W, 0)
    W = W.astype(np.float32)
    return B + (W @ Z) / ((W @ M) + 2.0)


def meta_si_presente(n_films):
    return pa.charger_metadonnees(n_films, [os.getcwd()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-sensitivity", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train, test, both = load_data()
    val = make_val_splits(train, both)
    meta = meta_si_presente(train.shape[1])
    print("métadonnées :", "trouvées" if meta is not None else "absentes (étape genres sautée)")

    # --- réglages choisis sur la première graine de V
    sg_paper = min((rmse(val[0][1], clip(pd_papier(val[0][0], s))), s)
                   for s in (1.0, 2.5, 5.0, 10.0, 20.0))[1]
    sg_r1 = min((rmse(val[0][1], clip(pd_residus_poids_papier(val[0][0], s))), s)
                for s in (16.0, 32.0, 64.0, 128.0))[1]
    print("sigma du PD du papier :", sg_paper, "| sigma de l'étape (1) :", sg_r1)

    films0 = dict(voisins="films", sigma=0.7, kappa=0.25, overlap=160.0, min_common=3, top_k=None)
    sans_meta = dict(verbose=False, metadonnees=False)
    methodes = {
        "baseline": lambda R: pa.baseline_biais(R, np.isfinite(R), 3.0, 5.0),
        "mf_starter": lambda R: matrix_factorization_gd(R, n_factors=32, epochs=30, learning_rate=0.005,
                                                        reg=0.02, seed=42, verbose=False),
        "pd_paper": lambda R: pd_papier(R, sg_paper),
        "r1_residuals": lambda R: pd_residus_poids_papier(R, sg_r1),
        "r2_mean_gap": pd_ecart_moyen,
        "r3_items": lambda R: pa.pd_ameliore(R, etages=(films0,), **sans_meta),
        "r4_stages": lambda R: pa.pd_ameliore(R, etages=(films0, pa.ETAGES[1], pa.ETAGES[2]), **sans_meta),
        "r5_final": lambda R: pa.pd_ameliore(R, **sans_meta),
    }
    if meta is not None:
        methodes["r6_meta"] = lambda R: pa.pd_ameliore(R, verbose=False, metadonnees=meta)

    res = {"sigma_paper": sg_paper, "sigma_r1": sg_r1}
    for nom, f in methodes.items():
        v = [rmse(y, clip(f(s))) for s, y in val]
        t0 = time.time()
        p = f(train)
        dt = time.time() - t0
        res[nom] = dict(val=v, test=rmse(test, clip(p)), acc_test=accuracy(test, p), time_test=dt)
        print("%-13s V %.4f   T %.4f   acc T %.3f   %.1f s" % (nom, np.mean(v), res[nom]["test"], res[nom]["acc_test"], dt),
              flush=True)

    # --- gain par niveau de données (utilisateurs / films), V, 3 graines
    lignes = []
    for s, y in val:
        m = np.isfinite(s)
        nu = m.sum(1)[:, None] * np.ones((1, s.shape[1]))
        ni = m.sum(0)[None, :] * np.ones((s.shape[0], 1))
        pb, pf = clip(methodes["baseline"](s)), clip(methodes["r5_final"](s))
        h = np.isfinite(y)
        lignes.append(np.column_stack([nu[h], ni[h], (y[h] - pb[h]) ** 2, (y[h] - pf[h]) ** 2]))
    A = np.vstack(lignes)

    def groupes(col, bornes, etiquettes):
        out = []
        for lo, hi, lab in zip(bornes[:-1], bornes[1:], etiquettes):
            k = (A[:, col] >= lo) & (A[:, col] < hi)
            out.append(dict(label=lab, n=int(k.sum()), share=float(k.mean()),
                            base=float(np.sqrt(A[k, 2].mean())), pd=float(np.sqrt(A[k, 3].mean()))))
        return out

    res["buckets_user"] = groupes(0, [0, 25, 50, 100, 200, 1e9], ["<25", "25-49", "50-99", "100-199", "200+"])
    res["buckets_item"] = groupes(1, [0, 3, 6, 12, 30, 1e9], ["0-2", "3-5", "6-11", "12-29", "30+"])

    meta_res = res.pop("r6_meta", None)
    json.dump(res, open(os.path.join(args.out, "rep1.json"), "w"), indent=1)
    if meta_res is not None:
        json.dump(meta_res, open(os.path.join(args.out, "rep5_meta.json"), "w"), indent=1)
    print("écrit :", os.path.join(args.out, "rep1.json"))
    if args.no_sensitivity:
        return

    # --- sensibilité du PD final (sans genres) : une valeur change, les autres restent finales
    def score(etages, reg=(3.0, 5.0)):
        v = [rmse(y, pa.pd_ameliore(s, etages=etages, reg_item=reg[0], reg_user=reg[1], **sans_meta)) for s, y in val]
        return float(np.mean(v)), float(np.std(v))

    finale = [dict(e) for e in pa.ETAGES]
    sens = {"final": score(tuple(finale))}
    grille = {
        0: {"sigma": [0.5, 0.7, 1.0, 1.5, 2.0, 3.0], "kappa": [0.1, 0.25, 0.5, 1.0],
            "overlap": [40, 160, 320, 640, 1280], "min_common": [2, 3, 5, 8, 12, 16]},
        1: {"sigma": [0.3, 0.5, 0.7, 1.0], "kappa": [0.25, 0.5, 1.0, 2.0],
            "overlap": [20, 40, 80, 160, 320], "min_common": [3, 5, 8, 12]},
        2: {"sigma": [0.5, 0.7, 1.0, 1.5], "kappa": [0.1, 0.25, 0.5, 1.0],
            "overlap": [40, 80, 160, 320], "min_common": [2, 3, 5, 8]},
    }
    for st, g in grille.items():
        for nom, valeurs in g.items():
            sortie = []
            for x in valeurs:
                cfg = [dict(e) for e in finale]
                cfg[st][nom] = x
                sortie.append((x,) + score(tuple(cfg)))
            sens["stage%d_%s" % (st + 1, nom)] = sortie
            print("étage %d %-10s" % (st + 1, nom), [(x, round(m, 4)) for x, m, _ in sortie], flush=True)
    sortie = []
    for K in (20, 50, 100, 200, 500, None):
        cfg = [dict(e) for e in finale]
        cfg[0]["top_k"] = K
        sortie.append(("all" if K is None else K,) + score(tuple(cfg)))
    sens["stage1_topk"] = sortie
    sens["n_stages"] = [(n,) + score(tuple(finale[:n])) for n in (0, 1, 2, 3)]
    sens["order_U_I_I"] = score((finale[1], finale[0], finale[2]))
    reg = []
    for r in ((1.0, 1.0), (2.0, 3.0), (3.0, 5.0), (5.0, 8.0), (10.0, 10.0), (20.0, 20.0)):
        base = float(np.mean([rmse(y, clip(pa.baseline_biais(s, np.isfinite(s), r[0], r[1]))) for s, y in val]))
        reg.append((r, base, score(tuple(finale), r)[0]))
    sens["baseline_reg"] = reg
    json.dump(sens, open(os.path.join(args.out, "rep2.json"), "w"), indent=1)
    print("écrit :", os.path.join(args.out, "rep2.json"))


if __name__ == "__main__":
    main()
