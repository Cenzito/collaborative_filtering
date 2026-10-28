"""
Évalue le PD amélioré étage par étage.

Validation (sert à choisir les hyperparamètres) : on part de train + test (comme la plateforme,
qui reçoit toutes les notes connues) et on cache 30 % des notes du TRAIN, soit autant de notes
que 15 % de train + test. Les notes du TEST ne servent jamais de cible pendant le choix.
Test : une seule mesure, train seul en entrée, cible = test.

Usage (dans le dossier du dépôt, à côté de pd_ameliore.py ; namesngenre2.npy à côté pour la ligne métadonnées) :
    python3 evaluate_pd_ameliore.py
"""
import time

import numpy as np

from pd_ameliore import ETAGES, baseline_biais, charger_metadonnees, pd_ameliore, rmse

TRAIN_PATH = "ratings_train.npy"
TEST_PATH = "ratings_test.npy"
HIDE_FRAC = 0.30
SEEDS = (0, 1, 2)

train = np.load(TRAIN_PATH).astype(float)
test = np.load(TEST_PATH).astype(float)
train[train == 0] = np.nan
test[test == 0] = np.nan
both = np.where(np.isfinite(train), train, test)
lo, hi = np.nanmin(train), np.nanmax(train)


def split(seed):
    rng = np.random.default_rng(1000 + seed)
    ii, jj = np.where(np.isfinite(train))
    pick = rng.permutation(len(ii))[: int(HIDE_FRAC * len(ii))]
    sub = both.copy()
    sub[ii[pick], jj[pick]] = np.nan
    val = np.full_like(both, np.nan)
    val[ii[pick], jj[pick]] = train[ii[pick], jj[pick]]
    return sub, val


splits = [split(s) for s in SEEDS]

premier_etage_sans_topk = dict(ETAGES[0], sigma=0.7, kappa=0.25, overlap=160.0, min_common=3, top_k=None)
modeles = {
    "Baseline à biais": lambda R: np.clip(baseline_biais(R, np.isfinite(R)), lo, hi),
    "PD, 1 étage (films)": lambda R: pd_ameliore(R, etages=(premier_etage_sans_topk,), verbose=False, metadonnees=False),
    "PD, 3 étages + top-100": lambda R: pd_ameliore(R, verbose=False, metadonnees=False),
}
meta = charger_metadonnees(train.shape[1])
if meta is not None:
    modeles["PD, 3 étages + métadonnées (final)"] = lambda R: pd_ameliore(R, verbose=False, metadonnees=meta)
else:
    print("namesngenre2.npy introuvable : la ligne « + métadonnées » est sautée.")

print(f"{'modèle':34s} {'RMSE validation':>16s} {'RMSE test':>10s}")
for nom, f in modeles.items():
    t0 = time.time()
    val = np.mean([rmse(v, f(s)) for s, v in splits])
    tst = rmse(test, f(train))
    print(f"{nom:34s} {val:16.4f} {tst:10.4f}   ({time.time() - t0:.0f} s)")
