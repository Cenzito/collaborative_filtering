"""
PD amélioré (Personality Diagnosis adapté aux données éparses).

Base : Pennock et al. (2000) — le poids d'un voisin est une vraisemblance gaussienne
exp(-écart^2 / 2 sigma^2) mesurée sur les éléments notés en commun.
La prédiction reste une moyenne pondérée des notes des voisins (espérance).

Modifications par rapport au PD du papier, chacune choisie sur un jeu de VALIDATION :
  1. Résidus : PD travaille sur z = note - baseline (moyenne globale + biais utilisateur
     + biais film), pas sur les notes brutes.
  2. Voisins = films au premier étage : on compare deux films sur les utilisateurs qui les
     ont notés. Mesuré : meilleur que voisins = utilisateurs (RMSE validation 0,849 vs 0,853).
     Cause non démontrée ; piste : un utilisateur a ~46 notes (médiane) contre ~6 par film,
     donc une case reçoit plus de votes de films voisins que d'utilisateurs voisins.
  3. Écart moyen + confiance : l'écart est ramené à une moyenne par élément commun, et le
     poids est multiplié par c / (c + overlap), où c = nombre d'éléments communs.
  4. Rétrécissement kappa : résidu prédit = somme(poids * résidus) / (somme(poids) + kappa).
     Sans voisin fiable, la prédiction retombe sur la baseline.
  5. Étages enchaînés : un deuxième PD (voisins = utilisateurs), puis un troisième (voisins =
     films), sont appliqués aux résidus que le PD précédent n'a pas expliqués.
  6. Top-k : au premier étage, chaque film ne garde que ses k meilleurs voisins.
  7. Métadonnées (optionnel) : après les étages PD, une régression ridge PAR UTILISATEUR de
     ses résidus restants sur les caractéristiques du film (genres + époque, déduite du titre).
     Elle capte ce que les notes seules disent mal : « cet utilisateur aime les films noirs
     des années 50 ». Sans fichier de métadonnées, cette étape est simplement sautée.

Interface comme mf.py / personality_diagnosis.py : matrice utilisateurs x films,
NaN = note inconnue, retourne une matrice dense de prédictions.
"""
import os
import re

import numpy as np

# Un étage = un PD appliqué aux résidus laissés par les étages précédents.
ETAGES = (
    dict(voisins="films",        sigma=1.5, kappa=0.25, overlap=640.0, min_common=8, top_k=100),
    dict(voisins="utilisateurs", sigma=0.5, kappa=0.5,  overlap=80.0,  min_common=5, top_k=None),
    dict(voisins="films",        sigma=0.7, kappa=0.25, overlap=80.0,  min_common=3, top_k=None),
)


# Réglage de l'étape métadonnées (choisi sur la validation, voir le rapport).
LAMBDA_META = 20.0
# Bornes des époques (année du film, lue dans le titre « Titre (1995) »).
BORNES_EPOQUES = (0, 1950, 1970, 1980, 1990, 1995, 2000, 3000)
NOM_FICHIER_META = "namesngenre2.npy"


def caracteristiques_films(metadonnees):
    """
    metadonnees : tableau (n_films, 2) de chaînes [titre, 'Genre1|Genre2|...'], ligne j = colonne j des notes.
    Retourne F (n_films, n_genres + 7) : un indicateur par genre, puis un indicateur d'époque.
    """
    metadonnees = np.asarray(metadonnees)
    genres = sorted({g for s in metadonnees[:, 1] for g in str(s).split("|")})
    index = {g: k for k, g in enumerate(genres)}
    n_ep = len(BORNES_EPOQUES) - 1
    F = np.zeros((len(metadonnees), len(genres) + n_ep))
    for j, (titre, gs) in enumerate(metadonnees):
        for g in str(gs).split("|"):
            F[j, index[g]] = 1.0
        m = re.search(r"\((\d{4})\)\s*$", str(titre).strip())
        if m:
            annee = int(m.group(1))
            for b in range(n_ep):
                if BORNES_EPOQUES[b] <= annee < BORNES_EPOQUES[b + 1]:
                    F[j, len(genres) + b] = 1.0
    return F


def charger_metadonnees(n_films, dossiers=None):
    """
    Cherche namesngenre2.npy (dossier du script, dossier courant, dossier parent) et le charge SANS pickle.
    Retourne le tableau, ou None s'il est absent ou ne correspond pas aux n_films colonnes des notes.
    """
    if dossiers is None:
        ici = os.path.dirname(os.path.abspath(__file__))
        dossiers = [ici, os.getcwd(), os.path.dirname(ici)]
    for d in dossiers:
        chemin = os.path.join(d, NOM_FICHIER_META)
        if os.path.isfile(chemin):
            try:
                meta = np.load(chemin, allow_pickle=False)
            except Exception:
                continue
            if meta.ndim == 2 and meta.shape == (n_films, 2):
                return meta
    return None


def etage_metadonnees(R, mask, pred, F, lam=LAMBDA_META):
    """
    Pour chaque utilisateur u : w_u = argmin sum_{j noté par u} (résidu_uj - F_j . w)^2 + lam ||w||^2,
    puis résidu prédit (u, j) = F_j . w_u. Tous les utilisateurs sont résolus d'un coup.
    """
    n_u, n_i = R.shape
    k = F.shape[1]
    res = np.where(mask, R - pred, 0.0)
    FF = (F[:, :, None] * F[:, None, :]).reshape(n_i, k * k)
    A = (mask.astype(np.float64) @ FF).reshape(n_u, k, k) + lam * np.eye(k)[None, :, :]
    b = res @ F
    w = np.linalg.solve(A, b[:, :, None])[:, :, 0]
    return w @ F.T


def rmse(actual, predicted):
    """RMSE calculé uniquement sur les notes observées."""
    mask = np.isfinite(actual)
    if not np.any(mask):
        raise ValueError("Aucune note observée pour calculer le RMSE.")
    return float(np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2)))


def baseline_biais(R, mask, reg_item=3.0, reg_user=5.0, n_iter=10):
    """Moyenne globale + biais film + biais utilisateur (régularisés), 10 itérations alternées."""
    Rz = np.where(mask, R, 0.0)
    mu = Rz[mask].mean()
    b_user = np.zeros(R.shape[0])
    b_item = np.zeros(R.shape[1])
    for _ in range(n_iter):
        b_item = ((Rz - mu - b_user[:, None]) * mask).sum(axis=0) / (reg_item + mask.sum(axis=0))
        b_user = ((Rz - mu - b_item[None, :]) * mask).sum(axis=1) / (reg_user + mask.sum(axis=1))
    return mu + b_user[:, None] + b_item[None, :]


def _etage_pd(Z, M, sigma, kappa, overlap, min_common, top_k):
    """
    Un étage de PD. Lignes de Z = les éléments comparés (films ou utilisateurs), colonnes = les autres.
    Retourne le résidu prédit pour chaque case (même forme que Z).
    """
    Z = Z.astype(np.float32)
    M = M.astype(np.float32)
    Z2 = Z ** 2

    # D[a, i] = somme des (z_a - z_i)^2 sur les colonnes notées par a ET par i.
    D = np.maximum(M @ Z2.T + Z2 @ M.T - 2.0 * (Z @ Z.T), 0.0)
    C = M @ M.T                                     # éléments communs
    W = np.exp(-(D / np.maximum(C, 1.0)) / (2.0 * sigma ** 2)) * (C / (C + overlap))
    W[C < min_common] = 0.0
    np.fill_diagonal(W, 0.0)
    del D, C
    if top_k is not None and top_k < W.shape[1]:
        seuil = -np.partition(-W, top_k, axis=1)[:, top_k][:, None]
        W = np.where(W > seuil, W, 0.0)
    W = W.astype(np.float32)

    # résidu prédit pour (a, colonne k) = sum_i W[a, i] z[i, k] / (sum_i W[a, i] m[i, k] + kappa)
    return (W @ Z) / ((W @ M) + kappa)


def pd_ameliore(
    ratings,
    etages=ETAGES,
    reg_item=3.0,
    reg_user=5.0,
    verbose=True,
    metadonnees=None,
    lam_meta=LAMBDA_META,
):
    """
    ratings : matrice utilisateurs x films, NaN (ou 0) = note inconnue.
    metadonnees : tableau (n_films, 2) [titre, genres] aligné sur les colonnes (namesngenre2.npy), ou None.
        None = on cherche namesngenre2.npy à côté du script ; absent => pas d'étape métadonnées.
        Passer False pour la désactiver explicitement.
    lam_meta : régularisation ridge de l'étape métadonnées.
    etages  : liste d'étages PD (voisins, sigma, kappa, overlap, min_common, top_k).
        sigma      : écart-type du bruit gaussien (échelle des résidus, pas des notes brutes).
        kappa      : rétrécissement vers la prédiction précédente (0 = aucun).
        overlap    : confiance c / (c + overlap) ; plus il est grand, plus on exige d'éléments communs.
        min_common : nombre minimal de notes en commun pour être voisin.
        top_k      : nombre de voisins gardés par élément (None = tous).
    reg_item, reg_user : régularisation des biais de la baseline.
    """
    R = np.asarray(ratings, dtype=np.float64).copy()
    if R.ndim != 2:
        raise ValueError("La matrice doit être bidimensionnelle.")
    R[R == 0] = np.nan
    mask = np.isfinite(R)
    if not mask.any():
        raise ValueError("La matrice ne contient aucune note observée.")

    pred = baseline_biais(R, mask, reg_item, reg_user)
    for e in etages:
        if e["sigma"] <= 0:
            raise ValueError("sigma doit être strictement positif.")
        Z = np.where(mask, R - pred, 0.0)            # résidus que les étages précédents laissent
        if e["voisins"] == "films":
            res = _etage_pd(Z.T, mask.T, e["sigma"], e["kappa"], e["overlap"], e["min_common"], e["top_k"]).T
        else:
            res = _etage_pd(Z, mask, e["sigma"], e["kappa"], e["overlap"], e["min_common"], e["top_k"])
        pred = pred + res
        if verbose:
            print(f"PD étage {e['voisins']} (sigma={e['sigma']}, kappa={e['kappa']}, "
                  f"overlap={e['overlap']}, min_common={e['min_common']}, top_k={e['top_k']})")

    if metadonnees is None:
        metadonnees = charger_metadonnees(R.shape[1])
    if metadonnees is not None and metadonnees is not False:
        metadonnees = np.asarray(metadonnees)
        if metadonnees.shape != (R.shape[1], 2):
            raise ValueError("metadonnees doit avoir une ligne par colonne de notes : (n_films, 2).")
        F = caracteristiques_films(metadonnees)
        pred = pred + etage_metadonnees(R, mask, pred, F, lam_meta)
        if verbose:
            print(f"Étage métadonnées : ridge par utilisateur sur {F.shape[1]} caractéristiques (lambda={lam_meta})")
    elif verbose and metadonnees is None:
        print(f"Pas de {NOM_FICHIER_META} trouvé : étape métadonnées sautée.")

    return np.clip(pred, R[mask].min(), R[mask].max())


def personality_diagnosis_predict(ratings, sigma=None, **kwargs):
    """Alias compatible avec evaluate_pd.py (sigma ignoré : réglé par étage dans ETAGES)."""
    kwargs.setdefault("verbose", False)
    return pd_ameliore(ratings, **kwargs)
