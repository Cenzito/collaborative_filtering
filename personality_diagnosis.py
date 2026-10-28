
import numpy as np


def personality_diagnosis_predict(ratings, sigma=1.0):
    """
    Personality Diagnosis (PD), adapte a la prediction de notes.

    Parametres
    ----------
    ratings : array (films, utilisateurs)
        Notes observees ; les notes manquantes sont NaN.
    sigma : float
        Ecart-type du bruit gaussien suppose sur les notes.

    Retour
    ------
    predictions : array (films, utilisateurs)
        Notes predites pour tous les couples film-utilisateur.
        Les notes observees sont conservees.
    """
    R = np.asarray(ratings, dtype=np.float64)

    if R.ndim != 2:
        raise ValueError("La matrice doit etre bidimensionnelle.")
    if sigma <= 0:
        raise ValueError("sigma doit etre strictement positif.")

    n_items, n_users = R.shape
    mask = np.isfinite(R)

    if not mask.any():
        raise ValueError("Aucune note observee.")

    global_mean = float(R[mask].mean())

    # Moyennes des films, utilisees comme solution de secours.
    item_counts = mask.sum(axis=1)
    item_means = np.full(n_items, global_mean)
    np.divide(
        np.where(mask, R, 0.0).sum(axis=1),
        item_counts,
        out=item_means,
        where=item_counts > 0,
    )

    # Matrice de predictions initiale.
    predictions = np.broadcast_to(
        item_means[:, None], (n_items, n_users)
    ).copy()
    predictions[mask] = R[mask]

    # On travaille avec les utilisateurs en lignes.
    ratings_u = R.T
    mask_u = mask.T

    # Pour chaque utilisateur cible, calculer la probabilite
    # relative de chaque profil candidat.
    for u in range(n_users):
        observed_items = np.flatnonzero(mask_u[u])
        missing_items = np.flatnonzero(~mask_u[u])

        if missing_items.size == 0:
            continue

        if observed_items.size == 0:
            predictions[missing_items, u] = item_means[missing_items]
            continue

        # Profils candidats : tous les autres utilisateurs.
        candidates = np.arange(n_users) != u

        candidate_ratings = ratings_u[candidates][:, observed_items]
        candidate_mask = mask_u[candidates][:, observed_items]

        target_ratings = ratings_u[u, observed_items][None, :]

        # Vraisemblance gaussienne sur les notes communes :
        # log P(profil cible | profil candidat)
        common = candidate_mask
        squared_errors = np.where(
            common,
            (candidate_ratings - target_ratings) ** 2,
            0.0,
        )

        log_likelihood = (
            -squared_errors.sum(axis=1) / (2.0 * sigma**2)
        )

        # Un profil sans aucune note commune n'apporte pas
        # d'information : on lui attribue un poids nul.
        overlap = common.sum(axis=1)
        log_likelihood[overlap == 0] = -np.inf

        # Normalisation numeriquement stable.
        finite = np.isfinite(log_likelihood)

        if not finite.any():
            predictions[missing_items, u] = item_means[missing_items]
            continue

        max_log = np.max(log_likelihood[finite])
        weights = np.zeros(n_users - 1, dtype=np.float64)
        weights[finite] = np.exp(log_likelihood[finite] - max_log)

        total_weight = weights.sum()
        if total_weight <= 0:
            predictions[missing_items, u] = item_means[missing_items]
            continue

        weights /= total_weight

        candidate_ids = np.flatnonzero(candidates)

        # Pour chaque film manquant, ne garder que les profils
        # candidats qui ont effectivement note ce film.
        for item in missing_items:
            rated = mask[item, candidate_ids]
            if not rated.any():
                predictions[item, u] = item_means[item]
                continue

            ids = candidate_ids[rated]
            local_weights = weights[rated]

            weight_sum = local_weights.sum()
            if weight_sum <= 0:
                predictions[item, u] = item_means[item]
            else:
                predictions[item, u] = (
                    np.dot(local_weights, R[item, ids]) / weight_sum
                )

    return predictions
