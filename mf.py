
import numpy as np


def rmse(actual, predicted):
    """RMSE calculé uniquement sur les notes observées."""
    mask = np.isfinite(actual)

    if not np.any(mask):
        raise ValueError("Aucune note observée pour calculer le RMSE.")

    return np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2))


def matrix_factorization_gd(
    ratings,
    n_factors=32,
    epochs=30,
    learning_rate=0.005,
    reg=0.02,
    seed=42,
    verbose=True,
):
    """
    ratings : matrice films x utilisateurs.
              Les notes inconnues doivent être représentées par NaN.
    Retourne une matrice dense de prédictions.
    """
    R = np.asarray(ratings, dtype=np.float64)

    if R.ndim != 2:
        raise ValueError("La matrice doit être bidimensionnelle.")

    mask = np.isfinite(R)
    items, users = R.shape

    if not np.any(mask):
        raise ValueError("La matrice ne contient aucune note observée.")

    # Coordonnées des notes connues : ne jamais entraîner sur les NaN.
    item_idx, user_idx = np.where(mask)
    values = R[item_idx, user_idx]

    # Moyenne globale et biais initiaux.
    mean_rating = values.mean()
    item_bias = np.zeros(items)
    user_bias = np.zeros(users)

    # Initialisation reproductible des facteurs latents.
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 0.1, size=(items, n_factors))
    Q = rng.normal(0, 0.1, size=(users, n_factors))

    for epoch in range(epochs):
        order = rng.permutation(len(values))

        for idx in order:
            i = item_idx[idx]
            u = user_idx[idx]
            r = values[idx]

            # Prédiction avant la mise à jour.
            pred = mean_rating + item_bias[i] + user_bias[u]
            pred += P[i] @ Q[u]

            error = r - pred

            # Copies des anciens vecteurs : mises à jour cohérentes.
            p_old = P[i].copy()
            q_old = Q[u].copy()

            item_bias[i] += learning_rate * (
                error - reg * item_bias[i]
            )
            user_bias[u] += learning_rate * (
                error - reg * user_bias[u]
            )

            P[i] += learning_rate * (
                error * q_old - reg * p_old
            )
            Q[u] += learning_rate * (
                error * p_old - reg * q_old
            )

        if verbose:
            pred_train = (
                mean_rating
                + item_bias[:, None]
                + user_bias[None, :]
                + P @ Q.T
            )
            print(
                f"Epoch {epoch + 1:02d}/{epochs} "
                f"- Train RMSE: {rmse(R, pred_train):.4f}"
            )

    # Prédictions de tous les couples film-utilisateur.
    predictions = (
        mean_rating
        + item_bias[:, None]
        + user_bias[None, :]
        + P @ Q.T
    )

    return predictions
