import numpy as np


def rmse(actual, predicted):
    """RMSE calculé uniquement sur les notes observées."""
    mask = np.isfinite(actual)

    if not np.any(mask):
        raise ValueError("Aucune note observée pour calculer le RMSE.")

    return np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2))


def matrix_factorization_observed(
    R,
    k=10,
    learning_rate_I=0.001,
    learning_rate_U=0.001,
    reg_I=0.1,
    reg_U=0.1,
    n_epochs=100
):
    """
    Factorize a users-by-items ratings matrix as I @ U.T.

    Minimize the squared Frobenius reconstruction error on observed ratings,
    plus separate Frobenius-norm regularization for I and U:
    ||R - I @ U.T||_F^2 + reg_I * ||I||_F^2 + reg_U * ||U||_F^2.
    Missing ratings must be represented by NaN.
    """
    R = np.asarray(R, dtype=np.float64)

    if R.ndim != 2:
        raise ValueError("R must be a two-dimensional matrix.")

    observed = np.isfinite(R)
    if not np.any(observed):
        raise ValueError("R must contain at least one observed rating.")

    R_filled = np.where(observed, R, 0.0)

    n_users, n_items = R.shape
    rng = np.random.default_rng(42)

    I = rng.normal(0, 0.1, (n_users, k))
    U = rng.normal(0, 0.1, (n_items, k))

    for t in range(n_epochs):

        # Mask the reconstruction residual so missing entries contribute
        # neither to the squared Frobenius error nor to its gradients.
        error = np.where(
            observed,
            R_filled - I @ U.T,
            0.0
        )

        # Gradients of the regularized squared Frobenius objective.
        grad_I = -2 * error @ U + 2 * reg_I * I
        grad_U = -2 * error.T @ I + 2 * reg_U * U

        # Simultaneous batch updates
        I_new = I - learning_rate_I * grad_I
        U_new = U - learning_rate_U * grad_U

        I, U = I_new, U_new

    R_pred = I @ U.T

    return R_pred

