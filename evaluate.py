
import numpy as np
from mf import matrix_factorization_gd, rmse
from mf_original import matrix_factorization_observed 

train = np.load("datalab-a1/ratings_train.npy")
test = np.load("datalab-a1/ratings_test.npy")

# Entraînement uniquement sur train.
predictions = matrix_factorization_gd(
    train,
    n_factors=32,
    epochs=50,
    learning_rate=0.001,
    reg=0.01,
    seed=7,
    verbose=False,
)

# Évaluation uniquement sur les notes connues de test.
test_mask = np.isfinite(test)
score = rmse(test, predictions)

print(f"Test RMSE Updated: {score:.4f}")
print("Nombre de notes test:", test_mask.sum())

predictions_observed = matrix_factorization_observed(
    train,
    k=32,
    learning_rate_I=0.001,
    learning_rate_U=0.001,
    reg_I=0.01,
    reg_U=0.01,
    n_epochs=50
)

# Évaluation uniquement sur les notes connues de test.
test_mask = np.isfinite(test)
score_new = rmse(test, predictions_observed)

print(f"Test RMSE Original: {score_new:.4f}")
print("Nombre de notes test:", test_mask.sum())