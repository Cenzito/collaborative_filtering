
import csv
import time

import numpy as np
from personality_diagnosis import personality_diagnosis_predict


TRAIN_PATH = "datalab-a1/ratings_train.npy"
TEST_PATH = "datalab-a1/ratings_test.npy"

SIGMA_VALUES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
MF_REFERENCE_RMSE = 0.885


def calculate_rmse(reference, predictions):
    mask = np.isfinite(reference)
    if not mask.any():
        raise ValueError("Aucune note de reference dans le test.")

    errors = reference[mask] - predictions[mask]
    return float(np.sqrt(np.mean(errors**2)))


train = np.load(TRAIN_PATH)
test = np.load(TEST_PATH)

if train.shape != test.shape:
    raise ValueError("Les matrices train et test doivent avoir la meme forme.")

results = []

for sigma in SIGMA_VALUES:
    print(f"\nPersonality Diagnosis | sigma={sigma}")
    start = time.time()

    predictions = personality_diagnosis_predict(
        train,
        sigma=sigma,
    )

    score = calculate_rmse(test, predictions)
    duration = time.time() - start

    results.append({
        "sigma": sigma,
        "test_rmse": score,
        "seconds": round(duration, 2),
    })

    print(f"Test RMSE : {score:.4f}")
    print(f"Duree     : {duration:.2f} s")

results.sort(key=lambda result: result["test_rmse"])

print("\n========== CLASSEMENT FINAL ==========")
for rank, result in enumerate(results, start=1):
    print(
        f"{rank}. sigma={result['sigma']} | "
        f"RMSE={result['test_rmse']:.4f} | "
        f"{result['seconds']:.2f}s"
    )

best = results[0]

print("\n========== MEILLEURE CONFIGURATION ==========")
print(f"Meilleur sigma : {best['sigma']}")
print(f"RMSE           : {best['test_rmse']:.4f}")
print(f"Reference MF   : {MF_REFERENCE_RMSE:.4f}")

if best["test_rmse"] < MF_REFERENCE_RMSE:
    print("PD bat le score de reference.")
else:
    print("La factorisation matricielle reste meilleure.")

with open("pd_results.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\nResultats sauvegardes dans pd_results.csv")
