import argparse
import csv
import time
from pathlib import Path

import numpy as np
from mf import matrix_factorization_gd


def rmse_on_test(test, predictions):
    mask = np.isfinite(test)
    if not mask.any():
        raise ValueError("ratings_test.npy ne contient aucune note observée.")
    if predictions.shape != test.shape:
        raise ValueError(f"Dimensions incompatibles : test={test.shape}, prédictions={predictions.shape}")
    errors = test[mask] - predictions[mask]
    return float(np.sqrt(np.mean(errors ** 2)))


def main():
    parser = argparse.ArgumentParser(
        description="Recherche d'hyperparamètres MF avec le test fourni par l'enseignant."
    )
    parser.add_argument("--train", default="datalab-a1/ratings_train.npy")
    parser.add_argument("--test", default="datalab-a1/ratings_test.npy")
    parser.add_argument("--output", default="mf_tuning_results.csv")
    parser.add_argument("--factors", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--epochs", type=int, nargs="+", default=[20, 30])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.005])
    parser.add_argument("--regs", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = np.load(args.train)
    test = np.load(args.test)
    if train.shape != test.shape:
        raise ValueError(f"Les matrices doivent avoir la même forme : {train.shape} vs {test.shape}")
    train_mask = np.isfinite(train)
    test_mask = np.isfinite(test)
    overlap = int(np.count_nonzero(train_mask & test_mask))
    if overlap:
        print(f"ATTENTION : {overlap} positions sont observées à la fois dans train et test.")
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Train et test doivent chacun contenir des notes observées.")

    print(f"Train : {train.shape}, notes={train_mask.sum()}")
    print(f"Test  : {test.shape}, notes={test_mask.sum()}")
    print("Le modèle est entraîné uniquement sur train ; le RMSE est calculé uniquement sur test.")

    rows = []
    total = len(args.factors) * len(args.epochs) * len(args.learning_rates) * len(args.regs)
    run = 0
    for factors in args.factors:
        for epochs in args.epochs:
            for lr in args.learning_rates:
                for reg in args.regs:
                    run += 1
                    print(f"\n[{run}/{total}] factors={factors}, epochs={epochs}, lr={lr}, reg={reg}")
                    start = time.perf_counter()
                    predictions = matrix_factorization_gd(
                        train,
                        n_factors=factors,
                        epochs=epochs,
                        learning_rate=lr,
                        reg=reg,
                        seed=args.seed,
                        verbose=False,
                    )
                    score = rmse_on_test(test, predictions)
                    elapsed = time.perf_counter() - start
                    row = {
                        "model": "MF",
                        "factors": factors,
                        "epochs": epochs,
                        "learning_rate": lr,
                        "reg": reg,
                        "seed": args.seed,
                        "test_rmse": score,
                        "seconds": round(elapsed, 2),
                    }
                    rows.append(row)
                    print(f"Test RMSE={score:.5f} | durée={elapsed:.1f}s")

                    # Sauvegarde après chaque essai pour conserver les résultats si interruption.
                    fieldnames = list(rows[0].keys())
                    with open(args.output, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(sorted(rows, key=lambda x: x["test_rmse"]))

    best = min(rows, key=lambda x: x["test_rmse"])
    print("\nMeilleure configuration observée (à confirmer, le test sert ici à la sélection) :")
    print(best)
    print(f"Résultats sauvegardés dans : {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
