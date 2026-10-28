"""Tune and compare PD, PD+genres and bias baseline using validation RMSE."""
import argparse, csv, os
import numpy as np
from pd_genre import load_genres, predict_entries, _bias_baseline, rmse


def floats(s): return [float(x.strip()) for x in s.split(',') if x.strip()]
def ints(s): return [int(x.strip()) for x in s.split(',') if x.strip()]
def load_matrix(path):
    x = np.asarray(np.load(path), dtype=float)
    if x.ndim != 2: raise ValueError(f'{path} must be a 2-D matrix, got {x.shape}')
    return x


def score(train, target, genres, method, sigma, overlap, weight, bias_reg):
    ii, uu = np.where(np.isfinite(target))
    actual = target[ii, uu]
    if method == 'baseline':
        _, _, _, base = _bias_baseline(train, reg=bias_reg)
        pred = base[ii, uu]
        vals = train[np.isfinite(train)]
        pred = np.clip(pred, vals.min(), vals.max())
    else:
        # Zero genre features disable content similarity for PD-only comparison.
        g = genres if method == 'hybrid' else np.zeros_like(genres)
        w = weight if method == 'hybrid' else 1.0
        pred = predict_entries(train, ii, uu, g, sigma=sigma, min_overlap=overlap,
                               pd_weight=w, bias_reg=bias_reg)
    return rmse(actual, pred)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', default='datalab-a1/ratings_train.npy')
    ap.add_argument('--test', default='datalab-a1/ratings_test.npy')
    ap.add_argument('--genres', default='datalab-a1/namesngenre2.npy')
    ap.add_argument('--movie-ids', default='datalab-a1/movie_ids.npy', help='IDs in the same order as rating movie columns')
    ap.add_argument('--validation-fraction', type=float, default=0.20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--sigmas', default='1,2.5,5,10')
    ap.add_argument('--min-overlaps', default='1,2,3')
    ap.add_argument('--pd-weights', default='0,0.5,1', help='1=PD seul, 0=genres seuls, 0.5=mélange')
    ap.add_argument('--bias-reg', type=float, default=10.0)
    ap.add_argument('--results', default='pd_genre_results.csv')
    ap.add_argument('--skip-test', action='store_true', help='Ne pas lancer le test final')
    args = ap.parse_args()

    # The professor's corrected dataset has movies in COLUMNS: raw shape is users x movies.
    R_raw = load_matrix(args.train); test_raw = load_matrix(args.test)
    if test_raw.shape != R_raw.shape: raise ValueError(f'train shape {R_raw.shape} != test shape {test_raw.shape}')
    n_users, n_movies = R_raw.shape
    movie_ids = np.asarray(np.load(args.movie_ids, allow_pickle=True)).reshape(-1)
    if len(movie_ids) != n_movies:
        raise ValueError(f'{args.movie_ids} has {len(movie_ids)} IDs, but ratings have {n_movies} movie columns.')
    if len(set(map(str, movie_ids.tolist()))) != len(movie_ids):
        raise ValueError(f'{args.movie_ids} contains duplicate movieId values.')
    # namesngenre2 rows and movie_ids follow the corrected movie order; never align by title.
    genres, vocab = load_genres(args.genres, n_movies)
    if np.any(np.isfinite(R_raw) & np.isfinite(test_raw)):
        print('WARNING: train/test have overlapping observed entries; verify your split.')
    print(f'Input matrix: {n_users} users x {n_movies} movies; validated {len(movie_ids)} movie IDs.')
    # Existing PD implementation uses (items, users), so transpose internally only.
    R = R_raw.T.copy(); test = test_raw.T.copy()
    if not 0 < args.validation_fraction < 1: raise ValueError('validation-fraction must be between 0 and 1')

    rng = np.random.default_rng(args.seed)
    obs = np.flatnonzero(np.isfinite(R))
    nval = max(1, int(round(len(obs) * args.validation_fraction)))
    held = rng.choice(obs, size=nval, replace=False)
    val_mask = np.zeros(R.size, dtype=bool); val_mask[held] = True; val_mask = val_mask.reshape(R.shape)
    val = np.full_like(R, np.nan); val[val_mask] = R[val_mask]
    inner = R.copy(); inner[val_mask] = np.nan

    sigmas, overlaps, weights = floats(args.sigmas), ints(args.min_overlaps), floats(args.pd_weights)
    if not sigmas or not overlaps or not weights: raise ValueError('Hyperparameter lists cannot be empty')
    if any(not 0 <= w <= 1 for w in weights): raise ValueError('pd-weights must be in [0,1]')
    rows = []
    for sigma in sigmas:
        for overlap in overlaps:
            # PD-only does not depend on pd_weight, so evaluate it once.
            value = score(inner, val, genres, 'pd', sigma, overlap, 1.0, args.bias_reg)
            row = {'split':'validation','method':'pd','sigma':sigma,'min_overlap':overlap,
                   'pd_weight':1.0,'rmse':value}
            rows.append(row)
            print(f"VALIDATION | {'pd':6s} sigma={sigma:g} min_overlap={overlap} RMSE={value:.5f}")
            for weight in weights:
                value = score(inner, val, genres, 'hybrid', sigma, overlap, weight, args.bias_reg)
                row = {'split':'validation','method':'hybrid','sigma':sigma,'min_overlap':overlap,
                       'pd_weight':weight,'rmse':value}
                rows.append(row)
                print(f"VALIDATION | {'hybrid':6s} sigma={sigma:g} min_overlap={overlap} "
                      f"pd_weight={weight:.2f} RMSE={value:.5f}")
    base_score = score(inner, val, genres, 'baseline', 0, 0, 0, args.bias_reg)
    rows.append({'split':'validation','method':'baseline','sigma':'','min_overlap':'','pd_weight':'','rmse':base_score})
    print(f'VALIDATION | baseline RMSE={base_score:.5f}')
    candidates = [r for r in rows if r['split']=='validation' and r['method'] in ('pd','hybrid')]
    best = min(candidates, key=lambda x: x['rmse'])
    print('\nBest validation configuration:', best)

    if not args.skip_test:
        test_score = score(R, test, genres, best['method'], float(best['sigma']),
                           int(best['min_overlap']), float(best['pd_weight']), args.bias_reg)
        rows.append({'split':'test','method':best['method'],'sigma':best['sigma'],
                     'min_overlap':best['min_overlap'],'pd_weight':best['pd_weight'],'rmse':test_score})
        print(f"FINAL TEST | {best['method']} sigma={best['sigma']} min_overlap={best['min_overlap']} "
              f"pd_weight={best['pd_weight']} RMSE={test_score:.5f}")
        print('Do not retune on test: it is only for the final comparison.')

    with open(args.results, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['split','method','sigma','min_overlap','pd_weight','rmse'])
        writer.writeheader(); writer.writerows(rows)
    print(f'\nResults saved to: {os.path.abspath(args.results)}')
    print(f'Genres parsed ({len(vocab)}): {", ".join(vocab[:20])}')

if __name__ == '__main__': main()
