"""Experimental hybrid of Personality Diagnosis (PD), movie genres and a bias baseline.

Internal ratings matrix convention: (n_items, n_users), missing values are NaN.
The evaluator transposes the official users x movies matrices internally.
This is an experimental hybrid, not an exact implementation of the cited AAAI random-walk method.
"""
import re
import numpy as np


def rmse(actual, predicted):
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2)))


def _split_genres(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set, np.ndarray)):
        out = []
        for x in value:
            out.extend(_split_genres(x))
        return sorted(set(out))
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "unknown", "[]"}:
        return []
    s = s.strip("[](){}")
    return sorted(set(p.strip().strip("'\" ") for p in re.split(r"[|,;/]+", s)
                      if p.strip().strip("'\" ").lower() not in {"", "nan", "none", "unknown"}))


def load_genres(path, n_items):
    """Parse genre metadata; rows must match corrected rating-column order (movieId)."""
    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.shape == ():
        raw = raw.item()
    if isinstance(raw, dict):
        key = next((k for k in raw if str(k).lower() in
                    {"genre", "genres", "genres_by_item", "movie_genres"}), None)
        if key is None:
            raise ValueError(f"No genre key found in {path}; keys={list(raw)[:20]}")
        rows = list(raw[key])
    elif isinstance(raw, np.ndarray) and raw.dtype.names:
        fields = {n.lower(): n for n in raw.dtype.names}
        key = next((fields[k] for k in ("genres", "genre", "movie_genres") if k in fields), None)
        if key is None:
            raise ValueError(f"No genre field found; fields={raw.dtype.names}")
        rows = list(raw[key])
    else:
        arr = np.asarray(raw, dtype=object)
        if arr.ndim == 1:
            rows = list(arr)
        elif arr.ndim == 2 and arr.shape[0] == n_items:
            rows = list(arr[:, -1])
        elif arr.ndim == 2 and arr.shape[1] == n_items:
            # A transposed array: select the row with most parseable genre entries.
            rows = max((list(arr[i, :]) for i in range(arr.shape[0])),
                       key=lambda r: sum(bool(_split_genres(x)) for x in r))
        else:
            raise ValueError(f"Unsupported shape {arr.shape}; expected one row per item ({n_items}).")
    if len(rows) != n_items:
        raise ValueError(f"Genre rows={len(rows)}, but ratings contain {n_items} items. Check row order/format.")
    parsed = [_split_genres(x) for x in rows]
    vocab = sorted({g for row in parsed for g in row})
    if not vocab:
        raise ValueError("No genres parsed. Inspect namesngenre.npy and adapt load_genres().")
    idx = {g: j for j, g in enumerate(vocab)}
    X = np.zeros((n_items, len(vocab)), dtype=float)
    for i, row in enumerate(parsed):
        for g in row:
            X[i, idx[g]] = 1.0
    return X, vocab


def _bias_baseline(R, reg=10.0, n_iter=5):
    mask = np.isfinite(R)
    if not mask.any():
        raise ValueError("No observed ratings in training matrix.")
    mu = float(R[mask].mean())
    ni, nu = R.shape
    bi = np.zeros(ni); bu = np.zeros(nu)
    for _ in range(n_iter):
        for i in range(ni):
            m = mask[i]
            if m.any():
                bi[i] = np.sum(R[i, m] - mu - bu[m]) / (reg + m.sum())
        for u in range(nu):
            m = mask[:, u]
            if m.any():
                bu[u] = np.sum(R[m, u] - mu - bi[m]) / (reg + m.sum())
    return mu, bi, bu, mu + bi[:, None] + bu[None, :]


def _genre_similarity(genres):
    norms = np.linalg.norm(genres, axis=1)
    denom = norms[:, None] * norms[None, :]
    sim = np.divide(genres @ genres.T, denom, out=np.zeros((len(genres), len(genres))), where=denom > 0)
    np.fill_diagonal(sim, 0.0)
    return sim


def _pd_weights(R, user, sigma, min_overlap, rating_levels):
    """Gaussian profile likelihood, uniform likelihood for candidate missing ratings."""
    rated_items = np.flatnonzero(np.isfinite(R[:, user]))
    w = np.zeros(R.shape[1], dtype=float)
    if not len(rated_items):
        return w
    target = R[rated_items, user]
    candidate = R[rated_items, :]
    common = np.isfinite(candidate)
    overlap = common.sum(axis=0)
    sqerr = np.where(common, (candidate - target[:, None]) ** 2, 0.0).sum(axis=0)
    nc = overlap.astype(float)
    nm = len(rated_items) - nc
    sigma = max(float(sigma), 1e-8)
    logp = -0.5 * nc * np.log(2 * np.pi * sigma * sigma) - sqerr / (2 * sigma * sigma) \
           - nm * np.log(max(float(rating_levels), 1.0))
    valid = overlap >= int(min_overlap)
    valid[user] = False
    if not valid.any():
        return w
    logp[~valid] = -np.inf
    finite = np.isfinite(logp)
    if not finite.any():
        return w
    logp[finite] -= np.max(logp[finite])
    w[finite] = np.exp(np.clip(logp[finite], -700, 0))
    if w.sum() > 0:
        w /= w.sum()
    return w


def predict_entries(train, item_idx, user_idx, genres, sigma=5.0, min_overlap=2,
                    pd_weight=0.5, bias_reg=10.0, min_genre_sim=0.0):
    """Predict selected entries. Blend residuals from PD and genre similarity."""
    R = np.asarray(train, dtype=float)
    item_idx = np.asarray(item_idx, dtype=int); user_idx = np.asarray(user_idx, dtype=int)
    genres = np.asarray(genres, dtype=float)
    if R.ndim != 2 or len(item_idx) != len(user_idx):
        raise ValueError("Expected a 2-D item x user matrix and equally sized index arrays.")
    if genres.shape[0] != R.shape[0]:
        raise ValueError(f"Genre rows {genres.shape[0]} != item count {R.shape[0]}.")
    if not 0 <= pd_weight <= 1:
        raise ValueError("pd_weight must be between 0 and 1.")
    _, _, _, base = _bias_baseline(R, reg=bias_reg)
    mask = np.isfinite(R)
    residual = np.where(mask, R - base, 0.0)
    sim = _genre_similarity(genres)
    vals = R[mask]
    levels = max(float(vals.max() - vals.min() + 1), 1.0)
    preds = np.empty(len(item_idx), dtype=float)

    for u in np.unique(user_idx):
        pos = np.flatnonzero(user_idx == u)
        items = item_idx[pos]
        w = _pd_weights(R, int(u), sigma, min_overlap, levels)
        avail = mask[items, :]
        weighted_avail = avail * w[None, :]
        den_pd = weighted_avail.sum(axis=1)
        num_pd = (weighted_avail * residual[items, :]).sum(axis=1)
        pd_resid = np.divide(num_pd, den_pd, out=np.zeros_like(num_pd), where=den_pd > 1e-15)
        has_pd = den_pd > 1e-15

        rated = np.flatnonzero(mask[:, int(u)])
        gen_resid = np.zeros(len(items)); has_gen = np.zeros(len(items), dtype=bool)
        if len(rated):
            sg = sim[np.ix_(items, rated)].copy()
            sg[sg < min_genre_sim] = 0.0
            dg = sg.sum(axis=1)
            ng = sg @ residual[rated, int(u)]
            gen_resid = np.divide(ng, dg, out=np.zeros_like(ng), where=dg > 1e-15)
            has_gen = dg > 1e-15

        combined = np.zeros(len(items))
        both = has_pd & has_gen
        only_pd = has_pd & ~has_gen
        only_gen = ~has_pd & has_gen
        combined[both] = pd_weight * pd_resid[both] + (1 - pd_weight) * gen_resid[both]
        combined[only_pd] = pd_resid[only_pd]
        combined[only_gen] = gen_resid[only_gen]
        preds[pos] = base[items, int(u)] + combined

    return np.clip(preds, vals.min(), vals.max())


def predict_full_matrix(train, genres, sigma=5.0, min_overlap=2, pd_weight=0.5, bias_reg=10.0):
    """Fill missing cells and preserve all observed ratings."""
    R = np.asarray(train, dtype=float).copy()
    ii, uu = np.where(~np.isfinite(R))
    if len(ii):
        R[ii, uu] = predict_entries(train, ii, uu, genres, sigma=sigma,
                                    min_overlap=min_overlap, pd_weight=pd_weight,
                                    bias_reg=bias_reg)
    return R
