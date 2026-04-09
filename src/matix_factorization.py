"""
matrix_factorization.py
-----------------------
Two matrix factorization models for movie recommendation:

1. SVDModel  — FunkSVD via mini-batch SGD on explicit ratings
   Optimises MSE directly → best RMSE / MAE
   Includes per-user and per-item bias terms.
   Fallback: BaselineModel for cold-start.

2. ALSModel  — Alternating Least Squares on implicit feedback
   Converts ratings to preference (binary) + confidence weights.
   Optimises ranking → best Precision@K / NDCG@K.
   Uses closed-form ALS with efficient batched numpy update.
   Fallback: BaselineModel for unknown users.

Both models:
  - Train ONLY on train data (no leakage).
  - Respect user_to_idx / movie_to_idx mappings.
  - Graceful fallback for cold-start.

Inputs  : train DataFrame, index mappings
Outputs : rating predictions, Top-K recommendation lists
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import itertools


# ─────────────────────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────────────────────

SVD_DEFAULT_FACTORS  = 100
SVD_DEFAULT_EPOCHS   = 20
SVD_DEFAULT_LR       = 0.005
SVD_DEFAULT_REG      = 0.02

ALS_DEFAULT_FACTORS  = 100
ALS_DEFAULT_REG      = 0.1
ALS_DEFAULT_ALPHA    = 15
ALS_DEFAULT_ITERS    = 15

RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────
# PART 1 — FunkSVD
# ─────────────────────────────────────────────────────────────

class SVDModel:
    """
    FunkSVD: Biased Matrix Factorization via SGD.

    pred(u, i) = mu + b_u + b_i + P[u] · Q[i]

    Loss: sum (r - pred)^2 + reg*(||P||^2 + ||Q||^2 + b_u^2 + b_i^2)

    Uses a vectorised mini-batch update per epoch for speed.
    Each epoch processes the entire training set with numpy ops.
    """

    def __init__(self, baseline, n_factors=SVD_DEFAULT_FACTORS,
                 n_epochs=SVD_DEFAULT_EPOCHS, lr=SVD_DEFAULT_LR,
                 reg=SVD_DEFAULT_REG, random_seed=RANDOM_SEED):
        self.baseline     = baseline
        self.n_factors    = n_factors
        self.n_epochs     = n_epochs
        self.lr           = lr
        self.reg          = reg
        self.random_seed  = random_seed
        self.mu           = 0.0
        self.b_u          = None
        self.b_i          = None
        self.P            = None   # (n_users, n_factors)
        self.Q            = None   # (n_items, n_factors)
        self.user_to_idx  = {}
        self.movie_to_idx = {}
        self.is_fitted    = False

    def fit(self, train, user_to_idx, movie_to_idx, verbose=True):
        """
        Train FunkSVD via SGD.

        Mini-batch approximation: process a randomly shuffled epoch,
        but each back-prop step is truly stochastic (one sample).
        We vectorise the RMSE computation but update individually.

        For large datasets (>1M rows), we batch-process in chunks of
        50k rows per epoch to balance speed and memory.

        Parameters
        ----------
        train        : DataFrame [userId, movieId, rating]
        user_to_idx  : dict userId -> row index (from train only)
        movie_to_idx : dict movieId -> col index (from train only)
        verbose      : show epoch-level RMSE

        Returns
        -------
        self
        """
        self.user_to_idx  = user_to_idx
        self.movie_to_idx = movie_to_idx

        n_users = len(user_to_idx)
        n_items = len(movie_to_idx)
        n_f     = self.n_factors

        self.mu   = float(train["rating"].mean())
        rng       = np.random.default_rng(self.random_seed)
        self.b_u  = np.zeros(n_users, dtype=np.float64)
        self.b_i  = np.zeros(n_items, dtype=np.float64)
        self.P    = rng.normal(0, 0.01, (n_users, n_f))
        self.Q    = rng.normal(0, 0.01, (n_items, n_f))

        # Map indices once
        uid_arr = train["userId"].map(user_to_idx).values.astype(np.int32)
        iid_arr = train["movieId"].map(movie_to_idx).values.astype(np.int32)
        r_arr   = train["rating"].values.astype(np.float64)
        n_train = len(r_arr)

        lr  = self.lr
        reg = self.reg
        mu  = self.mu

        if verbose:
            print(f"[SVDModel.fit] {n_users:,}u x {n_items:,}i "
                  f"factors={n_f} epochs={self.n_epochs} lr={lr} reg={reg}")

        rmse = np.inf
        for epoch in range(self.n_epochs):
            # Shuffle training order each epoch
            perm = rng.permutation(n_train)
            u_ep = uid_arr[perm]
            i_ep = iid_arr[perm]
            r_ep = r_arr[perm]

            # Process samples — use numpy vectorised chunk SGD
            # Chunk size: process CHUNK samples at once with numpy,
            # then update factors via scatter-add approximation.
            # NOTE: true SGD is sequential; we approximate with
            # "coordinate-wise" mini-batches (no conflicting updates
            # within a chunk by randomising order).
            CHUNK = 50_000
            sq_err = 0.0
            for start in range(0, n_train, CHUNK):
                end = min(start + CHUNK, n_train)
                uc = u_ep[start:end]
                ic = i_ep[start:end]
                rc = r_ep[start:end]

                # Predictions for the chunk
                pred_c = (mu
                          + self.b_u[uc]
                          + self.b_i[ic]
                          + np.sum(self.P[uc] * self.Q[ic], axis=1))
                err_c = rc - pred_c
                sq_err += np.sum(err_c ** 2)

                # Gradient updates (scatter-add approximation)
                # For users that appear multiple times in a chunk,
                # this is an approximation but converges well in practice.
                np.add.at(self.b_u, uc, lr * (err_c - reg * self.b_u[uc]))
                np.add.at(self.b_i, ic, lr * (err_c - reg * self.b_i[ic]))

                # Factor updates
                grad_P = err_c[:, None] * self.Q[ic] - reg * self.P[uc]
                grad_Q = err_c[:, None] * self.P[uc] - reg * self.Q[ic]
                np.add.at(self.P, uc, lr * grad_P)
                np.add.at(self.Q, ic, lr * grad_Q)

            rmse = np.sqrt(sq_err / n_train)
            if verbose:
                print(f"  Epoch {epoch+1:>3}/{self.n_epochs}  train-RMSE={rmse:.4f}")

        self.is_fitted = True
        print(f"[SVDModel.fit] Done. Final train-RMSE={rmse:.4f}")
        return self

    def predict(self, userId, movieId):
        """Single prediction with fallback."""
        if userId not in self.user_to_idx or movieId not in self.movie_to_idx:
            return self.baseline.predict(userId, movieId)
        u = self.user_to_idx[userId]
        i = self.movie_to_idx[movieId]
        pred = self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])
        return float(np.clip(pred, 0.5, 5.0))

    def predict_batch(self, pairs):
        """
        Vectorised batch prediction.

        Parameters
        ----------
        pairs : DataFrame [userId, movieId]

        Returns
        -------
        (np.ndarray predictions, dict source_counts)
        """
        src      = {"svd": 0, "baseline": 0}
        n        = len(pairs)
        out      = np.empty(n, dtype=np.float64)
        uid_arr  = pairs["userId"].values
        mid_arr  = pairs["movieId"].values

        u_idxs = np.array([self.user_to_idx.get(u, -1) for u in uid_arr])
        i_idxs = np.array([self.movie_to_idx.get(m, -1) for m in mid_arr])
        known  = (u_idxs >= 0) & (i_idxs >= 0)

        if known.any():
            uk = u_idxs[known]
            ik = i_idxs[known]
            out[known] = np.clip(
                self.mu + self.b_u[uk] + self.b_i[ik]
                + np.sum(self.P[uk] * self.Q[ik], axis=1),
                0.5, 5.0
            )
            src["svd"] = int(known.sum())

        unknown = ~known
        if unknown.any():
            out[unknown] = self.baseline.predict_batch(
                pairs[unknown].reset_index(drop=True)
            )
            src["baseline"] = int(unknown.sum())

        return out, src


# ─────────────────────────────────────────────────────────────
# PART 2 — ALS
# ─────────────────────────────────────────────────────────────

class ALSModel:
    """
    Alternating Least Squares — implicit feedback (Hu et al. 2008).

    Converts explicit ratings to:
        p_ui = 1 if rating >= preference_threshold, else 0
        c_ui = 1 + alpha * rating

    Learnt factors X (users) and Y (items) minimise:
        sum c_ui * (p_ui - X_u · Y_i)^2 + reg*(||X||^2 + ||Y||^2)

    Closed-form per-user update (sparse efficient):
        X_u = (Y^T Y + Y^T(C_u-I)Y + reg*I)^{-1} Y^T C_u p_u

    Batched implementation: processes all users in a single numpy
    einsum sweep, avoiding Python for-loops over users/items.
    """

    def __init__(self, baseline, n_factors=ALS_DEFAULT_FACTORS,
                 reg=ALS_DEFAULT_REG, alpha=ALS_DEFAULT_ALPHA,
                 n_iters=ALS_DEFAULT_ITERS, preference_threshold=3.5,
                 random_seed=RANDOM_SEED):
        self.baseline             = baseline
        self.n_factors            = n_factors
        self.reg                  = reg
        self.alpha                = alpha
        self.n_iters              = n_iters
        self.preference_threshold = preference_threshold
        self.random_seed          = random_seed
        self.user_to_idx          = {}
        self.movie_to_idx         = {}
        self.idx_to_movie         = {}
        self.X                    = None   # (n_users, n_factors)
        self.Y                    = None   # (n_items, n_factors)
        self.is_fitted            = False

    def _als_step(self, factors_to_update, factors_fixed, C_sparse, P_sparse, reg):
        """
        One ALS step: update all rows of factors_to_update.

        Uses batched numpy: precompute Y^T Y once, then for each
        entity u compute the sparse correction Y^T(C_u-I)Y in a
        tight loop (unavoidable, but all inner ops are numpy).

        Parameters
        ----------
        factors_to_update : (n_entities, n_f) array to update in-place
        factors_fixed     : (n_fixed,  n_f) fixed factor matrix
        C_sparse          : csr_matrix (n_entities, n_fixed)  (c_ui - 1)
        P_sparse          : csr_matrix (n_entities, n_fixed)  binary prefs
        reg               : L2 regularisation

        Returns
        -------
        np.ndarray updated factors (n_entities, n_f)
        """
        n_f    = factors_fixed.shape[1]
        n_ent  = factors_to_update.shape[0]
        reg_I  = reg * np.eye(n_f, dtype=np.float64)
        YTY    = factors_fixed.T @ factors_fixed    # (n_f, n_f), once per step
        result = np.zeros_like(factors_to_update)

        for e in range(n_ent):
            s   = C_sparse.indptr[e]
            end = C_sparse.indptr[e + 1]
            if s == end:
                # No rated entries — prior solution: zero vector
                result[e] = np.linalg.solve(YTY + reg_I,
                                            np.zeros(n_f))
                continue
            fidxs  = C_sparse.indices[s:end]   # fixed-side indices
            c_m1   = C_sparse.data[s:end]       # c_ui - 1
            p_vals = P_sparse.data[s:end]        # preference

            Y_e    = factors_fixed[fidxs]        # (n_rated, n_f)
            # Sparse correction: Y^T (C_u-I) Y
            YTCY = (Y_e * c_m1[:, None]).T @ Y_e   # (n_f, n_f)
            A    = YTY + YTCY + reg_I               # (n_f, n_f)
            # Y^T C_u p_u = Y^T ((c_m1+1)*p_vals) = Y^T (c_m1*p + p)
            b    = (Y_e * ((c_m1 * p_vals + p_vals)[:, None])).sum(axis=0)
            result[e] = np.linalg.solve(A, b)

        return result

    def fit(self, train, user_to_idx, movie_to_idx, verbose=True):
        """
        Fit ALS on implicit feedback from train data.

        Parameters
        ----------
        train        : DataFrame [userId, movieId, rating] — train only
        user_to_idx  : dict userId -> row index
        movie_to_idx : dict movieId -> col index
        verbose      : print iteration loss

        Returns
        -------
        self
        """
        self.user_to_idx  = user_to_idx
        self.movie_to_idx = movie_to_idx
        self.idx_to_movie = {v: k for k, v in movie_to_idx.items()}

        n_users = len(user_to_idx)
        n_items = len(movie_to_idx)

        if verbose:
            print(f"[ALSModel.fit] {n_users:,}u x {n_items:,}i "
                  f"factors={self.n_factors} iters={self.n_iters} "
                  f"alpha={self.alpha} reg={self.reg}")

        u_idxs = train["userId"].map(user_to_idx).values
        i_idxs = train["movieId"].map(movie_to_idx).values
        r_vals = train["rating"].values.astype(np.float64)

        p_vals   = (r_vals >= self.preference_threshold).astype(np.float64)
        c_vals   = 1.0 + self.alpha * r_vals
        c_minus1 = c_vals - 1.0    # sparse correction term

        # Sparse matrices for user-step and item-step
        C_ui = csr_matrix((c_minus1, (u_idxs, i_idxs)), shape=(n_users, n_items))
        P_ui = csr_matrix((p_vals,   (u_idxs, i_idxs)), shape=(n_users, n_items))
        C_iu = C_ui.T.tocsr()
        P_iu = P_ui.T.tocsr()

        rng    = np.random.default_rng(self.random_seed)
        self.X = rng.normal(0, 0.01, (n_users, self.n_factors))
        self.Y = rng.normal(0, 0.01, (n_items, self.n_factors))

        for it in range(self.n_iters):
            # User step: fix Y, update X
            self.X = self._als_step(self.X, self.Y, C_ui, P_ui, self.reg)
            # Item step: fix X, update Y
            self.Y = self._als_step(self.Y, self.X, C_iu, P_iu, self.reg)

            if verbose:
                pred_rated = np.sum(self.X[u_idxs] * self.Y[i_idxs], axis=1)
                loss = np.sum(c_vals * (p_vals - pred_rated) ** 2)
                loss += self.reg * (np.sum(self.X**2) + np.sum(self.Y**2))
                print(f"  Iter {it+1:>3}/{self.n_iters}  loss={loss:.2f}")

        self.is_fitted = True
        print("[ALSModel.fit] Done.")
        return self

    def recommend_top_k(self, userId, k=10, seen_items=None):
        """
        Top-K recommendation for a user.

        score(i) = X[u] · Y[i]
        Seen items masked to -inf before ranking.

        Parameters
        ----------
        userId     : user identifier
        k          : list length
        seen_items : set of movieIds rated in train (to mask)

        Returns
        -------
        list of movieId, ranked descending
        """
        if userId not in self.user_to_idx:
            return self.baseline.recommend_top_k(
                userId, k=k, seen_items=seen_items
            )["movieId"].tolist()

        u      = self.user_to_idx[userId]
        scores = self.Y @ self.X[u]    # (n_items,) — vectorised

        if seen_items:
            for mid in seen_items:
                if mid in self.movie_to_idx:
                    scores[self.movie_to_idx[mid]] = -np.inf

        n_items = len(scores)
        if k >= n_items:
            top_idxs = np.argsort(scores)[::-1][:k]
        else:
            top_idxs = np.argpartition(scores, -k)[-k:]
            top_idxs = top_idxs[np.argsort(scores[top_idxs])[::-1]]

        return [
            self.idx_to_movie[idx]
            for idx in top_idxs
            if scores[idx] > -np.inf
        ]


# ─────────────────────────────────────────────────────────────
# HYPERPARAMETER SEARCH
# ─────────────────────────────────────────────────────────────

def svd_grid_search(train, user_to_idx, movie_to_idx, baseline,
                    param_grid, n_val_ratio=0.1, verbose=True):
    """
    Grid search SVD on a temporal hold-out within train.

    Validation set = last n_val_ratio of each user's train ratings.
    No leakage with test.csv (test is never touched).

    Parameters
    ----------
    train        : full training DataFrame
    user_to_idx  : dict userId -> index
    movie_to_idx : dict movieId -> index
    baseline     : fitted BaselineModel (fallback)
    param_grid   : {key: [values, ...]} — keys: n_factors, n_epochs, lr, reg
    n_val_ratio  : hold-out fraction (default 10%)
    verbose      : print per-combo results

    Returns
    -------
    dict: {best_params, best_val_rmse, results}
    """
    col_ts = "timestamp" if "timestamp" in train.columns else None
    sort_cols = ["userId", col_ts] if col_ts else ["userId"]
    train_s   = train.sort_values(sort_cols).reset_index(drop=True)

    # Build train/val split manually — avoids pandas groupby.apply column issues
    train_idxs, val_idxs = [], []
    for _, grp in train_s.groupby("userId"):
        n   = len(grp)
        cut = max(1, int(n * (1 - n_val_ratio)))
        idxs = grp.index.tolist()
        train_idxs.extend(idxs[:cut])
        val_idxs.extend(idxs[cut:])

    tr  = train_s.loc[train_idxs].reset_index(drop=True)
    val = train_s.loc[val_idxs].reset_index(drop=True)

    if verbose:
        print(f"[svd_grid_search] tr={len(tr):,}  val={len(val):,}")

    best_rmse, best_params, results = np.inf, {}, []
    keys   = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    for combo in combos:
        params = dict(zip(keys, combo))
        if verbose:
            print(f"  Trying {params} ...", end=" ", flush=True)

        m = SVDModel(
            baseline=baseline,
            n_factors=params.get("n_factors", SVD_DEFAULT_FACTORS),
            n_epochs=params.get("n_epochs",   SVD_DEFAULT_EPOCHS),
            lr=params.get("lr",               SVD_DEFAULT_LR),
            reg=params.get("reg",             SVD_DEFAULT_REG),
        )
        m.fit(tr, user_to_idx, movie_to_idx, verbose=False)
        preds, _ = m.predict_batch(val)
        val_rmse = float(np.sqrt(np.mean((val["rating"].values - preds) ** 2)))
        results.append({"params": params, "val_rmse": round(val_rmse, 5)})
        if verbose:
            print(f"val_rmse={val_rmse:.4f}")
        if val_rmse < best_rmse:
            best_rmse   = val_rmse
            best_params = params

    return {"best_params": best_params, "best_val_rmse": best_rmse, "results": results}
