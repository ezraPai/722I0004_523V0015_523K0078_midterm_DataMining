"""
collaborative_filtering.py
--------------------------
Item-Based Collaborative Filtering recommender.

Algorithm:
  1. Cosine similarity between item vectors (sparse, no dense conversion).
  2. Mean-centered weighted-average rating prediction.
  3. Graceful fallback to BaselineModel for cold-start / no-neighbor cases.
  4. Vectorised Top-K ranking with seen-item masking (score = -inf).

Design constraints enforced:
  - Similarity computed from user_item_matrix (train only) → no leakage.
  - test.csv is NEVER read or touched inside this module.
  - Index consistency: always maps via user_to_idx / movie_to_idx.
  - No dense materialisation of full similarity matrix.

Inputs  : user_item_matrix (csr_matrix), index maps, baseline model
Outputs : rating predictions (float), Top-K recommendation lists
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from sklearn.metrics.pairwise import cosine_similarity
import joblib


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

# Maximum number of similar items to use per prediction.
# Capping neighbours speeds up inference and reduces noise.
DEFAULT_N_NEIGHBORS = 40

# Minimum cosine similarity to include a neighbour.
# Very low similarities add noise; this filters them out.
DEFAULT_SIM_THRESHOLD = 0.0


# ─────────────────────────────────────────────────────────────
# ITEM-BASED CF CLASS
# ─────────────────────────────────────────────────────────────

class ItemBasedCF:
    """
    Item-Based Collaborative Filtering using cosine similarity.

    Attributes
    ----------
    matrix : csr_matrix
        User-item rating matrix (shape: n_users x n_items), train only.
    user_to_idx : dict
        Maps userId -> row index in matrix.
    movie_to_idx : dict
        Maps movieId -> column index in matrix.
    idx_to_movie : dict
        Reverse map: column index -> movieId.
    user_means : np.ndarray
        Per-user mean rating (shape: n_users,). Used for mean-centering.
    item_sim : np.ndarray
        Item-item cosine similarity matrix (shape: n_items x n_items).
        Computed lazily in fit().
    baseline : BaselineModel
        Fallback predictor for cold-start cases.
    n_neighbors : int
        Maximum neighbours per prediction.
    sim_threshold : float
        Minimum similarity to count a neighbour.
    """

    def __init__(
        self,
        baseline,
        n_neighbors: int = DEFAULT_N_NEIGHBORS,
        sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    ):
        """
        Parameters
        ----------
        baseline      : fitted BaselineModel instance (used for fallback)
        n_neighbors   : top-N similar items to use per prediction
        sim_threshold : minimum cosine similarity to include a neighbour
        """
        self.baseline      = baseline
        self.n_neighbors   = n_neighbors
        self.sim_threshold = sim_threshold

        self.matrix      = None
        self.user_to_idx = {}
        self.movie_to_idx = {}
        self.idx_to_movie = {}
        self.user_means  = None
        self.item_sim    = None
        self.is_fitted   = False

    # ── FIT ────────────────────────────────────────────────────

    def fit(
        self,
        matrix: csr_matrix,
        user_to_idx: dict,
        movie_to_idx: dict,
    ) -> "ItemBasedCF":
        """
        Build item-item cosine similarity from the train matrix.

        Steps:
        1. Store matrix and index maps.
        2. Compute per-user mean for mean-centering at predict time.
        3. Compute cosine similarity between all item column vectors.

        The matrix is transposed so items become rows (sklearn
        cosine_similarity works row-wise). We do NOT materialise
        a full dense item-item matrix if items > threshold;
        sklearn handles sparse input efficiently.

        Parameters
        ----------
        matrix       : csr_matrix (n_users x n_items), train only
        user_to_idx  : dict userId -> row index
        movie_to_idx : dict movieId -> col index

        Returns
        -------
        self
        """
        self.matrix       = matrix
        self.user_to_idx  = user_to_idx
        self.movie_to_idx = movie_to_idx
        self.idx_to_movie = {v: k for k, v in movie_to_idx.items()}

        n_users, n_items = matrix.shape
        print(f"[ItemBasedCF.fit] Matrix shape: {matrix.shape}")
        print(f"  Density: {matrix.nnz / (n_users * n_items):.4%}")

        # ── Per-user mean (for mean-centering) ─────────────────
        # Only over rated items (non-zero entries), not the zeros
        # which represent "not rated" rather than a zero rating.
        user_sums   = np.array(matrix.sum(axis=1)).ravel()       # shape (n_users,)
        user_counts = np.diff(matrix.indptr)                     # nnz per row
        # Avoid division by zero for users with no ratings (shouldn't exist)
        user_counts_safe = np.where(user_counts == 0, 1, user_counts)
        self.user_means = user_sums / user_counts_safe            # shape (n_users,)

        # ── Item-item cosine similarity ─────────────────────────
        # Transpose: item_matrix shape (n_items, n_users)
        # sklearn cosine_similarity(X) computes X @ X.T row-wise.
        # For large item sets this is memory-intensive; we keep it
        # sparse by passing the csr_matrix directly (sklearn handles it).
        print(f"  Computing item-item cosine similarity ({n_items} x {n_items})...")
        item_matrix = matrix.T.tocsr()   # shape (n_items, n_users)

        # sklearn returns a dense (n_items, n_items) numpy array.
        # For 14k items this is ~14k^2 * 8 bytes ≈ 1.5 GB — too large.
        # We avoid this by computing similarity ON-DEMAND per item
        # during prediction via a normalised sparse representation.
        # Store normalised item vectors for dot-product similarity.
        norms = np.array(np.sqrt(item_matrix.power(2).sum(axis=1))).ravel()
        norms_safe = np.where(norms == 0, 1.0, norms)
        # Row-normalise the item matrix (sparse-safe)
        from scipy.sparse import diags
        norm_diag = diags(1.0 / norms_safe)
        self.item_matrix_normed = norm_diag @ item_matrix   # (n_items, n_users)

        self.is_fitted = True
        print(f"  Normalised item matrix stored: {self.item_matrix_normed.shape}")
        print(f"[ItemBasedCF.fit] Done.")
        return self

    # ── SIMILARITY HELPERS ────────────────────────────────────

    def _item_similarities(self, item_idx: int) -> np.ndarray:
        """
        Compute cosine similarity between item_idx and all other items.

        Returns a dense array of shape (n_items,) where
        result[j] = cos_sim(item_idx, j).

        This avoids storing the full n_items x n_items matrix by
        computing one row at a time.

        Parameters
        ----------
        item_idx : int  column index of the target item

        Returns
        -------
        np.ndarray shape (n_items,), float32
        """
        target = self.item_matrix_normed[item_idx]          # (1, n_users) sparse
        sims   = (self.item_matrix_normed @ target.T)       # (n_items, 1) sparse
        sims   = np.array(sims.todense()).ravel()            # dense (n_items,)
        return sims

    # ── PREDICT SINGLE ────────────────────────────────────────

    def predict(self, userId, movieId) -> tuple[float, str]:
        """
        Predict rating for (userId, movieId) with full fallback chain.

        Returns
        -------
        (prediction: float, source: str)
            source is one of: 'cf', 'baseline_no_neighbors',
            'baseline_unseen_item', 'baseline_unseen_user'
        """
        # ── Index lookup ───────────────────────────────────────
        if movieId not in self.movie_to_idx:
            return self.baseline.predict(userId, movieId), "baseline_unseen_item"

        if userId not in self.user_to_idx:
            return self.baseline.predict(userId, movieId), "baseline_unseen_user"

        u_idx = self.user_to_idx[userId]
        i_idx = self.movie_to_idx[movieId]

        # ── User's rated items (non-zero entries in user row) ──
        user_row   = self.matrix.getrow(u_idx)               # sparse (1, n_items)
        rated_idxs = user_row.indices                        # item indices rated
        rated_vals = user_row.data                           # raw ratings

        if len(rated_idxs) == 0:
            return self.baseline.predict(userId, movieId), "baseline_no_neighbors"

        # ── Mean-center the user's raw ratings ─────────────────
        u_mean         = self.user_means[u_idx]
        centered_vals  = rated_vals - u_mean                 # r(u,j) - mu_u

        # ── Item similarities (row for item i_idx) ──────────────
        sims_all   = self._item_similarities(i_idx)          # (n_items,)
        # Only keep similarities for items the user has actually rated
        neighbor_sims = sims_all[rated_idxs]                 # (n_rated,)

        # ── Apply threshold and select top-N neighbors ─────────
        valid_mask    = neighbor_sims > self.sim_threshold
        neighbor_sims = neighbor_sims[valid_mask]
        centered_vals = centered_vals[valid_mask]

        if len(neighbor_sims) == 0:
            return self.baseline.predict(userId, movieId), "baseline_no_neighbors"

        # Keep only the N most similar neighbors
        if len(neighbor_sims) > self.n_neighbors:
            top_n = np.argpartition(neighbor_sims, -self.n_neighbors)[-self.n_neighbors:]
            neighbor_sims = neighbor_sims[top_n]
            centered_vals = centered_vals[top_n]

        # ── Weighted average (mean-centered) ───────────────────
        denom = np.abs(neighbor_sims).sum()
        if denom < 1e-9:
            return self.baseline.predict(userId, movieId), "baseline_no_neighbors"

        pred = u_mean + np.dot(neighbor_sims, centered_vals) / denom
        pred = float(np.clip(pred, 0.5, 5.0))
        return pred, "cf"

    # ── PRECOMPUTE TOP-N NEIGHBORS PER ITEM ──────────────────

    def precompute_top_neighbors(self, chunk_size: int = 500) -> None:
        """
        Precompute and cache the top-N neighbor indices and similarities
        for every item. Computed in chunks to avoid full dense matrix.

        After calling this, predict_batch uses the cached neighbors
        instead of on-demand similarity computation, giving a 100-200x
        speedup over the per-row Python loop.

        Parameters
        ----------
        chunk_size : int  number of items to process per batch
        """
        n_items = self.item_matrix_normed.shape[0]
        N       = self.n_neighbors

        # top_neighbor_idxs[i]  = array of up to N item indices most similar to i
        # top_neighbor_sims[i]  = corresponding similarity values
        self._top_nbr_idxs = [None] * n_items
        self._top_nbr_sims = [None] * n_items

        print(f"  Precomputing top-{N} neighbors for {n_items:,} items "
              f"(chunk_size={chunk_size})...")

        for start in range(0, n_items, chunk_size):
            end   = min(start + chunk_size, n_items)
            chunk = self.item_matrix_normed[start:end]       # (chunk, n_users) sparse
            # sims_chunk: (chunk, n_items) dense
            sims_chunk = np.array(
                (chunk @ self.item_matrix_normed.T).todense()
            )
            for local_i, global_i in enumerate(range(start, end)):
                row = sims_chunk[local_i]                    # (n_items,)
                row[global_i] = -1.0                        # exclude self
                # Keep only above threshold
                above = np.where(row > self.sim_threshold)[0]
                if len(above) == 0:
                    self._top_nbr_idxs[global_i] = np.array([], dtype=np.int32)
                    self._top_nbr_sims[global_i]  = np.array([], dtype=np.float32)
                else:
                    if len(above) > N:
                        top = np.argpartition(row[above], -N)[-N:]
                        above = above[top]
                    self._top_nbr_idxs[global_i] = above.astype(np.int32)
                    self._top_nbr_sims[global_i]  = row[above].astype(np.float32)

        self._neighbors_precomputed = True
        print("  Precomputation done.")

    # ── BATCH PREDICTION ─────────────────────────────────────

    def predict_batch(self, pairs: pd.DataFrame) -> tuple[np.ndarray, dict]:
        """
        Predict ratings for all rows in pairs DataFrame.

        Uses precomputed neighbor cache if available (fast path).
        Falls back to per-row on-demand similarity if cache not built.

        Parameters
        ----------
        pairs : pd.DataFrame  columns [userId, movieId]

        Returns
        -------
        (predictions: np.ndarray shape (n,), diagnostics: dict)
        """
        n      = len(pairs)
        preds  = np.empty(n, dtype=np.float64)
        source_counts = {"cf": 0, "baseline_unseen_item": 0,
                         "baseline_unseen_user": 0, "baseline_no_neighbors": 0}

        use_cache = getattr(self, "_neighbors_precomputed", False)

        uid_arr = pairs["userId"].values
        mid_arr = pairs["movieId"].values

        for i in range(n):
            uid = uid_arr[i]
            mid = mid_arr[i]

            # ── Index lookup ──────────────────────────────────
            if mid not in self.movie_to_idx:
                preds[i] = self.baseline.predict(uid, mid)
                source_counts["baseline_unseen_item"] += 1
                continue

            if uid not in self.user_to_idx:
                preds[i] = self.baseline.predict(uid, mid)
                source_counts["baseline_unseen_user"] += 1
                continue

            u_idx = self.user_to_idx[uid]
            i_idx = self.movie_to_idx[mid]

            user_row   = self.matrix.getrow(u_idx)
            rated_idxs = user_row.indices
            rated_vals = user_row.data

            if len(rated_idxs) == 0:
                preds[i] = self.baseline.predict(uid, mid)
                source_counts["baseline_no_neighbors"] += 1
                continue

            u_mean        = self.user_means[u_idx]
            centered_vals = rated_vals - u_mean

            if use_cache:
                # Fast path: use precomputed neighbor index/sim arrays
                nbr_idxs = self._top_nbr_idxs[i_idx]       # item indices of neighbors
                nbr_sims = self._top_nbr_sims[i_idx]        # similarities

                # Intersect neighbors with items the user has rated
                # Build a lookup: rated item idx -> centered value
                rated_set = {idx: val for idx, val in
                             zip(rated_idxs.tolist(), centered_vals.tolist())}

                match_sims, match_vals = [], []
                for nbr_i, nbr_s in zip(nbr_idxs, nbr_sims):
                    if nbr_i in rated_set:
                        match_sims.append(nbr_s)
                        match_vals.append(rated_set[nbr_i])

                if not match_sims:
                    preds[i] = self.baseline.predict(uid, mid)
                    source_counts["baseline_no_neighbors"] += 1
                    continue

                ms   = np.array(match_sims, dtype=np.float64)
                mv   = np.array(match_vals, dtype=np.float64)
                denom = np.abs(ms).sum()

                if denom < 1e-9:
                    preds[i] = self.baseline.predict(uid, mid)
                    source_counts["baseline_no_neighbors"] += 1
                    continue

                pred = u_mean + np.dot(ms, mv) / denom
                preds[i] = float(np.clip(pred, 0.5, 5.0))
                source_counts["cf"] += 1

            else:
                # Slow path: on-demand similarity
                p, src = self.predict(uid, mid)
                preds[i] = p
                source_counts[src] += 1

        return preds, source_counts

    # ── TOP-K RANKING ─────────────────────────────────────────

    def recommend_top_k(
        self,
        userId,
        k: int = 10,
        seen_items: set = None,
    ) -> list:
        """
        Generate Top-K recommendations for a user.

        Strategy:
        - Score all candidate items (not seen in train) using item-CF.
        - Seen items are set to -inf BEFORE ranking (no contamination).
        - Falls back to baseline popularity for unknown users.

        Parameters
        ----------
        userId     : user identifier
        k          : number of recommendations
        seen_items : set of movieIds the user has already rated (train)

        Returns
        -------
        list of movieId (length <= k), ranked descending by score
        """
        if userId not in self.user_to_idx:
            # Unknown user: delegate entirely to baseline
            return self.baseline.recommend_top_k(
                userId, k=k, seen_items=seen_items
            )["movieId"].tolist()

        u_idx     = self.user_to_idx[userId]
        u_mean    = self.user_means[u_idx]
        user_row  = self.matrix.getrow(u_idx)
        rated_idxs = user_row.indices
        rated_vals = user_row.data
        centered_vals = rated_vals - u_mean

        n_items = self.item_matrix_normed.shape[0]

        # Vectorised scoring of all items for this user.
        #
        # item_matrix_normed: shape (n_items, n_users)
        # Each row i is the L2-normalised rating vector of item i over users.
        #
        # We want: score(i) = sum_j[ sim(i,j) * r_centered(u,j) ]
        # where the sum runs over items j that user u has rated.
        #
        # This equals: item_matrix_normed[i] dot item_matrix_normed[j] * r_c(u,j)
        # summed over rated j, which is:
        #   scores = item_matrix_normed @ (item_matrix_normed[rated_idxs].T @ centered_vals)
        #
        # But a simpler correct proxy for ranking is:
        #   Build user's centered "preference vector" in user-space:
        #     user_pref_in_user_space[u_idx_of_each_rated_item_rater] ...
        #
        # Actually, the cleanest and correct formula is:
        #   score(i) = item_normed_row_i  dot  SUM_j_rated( r_c(u,j) * item_normed_row_j )
        #
        # Let weighted_sum = sum_j[ r_c(u,j) * item_matrix_normed[j] ]  (1 x n_users)
        # Then scores = item_matrix_normed @ weighted_sum.T              (n_items, 1)

        # Step 1: build weighted user preference vector in user-space
        # Shape: (n_users,)  (dense, but n_users=12k — fine in memory)
        n_users = self.item_matrix_normed.shape[1]
        weighted_sum = np.zeros(n_users, dtype=np.float64)
        for j_local, j_global in enumerate(rated_idxs):
            # item_matrix_normed[j_global] is the normalised row for item j
            row_j = np.array(self.item_matrix_normed[j_global].todense()).ravel()
            weighted_sum += centered_vals[j_local] * row_j    # (n_users,)

        # Step 2: scores = item_matrix_normed @ weighted_sum  → (n_items,)
        scores = np.array(
            self.item_matrix_normed @ weighted_sum
        ).ravel()  # (n_items,)

        # Add user mean back to put scores on the rating scale
        scores = scores + u_mean

        # Mask seen items → -inf (BEFORE ranking, critical)
        if seen_items:
            for mid in seen_items:
                if mid in self.movie_to_idx:
                    scores[self.movie_to_idx[mid]] = -np.inf

        # Select Top-K
        if k >= n_items:
            top_idxs = np.argsort(scores)[::-1][:k]
        else:
            top_idxs = np.argpartition(scores, -k)[-k:]
            top_idxs = top_idxs[np.argsort(scores[top_idxs])[::-1]]

        return [self.idx_to_movie[idx] for idx in top_idxs if scores[idx] > -np.inf]

