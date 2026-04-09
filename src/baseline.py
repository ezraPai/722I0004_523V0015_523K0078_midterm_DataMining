"""
baseline.py
-----------
Baseline recommender system implementing:
  1. Global Mean predictor
  2. Bias Model: pred(u, i) = mu + b_u + b_i
  3. Popularity ranking via weighted rating (Bayesian average)

All model statistics are computed EXCLUSIVELY from train data.
No test data is ever read or accessed in this module.

Inputs  : train DataFrame (userId, movieId, rating)
Outputs : predictions (float), popularity ranking (DataFrame)
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

# Minimum number of ratings a movie must have to receive a
# popularity score above the prior. Movies below this threshold
# are strongly pulled toward the global mean.
DEFAULT_POPULARITY_THRESHOLD = 50


# ─────────────────────────────────────────────────────────────
# BASELINE MODEL CLASS
# ─────────────────────────────────────────────────────────────

class BaselineModel:
    """
    A bias-corrected baseline recommender.

    Attributes
    ----------
    mu : float
        Global mean rating computed from train data.
    user_bias : dict[int, float]
        Per-user deviation from the global mean (b_u).
    item_bias : dict[int, float]
        Per-item deviation from the global mean (b_i).
    popularity_df : pd.DataFrame
        Movies ranked by weighted rating score.
    is_fitted : bool
        Whether fit() has been called.
    """

    def __init__(self, popularity_threshold: int = DEFAULT_POPULARITY_THRESHOLD):
        """
        Parameters
        ----------
        popularity_threshold : int
            Minimum vote count (m) used in the Bayesian weighted rating
            formula. Movies with fewer ratings are shrunk toward mu.
        """
        self.popularity_threshold = popularity_threshold
        self.mu: float = 0.0
        self.user_bias: dict = {}
        self.item_bias: dict = {}
        self.popularity_df: pd.DataFrame = pd.DataFrame()
        self.is_fitted: bool = False

    # ── FIT ────────────────────────────────────────────────────

    def fit(self, train: pd.DataFrame) -> "BaselineModel":
        """
        Fit the baseline model on training data ONLY.

        Computes:
        - Global mean mu
        - User biases b_u = mean(user ratings) - mu
        - Item biases b_i = mean(item ratings) - mu
        - Popularity-ranked movie list (weighted rating)

        Parameters
        ----------
        train : pd.DataFrame
            Training ratings with columns [userId, movieId, rating].

        Returns
        -------
        self
        """
        # ── Global mean (μ) ────────────────────────────────────
        self.mu = float(train["rating"].mean())

        # ── User biases (b_u) ──────────────────────────────────
        user_means = train.groupby("userId")["rating"].mean()
        self.user_bias = (user_means - self.mu).to_dict()

        # ── Item biases (b_i) ──────────────────────────────────
        item_means = train.groupby("movieId")["rating"].mean()
        item_counts = train.groupby("movieId")["rating"].count()
        self.item_bias = (item_means - self.mu).to_dict()

        # ── Popularity ranking (Bayesian weighted rating) ───────
        # score = (v / (v + m)) * R + (m / (v + m)) * mu
        # Where R = movie mean, v = vote count, m = threshold
        m = self.popularity_threshold
        pop = pd.DataFrame({
            "movieId": item_means.index,
            "R":       item_means.values,       # movie mean rating
            "v":       item_counts[item_means.index].values,  # vote count
        })
        pop["score"] = (
            (pop["v"] / (pop["v"] + m)) * pop["R"]
            + (m / (pop["v"] + m)) * self.mu
        )
        self.popularity_df = (
            pop.sort_values("score", ascending=False)
               .reset_index(drop=True)
        )

        self.is_fitted = True
        print(f"[BaselineModel.fit] Done.")
        print(f"  mu              = {self.mu:.4f}")
        print(f"  unique users    = {len(self.user_bias):,}")
        print(f"  unique items    = {len(self.item_bias):,}")
        print(f"  popularity rows = {len(self.popularity_df):,}")
        return self

    # ── PREDICT ────────────────────────────────────────────────

    def predict(self, userId, movieId) -> float:
        """
        Predict the rating for a (userId, movieId) pair.

        Fallback hierarchy:
          - Known user & item  →  mu + b_u + b_i
          - Unknown user only  →  mu + b_i
          - Unknown item only  →  mu + b_u
          - Both unknown       →  mu

        Parameters
        ----------
        userId  : user identifier (int or hashable)
        movieId : movie identifier (int or hashable)

        Returns
        -------
        float : predicted rating, always in [0.5, 5.0], never NaN
        """
        b_u = self.user_bias.get(userId, 0.0)
        b_i = self.item_bias.get(movieId, 0.0)
        pred = self.mu + b_u + b_i
        # Clamp to valid rating range to prevent out-of-range predictions
        return float(np.clip(pred, 0.5, 5.0))

    def predict_batch(self, pairs: pd.DataFrame) -> np.ndarray:
        """
        Vectorised batch prediction for a DataFrame of (userId, movieId) pairs.

        Parameters
        ----------
        pairs : pd.DataFrame
            Must contain columns [userId, movieId].

        Returns
        -------
        np.ndarray of float predictions, shape (len(pairs),)
        """
        b_u = pairs["userId"].map(self.user_bias).fillna(0.0).values
        b_i = pairs["movieId"].map(self.item_bias).fillna(0.0).values
        preds = self.mu + b_u + b_i
        return np.clip(preds, 0.5, 5.0)

    # ── TOP-K RANKING ─────────────────────────────────────────

    def recommend_top_k(
        self,
        userId,
        k: int = 10,
        seen_items: set = None,
    ) -> pd.DataFrame:
        """
        Return the top-K unseen popular movies for a user.

        For the popularity baseline, ranking is global and not
        personalised. Items the user has already rated in train are
        masked out before slicing the top K.

        Parameters
        ----------
        userId     : user identifier
        k          : number of recommendations to return
        seen_items : set of movieIds already rated by this user in train.
                     If None, no masking is applied (not recommended).

        Returns
        -------
        pd.DataFrame with columns [movieId, score, rank]
        """
        ranked = self.popularity_df.copy()

        if seen_items:
            ranked = ranked[~ranked["movieId"].isin(seen_items)]

        top_k = ranked.head(k).copy()
        top_k["rank"] = range(1, len(top_k) + 1)
        return top_k[["movieId", "score", "rank"]]

    # ── DIAGNOSTICS ───────────────────────────────────────────

    def fallback_stats(self, test: pd.DataFrame) -> dict:
        """
        Compute fallback usage statistics on a test set.

        Parameters
        ----------
        test : pd.DataFrame
            Test set with columns [userId, movieId].

        Returns
        -------
        dict with fallback counts and percentages.
        """
        n = len(test)
        unknown_users  = ~test["userId"].isin(self.user_bias)
        unknown_items  = ~test["movieId"].isin(self.item_bias)
        both_unknown   = unknown_users & unknown_items

        stats = {
            "total_test_rows":     n,
            "unseen_users":        int(unknown_users.sum()),
            "unseen_items":        int(unknown_items.sum()),
            "both_unseen":         int(both_unknown.sum()),
            "pct_unseen_users":    round(100 * unknown_users.mean(), 2),
            "pct_unseen_items":    round(100 * unknown_items.mean(), 2),
            "pct_both_unseen":     round(100 * both_unknown.mean(), 2),
        }
        return stats
