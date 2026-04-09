"""
evaluation.py
-------------
Shared evaluation utilities for the movie recommendation system.

Implements:
  - Rating prediction metrics: RMSE, MAE
  - Ranking metrics: Precision@K, NDCG@K
  - Sanity check runner

All evaluation is performed ONLY on test.csv.
No training data is ever read or modified here.

Inputs  : ground-truth test DataFrame, predicted scores (np.ndarray)
Outputs : metric dictionaries
"""

import numpy as np
import pandas as pd
from typing import Callable


# ─────────────────────────────────────────────────────────────
# RATING PREDICTION METRICS
# ─────────────────────────────────────────────────────────────

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    Parameters
    ----------
    y_true : np.ndarray  Ground-truth ratings from test.csv
    y_pred : np.ndarray  Predicted ratings (same length)

    Returns
    -------
    float : RMSE value
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    Parameters
    ----------
    y_true : np.ndarray  Ground-truth ratings from test.csv
    y_pred : np.ndarray  Predicted ratings (same length)

    Returns
    -------
    float : MAE value
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_rating_predictions(
    test: pd.DataFrame,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute RMSE and MAE on the test set, plus run sanity checks.

    Parameters
    ----------
    test   : pd.DataFrame with column [rating] (ground truth)
    y_pred : np.ndarray of predictions aligned with test rows

    Returns
    -------
    dict with keys: rmse, mae, n_preds, n_nan_preds
    """
    y_true = test["rating"].values.astype(float)

    # Sanity checks
    assert len(y_true) == len(y_pred), "Prediction count mismatch with test rows"
    n_nan = int(np.isnan(y_pred).sum())

    rmse = compute_rmse(y_true, y_pred)
    mae  = compute_mae(y_true, y_pred)

    return {
        "rmse":       round(rmse, 6),
        "mae":        round(mae, 6),
        "n_preds":    len(y_pred),
        "n_nan_preds": n_nan,
    }


# ─────────────────────────────────────────────────────────────
# RANKING METRICS
# ─────────────────────────────────────────────────────────────

def precision_at_k(relevant: set, recommended: list, k: int) -> float:
    """
    Precision@K for a single user.

    Parameters
    ----------
    relevant    : set of relevant movieIds (rating >= threshold)
    recommended : ordered list of recommended movieIds (top-K)
    k           : cutoff rank

    Returns
    -------
    float : Precision@K in [0, 1]
    """
    if not relevant or k == 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for m in top_k if m in relevant)
    return hits / k


def ndcg_at_k(relevant: set, recommended: list, k: int) -> float:
    """
    Normalised Discounted Cumulative Gain @ K for a single user.

    Parameters
    ----------
    relevant    : set of relevant movieIds (rating >= threshold)
    recommended : ordered list of recommended movieIds (top-K)
    k           : cutoff rank

    Returns
    -------
    float : NDCG@K in [0, 1]
    """
    if not relevant or k == 0:
        return 0.0

    top_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, m in enumerate(top_k)
        if m in relevant
    )
    # Ideal DCG: all relevant items at top positions
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking(
    test: pd.DataFrame,
    train: pd.DataFrame,
    recommend_fn: Callable,
    k: int = 10,
    relevance_threshold: float = 3.5,
) -> dict:
    """
    Compute macro-averaged Precision@K and NDCG@K across all test users.

    For each user:
      - Relevant items = test items with rating >= relevance_threshold
      - Seen items (masked out) = all items in train for that user
      - Recommendations = recommend_fn(userId, k, seen_items)

    Parameters
    ----------
    test                 : pd.DataFrame test set [userId, movieId, rating]
    train                : pd.DataFrame train set [userId, movieId]
    recommend_fn         : callable(userId, k, seen_items) -> list[movieId]
    k                    : top-K cutoff (default 10)
    relevance_threshold  : min rating to count as "relevant" (default 3.5)

    Returns
    -------
    dict with keys: precision_at_k, ndcg_at_k, n_users_evaluated, k
    """
    # Build a seen-items lookup from train (to mask out in ranking)
    train_seen = train.groupby("userId")["movieId"].apply(set).to_dict()

    precisions = []
    ndcgs = []

    for user_id, user_test in test.groupby("userId"):
        # Relevant = items the user genuinely liked
        relevant = set(
            user_test.loc[user_test["rating"] >= relevance_threshold, "movieId"]
        )
        if not relevant:
            continue

        seen = train_seen.get(user_id, set())
        recommended = recommend_fn(user_id, k, seen)

        precisions.append(precision_at_k(relevant, recommended, k))
        ndcgs.append(ndcg_at_k(relevant, recommended, k))

    return {
        "precision_at_k":    round(float(np.mean(precisions)), 6),
        "ndcg_at_k":         round(float(np.mean(ndcgs)), 6),
        "n_users_evaluated": len(precisions),
        "k":                 k,
        "relevance_threshold": relevance_threshold,
    }


# ─────────────────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────────────────

def run_sanity_checks(
    y_pred: np.ndarray,
    rmse: float,
    mae: float,
    rmse_range: tuple = (0.8, 1.5),
    mae_range: tuple = (0.6, 1.2),
) -> list[dict]:
    """
    Run a battery of sanity checks on predictions and metrics.

    Parameters
    ----------
    y_pred     : np.ndarray of predicted ratings
    rmse       : computed RMSE value
    mae        : computed MAE value
    rmse_range : (min, max) acceptable RMSE band
    mae_range  : (min, max) acceptable MAE band

    Returns
    -------
    list of dicts, each with {check, passed, detail}
    """
    checks = []

    # 1. No NaN predictions
    n_nan = int(np.isnan(y_pred).sum())
    checks.append({
        "check":  "No NaN predictions",
        "passed": n_nan == 0,
        "detail": f"{n_nan} NaN values found",
    })

    # 2. All predictions within valid rating range
    out_of_range = int(((y_pred < 0.5) | (y_pred > 5.0)).sum())
    checks.append({
        "check":  "All predictions in [0.5, 5.0]",
        "passed": out_of_range == 0,
        "detail": f"{out_of_range} out-of-range values",
    })

    # 3. RMSE in expected range
    checks.append({
        "check":  f"RMSE in {rmse_range}",
        "passed": rmse_range[0] <= rmse <= rmse_range[1],
        "detail": f"RMSE = {rmse:.4f}",
    })

    # 4. MAE in expected range
    checks.append({
        "check":  f"MAE in {mae_range}",
        "passed": mae_range[0] <= mae <= mae_range[1],
        "detail": f"MAE = {mae:.4f}",
    })

    return checks
