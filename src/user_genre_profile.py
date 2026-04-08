"""
user_genre_profile.py

Builds a User Genre Profile for every user, capturing:
  1. Weighted score  — average rating per genre, normalized 0-1 per user
  2. Binary liked    — count of movies rated >= 3.5 per genre, normalized 0-1
  3. Combined score  — blend of both (default 50/50)
  4. Top-N genres    — each user's favourite genres ranked
"""

import pandas as pd
import numpy as np
import os


# ─────────────────────────────────────────────────────────────
# CORE BUILDER
# ─────────────────────────────────────────────────────────────

def build_user_genre_profile(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    liked_threshold: float = 3.5,
    blend_weight: float = 0.5,
    top_n: int = 3
) -> pd.DataFrame:
    """
    Build a normalized user-genre preference profile.

    For each (user, genre) pair we compute:
      - weighted_score : mean rating the user gave to movies of that genre,
                         normalized 0-1 within the user (so users are comparable)
      - liked_score    : fraction of "liked" movies (rating >= threshold) in
                         that genre, normalized 0-1 within the user
      - combined_score : blend_weight * weighted_score +
                         (1 - blend_weight) * liked_score
      - is_top_genre   : True if genre is in the user's top-N by combined_score

    Args:
        ratings          : cleaned ratings DataFrame [userId, movieId, rating]
        movies           : cleaned movies DataFrame  [movieId, genres, ...]
        liked_threshold  : min rating to count as "liked"   (default 3.5)
        blend_weight     : weight for weighted_score in blend (default 0.5)
        top_n            : number of top genres to flag per user (default 3)

    Returns:
        Long-format DataFrame with one row per (userId, genre)
        Columns: userId, genre, avg_rating, weighted_score,
                 liked_count, liked_score, combined_score, is_top_genre
    """
    print("[build_user_genre_profile] Starting...")

    # ── Explode genres ────────────────────────────────────────
    if "genre_list" not in movies.columns:
        movies = movies.copy()
        movies["genre_list"] = movies["genres"].str.split("|")

    movies_exp = (
        movies[["movieId", "genre_list"]]
        .explode("genre_list")
        .rename(columns={"genre_list": "genre"})
    )
    movies_exp = movies_exp[movies_exp["genre"] != "(no genres listed)"]

    # ── Merge ratings with genres ─────────────────────────────
    merged = ratings[["userId", "movieId", "rating"]].merge(
        movies_exp, on="movieId", how="inner"
    )
    print(f"  Merged rows (ratings × genres): {len(merged):,}")

    # ── 1. Weighted score: avg rating per (user, genre) ───────
    weighted = (
        merged
        .groupby(["userId", "genre"])["rating"]
        .mean()
        .reset_index()
        .rename(columns={"rating": "avg_rating"})
    )

    # Normalize 0-1 per user
    weighted["weighted_score"] = (
        weighted
        .groupby("userId")["avg_rating"]
        .transform(lambda x: _minmax(x))
    )

    # ── 2. Liked score: count liked movies per (user, genre) ──
    liked = merged[merged["rating"] >= liked_threshold]
    liked_count = (
        liked
        .groupby(["userId", "genre"])["rating"]
        .count()
        .reset_index()
        .rename(columns={"rating": "liked_count"})
    )

    # ── 3. Merge and normalize liked_count ────────────────────
    profile = weighted.merge(liked_count, on=["userId", "genre"], how="left")
    profile["liked_count"] = profile["liked_count"].fillna(0).astype(int)

    profile["liked_score"] = (
        profile
        .groupby("userId")["liked_count"]
        .transform(lambda x: _minmax(x))
    )

    # ── 4. Combined score ─────────────────────────────────────
    profile["combined_score"] = (
        blend_weight * profile["weighted_score"] +
        (1 - blend_weight) * profile["liked_score"]
    )

    # ── 5. Top-N flag ─────────────────────────────────────────
    profile["rank"] = (
        profile
        .groupby("userId")["combined_score"]
        .rank(method="first", ascending=False)
    )
    profile["is_top_genre"] = profile["rank"] <= top_n
    profile = profile.drop(columns="rank")

    profile = profile.sort_values(["userId", "combined_score"], ascending=[True, False])
    profile = profile.reset_index(drop=True)

    print(f"  Users profiled : {profile['userId'].nunique():,}")
    print(f"  Genres tracked : {profile['genre'].nunique()}")
    print(f"[build_user_genre_profile] Done — shape: {profile.shape}")
    return profile


def _minmax(series: pd.Series) -> pd.Series:
    """Min-max normalize a series; returns 0 if all values are equal."""
    mn, mx = series.min(), series.max()
    if mx - mn < 1e-9:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mn) / (mx - mn)


# ─────────────────────────────────────────────────────────────
# PIVOT TABLE (wide format)
# ─────────────────────────────────────────────────────────────

def pivot_user_genre_profile(profile: pd.DataFrame, score_col: str = "combined_score") -> pd.DataFrame:
    """
    Convert long-format profile to wide format:
      rows = users, columns = genres, values = score_col

    Useful for feeding into downstream models or similarity computation.

    Args:
        profile   : output of build_user_genre_profile()
        score_col : which score column to use (default "combined_score")
    Returns:
        Wide DataFrame of shape (n_users, n_genres)
    """
    wide = profile.pivot_table(
        index="userId", columns="genre", values=score_col, fill_value=0
    )
    wide.columns.name = None
    print(f"[pivot_user_genre_profile] Shape: {wide.shape}")
    return wide


# ─────────────────────────────────────────────────────────────
# TOP-N LOOKUP
# ─────────────────────────────────────────────────────────────

def get_top_genres(profile: pd.DataFrame, user_id: int, n: int = 5) -> pd.DataFrame:
    """
    Return the top-N genres for a specific user.

    Args:
        profile : output of build_user_genre_profile()
        user_id : the userId to look up
        n       : number of top genres to return
    Returns:
        DataFrame with top-N genres and their scores
    """
    user_profile = profile[profile["userId"] == user_id]
    if user_profile.empty:
        print(f"  User {user_id} not found in profile.")
        return pd.DataFrame()
    return (
        user_profile
        .nlargest(n, "combined_score")
        [["genre", "avg_rating", "weighted_score", "liked_count", "liked_score", "combined_score"]]
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────────────────────

def save_user_genre_profile(profile: pd.DataFrame, processed_dir: str = "data/processed/") -> None:
    """Save long-format profile to CSV."""
    os.makedirs(processed_dir, exist_ok=True)
    path = processed_dir + "user_genre_profile.csv"
    profile.to_csv(path, index=False)
    print(f"[save] Saved to {path}")


def load_user_genre_profile(processed_dir: str = "data/processed/") -> pd.DataFrame:
    """Load long-format profile from CSV."""
    path = processed_dir + "user_genre_profile.csv"
    df = pd.read_csv(path)
    print(f"[load] Loaded {len(df):,} rows from {path}")
    return df
