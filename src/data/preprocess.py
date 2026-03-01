"""
Data preprocessing for AHF experiments.
Handles ML100K, ML1M, and Yelp datasets.
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


DATA_DIR = Path(__file__).parent.parent.parent / "data"


# ──────────────────────────────────────────────────────────────────────────────
# MovieLens 100K
# ──────────────────────────────────────────────────────────────────────────────

GENRES_ML = [
    "unknown", "Action", "Adventure", "Animation", "Childrens", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

ML100K_AGE_BINS = {1: "Under 18", 18: "18-24", 25: "25-34",
                   35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}


def load_ml100k(data_dir=None):
    data_dir = Path(data_dir or DATA_DIR / "ml-100k")
    
    # Ratings
    ratings = pd.read_csv(
        data_dir / "u.data",
        sep="\t", header=None,
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    
    # User demographics
    users = pd.read_csv(
        data_dir / "u.user",
        sep="|", header=None,
        names=["user_id", "age", "gender", "occupation", "zip"]
    )
    users["age_group"] = users["age"].apply(_ml_age_group)
    
    # Item genres (u.item has genre indicators at columns 6..24)
    items_raw = pd.read_csv(
        data_dir / "u.item",
        sep="|", header=None, encoding="latin-1",
        names=["item_id", "title", "release_date", "video_date", "imdb_url"] + GENRES_ML
    )
    genre_cols = GENRES_ML
    items = items_raw[["item_id"] + genre_cols].copy()
    
    # Normalize genre weights (phi_{v,c})
    genre_matrix = items[genre_cols].values.astype(float)
    row_sums = genre_matrix.sum(axis=1, keepdims=True).clip(min=1)
    items[genre_cols] = genre_matrix / row_sums
    
    return ratings, users, items, genre_cols


def load_ml1m(data_dir=None):
    data_dir = Path(data_dir or DATA_DIR / "ml-1m")
    
    ratings = pd.read_csv(
        data_dir / "ratings.dat",
        sep="::", header=None, engine="python",
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    
    users = pd.read_csv(
        data_dir / "users.dat",
        sep="::", header=None, engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"]
    )
    users["age_group"] = users["age"].apply(_ml_age_group)
    
    movies = pd.read_csv(
        data_dir / "movies.dat",
        sep="::", header=None, engine="python", encoding="latin-1",
        names=["item_id", "title", "genres"]
    )
    
    # Expand pipe-separated genres into indicator columns
    all_genres = set()
    for g in movies["genres"]:
        all_genres.update(g.split("|"))
    all_genres = sorted(all_genres)
    
    for g in all_genres:
        movies[g] = movies["genres"].apply(lambda x: int(g in x.split("|")))
    
    genre_matrix = movies[all_genres].values.astype(float)
    row_sums = genre_matrix.sum(axis=1, keepdims=True).clip(min=1)
    movies[all_genres] = genre_matrix / row_sums
    
    return ratings, users, movies, all_genres


def _ml_age_group(age):
    thresholds = [56, 50, 45, 35, 25, 18, 1]
    for t in thresholds:
        if age >= t:
            return ML100K_AGE_BINS[t]
    return "Under 18"


# ──────────────────────────────────────────────────────────────────────────────
# Yelp (following Mansoury et al. 2019 preprocessing)
# ──────────────────────────────────────────────────────────────────────────────

def load_yelp(data_dir=None):
    data_dir = Path(data_dir or DATA_DIR / "yelp_raw")
    
    reviews_path = data_dir / "yelp_academic_dataset_review.json"
    business_path = data_dir / "yelp_academic_dataset_business.json"
    
    # Load reviews
    reviews = []
    with open(reviews_path, "r") as f:
        for line in f:
            r = json.loads(line)
            reviews.append({
                "user_id": r["user_id"],
                "item_id": r["business_id"],
                "rating": r["stars"],
                "timestamp": pd.Timestamp(r["date"]).timestamp(),
            })
    reviews = pd.DataFrame(reviews)
    
    # Load businesses
    businesses = []
    with open(business_path, "r") as f:
        for line in f:
            b = json.loads(line)
            cats = b.get("categories") or ""
            businesses.append({
                "item_id": b["business_id"],
                "categories": cats,
            })
    businesses = pd.DataFrame(businesses)
    
    # Extract top-level categories (Mansoury et al. use 21 categories)
    YELP_CATS = [
        "Restaurants", "Shopping", "Beauty & Spas", "Health & Medical",
        "Home Services", "Automotive", "Nightlife", "Bars", "Sandwiches",
        "American (Traditional)", "American (New)", "Pizza", "Mexican",
        "Chinese", "Italian", "Japanese", "Fast Food", "Coffee & Tea",
        "Hotels & Travel", "Active Life", "Arts & Entertainment",
    ]
    
    for c in YELP_CATS:
        businesses[c] = businesses["categories"].apply(
            lambda x: int(c in (x or ""))
        )
    
    cat_matrix = businesses[YELP_CATS].values.astype(float)
    row_sums = cat_matrix.sum(axis=1, keepdims=True).clip(min=1)
    businesses[YELP_CATS] = cat_matrix / row_sums
    
    # Filter active users and businesses (≥5 interactions)
    for _ in range(3):
        user_counts = reviews["user_id"].value_counts()
        item_counts = reviews["item_id"].value_counts()
        reviews = reviews[
            reviews["user_id"].isin(user_counts[user_counts >= 5].index) &
            reviews["item_id"].isin(item_counts[item_counts >= 5].index)
        ]
    
    # Placeholder sensitive attribute: activity level (high/low based on review count)
    user_activity = reviews.groupby("user_id").size()
    median_activity = user_activity.median()
    users_df = user_activity.reset_index()
    users_df.columns = ["user_id", "review_count"]
    users_df["activity_group"] = users_df["review_count"].apply(
        lambda x: "high" if x >= median_activity else "low"
    )
    
    businesses = businesses[businesses["item_id"].isin(reviews["item_id"].unique())]
    
    return reviews, users_df, businesses, YELP_CATS


# ──────────────────────────────────────────────────────────────────────────────
# Train/Val/Test split (per-user holdout)
# ──────────────────────────────────────────────────────────────────────────────

def split_data(ratings, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Per-user holdout split. For each user, the last (by timestamp) 
    test_ratio fraction of interactions go to test, next val_ratio to val,
    remainder to train.
    """
    ratings = ratings.sort_values(["user_id", "timestamp"])
    
    train_rows, val_rows, test_rows = [], [], []
    
    for uid, group in ratings.groupby("user_id"):
        n = len(group)
        if n < 3:
            train_rows.append(group)
            continue
        n_test = max(1, int(n * test_ratio))
        n_val  = max(1, int(n * val_ratio))
        test_rows.append(group.iloc[-n_test:])
        val_rows.append(group.iloc[-(n_test + n_val):-n_test])
        train_rows.append(group.iloc[:-(n_test + n_val)])
    
    train = pd.concat(train_rows).reset_index(drop=True)
    val   = pd.concat(val_rows).reset_index(drop=True)
    test  = pd.concat(test_rows).reset_index(drop=True)
    
    return train, val, test


# ──────────────────────────────────────────────────────────────────────────────
# Category weights builder
# ──────────────────────────────────────────────────────────────────────────────

def build_phi(items, item_ids, cat_cols):
    """
    Returns phi[item_id] = np.array of shape (n_cats,) with normalized weights.
    """
    items_indexed = items.set_index("item_id")
    phi = {}
    for iid in item_ids:
        if iid in items_indexed.index:
            phi[iid] = items_indexed.loc[iid, cat_cols].values.astype(float)
        else:
            phi[iid] = np.zeros(len(cat_cols))
    return phi


# ──────────────────────────────────────────────────────────────────────────────
# Main preprocessing entry point
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_all(out_dir=None):
    import pickle
    out_dir = Path(out_dir or DATA_DIR / "processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for name, loader, sensitive_cols in [
        ("ml100k", load_ml100k, ["gender", "age_group"]),
        ("ml1m",   load_ml1m,   ["gender", "age_group"]),
    ]:
        print(f"Processing {name}...")
        try:
            ratings, users, items, cat_cols = loader()
        except FileNotFoundError:
            print(f"  Skipping {name} — data not found.")
            continue
        
        train, val, test = split_data(ratings)
        item_ids = items["item_id"].tolist()
        phi = build_phi(items, item_ids, cat_cols)
        
        # Re-index users/items to 0-based integers
        all_users = ratings["user_id"].unique().tolist()
        all_items = item_ids
        user2idx = {u: i for i, u in enumerate(all_users)}
        item2idx = {v: i for i, v in enumerate(all_items)}
        
        def remap(df):
            df = df.copy()
            df["user_idx"] = df["user_id"].map(user2idx)
            df["item_idx"] = df["item_id"].map(item2idx)
            return df.dropna(subset=["user_idx", "item_idx"])
        
        data = {
            "train": remap(train),
            "val":   remap(val),
            "test":  remap(test),
            "users": users,
            "items": items,
            "cat_cols": cat_cols,
            "phi": phi,  # item_id -> np.array
            "user2idx": user2idx,
            "item2idx": item2idx,
            "sensitive_cols": sensitive_cols,
            "n_users": len(all_users),
            "n_items": len(all_items),
        }
        
        with open(out_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(data, f)
        print(f"  Saved {name} to {out_dir / f'{name}.pkl'}")
    
    # Yelp
    print("Processing yelp...")
    try:
        ratings, users, items, cat_cols = load_yelp()
        train, val, test = split_data(ratings)
        all_users = ratings["user_id"].unique().tolist()
        all_items = items["item_id"].unique().tolist()
        user2idx = {u: i for i, u in enumerate(all_users)}
        item2idx = {v: i for i, v in enumerate(all_items)}
        phi = build_phi(items, all_items, cat_cols)
        
        def remap(df):
            df = df.copy()
            df["user_idx"] = df["user_id"].map(user2idx)
            df["item_idx"] = df["item_id"].map(item2idx)
            return df.dropna(subset=["user_idx", "item_idx"])
        
        data = {
            "train": remap(train),
            "val":   remap(val),
            "test":  remap(test),
            "users": users,
            "items": items,
            "cat_cols": cat_cols,
            "phi": phi,
            "user2idx": user2idx,
            "item2idx": item2idx,
            "sensitive_cols": ["activity_group"],
            "n_users": len(all_users),
            "n_items": len(all_items),
        }
        out_path = out_dir / "yelp.pkl"
        import pickle
        with open(out_path, "wb") as f:
            pickle.dump(data, f)
        print(f"  Saved yelp to {out_path}")
    except FileNotFoundError:
        print("  Skipping yelp — data not found.")


if __name__ == "__main__":
    preprocess_all()
