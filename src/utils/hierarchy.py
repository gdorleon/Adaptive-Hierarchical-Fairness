"""
Category hierarchy construction (Section 4.2).

For flat taxonomies (ML100K, ML1M): agglomerative clustering of
genre co-occurrence matrix → super-genre clusters.

For Yelp: use native two-level taxonomy.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def build_cooccurrence_matrix(train_df, phi, item2idx, n_cats):
    """
    Build genre co-occurrence matrix from training data.
    C[c1, c2] = number of items that have both genres c1 and c2.
    """
    co = np.zeros((n_cats, n_cats), dtype=float)
    counted = set()
    
    for item_id, w in phi.items():
        if item_id in counted:
            continue
        counted.add(item_id)
        w = np.array(w)
        # outer product of indicator (binarized) genre vector
        ind = (w > 0).astype(float)
        co += np.outer(ind, ind)
    
    return co


def build_genre_hierarchy_agglomerative(phi, n_cats, n_super=6, random_state=42):
    """
    Build a 2-level hierarchy by clustering genres via agglomerative clustering
    on the co-occurrence matrix.
    
    Returns:
        genre_to_super (n_cats,): integer super-genre id for each genre
        n_super: number of super-genres
        super_weights (n_items dict or array): phi at level 2
    """
    # Build co-occurrence from phi values
    co = np.zeros((n_cats, n_cats), dtype=float)
    for w in phi.values():
        w = np.array(w)
        ind = (w > 0).astype(float)
        co += np.outer(ind, ind)
    
    # Convert to distance matrix
    diag = np.diag(co).clip(min=1)
    similarity = co / np.sqrt(np.outer(diag, diag)).clip(min=1e-10)
    np.fill_diagonal(similarity, 1.0)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    
    # Agglomerative clustering
    condensed = squareform(distance, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, n_super, criterion="maxclust") - 1  # 0-indexed
    
    return labels, n_super


def build_hierarchy_from_flat(phi, cat_cols, n_super=6):
    """
    Main entry point for flat-category datasets.
    
    Returns:
        hierarchy: dict with keys 1..L, each mapping to:
            - 'cats': list of category names
            - 'phi': dict {item_id: np.array} of weights at this level
            - 'n_cats': int
        level1_to_level2: (n_cats_level1,) mapping from fine to coarse idx
    """
    n_cats = len(cat_cols)
    genre_to_super, n_super_actual = build_genre_hierarchy_agglomerative(
        phi, n_cats, n_super
    )
    
    # Level 1: original genres
    # Level 2: super-genres (sum of child genre weights)
    super_names = [f"Super-{i}" for i in range(n_super_actual)]
    
    phi_level2 = {}
    for item_id, w in phi.items():
        w = np.array(w)
        w_super = np.zeros(n_super_actual, dtype=float)
        for c, gs in enumerate(genre_to_super):
            w_super[gs] += w[c]
        # Normalize
        s = w_super.sum()
        if s > 0:
            w_super = w_super / s
        phi_level2[item_id] = w_super
    
    hierarchy = {
        1: {
            "cats": cat_cols,
            "phi": phi,
            "n_cats": n_cats,
        },
        2: {
            "cats": super_names,
            "phi": phi_level2,
            "n_cats": n_super_actual,
        },
    }
    
    return hierarchy, genre_to_super


def build_yelp_hierarchy(cat_cols):
    """
    Native two-level Yelp taxonomy.
    Maps Yelp leaf categories to top-level groups.
    """
    top_level_map = {
        "Restaurants": "Food & Dining",
        "Sandwiches":  "Food & Dining",
        "American (Traditional)": "Food & Dining",
        "American (New)": "Food & Dining",
        "Pizza":       "Food & Dining",
        "Mexican":     "Food & Dining",
        "Chinese":     "Food & Dining",
        "Italian":     "Food & Dining",
        "Japanese":    "Food & Dining",
        "Fast Food":   "Food & Dining",
        "Coffee & Tea":"Food & Dining",
        "Bars":        "Nightlife & Entertainment",
        "Nightlife":   "Nightlife & Entertainment",
        "Arts & Entertainment": "Nightlife & Entertainment",
        "Shopping":    "Retail & Services",
        "Beauty & Spas": "Retail & Services",
        "Home Services": "Retail & Services",
        "Automotive":  "Retail & Services",
        "Health & Medical": "Health & Wellness",
        "Active Life": "Health & Wellness",
        "Hotels & Travel": "Travel",
    }
    
    top_cats = sorted(set(top_level_map.values()))
    top_cat_idx = {c: i for i, c in enumerate(top_cats)}
    
    # Map fine category → coarse index
    fine_to_coarse = np.array([
        top_cat_idx[top_level_map.get(c, "Other")]
        for c in cat_cols
    ])
    
    return top_cats, fine_to_coarse


def get_level_targets(fine_targets, fine_to_coarse, n_coarse):
    """
    Aggregate fine-grained targets to coarse level by summing child weights.
    
    Args:
        fine_targets: (U, C_fine) per-user fine targets
        fine_to_coarse: (C_fine,) mapping
        n_coarse: int
    Returns:
        coarse_targets: (U, C_coarse)
    """
    U, C = fine_targets.shape
    coarse = np.zeros((U, n_coarse), dtype=float)
    for c_fine in range(C):
        c_coarse = fine_to_coarse[c_fine]
        coarse[:, c_coarse] += fine_targets[:, c_fine]
    # Re-normalize rows
    row_sums = coarse.sum(axis=1, keepdims=True).clip(min=1e-10)
    return coarse / row_sums
