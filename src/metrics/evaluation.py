"""
Evaluation metrics for AHF experiments.

- NDCG@k  (Eq. from Jarvelin & Kekalainen 2002)
- CC Disparity   (Category Coverage Disparity, Eq. 4)
- CDCG Disparity (Discounted Category Gain Disparity, Eq. 5)

All match the protocol of Kheya et al. (2025).
"""
from __future__ import annotations

import math
import numpy as np
from itertools import combinations
from typing import Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# NDCG@k
# ──────────────────────────────────────────────────────────────────────────────

def ndcg_at_k(ranked_list: List, relevant: set, k: int = 20) -> float:
    """
    NDCG@k for a single user.
    Treats relevance as binary (item in test set = 1, else 0).
    """
    dcg = 0.0
    for j, item in enumerate(ranked_list[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(j + 1)
    
    # Ideal DCG
    n_relevant = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(j + 1) for j in range(1, n_relevant + 1))
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def mean_ndcg_at_k(
    ranked_lists: Dict[int, List],   # user_idx -> ranked list of item_ids
    test_items: Dict[int, set],      # user_idx -> set of relevant item_ids
    k: int = 20,
) -> float:
    scores = []
    for uid, rl in ranked_lists.items():
        rel = test_items.get(uid, set())
        if rel:
            scores.append(ndcg_at_k(rl, rel, k))
    return float(np.mean(scores)) if scores else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Category Coverage (CC) per group  (Eq. 4)
# ──────────────────────────────────────────────────────────────────────────────

def category_coverage(
    ranked_lists: Dict[int, List],  # user_idx -> list of item_ids
    phi: Dict,                       # item_id -> (C,)
    n_cats: int,
    k: int = 20,
) -> np.ndarray:
    """
    CC(c, U) = (1/|U|) * sum_u (1/k) * sum_{v in R_u} phi_{v,c}
    Returns (C,) array.
    """
    acc = np.zeros(n_cats, dtype=float)
    n_users = 0
    for uid, rl in ranked_lists.items():
        n_users += 1
        for iid in rl[:k]:
            acc += phi.get(iid, np.zeros(n_cats))
        # Divide by k for this user
    if n_users == 0:
        return acc
    return acc / (n_users * k)


# ──────────────────────────────────────────────────────────────────────────────
# Discounted Category Gain (CDCG) per group  (Eq. 5)
# ──────────────────────────────────────────────────────────────────────────────

def cdcg(
    ranked_lists: Dict[int, List],
    phi: Dict,
    n_cats: int,
    k: int = 20,
) -> np.ndarray:
    """
    CDCG(c, U) = (1/|U|) * sum_u (1/k) * sum_j phi_{v_j, c} / log2(j+1)
    Returns (C,) array.
    """
    acc = np.zeros(n_cats, dtype=float)
    n_users = 0
    for uid, rl in ranked_lists.items():
        n_users += 1
        for j, iid in enumerate(rl[:k], start=1):
            discount = 1.0 / math.log2(j + 1)
            acc += phi.get(iid, np.zeros(n_cats)) * discount
    if n_users == 0:
        return acc
    return acc / (n_users * k)


# ──────────────────────────────────────────────────────────────────────────────
# Disparity (sum of pairwise absolute differences across groups × categories)
# ──────────────────────────────────────────────────────────────────────────────

def group_disparity(
    cc_per_group: Dict,   # group_val -> (C,) CC or CDCG
) -> float:
    """
    Sum over categories of sum of pairwise |CC_g1 - CC_g2|.
    Lower is better; zero means perfect demographic parity.
    """
    groups = list(cc_per_group.keys())
    if len(groups) < 2:
        return 0.0
    
    total = 0.0
    for g1, g2 in combinations(groups, 2):
        total += float(np.abs(cc_per_group[g1] - cc_per_group[g2]).sum())
    return total


def compute_all_metrics(
    ranked_lists: Dict[int, List],    # user_idx -> ranked item list
    test_items: Dict[int, set],       # user_idx -> relevant items
    phi: Dict,                        # item_id -> (C,)
    n_cats: int,
    user_groups: Dict[int, str],      # user_idx -> group value
    k: int = 20,
) -> Dict[str, float]:
    """
    Compute NDCG@k, CC-Disparity, CDCG-Disparity.
    """
    ndcg = mean_ndcg_at_k(ranked_lists, test_items, k)
    
    # Split ranked lists by group
    groups = {}
    for uid, rl in ranked_lists.items():
        g = user_groups.get(uid, "unknown")
        groups.setdefault(g, {})[uid] = rl
    
    cc_per_group  = {g: category_coverage(rls, phi, n_cats, k) for g, rls in groups.items()}
    cdcg_per_group = {g: cdcg(rls, phi, n_cats, k)             for g, rls in groups.items()}
    
    cc_disp   = group_disparity(cc_per_group)
    cdcg_disp = group_disparity(cdcg_per_group)
    
    return {
        "ndcg@k":     ndcg,
        "cc_disp":    cc_disp,
        "cdcg_disp":  cdcg_disp,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Multi-run statistics
# ──────────────────────────────────────────────────────────────────────────────

def summarize_runs(results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Given list of per-run metric dicts, return {metric: {mean, std}}.
    """
    from collections import defaultdict
    per_metric = defaultdict(list)
    for r in results:
        for k, v in r.items():
            per_metric[k].append(v)
    
    summary = {}
    for metric, vals in per_metric.items():
        summary[metric] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
        }
    return summary
