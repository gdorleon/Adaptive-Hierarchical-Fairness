"""
AHF Re-ranking: exact greedy and LSH sketch approximation.

Implements:
  - Kheya et al. (2025) baseline: static CCP via KL
  - AHF exact: hierarchical KL with Bayesian targets
  - AHF sketch: LSH bucket templates + per-user adjustment (Algorithm 1)
"""
from __future__ import annotations

import math
import time
import numpy as np
from typing import Dict, List, Optional, Tuple


EPS0 = 1e-10  # numerical stability constant


# ──────────────────────────────────────────────────────────────────────────────
# RCP computation (Eq. 3)
# ──────────────────────────────────────────────────────────────────────────────

def compute_rcp(
    ranked_list: List[int],
    phi: np.ndarray,  # (N_pool, C) phi for pool items in order
    gamma: float = 0.1,
) -> np.ndarray:
    """
    Recommended Category Proportion for a ranked list.
    r(c|u,I) = sum_j phi_{v_j, c} / j^gamma  /  sum_j 1/j^gamma
    """
    if len(ranked_list) == 0:
        return np.zeros(phi.shape[1])
    weights = np.array([1.0 / ((j + 1) ** gamma) for j in range(len(ranked_list))])
    weighted = phi[ranked_list] * weights[:, None]
    return weighted.sum(axis=0) / (weights.sum() + EPS10)


EPS10 = 1e-10


def compute_rcp_incremental(
    current_rcp: np.ndarray,
    current_weight_sum: float,
    new_item_phi: np.ndarray,
    new_pos: int,
    gamma: float = 0.1,
) -> Tuple[np.ndarray, float]:
    """
    Incrementally update RCP when adding a new item at position new_pos (1-indexed).
    """
    w = 1.0 / (new_pos ** gamma)
    new_weight_sum = current_weight_sum + w
    new_rcp = (current_rcp * current_weight_sum + new_item_phi * w) / (new_weight_sum + EPS10)
    return new_rcp, new_weight_sum


# ──────────────────────────────────────────────────────────────────────────────
# KL divergence (single level)
# ──────────────────────────────────────────────────────────────────────────────

def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q)"""
    mask = p > EPS0
    return float(np.sum(p[mask] * np.log(p[mask] / (q[mask] + EPS0))))


def hierarchical_kl(
    targets: Dict[int, np.ndarray],   # level -> (C_l,) target distribution
    rcps: Dict[int, np.ndarray],      # level -> (C_l,) current RCP
    level_weights: Dict[int, float],
) -> float:
    """Hierarchical KL divergence (Eq. 8)."""
    total = 0.0
    for level, w in level_weights.items():
        total += w * kl_div(targets[level], rcps[level] + EPS0)
    return total


def delta_hierarchical_kl(
    targets: Dict[int, np.ndarray],
    rcps: Dict[int, np.ndarray],
    current_weight_sum: float,
    new_pos: int,
    item_phi_per_level: Dict[int, np.ndarray],
    level_weights: Dict[int, float],
    gamma: float = 0.1,
) -> float:
    """
    Compute marginal increase in hierarchical KL when adding item to list.
    """
    w = 1.0 / (new_pos ** gamma)
    new_ws = current_weight_sum + w
    
    total = 0.0
    for level, lw in level_weights.items():
        new_rcp = (rcps[level] * current_weight_sum + item_phi_per_level[level] * w) / (new_ws + EPS10)
        total += lw * kl_div(targets[level], new_rcp + EPS0)
    return total


# ──────────────────────────────────────────────────────────────────────────────
# Greedy re-ranking core
# ──────────────────────────────────────────────────────────────────────────────

def greedy_rerank(
    candidate_scores: np.ndarray,  # (N_pool,) base scores
    candidate_pool: np.ndarray,    # (N_pool,) item indices
    targets: Dict[int, np.ndarray],  # level -> (C_l,) target
    phi_per_level: Dict[int, np.ndarray],  # level -> (N_pool, C_l)
    k: int,
    beta: float,
    gamma: float = 0.1,
    level_weights: Optional[Dict[int, float]] = None,
) -> List[int]:
    """
    Greedy re-ranking as in Eq. (9).
    Returns list of k item indices (into candidate_pool).
    """
    L = list(targets.keys())
    if level_weights is None:
        # w_l ∝ 1/|C_l|
        raw = {l: 1.0 / len(targets[l]) for l in L}
        total_w = sum(raw.values())
        level_weights = {l: raw[l] / total_w for l in L}
    
    n_pool = len(candidate_pool)
    selected = []
    remaining = set(range(n_pool))
    
    # Current RCP state per level
    rcps = {l: np.zeros_like(targets[l]) for l in L}
    weight_sum = 0.0
    
    for pos in range(1, k + 1):
        best_idx = None
        best_score = -np.inf
        
        for idx in remaining:
            # Base score
            base = candidate_scores[idx]
            
            # Marginal KL increase
            item_phi = {l: phi_per_level[l][idx] for l in L}
            delta_kl = delta_hierarchical_kl(
                targets, rcps, weight_sum, pos, item_phi, level_weights, gamma
            )
            
            score = (1 - beta) * base - beta * delta_kl
            if score > best_score:
                best_score = score
                best_idx = idx
        
        if best_idx is None:
            break
        
        selected.append(best_idx)
        remaining.remove(best_idx)
        
        # Update RCP incrementally
        w = 1.0 / (pos ** gamma)
        weight_sum += w
        for l in L:
            rcps[l] = (rcps[l] * (weight_sum - w) + phi_per_level[l][best_idx] * w) / (weight_sum + EPS10)
    
    return [candidate_pool[i] for i in selected]


# ──────────────────────────────────────────────────────────────────────────────
# Kheya et al. (2025) Baseline
# ──────────────────────────────────────────────────────────────────────────────

class KheyaReranker:
    """
    Static counterfactual category proportion (CCP) re-ranking.
    Reference: Kheya et al. (2025).
    """
    
    def __init__(
        self,
        phi: Dict,           # item_id -> np.array (C,)
        cat_cols: List[str],
        sensitive_col: str,
        lambda_decay: float = 0.05,
        gamma: float = 0.1,
        beta: float = 0.5,
        k: int = 20,
        topn: int = 100,
    ):
        self.phi = phi
        self.cat_cols = cat_cols
        self.sensitive_col = sensitive_col
        self.lambda_decay = lambda_decay
        self.gamma = gamma
        self.beta = beta
        self.k = k
        self.topn = topn
        
        self._ccp = {}  # sensitive_value -> (C,)
    
    def compute_ccp(self, train_df, users_df, t_now: Optional[float] = None):
        """
        Compute CCP o(c|s_u) for each sensitive group value.
        """
        if t_now is None:
            t_now = train_df["timestamp"].max()
        
        # user_id -> m(c|u): time-decayed category preferences
        user_prefs = {}
        for uid, group in train_df.groupby("user_id"):
            decay = np.exp(-self.lambda_decay * (t_now - group["timestamp"].values))
            w_sum = decay.sum()
            cat_vecs = np.array([
                self.phi.get(iid, np.zeros(len(self.cat_cols)))
                for iid in group["item_id"].values
            ])
            if w_sum > 0:
                user_prefs[uid] = (cat_vecs * decay[:, None]).sum(axis=0) / w_sum
            else:
                user_prefs[uid] = cat_vecs.mean(axis=0)
        
        # CCP per sensitive group
        user_sens = users_df.set_index("user_id")[self.sensitive_col].to_dict()
        
        all_vals = set(user_sens.values())
        for sv in all_vals:
            other_users = [u for u, s in user_sens.items() if s != sv and u in user_prefs]
            if other_users:
                self._ccp[sv] = np.stack([user_prefs[u] for u in other_users]).mean(axis=0)
            else:
                self._ccp[sv] = np.ones(len(self.cat_cols)) / len(self.cat_cols)
        
        return self._ccp
    
    def rerank_user(
        self,
        user_id,
        sensitive_value,
        candidate_items: List,  # list of item_ids
        base_scores: np.ndarray,
    ) -> List:
        target = self._ccp.get(sensitive_value, np.ones(len(self.cat_cols)) / len(self.cat_cols))
        
        phi_pool = np.array([
            self.phi.get(iid, np.zeros(len(self.cat_cols)))
            for iid in candidate_items
        ])
        
        # Single-level greedy
        targets = {1: target}
        phi_per_level = {1: phi_pool}
        level_weights = {1: 1.0}
        
        pool_idx = np.arange(len(candidate_items))
        result_idx = greedy_rerank(
            base_scores, pool_idx, targets, phi_per_level,
            self.k, self.beta, self.gamma, level_weights
        )
        return [candidate_items[i] for i in result_idx]


# ──────────────────────────────────────────────────────────────────────────────
# AHF Exact Reranker
# ──────────────────────────────────────────────────────────────────────────────

class AHFReranker:
    """
    Full AHF: Bayesian targets + hierarchical KL.
    """
    
    def __init__(
        self,
        phi: Dict,
        hierarchy: Dict,    # level -> {cats, phi, n_cats}
        beta: float = 0.6,
        gamma: float = 0.1,
        k: int = 20,
        topn: int = 100,
    ):
        self.phi = phi
        self.hierarchy = hierarchy
        self.beta = beta
        self.gamma = gamma
        self.k = k
        self.topn = topn
        self.levels = sorted(hierarchy.keys())
        
        # level_weights: w_l ∝ 1/|C_l|
        raw = {l: 1.0 / hierarchy[l]["n_cats"] for l in self.levels}
        total_w = sum(raw.values())
        self.level_weights = {l: raw[l] / total_w for l in self.levels}
    
    def rerank_user(
        self,
        user_idx: int,
        targets: np.ndarray,       # (C_fine,) per-user blended target
        targets_per_level: Dict,   # level -> (C_l,) aggregated targets
        candidate_items: List,
        base_scores: np.ndarray,
    ) -> List:
        phi_per_level = {}
        for l in self.levels:
            phi_l = self.hierarchy[l]["phi"]
            phi_per_level[l] = np.array([
                phi_l.get(iid, np.zeros(self.hierarchy[l]["n_cats"]))
                for iid in candidate_items
            ])
        
        pool_idx = np.arange(len(candidate_items))
        result_idx = greedy_rerank(
            base_scores, pool_idx, targets_per_level, phi_per_level,
            self.k, self.beta, self.gamma, self.level_weights
        )
        return [candidate_items[i] for i in result_idx]


# ──────────────────────────────────────────────────────────────────────────────
# LSH Sketch
# ──────────────────────────────────────────────────────────────────────────────

class RandomHyperplaneLSH:
    """
    Simple random hyperplane LSH for similarity hashing of probability vectors.
    Signature = binary hash of shape (n_bits,).
    """
    
    def __init__(self, dim: int, n_bits: int = 32, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.planes = rng.randn(n_bits, dim)  # (n_bits, dim)
    
    def hash(self, v: np.ndarray) -> int:
        """v: (dim,) → integer hash."""
        bits = (self.planes @ v) > 0  # (n_bits,)
        val = 0
        for b in bits:
            val = (val << 1) | int(b)
        return val
    
    def hash_batch(self, V: np.ndarray) -> np.ndarray:
        """V: (U, dim) → (U,) integer hashes."""
        signs = (V @ self.planes.T) > 0  # (U, n_bits)
        hashes = np.packbits(signs.astype(np.uint8), axis=1, bitorder="big")
        # Use first 4 bytes as integer
        result = hashes[:, :4].view(np.uint32).flatten().astype(int)
        return result


def ahf_sketch(
    user_targets: np.ndarray,          # (U, C) blended targets
    user_targets_per_level: Dict,      # level -> (U, C_l)
    candidate_scores_per_user: Dict,   # user_idx -> (N_pool,) base scores
    candidate_items_per_user: Dict,    # user_idx -> list of item_ids
    hierarchy: Dict,
    phi: Dict,
    beta: float,
    gamma: float,
    k: int,
    n_buckets: Optional[int] = None,
    seed: int = 42,
) -> Dict[int, List]:
    """
    Algorithm 1 from the paper: LSH sketch approximation.
    
    Returns {user_idx: ranked_list_of_item_ids}
    """
    U = user_targets.shape[0]
    if n_buckets is None:
        n_buckets = max(1, int(math.sqrt(U)))
    
    levels = sorted(hierarchy.keys())
    level_weights_raw = {l: 1.0 / hierarchy[l]["n_cats"] for l in levels}
    total_w = sum(level_weights_raw.values())
    level_weights = {l: level_weights_raw[l] / total_w for l in levels}
    
    # Step 1: LSH signatures
    lsh = RandomHyperplaneLSH(dim=user_targets.shape[1], n_bits=32, seed=seed)
    hashes = lsh.hash_batch(user_targets)  # (U,)
    
    # Step 2: Cluster into n_buckets via hash modulo
    bucket_ids = hashes % n_buckets  # (U,)
    buckets: Dict[int, List[int]] = {}
    for u, b in enumerate(bucket_ids):
        buckets.setdefault(b, []).append(u)
    
    # Step 3: Per-bucket template
    templates: Dict[int, List] = {}  # bucket_id -> ordered list of item positions
    bucket_centroids: Dict[int, np.ndarray] = {}
    bucket_centroids_per_level: Dict[int, Dict] = {}
    
    for b, users_in_bucket in buckets.items():
        centroid = user_targets[users_in_bucket].mean(axis=0)
        centroid /= centroid.sum() + EPS10
        bucket_centroids[b] = centroid
        
        c_per_level = {}
        for l in levels:
            cl = user_targets_per_level[l][users_in_bucket].mean(axis=0)
            cl /= cl.sum() + EPS10
            c_per_level[l] = cl
        bucket_centroids_per_level[b] = c_per_level
        
        # Pick a representative user for the template
        rep_user = users_in_bucket[0]
        rep_candidates = candidate_items_per_user[rep_user]
        rep_scores = candidate_scores_per_user[rep_user]
        
        phi_per_level = {}
        for l in levels:
            phi_l = hierarchy[l]["phi"]
            phi_per_level[l] = np.array([
                phi_l.get(iid, np.zeros(hierarchy[l]["n_cats"]))
                for iid in rep_candidates
            ])
        
        template = greedy_rerank(
            rep_scores,
            np.arange(len(rep_candidates)),
            c_per_level,
            phi_per_level,
            k, beta, gamma, level_weights
        )
        templates[b] = template  # indices into rep_candidates
        # Note: templates store indices; we map to item_ids per user later
    
    # Step 4: Per-user lightweight adjustment
    results = {}
    
    for b, users_in_bucket in buckets.items():
        centroid = bucket_centroids[b]
        template = templates[b]
        
        for u in users_in_bucket:
            delta = user_targets[u] - centroid  # (C,)
            
            cands = candidate_items_per_user[u]
            base_scores = candidate_scores_per_user[u]
            
            # Re-score template items
            scored = []
            for idx in template:
                if idx < len(cands):
                    iid = cands[idx]
                    phi_v = phi.get(iid, np.zeros(user_targets.shape[1]))
                    adj_score = base_scores[idx] - beta * float(delta @ phi_v)
                    scored.append((adj_score, iid))
            
            scored.sort(key=lambda x: -x[0])
            results[u] = [iid for _, iid in scored[:k]]
    
    return results


# ──────────────────────────────────────────────────────────────────────────────
# FA*IR baseline (simplified)
# ──────────────────────────────────────────────────────────────────────────────

class FAIRReranker:
    """
    Simplified FA*IR: ensures at least p_min proportion of each protected group
    appears in top-k. Here we adapt to category fairness by ensuring each
    category gets minimum representation.
    """
    
    def __init__(self, phi, cat_cols, beta=0.5, k=20, min_cat_frac=None):
        self.phi = phi
        self.cat_cols = cat_cols
        self.beta = beta
        self.k = k
        n_cats = len(cat_cols)
        self.min_cat_frac = min_cat_frac or (1.0 / n_cats)
    
    def rerank_user(self, candidate_items, base_scores) -> List:
        n_cats = len(self.cat_cols)
        cat_counts = np.zeros(n_cats)
        selected = []
        remaining = list(range(len(candidate_items)))
        
        for pos in range(self.k):
            if not remaining:
                break
            
            # Check which categories are under-represented
            deficit = self.min_cat_frac * (pos + 1) - cat_counts
            
            best_idx = None
            best_score = -np.inf
            
            for i in remaining:
                iid = candidate_items[i]
                phi_v = self.phi.get(iid, np.zeros(n_cats))
                fair_bonus = float(deficit @ phi_v)
                score = (1 - self.beta) * base_scores[i] + self.beta * fair_bonus
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            if best_idx is None:
                break
            
            iid = candidate_items[best_idx]
            cat_counts += self.phi.get(iid, np.zeros(n_cats))
            selected.append(iid)
            remaining.remove(best_idx)
        
        return selected


# ──────────────────────────────────────────────────────────────────────────────
# CPFair baseline (simplified consumer-provider)
# ──────────────────────────────────────────────────────────────────────────────

class CPFairReranker:
    """
    Simplified CPFair: balances consumer-side category exposure while
    maintaining relevance. Uses a softened fairness constraint.
    """
    
    def __init__(self, phi, cat_cols, target_dist, beta=0.5, k=20):
        self.phi = phi
        self.cat_cols = cat_cols
        self.target_dist = target_dist  # (C,)
        self.beta = beta
        self.k = k
    
    def rerank_user(self, candidate_items, base_scores) -> List:
        n_cats = len(self.cat_cols)
        targets = {1: self.target_dist}
        phi_per_level = {1: np.array([
            self.phi.get(iid, np.zeros(n_cats))
            for iid in candidate_items
        ])}
        level_weights = {1: 1.0}
        
        pool_idx = np.arange(len(candidate_items))
        result_idx = greedy_rerank(
            base_scores, pool_idx, targets, phi_per_level,
            self.k, self.beta, self.gamma if hasattr(self, "gamma") else 0.1,
            level_weights
        )
        return [candidate_items[i] for i in result_idx]
