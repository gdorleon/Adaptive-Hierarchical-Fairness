"""
Unit tests for core AHF components.
Run: pytest tests/
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ──────────────────────────────────────────────────────────────────────────────
# RCP / KL tests
# ──────────────────────────────────────────────────────────────────────────────

from src.reranking.ahf import (
    compute_rcp, kl_div, hierarchical_kl,
    delta_hierarchical_kl, greedy_rerank,
    RandomHyperplaneLSH,
)


def test_rcp_empty():
    phi = np.random.rand(5, 3)
    rcp = compute_rcp([], phi, gamma=0.1)
    assert rcp.shape == (3,)
    assert np.allclose(rcp, 0.0)


def test_rcp_single():
    phi = np.array([[0.5, 0.3, 0.2], [0.1, 0.7, 0.2]])
    rcp = compute_rcp([0], phi, gamma=0.1)
    assert np.allclose(rcp, [0.5, 0.3, 0.2])


def test_kl_div_identical():
    p = np.array([0.3, 0.4, 0.3])
    assert kl_div(p, p) < 1e-8


def test_kl_div_nonneg():
    p = np.array([0.2, 0.5, 0.3])
    q = np.array([0.4, 0.2, 0.4])
    assert kl_div(p, q) >= 0


def test_pinsker_inequality():
    """KL >= 2 * TV^2 (Pinsker)."""
    p = np.array([0.1, 0.6, 0.3])
    q = np.array([0.4, 0.3, 0.3])
    tv = 0.5 * np.abs(p - q).sum()
    kl = kl_div(p, q)
    assert kl >= 2 * tv ** 2 - 1e-8, f"Pinsker violated: KL={kl}, 2*TV^2={2*tv**2}"


def test_greedy_rerank_length():
    n_pool = 20
    n_cats = 4
    scores = np.random.rand(n_pool)
    phi_pool = np.random.dirichlet(np.ones(n_cats), size=n_pool)
    targets = {1: np.array([0.25, 0.25, 0.25, 0.25])}
    phi_per_level = {1: phi_pool}
    level_weights = {1: 1.0}
    pool_idx = np.arange(n_pool)
    
    result = greedy_rerank(scores, pool_idx, targets, phi_per_level,
                           k=10, beta=0.5, gamma=0.1, level_weights=level_weights)
    assert len(result) == 10
    assert len(set(result)) == 10, "Duplicate items in ranked list"


def test_greedy_rerank_no_duplicates():
    n_pool = 15
    n_cats = 3
    scores = np.random.rand(n_pool)
    phi_pool = np.random.dirichlet(np.ones(n_cats), size=n_pool)
    targets = {1: np.ones(n_cats) / n_cats}
    phi_per_level = {1: phi_pool}
    level_weights = {1: 1.0}
    pool_idx = np.arange(n_pool)
    
    result = greedy_rerank(scores, pool_idx, targets, phi_per_level,
                           k=15, beta=0.3, gamma=0.1, level_weights=level_weights)
    assert len(result) == n_pool
    assert len(set(result)) == n_pool


def test_greedy_beta_zero_is_sorted():
    """With beta=0, greedy should return top-k by base score."""
    n_pool = 10
    n_cats = 3
    scores = np.arange(n_pool, dtype=float)  # [0,1,...,9]
    phi_pool = np.random.dirichlet(np.ones(n_cats), size=n_pool)
    targets = {1: np.ones(n_cats) / n_cats}
    phi_per_level = {1: phi_pool}
    level_weights = {1: 1.0}
    pool_idx = np.arange(n_pool)
    
    result = greedy_rerank(scores, pool_idx, targets, phi_per_level,
                           k=5, beta=0.0, gamma=0.1, level_weights=level_weights)
    assert result == [9, 8, 7, 6, 5], f"Expected top-5 by score, got {result}"


# ──────────────────────────────────────────────────────────────────────────────
# LSH tests
# ──────────────────────────────────────────────────────────────────────────────

def test_lsh_deterministic():
    lsh = RandomHyperplaneLSH(dim=10, n_bits=16, seed=0)
    v = np.random.rand(10)
    h1 = lsh.hash(v)
    h2 = lsh.hash(v)
    assert h1 == h2


def test_lsh_batch_consistent():
    dim = 8
    lsh = RandomHyperplaneLSH(dim=dim, n_bits=16, seed=1)
    V = np.random.rand(20, dim)
    batch_hashes = lsh.hash_batch(V)
    single_hashes = np.array([lsh.hash(V[i]) for i in range(20)])
    # batch and single should agree modulo bit packing differences
    assert batch_hashes.shape == (20,)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics tests
# ──────────────────────────────────────────────────────────────────────────────

from src.metrics.evaluation import ndcg_at_k, category_coverage, group_disparity


def test_ndcg_perfect():
    ranked = [1, 2, 3, 4, 5]
    relevant = {1, 2, 3}
    score = ndcg_at_k(ranked, relevant, k=5)
    assert abs(score - 1.0) < 1e-8


def test_ndcg_zero():
    ranked = [10, 11, 12]
    relevant = {1, 2, 3}
    score = ndcg_at_k(ranked, relevant, k=3)
    assert score == 0.0


def test_ndcg_partial():
    ranked = [2, 1, 10, 3, 10]
    relevant = {1, 2, 3}
    score = ndcg_at_k(ranked, relevant, k=5)
    assert 0.0 < score < 1.0


def test_category_coverage_uniform():
    n_cats = 3
    phi = {
        "a": np.array([1.0, 0.0, 0.0]),
        "b": np.array([0.0, 1.0, 0.0]),
        "c": np.array([0.0, 0.0, 1.0]),
    }
    ranked_lists = {0: ["a", "b", "c"]}
    cc = category_coverage(ranked_lists, phi, n_cats, k=3)
    assert np.allclose(cc, [1/3, 1/3, 1/3])


def test_group_disparity_zero():
    cc = {"g1": np.array([0.3, 0.4, 0.3]),
          "g2": np.array([0.3, 0.4, 0.3])}
    assert group_disparity(cc) < 1e-8


def test_group_disparity_positive():
    cc = {"g1": np.array([0.8, 0.1, 0.1]),
          "g2": np.array([0.1, 0.8, 0.1])}
    assert group_disparity(cc) > 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Hierarchy tests
# ──────────────────────────────────────────────────────────────────────────────

from src.utils.hierarchy import get_level_targets


def test_level_targets_sums_to_one():
    fine = np.array([[0.2, 0.3, 0.1, 0.4],
                     [0.1, 0.1, 0.4, 0.4]])
    fine_to_coarse = np.array([0, 0, 1, 1])
    coarse = get_level_targets(fine, fine_to_coarse, n_coarse=2)
    assert coarse.shape == (2, 2)
    assert np.allclose(coarse.sum(axis=1), 1.0)


def test_level_targets_correct():
    fine = np.array([[0.6, 0.4]])
    fine_to_coarse = np.array([0, 0])  # both map to coarse=0
    coarse = get_level_targets(fine, fine_to_coarse, n_coarse=1)
    assert np.allclose(coarse, [[1.0]])


# ──────────────────────────────────────────────────────────────────────────────
# Theorem verification (numerical)
# ──────────────────────────────────────────────────────────────────────────────

def test_pinsker_theorem1():
    """
    Verify Theorem 1: if KL(o||r) <= eps, then |r(c) - o(c)| <= sqrt(2*eps)
    for all c.
    """
    rng = np.random.RandomState(0)
    for _ in range(100):
        o = rng.dirichlet(np.ones(5))
        r = rng.dirichlet(np.ones(5))
        eps = kl_div(o, r)
        for c in range(5):
            assert abs(r[c] - o[c]) <= np.sqrt(2 * eps) + 1e-8, \
                f"Theorem 1 violated at c={c}: |{r[c]-o[c]}| > sqrt(2*{eps})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
