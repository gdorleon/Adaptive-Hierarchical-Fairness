"""
Efficiency analysis: Exact AHF vs Sketch AHF (Table 4).
ML1M + VAE-CF.

Usage:
    python experiments/run_efficiency.py --dataset ml1m --backbone vaecf
"""
import argparse
import pickle
import time
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.recommenders import get_model
from src.bayesian.hierarchical_model import HierarchicalBayesianModel, build_xuc
from src.utils.hierarchy import build_hierarchy_from_flat, get_level_targets
from src.reranking.ahf import AHFReranker, ahf_sketch
from src.metrics.evaluation import compute_all_metrics, summarize_runs

SEEDS = [42, 43, 44, 45, 46]
K = 20
TOPN = 100


def run_efficiency(dataset_name="ml1m", backbone="vaecf", n_runs=5):
    cfg = {
        "ml1m": {"data_path": "data/processed/ml1m.pkl",
                 "beta": 0.5, "gamma": 0.1, "lambda_decay": 0.05, "kappa": 0.5,
                 "svi_steps": 80_000, "n_super": 6,
                 "sensitive_cols": ["gender", "age_group"]},
        "ml100k": {"data_path": "data/processed/ml100k.pkl",
                   "beta": 0.6, "gamma": 0.1, "lambda_decay": 0.05, "kappa": 0.5,
                   "svi_steps": 50_000, "n_super": 6,
                   "sensitive_cols": ["gender", "age_group"]},
    }[dataset_name]
    
    with open(cfg["data_path"], "rb") as f:
        data = pickle.load(f)
    
    train_df = data["train"]
    phi      = data["phi"]
    cat_cols = data["cat_cols"]
    n_users  = data["n_users"]
    n_items  = data["n_items"]
    n_cats   = len(cat_cols)
    users_df = data["users"]
    user2idx = data["user2idx"]
    item2idx = data["item2idx"]
    idx2item = {v: k for k, v in item2idx.items()}
    
    test_items_per_user = {}
    for _, row in data["test"].iterrows():
        test_items_per_user.setdefault(int(row["user_idx"]), set()).add(row["item_id"])
    
    sensitive_col = cfg["sensitive_cols"][0]
    uid_map = users_df.set_index("user_id")
    user_groups = {}
    for uid_orig, uidx in user2idx.items():
        if uid_orig in uid_map.index:
            user_groups[uidx] = str(uid_map.loc[uid_orig, sensitive_col])
    
    group_vals = sorted(set(user_groups.values()))
    group2idx  = {g: i for i, g in enumerate(group_vals)}
    user_group_arr = np.array([group2idx.get(user_groups.get(u, group_vals[0]), 0)
                                for u in range(n_users)])
    
    exact_results = []
    sketch_results = []
    exact_times   = []
    sketch_times  = []
    
    for seed in SEEDS[:n_runs]:
        np.random.seed(seed)
        model = get_model(backbone, n_users, n_items, seed=seed)
        model.fit(train_df)
        
        train_items_per_user = {}
        for _, row in train_df.iterrows():
            train_items_per_user.setdefault(int(row["user_idx"]), set()).add(row["item_id"])
        
        candidate_items_per_user = {}
        candidate_scores_per_user = {}
        for uid in range(n_users):
            seen = train_items_per_user.get(uid, set())
            cand_indices = np.array([i for i in range(n_items)
                                     if idx2item.get(i) not in seen])
            if len(cand_indices) == 0:
                continue
            scores = model.score(uid, cand_indices)
            top_idx = np.argsort(scores)[::-1][:TOPN]
            candidate_items_per_user[uid]  = [idx2item[i] for i in cand_indices[top_idx]]
            candidate_scores_per_user[uid] = scores[top_idx]
        
        x_uc = build_xuc(train_df, phi, n_users, n_cats, user2idx, item2idx)
        bayes = HierarchicalBayesianModel(
            n_users=n_users, n_items=n_items, n_cats=n_cats,
            n_groups=len(group_vals), lr=1e-2, batch_size=256,
        )
        bayes.fit(x_uc, user_group_arr, n_steps=cfg["svi_steps"], seed=seed)
        
        targets_fine = bayes.compute_targets(user_group_arr, cfg["kappa"])
        hierarchy, g2s = build_hierarchy_from_flat(phi, cat_cols, cfg["n_super"])
        n_coarse = hierarchy[2]["n_cats"]
        targets_coarse = get_level_targets(targets_fine, g2s, n_coarse)
        targets_per_level = {1: targets_fine, 2: targets_coarse}
        
        # ── Exact AHF ───────────────────────────────────────────────────────
        reranker = AHFReranker(phi, hierarchy, beta=cfg["beta"],
                               gamma=cfg["gamma"], k=K)
        t0 = time.time()
        rls_exact = {}
        for uid, cands in candidate_items_per_user.items():
            tpl = {l: targets_per_level[l][uid] for l in [1, 2]}
            rls_exact[uid] = reranker.rerank_user(
                uid, targets_fine[uid], tpl, cands, candidate_scores_per_user[uid]
            )
        elapsed_exact = (time.time() - t0) * 1000 / len(rls_exact)
        exact_times.append(elapsed_exact)
        exact_results.append(
            compute_all_metrics(rls_exact, test_items_per_user, phi, n_cats, user_groups, K))
        
        # ── Sketch AHF ──────────────────────────────────────────────────────
        n_buckets = max(1, int(np.sqrt(n_users)))
        t0 = time.time()
        rls_sketch = ahf_sketch(
            targets_fine, targets_per_level,
            candidate_scores_per_user, candidate_items_per_user,
            hierarchy, phi,
            beta=cfg["beta"], gamma=cfg["gamma"], k=K,
            n_buckets=n_buckets, seed=seed,
        )
        elapsed_sketch = (time.time() - t0) * 1000 / len(rls_sketch)
        sketch_times.append(elapsed_sketch)
        sketch_results.append(
            compute_all_metrics(rls_sketch, test_items_per_user, phi, n_cats, user_groups, K))
    
    # Summarize
    exact_summary  = summarize_runs(exact_results)
    sketch_summary = summarize_runs(sketch_results)
    
    print(f"\nEfficiency Analysis ({dataset_name.upper()}, {backbone.upper()})")
    print(f"{'Method':<15} {'NDCG@20':>12} {'CC-Disp':>12} {'Time (ms/user)':>16}")
    print("-"*60)
    
    for name, summary, times in [
        ("Exact AHF",  exact_summary,  exact_times),
        ("Sketch AHF", sketch_summary, sketch_times),
    ]:
        ndcg = summary["ndcg@k"]
        cc   = summary["cc_disp"]
        t_mean = np.mean(times)
        t_std  = np.std(times)
        print(f"{name:<15} {ndcg['mean']:.3f}±{ndcg['std']:.3f}  "
              f"{cc['mean']:.3f}±{cc['std']:.3f}  {t_mean:.2f}±{t_std:.2f}")
    
    speedup = np.mean(exact_times) / np.mean(sketch_times)
    print(f"\n  Speedup: {speedup:.1f}×")
    
    return {"exact": exact_summary, "sketch": sketch_summary,
            "exact_times": exact_times, "sketch_times": sketch_times}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  default="ml1m")
    parser.add_argument("--backbone", default="vaecf")
    parser.add_argument("--n_runs",   type=int, default=5)
    args = parser.parse_args()
    run_efficiency(args.dataset, args.backbone, args.n_runs)
