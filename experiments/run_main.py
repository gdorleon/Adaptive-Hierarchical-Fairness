"""
Main experiment runner for AHF paper.
Reproduces Tables 1–4 (main results, baselines, ablation, efficiency).

Usage:
    python experiments/run_main.py --dataset ml100k --backbone vaecf --method ahf
    python experiments/run_main.py --all   # full grid
"""
import argparse
import pickle
import time
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.recommenders import get_model, build_interaction_matrix
from src.bayesian.hierarchical_model import HierarchicalBayesianModel, build_xuc
from src.utils.hierarchy import (
    build_hierarchy_from_flat, build_yelp_hierarchy, get_level_targets
)
from src.reranking.ahf import (
    AHFReranker, KheyaReranker, FAIRReranker, ahf_sketch
)
from src.metrics.evaluation import compute_all_metrics, summarize_runs


SEEDS = [42, 43, 44, 45, 46]
K = 20
TOPN = 100

DATASET_CONFIGS = {
    "ml100k": {
        "data_path": "data/processed/ml100k.pkl",
        "beta": 0.6, "gamma": 0.1, "lambda_decay": 0.05, "kappa": 0.5,
        "svi_steps": 50_000, "n_super": 6,
        "sensitive_cols": ["gender", "age_group"],
    },
    "ml1m": {
        "data_path": "data/processed/ml1m.pkl",
        "beta": 0.5, "gamma": 0.1, "lambda_decay": 0.05, "kappa": 0.5,
        "svi_steps": 80_000, "n_super": 6,
        "sensitive_cols": ["gender", "age_group"],
    },
    "yelp": {
        "data_path": "data/processed/yelp.pkl",
        "beta": 0.7, "gamma": 0.1, "lambda_decay": 0.05, "kappa": 0.5,
        "svi_steps": 50_000, "n_super": None,  # native hierarchy
        "sensitive_cols": ["activity_group"],
    },
}

BACKBONE_NAMES = ["bmf", "wmf", "neumf", "vaecf"]
METHOD_NAMES   = ["base", "kheya", "ahf", "ahf_sketch", "fair", "cpfair"]


def run_one(dataset_name, backbone_name, method_name, seed=42, verbose=True):
    cfg = DATASET_CONFIGS[dataset_name]
    
    # Load preprocessed data
    with open(cfg["data_path"], "rb") as f:
        data = pickle.load(f)
    
    train_df = data["train"]
    val_df   = data["val"]
    test_df  = data["test"]
    phi      = data["phi"]        # item_id -> (C,)
    cat_cols = data["cat_cols"]
    n_users  = data["n_users"]
    n_items  = data["n_items"]
    n_cats   = len(cat_cols)
    users_df = data["users"]
    user2idx = data["user2idx"]
    item2idx = data["item2idx"]
    idx2item = {v: k for k, v in item2idx.items()}
    all_items = list(item2idx.keys())
    
    # Convert phi keys to item_ids (original)
    phi_by_iid = phi  # already item_id keyed
    
    # phi array indexed by item_idx
    phi_arr = np.array([phi.get(idx2item.get(i, -1), np.zeros(n_cats))
                        for i in range(n_items)])  # (n_items, C)
    
    # Test relevance sets
    test_items_per_user = {}
    for _, row in test_df.iterrows():
        uid = int(row["user_idx"])
        iid = row["item_id"]
        test_items_per_user.setdefault(uid, set()).add(iid)
    
    # ── Train base recommender ──────────────────────────────────────────────
    np.random.seed(seed)
    model = get_model(backbone_name, n_users, n_items, seed=seed)
    model.fit(train_df)
    
    # ── Get top-N candidates per user ───────────────────────────────────────
    candidate_items_per_user = {}
    candidate_scores_per_user = {}
    
    train_items_per_user = {}
    for _, row in train_df.iterrows():
        train_items_per_user.setdefault(int(row["user_idx"]), set()).add(row["item_id"])
    
    for uid in range(n_users):
        seen = train_items_per_user.get(uid, set())
        # Candidate pool: exclude training items
        cand_indices = np.array([
            i for i in range(n_items)
            if idx2item.get(i) not in seen
        ])
        if len(cand_indices) == 0:
            continue
        scores = model.score(uid, cand_indices)
        top_idx = np.argsort(scores)[::-1][:TOPN]
        top_cand_idx = cand_indices[top_idx]
        
        candidate_items_per_user[uid] = [idx2item[i] for i in top_cand_idx]
        candidate_scores_per_user[uid] = scores[top_idx]
    
    # ── Sensitive attribute mapping ─────────────────────────────────────────
    sensitive_col = cfg["sensitive_cols"][0]
    user_groups = {}
    uid_map = users_df.set_index("user_id")
    for uid_orig, uidx in user2idx.items():
        if uid_orig in uid_map.index:
            user_groups[uidx] = str(uid_map.loc[uid_orig, sensitive_col])
    
    if method_name == "base":
        # Return top-k unre-ranked
        ranked_lists = {}
        for uid, cands in candidate_items_per_user.items():
            ranked_lists[uid] = cands[:K]
        
        metrics = compute_all_metrics(
            ranked_lists, test_items_per_user,
            phi_by_iid, n_cats, user_groups, K
        )
        return metrics
    
    # ── Build category hierarchy ─────────────────────────────────────────────
    if dataset_name == "yelp":
        top_cats, fine_to_coarse = build_yelp_hierarchy(cat_cols)
        n_coarse = len(top_cats)
        phi_coarse = {}
        for iid, w in phi_by_iid.items():
            w_c = np.zeros(n_coarse)
            for c, gc in enumerate(fine_to_coarse):
                w_c[gc] += w[c]
            s = w_c.sum()
            phi_coarse[iid] = w_c / s if s > 0 else w_c
        
        hierarchy = {
            1: {"cats": cat_cols,  "phi": phi_by_iid, "n_cats": n_cats},
            2: {"cats": top_cats,  "phi": phi_coarse,  "n_cats": n_coarse},
        }
        genre_to_super = fine_to_coarse
    else:
        hierarchy, genre_to_super = build_hierarchy_from_flat(
            phi_by_iid, cat_cols, n_super=cfg["n_super"]
        )
    
    n_coarse = hierarchy[2]["n_cats"]
    
    # ── Kheya baseline ──────────────────────────────────────────────────────
    if method_name == "kheya":
        reranker = KheyaReranker(
            phi_by_iid, cat_cols, sensitive_col,
            lambda_decay=cfg["lambda_decay"], gamma=cfg["gamma"],
            beta=cfg["beta"], k=K, topn=TOPN,
        )
        reranker.compute_ccp(train_df, users_df)
        ranked_lists = {}
        for uid, cands in candidate_items_per_user.items():
            sv = user_groups.get(uid, "unknown")
            ranked_lists[uid] = reranker.rerank_user(
                uid, sv, cands, candidate_scores_per_user[uid]
            )
        return compute_all_metrics(ranked_lists, test_items_per_user,
                                   phi_by_iid, n_cats, user_groups, K)
    
    # ── Bayesian model (shared for AHF methods) ─────────────────────────────
    # Build x_{u,c}
    x_uc = build_xuc(train_df, phi_by_iid, n_users, n_cats, user2idx, item2idx)
    
    # Map users to integer group ids
    group_vals = sorted(set(user_groups.values()))
    group2idx = {g: i for i, g in enumerate(group_vals)}
    user_group_arr = np.array([group2idx.get(user_groups.get(u, group_vals[0]), 0)
                                for u in range(n_users)])
    
    bayes_model = HierarchicalBayesianModel(
        n_users=n_users, n_items=n_items, n_cats=n_cats,
        n_groups=len(group_vals),
        lr=1e-2, batch_size=256,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )
    bayes_model.fit(x_uc, user_group_arr, n_steps=cfg["svi_steps"], seed=seed)
    
    # Compute per-user blended targets
    targets_fine = bayes_model.compute_targets(user_group_arr, kappa=cfg["kappa"])  # (U, C)
    targets_coarse = get_level_targets(targets_fine, genre_to_super, n_coarse)  # (U, C_coarse)
    
    targets_per_level = {1: targets_fine, 2: targets_coarse}
    
    if method_name == "ahf":
        reranker = AHFReranker(phi_by_iid, hierarchy, beta=cfg["beta"],
                               gamma=cfg["gamma"], k=K, topn=TOPN)
        ranked_lists = {}
        for uid, cands in candidate_items_per_user.items():
            tpl = {l: targets_per_level[l][uid] for l in [1, 2]}
            ranked_lists[uid] = reranker.rerank_user(
                uid, targets_fine[uid], tpl, cands, candidate_scores_per_user[uid]
            )
        return compute_all_metrics(ranked_lists, test_items_per_user,
                                   phi_by_iid, n_cats, user_groups, K)
    
    if method_name == "ahf_sketch":
        n_buckets = max(1, int(np.sqrt(n_users)))
        t0 = time.time()
        ranked_lists = ahf_sketch(
            targets_fine, targets_per_level,
            candidate_scores_per_user, candidate_items_per_user,
            hierarchy, phi_by_iid,
            beta=cfg["beta"], gamma=cfg["gamma"], k=K,
            n_buckets=n_buckets, seed=seed,
        )
        elapsed = (time.time() - t0) * 1000 / len(ranked_lists)
        metrics = compute_all_metrics(ranked_lists, test_items_per_user,
                                      phi_by_iid, n_cats, user_groups, K)
        metrics["time_ms_per_user"] = elapsed
        return metrics
    
    raise ValueError(f"Unknown method: {method_name}")


def run_grid(datasets=None, backbones=None, methods=None, n_runs=5):
    datasets  = datasets  or list(DATASET_CONFIGS.keys())
    backbones = backbones or BACKBONE_NAMES
    methods   = methods   or ["base", "kheya", "ahf"]
    
    all_results = {}
    
    for ds in datasets:
        for bb in backbones:
            for method in methods:
                key = (ds, bb, method)
                print(f"\n{'='*60}")
                print(f"  Dataset={ds}  Backbone={bb}  Method={method}")
                print(f"{'='*60}")
                
                run_results = []
                for seed in SEEDS[:n_runs]:
                    metrics = run_one(ds, bb, method, seed=seed)
                    run_results.append(metrics)
                    print(f"  seed={seed}  {metrics}")
                
                from src.metrics.evaluation import summarize_runs
                summary = summarize_runs(run_results)
                all_results[key] = summary
                print(f"\n  SUMMARY: {summary}")
    
    return all_results


def print_table(all_results):
    """Print results in LaTeX-style table format."""
    print("\n" + "="*100)
    print(f"{'Dataset':<10} {'Backbone':<10} {'Method':<15} "
          f"{'NDCG@20':>12} {'CC-Disp':>12} {'CDCG-Disp':>12}")
    print("-"*100)
    
    for (ds, bb, method), summary in sorted(all_results.items()):
        ndcg = summary.get("ndcg@k", {})
        cc   = summary.get("cc_disp", {})
        cdcg = summary.get("cdcg_disp", {})
        print(f"{ds:<10} {bb:<10} {method:<15} "
              f"{ndcg.get('mean', 0):.3f}±{ndcg.get('std', 0):.3f}  "
              f"{cc.get('mean', 0):.3f}±{cc.get('std', 0):.3f}  "
              f"{cdcg.get('mean', 0):.3f}±{cdcg.get('std', 0):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  default="ml100k",
                        choices=list(DATASET_CONFIGS.keys()) + ["all"])
    parser.add_argument("--backbone", default="vaecf",
                        choices=BACKBONE_NAMES + ["all"])
    parser.add_argument("--method",   default="ahf",
                        choices=METHOD_NAMES + ["all"])
    parser.add_argument("--n_runs",   type=int, default=5)
    parser.add_argument("--all",      action="store_true", help="Run full grid")
    args = parser.parse_args()
    
    if args.all or args.dataset == "all":
        results = run_grid(n_runs=args.n_runs)
    else:
        datasets  = list(DATASET_CONFIGS.keys()) if args.dataset == "all" else [args.dataset]
        backbones = BACKBONE_NAMES if args.backbone == "all" else [args.backbone]
        methods   = METHOD_NAMES   if args.method   == "all" else [args.method]
        results   = run_grid(datasets, backbones, methods, n_runs=args.n_runs)
    
    print_table(results)
