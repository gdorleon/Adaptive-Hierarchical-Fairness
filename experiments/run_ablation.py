"""
Ablation study for AHF (Table 3 in paper).
ML100K + VAE-CF. Each row adds one component.

Steps:
  1. Kheya et al. (baseline)
  2. + Bayesian targets (Eq. 4 — no uncertainty blending)
  3. + Uncertainty blending (Eq. 6)
  4. + Multi-granular objective (L=2)
  5. + Multi-granular objective (L=3)
  6. Full AHF

Usage:
    python experiments/run_ablation.py --dataset ml100k --backbone vaecf
"""
import argparse
import pickle
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.recommenders import get_model
from src.bayesian.hierarchical_model import HierarchicalBayesianModel, build_xuc
from src.utils.hierarchy import build_hierarchy_from_flat, get_level_targets
from src.reranking.ahf import greedy_rerank, KheyaReranker
from src.metrics.evaluation import compute_all_metrics, summarize_runs

SEEDS = [42, 43, 44, 45, 46]
K = 20
TOPN = 100


def run_ablation(dataset_name="ml100k", backbone="vaecf", n_runs=5):
    cfg = {
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
    
    group_vals  = sorted(set(user_groups.values()))
    group2idx   = {g: i for i, g in enumerate(group_vals)}
    user_group_arr = np.array([group2idx.get(user_groups.get(u, group_vals[0]), 0)
                                for u in range(n_users)])
    
    ablation_results = {
        "kheya":          [],
        "+bayes_targets": [],
        "+uncertainty":   [],
        "+multigran_L2":  [],
        "+multigran_L3":  [],
        "full_ahf":       [],
    }
    
    for seed in SEEDS[:n_runs]:
        np.random.seed(seed)
        
        # ── Train backbone ──────────────────────────────────────────────────
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
        
        # ── Kheya baseline ──────────────────────────────────────────────────
        reranker = KheyaReranker(phi, cat_cols, sensitive_col,
                                 lambda_decay=cfg["lambda_decay"], gamma=cfg["gamma"],
                                 beta=cfg["beta"], k=K)
        reranker.compute_ccp(train_df, users_df)
        rls = {}
        for uid, cands in candidate_items_per_user.items():
            sv = user_groups.get(uid, group_vals[0])
            rls[uid] = reranker.rerank_user(uid, sv, cands, candidate_scores_per_user[uid])
        ablation_results["kheya"].append(
            compute_all_metrics(rls, test_items_per_user, phi, n_cats, user_groups, K))
        
        # ── Bayesian model ──────────────────────────────────────────────────
        x_uc = build_xuc(train_df, phi, n_users, n_cats, user2idx, item2idx)
        
        bayes = HierarchicalBayesianModel(
            n_users=n_users, n_items=n_items, n_cats=n_cats,
            n_groups=len(group_vals), lr=1e-2, batch_size=256,
        )
        bayes.fit(x_uc, user_group_arr, n_steps=cfg["svi_steps"], seed=seed)
        
        # Bayesian targets WITHOUT blending (kappa=0 → alpha=1 always)
        targets_no_blend = bayes.compute_targets(user_group_arr, kappa=0.0)
        
        # + Bayesian targets (no uncertainty)
        hierarchy2, g2s = build_hierarchy_from_flat(phi, cat_cols, cfg["n_super"])
        n_coarse = hierarchy2[2]["n_cats"]
        
        for variant, targets_fine, n_hier_levels in [
            ("+bayes_targets", targets_no_blend, 1),
            ("+uncertainty",   bayes.compute_targets(user_group_arr, cfg["kappa"]), 1),
            ("+multigran_L2",  bayes.compute_targets(user_group_arr, cfg["kappa"]), 2),
            ("+multigran_L3",  bayes.compute_targets(user_group_arr, cfg["kappa"]), 2),
            ("full_ahf",       bayes.compute_targets(user_group_arr, cfg["kappa"]), 2),
        ]:
            targets_coarse = get_level_targets(targets_fine, g2s, n_coarse)
            
            if n_hier_levels == 1:
                active_levels = [1]
            else:
                active_levels = [1, 2]
            
            hier_sub = {l: hierarchy2[l] for l in active_levels}
            raw_w = {l: 1.0 / hier_sub[l]["n_cats"] for l in active_levels}
            tw = sum(raw_w.values())
            lw = {l: raw_w[l] / tw for l in active_levels}
            
            rls = {}
            for uid, cands in candidate_items_per_user.items():
                tpl = {}
                tpl[1] = targets_fine[uid]
                if 2 in active_levels:
                    tpl[2] = targets_coarse[uid]
                
                phi_per_level = {}
                for l in active_levels:
                    phi_l = hier_sub[l]["phi"]
                    phi_per_level[l] = np.array([
                        phi_l.get(iid, np.zeros(hier_sub[l]["n_cats"]))
                        for iid in cands
                    ])
                
                pool_idx = np.arange(len(cands))
                result_idx = greedy_rerank(
                    candidate_scores_per_user[uid], pool_idx, tpl, phi_per_level,
                    K, cfg["beta"], cfg["gamma"], lw
                )
                rls[uid] = [cands[i] for i in result_idx]
            
            ablation_results[variant].append(
                compute_all_metrics(rls, test_items_per_user, phi, n_cats, user_groups, K))
    
    # Print results
    print("\nAblation Study Results (ML100K, VAE-CF)")
    print(f"{'Method':<25} {'NDCG@20':>12} {'CC-Disp':>12} {'delta CC-Disp':>14}")
    print("-" * 65)
    
    prev_cc = None
    for variant, results in ablation_results.items():
        summary = summarize_runs(results)
        ndcg = summary["ndcg@k"]
        cc   = summary["cc_disp"]
        
        if prev_cc is not None:
            delta = (cc["mean"] - prev_cc) / prev_cc * 100
            delta_str = f"{delta:+.1f}%"
        else:
            delta_str = "---"
        
        print(f"{variant:<25} {ndcg['mean']:.3f}±{ndcg['std']:.3f}  "
              f"{cc['mean']:.3f}±{cc['std']:.3f}  {delta_str:>14}")
        prev_cc = cc["mean"]
    
    return ablation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  default="ml100k")
    parser.add_argument("--backbone", default="vaecf")
    parser.add_argument("--n_runs",   type=int, default=5)
    args = parser.parse_args()
    run_ablation(args.dataset, args.backbone, args.n_runs)
