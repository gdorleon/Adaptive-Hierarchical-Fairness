# Adaptive Hierarchical Fairness (AHF)

> **"Beyond Static Aggregation: Adaptive Hierarchical Fairness for Recommendation Systems"**

A post-processing re-ranking framework that reduces demographic category-distribution biases in recommendation systems through:
1. **Bayesian hierarchical preference modeling** with uncertainty-aware blending
2. **Multi-granularity fairness** over category hierarchies (hierarchical KL divergence)
3. **Theoretical bounds** linking per-user KL control to demographic parity gaps
4. **LSH sketch approximation** for 22–41× speedups

---

## Results Summary

| Dataset | Backbone | Base CC-Disp | Kheya CC-Disp | AHF CC-Disp | NDCG@20 |
|---------|----------|-------------|--------------|------------|---------|
| ML100K  | VAE-CF   | 0.215       | 0.097        | **0.067**  | **0.481** |
| ML1M    | VAE-CF   | 0.201       | 0.091        | **0.060**  | **0.568** |
| Yelp    | VAE-CF   | 0.183       | 0.089        | **0.064**  | **0.388** |

AHF reduces CC disparity **31–48%** relative to the strongest prior baseline while improving NDCG.

---

## Installation

```bash
git clone https://github.com/yourname/ahf
cd ahf
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch 2.0+, Pyro 1.9+

---

## Data

Download and preprocess:
```bash
bash data/download.sh          # downloads ML100K, ML1M, Yelp
python src/data/preprocess.py  # creates processed splits
```

---

## Quick Start

### Run all experiments (main table)
```bash
python experiments/run_main.py --dataset ml100k --backbone vaecf --method ahf
python experiments/run_main.py --dataset ml1m   --backbone vaecf --method ahf
python experiments/run_main.py --dataset yelp   --backbone vaecf --method ahf
```

### Run full main results table (all backbones × datasets)
```bash
python experiments/run_main.py --all
```

### Ablation study
```bash
python experiments/run_ablation.py --dataset ml100k --backbone vaecf
```

### Efficiency analysis (exact vs. sketch)
```bash
python experiments/run_efficiency.py --dataset ml1m --backbone vaecf
```

---

## Repository Structure

```
ahf/
├── src/
│   ├── data/           # Dataset loading and preprocessing
│   ├── models/         # Base recommenders (BMF, WMF, NeuMF, VAE-CF)
│   ├── bayesian/       # Hierarchical Bayesian preference model (Pyro/SVI)
│   ├── reranking/      # AHF, Kheya et al., FA*IR, CPFair re-rankers
│   ├── metrics/        # NDCG, CC-Disparity, CDCG-Disparity
│   └── utils/          # Category hierarchy construction, LSH sketch
├── experiments/        # Experiment scripts
├── configs/            # YAML configs per dataset
└── data/               # Raw and processed data
```

---

## Configuration

Edit `configs/ml100k.yaml` etc. Key hyperparameters:

```yaml
beta: 0.6          # fairness weight in re-ranking objective
gamma: 0.1         # position discount exponent for RCP
lambda_decay: 0.05 # time decay for CCP
kappa: 0.5         # uncertainty scale factor
n_buckets: auto    # sqrt(|U|) for sketch
svi_steps: 50000   # variational inference steps
```

---

## Citation

```bibtex
@inproceedings{ahf2024,
  title={Beyond Static Aggregation: Adaptive Hierarchical Fairness for Recommendation Systems},
  booktitle={...},
  year={2026}
}
```
