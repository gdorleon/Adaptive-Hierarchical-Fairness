"""
Bayesian Hierarchical Preference Model (Section 4.1 of AHF paper).

Three-level Gaussian hierarchy:
    theta_{u,c} ~ N(mu_{g(u),c}, sigma_ind^2)   [user-level log-affinity]
    mu_{g,c}    ~ N(mu_global_c, sigma_group^2)  [group-level mean]
    mu_global_c ~ N(0, sigma_global^2)           [global mean]

Observation:
    x_{u,c} ~ Poisson(exp(theta_{u,c}))

Inference: SVI with mean-field variational family (Adam, minibatches).
"""
from __future__ import annotations

import math
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch.nn.functional import softmax
from tqdm import trange


class HierarchicalBayesianModel:
    """
    Variational parameters (stored after fit):
        mu_global_loc  : (C,)
        mu_global_scale: (C,)
        mu_group_loc   : (G, C)
        mu_group_scale : (G, C)
        theta_loc      : (U, C)
        theta_scale    : (U, C)
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_cats: int,
        n_groups: int,
        sigma_global: float = 1.0,
        sigma_group: float = 0.5,
        sigma_ind: float = 0.3,
        lr: float = 1e-2,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        self.n_users = n_users
        self.n_cats = n_cats
        self.n_groups = n_groups
        self.sigma_global = sigma_global
        self.sigma_group = sigma_group
        self.sigma_ind = sigma_ind
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

        # Variational parameters (initialized in fit)
        self._params_initialized = False

    # ------------------------------------------------------------------
    # Pyro model and guide
    # ------------------------------------------------------------------

    def model(self, x_uc, user_group):
        """
        x_uc      : (B, C) observed category-weighted interaction counts
        user_group: (B,)   integer group id per user
        """
        C = self.n_cats
        G = self.n_groups

        # Global mean
        mu_global = pyro.sample(
            "mu_global",
            dist.Normal(torch.zeros(C, device=self.device),
                        self.sigma_global * torch.ones(C, device=self.device))
            .to_event(1)
        )  # (C,)

        # Group means
        with pyro.plate("groups", G):
            mu_group = pyro.sample(
                "mu_group",
                dist.Normal(
                    mu_global.unsqueeze(0).expand(G, C),
                    self.sigma_group * torch.ones(G, C, device=self.device)
                ).to_event(1)
            )  # (G, C)

        # User affinities
        B = x_uc.shape[0]
        mu_for_users = mu_group[user_group]  # (B, C)

        with pyro.plate("users", B):
            theta = pyro.sample(
                "theta",
                dist.Normal(mu_for_users,
                            self.sigma_ind * torch.ones(B, C, device=self.device))
                .to_event(1)
            )  # (B, C)
            # Poisson likelihood
            rate = theta.exp()
            pyro.sample(
                "x_obs",
                dist.Poisson(rate).to_event(1),
                obs=x_uc,
            )

    def guide(self, x_uc, user_group):
        C = self.n_cats
        G = self.n_groups
        B = x_uc.shape[0]

        # Global mean variational params
        mu_global_loc = pyro.param(
            "mu_global_loc", torch.zeros(C, device=self.device)
        )
        mu_global_scale = pyro.param(
            "mu_global_scale",
            0.1 * torch.ones(C, device=self.device),
            constraint=dist.constraints.positive,
        )
        pyro.sample(
            "mu_global",
            dist.Normal(mu_global_loc, mu_global_scale).to_event(1)
        )

        # Group mean variational params
        mu_group_loc = pyro.param(
            "mu_group_loc", torch.zeros(G, C, device=self.device)
        )
        mu_group_scale = pyro.param(
            "mu_group_scale",
            0.1 * torch.ones(G, C, device=self.device),
            constraint=dist.constraints.positive,
        )
        with pyro.plate("groups", G):
            pyro.sample(
                "mu_group",
                dist.Normal(mu_group_loc, mu_group_scale).to_event(1)
            )

        # User affinity variational params (full U × C, index by batch)
        theta_loc = pyro.param(
            "theta_loc", torch.zeros(self.n_users, C, device=self.device)
        )
        theta_scale = pyro.param(
            "theta_scale",
            0.1 * torch.ones(self.n_users, C, device=self.device),
            constraint=dist.constraints.positive,
        )

        # Determine which users are in this batch via unique user indices
        # We pass user indices as extra context stored in x_uc's metadata
        user_indices = self._current_user_indices

        with pyro.plate("users", B):
            pyro.sample(
                "theta",
                dist.Normal(
                    theta_loc[user_indices],
                    theta_scale[user_indices]
                ).to_event(1)
            )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        x_uc: np.ndarray,     # (U, C) category-weighted interaction counts
        user_group: np.ndarray,  # (U,) integer group ids
        n_steps: int = 50_000,
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        pyro.set_rng_seed(seed)
        pyro.clear_param_store()

        x_uc_t = torch.tensor(x_uc, dtype=torch.float32, device=self.device)
        ug_t   = torch.tensor(user_group, dtype=torch.long, device=self.device)

        optimizer = Adam({"lr": self.lr})
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())

        U = x_uc.shape[0]
        losses = []

        for step in trange(n_steps, desc="SVI"):
            idx = torch.randperm(U)[:self.batch_size]
            self._current_user_indices = idx
            loss = svi.step(x_uc_t[idx], ug_t[idx])
            losses.append(loss)

            if step % 5000 == 0:
                print(f"  step {step:6d}  ELBO={-loss:.2f}")

        self._params_initialized = True
        return losses

    # ------------------------------------------------------------------
    # Extract posterior parameters
    # ------------------------------------------------------------------

    def _get_params(self):
        assert self._params_initialized, "Call fit() first."
        return {
            "mu_global_loc":   pyro.param("mu_global_loc").detach().cpu().numpy(),
            "mu_global_scale": pyro.param("mu_global_scale").detach().cpu().numpy(),
            "mu_group_loc":    pyro.param("mu_group_loc").detach().cpu().numpy(),
            "mu_group_scale":  pyro.param("mu_group_scale").detach().cpu().numpy(),
            "theta_loc":       pyro.param("theta_loc").detach().cpu().numpy(),
            "theta_scale":     pyro.param("theta_scale").detach().cpu().numpy(),
        }

    def get_group_distributions(self):
        """
        Returns:
            o_bayes (G, C): softmax of group posterior mean  (Eq. 4 in paper)
            sigma_group (G, C): posterior std for groups
        """
        params = self._get_params()
        mu_g = params["mu_group_loc"]        # (G, C)
        sig_g = params["mu_group_scale"]     # (G, C)
        
        # Softmax over categories for each group
        exp_mu = np.exp(mu_g - mu_g.max(axis=1, keepdims=True))
        o_bayes = exp_mu / exp_mu.sum(axis=1, keepdims=True)
        
        return o_bayes, sig_g

    def get_global_distribution(self):
        """
        Global category distribution from all users' theta_loc.
        """
        params = self._get_params()
        theta = params["theta_loc"]  # (U, C)
        exp_theta = np.exp(theta - theta.max(axis=1, keepdims=True))
        per_user = exp_theta / exp_theta.sum(axis=1, keepdims=True)
        return per_user.mean(axis=0)  # (C,)

    def compute_targets(self, user_group: np.ndarray, kappa: float = 0.5):
        """
        Compute blended preference targets \tilde{o}_bayes(c|u) (Eq. 6).
        
        Args:
            user_group: (U,) integer group id per user
            kappa: uncertainty scale
        Returns:
            targets (U, C): per-user blended category distribution
        """
        o_bayes, sig_g = self.get_group_distributions()  # (G, C)
        o_global = self.get_global_distribution()        # (C,)
        
        # Average posterior std per group (Eq. 5)
        sigma_bar_g = sig_g.mean(axis=1)  # (G,)
        
        # Credibility weight alpha_g
        alpha_g = 1.0 / (1.0 + kappa * sigma_bar_g)  # (G,)
        
        # Blend for each user
        U = len(user_group)
        C = o_bayes.shape[1]
        targets = np.zeros((U, C))
        
        for u in range(U):
            g = user_group[u]
            alpha = alpha_g[g]
            targets[u] = alpha * o_bayes[g] + (1 - alpha) * o_global
        
        return targets


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build x_{u,c} from training interactions
# ──────────────────────────────────────────────────────────────────────────────

def build_xuc(train_df, phi, n_users, n_cats, user2idx, item2idx):
    """
    x_{u,c} = sum_{v in V_u} phi_{v,c}
    """
    x_uc = np.zeros((n_users, n_cats), dtype=np.float32)
    for _, row in train_df.iterrows():
        uidx = int(row["user_idx"])
        iidx = int(row["item_idx"])
        item_id = row["item_id"]
        if item_id in phi:
            x_uc[uidx] += phi[item_id]
    return x_uc
