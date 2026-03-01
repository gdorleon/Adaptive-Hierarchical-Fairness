"""
Base recommender models for AHF experiments.

Implements:
  - BMF  : Bayesian Matrix Factorization (Koren et al. 2009)
  - WMF  : Weighted Matrix Factorization (Hu et al. 2008)
  - NeuMF: Neural Matrix Factorization (He et al. 2017)
  - VAE-CF: Variational Autoencoder for CF (Liang et al. 2018)

All models expose a `score(user_idx, item_indices)` method.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from tqdm import trange
from typing import Dict, List, Optional


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Interaction matrix builder
# ──────────────────────────────────────────────────────────────────────────────

def build_interaction_matrix(train_df, n_users: int, n_items: int) -> csr_matrix:
    """Binary implicit feedback matrix."""
    rows = train_df["user_idx"].astype(int).values
    cols = train_df["item_idx"].astype(int).values
    data = np.ones(len(rows))
    return csr_matrix((data, (rows, cols)), shape=(n_users, n_items))


def build_rating_matrix(train_df, n_users: int, n_items: int) -> csr_matrix:
    rows = train_df["user_idx"].astype(int).values
    cols = train_df["item_idx"].astype(int).values
    data = train_df["rating"].values.astype(float)
    return csr_matrix((data, (rows, cols)), shape=(n_users, n_items))


# ──────────────────────────────────────────────────────────────────────────────
# BMF (Probabilistic Matrix Factorization / MAP)
# ──────────────────────────────────────────────────────────────────────────────

class BMF:
    """
    Bayesian (MAP) Matrix Factorization with L2 regularization.
    Optimized via SGD on rating data.
    """
    
    def __init__(
        self,
        n_users: int, n_items: int,
        n_factors: int = 64,
        lr: float = 0.01,
        reg: float = 0.01,
        n_epochs: int = 50,
        seed: int = 42,
    ):
        np.random.seed(seed)
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        
        std = 0.1
        self.U = np.random.randn(n_users, n_factors) * std
        self.V = np.random.randn(n_items, n_factors) * std
        self.bu = np.zeros(n_users)
        self.bv = np.zeros(n_items)
        self.global_mean = 0.0
    
    def fit(self, train_df):
        users  = train_df["user_idx"].astype(int).values
        items  = train_df["item_idx"].astype(int).values
        ratings = train_df["rating"].values.astype(float)
        self.global_mean = ratings.mean()
        
        for epoch in range(self.n_epochs):
            idx = np.random.permutation(len(users))
            total_loss = 0.0
            for i in idx:
                u, v, r = users[i], items[i], ratings[i]
                pred = (self.U[u] @ self.V[v] + self.bu[u] + self.bv[v]
                        + self.global_mean)
                err = r - pred
                total_loss += err ** 2
                
                self.U[u]  += self.lr * (err * self.V[v] - self.reg * self.U[u])
                self.V[v]  += self.lr * (err * self.U[u] - self.reg * self.V[v])
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bv[v] += self.lr * (err - self.reg * self.bv[v])
            
            if (epoch + 1) % 10 == 0:
                rmse = (total_loss / len(users)) ** 0.5
                print(f"  BMF epoch {epoch+1}/{self.n_epochs}  RMSE={rmse:.4f}")
    
    def score(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        return (self.U[user_idx] @ self.V[item_indices].T
                + self.bu[user_idx] + self.bv[item_indices]
                + self.global_mean)


# ──────────────────────────────────────────────────────────────────────────────
# WMF (Weighted Matrix Factorization — ALS)
# ──────────────────────────────────────────────────────────────────────────────

class WMF:
    """
    Weighted MF (Hu et al. 2008) with implicit feedback.
    Uses alternating least squares.
    """
    
    def __init__(
        self,
        n_users: int, n_items: int,
        n_factors: int = 64,
        alpha: float = 40.0,
        reg: float = 0.01,
        n_epochs: int = 20,
        seed: int = 42,
    ):
        np.random.seed(seed)
        self.n_factors = n_factors
        self.alpha = alpha
        self.reg = reg
        self.n_epochs = n_epochs
        self.n_users = n_users
        self.n_items = n_items
        
        std = 0.01
        self.U = np.random.randn(n_users, n_factors) * std
        self.V = np.random.randn(n_items, n_factors) * std
    
    def fit(self, train_df):
        R = build_interaction_matrix(
            train_df, self.n_users, self.n_items
        ).toarray().astype(float)
        C = 1 + self.alpha * R  # confidence matrix
        
        for epoch in range(self.n_epochs):
            # Fix V, solve for U
            VtV = self.V.T @ self.V
            for u in range(self.n_users):
                Cu = np.diag(C[u])
                A = self.V.T @ Cu @ self.V + self.reg * np.eye(self.n_factors)
                b = self.V.T @ Cu @ R[u]
                self.U[u] = np.linalg.solve(A, b)
            
            # Fix U, solve for V
            UtU = self.U.T @ self.U
            for v in range(self.n_items):
                Cv = np.diag(C[:, v])
                A = self.U.T @ Cv @ self.U + self.reg * np.eye(self.n_factors)
                b = self.U.T @ Cv @ R[:, v]
                self.V[v] = np.linalg.solve(A, b)
            
            print(f"  WMF epoch {epoch+1}/{self.n_epochs}")
    
    def score(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        return self.U[user_idx] @ self.V[item_indices].T


# ──────────────────────────────────────────────────────────────────────────────
# NeuMF (He et al. 2017)
# ──────────────────────────────────────────────────────────────────────────────

class NeuMFNet(nn.Module):
    def __init__(self, n_users, n_items, mf_dim=32, mlp_layers=(64, 32, 16)):
        super().__init__()
        # MF embeddings
        self.mf_user = nn.Embedding(n_users, mf_dim)
        self.mf_item = nn.Embedding(n_items, mf_dim)
        # MLP embeddings
        mlp_dim = mlp_layers[0] // 2
        self.mlp_user = nn.Embedding(n_users, mlp_dim)
        self.mlp_item = nn.Embedding(n_items, mlp_dim)
        # MLP layers
        layers = []
        in_dim = mlp_layers[0]
        for out_dim in mlp_layers[1:]:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        self.predict = nn.Linear(mf_dim + mlp_layers[-1], 1)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, user_idx, item_idx):
        mf_out = self.mf_user(user_idx) * self.mf_item(item_idx)
        mlp_in = torch.cat([self.mlp_user(user_idx), self.mlp_item(item_idx)], dim=-1)
        mlp_out = self.mlp(mlp_in)
        out = self.predict(torch.cat([mf_out, mlp_out], dim=-1))
        return out.squeeze(-1)


class NeuMF:
    def __init__(
        self,
        n_users, n_items,
        mf_dim=32, mlp_layers=(64, 32, 16),
        lr=1e-3, n_epochs=30, batch_size=1024,
        n_neg=4, seed=42,
        device=DEVICE,
    ):
        torch.manual_seed(seed)
        self.n_users = n_users
        self.n_items = n_items
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_neg = n_neg
        self.device = device
        
        self.net = NeuMFNet(n_users, n_items, mf_dim, mlp_layers).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
    
    def fit(self, train_df):
        R = build_interaction_matrix(self.n_users, self.n_items, train_df)
        users_pos = train_df["user_idx"].astype(int).values
        items_pos = train_df["item_idx"].astype(int).values
        
        for epoch in range(self.n_epochs):
            self.net.train()
            idx = np.random.permutation(len(users_pos))
            total_loss = 0.0
            n_batches = 0
            
            for start in range(0, len(idx), self.batch_size):
                batch = idx[start:start + self.batch_size]
                u_b = users_pos[batch]
                v_pos_b = items_pos[batch]
                
                # Negative sampling
                v_neg_b = np.random.randint(0, self.n_items, (len(batch), self.n_neg))
                
                all_u = np.concatenate([u_b] * (1 + self.n_neg))
                all_v = np.concatenate([v_pos_b] + [v_neg_b[:, j] for j in range(self.n_neg)])
                all_y = np.concatenate([np.ones(len(u_b))] + [np.zeros(len(u_b))] * self.n_neg)
                
                u_t = torch.tensor(all_u, dtype=torch.long, device=self.device)
                v_t = torch.tensor(all_v, dtype=torch.long, device=self.device)
                y_t = torch.tensor(all_y, dtype=torch.float32, device=self.device)
                
                self.optimizer.zero_grad()
                pred = self.net(u_t, v_t)
                loss = F.binary_cross_entropy_with_logits(pred, y_t)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            
            if (epoch + 1) % 5 == 0:
                print(f"  NeuMF epoch {epoch+1}  loss={total_loss/n_batches:.4f}")
    
    @torch.no_grad()
    def score(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        self.net.eval()
        u_t = torch.tensor([user_idx] * len(item_indices), dtype=torch.long, device=self.device)
        v_t = torch.tensor(item_indices, dtype=torch.long, device=self.device)
        return self.net(u_t, v_t).cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# VAE-CF (Liang et al. 2018)
# ──────────────────────────────────────────────────────────────────────────────

class VAECFNet(nn.Module):
    def __init__(self, n_items, hidden_dim=600, latent_dim=200, dropout=0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_items, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.mu_head    = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_items),
        )
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class VAECF:
    def __init__(
        self,
        n_users, n_items,
        hidden_dim=600, latent_dim=200,
        lr=1e-3, n_epochs=100, batch_size=500,
        beta_kl=0.2, dropout=0.5, seed=42,
        device=DEVICE,
    ):
        torch.manual_seed(seed)
        self.n_users = n_users
        self.n_items = n_items
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.beta_kl = beta_kl
        self.device = device
        
        self.net = VAECFNet(n_items, hidden_dim, latent_dim, dropout).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self._user_profiles = None  # stored after fit
    
    def fit(self, train_df):
        X = build_interaction_matrix(
            train_df, self.n_users, self.n_items
        ).toarray().astype(np.float32)
        X_t = torch.tensor(X, device=self.device)
        
        for epoch in range(self.n_epochs):
            self.net.train()
            idx = np.random.permutation(self.n_users)
            total_loss = 0.0
            n_batches = 0
            
            for start in range(0, self.n_users, self.batch_size):
                batch = idx[start:start + self.batch_size]
                x_b = X_t[batch]
                # Normalize
                norm = x_b.sum(1, keepdim=True).clamp(min=1)
                x_norm = x_b / norm
                
                self.optimizer.zero_grad()
                logits, mu, logvar = self.net(x_norm)
                
                # Multinomial likelihood
                log_softmax = F.log_softmax(logits, dim=-1)
                recon_loss = -(x_b * log_softmax).sum(dim=-1).mean()
                
                # KL divergence
                kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
                
                loss = recon_loss + self.beta_kl * kl
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            
            if (epoch + 1) % 20 == 0:
                print(f"  VAE-CF epoch {epoch+1}  loss={total_loss/n_batches:.4f}")
        
        # Pre-compute user scores
        self.net.eval()
        self._X = X
    
    @torch.no_grad()
    def score(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        self.net.eval()
        x = torch.tensor(self._X[user_idx:user_idx+1], device=self.device)
        norm = x.sum(1, keepdim=True).clamp(min=1)
        logits, _, _ = self.net(x / norm)
        scores = F.softmax(logits, dim=-1).cpu().numpy()[0]
        return scores[item_indices]


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def get_model(name: str, n_users: int, n_items: int, seed: int = 42, **kwargs):
    name = name.lower()
    if name == "bmf":
        return BMF(n_users, n_items, seed=seed, **kwargs)
    elif name == "wmf":
        return WMF(n_users, n_items, seed=seed, **kwargs)
    elif name in ("neumf", "ncf"):
        return NeuMF(n_users, n_items, seed=seed, **kwargs)
    elif name in ("vaecf", "vae-cf", "vae_cf"):
        return VAECF(n_users, n_items, seed=seed, **kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")
