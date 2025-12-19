#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modèle de prédiction de trafic pour Montréal à partir de :

    - features.npy   : [T, N, F]
        F >= 1, avec le canal 0 = TravelTime_s (cible)
        Les autres canaux peuvent être des features temporelles (heure/jour, etc.).
    - adjacency.npy  : [N, N]
    - timestamps.npy : [T] (non utilisé dans le modèle mais chargé pour info)

Le modèle :
    - prend en entrée une séquence de longueur seq_len
    - prédit le prochain pas de temps (horizon = 1) pour tous les noeuds,
      pour la seule cible TravelTime_s (canal 0).

Deux architectures possibles :
    - GCRNN : RNN avec convolution de graphe à chaque pas de temps
    - LSTM  : baseline sans graphe (on ignore adjacency)

Usage exemple :

    python traffic_model.py \
        --data_dir my_msltd_montreal \
        --model_type gcrnn \
        --seq_len 12 \
        --epochs 20 \
        --batch_size 32
"""

import os
import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ============================
#   DATA UTILITIES
# ============================

class TrafficDataset(Dataset):
    """
    Dataset temporel glissant :
      - X : séquences de longueur seq_len, toutes les features
      - y : prochaine frame, uniquement la cible TravelTime_s (canal 0)

    features : [T, N, F]
    """

    def __init__(self, features: np.ndarray, seq_len: int):
        super().__init__()
        assert features.ndim == 3, "features doit être de forme [T, N, F]"
        self.features = features.astype(np.float32)
        self.seq_len = seq_len
        self.T = self.features.shape[0]
        self.num_samples = self.T - self.seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # X : [seq_len, N, F] – toutes les features (entrée)
        X = self.features[idx : idx + self.seq_len]          # [T_in, N, F]
        # y : [N, 1] – TravelTime_s (canal 0) à l'instant suivant
        y_full = self.features[idx + self.seq_len]          # [N, F]
        y = y_full[:, 0:1]                                  # [N, 1]
        return X, y


def load_data(data_dir: str):
    """
    Charge features.npy, adjacency.npy, timestamps.npy.
    """
    features_path = os.path.join(data_dir, "features.npy")
    adjacency_path = os.path.join(data_dir, "adjacency.npy")
    timestamps_path = os.path.join(data_dir, "timestamps.npy")

    features = np.load(features_path)                       # [T, N, F]
    adjacency = np.load(adjacency_path)                     # [N, N]
    timestamps = np.load(timestamps_path, allow_pickle=True)  # [T]

    print(f"[INFO] features shape : {features.shape} (T, N, F)")
    print(f"[INFO] adjacency shape : {adjacency.shape} (N, N)")
    print(f"[INFO] timestamps count : {len(timestamps)}")

    return features, adjacency, timestamps


def normalize_features(features: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalisation standard globale (par feature, sur tous les temps et noeuds).

    Retourne :
      - features normalisées
      - moyenne (pour éventuelle dénormalisation)
      - écart-type
    """
    # features : [T, N, F]
    mean = features.mean(axis=(0, 1), keepdims=True)        # [1, 1, F]
    std = features.std(axis=(0, 1), keepdims=True) + eps    # [1, 1, F]
    normed = (features - mean) / std
    return normed.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def prepare_dataloaders(
    features: np.ndarray,
    seq_len: int,
    batch_size: int,
    test_size: float = 0.2,
    val_size: float = 0.1,
):
    """
    Crée DataLoader train/val/test à partir des features [T, N, F]
    via un split temporel (on ne shuffle pas les indices globaux).
    """
    dataset = TrafficDataset(features, seq_len)
    num_samples = len(dataset)
    indices = np.arange(num_samples)

    # Split train / (val+test)
    train_idx, temp_idx = train_test_split(
        indices, test_size=test_size + val_size, shuffle=False
    )

    # Split (val+test) en val / test
    val_rel = val_size / (test_size + val_size)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=1 - val_rel, shuffle=False
    )

    def make_loader(idxs, shuffle: bool):
        subset = torch.utils.data.Subset(dataset, idxs)
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader = make_loader(val_idx, shuffle=False)
    test_loader = make_loader(test_idx, shuffle=False)

    print(
        f"[INFO] #train samples: {len(train_idx)}, "
        f"#val samples: {len(val_idx)}, #test samples: {len(test_idx)}"
    )

    return train_loader, val_loader, test_loader


# ============================
#   GRAPH UTILITIES
# ============================

def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """
    Normalisation de la matrice d'adjacence :

        Â = D^{-1/2} (A + I) D^{-1/2}

    A : [N, N]
    """
    N = A.shape[0]
    A_tilde = A + np.eye(N, dtype=A.dtype)
    D = A_tilde.sum(axis=1)
    D_inv_sqrt = np.power(D, -0.5, where=D > 0)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
    D_mat = np.diag(D_inv_sqrt)
    A_norm = D_mat @ A_tilde @ D_mat
    return A_norm.astype(np.float32)


# ============================
#   MODELES
# ============================

class GraphConv(nn.Module):
    """
    Convolution de graphe simple : H = Â X W

      Â : [N, N] (fixe)
      X : [B, N, C_in]
      W : [C_in, C_out]
    """

    def __init__(self, in_dim: int, out_dim: int, A_hat: torch.Tensor):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.register_buffer("A_hat", A_hat)  # matrice fixe
        self.W = nn.Parameter(torch.randn(in_dim, out_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [B, N, C_in]
        Retour : [B, N, C_out]
        """
        AX = torch.einsum("ij,bjk->bik", self.A_hat, X)       # [B, N, C_in]
        out = torch.einsum("bik,kh->bih", AX, self.W) + self.bias
        return out


class GCRNNCell(nn.Module):
    """
    Cellule GRU avec convolution de graphe dans chaque gate.
    """

    def __init__(self, num_nodes: int, in_dim: int, hidden_dim: int, A_hat: torch.Tensor):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        gc_in = in_dim + hidden_dim  # concat(X, H)

        self.gc_z = GraphConv(gc_in, hidden_dim, A_hat)
        self.gc_r = GraphConv(gc_in, hidden_dim, A_hat)
        self.gc_h = GraphConv(in_dim + hidden_dim, hidden_dim, A_hat)

    def forward(self, X_t: torch.Tensor, H_prev: torch.Tensor) -> torch.Tensor:
        """
        X_t:   [B, N, in_dim]
        H_prev: [B, N, hidden_dim]
        """
        XH = torch.cat([X_t, H_prev], dim=-1)   # [B, N, in+hidden]

        z = torch.sigmoid(self.gc_z(XH))
        r = torch.sigmoid(self.gc_r(XH))

        X_rH = torch.cat([X_t, r * H_prev], dim=-1)
        H_tilde = torch.tanh(self.gc_h(X_rH))

        H = (1.0 - z) * H_prev + z * H_tilde
        return H


class GCRNN(nn.Module):
    """
    GCRNN complet :
      - lit une séquence X [B, T_in, N, F_in]
      - propage via la cellule GCRNN
      - prédit la cible suivante y_hat [B, N, F_out] (ici F_out = 1)
    """

    def __init__(self, num_nodes: int, in_dim: int, hidden_dim: int,
                 out_dim: int, A_hat: np.ndarray):
        super().__init__()
        A_hat_tensor = torch.tensor(A_hat, dtype=torch.float32)
        self.cell = GCRNNCell(num_nodes, in_dim, hidden_dim, A_hat_tensor)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [B, T_in, N, F_in]
        """
        B, T_in, N, F_in = X.shape
        device = X.device
        H = torch.zeros(B, N, self.cell.hidden_dim, device=device)

        for t in range(T_in):
            X_t = X[:, t]                  # [B, N, F_in]
            H = self.cell(X_t, H)          # [B, N, hidden_dim]

        y_hat = self.out_proj(H)           # [B, N, out_dim]
        return y_hat


class LSTMModel(nn.Module):
    """
    Baseline LSTM sans graphe :
      - on "aplatit" nodes + features dans le temps.
      - entrée : X [B, T_in, N, F_in] -> [B, T_in, N*F_in]
      - sortie : y_hat [B, N, 1]
    """

    def __init__(self, num_nodes: int, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.lstm = nn.LSTM(
            input_size=num_nodes * in_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, num_nodes * out_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X : [B, T_in, N, F_in]
        """
        B, T_in, N, F_in = X.shape
        X_flat = X.reshape(B, T_in, N * F_in)                # [B, T_in, N*F_in]
        out_seq, (h_T, _) = self.lstm(X_flat)                # h_T : [1, B, hidden_dim]
        h_last = h_T[-1]                                     # [B, hidden_dim]
        y_flat = self.fc(h_last)                             # [B, N*out_dim]
        y_hat = y_flat.view(B, N, self.out_dim)              # [B, N, out_dim]
        return y_hat


# ============================
#   TRAIN / EVAL
# ============================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X = X.to(device)          # [B, T_in, N, F]
        y = y.to(device)          # [B, N, 1]

        optimizer.zero_grad()
        y_hat = model(X)          # [B, N, 1]
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            y_hat = model(X)
            loss = criterion(y_hat, y)
            total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


# ============================
#   MAIN
# ============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", required=True,
        help="Dossier contenant features.npy, adjacency.npy, timestamps.npy",
    )
    parser.add_argument(
        "--model_type", type=str, default="gcrnn",
        choices=["gcrnn", "lstm"],
        help="Type de modèle à utiliser (gcrnn ou lstm).",
    )
    parser.add_argument("--seq_len", type=int, default=12,
                        help="Longueur de la séquence d'entrée.")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Dimension cachée (GCRNN ou LSTM).")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    print(f"[INFO] Device : {args.device}")
    print(f"[INFO] Model type : {args.model_type}")

    # 1) Chargement des données
    features, adjacency, timestamps = load_data(args.data_dir)
    T, N, F_in = features.shape

    # 2) Normalisation des features
    features_norm, mean, std = normalize_features(features)

    # 3) Normalisation de la matrice d'adjacence (pour GCRNN)
    A_hat = normalize_adjacency(adjacency)

    # 4) DataLoaders
    train_loader, val_loader, test_loader = prepare_dataloaders(
        features_norm,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )

    # 5) Modèle
    if args.model_type == "gcrnn":
        model = GCRNN(
            num_nodes=N,
            in_dim=F_in,
            hidden_dim=args.hidden_dim,
            out_dim=1,           # on prédit seulement TravelTime_s
            A_hat=A_hat,
        )
    else:  # lstm
        model = LSTMModel(
            num_nodes=N,
            in_dim=F_in,
            hidden_dim=args.hidden_dim,
            out_dim=1,
        )

    device = torch.device(args.device)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 6) Entraînement
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    # 7) Évaluation finale sur le test
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"[RESULT] Test MSE (sur données normalisées) : {test_loss:.6f}")

    # 8) Sauvegarde du modèle
    model_path = os.path.join(args.data_dir, f"{args.model_type}_model.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "seq_len": args.seq_len,
            "hidden_dim": args.hidden_dim,
            "model_type": args.model_type,
        },
        model_path,
    )
    print(f"[OK] Modèle sauvegardé dans : {model_path}")


if __name__ == "__main__":
    main()
