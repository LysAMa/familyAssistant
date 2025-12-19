#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utiliser un modèle entraîné (GCRNN ou LSTM) pour prédire les temps de parcours
sur tous les segments, à partir de features.npy / adjacency.npy / timestamps.npy.

- Charge le modèle sauvegardé {gcrnn,lstm}_model.pt
- Charge features.npy (NON normalisé) et adjacency.npy
- Re-normalise avec mean/std sauvegardés
- Fait une prédiction à partir de la dernière fenêtre temporelle
- Donne les temps de parcours prédits (en secondes) pour chaque segment

Optionnel : montre comment agréger le temps total sur un trajet (liste de segments).
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn

from traffic_model import (
    GCRNN,
    LSTMModel,
    normalize_adjacency,
    load_data,
)

def describe_predictions(y_hat_real: np.ndarray, route_nodes: list[int] | None = None) -> str:
    """
    Génère un court résumé en langage naturel à partir des prédictions du modèle.
    - y_hat_real : [N] temps de parcours prédits (en secondes) pour chaque segment
    - route_nodes : séquence des segments du trajet (node_index)
    """
    route_nodes = [int(n) for n in (route_nodes or [])]
    try:
        y = np.asarray(y_hat_real, dtype=float)
        median_seg = float(np.median(y))
        slow_seg = int(np.argmax(y))
        slow_time = float(np.max(y))
        if route_nodes:
            y = np.asarray(y_hat_real, dtype=np.float32)
            eta_seconds = float(np.sum(y[np.asarray(route_nodes, dtype=int)]))
            eta_minutes = eta_seconds / 60.0
            msg = (
                f"Selon le modèle, le trajet couvre {len(route_nodes)} segments "
                f"pour environ {eta_minutes:.1f} minutes. "
            )
        else:
            msg = "Selon le modèle, "
            eta_minutes = None
        msg += (
            f"Le segment le plus lent est {slow_seg} avec ~{slow_time:.1f} s. "
            f"Le temps médian par segment récent est ~{median_seg:.1f} s."
        )
        if eta_minutes is not None:
            msg += " Prédiction basée sur les conditions récentes apprises."
        return msg
    except Exception:
        return "Résumé indisponible pour les prédictions du modèle."


def load_trained_model(data_dir: str, device: torch.device):
    """
    Charge le modèle entraîné et les méta-infos (mean, std, seq_len, etc.).
    Retourne :
      - model (en mode eval)
      - checkpoint (dict)
    """
    # On cherche d'abord un modèle GCRNN, sinon LSTM
    gcrnn_path = os.path.join(data_dir, "gcrnn_model.pt")
    lstm_path = os.path.join(data_dir, "lstm_model.pt")

    if os.path.exists(gcrnn_path):
        model_path = gcrnn_path
    elif os.path.exists(lstm_path):
        model_path = lstm_path
    else:
        raise FileNotFoundError(
            f"Aucun modèle trouvé dans {data_dir} (gcrnn_model.pt ou lstm_model.pt)."
        )

    print(f"[INFO] Chargement du modèle : {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model_type = checkpoint["model_type"]
    seq_len = checkpoint["seq_len"]
    hidden_dim = checkpoint["hidden_dim"]
    mean = checkpoint["mean"]      # [1,1,F]
    std = checkpoint["std"]        # [1,1,F]

    # Charger les données brutes
    features, adjacency, timestamps = load_data(data_dir)
    # Sécuriser les types numériques
    features = np.asarray(features, dtype=np.float32)
    adjacency = np.asarray(adjacency, dtype=np.float32)
    T, N, F_in = features.shape

    # Normaliser l'adjacence (pour GCRNN)
    A_hat = normalize_adjacency(adjacency)

    # Construire le modèle
    if model_type == "gcrnn":
        model = GCRNN(
            num_nodes=N,
            in_dim=F_in,
            hidden_dim=hidden_dim,
            out_dim=1,
            A_hat=A_hat,
        )
    elif model_type == "lstm":
        model = LSTMModel(
            num_nodes=N,
            in_dim=F_in,
            hidden_dim=hidden_dim,
            out_dim=1,
        )
    else:
        raise ValueError(f"model_type inconnu dans le checkpoint : {model_type}")

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    print(f"[INFO] Modèle chargé ({model_type}), seq_len={seq_len}, hidden_dim={hidden_dim}")
    return model, checkpoint, features, adjacency, timestamps


def predict_next_step(
    model: nn.Module,
    checkpoint: dict,
    features: np.ndarray,
    device: torch.device,
    t_end: int | None = None,
):
    """
    Fait une prédiction d'un pas de temps à partir d'une fenêtre de longueur seq_len.

    - features : données NON normalisées [T, N, F]
    - t_end : index temporel de fin (exclusif) pour la fenêtre.
              Si None -> on prend la dernière fenêtre possible (prévision une step après la fin du dataset).

    Retourne :
      - y_hat_real : [N] temps de parcours prédits (en secondes) pour chaque segment.
      - t_input_start, t_input_end : indices utilisés pour la fenêtre.
    """
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)      # [1,1,F]
    std = np.asarray(checkpoint["std"], dtype=np.float32)        # [1,1,F]
    seq_len = checkpoint["seq_len"]

    T, N, F_in = features.shape

    # Normaliser tous les features avec mean/std appris
    features_norm = (features.astype(np.float32) - mean) / std

    # Déterminer la fenêtre temporelle utilisée
    if t_end is None:
        # on prend la dernière fenêtre possible : [T-seq_len, T)
        t_end = T
    if t_end <= seq_len:
        raise ValueError(
            f"t_end={t_end} <= seq_len={seq_len} : pas assez d'historique."
        )

    t_start = t_end - seq_len   # fenêtre [t_start, t_end)

    X_window = features_norm[t_start:t_end]         # [seq_len, N, F_in]
    X_window = np.expand_dims(X_window, axis=0)     # [1, seq_len, N, F_in]

    X_tensor = torch.from_numpy(X_window).to(device)  # [1, T_in, N, F_in]

    with torch.no_grad():
        y_hat_norm = model(X_tensor)              # [1, N, 1]

    # y_hat_norm est dans l'espace normalisé -> on dénormalise sur le canal TravelTime (indice 0)
    # std et mean : [1,1,F], on prend [:,:,0:1] pour canal 0
    mean_tt = mean[:, :, 0:1]   # [1,1,1]
    std_tt = std[:, :, 0:1]     # [1,1,1]

    y_hat_real = y_hat_norm.cpu().numpy() * std_tt + mean_tt  # [1, N, 1]
    y_hat_real = y_hat_real[0, :, 0]   # [N]

    return y_hat_real, t_start, t_end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", required=True,
        help="Dossier contenant features.npy, adjacency.npy, timestamps.npy et le modèle sauvegardé.",
    )
    parser.add_argument(
        "--t_end", type=int, default=None,
        help="Index temporel de fin pour la fenêtre (optionnel). "
             "Par défaut, utilise la dernière fenêtre pour prédire un pas au-delà du dataset.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[INFO] Device : {device}")

    # 1) Charger le modèle + données
    model, checkpoint, features, adjacency, timestamps = load_trained_model(
        args.data_dir, device
    )

    # 2) Prédire le prochain pas de temps
    y_hat_real, t_start, t_end = predict_next_step(
        model, checkpoint, features, device, t_end=args.t_end
    )

    print(f"[INFO] Fenêtre temporelle utilisée : [{t_start}, {t_end}) "
          f"-> prédiction pour l'instant t={t_end}")

    if args.t_end is not None and args.t_end < len(timestamps):
        print(f"[INFO] Timestamp approx. prédite : après {timestamps[args.t_end-1]}")
    else:
        print("[INFO] Prédiction pour un pas de temps juste après la fin des données disponibles.")

    print(f"[INFO] Nombre de segments : {len(y_hat_real)}")
    print("[INFO] Exemple de temps de parcours prédits (en secondes) pour les 10 premiers segments :")
    print(y_hat_real[:10])

    # 3) Exemple d'ETA pour un trajet (facultatif) :
    # Imaginons un chemin composé des segments [0, 3, 5, 10]
    y_hat_real = np.asarray(y_hat_real, dtype=float)
    route_nodes = [0, 3, 5, 10]
    eta_seconds = float(y_hat_real[route_nodes].sum())
    eta_minutes = eta_seconds / 60.0


    print(f"[DEMO] ETA pour le chemin {route_nodes} ≈ {eta_seconds:.1f} s "
          f"({eta_minutes:.1f} minutes)")

if __name__ == "__main__":
    main()
