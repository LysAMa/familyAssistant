#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ajoute des features temporels (heure / jour de semaine, en codage cyclique)
à features.npy, en utilisant timestamps.npy.

Entrée :
  - features.npy : [T, N, F] (F=1 si tu n'avais que TravelTime_s)
  - timestamps.npy : [T] (strings ISO)

Sortie :
  - features.npy écrasé par une version augmentée [T, N, F+4]
    où les 4 nouvelles features sont :
      - sin_time_of_day, cos_time_of_day
      - sin_day_of_week, cos_day_of_week
"""

import numpy as np
import pandas as pd
import os


def main(data_dir: str):
    features_path = os.path.join(data_dir, "features.npy")
    timestamps_path = os.path.join(data_dir, "timestamps.npy")

    features = np.load(features_path)  # [T, N, F]
    timestamps = np.load(timestamps_path, allow_pickle=True)  # [T]

    print(f"[INFO] features shape (before) : {features.shape}")
    T, N, F = features.shape

    # Convertir les timestamps en datetime pandas
    # On passe par une Series pour être sûr d'avoir l'attribut .dt
    dt = pd.to_datetime(pd.Series(timestamps))

    # Heure de la journée (0-24)
    hours = dt.dt.hour.to_numpy() + dt.dt.minute.to_numpy() / 60.0
    hour_angle = 2.0 * np.pi * hours / 24.0

    # Jour de semaine (0=Monday ... 6=Sunday)
    dow = dt.dt.weekday.to_numpy()
    dow_angle = 2.0 * np.pi * dow / 7.0

    # Features cycliques [T, 4]
    time_sin = np.sin(hour_angle)
    time_cos = np.cos(hour_angle)
    dow_sin = np.sin(dow_angle)
    dow_cos = np.cos(dow_angle)

    time_feats = np.stack([time_sin, time_cos, dow_sin, dow_cos], axis=-1)  # [T, 4]

    # Étendre à tous les noeuds : [T, 1, 4] -> [T, N, 4]
    time_feats = np.repeat(time_feats[:, None, :], N, axis=1)  # [T, N, 4]

    # Concaténer : TravelTime_s (ou autres) + 4 features temporels
    features_aug = np.concatenate([features, time_feats.astype(np.float32)], axis=-1)  # [T, N, F+4]

    print(f"[INFO] features shape (after) : {features_aug.shape}")

    # Écraser features.npy (ou change le nom si tu préfères garder l'original)
    np.save(features_path, features_aug)
    print(f"[OK] features.npy mis à jour dans : {features_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Dossier contenant features.npy et timestamps.npy")
    args = parser.parse_args()
    main(args.data_dir)
