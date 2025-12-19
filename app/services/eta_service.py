#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
eta_ml_service.py

Service pour calculer un ETA à partir :
  - d'un chemin fourni sous forme de polyline (liste de [lon, lat])
  - du modèle entraîné (GCRNN/LSTM) et du dataset Montréal (features.npy, adjacency.npy, segment_info.csv).

Idée :
  1) ORS donne une geometrie (LineString) = liste de points [lon, lat]
  2) On projette chaque point sur le segment Bluetooth le plus proche
     (via les milieux des segments dans segment_info.csv)
  3) On obtient une séquence de node_index représentant approximativement le trajet
  4) On demande au modèle de prédire TravelTime_s pour tous les segments,
     puis on somme sur les segments de la route.
"""

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from use_model import load_trained_model, predict_next_step


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Distance approx (en mètres) entre deux points lat/lon (WGS84).
    """
    R = 6371000.0  # rayon Terre en m

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return R * c


class ETAModel:
    def __init__(self, data_dir: str, device: str | None = None):
        """
        :param data_dir: dossier contenant :
           - features.npy
           - adjacency.npy
           - timestamps.npy
           - segment_info.csv
           - gcrnn_model.pt / lstm_model.pt
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Charger modèle + données brutes
        (
            self.model,
            self.checkpoint,
            self.features,
            self.adjacency,
            self.timestamps,
        ) = load_trained_model(data_dir, self.device)

        # Charger les segments
        seg_path = os.path.join(data_dir, "segment_info.csv")
        self.segments_df = pd.read_csv(seg_path)

        if "node_index" not in self.segments_df.columns:
            raise ValueError("segment_info.csv doit contenir une colonne 'node_index'.")

        # On prépare aussi les points représentatifs (milieu du segment) pour map-matching
        required = ["SrcLatitude", "SrcLongitude", "DestLatitude", "DestLongitude"]
        for c in required:
            if c not in self.segments_df.columns:
                raise ValueError(f"Colonne '{c}' manquante dans segment_info.csv : {c}")

        df = self.segments_df.copy()
        df["node_index"] = df["node_index"].astype(int)

        lat_mid = (df["SrcLatitude"] + df["DestLatitude"]) / 2.0
        lon_mid = (df["SrcLongitude"] + df["DestLongitude"]) / 2.0

        df["lat_mid"] = lat_mid
        df["lon_mid"] = lon_mid

        # On ordonne par node_index pour aligner avec adjacency/features
        df = df.sort_values("node_index")

        self.rep_points = df[["lat_mid", "lon_mid"]].to_numpy()   # [N, 2]
        self.node_indices = df["node_index"].to_numpy()           # [N]

        print(
            f"[INFO] ETAModel initialisé sur {device} avec "
            f"{len(self.node_indices)} segments Bluetooth."
        )

    # ------------- helpers -------------

    def _nearest_node_for_point(self, lat: float, lon: float) -> int:
        """
        Trouve le node_index du segment dont le milieu est le plus proche du point GPS.
        """
        lat_arr = self.rep_points[:, 0]
        lon_arr = self.rep_points[:, 1]
        dists = haversine_distance(lat, lon, lat_arr, lon_arr)
        idx_min = int(np.argmin(dists))
        return int(self.node_indices[idx_min])

    def route_from_polyline(self, coords: List[List[float]]) -> List[int]:
        """
        Convertit une polyline ORS (liste de [lon, lat]) en séquence de node_index.
        On prend, pour chaque point, le segment Bluetooth le plus proche.
        On simplifie ensuite en supprimant les répétitions consécutives.

        :param coords: liste de [lon, lat]
        :return: liste de node_index représentant la route
        """
        if not coords:
            raise ValueError("Polyline vide reçue pour la route.")

        route_nodes: List[int] = []
        last_node = None

        for lon, lat in coords:
            node = self._nearest_node_for_point(lat, lon)
            if node != last_node:
                route_nodes.append(node)
                last_node = node

        return route_nodes

    # ------------- ETA -------------

    def eta_for_nodes(self, route_nodes: List[int], departure_index: int | None = None) -> Tuple[float, float, np.ndarray]:
        """
        ETA pour un chemin déjà exprimé en node_index.
        """
        y_hat_real, t_start, t_end = predict_next_step(
            self.model,
            self.checkpoint,
            self.features,
            self.device,
            t_end=departure_index,
        )
        # y_hat_real : [N] TravelTime_s prédit pour chaque segment
        y_hat_real = np.asarray(y_hat_real, dtype=float)
        route_nodes_int = [int(n) for n in route_nodes]
        seg_times = y_hat_real[route_nodes_int]   # [len(route_nodes)]

        eta_seconds = float(seg_times.sum())
        eta_minutes = eta_seconds / 60.0
        return eta_seconds, eta_minutes, seg_times

    def eta_for_polyline(self, coords: List[List[float]], departure_index: int | None = None):
        """
        ETA pour une polyline ORS (liste [lon, lat]).

        :return: (eta_seconds, eta_minutes, route_nodes, per_segment_times)
        """
        route_nodes = self.route_from_polyline(coords)
        eta_seconds, eta_minutes, seg_times = self.eta_for_nodes(
            route_nodes, departure_index=departure_index
        )
        return eta_seconds, eta_minutes, route_nodes, seg_times
