#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Construction d'un dataset de trafic urbain de type MSLTD
à partir des données ouvertes de la Ville de Montréal :

1) Temps de parcours sur des segments routiers (historique)
   https://donnees.montreal.ca/dataset/temps-de-parcours-sur-des-segments-routiers-historique

2) Segments routiers de collecte des temps de parcours
   https://donnees.montreal.ca/dataset/segments-routiers-de-collecte-des-temps-de-parcours

Sorties (dans --out_dir) :
    - features.npy   : tableau [T, N, 1] de TravelTime_s agrégé toutes les 15 minutes
    - adjacency.npy  : matrice [N, N] (segments adjacents via capteurs Bluetooth)
    - timestamps.npy : tableau [T] de timestamps ISO (str)
    - segment_info.csv : métadonnées sur les segments + index de nœud

Licence : respecter la licence d’utilisation des données ouvertes de Montréal.
"""

import argparse
import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


# ------------------------------
# Chargement des données sources
# ------------------------------

def load_segments(segments_csv_path: str) -> pd.DataFrame:
    """
    Charge le fichier CSV :
    'Segments de temps de parcours au format CSV' (IdLink, LinkID, SrcDetectorId, etc.)
    https://donnees.montreal.ca/dataset/segments-routiers-de-collecte-des-temps-de-parcours
    """
    df = pd.read_csv(segments_csv_path)

    # Normalisation légère des noms de colonnes attendus
    # D'après le dictionnaire de données officiel :
    #   IdLink, LinkID, SrcDetectorId, DestDetectorId, SrcLatitude, SrcLongitude,
    #   DestLatitude, DestLongitude, LinkName, RouteDirectionName,
    #   LineDistance_m, etc.
    expected_cols = [
        "IdLink", "LinkID", "SrcDetectorId", "DestDetectorId",
        "SrcLatitude", "SrcLongitude", "DestLatitude", "DestLongitude",
        "LinkName", "RouteDirectionName", "LineDistance_m"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print(f"[AVERTISSEMENT] Colonnes manquantes dans segments CSV: {missing}")
        print("           Le script peut encore fonctionner, mais les métadonnées seront incomplètes.")

    # Normaliser LinkID en texte cohérent
    if "LinkID" in df.columns:
        df["LinkID"] = df["LinkID"].astype(str).str.strip()
        # pour simplifier les jointures ensuite
        df["LinkId"] = df["LinkID"]  # alias cohérent avec le dataset historique
    elif "LinkId" in df.columns:
        df["LinkId"] = df["LinkId"].astype(str).str.strip()
    else:
        raise ValueError("Impossible de trouver une colonne 'LinkID' ou 'LinkId' dans le fichier segments.")

    return df

def load_travel_times(travel_csv_paths: List[str]) -> pd.DataFrame:
    """
    Charge et concatène les CSV 'Temps de parcours AAAA' provenant de :
    https://donnees.montreal.ca/dataset/temps-de-parcours-sur-des-segments-routiers-historique

    Dictionnaire de données officiel :
      * LinkId (texte)
      * SrcDetectorId (texte)
      * DestDetectorId (texte)
      * PathDistance_m (entier)
      * TripStart_dt (datetime)
      * TripEnd_dt (datetime)
      * Speed_kmh (numérique)
      * TravelTime_s (entier)
    """
    frames = []
    for path in travel_csv_paths:
        print(f"[INFO] Chargement du fichier de temps de parcours : {path}")
        # On force certains champs en texte pour éviter les types mixtes
        df = pd.read_csv(
            path,
            low_memory=False,  # évite les warnings de type
            dtype={
                "LinkId": "string",
                "SrcDetectorId": "string",
                "DestDetectorId": "string",
            },
        )
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    # Vérification des colonnes attendues
    required_cols = [
        "LinkId",
        "TripStart_dt",
        "TripEnd_dt",
        "Speed_kmh",
        "TravelTime_s",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans les fichiers de temps de parcours : {missing}")

    # Nettoyage : normaliser LinkId en string propre
    df["LinkId"] = df["LinkId"].astype(str).str.strip()

    # Parsing des dates
    df["TripStart_dt"] = pd.to_datetime(df["TripStart_dt"], errors="coerce")
    df["TripEnd_dt"] = pd.to_datetime(df["TripEnd_dt"], errors="coerce")

    # Conversion robuste en numérique (certaines lignes peuvent contenir du texte, vide, etc.)
    df["Speed_kmh"] = pd.to_numeric(df["Speed_kmh"], errors="coerce")
    df["TravelTime_s"] = pd.to_numeric(df["TravelTime_s"], errors="coerce")

    # On enlève les lignes sans info temporelle ou sans mesure valide
    df = df.dropna(subset=["TripStart_dt", "TripEnd_dt", "Speed_kmh", "TravelTime_s"])

    # Filtrage basique : on garde uniquement des vitesses et temps > 0
    df = df[df["Speed_kmh"] > 0]
    df = df[df["TravelTime_s"] > 0]

    return df

# ------------------------------
# Construction des nœuds & graph
# ------------------------------

def build_node_index(
    segments_df: pd.DataFrame,
    travel_df: pd.DataFrame,
    min_obs_per_segment: int = 50,
) -> Tuple[List[str], Dict[str, int], pd.DataFrame, pd.DataFrame]:
    """
    Détermine la liste finale des LinkId (segments) à garder comme nœuds.

    On garde seulement les LinkId présents :
      - dans segments_df
      - ET dans travel_df
      - ET avec au moins min_obs_per_segment observations de TravelTime_s
    """

    # Intersection des segments présents dans les deux jeux
    seg_linkids = set(segments_df["LinkId"].astype(str))
    trav_linkids = set(travel_df["LinkId"].astype(str))
    common = seg_linkids.intersection(trav_linkids)

    print(f"[INFO] Nombre de LinkId dans segments : {len(seg_linkids)}")
    print(f"[INFO] Nombre de LinkId dans temps de parcours : {len(trav_linkids)}")
    print(f"[INFO] Intersection initiale : {len(common)} segments")

    # Filtrer travel_df à ces segments communs
    travel_df = travel_df[travel_df["LinkId"].isin(common)].copy()

    # Appliquer un filtre de fréquence minimale (évite les segments quasi vides)
    counts = travel_df.groupby("LinkId").size()
    valid_ids = counts[counts >= min_obs_per_segment].index.tolist()
    print(f"[INFO] Segments avec au moins {min_obs_per_segment} observations : {len(valid_ids)}")

    # Trie pour stabilité
    valid_ids = sorted(valid_ids)

    # Construction du mapping LinkId -> index de nœud
    node_index = {lid: i for i, lid in enumerate(valid_ids)}

    # Filtrer segments_df à ces LinkId
    segments_df = segments_df[segments_df["LinkId"].isin(valid_ids)].copy()

    return valid_ids, node_index, segments_df, travel_df


def build_adjacency(
    segments_df: pd.DataFrame,
    node_index: Dict[str, int],
    undirected: bool = True,
) -> np.ndarray:
    """
    Construit la matrice d'adjacence [N, N] des segments.

    Logique de base :
      - Un segment A -> B si
          DestDetectorId(A) == SrcDetectorId(B)
    On peut ensuite symétriser si undirected=True.
    """

    if not {"LinkId", "SrcDetectorId", "DestDetectorId"}.issubset(segments_df.columns):
        raise ValueError("Les colonnes 'LinkId', 'SrcDetectorId', 'DestDetectorId' sont requises dans segments_df.")

    n = len(node_index)
    adj = np.zeros((n, n), dtype=np.float32)

    # On indexe segments_df par LinkId pour accès rapide
    seg = segments_df[["LinkId", "SrcDetectorId", "DestDetectorId"]].copy()
    seg["SrcDetectorId"] = seg["SrcDetectorId"].astype(str)
    seg["DestDetectorId"] = seg["DestDetectorId"].astype(str)

    # Préparer un mapping DestDetectorId -> liste de LinkId
    dest_to_links = {}
    for _, row in seg.iterrows():
        dest = row["DestDetectorId"]
        lid = row["LinkId"]
        dest_to_links.setdefault(dest, []).append(lid)

    # Pour chaque segment A, on cherche les B dont SrcDetectorId == DestDetectorId(A)
    for _, row in seg.iterrows():
        lid_a = row["LinkId"]
        idx_a = node_index[lid_a]
        dest_a = row["DestDetectorId"]
        # voisins B
        neighbors = dest_to_links.get(dest_a, [])
        for lid_b in neighbors:
            idx_b = node_index.get(lid_b)
            if idx_b is None:
                continue
            adj[idx_a, idx_b] = 1.0
            if undirected:
                adj[idx_b, idx_a] = 1.0

    print(f"[INFO] Nombre total d'arêtes (non pondérées) : {int(adj.sum())}")
    return adj


# ------------------------------
# Construction des features
# ------------------------------

def build_features(
    travel_df: pd.DataFrame,
    node_ids: List[str],
    freq: str = "15min",
    impute: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construit la matrice de features [T, N, 1] et le vecteur de timestamps [T].

    - On utilise TravelTime_s comme feature principale (comme MSLTD utilise temps/vitesse).
    - On agrège par moyenne sur des intervalles réguliers (freq = 15 min par défaut)
      en fonction de TripStart_dt.

    Retourne:
      features : np.array [T, N, 1]
      timestamps : np.array [T] (dtype=object, strings ISO)
    """

    required_cols = {"LinkId", "TripStart_dt", "TravelTime_s"}
    if not required_cols.issubset(travel_df.columns):
        raise ValueError(f"Colonnes nécessaires manquantes dans travel_df : {required_cols - set(travel_df.columns)}")

    # Filtrer sur les LinkId retenus
    travel_df = travel_df[travel_df["LinkId"].isin(node_ids)].copy()

    # Binning temporel
    travel_df["dt_bin"] = travel_df["TripStart_dt"].dt.floor(freq)

    # Agrégation par moyenne de TravelTime_s
    grouped = (
        travel_df
        .groupby(["dt_bin", "LinkId"])["TravelTime_s"]
        .mean()
        .reset_index()
    )

    # Pivot pour obtenir [T, N]
    pivot = grouped.pivot(index="dt_bin", columns="LinkId", values="TravelTime_s")

    # Ordonner les colonnes selon node_ids (et garantir toutes les colonnes)
    pivot = pivot.reindex(columns=node_ids)

    # Trier les timestamps
    pivot = pivot.sort_index()

    # Gestion des valeurs manquantes
    if impute:
        # Imputation simple : médiane par colonne, puis 0 pour les colonnes entièrement NaN
        col_medians = pivot.median(axis=0)
        pivot = pivot.fillna(col_medians)
        pivot = pivot.fillna(0.0)  # si une colonne est entièrement NaN
    else:
        print("[INFO] Imputation désactivée : des NaN peuvent persister dans features.npy.")

    # Conversion en numpy
    values = pivot.to_numpy(dtype=np.float32)  # [T, N]
    # Ajout de la dimension "feature" = 1
    features = values[:, :, None]  # [T, N, 1]

    # Timestamps au format string ISO
    timestamps = pivot.index.strftime("%Y-%m-%d %H:%M:%S").to_numpy()

    print(f"[INFO] Features shape : {features.shape} (T, N, 1)")
    print(f"[INFO] Nombre de timestamps : {len(timestamps)}")
    return features, timestamps


# ------------------------------
# Fonction principale
# ------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Construire un dataset de trafic urbain de type MSLTD pour Montréal "
                    "à partir des données Bluetooth (temps de parcours + segments)."
    )
    parser.add_argument(
        "--segments_csv",
        required=True,
        help="Chemin vers le CSV 'Segments de temps de parcours' (segments routiers de collecte).",
    )
    parser.add_argument(
        "--travel_csvs",
        nargs="+",
        required=True,
        help="Liste de chemins vers les CSV 'Temps de parcours AAAA' (historique). "
             "On peut mettre plusieurs années.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Dossier de sortie pour features.npy, adjacency.npy, timestamps.npy, segment_info.csv",
    )
    parser.add_argument(
        "--freq",
        default="15min",
        help="Fréquence temporelle pour l'agrégation (par défaut : 15min, comme MSLTD).",
    )
    parser.add_argument(
        "--min_obs_per_segment",
        type=int,
        default=50,
        help="Nombre minimum d'observations TravelTime_s pour qu'un segment soit conservé.",
    )
    parser.add_argument(
        "--no_impute",
        action="store_true",
        help="Si présent, n'impute pas les valeurs manquantes (laisse des NaN dans features).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=== Étape 1 : Chargement des segments ===")
    segments_df = load_segments(args.segments_csv)

    print("=== Étape 2 : Chargement des temps de parcours ===")
    travel_df = load_travel_times(args.travel_csvs)

    print("=== Étape 3 : Construction des nœuds (LinkId) ===")
    node_ids, node_index, segments_df_filt, travel_df_filt = build_node_index(
        segments_df, travel_df, min_obs_per_segment=args.min_obs_per_segment
    )
    print(f"[INFO] Nombre final de segments (nœuds) : {len(node_ids)}")

    print("=== Étape 4 : Construction de la matrice d'adjacence ===")
    adjacency = build_adjacency(segments_df_filt, node_index, undirected=True)

    print("=== Étape 5 : Construction des features temporels ===")
    features, timestamps = build_features(
        travel_df_filt,
        node_ids=node_ids,
        freq=args.freq,
        impute=not args.no_impute,
    )

    print("=== Étape 6 : Sauvegarde des fichiers ===")
    # features.npy
    features_path = os.path.join(args.out_dir, "features.npy")
    np.save(features_path, features)
    print(f"[OK] features.npy sauvegardé : {features_path}")

    # adjacency.npy
    adjacency_path = os.path.join(args.out_dir, "adjacency.npy")
    np.save(adjacency_path, adjacency.astype(np.float32))
    print(f"[OK] adjacency.npy sauvegardé : {adjacency_path}")

    # timestamps.npy
    timestamps_path = os.path.join(args.out_dir, "timestamps.npy")
    np.save(timestamps_path, timestamps)
    print(f"[OK] timestamps.npy sauvegardé : {timestamps_path}")

    # segment_info.csv
    # ajouter l'index de nœud pour faciliter l'interprétation
    segments_df_filt = segments_df_filt.copy()
    segments_df_filt["node_index"] = segments_df_filt["LinkId"].map(node_index)
    info_path = os.path.join(args.out_dir, "segment_info.csv")
    segments_df_filt.to_csv(info_path, index=False)
    print(f"[OK] segment_info.csv sauvegardé : {info_path}")

    print("=== Terminé ===")
    print(f"Dataset de type MSLTD prêt dans : {args.out_dir}")


if __name__ == "__main__":
    main()
