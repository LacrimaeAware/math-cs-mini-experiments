from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

EPS = 1e-12


def _rolling_windows(x: np.ndarray, window: int):
    n = len(x)
    X = np.full((n, window), np.nan)
    for i in range(window - 1, n):
        X[i, :] = x[i - window + 1:i + 1]
    valid = np.isfinite(X).all(axis=1)
    return X, valid


def _zrow(X: np.ndarray):
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True)
    sd = np.where((~np.isfinite(sd)) | (sd < EPS), 1.0, sd)
    return (X - mu) / sd


def fit_transform_learned_events(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, config):
    """Fit unsupervised event clusters on train only and transform all splits.

    These are not assumed useful; they are candidate event ingredients and must pass robustness tests.
    """
    if not getattr(config, "include_learned_events", True):
        return train, val, test

    train = train.copy(); val = val.copy(); test = test.copy()
    combined = pd.concat([train.assign(__split="train"), val.assign(__split="val"), test.assign(__split="test")], axis=0)
    ret = combined["log_ret_1"].to_numpy(dtype=float)
    new_frames = []

    for window in getattr(config, "learned_event_windows", (30,)):
        Xr, valid = _rolling_windows(ret, int(window))
        Xp = np.cumsum(np.where(np.isfinite(Xr), Xr, 0.0), axis=1)
        X = np.concatenate([_zrow(Xr), _zrow(Xp)], axis=1)
        train_mask = (combined["__split"].to_numpy() == "train") & valid
        prefix = f"learned_w{window}"
        frame = pd.DataFrame(index=combined.index)
        frame[f"{prefix}_cluster"] = -1
        frame[f"{prefix}_confidence"] = 0.0
        frame[f"{prefix}_distance"] = np.nan
        frame[f"{prefix}_changed"] = 0
        if train_mask.sum() < max(80, config.learned_event_clusters * 20):
            new_frames.append(frame)
            continue
        ncomp = min(config.learned_event_pca_components, X.shape[1], int(train_mask.sum()) - 1)
        nclust = min(config.learned_event_clusters, max(2, int(train_mask.sum()) // 50))
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        pca = PCA(n_components=ncomp, random_state=config.seed + int(window))
        km = KMeans(n_clusters=nclust, random_state=config.seed + int(window), n_init=10)
        Z_train = pca.fit_transform(scaler.fit_transform(imputer.fit_transform(X[train_mask])))
        km.fit(Z_train)
        valid_idx = np.where(valid)[0]
        Z = pca.transform(scaler.transform(imputer.transform(X[valid])))
        dist = km.transform(Z)
        lab = np.argmin(dist, axis=1).astype(int)
        best = dist[np.arange(len(lab)), lab]
        frame.iloc[valid_idx, frame.columns.get_loc(f"{prefix}_cluster")] = lab
        frame.iloc[valid_idx, frame.columns.get_loc(f"{prefix}_confidence")] = 1.0 / (1.0 + best)
        frame.iloc[valid_idx, frame.columns.get_loc(f"{prefix}_distance")] = best
        frame[f"{prefix}_prev_cluster"] = frame[f"{prefix}_cluster"].shift(1).fillna(-1).astype(int)
        frame[f"{prefix}_changed"] = (frame[f"{prefix}_cluster"] != frame[f"{prefix}_prev_cluster"]).astype(float)
        for k in range(nclust):
            flag = (frame[f"{prefix}_cluster"] == k).astype(float)
            frame[f"{prefix}_evt_{k}"] = flag
            frame[f"{prefix}_evt_{k}_freq30"] = flag.rolling(30, min_periods=1).mean()
        for j in range(ncomp):
            frame[f"{prefix}_pc{j}"] = 0.0
            frame.iloc[valid_idx, frame.columns.get_loc(f"{prefix}_pc{j}")] = Z[:, j]
        new_frames.append(frame)

    if new_frames:
        combined = pd.concat([combined] + new_frames, axis=1)
    combined = combined.drop(columns=["__split"])
    new_train = combined.iloc[:len(train)].copy()
    new_val = combined.iloc[len(train):len(train) + len(val)].copy()
    new_test = combined.iloc[len(train) + len(val):].copy()
    return new_train, new_val, new_test
