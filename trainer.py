# --- DROP-IN PATCH: Make the pipeline robust to your CSVs and missing sources ---
# Place this near the top of dte_model_trainer.py (after your standard imports), or replace the file with this.
# It defines all helpers referenced in main(), robust CSV loaders, and fixes your undefined variables.

import os
import re
import json
import random
import warnings
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def list_metasys_columns_from_file(path: str, max_show: int = 200) -> list[str]:
    df = load_metasys_csv(path)
    return df.columns.tolist()[:max_show]

def resolve_target_alias(requested: str, available_cols: list[str]) -> str | None:
    """
    If exact match not found, try regex/aliases that map 'SteamFlow_Main' to real Metasys names.
    Customize patterns as you learn your site’s tags.
    """
    # Exact
    if requested in available_cols:
        return requested

    # Common aliases -> regex patterns (edit to taste)
    alias_patterns = {
        "SteamFlow_Main": r"(Steam.*Flow|HX1.*FLOW|CHPWTR.*MTR.*Present_Value|HX1.*FLOW.*Present_Value)",
        "Boiler1_FiringRate": r"Firing_Rate_Boiler_1_.*",
        "Boiler2_FiringRate": r"Firing_Rate_Boiler_2_.*",
        "Boiler3_FiringRate": r"Firing_Rate_Boiler_3_.*",
        "CHP_Water_Meter": r"CHPWTR_MTR_.*Present_Value.*",
    }
    pattern = alias_patterns.get(requested, requested)  # let user pass a regex, too
    try:
        rx = re.compile(pattern, re.I)
    except re.error:
        return None
    matches = [c for c in available_cols if rx.search(c)]
    if matches:
        # Heuristic: prefer columns containing FLOW or Present_Value
        matches.sort(key=lambda s: (("FLOW" not in s.upper()), ("PRESENT_VALUE" not in s.upper()), len(s)))
        return matches[0]
    return None


def _is_finite_array(x: np.ndarray) -> bool:
    return x is not None and np.isfinite(x).all() and x.size > 0

def _safe_metrics_inv(y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray, y_scaler) -> tuple[dict, bool]:
    """
    Inverse-transform y, drop any rows with NaN/Inf, and compute metrics.
    Returns (metrics_dict, ok_flag). If not ok, metrics_dict has rmse=inf.
    """
    try:
        y_true_inv = y_scaler.inverse_transform(y_true_scaled)
        y_pred_inv = y_scaler.inverse_transform(y_pred_scaled)
    except Exception:
        return {"rmse": float("inf"), "mae": float("inf"), "mape": float("inf"), "r2": float("-inf")}, False

    if y_true_inv.ndim == 1:
        y_true_inv = y_true_inv.reshape(-1, 1)
    if y_pred_inv.ndim == 1:
        y_pred_inv = y_pred_inv.reshape(-1, 1)

    # Drop rows where either side is non-finite
    mask = np.isfinite(y_true_inv.ravel()) & np.isfinite(y_pred_inv.ravel())
    if mask.sum() == 0:
        return {"rmse": float("inf"), "mae": float("inf"), "mape": float("inf"), "r2": float("-inf")}, False

    yt = y_true_inv.ravel()[mask]
    yp = y_pred_inv.ravel()[mask]
    try:
        m = metrics(yt, yp)  # your existing metrics(y_true, y_pred)
        return m, True
    except Exception:
        return {"rmse": float("inf"), "mae": float("inf"), "mape": float("inf"), "r2": float("-inf")}, False


def _exists(path: str) -> bool:
    return isinstance(path, str) and len(path.strip()) > 0 and os.path.exists(path)

def _try_load(label: str, path: str, loader_fn):
    if not _exists(path):
        warnings.warn(f"{label}: path missing or not found: {path}")
        return None
    try:
        df = loader_fn(path)
        if df is None or len(df) == 0:
            warnings.warn(f"{label}: loaded but empty: {path}")
            return None
        return df
    except Exception as e:
        warnings.warn(f"{label}: failed to load {path}: {e}")
        return None

def _report_coverage(dct, out_dir):
    lines = ["label,rows,start,end,non_null_target_pct"]
    for lbl, df in dct.items():
        if df is None or len(df) == 0:
            lines.append(f"{lbl},0,,,")
            continue
        start = getattr(df.index.min(), "isoformat", lambda: str(df.index.min()))()
        end   = getattr(df.index.max(), "isoformat", lambda: str(df.index.max()))()
        lines.append(f"{lbl},{len(df)},{start},{end},")
    p = os.path.join(out_dir, "source_coverage.csv")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Wrote source coverage to {p}")


def drop_sparse_columns(
    df: pd.DataFrame,
    max_nan_ratio: float = 0.5,
    keep: list | None = None,
    within_mask: pd.Series | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop columns whose NaN ratio exceeds max_nan_ratio.
    - keep: columns to protect from dropping
    - within_mask: measure sparsity only within rows where mask is True
      (useful to measure sparsity within the target's time coverage)
    Returns: (df_pruned, dropped_cols)
    """
    keep = keep or []
    view = df.loc[within_mask] if within_mask is not None else df
    if view.empty:
        return df, []

    nan_ratio = view.isna().mean()
    to_drop = [c for c, r in nan_ratio.items() if (r > max_nan_ratio and c not in keep)]
    if to_drop:
        df = df.drop(columns=to_drop, errors="ignore")
    return df, to_drop


def light_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Forward/backward fill within reasonable limits; then interpolate."""
    out = df.copy()
    out = out.ffill(limit=6).bfill(limit=2)
    # Interpolate numeric columns only
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].interpolate(limit=4, limit_direction="both")
    return out


# -------------------- Repro --------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -------------------- Time Features --------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must be indexed by datetime before adding time features.")
    out = df.copy()
    hod = out.index.hour.to_numpy()
    dow = out.index.dayofweek.to_numpy()
    out["hod_sin"] = np.sin(2 * np.pi * hod / 24.0)
    out["hod_cos"] = np.cos(2 * np.pi * hod / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return out


# -------------------- Feature Engineering --------------------
def _existing(cols: List[str], df_cols: pd.Index) -> List[str]:
    s = set(df_cols)
    return [c for c in cols if c in s]

def add_lagged_features(df: pd.DataFrame, base_cols: list[str], lags: list[int]) -> pd.DataFrame:
    base_cols = [c for c in base_cols if c in df.columns]
    if not base_cols or not lags:
        return df
    lagged = {}
    for c in base_cols:
        for L in lags:
            lagged[f"{c}_lag{L}"] = df[c].shift(L)
    lag_df = pd.DataFrame(lagged, index=df.index)
    return pd.concat([df, lag_df], axis=1)

def add_rolling_features(df: pd.DataFrame, base_cols: List[str], windows: List[int]) -> pd.DataFrame:
    out = df.copy()
    base_cols = _existing(base_cols, out.columns)
    for c in base_cols:
        for w in windows:
            out[f"{c}_roll{w}m"] = out[c].rolling(window=w, min_periods=max(1, w // 3)).mean()
    return out


# -------------------- Sequence Builder --------------------
def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    # Predict y at t + horizon using history [t-seq_len+1 ... t]
    for t in range(seq_len - 1, len(X) - horizon):
        xs.append(X[t - seq_len + 1 : t + 1, :])
        ys.append(y[t + horizon, :])
    return np.asarray(xs), np.asarray(ys)


# -------------------- Model & Metrics --------------------
def build_lstm_model(seq_len: int, n_features: int, units: int = 64, dropout: float = 0.2, lr: float = 1e-3) -> tf.keras.Model:
    model = Sequential()
    model.add(LSTM(units, input_shape=(seq_len, n_features), return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-9, None))) * 100.0)
    r2   = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}

def safe_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        print("[WARN] No finite points to score; returning NaNs for metrics.")
        return {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan"), "r2": float("nan")}
    return metrics(y_true[mask], y_pred[mask])





# -------------------- Robust CSV Loaders --------------------
def _parse_dt(s: pd.Series, fmts: List[str]) -> pd.Series:
    s = s.astype(str).str.strip()
    for f in fmts:
        try:
            dt = pd.to_datetime(s, format=f, errors="raise")
            return dt
        except Exception:
            continue
    return pd.to_datetime(s, errors="coerce")

def ensure_datetime_index(df: pd.DataFrame, ts_candidates: List[str], tz=None) -> pd.DataFrame:
    found = None
    for c in ts_candidates:
        if c in df.columns:
            found = c
            break
    if found is None:
        raise ValueError(f"No timestamp column found among {ts_candidates}. Columns={df.columns.tolist()}")

    fmts = [
        "%Y-%m-%d %H:%M:%S",   # 2023-01-26 00:15:00
        "%m/%d/%Y %H:%M",      # 6/20/2023 14:00
        "%H:%M %m/%d/%Y",      # 08:02 5/28/2023
        "%Y-%m-%d %H:%M",      # 2023-05-28 00:00
    ]
    ts = _parse_dt(df[found], fmts)
    df = df.copy()
    df.drop(columns=[found], inplace=True)
    df.insert(0, "timestamp", ts)
    df = df.dropna(subset=["timestamp"])
    if tz:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
    df = df.set_index("timestamp").sort_index()
    return df

def load_metasys_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]
    req = {"Date / Time", "Object Name", "Object Value"}
    if not req.issubset(df.columns):
        raise ValueError(f"Metasys CSV missing required columns {req}. Got {df.columns.tolist()}")
    df = ensure_datetime_index(df, ["Date / Time"])
    df["Object Value"] = pd.to_numeric(df["Object Value"], errors="coerce")
    wide = df.pivot_table(values="Object Value", index=df.index, columns="Object Name", aggfunc="last")
    wide.columns = (
        wide.columns.astype(str)
        .str.strip()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace(r"_+$", "", regex=True)
    )
    return wide

def load_noaa_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Trim all header names to handle leading/trailing spaces
    df.columns = [c.strip() for c in df.columns]
    # Handle common variants like "Date " / " date"
    if "Date" not in df.columns:
        # Try a fuzzy match
        candidates = [c for c in df.columns if c.strip().lower() == "date"]
        if not candidates:
            raise ValueError(f"NOAA CSV missing 'Date' column. Columns={df.columns.tolist()}")
        df.rename(columns={candidates[0]: "Date"}, inplace=True)

    df = ensure_datetime_index(df, ["Date"])
    # Select HR 0 columns robustly (allow multiple spaces)
    hr0 = df.filter(regex=r"^HR\s*0\s+", axis=1).copy()

    # Map/rename whatever HR 0 fields exist
    rename_map = {
        "HR 0 Temperature": "noaa_temp",
        "HR 0 Humidity": "noaa_hum",
        "HR 0 Wind Speed": "noaa_wind",
        "HR 0 Wind Direction": "noaa_wdir",
        "HR 0 Cloud Amount": "noaa_cloud",
        "HR 0 Precipitation": "noaa_precip",
    }
    # Also allow collapsed space variants like "HR 0  Temperature"
    for k in list(rename_map.keys()):
        if k not in hr0.columns:
            # try forgiving search
            matches = [c for c in hr0.columns if c.replace("  ", " ").strip() == k]
            if matches:
                hr0.rename(columns={matches[0]: k}, inplace=True)

    keep = {k: v for k, v in rename_map.items() if k in hr0.columns}
    if not keep:
        # If HR 0 block is missing for some reason, fall back to anything useful
        fallback = df.copy()
        for col in fallback.columns:
            fallback[col] = pd.to_numeric(fallback[col], errors="coerce")
        fallback = fallback.dropna(how="all")
        return fallback

    hr0 = hr0.rename(columns=keep)
    for col in hr0.columns:
        hr0[col] = pd.to_numeric(hr0[col], errors="coerce")
    hr0 = hr0.dropna(how="all")
    return hr0

def load_onsite_weather_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().strip(",") for c in df.columns]
    time_candidates = ["Date / Time", "Date/Time", "DateTime", "Time", df.columns[0]]
    df = ensure_datetime_index(df, time_candidates)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    rename_map = {
        "outTemp": "ws_outTemp",
        "outHumi": "ws_outHum",
        "avgwind": "ws_wind",
        "solarrad": "ws_solarrad",
        "uvi": "ws_uvi",
        "rainofhourly": "ws_rain_hour",
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)
    return df

def load_window_sensors_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Trim and drop completely empty columns (many CSVs have trailing commas -> empty headers)
    new_cols = []
    for c in df.columns:
        cc = (c if isinstance(c, str) else str(c)).strip()
        new_cols.append(cc)
    df.columns = new_cols
    df = df.drop(columns=[c for c in df.columns if c == "" or df[c].isna().all()], errors="ignore")

    df = ensure_datetime_index(df, ["observed"])

    # Find *_temperature / *_humidity columns after cleanup
    temp_cols = [c for c in df.columns if isinstance(c, str) and c.endswith("_temperature")]
    hum_cols  = [c for c in df.columns if isinstance(c, str) and c.endswith("_humidity")]

    out = pd.DataFrame(index=df.index)
    if temp_cols:
        temps = df[temp_cols].apply(pd.to_numeric, errors="coerce")
        out["win_temp_mean"] = temps.mean(axis=1, skipna=True)
    if hum_cols:
        hums = df[hum_cols].apply(pd.to_numeric, errors="coerce")
        out["win_hum_mean"] = hums.mean(axis=1, skipna=True)

    # If nothing usable, return empty to let the pipeline skip it gracefully
    return out

def load_occupancy_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = ensure_datetime_index(df, ["Interval stop", "Interval Stop", "interval_stop"])
    keep = {}
    for c in ["Occupancy", "Total In", "Total Out"]:
        if c in df.columns:
            keep[c.lower().replace(" ", "_")] = pd.to_numeric(df[c], errors="coerce")
    occ = pd.DataFrame(keep, index=df.index)
    return occ

def load_gas_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = ensure_datetime_index(df, ["date_time", "timestamp", "Date/Time"])
    out = pd.DataFrame(index=df.index)
    if "meter_readings" in df.columns:
        out["gas_meter_readings"] = pd.to_numeric(df["meter_readings"], errors="coerce")
    if "flow" in df.columns:
        out["gas_flow"] = pd.to_numeric(df["flow"], errors="coerce")
    return out

def merge_sources_hourly(metasys=None, noaa=None, onsite=None, windows=None, occupancy=None, gas=None, how="outer"):
    used, frames = [], []
    def _prep(name, df):
        if df is None or df.empty:
            return
        x = df.copy()
        # Resample to hourly and forward-fill short gaps (up to 6h)
        x = x.resample("1h").mean().ffill(limit=6)
        frames.append(x)
        used.append(name)

    _prep("metasys", metasys)
    _prep("noaa", noaa)
    _prep("onsite_weather", onsite)
    _prep("window_sensors", windows)
    _prep("occupancy", occupancy)
    _prep("gas_meter", gas)

    if not frames:
        raise RuntimeError("No data sources available to merge.")
    merged = pd.concat(frames, axis=1, join=how).sort_index()
    merged = merged.dropna(how="all")
    return merged, used

def load_all_sources_from_args(args) -> tuple[pd.DataFrame, list[str]]:
    """
    Load Metasys (required) and any optional sources, align to hourly, and outer-join.
    Also resolves the target against Metasys columns *before* merging others.
    Returns:
      df (pd.DataFrame): merged table indexed by hourly timestamp
      used_sources (list[str]): which logical sources were actually included
    """
    used = []
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # --- helpers for optional loading ---
    def _exists(p): 
        return (p is not None) and isinstance(p, (str, os.PathLike)) and os.path.exists(p)

    def _describe_source(name: str, x: pd.DataFrame | None) -> dict:
        if x is None or len(x) == 0:
            return {"source": name, "rows": 0, "start": "", "end": ""}
        idx = x.index if isinstance(x.index, pd.DatetimeIndex) else None
        start = str(idx.min()) if idx is not None and len(idx) else ""
        end   = str(idx.max()) if idx is not None and len(idx) else ""
        return {"source": name, "rows": int(len(x)), "start": start, "end": end}

    # --- Metasys (required) ---
    if not _exists(args.metasys_csv):
        raise FileNotFoundError(f"Metasys CSV not found: {args.metasys_csv}")
    metasys_raw = load_metasys_csv(args.metasys_csv)  # assumed helper
    if metasys_raw is None or len(metasys_raw) == 0:
        raise ValueError("Metasys CSV loaded but empty.")
    # Hourly align (mean within hour)
    metasys = metasys_raw.sort_index().resample("1h").mean()

    # Resolve target *against Metasys columns* before merging others
    metasys_cols = list(metasys.columns)
    meta_cols_path = os.path.join(out_dir, "metasys_columns_premerge.txt")
    with open(meta_cols_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metasys_cols))
    print(f"[INFO] Wrote Metasys column list to {meta_cols_path}")

    if args.target not in metasys_cols:
        cand = resolve_target_alias(args.target, metasys_cols)  # assumed helper
        if cand and cand in metasys_cols:
            print(f"[INFO] Target '{args.target}' resolved to '{cand}' from Metasys columns.")
            args.target = cand
        else:
            print("\n[ERROR] Target not found in Metasys after sanitization.")
            print(f"[HINT] Choose one from {meta_cols_path}")
            raise ValueError(f"Target column '{args.target}' not in Metasys columns.")

    df = metasys.copy()
    used.append("metasys")

    # --- Optional sources: load if present, else skip quietly ---
    noaa = None
    if _exists(args.noaa_csv):
        try:
            noaa = load_noaa_csv(args.noaa_csv)  # assumed helper
            if noaa is not None and len(noaa) > 0:
                noaa = noaa.sort_index().resample("1h").mean()
                df = df.join(noaa, how="outer")
                used.append("noaa")
        except Exception as e:
            warnings.warn(f"NOAA: failed to load {args.noaa_csv}: {e}")

    onsite = None
    if _exists(args.onsite_csv):
        try:
            onsite = load_onsite_weather_csv(args.onsite_csv)  # assumed helper
            if onsite is not None and len(onsite) > 0:
                onsite = onsite.sort_index().resample("1h").mean()
                df = df.join(onsite, how="outer")
                used.append("onsite_weather")
        except Exception as e:
            warnings.warn(f"Onsite weather: failed to load {args.onsite_csv}: {e}")

    window = None
    if _exists(args.window_csv):
        try:
            window = load_window_sensors_csv(args.window_csv)  # assumed helper
            if window is not None and len(window) > 0:
                window = window.sort_index().resample("1h").mean()
                df = df.join(window, how="outer")
                used.append("window_sensors")
        except Exception as e:
            warnings.warn(f"Window sensors: failed to load {args.window_csv}: {e}")

    occupancy = None
    if _exists(args.occupancy_csv):
        try:
            occupancy = load_occupancy_csv(args.occupancy_csv)  # assumed helper
            if occupancy is not None and len(occupancy) > 0:
                occupancy = occupancy.sort_index().resample("1h").mean()
                df = df.join(occupancy, how="outer")
                used.append("occupancy")
        except Exception as e:
            warnings.warn(f"Occupancy: failed to load {args.occupancy_csv}: {e}")

    gas = None
    if _exists(args.gas_csv):
        try:
            gas = load_gas_csv(args.gas_csv)  # assumed helper
            if gas is not None and len(gas) > 0:
                gas = gas.sort_index().resample("1h").mean()
                df = df.join(gas, how="outer")
                used.append("gas_meter")
        except Exception as e:
            warnings.warn(f"Gas meter: failed to load {args.gas_csv}: {e}")

    # --- Write a quick coverage report for convenience ---
    coverage_rows = [
        _describe_source("metasys", metasys),
        _describe_source("noaa", noaa),
        _describe_source("onsite_weather", onsite),
        _describe_source("window_sensors", window),
        _describe_source("occupancy", occupancy),
        _describe_source("gas_meter", gas),
    ]
    cov_path = os.path.join(out_dir, "source_coverage.csv")
    pd.DataFrame(coverage_rows).to_csv(cov_path, index=False)
    print(f"[INFO] Wrote source coverage to {cov_path}")

    # --- Final tidy ---
    df = df.sort_index()
    cols_path = os.path.join(out_dir, "columns_available.txt")
    with open(cols_path, "w", encoding="utf-8") as f:
        f.write("\n".join(df.columns.tolist()))
    print(f"[INFO] Wrote column list to {cols_path}")

    return df, used


# -------------------- MAIN (patch your undefined variables & columns) --------------------
def main():
    parser = argparse.ArgumentParser(description="DTE End-to-End Forecasting Pipeline")
    # Inputs (some optional)
    parser.add_argument("--metasys_csv", required=True)
    parser.add_argument("--noaa_csv", required=False)
    parser.add_argument("--occupancy_csv", required=False)
    parser.add_argument("--window_csv", required=False)
    parser.add_argument("--gas_csv", required=False)
    parser.add_argument("--onsite_xml_dir", default=None)  # kept for CLI compat
    parser.add_argument("--onsite_csv", default=None)

    # Columns
    parser.add_argument("--target", default="SteamFlow_Main")

    # Modeling
    parser.add_argument("--seq_length", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=1, help="Predict t+H hours ahead")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=128)  # will be clamped safely
    parser.add_argument("--search_trials", type=int, default=12)

    # Split control
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--out_dir", default="./artifacts")

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------ Load & align all sources ------------------
    df, used_sources = load_all_sources_from_args(args)

    # Try to resolve target if not present (Metasys aliases)
    if args.target not in df.columns:
        if "metasys" in used_sources:
            metasys_cols = [c for c in df.columns if c.endswith("Trend1") or "CHPWTR" in c or "HX1" in c]
            resolved = resolve_target_alias(args.target, metasys_cols)
            if resolved and resolved in df.columns:
                print(f"[INFO] Target '{args.target}' resolved to '{resolved}'.")
                args.target = resolved

    if args.target not in df.columns:
        print("\n[ERROR] Target column not found after merging.")
        if "metasys" in used_sources:
            metasys_cols_all = [c for c in df.columns if c.endswith("Trend1") or "CHPWTR" in c or "HX1" in c]
            print("Metasys columns sample (first 100):")
            print(metasys_cols_all[:100])
            print("Tip: re-run with one of the names above, e.g.:")
            print("  --target HX1HWR_F_HX1_FLOW_Trend_Present_Value_Trend1")
        else:
            print("Metasys not included. Ensure --metasys_csv path is correct.")
        raise ValueError(f"Target column '{args.target}' not found.")

    # ------------------ Feature engineering ------------------
    # Keep FE light when dataset is tiny; heavy FE can annihilate rows
    small_df = len(df) < 200
    if small_df:
        print(f"[INFO] Small merged dataset (rows={len(df)}). Using lighter FE.")
        lag_candidates = [args.target, "noaa_temp", "ws_outTemp", "gas_flow", "occupancy"]
        df = add_lagged_features(df, lag_candidates, lags=[1, 2])
        # no rolling features on tiny data
    else:
        base_lag_candidates = [
            args.target,
            "noaa_temp", "noaa_hum", "noaa_wind", "noaa_cloud", "noaa_precip",
            "ws_outTemp", "ws_outHum", "ws_wind", "ws_solarrad", "ws_rain_hour",
            "win_temp_mean", "win_hum_mean",
            "gas_flow", "gas_meter_readings",
            "occupancy", "total_in", "total_out",
        ]
        df = add_lagged_features(df, base_lag_candidates, lags=[1, 2, 3, 6, 12, 24])

        roll_candidates = [
            args.target,
            "noaa_temp", "noaa_hum", "noaa_wind", "ws_outTemp", "ws_outHum", "ws_wind",
            "ws_solarrad", "gas_flow"
        ]
        df = add_rolling_features(df, roll_candidates, windows=[3, 6, 12, 24])

    # Drop rows with NaNs introduced by lags/rolls and ensure target finite
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[args.target]).dropna()
    prepared_path = os.path.join(args.out_dir, "prepared_dataset.csv")
    try:
        df.to_csv(prepared_path, index=True)
        print(f"[INFO] Wrote prepared dataset to {prepared_path}")
    except Exception:
        pass

    if len(df) < (args.seq_length + args.horizon + 2):
        print(f"[WARN] Very little data after prep (len={len(df)}). "
              f"Consider smaller --seq_length or fewer lags/rolls for a smoke run.")

    # ------------------ Train/Valid/Test split by time ------------------
    target = df[args.target].values.reshape(-1, 1)
    features = df.drop(columns=[args.target])

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = X_scaler.fit_transform(features.values) if len(features) > 0 else np.empty((0, 0))
    y_scaled = y_scaler.fit_transform(target)

    X_seq, y_seq = build_sequences(X_scaled, y_scaled, args.seq_length, args.horizon)

    N = len(X_seq)
    n_test = int(N * args.test_ratio)
    n_valid = int(N * args.valid_ratio)
    n_train = N - n_valid - n_test

    if n_train <= 0:
        raise ValueError("Insufficient data after sequence building. Reduce seq_length or adjust split ratios.")

    X_train, y_train = X_seq[:n_train], y_seq[:n_train]
    X_valid, y_valid = (X_seq[n_train:n_train + n_valid], y_seq[n_train:n_train + n_valid]) if n_valid > 0 else (None, None)
    X_test,  y_test  = (X_seq[n_train + n_valid:],        y_seq[n_train + n_valid:])        if n_test  > 0 else (None, None)

    n_features = X_train.shape[2]

    # Safe batch size for tiny data
    min_split = len(X_train) if X_valid is None else max(1, min(len(X_train), len(X_valid)))
    safe_batch = max(1, min(args.batch_size, min(8, min_split)))

    # ------------------ Hyperparameter search (robust to tiny data) ------------------
    search_space = {
        "units": [32, 64, 128],
        "dropout": [0.1, 0.2, 0.3],
        "lr": [5e-4, 1e-3, 2e-3],
    }

    best = {"score": float("inf")}
    for trial in range(1, args.search_trials + 1):
        units = random.choice(search_space["units"])
        dropout = random.choice(search_space["dropout"])
        lr = random.choice(search_space["lr"])
        batch_size = safe_batch  # clamp

        # Build model
        model = build_lstm_model(args.seq_length, n_features, units=units, dropout=dropout, lr=lr)
        ckpt_path = os.path.join(args.out_dir, f"model_trial{trial:02d}.keras")

        use_val = X_valid is not None and len(X_valid) > 0
        callbacks = []
        if use_val:
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
                ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, save_weights_only=False),
            ]

        # Train with guards
        try:
            if use_val:
                model.fit(
                    X_train, y_train,
                    validation_data=(X_valid, y_valid),
                    epochs=args.epochs,
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=callbacks,
                )
                pred_scaled = model.predict(X_valid, verbose=0)
                true_scaled = y_valid
            else:
                model.fit(
                    X_train, y_train,
                    epochs=args.epochs,
                    batch_size=batch_size,
                    verbose=0,
                )
                pred_scaled = model.predict(X_train, verbose=0)
                true_scaled = y_train
        except Exception as e:
            print(f"TRIAL {{\"trial\": {trial}, \"units\": {units}, \"dropout\": {dropout}, \"lr\": {lr}, "
                  f"\"batch_size\": {batch_size}, \"score\": \"failed/train:{str(e)}\"}}")
            continue

        # Inverse transform + metrics with NaN guards
        try:
            pred_inv = y_scaler.inverse_transform(pred_scaled)
            true_inv = y_scaler.inverse_transform(true_scaled)
            if (not np.isfinite(pred_inv).all()) or (not np.isfinite(true_inv).all()):
                raise ValueError("non-finite values in predictions/targets after inverse transform")
            m = metrics(true_inv.ravel(), pred_inv.ravel())
            score = float(m["rmse"]) if np.isfinite(list(m.values())).all() else float("inf")
        except Exception as e:
            print(f"[WARN] Non-finite score (likely NaNs) for trial; units={units}, dropout={dropout}, lr={lr}, batch={batch_size}")
            m = {"rmse": float("inf"), "mae": float("inf"), "mape": float("inf"), "r2": float("-inf")}
            score = float("inf")

        # Try saving the model for reference
        try:
            model.save(ckpt_path)
        except Exception:
            ckpt_path = None

        trial_log = {
            "trial": trial, "units": units, "dropout": dropout, "lr": lr, "batch_size": batch_size,
            "train_rmse": float(m.get("rmse", np.inf)),
            "train_mae":  float(m.get("mae",  np.inf)),
            "train_mape": float(m.get("mape", np.inf)),
            "train_r2":   float(m.get("r2",  -np.inf)),
            "score": score
        }
        print("TRIAL", json.dumps(trial_log))
        _append_jsonl(os.path.join(args.out_dir, "search_log.jsonl"), trial_log)

        if np.isfinite(score) and score < best["score"]:
            best = {**trial_log, "ckpt": ckpt_path}

    print("BEST", json.dumps({k: v for k, v in best.items() if k != "ckpt"}, indent=2))

    # If no usable trial, exit gracefully with a summary
    if not np.isfinite(best.get("score", np.inf)):
        print("[ERROR] No usable trials (all produced non-finite predictions).")
        print("Hints:")
        print(" - Reduce FE for a smoke test: only lags=[1,2], no rolling.")
        print(" - Lower --seq_length (e.g., 3) and --epochs (e.g., 5).")
        print(" - Temporarily train on Metasys only to verify target signal.")
        print(" - Inspect prepared_dataset.csv to confirm finite target values.")
        summary = {
            "n_samples": int(len(df)),
            "n_seq": int(len(X_seq)) if 'X_seq' in locals() else 0,
            "seq_length": args.seq_length,
            "horizon": args.horizon,
            "splits": {"train": int(len(X_train)), "valid": int(len(X_valid) if X_valid is not None else 0), "test": int(len(X_test) if X_test is not None else 0)},
            "best": best,
            "sources_used": used_sources,
            "artifacts": {"prepared_dataset": prepared_path},
            "note": "No usable trials; all scores were non-finite."
        }
        with open(os.path.join(args.out_dir, "run_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return

    # ------------------ Final model & optional TEST ------------------
    best_ckpt = best.get("ckpt")
    if best_ckpt and os.path.exists(best_ckpt):
        best_model = tf.keras.models.load_model(best_ckpt)
    else:
        best_model = build_lstm_model(args.seq_length, n_features, units=best["units"], dropout=best["dropout"], lr=best["lr"])

    if X_test is not None and len(X_test) > 0:
        test_pred = best_model.predict(X_test, verbose=0)
        try:
            test_pred_inv = y_scaler.inverse_transform(test_pred)
            test_true_inv = y_scaler.inverse_transform(y_test)
            if (not np.isfinite(test_pred_inv).all()) or (not np.isfinite(test_true_inv).all()):
                raise ValueError("non-finite values in TEST predictions/targets")
            test_metrics = metrics(test_true_inv.ravel(), test_pred_inv.ravel())
        except Exception:
            test_metrics = {"rmse": float("inf"), "mae": float("inf"), "mape": float("inf"), "r2": float("-inf")}
    else:
        print("[INFO] No test split (dataset too small). Skipping TEST evaluation.")
        test_metrics = None

    # Save artifacts
    best_model_path = os.path.join(args.out_dir, "best_model.keras")
    best_model.save(best_model_path)

    import joblib
    joblib.dump(X_scaler, os.path.join(args.out_dir, "X_scaler.pkl"))
    joblib.dump(y_scaler, os.path.join(args.out_dir, "y_scaler.pkl"))
    with open(os.path.join(args.out_dir, "feature_columns.json"), "w") as f:
        json.dump(features.columns.tolist(), f, indent=2)

    summary = {
        "n_samples": int(len(df)),
        "n_seq": int(len(X_seq)),
        "seq_length": args.seq_length,
        "horizon": args.horizon,
        "splits": {"train": int(len(X_train)), "valid": int(len(X_valid) if X_valid is not None else 0), "test": int(len(X_test) if X_test is not None else 0)},
        "best": {k: v for k, v in best.items() if k != "ckpt"},
        "test_metrics": test_metrics,
        "sources_used": used_sources,
        "artifacts": {
            "model": best_model_path,
            "X_scaler": os.path.join(args.out_dir, "X_scaler.pkl"),
            "y_scaler": os.path.join(args.out_dir, "y_scaler.pkl"),
            "features": os.path.join(args.out_dir, "feature_columns.json"),
            "search_log": os.path.join(args.out_dir, "search_log.jsonl"),
            "prepared_dataset": prepared_path,
        }
    }
    with open(os.path.join(args.out_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Artifacts saved to:", os.path.abspath(args.out_dir))


def _append_jsonl(path: str, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
