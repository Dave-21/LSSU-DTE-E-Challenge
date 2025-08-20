# -*- coding: utf-8 -*-
from pathlib import Path
import os, zipfile, traceback, shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

tf.keras.utils.disable_interactive_logging()

# ---------- Paths ----------
ROOT    = Path(__file__).resolve().parent
DATA    = ROOT / "data"
# Primary models dir + common legacy dirs that held models previously
MODELS  = ROOT / "models"
EXTRA_MODEL_DIRS = [
    ROOT / "Archived KERAS Files",
    ROOT / "AIRunsRiley" / "Models",
]
RESULTS = ROOT / "results" / "ModelTester"
PLOTS   = RESULTS / "plots"
for p in (RESULTS, PLOTS):
    p.mkdir(parents=True, exist_ok=True)

# ---------- Config ----------
SEQUENCE_LENGTH = 10
TARGET_COL = "Library - Tomorrow Max Energy"

# Choose your hold-out CSV here (must include TARGET_COL).
# Tip: for older archived models trained circa May23–May24, set this to:
# DATA / "Library_Data_May23-May24.csv"
EVAL_CSV   = DATA / "Library_Data_June24-Jul24.csv"

# Preferred scaler source (if present). Otherwise falls back to MODEL_INPUT, then EVAL_CSV.
TRAIN_DS   = DATA / "TrainingDataset.csv"
MODEL_INPUT= DATA / "Model_Input_Data.csv"
SCHEMA_TXT = DATA / "columns_in_use.txt"  # definitive feature schema (order)

# Candidate targets present in your eval CSV
CANDIDATE_TARGETS = [
    ("Library - Energy Consumed Hourly (Kilowatts)", 15),   # ±15 kW for hourly
    ("Library - Daily Max Energy", 150),                    # ±150 kW for daily
    ("Library - Tomorrow Max Energy", 150),                 # ±150 kW for tomorrow max
]

# Optional: force a specific target via env var (hourly|daily|tomorrow)
FORCE_TARGET = os.getenv("DTELIB_TARGET", "").strip().lower()
TARGET_ALIAS = {
    "hourly": "Library - Energy Consumed Hourly (Kilowatts)",
    "daily": "Library - Daily Max Energy",
    "tomorrow": "Library - Tomorrow Max Energy",
}

import numpy as np

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size: return np.nan
    a = a.astype(np.float64); b = b.astype(np.float64)
    ad = a - a.mean(); bd = b - b.mean()
    denom = np.sqrt((ad*ad).sum() * (bd*bd).sum())
    return float((ad*bd).sum() / denom) if denom > 0 else np.nan

def pick_target_for_model(eval_df, y_pred, forced_alias_or_name: str | None):
    """
    Decide which column in eval_df is the model's true target.
    If forced_alias_or_name is provided, try that first (accepts alias 'hourly'/'daily'/'tomorrow'
    or an exact column name). Otherwise pick the column with the lowest MSE (corr as tiebreaker).
    Returns (col_name, y_true_np, mse, corr)
    """
    # resolve forced
    if forced_alias_or_name:
        forced = TARGET_ALIAS.get(forced_alias_or_name, forced_alias_or_name)
        if forced not in eval_df.columns:
            raise RuntimeError(f"Forced target '{forced}' not found in eval CSV columns.")
        y_true = eval_df[forced].to_numpy(dtype=np.float32)
        mse = float(np.mean((y_pred - y_true) ** 2))
        corr = _safe_corr(y_pred, y_true)
        return forced, y_true, mse, corr

    best = None
    for col in CANDIDATE_TARGETS:
        if col not in eval_df.columns:
            continue
        y_true = eval_df[col].to_numpy(dtype=np.float32)
        mse = float(np.mean((y_pred - y_true) ** 2))
        corr = _safe_corr(y_pred, y_true)
        key = (mse, -abs(corr))
        if best is None or key < best["key"]:
            best = {"col": col, "y": y_true, "mse": mse, "corr": corr, "key": key}
    if best is None:
        raise RuntimeError("None of the candidate targets are present in the eval CSV.")
    return best["col"], best["y"], best["mse"], best["corr"]

def _keras3_available():
    try:
        import keras as k3
        return getattr(k3, "__version__", "").startswith("3")
    except Exception:
        return False

# --- Add this shim right after your imports ---
# A drop-in LSTM that ignores unknown config keys like `time_major`
@tf.keras.utils.register_keras_serializable(package="Compat")
class CompatLSTM(tf.keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)   # swallow legacy arg
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        # Some legacy saves put time_major in the serialized config
        config.pop("time_major", None)
        return super().from_config(config)


def _keras3_available():
    try:
        import keras  # Keras 3
        return True
    except Exception:
        return False


def load_model_any(path):
    """
    Load legacy tf.keras `.keras`/H5 models and Keras 3 zip `.keras` models.
    Always returns an *uncompiled* model ready for predict().
    """
    import os, shutil, zipfile
    p = str(path)

    # Custom maps for both tf.keras and Keras 3 paths
    tf_custom = {
        "LSTM": CompatLSTM,                           # <-- key line
        "Bidirectional": tf.keras.layers.Bidirectional,
        "Dense": tf.keras.layers.Dense,
        "Dropout": tf.keras.layers.Dropout,
        "InputLayer": tf.keras.layers.InputLayer,
    }

    # 1) Try tf.keras loader first (best for legacy saves)
    try:
        return tf.keras.models.load_model(p, compile=False, custom_objects=tf_custom)
    except Exception as e1:
        last_err = e1

    # 2) If it’s a Keras 3 zip, try keras.models.load_model
    is_zip = False
    try:
        is_zip = zipfile.is_zipfile(p)
    except Exception:
        pass

    if is_zip and _keras3_available():
        try:
            import keras as k3
            k3_custom = {
                "LSTM": CompatLSTM,                       # still OK, it’s a Keras-serializable class
                "Bidirectional": k3.layers.Bidirectional,
                "Dense": k3.layers.Dense,
                "Dropout": k3.layers.Dropout,
                "InputLayer": k3.layers.InputLayer,
            }
            return k3.models.load_model(p, compile=False, custom_objects=k3_custom)
        except Exception as e2:
            last_err = e2

    # 3) Some legacy `.keras` files are actually HDF5—try that path
    if (not is_zip) and p.lower().endswith(".keras") and os.path.isfile(p):
        tmp_h5 = p + ".as_h5.h5"
        try:
            shutil.copy2(p, tmp_h5)
            return tf.keras.models.load_model(tmp_h5, compile=False, custom_objects=tf_custom)
        except Exception as e3:
            last_err = e3
        finally:
            try: os.remove(tmp_h5)
            except Exception: pass

    # Nothing worked
    raise last_err

# ---------- Helper functions ----------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Trim header whitespace and collapse internal double spaces
    ren = {c: " ".join(c.strip().split()) for c in df.columns}
    return df.rename(columns=ren)

def read_schema_cols(path: Path):
    # Read columns_in_use.txt, strip whitespace, drop target from list, keep order
    if not path.exists(): return None
    txt = path.read_text(encoding="utf-8")
    raw = [r for r in (txt.splitlines() if "\n" in txt else txt.split(",")) if r.strip()]
    cols = [" ".join(r.strip().split()) for r in raw]
    return [c for c in cols if c != TARGET_COL]  # drop target

def ensure_time_columns(df):
    need = {"Hour", "Year sin", "Year cos", "Day sin", "Day cos"}
    if need.issubset(df.columns): return df
    if "Date / Time" not in df.columns: return df
    ts = pd.to_datetime(df["Date / Time"])
    df = df.copy()
    df["Hour"] = ts.dt.hour
    day_of_year = ts.dt.dayofyear
    seconds_in_day = ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second
    df["Year sin"] = np.sin(day_of_year * (2*np.pi/365.2425))
    df["Year cos"] = np.cos(day_of_year * (2*np.pi/365.2425))
    df["Day sin"]  = np.sin(seconds_in_day * (2*np.pi/86400))
    df["Day cos"]  = np.cos(seconds_in_day * (2*np.pi/86400))
    return df

def enrich_from_model_input(eval_df, expected_features):
    """If eval_df is missing any expected features and we have Model_Input_Data.csv with Date / Time,
       left-join the missing columns onto eval_df."""
    if not MODEL_INPUT.exists() or "Date / Time" not in eval_df.columns:
        return eval_df
    mi = pd.read_csv(MODEL_INPUT)
    mi = normalize_columns(mi)
    if "Date / Time" not in mi.columns:
        return eval_df
    missing = [c for c in expected_features if c not in eval_df.columns and c in mi.columns]
    if not missing:
        return eval_df
    cols_to_add = ["Date / Time"] + missing
    merged = eval_df.merge(mi[cols_to_add], on="Date / Time", how="left")
    return merged

def difference_plot(df, tag, acc15, mse):
    x = df.index.to_list()
    actual_values = df["actual"].to_numpy(dtype=float).copy()
    pred_values   = df["prediction"].to_numpy(dtype=float)
    actual_values[actual_values == 0] = 1e-6
    pct_err = ((pred_values - actual_values) / actual_values) * 100.0

    colors = ['blue', 'green', 'red']  # under, accurate, over
    cmap = LinearSegmentedColormap.from_list('blue-green-red', colors)
    norm = TwoSlopeNorm(vmin=-50, vcenter=0, vmax=50)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(x, actual_values, color='green', marker='o', linestyle='None')
    axes[0].set_xlabel('Testing Hour Indices'); axes[0].set_ylabel('Energy (kWh)')
    axes[0].set_title('Actual Energy Consumption')
    axes[1].scatter(x, pred_values, c=pct_err, cmap=cmap, norm=norm, marker='o')
    axes[1].set_xlabel('Testing Hour Indices'); axes[1].set_ylabel('Energy (kWh)')
    axes[1].set_title('Prediction Error (%)')
    y_min = float(min(actual_values.min(), pred_values.min()))
    y_max = float(max(actual_values.max(), pred_values.max()))
    axes[0].set_ylim(y_min, y_max); axes[1].set_ylim(y_min, y_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=axes[1], label="Prediction Error (%)")
    outpath = PLOTS / f"{tag}_acc{acc15:.2f}_mse{mse:.2f}.png"
    plt.savefig(outpath); plt.close()
    return outpath

def acc15_from_df(df):
    a = df["actual"].to_numpy(dtype=float).copy()
    p = df["prediction"].to_numpy(dtype=float)
    a[a == 0] = 1e-6
    diff_pct = np.abs(p - a) / a * 100.0
    return (diff_pct < 15).mean() * 100.0

def build_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len + 1):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len-1])
    return np.asarray(X_seq), np.asarray(y_seq)

# --- Model file helpers (inventory + safe loader) ---
def find_all_model_files():
    files = []
    # main dir
    if MODELS.exists():
        files += sorted(MODELS.glob("*.keras"))
    # extras
    for d in EXTRA_MODEL_DIRS:
        if d.exists():
            files += sorted(d.glob("*.keras"))
    # de-duplicate by absolute path
    seen = set()
    uniq = []
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(rp)
    return uniq

def inventory(models_list):
    print("\nModel inventory:")
    for p in models_list:
        try:
            exists = p.exists()
            is_file = p.is_file()
            size = (p.stat().st_size if exists and is_file else 0)
            try:
                is_zip = zipfile.is_zipfile(p) if exists and is_file else False
            except Exception:
                is_zip = False
            print(f"  - {p} | exists={exists} is_file={is_file} size={size}B zip={is_zip}")
        except Exception as e:
            print(f"  - {p} | inventory error: {e}")

def safe_load_model(p: Path):
    # Catch common “zip/safe-mode”/placeholder issues early
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Missing or not a file: {p}")
    if p.stat().st_size == 0:
        raise FileNotFoundError(f"Zero-byte file: {p}")
    # Keras 3 loader with safe_mode disabled
    return tf.keras.saving.load_model(str(p), compile=False, safe_mode=False)

def score_one_model(model_path, X_eval, eval_df):
    """
    Loads a model, checks feature width, predicts, auto-selects the most likely
    target column from eval_df (or respects DTELIB_TARGET if set), and computes metrics.
    Returns: (mse, acc15pct, y_pred, matched_target_col)
    """
    # Load
    model = load_model_any(model_path)

    # Feature width check
    want_feats = int(model.inputs[0].shape[-1])
    have_feats = X_eval.shape[-1]
    if want_feats != have_feats:
        raise ValueError(
            f"Feature mismatch for {os.path.basename(model_path)}: "
            f"model expects {want_feats}, eval has {have_feats}."
        )

    # Predict (no compile needed)
    y_pred = model.predict(X_eval, verbose=0)
    y_pred = np.asarray(y_pred).reshape(-1)  # ensure 1D

    # Pick the best-matching target (or forced)
    forced = FORCE_TARGET if FORCE_TARGET else None
    matched_col, y_true, mse, corr = pick_target_for_model(eval_df, y_pred, forced)

    # Your existing metric definition: "accuracy in 15%" (relative)
    denom = np.maximum(np.abs(y_true), 1e-8)  # avoid divide-by-zero when actual is 0
    acc15pct = float(np.mean(np.abs(y_pred - y_true) <= 0.15 * denom) * 100.0)

    # Optional transparency in console
    print(f"{os.path.basename(model_path)} matched_target='{matched_col}', corr={corr:.3f}, MSE={mse:.2f}, acc15%={acc15pct:.2f}%")

    return mse, acc15pct, y_pred, matched_col

# ---------- Load schema (training feature order) ----------
schema_cols = read_schema_cols(SCHEMA_TXT)
if not schema_cols:
    raise FileNotFoundError(
        f"Couldn't read feature schema from {SCHEMA_TXT}. "
        f"Please keep the training columns list there (with target last)."
    )
expected_features = schema_cols  # 31 features expected (target is NOT included)

# ---------- Load evaluation CSV & normalize ----------
if not EVAL_CSV.exists():
    cands = sorted(DATA.glob("Library_Data_*.csv"))
    if not cands:
        raise FileNotFoundError(f"No eval CSV at {EVAL_CSV} and no Library_Data_*.csv under {DATA}")
    EVAL_CSV = cands[0]

eval_df = pd.read_csv(EVAL_CSV)
eval_df = normalize_columns(eval_df)
eval_df = ensure_time_columns(eval_df)

if TARGET_COL not in eval_df.columns:
    raise ValueError(
        f"Eval CSV '{EVAL_CSV.name}' is missing target '{TARGET_COL}'. "
        f"Pick a Library_Data_*.csv that includes it."
    )

# Try to enrich missing features from Model_Input_Data.csv (merge on Date / Time)
eval_df = enrich_from_model_input(eval_df, expected_features)

# After enrich, verify we truly have the full training feature set
still_missing = [c for c in expected_features if c not in eval_df.columns]
if still_missing:
    raise ValueError(
        "Your evaluation CSV is missing required training features.\n"
        f"Missing ({len(still_missing)}): {still_missing}"
    )

# ---------- Build scaler source ----------
if TRAIN_DS.exists():
    scaler_source = pd.read_csv(TRAIN_DS)
    scaler_source = normalize_columns(scaler_source)
elif MODEL_INPUT.exists():
    scaler_source = pd.read_csv(MODEL_INPUT)
    scaler_source = normalize_columns(scaler_source)
else:
    scaler_source = eval_df  # last resort

# Keep only expected features (correct order)
missing_in_scaler = [c for c in expected_features if c not in scaler_source.columns]
if missing_in_scaler:
    # If we fell back to TRAIN_DS but it misses a few, try merging from Model_Input on Date / Time
    if scaler_source is not eval_df and MODEL_INPUT.exists() and "Date / Time" in scaler_source.columns:
        mi = normalize_columns(pd.read_csv(MODEL_INPUT))
        if "Date / Time" in mi.columns:
            cols_to_add = ["Date / Time"] + [c for c in missing_in_scaler if c in mi.columns]
            scaler_source = scaler_source.merge(mi[cols_to_add], on="Date / Time", how="left")
            missing_in_scaler = [c for c in expected_features if c not in scaler_source.columns]
if missing_in_scaler:
    raise ValueError(
        "Scaler source is missing required training features.\n"
        f"Missing ({len(missing_in_scaler)}): {missing_in_scaler}\n"
        "Ensure TrainingDataset.csv or Model_Input_Data.csv contains all training features."
    )

# Fit scaler
X_train_feats = scaler_source[expected_features].copy()
scaler = StandardScaler().fit(X_train_feats)

# ---------- Prepare eval arrays in exact training order ----------
X_new = eval_df[expected_features].copy()
y_new = eval_df[TARGET_COL].to_numpy(dtype=float)
X_new_scaled = scaler.transform(X_new)
X_eval, y_eval = build_sequences(X_new_scaled, y_new, SEQUENCE_LENGTH)

print(f"Eval file: {EVAL_CSV.name}")
print(f"Expected features: {len(expected_features)}")
print(f"Eval shapes -> X: {X_eval.shape}, y: {y_eval.shape}")

# ---------- Find and inventory models ----------
model_files = find_all_model_files()
print("Scoring all models from these folders:")
print(f"  - {MODELS}")
for d in EXTRA_MODEL_DIRS:
    print(f"  - {d}")
inventory(model_files)

# ---------- Score every model ----------
rows = []
best = {"mse": float("inf"), "acc15": -1, "model": None, "df": None}

for model_path in model_files:
    try:
        mse, acc15, y_pred, matched_col = score_one_model(model_path, X_eval, eval_df)

        df = pd.DataFrame({"prediction": y_pred, "actual": y_eval[:len(y_pred)]})
        rows.append({
            "model": model_path.name,
            "mse": mse,
            "acc15pct": acc15,
            "matched_target": matched_col
        })
        print(f"{model_path.name}: MSE={mse:.2f}, acc15={acc15:.2f}%")

        if (mse < best["mse"]) or (np.isclose(mse, best["mse"]) and acc15 > best["acc15"]):
            best = {"mse": mse, "acc15": acc15, "model": model_path.name, "df": df}
    except Exception as e:
        print(f"ERROR scoring {model_path.name}: {e}")
        # For quick debugging, uncomment next line:
        # traceback.print_exc(limit=1)

if not rows:
    raise RuntimeError(f"No models were scored. Ensure there are .keras files in '{MODELS}' or extra dirs.")

leaderboard = pd.DataFrame(rows).sort_values(["mse", "acc15"], ascending=[True, False])
lb_path = RESULTS / "model_selection.csv"
leaderboard.to_csv(lb_path, index=False)
print(f"\nSaved leaderboard => {lb_path}")

tag = Path(best["model"]).stem
preds_path = RESULTS / f"predictions_{tag}.csv"
best["df"].to_csv(preds_path, index=False)
plot_path = difference_plot(best["df"], tag, best["acc15"], best["mse"])

print(f"Best model: {best['model']} | MSE={best['mse']:.2f}, acc15={best['acc15']:.2f}%")
print(f"Saved best predictions => {preds_path}")
print(f"Saved error plot => {plot_path}")
