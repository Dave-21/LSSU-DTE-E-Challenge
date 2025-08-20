# -*- coding: utf-8 -*-
from pathlib import Path
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
MODELS  = ROOT / "models"
RESULTS = ROOT / "results" / "ModelTester"
PLOTS   = RESULTS / "plots"
for p in (RESULTS, PLOTS):
    p.mkdir(parents=True, exist_ok=True)

# ---------- Config ----------
SEQUENCE_LENGTH = 10
TARGET_COL = "Library - Tomorrow Max Energy"

# Choose your hold-out CSV here (must include TARGET_COL)
EVAL_CSV   = DATA / "Library_Data_June24-Jul24.csv"   # change if you prefer
TRAIN_DS   = DATA / "TrainingDataset.csv"             # preferred scaler source (if present)
MODEL_INPUT= DATA / "Model_Input_Data.csv"            # fallback for fitting scaler
SCHEMA_TXT = DATA / "columns_in_use.txt"              # definitive feature schema (order)

# ---------- Helpers ----------
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
scaler_source = None
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
    # If we fell back to eval_df, they won't be missing; otherwise try merging from Model_Input
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

# Drop non-feature columns and fit scaler
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
print("Scoring all models in:", MODELS)

# ---------- Score every .keras model ----------
rows = []
best = {"mse": float("inf"), "acc15": -1, "model": None, "df": None}

for model_path in sorted(MODELS.glob("*.keras")):
    try:
        model = tf.keras.models.load_model(model_path)
        loss = model.evaluate(X_eval, y_eval, verbose=0)
        mse = float(loss[0]) if isinstance(loss, (list, tuple, np.ndarray)) else float(loss)
        preds = model.predict(X_eval, verbose=0).reshape(-1)
        df = pd.DataFrame({"prediction": preds, "actual": y_eval[:len(preds)]})
        acc15 = acc15_from_df(df)
        rows.append({"model": model_path.name, "mse": mse, "acc15": acc15})
        print(f"{model_path.name}: MSE={mse:.2f}, acc15={acc15:.2f}%")
        if (mse < best["mse"]) or (np.isclose(mse, best["mse"]) and acc15 > best["acc15"]):
            best = {"mse": mse, "acc15": acc15, "model": model_path.name, "df": df}
    except Exception as e:
        print(f"ERROR scoring {model_path.name}: {e}")

if not rows:
    raise RuntimeError(f"No models were scored. Ensure there are .keras files in '{MODELS}'.")

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
