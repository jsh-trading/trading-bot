"""
ml/train_model.py

Loads ml/training_data.csv, trains a Random Forest classifier to predict
whether a stock will gain more than 5% within the next 10 trading days,
and saves the trained model to ml/model.pkl.

Split strategy: temporal (date-ordered) 80 / 20.  Training data are the
earliest 80% of dates; test data are the most recent 20%.  This prevents
future data from leaking into training — the correct approach for any
time-series financial model.

Run:
    python3 ml/train_model.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

ML_DIR     = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(ML_DIR, "training_data.csv")
MODEL_PATH = os.path.join(ML_DIR, "model.pkl")

# Import canonical feature list from feature_builder to guarantee consistency.
from ml.feature_builder import FEATURE_COLS


# ── data loading and splitting ────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Training data not found at {CSV_PATH}.\n"
            "Run first:  python3 ml/feature_builder.py"
        )
    df = pd.read_csv(CSV_PATH)
    # Normalise the date column name (feature_builder may write 'date' or 'Date').
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    if date_col:
        df.rename(columns={date_col: "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.dropna(subset=FEATURE_COLS + ["label"], inplace=True)
    df["label"] = df["label"].astype(int)
    return df


def temporal_split(df: pd.DataFrame, train_frac: float = 0.80):
    """
    Split by date rather than randomly.  The earliest train_frac of unique
    dates go to training; the rest go to the test set.

    Random splitting on time-series data would let the model 'see the future'
    during training — a common mistake that inflates reported accuracy.
    """
    sorted_dates = np.sort(df["date"].unique())
    cutoff = sorted_dates[int(len(sorted_dates) * train_frac)]
    return df[df["date"] < cutoff], df[df["date"] >= cutoff]


# ── training ──────────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  MODEL TRAINER — Random Forest Classifier")
    print("=" * 60)

    data = load_data()
    print(f"\n  Loaded {len(data):,} rows from training_data.csv")
    print(f"  Features: {len(FEATURE_COLS)}  ({', '.join(FEATURE_COLS)})")

    train_df, test_df = temporal_split(data)
    cutoff = test_df["date"].min().strftime("%Y-%m-%d")
    print(f"\n  Train: {len(train_df):,} rows  (before {cutoff})")
    print(f"  Test:  {len(test_df):,} rows  ({cutoff} onwards)")

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["label"].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df["label"].values

    pos_pct = 100 * y_train.mean()
    print(f"\n  Training class balance: {pos_pct:.1f}% positive  "
          f"({y_train.sum():,} buys / {len(y_train):,} total days)")

    # ── model ────────────────────────────────────────────────────────────────
    # 300 trees keeps variance low without being slow.  max_depth=6 prevents
    # overfitting on the ~8k training rows.  class_weight='balanced' corrects
    # for the typical 30/70 positive/negative imbalance automatically.
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    print("\n  Training... ", end="", flush=True)
    clf.fit(X_train, y_train)
    print("done.\n")

    # ── evaluation ───────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    # Precision = win rate: of all days the model flags as "buy", how many
    # actually ended up gaining 5%+ in the next 10 days.
    win_rate = precision_score(y_test, y_pred, zero_division=0)
    recall   = recall_score(y_test, y_pred, zero_division=0)

    print(f"  ── Test Results (most recent 20% of dates) ─────────────────")
    print(f"  Overall accuracy        {accuracy:.1%}")
    print(f"  Win rate (precision)    {win_rate:.1%}"
          "  ← when model says 'buy', correct this % of the time")
    print(f"  Recall                  {recall:.1%}"
          "  ← fraction of real opportunities the model caught")

    hc = y_prob >= 0.60
    if hc.sum() > 0:
        hc_acc = accuracy_score(y_test[hc], (y_prob[hc] >= 0.5).astype(int))
        print(f"  High-conf accuracy      {hc_acc:.1%}"
              f"  ({hc.sum()} signals at ≥60% confidence)")
    else:
        print("  High-conf accuracy      n/a  (no signals reached ≥60% confidence)")

    # ── feature importances ───────────────────────────────────────────────────
    ranked = sorted(
        zip(FEATURE_COLS, clf.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    print(f"\n  ── Feature Importance (most → least predictive) ────────────")
    for feat, imp in ranked:
        bar = "█" * round(imp * 80)
        print(f"  {feat:<22}  {imp:.3f}  {bar}")

    # ── save ─────────────────────────────────────────────────────────────────
    joblib.dump({"model": clf, "features": FEATURE_COLS}, MODEL_PATH)
    print(f"\n  Model saved → {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train()
