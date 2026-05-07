"""
ml/feature_builder.py

Downloads 3 years of daily OHLCV data for every watchlist stock via yfinance,
computes the same technical indicators used by the signal engine, derives
ten normalised ML features, then labels each trading day:

  label = 1  if the stock closed more than 5% above today's close at any
              point in the next 10 trading days
  label = 0  otherwise

Saves the result to ml/training_data.csv.

Run:
    python3 ml/feature_builder.py
"""

import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
for _log in ("yfinance", "yfinance.base", "urllib3"):
    logging.getLogger(_log).setLevel(logging.CRITICAL)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.watchlist import WATCHLIST
from signals.indicators import add_indicators

ML_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(ML_DIR, "training_data.csv")

LABEL_HORIZON   = 10    # look-forward days for the label
LABEL_THRESHOLD = 0.05  # 5% gain threshold

# ── canonical feature list ─────────────────────────────────────────────────
# All features are dimensionless ratios so the model generalises across
# stocks with different price levels.  Must stay in sync with train_model.py,
# backtest.py, and scorer.py.

FEATURE_COLS = [
    "price_to_sma50",   # close / sma_50       — how extended vs short-term trend
    "price_to_sma200",  # close / sma_200       — position vs long-term trend
    "sma50_to_sma200",  # sma_50 / sma_200      — golden/death-cross ratio (>1 = golden)
    "rsi",              # RSI(14)               — 0–100 momentum oscillator
    "macd_hist",        # MACD histogram        — momentum direction and strength
    "macd_to_price",    # macd_line / close     — normalised MACD scale
    "vol_ratio",        # volume / vol_avg_20   — activity vs 20-day baseline
    "daily_return",     # pct change from yesterday's close
    "high_to_close",    # high / close          — intraday bullish range
    "low_to_close",     # low  / close          — intraday dip depth
]


# ── data helpers ──────────────────────────────────────────────────────────────

def _download(ticker: str, period: str = "3y") -> pd.DataFrame | None:
    """Download OHLCV for *ticker*.  Returns None on failure or empty data."""
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty:
            return None
        # Newer yfinance may wrap columns in a MultiIndex — flatten it.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df
    except Exception as exc:
        print(f"    [WARN] {ticker}: {exc}")
        return None


# ── feature engineering ───────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ML feature columns to *df* (which must already have indicator columns
    from add_indicators).  Returns a new DataFrame; does not modify in place.
    """
    df = df.copy()

    # Ratios normalise absolute price so a $500 stock and a $10 stock are
    # directly comparable in the model.
    df["price_to_sma50"]  = df["close"] / df["sma_50"]
    df["price_to_sma200"] = df["close"] / df["sma_200"]
    df["sma50_to_sma200"] = df["sma_50"] / df["sma_200"]

    # MACD line divided by price removes the dollar-scale bias.
    df["macd_to_price"] = df["macd_line"] / df["close"]

    # Percentage change from the previous close — the day's "news".
    df["daily_return"] = df["close"].pct_change()

    # How far intraday extremes were from the close.  high > close means the
    # stock tried to go higher but gave back gains; low < close means buyers
    # stepped in to push it back up.
    df["high_to_close"] = df["high"] / df["close"]
    df["low_to_close"]  = df["low"]  / df["close"]

    return df


def compute_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'label' column.  For each row, look at the next LABEL_HORIZON closes
    and check whether any exceed today's close by LABEL_THRESHOLD.

    The last LABEL_HORIZON rows get NaN labels (no future data) and are
    dropped before saving.
    """
    df = df.copy()
    # Stack the next N closes as columns, take the row-wise max.
    future_max = pd.concat(
        [df["close"].shift(-i) for i in range(1, LABEL_HORIZON + 1)], axis=1
    ).max(axis=1)
    df["label"] = ((future_max / df["close"] - 1) > LABEL_THRESHOLD).astype("Int64")
    df.dropna(subset=["label"], inplace=True)
    return df


# ── main build routine ────────────────────────────────────────────────────────

def build_training_data() -> pd.DataFrame:
    frames = []
    for ticker in WATCHLIST:
        print(f"  {ticker}...", end=" ", flush=True)
        df = _download(ticker)
        if df is None or len(df) < 250:
            print("skipped (not enough data)")
            continue

        df = add_indicators(df)   # adds sma_50, sma_200, rsi, macd_*, vol_* columns
        df = compute_features(df) # adds normalised ML feature columns
        df = compute_label(df)    # adds label column; drops last 10 rows

        # Drop warm-up NaNs (indicator columns need history to be valid).
        df.dropna(subset=FEATURE_COLS + ["label"], inplace=True)

        df["ticker"] = ticker
        frames.append(df)
        pos = int((df["label"] == 1).sum())
        print(f"{len(df)} rows  ({pos} positive)")

    if not frames:
        raise RuntimeError("No data downloaded — check your internet connection.")

    combined = pd.concat(frames, ignore_index=False)
    combined.sort_index(inplace=True)
    return combined


if __name__ == "__main__":
    os.makedirs(ML_DIR, exist_ok=True)

    print("=" * 56)
    print("  FEATURE BUILDER — Building ML Training Dataset")
    print("=" * 56)
    print(f"\n  Downloading 3 years of data for {len(WATCHLIST)} tickers...\n")

    data = build_training_data()

    # Save — reset the DatetimeIndex to a plain 'date' column.
    out = data.reset_index().rename(columns={"index": "date", "Date": "date"})
    out.to_csv(CSV_PATH, index=False)

    n_total = len(data)
    n_pos   = int((data["label"] == 1).sum())
    n_neg   = int((data["label"] == 0).sum())
    ratio   = n_pos / n_neg if n_neg else float("inf")

    print(f"\n{'=' * 56}")
    print(f"  Saved → {CSV_PATH}")
    print(f"  Total rows        {n_total:>7,}")
    print(f"  Positive (1)      {n_pos:>7,}  ({100 * n_pos / n_total:.1f}%)")
    print(f"  Negative (0)      {n_neg:>7,}  ({100 * n_neg / n_total:.1f}%)")
    print(f"  Pos / Neg ratio   {ratio:>7.2f}")
    print("=" * 56)
