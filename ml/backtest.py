"""
ml/backtest.py

Simulates the combined signal-engine + ML strategy over the past year.

For each watchlist stock, 3 years of daily data are downloaded so that all
technical indicators have a full warm-up window.  Only the most recent
252 trading days (~1 year) are used as the backtest period, ensuring every
indicator is fully valid on every simulated trade.

Entry rules:
  • Signal score  ≥ 70  (rule-based scorer from signals/scorer.py)
  • ML confidence ≥ 60% (Random Forest probability from ml/model.pkl)

Exit:  close position 10 trading days after entry (at that day's close price).
No slippage or commissions are modelled — results are optimistic upper bounds.

Run:
    python3 ml/backtest.py
"""

import logging
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ResourceWarning)
for _log in ("yfinance", "yfinance.base", "urllib3", "sklearn"):
    logging.getLogger(_log).setLevel(logging.CRITICAL)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.watchlist import WATCHLIST
from signals.indicators import add_indicators
from signals.scorer import score_stock
from ml.feature_builder import FEATURE_COLS, compute_features

ML_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ML_DIR, "model.pkl")

BACKTEST_DAYS  = 252   # ~1 trading year
MIN_SCORE      = 70
MIN_CONFIDENCE = 0.60


# ── model loading ─────────────────────────────────────────────────────────────

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}.\n"
            "Run first:  python3 ml/train_model.py"
        )
    payload = joblib.load(MODEL_PATH)
    clf = payload["model"]
    # Force single-threaded inference so parallel-worker warnings don't flood
    # the output when predict_proba is called in a tight loop.
    clf.n_jobs = 1
    return clf


def ml_confidence(model, row: pd.Series) -> float:
    """Return P(label=1) for a single feature row.  Returns 0.0 if any NaN."""
    vals = [row.get(f, np.nan) for f in FEATURE_COLS]
    if any(np.isnan(v) for v in vals):
        return 0.0
    return float(model.predict_proba(np.array(vals).reshape(1, -1))[0][1])


# ── data download ─────────────────────────────────────────────────────────────

def download_ticker(ticker: str) -> pd.DataFrame | None:
    """Download 3y of OHLCV for *ticker*, lowercase columns, sorted by date."""
    try:
        df = yf.download(ticker, period="3y", auto_adjust=True, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df
    except Exception:
        return None


# ── backtest engine ───────────────────────────────────────────────────────────

def run_backtest(model) -> list:
    trades = []

    for ticker in WATCHLIST:
        df_raw = download_ticker(ticker)
        if df_raw is None or len(df_raw) < 220:
            print(f"    [SKIP] {ticker}: not enough data")
            continue

        # Compute all indicators on the full 3y so the last year is fully warmed up.
        df = add_indicators(df_raw)
        df = compute_features(df)
        df.dropna(subset=FEATURE_COLS, inplace=True)

        # Restrict the backtested period to the most recent BACKTEST_DAYS rows.
        if len(df) > BACKTEST_DAYS + 10:
            df = df.iloc[-(BACKTEST_DAYS + 10):]   # extra 10 so last entries have exits

        n = len(df)
        ticker_trades = 0

        # i starts at 1 so score_stock always has a previous row for MACD cross.
        for i in range(1, n - 10):
            row_df = df.iloc[: i + 1]
            score, _ = score_stock(row_df)
            if score < MIN_SCORE:
                continue

            conf = ml_confidence(model, df.iloc[i])
            if conf < MIN_CONFIDENCE:
                continue

            entry_date  = df.index[i]
            exit_date   = df.index[i + 10]
            entry_price = float(df["close"].iloc[i])
            exit_price  = float(df["close"].iloc[i + 10])
            ret = (exit_price - entry_price) / entry_price

            trades.append({
                "ticker":     ticker,
                "entry_date": entry_date,
                "exit_date":  exit_date,
                "entry":      entry_price,
                "exit":       exit_price,
                "return":     ret,
                "score":      score,
                "confidence": conf,
            })
            ticker_trades += 1

        print(f"    {ticker}: {ticker_trades} trade(s) triggered")

    return trades


# ── performance metrics ───────────────────────────────────────────────────────

def max_drawdown(returns: list) -> float:
    """Max peak-to-trough decline on a compounded equal-weight equity curve."""
    if not returns:
        return 0.0
    equity  = np.cumprod([1 + r for r in returns])
    peak    = np.maximum.accumulate(equity)
    dd      = (equity - peak) / peak
    return float(dd.min())


# ── report ────────────────────────────────────────────────────────────────────

W = 62

def print_report(trades: list):
    print("\n" + "=" * W)
    print("  BACKTEST REPORT — Rule-Based Signal + ML Strategy")
    print(f"  Entry:  score ≥ {MIN_SCORE}  AND  ML confidence ≥ {MIN_CONFIDENCE:.0%}")
    print("  Exit:   hold 10 trading days, sell at close")
    print("  Period: most recent 252 trading days (~1 year)")
    print("=" * W)

    if not trades:
        print("\n  No trades triggered.  The combined filters (score ≥ 70 and\n"
              "  ML confidence ≥ 60%) are intentionally strict.  Try running\n"
              "  with min_score=60 to see more activity.\n")
        print("=" * W)
        return

    rets   = [t["return"] for t in trades]
    wins   = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    n      = len(rets)

    total_ret  = float(np.prod([1 + r for r in rets]) - 1)
    win_rate   = len(wins) / n
    avg_gain   = float(np.mean(wins))  if wins   else 0.0
    avg_loss   = float(np.mean(losses)) if losses else 0.0
    mdd        = max_drawdown(rets)
    avg_conf   = float(np.mean([t["confidence"] for t in trades]))
    avg_score  = float(np.mean([t["score"] for t in trades]))

    print(f"\n  Total trades         {n}")
    print(f"  Win rate             {win_rate:.1%}  ({len(wins)}W / {len(losses)}L)")
    print(f"  Total return         {total_ret:+.2%}  (compounded, equal-weight)")
    print(f"  Average gain         {avg_gain:+.2%}")
    print(f"  Average loss         {avg_loss:+.2%}")
    print(f"  Max drawdown         {mdd:.2%}")
    print(f"  Avg ML confidence    {avg_conf:.1%}")
    print(f"  Avg signal score     {avg_score:.0f}/100")

    print(f"\n  {'Ticker':<6}  {'Entry':>10}  {'Exit':>10}  {'Score':>5}  "
          f"{'Conf':>5}  {'Return':>8}")
    print("  " + "─" * (W - 2))
    for t in sorted(trades, key=lambda x: x["entry_date"]):
        arrow = "▲" if t["return"] > 0 else "▼"
        print(
            f"  {t['ticker']:<6}  "
            f"{t['entry_date'].strftime('%Y-%m-%d'):>10}  "
            f"{t['exit_date'].strftime('%Y-%m-%d'):>10}  "
            f"{t['score']:>5}  "
            f"{t['confidence']:>4.0%}  "
            f"{arrow} {t['return']:>+6.2%}"
        )

    print("\n" + "=" * W)


if __name__ == "__main__":
    print("Loading model...")
    model = load_model()
    print(f"Downloading 3y data for {len(WATCHLIST)} tickers (for indicator warm-up)...\n")
    trades = run_backtest(model)
    print_report(trades)
