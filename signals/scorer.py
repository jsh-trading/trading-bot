"""
signals/scorer.py

Scores each watchlist stock 0–100 based on five bullish conditions (20 pts each),
then prints a ranked summary showing which conditions triggered for each stock.

If ml/model.pkl exists, the ML model's confidence (probability that the stock
gains 5%+ in the next 10 days) is shown alongside the rule-based score.

Run standalone:
    python3 signals/scorer.py
Import from other modules:
    from signals.scorer import run_scorer
"""

import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from signals.indicators import get_all_indicators

# ── optional ML integration ───────────────────────────────────────────────────
# Load the model once at import time.  If it doesn't exist yet (before
# ml/train_model.py has been run), the scorer still works — it just omits
# the ML confidence column.

_ML_MODEL    = None
_ML_FEATURES = None

try:
    import warnings as _warnings
    import joblib
    from ml.feature_builder import FEATURE_COLS, compute_features as _compute_features
    _model_path = os.path.join(_PROJECT_ROOT, "ml", "model.pkl")
    if os.path.exists(_model_path):
        _payload     = joblib.load(_model_path)
        _ML_MODEL    = _payload["model"]
        _ML_MODEL.n_jobs = 1   # single-threaded inference avoids per-call parallel warnings
        _ML_FEATURES = _payload["features"]
    _warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    _warnings.filterwarnings("ignore", category=ResourceWarning)
except Exception:
    pass   # ML layer not set up yet — silent fallback


def _ml_confidence(df) -> float | None:
    """
    Return the model's P(gain 5%+ in 10 days) for the latest row in *df*.
    Returns None if the model is not loaded or features contain NaN.
    """
    if _ML_MODEL is None:
        return None
    try:
        feat_df = _compute_features(df)
        row     = feat_df.iloc[-1]
        vals    = [float(row.get(f, np.nan)) for f in _ML_FEATURES]
        if any(np.isnan(v) for v in vals):
            return None
        X = np.array(vals).reshape(1, -1)
        return float(_ML_MODEL.predict_proba(X)[0][1])
    except Exception:
        return None


# ── scoring logic ─────────────────────────────────────────────────────────────

def score_stock(df) -> tuple:
    """
    Evaluate five bullish conditions against the most recent row in *df*.

    Returns
    -------
    (score: int, conditions: dict[str, bool])
        score       — 0, 20, 40, 60, 80, or 100
        conditions  — ordered dict mapping condition label to True/False
    """
    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    conditions = {}

    # Condition 1 — Price above 50-day SMA (20 pts) ───────────────────────────
    # If the stock is trading above its own 50-day average it means buyers have
    # been in control over the past two months. This is the most basic "in an
    # uptrend" check.
    conditions["Price above 50 SMA"] = bool(latest["close"] > latest["sma_50"])

    # Condition 2 — 50 SMA above 200 SMA (20 pts) ────────────────────────────
    # Known as the "golden cross" when it first happens, this means the medium-
    # term trend (50-day) is pointing higher than the long-term trend (200-day).
    # It signals the stock has moved from a bear phase into a sustained uptrend.
    conditions["50 SMA above 200 SMA"] = bool(latest["sma_50"] > latest["sma_200"])

    # Condition 3 — RSI between 30 and 50: oversold but recovering (20 pts) ──
    # An RSI in this band means the stock has been under selling pressure (not
    # yet above the neutral 50 line) but isn't in free-fall (above 30 — no
    # longer deeply oversold). This is the ideal entry zone: weak enough that
    # most sellers have exhausted themselves, but not yet bid up by buyers.
    conditions["RSI 30–50 (recovering from oversold)"] = bool(30 <= latest["rsi"] <= 50)

    # Condition 4 — MACD line just crossed above signal line (20 pts) ─────────
    # "Just crossed" means: yesterday the MACD line was below the signal line
    # (bearish), and today it flipped to at-or-above (bullish). This crossover
    # is one of the most-cited momentum buy signals because it means short-term
    # price acceleration just started outpacing the longer-term trend.
    macd_was_below = prev["macd_line"] < prev["macd_signal"]
    macd_now_above = latest["macd_line"] >= latest["macd_signal"]
    conditions["MACD crossed above signal"] = bool(macd_was_below and macd_now_above)

    # Condition 5 — Volume at least 20 % above its 20-day average (20 pts) ───
    # High volume confirms that price action has real conviction behind it.
    # A move on thin volume can reverse easily; a move on 1.2× normal volume
    # suggests genuine demand (or supply) is driving it.
    conditions["Volume 20%+ above 20-day avg"] = bool(latest["vol_ratio"] >= 1.20)

    score = sum(20 for met in conditions.values() if met)
    return score, conditions


# ── output ────────────────────────────────────────────────────────────────────

_W = 66


def run_scorer():
    ml_active = _ML_MODEL is not None
    print("=" * _W)
    print("  SIGNAL SCORER — Watchlist Rankings")
    header = "  Each condition = 20 pts  |  Max score = 100"
    if ml_active:
        header += "  |  ML confidence shown"
    print(header)
    print("=" * _W)

    all_data = get_all_indicators()
    if not all_data:
        print("\n  No data available. Run:  python3 data/market_data.py\n")
        return

    if not ml_active:
        print("\n  [ML] model not found — run ml/feature_builder.py then"
              " ml/train_model.py to enable ML confidence.\n")

    # Score every stock and optionally get ML confidence
    scored = []
    for ticker, df in all_data.items():
        score, conditions = score_stock(df)
        conf = _ml_confidence(df) if ml_active else None
        scored.append((ticker, score, conditions, df.iloc[-1], conf))

    # Sort: primary = ML confidence (if available), secondary = rule score
    if ml_active:
        scored.sort(key=lambda x: (-(x[4] or 0), -x[1], x[0]))
    else:
        scored.sort(key=lambda x: (-x[1], x[0]))

    for ticker, score, conditions, latest, conf in scored:
        filled = score // 5
        empty  = (100 - score) // 5
        bar    = "█" * filled + "░" * empty

        ml_str = f"  ML: {conf:.0%}" if conf is not None else ""
        print(f"\n  {ticker:<6}  {score:>3}/100  [{bar}]{ml_str}")
        print(f"         Close ${latest['close']:.2f}"
              f"  |  RSI {latest['rsi']:.1f}"
              f"  |  Vol ratio {latest['vol_ratio']:.2f}x")

        for label, met in conditions.items():
            icon   = "✓" if met else "✗"
            status = "YES" if met else "no"
            print(f"    {icon}  {label:<40}  {status}")

    print()
    print("─" * _W)
    top    = scored[0]
    bottom = scored[-1]
    summary = (f"  Scored {len(scored)} stocks  |  "
               f"Top: {top[0]} ({top[1]}/100)")
    if ml_active and top[4] is not None:
        summary += f" ML {top[4]:.0%}"
    summary += f"  |  Bottom: {bottom[0]} ({bottom[1]}/100)"
    print(summary)
    print("=" * _W)


if __name__ == "__main__":
    run_scorer()
