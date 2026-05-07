"""
run_daily.py — Daily trading bot pipeline

Steps
─────
  1. data/market_data.py    refresh OHLCV for 15 watchlist tickers
  2. signals/scorer.py      score and rank every stock by Buy Signal
  3. signals/screener.py    scan 101 small/mid-cap tickers for setups

Buy Signal formula: (technical_score × 0.4) + (ml_confidence × 0.6)

Manual run:
    python3 run_daily.py

Automated via launchd — see com.tradingbot.daily.plist in this directory.
Each run appends to  logs/run_YYYY-MM-DD.log
"""

import os
import sys
import subprocess
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON       = sys.executable   # same interpreter running this script

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LOG_DIR  = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, f"run_{datetime.now().strftime('%Y-%m-%d')}.log")

W = 66


# ── Tee stdout → terminal + date-stamped log file ────────────────────────────

class _Tee:
    """Mirrors every print() to both the original stdout and a log file."""
    def __init__(self, path):
        self._file = open(path, "a", buffering=1, encoding="utf-8")

    def write(self, text):
        if sys.__stdout__ is not None:
            sys.__stdout__.write(text)
        self._file.write(text)

    def flush(self):
        if sys.__stdout__ is not None:
            sys.__stdout__.flush()
        self._file.flush()

    def __getattr__(self, attr):
        # Forward anything else (encoding, fileno, etc.) to the real stdout
        return getattr(sys.__stdout__, attr)


sys.stdout = _Tee(LOG_PATH)


# ── Step 1 — Refresh market data ─────────────────────────────────────────────

def step1_market_data() -> bool:
    print("\n[1/3]  Refreshing market data…")
    t0 = time.time()

    result = subprocess.run(
        [PYTHON, os.path.join(PROJECT_ROOT, "data", "market_data.py")],
        cwd=os.path.join(PROJECT_ROOT, "data"),   # needed for its relative import
        capture_output=True,
        text=True,
    )

    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  ✗  FAILED (exit {result.returncode})")
        for line in (result.stderr or result.stdout).strip().splitlines():
            print(f"     {line}")
        return False

    # Count total new rows from "Saved N new rows" lines
    new_rows = 0
    for line in result.stdout.splitlines():
        parts = line.split()
        if "Saved" in parts:
            idx = parts.index("Saved")
            if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                new_rows += int(parts[idx + 1])

    print(f"  ✓  Done in {elapsed:.1f}s — {new_rows} new price row(s) added")
    return True


# ── Step 2 — Score watchlist ──────────────────────────────────────────────────

def step2_scorer() -> list:
    print("\n[2/3]  Scoring watchlist  (Buy Signal = Score × 0.4 + ML Conf × 0.6)…\n")

    import warnings, logging
    warnings.filterwarnings("ignore")
    for lg in ("yfinance", "yfinance.base", "urllib3", "sklearn"):
        logging.getLogger(lg).setLevel(logging.CRITICAL)

    from signals.indicators import get_all_indicators
    from signals.scorer import score_stock, _ml_confidence, _ML_MODEL

    all_data = get_all_indicators()
    rows = []
    for ticker, df in all_data.items():
        score, _  = score_stock(df)
        conf      = _ml_confidence(df) if _ML_MODEL is not None else None
        latest    = df.iloc[-1]
        score_i   = int(score)
        conf_pct  = round(conf * 100, 1) if conf is not None else None
        buy_sig   = (
            round(score_i * 0.4 + conf_pct * 0.6, 1)
            if conf_pct is not None else float(score_i)
        )
        rows.append({
            "ticker":     ticker,
            "buy_signal": buy_sig,
            "score":      score_i,
            "ml_conf":    conf_pct,
            "close":      float(latest["close"]),
            "rsi":        float(latest["rsi"]),
        })

    rows.sort(key=lambda r: -r["buy_signal"])

    col = "  {:<6}  {:>10}  {:>8}  {:>8}  {:>9}  {:>5}"
    sep = "  " + "─" * (W - 3)
    print(col.format("Ticker", "Buy Signal", "Score", "ML Conf", "Close", "RSI"))
    print(sep)
    for r in rows:
        ml_str = f"{r['ml_conf']:.1f}%" if r["ml_conf"] is not None else "—"
        print(col.format(
            r["ticker"],
            f"{r['buy_signal']:.1f}",
            f"{r['score']}/100",
            ml_str,
            f"${r['close']:.2f}",
            f"{r['rsi']:.1f}",
        ))
    print(sep)

    return rows


# ── Step 3 — Momentum screener ────────────────────────────────────────────────

def step3_screener():
    print("\n[3/3]  Running momentum screener across 101 small/mid-cap tickers…")

    import warnings, logging
    warnings.filterwarnings("ignore")
    for lg in ("yfinance", "yfinance.base", "urllib3"):
        logging.getLogger(lg).setLevel(logging.CRITICAL)

    from signals.screener import download_universe_data, screen_stocks

    data    = download_universe_data()
    results = screen_stocks(data)

    if results.empty:
        print("  → 0 candidates passed all four filters today.")
    else:
        print(f"  → {len(results)} candidate(s) found:\n")
        col = "  {:<4} {:<8} {:>7} {:>6} {:>9}  {:>14}"
        sep = "  " + "─" * 52
        print(col.format("#", "Ticker", "Price", "RSI", "5d Chg", "Avg Vol (20d)"))
        print(sep)
        for idx, row in results.iterrows():
            print(col.format(
                idx,
                row["Ticker"],
                f"${row['Price']:.2f}",
                f"{row['RSI']:.1f}",
                f"{row['5d Change %']:+.2f}%",
                f"{row['Avg Vol (20d)']:,.0f}",
            ))

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    now     = datetime.now()

    print("=" * W)
    print(f"  DAILY TRADING BRIEF — {now.strftime('%A, %B %d %Y')}")
    print(f"  Started: {now.strftime('%H:%M:%S')}  |  Log → {os.path.relpath(LOG_PATH)}")
    print("=" * W)

    if not step1_market_data():
        print("\n  Aborting — market data refresh failed.")
        sys.exit(1)

    scored_rows = step2_scorer()
    screener_df = step3_screener()
    elapsed     = time.time() - t_start

    # ── Summary ───────────────────────────────────────────────────────────────
    attention  = [r for r in scored_rows if r["buy_signal"] >= 40]
    screen_hit = not screener_df.empty

    print("\n" + "=" * W)
    print("  ★  SUMMARY")
    print("─" * W)

    if attention:
        print(f"  Watchlist — {len(attention)} stock(s) with Buy Signal ≥ 40:")
        for r in attention:
            ml_str = f"  ML {r['ml_conf']:.0f}%" if r["ml_conf"] is not None else ""
            print(
                f"    →  {r['ticker']:<5}  Buy Signal {r['buy_signal']:.1f}"
                f"  |  Score {r['score']}/100{ml_str}"
            )
    else:
        print("  Watchlist — no stocks reached Buy Signal ≥ 40 today.")

    if screen_hit:
        tickers = "  ".join(screener_df["Ticker"].tolist())
        print(f"\n  Screener — {len(screener_df)} small/mid-cap candidate(s):  {tickers}")
    else:
        print("\n  Screener — no momentum candidates today.")

    print(f"\n  Completed in {elapsed:.1f}s")
    print("=" * W)


if __name__ == "__main__":
    main()
