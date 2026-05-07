"""
signals/indicators.py

Calculates technical indicators for every stock in the watchlist using
historical price data stored in data/trading_bot.db.

Run standalone:
    python3 signals/indicators.py
Import from other modules:
    from signals.indicators import get_all_indicators
"""

import os
import sys
import sqlite3

import pandas as pd
import ta

# Allow running from any directory by ensuring the project root is on the path.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.watchlist import WATCHLIST

DB_PATH = os.path.join(_PROJECT_ROOT, "data", "trading_bot.db")


# ── data loading ─────────────────────────────────────────────────────────────

def load_price_data(ticker: str, conn: sqlite3.Connection) -> pd.DataFrame:
    """Return a DataFrame of daily OHLCV data for *ticker*, sorted oldest-first."""
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume "
        "FROM daily_prices WHERE ticker = ? ORDER BY date ASC",
        conn,
        params=(ticker,),
    )
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


# ── indicator calculation ─────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicator columns to *df* (in place) and return it.

    New columns added:
        sma_50      — 50-day Simple Moving Average
        sma_200     — 200-day Simple Moving Average
        rsi         — RSI with 14 periods
        macd_line   — MACD line (12-day EMA minus 26-day EMA)
        macd_signal — Signal line (9-day EMA of the MACD line)
        macd_hist   — Histogram (MACD line minus signal line)
        vol_avg_20  — Rolling 20-day average volume
        vol_ratio   — Today's volume divided by the 20-day average
    """
    close = df["close"]
    volume = df["volume"]

    # 50-day Simple Moving Average ─────────────────────────────────────────────
    # The average closing price over the last 50 trading days (~2.5 months).
    # When price is above this line the stock is in a short-to-medium-term uptrend.
    # Traders often use it as a first line of support during pullbacks.
    df["sma_50"] = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()

    # 200-day Simple Moving Average ────────────────────────────────────────────
    # The average closing price over the last 200 trading days (~10 months).
    # This is the most widely-watched long-term trend line in the market.
    # Price above = bull-market territory; price below = bear-market territory.
    df["sma_200"] = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()

    # RSI — Relative Strength Index (14 periods) ───────────────────────────────
    # Measures price momentum on a 0–100 scale.
    # Below 30: oversold — the stock has been hit hard and may be due for a bounce.
    # Above 70: overbought — the rally may be running out of steam.
    # 30–50: the sweet spot for a buying opportunity — selling pressure is easing
    #        but the stock hasn't yet been "discovered" by momentum chasers.
    df["rsi"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    # MACD — Moving Average Convergence Divergence ─────────────────────────────
    # Standard settings: fast EMA=12 days, slow EMA=26 days, signal EMA=9 days.
    # The MACD line is the fast EMA minus the slow EMA — it measures the gap
    # between short-term and long-term momentum.
    # The Signal line is a 9-day EMA of the MACD line — it smooths the signal.
    # A MACD line crossover above the Signal line is a classic buy signal:
    # it means short-term momentum just flipped from lagging behind to leading.
    # The Histogram shows the gap between MACD and Signal — positive = bullish,
    # growing histogram = strengthening momentum.
    _macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["macd_line"]   = _macd.macd()          # The main MACD line
    df["macd_signal"] = _macd.macd_signal()   # The slower signal line
    df["macd_hist"]   = _macd.macd_diff()     # Histogram = MACD minus signal

    # Volume vs. 20-day average ────────────────────────────────────────────────
    # The average number of shares traded per day over the last ~1 month.
    # When today's volume is well above this baseline it signals real conviction:
    # buyers (or sellers) are showing up in force, not just drifting aimlessly.
    df["vol_avg_20"] = volume.rolling(window=20).mean()

    # Volume ratio: today's volume ÷ 20-day average volume.
    # 1.20 means 20 % more trading activity than normal — noteworthy.
    # 2.00 means twice the normal volume — a very high-conviction move.
    df["vol_ratio"] = volume / df["vol_avg_20"]

    return df


# ── public API ────────────────────────────────────────────────────────────────

def get_all_indicators(conn: sqlite3.Connection = None) -> dict:
    """
    Calculate indicators for every ticker in WATCHLIST.

    Returns
    -------
    dict mapping ticker (str) → full DataFrame with indicator columns.
    Tickers with fewer than 200 rows are skipped (not enough history for SMA-200).
    """
    should_close = conn is None
    if should_close:
        conn = sqlite3.connect(DB_PATH)

    results = {}
    for ticker in WATCHLIST:
        df = load_price_data(ticker, conn)
        if len(df) < 200:
            print(f"  [SKIP] {ticker}: only {len(df)} rows — need 200+ for SMA-200")
            continue
        df = add_indicators(df)
        results[ticker] = df

    if should_close:
        conn.close()

    return results


# ── standalone output ─────────────────────────────────────────────────────────

def _print_separator(width=52):
    print("─" * width)


if __name__ == "__main__":
    print("Calculating technical indicators for all watchlist stocks…\n")
    all_data = get_all_indicators()

    for ticker, df in all_data.items():
        latest = df.iloc[-1]
        prev   = df.iloc[-2]

        # Detect a MACD crossover: MACD was below signal yesterday, at-or-above today.
        macd_crossed = (
            prev["macd_line"] < prev["macd_signal"]
            and latest["macd_line"] >= latest["macd_signal"]
        )

        _print_separator()
        print(f"  {ticker}  |  {df.index[-1].strftime('%Y-%m-%d')}")
        _print_separator()
        print(f"  Close        ${latest['close']:.2f}")
        print(f"  SMA 50       ${latest['sma_50']:.2f}"
              f"  (price {'above' if latest['close'] > latest['sma_50'] else 'below'} SMA)")
        print(f"  SMA 200      ${latest['sma_200']:.2f}"
              f"  (SMA50 {'above' if latest['sma_50'] > latest['sma_200'] else 'below'} SMA200)")
        print(f"  RSI (14)     {latest['rsi']:.1f}")
        print(f"  MACD line    {latest['macd_line']:.4f}"
              f"  |  Signal  {latest['macd_signal']:.4f}"
              f"  |  Hist  {latest['macd_hist']:.4f}")
        print(f"  MACD cross   {'YES — just crossed above signal line' if macd_crossed else 'No'}")
        print(f"  Volume       {int(latest['volume']):>12,}"
              f"  ({latest['vol_ratio']:.2f}x 20-day avg)")
        print()
