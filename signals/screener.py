"""
signals/screener.py

Scans a broad universe of ~200 popular small- and mid-cap stocks (typically
priced under $75) using live data from yfinance, then filters down to the
stocks most likely to make a bullish move in the near term.

Filters applied
───────────────
  1. Current price < $75
  2. 20-day average daily volume > 1,000,000 shares  (liquid enough to trade)
  3. RSI(14) < 50                                     (not yet overbought)
  4. Price up > 3 % over the last 5 trading days      (recent upward momentum)

Run standalone:
    python3 signals/screener.py
"""

import os
import sys
import logging
import warnings

import pandas as pd
import yfinance as yf
import ta

warnings.filterwarnings("ignore")
# yfinance logs "possibly delisted" / HTTP errors via its own logger — silence them.
for _log in ("yfinance", "yfinance.base", "urllib3", "peewee"):
    logging.getLogger(_log).setLevel(logging.CRITICAL)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ── screening universe ────────────────────────────────────────────────────────
# ~100 well-known small- and mid-cap names that typically trade under $30.
# The screener will filter these on current market data, so any that have
# moved above $30 or been delisted simply won't appear in the results.

SCREENING_UNIVERSE = [
    # Electric Vehicles & Clean Energy
    "NIO",  "XPEV", "LI",   "LCID", "RIVN", "JOBY", "ACHR", "WKHS", "NKLA", "BLDE",
    "GOEV", "PSNY", "SOLO", "AYRO", "EVGO", "CHPT", "BLNK", "SBE",  "ARVL", "MULN",
    # Technology & Fintech
    "SOFI", "HOOD", "BB",   "NOK",  "MVIS", "HIMS", "OPEN", "CLOV", "SKLZ", "FUBO",
    "BARK", "LMND", "PAYO", "BTBT", "MVST", "DAVE", "MQ",   "UPST", "AFRM", "LPRO",
    "STEM", "GREE", "XMTR", "MAPS", "OPAD", "SPIR", "KRTX", "RKLB", "ASTS", "LUNR",
    # AI & Quantum
    "SOUN", "BBAI", "RGTI", "QBTS", "IONQ", "ARQQ", "QUBT", "BKSY", "PRCT", "GFAI",
    "RNLX", "AIXI", "INPX", "BBIO", "SMAR", "CODA", "PERI", "GFGD", "BRSH", "ARCT",
    # Crypto & Blockchain
    "MARA", "RIOT", "CIFR", "HUT",  "BITF", "CLSK", "BTDR", "CORZ", "IREN", "WULF",
    "COIN", "MSTR","BTCS", "BTCM", "DMGI",
    # Mining & Metals
    "VALE", "CLF",  "MT",   "HL",   "PAAS", "AG",   "CDE",  "EXK",  "KGC",  "AUY",
    "MAG",  "FSM",  "SSRM", "GATO", "NGD",  "IAG",  "SBSW", "ABEV", "CENX", "TMST",
    # Energy (oil/gas — small/mid)
    "RIG",  "PBF",  "SWN",  "AR",   "SM",   "CNX",  "BORR", "TELL", "BTU",  "ARCH",
    "DEN",  "NINE", "NOG",  "TALO", "VTLE", "CIVI", "ESTE", "MGY",  "WTTR", "PUMP",
    # Healthcare & Biotech
    "MNKD", "NVAX", "ADMA", "AGEN", "INO",  "OCGN", "VXRT", "CRIS", "SIGA", "ACAD",
    "IMVT", "ARDX", "TLRY", "CGON", "PRAX", "FOLD", "CNTA", "AGIO", "FIXX", "VTYX",
    # Media, Entertainment & Gaming
    "AMC",  "PARA", "WBD",  "GRINDR","GME", "SKLZ", "HUYA", "DOYU", "NERD", "GENI",
    # Retail & Consumer
    "EXPR", "LOVE", "PRTY", "BIG",  "TLYS", "CATO", "VVPR", "WOOF", "PET",  "BARK",
    # Finance & Regional Banks
    "UWMC", "OPFI", "CURO", "SLM",  "KEY",  "RF",   "ZION", "FITB", "HBAN", "CFG",
    "NYCB", "PFSI", "GHLD", "COOP", "PFBC", "BRMK", "PNFP", "FBIZ", "FFBC", "WSFS",
    # Chinese ADRs & International
    "BILI", "VNET", "IQ",   "JMIA", "TIGR", "TUYA", "GRAB", "DIDI", "BEKE", "JKS",
    "XNET", "CIFS", "FINV", "BZUN", "GOTU", "CNF",  "KUKE",
    # Telecom & Legacy Tech
    "T",    "WBA",  "DISH", "LUMN", "SIRI", "AMCX", "NWSA", "NYT",  "VIAC", "CMCSA",
    # Autos & Industrials
    "F",    "IDEX", "HYLN", "UAVS", "XPEV", "RIDE", "KANDI","AYRO", "HYZN", "ZEV",
    # Other notable mid-caps (under $75)
    "SNAP", "UBER", "LYFT", "PLTR", "BAC",  "SPCE", "NKTR", "AMD",  "SOFI", "NOK",
]

# De-duplicate while preserving order
SCREENING_UNIVERSE = list(dict.fromkeys(SCREENING_UNIVERSE))


# ── data download ─────────────────────────────────────────────────────────────

def download_universe_data(period: str = "3mo") -> dict:
    """
    Batch-download historical data for the full screening universe.

    Uses a 3-month window so we have enough history for RSI (needs 14 bars)
    and the 20-day average volume (needs 20 bars), with a comfortable buffer.

    Returns a dict mapping ticker → DataFrame (columns: Open, High, Low, Close, Volume).
    Tickers with fewer than 30 rows of data are dropped silently.
    """
    print(f"  Downloading {len(SCREENING_UNIVERSE)} tickers (period={period}) …", flush=True)

    raw = yf.download(
        tickers=SCREENING_UNIVERSE,
        period=period,
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    data = {}
    for ticker in SCREENING_UNIVERSE:
        try:
            df = raw[ticker].copy() if len(SCREENING_UNIVERSE) > 1 else raw.copy()
            df.dropna(subset=["Close"], inplace=True)
            if len(df) >= 30:
                data[ticker] = df
        except (KeyError, TypeError):
            pass

    print(f"  Usable data for {len(data)} / {len(SCREENING_UNIVERSE)} tickers.\n")
    return data


# ── screening filters ─────────────────────────────────────────────────────────

def _add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Return a copy of *df* with an 'rsi' column added."""
    df = df.copy()
    df["rsi"] = ta.momentum.RSIIndicator(close=df["Close"], window=window).rsi()
    return df


def screen_stocks(data: dict) -> pd.DataFrame:
    """
    Apply the four screening filters to each ticker in *data*.

    Returns a DataFrame of passing stocks sorted by 5-day % gain (best first).
    Returns an empty DataFrame if nothing passes all filters.
    """
    rows = []

    for ticker, df in data.items():
        try:
            df = _add_rsi(df)
            latest = df.iloc[-1]

            current_price = float(latest["Close"])
            current_rsi   = float(latest["rsi"])
            avg_vol_20    = float(df["Volume"].tail(20).mean())

            # Need at least 6 rows to measure a 5-day change (today vs 5 days ago)
            if len(df) < 6:
                continue
            price_5d_ago  = float(df["Close"].iloc[-6])
            pct_change_5d = (current_price - price_5d_ago) / price_5d_ago * 100

            # ── filter 1: price under $75 ──────────────────────────────────────
            if current_price >= 75:
                continue

            # ── filter 2: 20-day average volume > 1,000,000 shares ────────────
            # Ensures the stock is liquid enough to enter and exit without slippage.
            if avg_vol_20 < 1_000_000:
                continue

            # ── filter 3: RSI under 50 — not yet overbought ───────────────────
            # We want stocks that still have room to run. RSI above 50 means momentum
            # buyers are already in; RSI under 50 means we're not late to the party.
            if current_rsi >= 50:
                continue

            # ── filter 4: up more than 3 % over the last 5 trading days ───────
            # This confirms the stock is already moving in the right direction.
            # A 3 % gain in 5 days is meaningful without being so large that the
            # move is already exhausted.
            if pct_change_5d <= 3.0:
                continue

            rows.append({
                "Ticker":       ticker,
                "Price":        round(current_price, 2),
                "RSI":          round(current_rsi, 1),
                "5d Change %":  round(pct_change_5d, 2),
                "Avg Vol (20d)": int(avg_vol_20),
            })

        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    results = pd.DataFrame(rows)
    results.sort_values("5d Change %", ascending=False, inplace=True)
    results.reset_index(drop=True, inplace=True)
    results.index += 1   # rank from 1
    return results


# ── output ────────────────────────────────────────────────────────────────────

_W = 62


def run_screener():
    print("=" * _W)
    print("  STOCK SCREENER — Small/Mid-Cap Momentum Candidates")
    print("=" * _W)
    print()
    print("  Filters:")
    print("    • Price under $75")
    print("    • Average daily volume > 1,000,000 shares")
    print("    • RSI(14) < 50  (not yet overbought — room to run)")
    print("    • Up more than +3 % in the last 5 trading days")
    print()

    data = download_universe_data()
    results = screen_stocks(data)

    if results.empty:
        print("  No stocks passed all four filters today.")
        print()
        print("=" * _W)
        return

    # Header row
    print(f"  {'#':<4} {'Ticker':<8} {'Price':>7} {'RSI':>6} {'5d Chg':>9}  {'Avg Vol (20d)':>14}")
    print("  " + "─" * (_W - 2))

    for idx, row in results.iterrows():
        vol_str   = f"{row['Avg Vol (20d)']:>14,.0f}"
        chg_str   = f"{row['5d Change %']:>+8.2f}%"
        print(
            f"  {idx:<4} {row['Ticker']:<8}"
            f" ${row['Price']:>6.2f}"
            f" {row['RSI']:>6.1f}"
            f" {chg_str}"
            f"  {vol_str}"
        )

    print()
    print("─" * _W)
    print(f"  {len(results)} candidate(s) from {len(SCREENING_UNIVERSE)}-ticker universe")
    print("=" * _W)


if __name__ == "__main__":
    run_screener()
