"""
dashboard/app.py

Light-themed trading dashboard — four tabs:
  1. Today's Signals  — watchlist scores + momentum screener
  2. Options Desk     — active positions tracker + options scanner + glossary
  3. Research         — AI deep-dive reports via Claude
  4. Trade Log        — SQLite-backed trade journal

Run:
    streamlit run dashboard/app.py
"""

import os
import sys
import math
import sqlite3
import warnings
import logging
from datetime import date, datetime, timedelta

import pandas as pd
import yfinance as yf
import streamlit as st
import anthropic

warnings.filterwarnings("ignore")
for _log in ("yfinance", "yfinance.base", "urllib3"):
    logging.getLogger(_log).setLevel(logging.CRITICAL)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from signals.indicators import get_all_indicators
from signals.scorer import score_stock, _ml_confidence, _ML_MODEL
from signals.screener import download_universe_data, screen_stocks
from research.prompts import DEEP_DIVE, SHORT_VERSION
from research.analyst import fetch_financials, _load_env

_load_env()

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'trading_bot.db'))

# ── challenge tracker constants ──────────────────────────────────────────────
CHALLENGE_START = 201.99
CHALLENGE_GOAL  = 1_000.00
_BALANCE_PATH   = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'balance.txt'))
try:
    CHALLENGE_CURRENT = float(open(_BALANCE_PATH).read().strip())
except Exception:
    CHALLENGE_CURRENT = 325.75


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Trading Desk",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── light theme CSS ───────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* ── base & typography ── */
  html, body, [data-testid="stAppViewContainer"],
  [data-testid="stApp"], section.main {
    background:#ffffff !important;
    color:#000000 !important;
    font-family:-apple-system,BlinkMacSystemFont,'SF Pro Display','Segoe UI',Helvetica,Arial,sans-serif !important;
  }
  [data-testid="stSidebar"] { background:#f8f9fa !important; }
  [data-testid="stHeader"]  { background:#ffffff !important; box-shadow:none !important; }

  /* ── pill tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background:#f2f2f2 !important;
    border-radius:100px;
    padding:5px;
    gap:2px;
  }
  .stTabs [data-baseweb="tab"] {
    background:transparent !important;
    color:#424242 !important;
    font-size:0.88rem;
    font-weight:600;
    padding:0.45rem 1.2rem;
    border-radius:100px;
    border:none !important;
    transition:all .2s ease;
  }
  .stTabs [aria-selected="true"] {
    background:#000000 !important;
    color:#ffffff !important;
    box-shadow:0 2px 8px rgba(0,0,0,0.20);
  }

  /* ── cards ── */
  .card {
    background:#ffffff;
    border:1px solid #f0f0f0;
    border-radius:16px;
    padding:20px 24px;
    margin-bottom:12px;
    box-shadow:0 2px 12px rgba(0,0,0,0.07);
  }
  .metric-card {
    background:#ffffff;
    border:1px solid #f0f0f0;
    border-radius:16px;
    padding:20px 20px;
    text-align:center;
    box-shadow:0 2px 12px rgba(0,0,0,0.07);
  }
  .metric-card .label { font-size:0.68rem;color:#424242;text-transform:uppercase;letter-spacing:.09em;font-weight:700;margin-bottom:6px; }
  .metric-card .value { font-size:2.1rem;font-weight:800;color:#000;line-height:1.1;letter-spacing:-.025em; }
  .metric-card .sub   { font-size:0.74rem;color:#999;margin-top:5px; }

  /* ── gain / loss / warn ── */
  .gain { color:#00c853 !important; }
  .loss { color:#ff1744 !important; }
  .flat { color:#888 !important; }
  .warn { color:#e65100 !important; }

  /* ── badges ── */
  .badge {
    display:inline-block;padding:3px 10px;
    border-radius:100px;font-size:0.69rem;
    font-weight:700;letter-spacing:.03em;
  }
  .badge-green  { background:#e8f5e9;color:#1b5e20;border:1px solid #c8e6c9; }
  .badge-red    { background:#ffebee;color:#b71c1c;border:1px solid #ffcdd2; }
  .badge-yellow { background:#fff8e1;color:#e65100;border:1px solid #ffe082; }
  .badge-blue   { background:#e3f2fd;color:#1565c0;border:1px solid #bbdefb; }
  .badge-purple { background:#f3e5f5;color:#6a1b9a;border:1px solid #e1bee7; }
  .badge-orange { background:#fff3e0;color:#e65100;border:1px solid #ffcc80; }

  /* ── options cards ── */
  .options-card {
    background:#ffffff;border:1px solid #f0f0f0;border-radius:16px;
    padding:18px 22px;margin-bottom:12px;
    box-shadow:0 2px 12px rgba(0,0,0,0.07);
    transition:box-shadow .2s,border-color .2s;
  }
  .options-card:hover { border-color:#d0d0d0;box-shadow:0 4px 20px rgba(0,0,0,0.12); }
  .options-card .ticker { font-size:1.3rem;font-weight:800;color:#000;letter-spacing:-.02em; }
  .options-card .thesis { font-size:0.82rem;color:#424242;margin-top:8px;line-height:1.6; }

  /* ── data tables ── */
  [data-testid="stDataFrame"] { border-radius:12px;overflow:hidden;border:1px solid #f0f0f0;box-shadow:0 1px 6px rgba(0,0,0,0.05); }
  [data-testid="stDataFrame"] th { background:#f8f9fa !important;color:#424242 !important;font-size:0.78rem;font-weight:700; }

  /* ── inputs ── */
  [data-testid="stTextInput"] input,
  [data-testid="stNumberInput"] input,
  textarea {
    border:1.5px solid #e0e0e0 !important;
    border-radius:10px !important;
    background:#fff !important;
  }
  [data-testid="stTextInput"] input:focus,
  [data-testid="stNumberInput"] input:focus {
    border-color:#000 !important;
    box-shadow:0 0 0 3px rgba(0,0,0,0.07) !important;
  }

  /* ── divider ── */
  hr { border-color:#f2f2f2 !important; }

  /* ── captions ── */
  .stCaption, small { color:#999 !important; }

  /* ── spinner ── */
  .stSpinner > div { border-top-color:#000 !important; }

  /* ── scrollbar ── */
  ::-webkit-scrollbar { width:5px;height:5px; }
  ::-webkit-scrollbar-track { background:#f8f9fa; }
  ::-webkit-scrollbar-thumb { background:#e0e0e0;border-radius:10px; }

  /* ── layout ── */
  .block-container { padding-top:2rem;padding-bottom:2rem; }

  /* ── page header ── */
  .page-header {
    display:flex;align-items:center;gap:10px;
    padding:12px 0 10px;border-bottom:1px solid #f2f2f2;margin-bottom:1.2rem;
  }
  .page-header .logo { font-size:1.5rem;line-height:1; }
  .page-header .title { font-size:1.35rem;font-weight:800;color:#000;letter-spacing:-.03em; }
  .page-header .timestamp { font-size:0.76rem;color:#bbb;margin-left:auto; }

  /* ── glossary expander ── */
  [data-testid="stExpander"] {
    border:1px solid #f0f0f0 !important;
    border-radius:12px !important;
    background:#fafafa !important;
  }

  /* ── watchlist chips ── */
  [data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background:#000 !important;color:#fff !important;
    border:none !important;border-radius:100px !important;
    font-weight:700 !important;font-size:0.75rem !important;
  }
  [data-testid="stMultiSelect"] [data-baseweb="tag"] svg { fill:#fff !important; }

  /* ── mobile responsive ── */
  @media (max-width:640px) {
    .block-container { padding-left:0.75rem;padding-right:0.75rem; }
    .stTabs [data-baseweb="tab"] { padding:0.38rem 0.7rem;font-size:0.8rem; }
    .metric-card .value { font-size:1.6rem; }
    .options-card { padding:14px 16px; }
  }
</style>
""", unsafe_allow_html=True)


# ── cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _scored_stocks() -> pd.DataFrame:
    all_data = get_all_indicators()
    rows = []
    for ticker, df in all_data.items():
        score, _ = score_stock(df)
        conf = _ml_confidence(df) if _ML_MODEL is not None else None
        latest   = df.iloc[-1]
        score_i  = int(score)
        conf_pct = round(conf * 100, 1) if conf is not None else float("nan")
        if pd.isna(conf_pct):
            buy_signal = float(score_i)
        else:
            buy_signal = round(score_i * 0.4 + conf_pct * 0.6, 1)
        rows.append({
            "Ticker":     ticker,
            "Buy Signal": buy_signal,
            "Score":      score_i,
            "ML Conf":    conf_pct,
            "Close":      round(float(latest["close"]), 2),
            "RSI":        round(float(latest["rsi"]), 1),
            "Vol Ratio":  round(float(latest["vol_ratio"]), 2),
        })
    df = pd.DataFrame(rows)
    df.sort_values("Buy Signal", ascending=False, na_position="last", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def _screener_results() -> pd.DataFrame:
    data = download_universe_data()
    return screen_stocks(data)


@st.cache_data(ttl=120, show_spinner=False)
def _live_price(ticker: str) -> float | None:
    try:
        hist = yf.Ticker(ticker).history(period="1d")
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 2)
    except Exception:
        pass
    return None


# ── db connection ─────────────────────────────────────────────────────────────

def get_db_connection():
    return sqlite3.connect(DB_PATH)


# ── trade log helpers ─────────────────────────────────────────────────────────

def _init_db():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT    NOT NULL,
                trade_date  TEXT    NOT NULL,
                entry_price REAL    NOT NULL,
                exit_price  REAL,
                notes       TEXT,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS options_positions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker        TEXT    NOT NULL,
                type          TEXT    NOT NULL,
                expiry        TEXT    NOT NULL,
                earnings_date TEXT,
                strike        REAL    NOT NULL,
                qty           INTEGER NOT NULL,
                entry_price   REAL    NOT NULL,
                stop_loss     REAL,
                target1       REAL,
                target2       REAL,
                target3       REAL,
                status        TEXT    DEFAULT 'Open',
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def _save_trade(ticker, trade_date, entry, exit_price, notes):
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO trades (ticker, trade_date, entry_price, exit_price, notes) "
            "VALUES (?, ?, ?, ?, ?)",
            (ticker.upper(), str(trade_date), float(entry),
             float(exit_price) if exit_price else None,
             notes.strip() or None),
        )


def _load_trades() -> pd.DataFrame:
    with get_db_connection() as conn:
        df = pd.read_sql_query(
            "SELECT id, ticker, trade_date, entry_price, exit_price, notes "
            "FROM trades ORDER BY trade_date DESC, id DESC", conn)
    if df.empty:
        return df
    df["P/L %"] = df.apply(
        lambda r: round((r.exit_price - r.entry_price) / r.entry_price * 100, 2)
                  if pd.notna(r.exit_price) else None, axis=1)
    return df


def _save_options_position(data: dict):
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO options_positions
              (ticker, type, expiry, earnings_date, strike, qty, entry_price,
               stop_loss, target1, target2, target3, status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            data["ticker"], data["type"], data["expiry"], data.get("earnings_date"),
            data["strike"], data["qty"], data["entry_price"],
            data.get("stop_loss"), data.get("target1"),
            data.get("target2"), data.get("target3"), "Open",
        ))


def _load_options_positions() -> pd.DataFrame:
    with get_db_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM options_positions ORDER BY created_at DESC", conn)
    return df


def _delete_options_position(pos_id: int):
    with get_db_connection() as conn:
        conn.execute("DELETE FROM options_positions WHERE id=?", (pos_id,))


_init_db()


# ── options scanner ───────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_earnings_date(ticker: str):
    """Return the next earnings date as a date object, or None if unavailable."""
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is None:
            return None
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
            if hasattr(dates, "__iter__"):
                for d in dates:
                    if hasattr(d, "date"):
                        return d.date()
                    if hasattr(d, "year"):
                        return d
        elif hasattr(cal, "loc"):
            if "Earnings Date" in cal.index.tolist():
                val = cal.loc["Earnings Date"]
                if hasattr(val, "iloc"):
                    val = val.iloc[0]
                if hasattr(val, "date"):
                    return val.date()
    except Exception:
        pass
    return None


def _norm_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def _bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes price for a European call option."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0.0)
    sqT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def _evaluate_play(ticker: str, strike: float, expiry_date: date) -> dict:
    """Fetch live data and compute Black-Scholes estimate for a specific options play."""
    try:
        hist = yf.Ticker(ticker).history(period="3mo")
        if hist.empty:
            return {"error": f"No price data found for {ticker}. Check the ticker symbol."}
        price = round(float(hist["Close"].iloc[-1]), 2)

        rets = hist["Close"].pct_change().dropna()
        hv   = max(float(rets.tail(30).std() * (252 ** 0.5)), 0.10)

        dte = (expiry_date - date.today()).days
        if dte <= 0:
            return {"error": "Expiry date must be in the future."}

        raw = _bs_call(price, strike, dte / 365.0, 0.045, hv)
        premium  = round(max(round(raw / 0.05) * 0.05, 0.05), 2)
        cost     = round(premium * 100, 2)
        breakeven = round(strike + premium, 2)

        earnings    = _fetch_earnings_date(ticker)
        days_to_earn = (earnings - date.today()).days if earnings else None

        return {
            "ticker":       ticker,
            "price":        price,
            "strike":       strike,
            "expiry":       str(expiry_date),
            "dte":          dte,
            "hv_pct":       round(hv * 100, 1),
            "premium":      premium,
            "cost":         cost,
            "breakeven":    breakeven,
            "earnings":     earnings,
            "days_to_earn": days_to_earn,
        }
    except Exception as exc:
        return {"error": str(exc)}


@st.cache_data(ttl=300, show_spinner=False)
def _scan_options_candidates() -> list[dict]:
    """
    Combines watchlist (full scorer) + screener universe (momentum filters)
    to find call option plays under $100/contract.
    """
    all_candidates: dict[str, dict] = {}

    # Source 1 — watchlist via full indicator + scorer pipeline
    try:
        all_ind = get_all_indicators()
    except Exception:
        all_ind = {}

    for ticker, df in all_ind.items():
        score, _ = score_stock(df)
        conf = _ml_confidence(df) if _ML_MODEL is not None else None
        conf_pct = round(conf * 100, 1) if conf is not None else 50.0
        buy_signal = round(score * 0.4 + conf_pct * 0.6, 1)
        latest = df.iloc[-1]
        try:
            rets = df["close"].pct_change().dropna()
            hv = round(float(rets.tail(30).std() * (252 ** 0.5) * 100), 1)
        except Exception:
            hv = 30.0
        all_candidates[ticker] = {
            "buy_signal": buy_signal,
            "price": round(float(latest["close"]), 2),
            "rsi": round(float(latest["rsi"]), 1),
            "vol_ratio": round(float(latest["vol_ratio"]), 2),
            "source": "Watchlist",
            "hv": hv,
        }

    # Source 2 — screener universe (already filtered for momentum)
    try:
        screen_data = download_universe_data()
        screen_df   = screen_stocks(screen_data)
        if not screen_df.empty:
            for _, row in screen_df.iterrows():
                ticker = row["Ticker"]
                if ticker in all_candidates:
                    continue  # already scored via watchlist pipeline

                rsi    = float(row["RSI"])
                pct5d  = float(row["5d Change %"])
                avg_vol = float(row["Avg Vol (20d)"])
                price  = float(row["Price"])

                # Simplified signal: RSI recovery + momentum + volume
                rsi_pts = 40 if rsi < 30 else (30 if rsi < 40 else 20)
                mom_pts = 30 if pct5d > 10 else (20 if pct5d > 5 else 10)
                vol_pts = 20 if avg_vol > 3_000_000 else (15 if avg_vol > 2_000_000 else 10)

                all_candidates[ticker] = {
                    "buy_signal": float(min(rsi_pts + mom_pts + vol_pts, 100)),
                    "price":      price,
                    "rsi":        rsi,
                    "vol_ratio":  round(avg_vol / 1_000_000, 1),
                    "source":     "Screener",
                    "hv":         min(round(abs(pct5d) * 8, 1), 200.0),
                }
    except Exception:
        pass

    # Build suggestion cards
    candidates: list[dict] = []
    for ticker, info in all_candidates.items():
        if info["buy_signal"] < 30:
            continue
        price = info["price"]
        if price <= 0:
            continue

        # Standard strike increments
        if price < 5:
            step = 0.50
        elif price < 20:
            step = 1.0
        elif price < 50:
            step = 2.5
        else:
            step = 5.0

        # One strike OTM from current price
        strike = round(round(price / step) * step + step, 2)

        # Estimated premium: ~4% of stock price for ~35 DTE ATM+1 call
        # Rounded to nearest $0.05 — matches typical options chain quoting
        raw_premium = price * 0.04
        premium = round(round(raw_premium / 0.05) * 0.05, 2)
        # Contract cost = premium × 100 (one contract = 100 shares)
        cost_per_contract = round(premium * 100, 2)

        if cost_per_contract > 100:
            continue  # only show plays ≤ $100 total outlay

        # Expiry ~35 days out, rolled to next Friday
        target_exp = date.today() + timedelta(days=35)
        days_to_fri = (4 - target_exp.weekday()) % 7
        expiry_date = target_exp + timedelta(days=days_to_fri)

        sig    = info["buy_signal"]
        source = info["source"]
        strength = "Strong" if sig >= 70 else ("Moderate" if sig >= 50 else "Speculative")

        thesis = (
            f"{strength} setup · {source} signal {sig:.0f}/100 · "
            f"RSI {info['rsi']:.0f} · {info['vol_ratio']:.1f}× avg volume. "
            f"Contract cost = ${cost_per_contract:.0f} "
            f"(${premium:.2f} premium × 100 shares). "
            f"Target strike ${strike:.2f}, break-even ~${strike + premium:.2f}."
        )

        candidates.append({
            "ticker":      ticker,
            "signal":      sig,
            "stock_price": price,
            "strike":      strike,
            "expiry":      expiry_date.strftime("%Y-%m-%d"),
            "premium":     premium,
            "cost":        cost_per_contract,
            "thesis":      thesis,
            "source":      source,
            "hv":          info.get("hv", 30.0),
        })

    candidates.sort(key=lambda x: x["signal"], reverse=True)
    return candidates[:15]


# ── styling helpers ───────────────────────────────────────────────────────────

def _style_signals_table(row):
    sig = row["Buy Signal"]
    anchor = sig if pd.notna(sig) else row["Score"]
    if anchor >= 70:
        bg = "background-color:#e8f5e9; color:#1b5e20"
    elif anchor >= 40:
        bg = "background-color:#fff8e1; color:#e65100"
    else:
        bg = "background-color:#ffebee; color:#b71c1c"
    return [
        "background-color:#1565c0; color:#ffffff; font-weight:700"
        if col == "Buy Signal" else bg
        for col in row.index
    ]


def metric_card(label: str, value: str, sub: str = "", cls: str = "") -> str:
    sub_html = f'<div class="sub {cls}">{sub}</div>' if sub else ""
    return f"""
<div class="metric-card">
  <div class="label">{label}</div>
  <div class="value {cls}">{value}</div>
  {sub_html}
</div>"""


# ── field label helper ────────────────────────────────────────────────────────

def _field(label: str, value: str, color: str = "#000000") -> str:
    return (
        f'<div>'
        f'<div style="color:#999;font-size:0.7rem;text-transform:uppercase;letter-spacing:.05em;font-weight:600;">{label}</div>'
        f'<div style="color:{color};font-weight:700;font-size:0.95rem;">{value}</div>'
        f'</div>'
    )


# ── persistence helpers ───────────────────────────────────────────────────────

def _save_balance(balance: float) -> None:
    with open(_BALANCE_PATH, "w") as f:
        f.write(f"{balance:.2f}")


def _save_watchlist(tickers: list) -> None:
    path = os.path.join(_ROOT, "data", "watchlist.py")
    with open(path, "w") as f:
        f.write("# watchlist.py\n")
        f.write("# Fundamentally strong stocks, all priced under $75.\n")
        f.write("\n")
        f.write("WATCHLIST = [\n")
        for t in tickers:
            f.write(f'    "{t}",\n')
        f.write("]\n")
    import signals.indicators as _si
    _si.WATCHLIST = list(tickers)


# ── session state init ────────────────────────────────────────────────────────

if "wl_tickers" not in st.session_state:
    from data.watchlist import WATCHLIST as _WL_INIT
    st.session_state.wl_tickers = list(_WL_INIT)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="page-header">'
    '<span class="logo">📈</span>'
    '<span class="title">Trading Desk</span>'
    f'<span class="timestamp">{datetime.now().strftime("%A, %B %-d · %I:%M %p")}</span>'
    '</div>',
    unsafe_allow_html=True,
)

# ── challenge tracker ────────────────────────────────────────────────────────
_CH_MILESTONES = [
    (CHALLENGE_START, 400,            "$200 → $400  ·  First Double"),
    (400,             800,            "$400 → $800  ·  Second Double"),
    (800,             CHALLENGE_GOAL, "$800 → $1,000  ·  Final Push"),
]
_ms_lo, _ms_hi, _ms_label = next(
    ((lo, hi, lbl) for lo, hi, lbl in _CH_MILESTONES if CHALLENGE_CURRENT < hi),
    _CH_MILESTONES[-1],
)
_ch_pct_overall  = min(1.0, (CHALLENGE_CURRENT - CHALLENGE_START) / (CHALLENGE_GOAL - CHALLENGE_START))
_ch_pct_ms       = min(1.0, max(0.0, (CHALLENGE_CURRENT - _ms_lo) / (_ms_hi - _ms_lo)))
_ch_gain         = CHALLENGE_CURRENT - CHALLENGE_START
_ring_r          = 38
_ring_circ       = round(2 * math.pi * _ring_r, 2)
_ring_offset     = round(_ring_circ * (1 - _ch_pct_overall), 2)

_ch_tracker_col, _ch_btn_col = st.columns([11, 1])
with _ch_tracker_col:
    st.markdown(f"""
<div style="background:#fff;border:1px solid #f0f0f0;border-radius:16px;box-shadow:0 2px 16px rgba(0,0,0,0.08);padding:22px 26px;margin-bottom:4px;">
  <div style="display:flex;align-items:center;gap:28px;flex-wrap:wrap;">
    <div style="min-width:150px;">
      <div style="font-size:0.63rem;color:#424242;text-transform:uppercase;letter-spacing:.09em;font-weight:700;">Account Balance</div>
      <div style="font-size:3.0rem;font-weight:900;color:#00c853;line-height:1.0;letter-spacing:-.04em;margin-top:4px;">${CHALLENGE_CURRENT:,.2f}</div>
      <div style="font-size:0.74rem;color:#999;margin-top:6px;">+${_ch_gain:.2f} from ${CHALLENGE_START:,.2f}</div>
    </div>
    <div style="position:relative;width:110px;height:110px;flex-shrink:0;">
      <svg width="110" height="110" viewBox="0 0 100 100" style="transform:rotate(-90deg);">
        <circle cx="50" cy="50" r="38" fill="none" stroke="#f0f0f0" stroke-width="10"/>
        <circle cx="50" cy="50" r="38" fill="none" stroke="#00c853" stroke-width="10" stroke-dasharray="{_ring_circ}" stroke-dashoffset="{_ring_offset}" stroke-linecap="round"/>
      </svg>
      <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;line-height:1.2;">
        <div style="font-size:1.15rem;font-weight:900;color:#000;">{_ch_pct_overall*100:.0f}%</div>
        <div style="font-size:0.58rem;color:#999;text-transform:uppercase;font-weight:600;letter-spacing:.05em;">to goal</div>
      </div>
    </div>
    <div style="flex:1;min-width:200px;">
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;">
        <span style="font-size:0.8rem;color:#000;font-weight:700;">🎯 {_ms_label}</span>
        <span style="font-size:0.78rem;color:#00c853;font-weight:700;">{_ch_pct_ms*100:.1f}%</span>
      </div>
      <div style="background:#f2f2f2;border-radius:100px;height:6px;overflow:hidden;">
        <div style="background:#00c853;border-radius:100px;height:6px;width:{_ch_pct_ms*100:.2f}%;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:6px;">
        <span style="font-size:0.70rem;color:#999;">${_ms_lo:,.0f}</span>
        <span style="font-size:0.70rem;color:#999;">${_ms_hi:,.0f}</span>
      </div>
    </div>
    <div style="min-width:110px;text-align:right;">
      <div style="font-size:0.63rem;color:#424242;text-transform:uppercase;letter-spacing:.09em;font-weight:700;">Goal</div>
      <div style="font-size:1.9rem;font-weight:800;color:#000;letter-spacing:-.02em;margin-top:4px;">${CHALLENGE_GOAL:,.0f}</div>
      <div style="font-size:0.72rem;color:#999;margin-top:6px;">×{CHALLENGE_GOAL/CHALLENGE_START:.1f}x from start</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
with _ch_btn_col:
    st.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)
    if st.button("✏️", key="ch_edit_btn", help="Update balance", use_container_width=True):
        st.session_state["ch_editing"] = not st.session_state.get("ch_editing", False)

if st.session_state.get("ch_editing", False):
    _bv1, _bv2, _bv3 = st.columns([3, 1, 7])
    _new_bal = _bv1.number_input(
        "balance",
        value=float(CHALLENGE_CURRENT),
        min_value=0.01,
        step=0.01,
        format="%.2f",
        key="ch_bal_input",
        label_visibility="collapsed",
    )
    if _bv2.button("Save", key="ch_bal_save", use_container_width=True):
        _save_balance(float(_new_bal))
        st.session_state["ch_editing"] = False
        st.rerun()

# ── tabs (Options Desk is first / default) ────────────────────────────────────
tab2, tab1, tab3, tab4 = st.tabs([
    "⚡  Options Desk",
    "📊  Signals",
    "🔍  Research",
    "📒  Trade Log",
])


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Today's Signals
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    hdr, btn = st.columns([6, 1])
    hdr.markdown('<p style="font-size:1.1rem;font-weight:700;color:#1a1a1a;margin:0;">Watchlist Signal Scores</p>', unsafe_allow_html=True)
    if btn.button("↺ Refresh", use_container_width=True, key="refresh_signals"):
        _scored_stocks.clear()
        _screener_results.clear()
        st.rerun()

    with st.spinner("Scoring watchlist…"):
        df_scores = _scored_stocks()

    if df_scores.empty:
        st.warning("No price data. Run `python3 data/market_data.py` first.")
    else:
        if _ML_MODEL is None:
            st.caption("ML model not loaded — run `python3 ml/train_model.py` to enable ML confidence.")

        top      = df_scores.iloc[0]
        n_strong = int((df_scores["Buy Signal"] >= 70).sum())
        n_watch  = int(((df_scores["Buy Signal"] >= 40) & (df_scores["Buy Signal"] < 70)).sum())
        avg_sig  = df_scores["Buy Signal"].mean()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card("Top Pick", top["Ticker"], f'Signal {top["Buy Signal"]:.0f}', "gain"), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("Strong Signals", str(n_strong), "≥ 70 score", "gain"), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("Watch List", str(n_watch), "40–69 score", "warn"), unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card("Avg Signal", f"{avg_sig:.1f}", "all watchlist"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.dataframe(
            df_scores.style
            .apply(_style_signals_table, axis=1)
            .format({
                "Buy Signal": "{:.1f}",
                "Score":      "{}/100",
                "ML Conf":    "{:.0f}%",
                "Close":      "${:.2f}",
                "RSI":        "{:.1f}",
                "Vol Ratio":  "{:.2f}x",
            }, na_rep="—"),
            use_container_width=True,
            height=460,
            hide_index=True,
        )
        st.caption("Buy Signal = Score×0.4 + ML Conf×0.6  ·  🟢 ≥70  🟡 40–69  🔴 <40  ·  Cached 5 min")

    st.divider()

    st.markdown('<p style="font-size:1.05rem;font-weight:700;color:#1a1a1a;margin:0 0 4px;">Momentum Screener — Under $75</p>', unsafe_allow_html=True)
    st.caption("price < $75  ·  avg vol > 1M  ·  RSI < 50  ·  up > 3% in 5 days  ·  200-ticker universe")

    with st.spinner("Scanning 200+ tickers…"):
        df_screen = _screener_results()

    if df_screen.empty:
        st.markdown('<div class="card"><span style="color:#aaa;">No stocks passed all four filters today.</span></div>', unsafe_allow_html=True)
    else:
        st.dataframe(
            df_screen.reset_index().rename(columns={"index": "Rank"}).style.format({
                "Price":         "${:.2f}",
                "RSI":           "{:.1f}",
                "5d Change %":   "{:+.2f}%",
                "Avg Vol (20d)": "{:,.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"✅ {len(df_screen)} candidate(s) found")

    # ── Watchlist Manager ─────────────────────────────────────────────────────
    st.divider()
    st.markdown('<p style="font-size:1.05rem;font-weight:700;color:#1a1a1a;margin:0 0 4px;">Watchlist Manager</p>', unsafe_allow_html=True)
    st.caption("Add or remove tickers. Changes are saved to data/watchlist.py and persist after restart.")

    if "_wl_msg" in st.session_state:
        _wl_msg_text, _wl_msg_type = st.session_state.pop("_wl_msg")
        if _wl_msg_type == "success":
            st.success(_wl_msg_text)
        else:
            st.warning(_wl_msg_text)

    _wl_current = list(st.session_state.wl_tickers)
    if not df_scores.empty:
        _score_order = {t: i for i, t in enumerate(df_scores["Ticker"])}
        _wl_current = sorted(_wl_current, key=lambda t: _score_order.get(t, 999))

    _wl_selected = st.multiselect(
        "Current tickers — click × to remove:",
        options=_wl_current,
        default=_wl_current,
        key="wl_chip_select",
    )

    if sorted(_wl_selected) != sorted(_wl_current):
        _removed = [t for t in _wl_current if t not in _wl_selected]
        st.session_state.wl_tickers = _wl_selected
        _save_watchlist(_wl_selected)
        _scored_stocks.clear()
        st.session_state["_wl_msg"] = (f"Removed {', '.join(_removed)} from watchlist.", "success")
        del st.session_state["wl_chip_select"]
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    _add_c1, _add_c2 = st.columns([4, 1])
    _new_ticker_input = _add_c1.text_input(
        "Add ticker",
        placeholder="Ticker symbol, e.g. NVDA",
        key="wl_add_input",
        label_visibility="collapsed",
    )
    if _add_c2.button("+ Add", key="wl_add_btn", use_container_width=True):
        _t = _new_ticker_input.strip().upper()
        if not _t:
            st.session_state["_wl_msg"] = ("Enter a ticker symbol first.", "warning")
        elif _t in st.session_state.wl_tickers:
            st.session_state["_wl_msg"] = (f"{_t} is already in the watchlist.", "warning")
        else:
            _new_list = list(st.session_state.wl_tickers) + [_t]
            st.session_state.wl_tickers = _new_list
            _save_watchlist(_new_list)
            _scored_stocks.clear()
            st.session_state["_wl_msg"] = (
                f"Added {_t} to watchlist. Run `python3 data/market_data.py` to fetch its price history.",
                "success",
            )
            for _k in ("wl_chip_select", "wl_add_input"):
                st.session_state.pop(_k, None)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Options Desk
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<p style="font-size:1.1rem;font-weight:700;color:#1a1a1a;margin:0 0 14px;">Options Desk</p>', unsafe_allow_html=True)
    od_tab_a, od_tab_b, od_tab_c = st.tabs(["📋  Active Positions", "🔎  Options Scanner", "🎯  Evaluate a Play"])

    # ────────────────────────────────────────────────────────────────────────
    # A — Active Positions
    # ────────────────────────────────────────────────────────────────────────

    with od_tab_a:
        positions = _load_options_positions()

        if not positions.empty:
            open_pos = positions[positions["status"] == "Open"]
            total_invested = float((open_pos["entry_price"] * open_pos["qty"] * 100).sum())
            today = date.today()
            n_warn = sum(
                1 for _, r in open_pos.iterrows()
                if r["earnings_date"] and
                0 <= (datetime.strptime(str(r["earnings_date"]), "%Y-%m-%d").date() - today).days <= 7
            )
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown(metric_card("Open Positions", str(len(open_pos))), unsafe_allow_html=True)
            with mc2:
                st.markdown(metric_card("Capital at Risk", f"${total_invested:,.0f}", "entry × qty × 100"), unsafe_allow_html=True)
            with mc3:
                st.markdown(metric_card("Earnings Alerts", str(n_warn), "within 7 days", "warn" if n_warn else ""), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        if positions.empty:
            st.markdown('<div class="card"><span style="color:#aaa;">No open positions. Add one below.</span></div>', unsafe_allow_html=True)
        else:
            today = date.today()
            for _, row in positions.iterrows():
                live = _live_price(row["ticker"])
                live_str = f"${live:.2f}" if live is not None else "—"

                earn_warn = ""
                if row["earnings_date"]:
                    try:
                        ed = datetime.strptime(str(row["earnings_date"]), "%Y-%m-%d").date()
                        d = (ed - today).days
                        if 0 <= d <= 7:
                            earn_warn = f'<span class="badge badge-yellow">⚠ Earnings in {d}d</span>'
                    except Exception:
                        pass

                type_cls   = "badge-blue" if "Call" in str(row["type"]) else "badge-purple"
                status_cls = "badge-green" if row["status"] == "Open" else "badge-red"

                targets_html = ""
                for lbl, col in [("T1", "target1"), ("T2", "target2"), ("T3", "target3")]:
                    if row[col]:
                        targets_html += f'<span style="font-size:0.78rem;color:#888;">{lbl}: <b>${row[col]:.2f}</b></span>&nbsp;&nbsp;'

                earn_row = (
                    f'<div style="margin-top:6px;font-size:0.78rem;color:#aaa;">Earnings: {row["earnings_date"]}</div>'
                    if row["earnings_date"] else ""
                )

                _ew_inline = (" " + earn_warn) if earn_warn else ""
                st.markdown(f"""
<div class="options-card">
  <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
    <span class="ticker">{row['ticker']}</span>
    <span class="badge {type_cls}">{row['type']}</span>
    <span class="badge {status_cls}">{row['status']}</span>{_ew_inline}
    <span style="margin-left:auto;color:#ccc;font-size:0.78rem;">#{int(row['id'])}</span>
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(115px,1fr));gap:10px;margin-top:14px;">
    {_field("Strike",    f"${row['strike']:.2f}")}
    {_field("Expiry",    row['expiry'])}
    {_field("Qty",       f"{int(row['qty'])} contract{'s' if row['qty']!=1 else ''}")}
    {_field("Entry",     f"${row['entry_price']:.2f}")}
    {_field("Stop Loss", f"${row['stop_loss']:.2f}" if row['stop_loss'] else "—", "#ff1744")}
    {_field("Live $",    live_str)}
    {_field("P&L $",     "—")}
    {_field("P&L %",     "—")}
  </div>
  <div style="margin-top:10px;">{targets_html}</div>{earn_row}
</div>
""", unsafe_allow_html=True)

                if st.button(f"🗑 Remove #{int(row['id'])}", key=f"del_{row['id']}"):
                    _delete_options_position(int(row["id"]))
                    st.rerun()

        st.divider()
        st.markdown('<p style="font-size:0.95rem;font-weight:700;color:#1a1a1a;margin:0 0 10px;">Add Position</p>', unsafe_allow_html=True)

        with st.form("options_position_form", clear_on_submit=True):
            r1c1, r1c2, r1c3 = st.columns(3)
            f_ticker  = r1c1.text_input("Ticker *", placeholder="PLTR")
            f_type    = r1c2.selectbox("Type *", ["Long Call", "Long Put"])
            f_expiry  = r1c3.date_input("Expiry *", value=date.today() + timedelta(days=30))

            r2c1, r2c2, r2c3 = st.columns(3)
            f_earnings = r2c1.date_input("Earnings Date", value=None)
            f_strike   = r2c2.number_input("Strike ($) *", min_value=0.01, step=0.5, format="%.2f")
            f_qty      = r2c3.number_input("Qty (contracts) *", min_value=1, step=1, value=1)

            r3c1, r3c2, r3c3 = st.columns(3)
            f_entry = r3c1.number_input("Entry Price ($) *", min_value=0.01, step=0.01, format="%.2f")
            f_stop  = r3c2.number_input("Stop Loss ($)", min_value=0.0, step=0.01, format="%.2f", value=0.0)
            f_t1    = r3c3.number_input("Target 1 ($)", min_value=0.0, step=0.01, format="%.2f", value=0.0)

            r4c1, r4c2 = st.columns(2)
            f_t2 = r4c1.number_input("Target 2 ($)", min_value=0.0, step=0.01, format="%.2f", value=0.0)
            f_t3 = r4c2.number_input("Target 3 ($)", min_value=0.0, step=0.01, format="%.2f", value=0.0)

            save_pos = st.form_submit_button("💾  Save Position", use_container_width=True)

        if save_pos:
            if not f_ticker.strip():
                st.error("Ticker is required.")
            else:
                _save_options_position({
                    "ticker":        f_ticker.strip().upper(),
                    "type":          f_type,
                    "expiry":        str(f_expiry),
                    "earnings_date": str(f_earnings) if f_earnings else None,
                    "strike":        float(f_strike),
                    "qty":           int(f_qty),
                    "entry_price":   float(f_entry),
                    "stop_loss":     float(f_stop) if f_stop > 0 else None,
                    "target1":       float(f_t1) if f_t1 > 0 else None,
                    "target2":       float(f_t2) if f_t2 > 0 else None,
                    "target3":       float(f_t3) if f_t3 > 0 else None,
                })
                st.success(f"✅ Saved — {f_ticker.upper()} ${f_strike:.2f} {f_type} exp {f_expiry}")
                st.rerun()

    # ────────────────────────────────────────────────────────────────────────
    # B — Options Scanner
    # ────────────────────────────────────────────────────────────────────────

    with od_tab_b:
        with st.expander("📋  Hard Rules — read before placing any trade", expanded=False):
            st.markdown(
                "**1. No trades within 7 days of earnings** — unless the card is intentionally labeled "
                "as an earnings play. IV is elevated before earnings and collapses after, "
                "destroying option value even when the stock moves in your favor.\n\n"

                "**2. Never deploy more than 50% of buying power in one play** — "
                "max risk per trade is half the current account balance. "
                "Two bad trades cannot wipe the account.\n\n"

                "**3. No entry if the stock is already up 5%+ at the open** — "
                "chasing a gap means buying premium at peak intraday IV. "
                "Wait for a pullback or skip it.\n\n"

                "**4. Set the stop loss before entering** — "
                "decide your exit price before emotion is involved. "
                "A common level is 40–50% of the premium paid.\n\n"

                "**5. Limit orders only** — "
                "never use market orders on options. Spreads are wide and "
                "market orders give away edge immediately.\n\n"

                "**6. Take profit at Target 1, never let a winner turn into a loser** — "
                "once the position hits the first target, sell at least half. "
                "Trail the stop up on the remainder. A locked-in gain is always better than a loss."
            )

        sc1, sc2 = st.columns([5, 1])
        sc1.markdown(
            '<p style="color:#666;font-size:0.88rem;margin:0;">'
            'Momentum plays from watchlist + 200-ticker screener universe — call contracts ≤ $100 total cost.'
            '</p>',
            unsafe_allow_html=True,
        )
        if sc2.button("↺ Scan", use_container_width=True, key="scan_options"):
            _scan_options_candidates.clear()
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        with st.spinner("Scanning watchlist + screener universe…"):
            candidates = _scan_options_candidates()

        # Pre-filter: hide any play with earnings within 7 days
        _today = date.today()
        _visible = []
        _earn_hidden = 0
        for c in candidates:
            _ed = _fetch_earnings_date(c["ticker"])
            if _ed is not None and 0 <= (_ed - _today).days <= 7:
                _earn_hidden += 1
            else:
                _visible.append((c, _ed))

        if not _visible:
            _no_msg = (
                'No setups found today. Check back after market open or lower your signal threshold.'
                if not candidates else
                f'All candidates hidden — earnings within 7 days for every play found. '
                'Use the <b>Evaluate a Play</b> tab to check them individually.'
            )
            st.markdown(f'<div class="card"><span style="color:#aaa;">{_no_msg}</span></div>', unsafe_allow_html=True)
        else:
            for c, _earn_date in _visible:
                badge_cls  = "badge-green" if c["signal"] >= 70 else "badge-yellow"
                source_cls = "badge-blue" if c["source"] == "Watchlist" else "badge-purple"

                _earn_field = "Verify before entry"
                if _earn_date is not None:
                    _earn_field = str(_earn_date)

                # IV warning — appended inline to avoid blank-line HTML block breaks
                _iv_badge = ""
                if c.get("hv", 0) > 80:
                    _iv_badge = '<span class="badge badge-orange">⚠️ Elevated IV</span>'
                _header_extras = (" " + _iv_badge if _iv_badge else "")

                st.markdown(f"""
<div class="options-card">
  <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
    <span class="ticker">{c['ticker']}</span>
    <span class="badge {badge_cls}">Signal {c['signal']:.0f}</span>
    <span class="badge badge-blue">Long Call</span>
    <span class="badge {source_cls}">{c['source']}</span>{_header_extras}
    <span style="margin-left:auto;font-size:1.15rem;font-weight:800;color:#00c853;">${c['cost']:.0f}</span><span style="color:#aaa;font-size:0.78rem;"> total / contract</span>
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-top:14px;">
    {_field("Stock Price",      f"${c['stock_price']:.2f}")}
    {_field("Suggested Strike", f"${c['strike']:.2f}")}
    {_field("Expiry (~35 DTE)", c['expiry'])}
    {_field("Est. Premium",     f"${c['premium']:.2f} / share")}
    {_field("Contract Cost",    f"${c['cost']:.0f}  (premium × 100)")}
    {_field("Break-even",       f"${c['strike'] + c['premium']:.2f}")}
    {_field("Earnings",         _earn_field)}
  </div>
  <div class="thesis">{c['thesis']}</div>
</div>
""", unsafe_allow_html=True)

        if _earn_hidden > 0:
            st.caption(
                f"⚠️ {_earn_hidden} play{'s' if _earn_hidden != 1 else ''} hidden — "
                "earnings within 7 days. Use **Evaluate a Play** tab to override."
            )

        # ── Glossary ──────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📖 Options Glossary — key terms explained"):
            st.markdown(
                "**Strike Price** — The price at which you have the right to buy (call) or sell (put) "
                "100 shares of the stock. A \\$20 strike call lets you buy 100 shares at \\$20 regardless "
                "of where the stock is actually trading.\n\n"

                "**Premium** — The per-share price of the option as quoted on the chain, for example \\$0.45. "
                "This is what you pay to own the contract.\n\n"

                "**Contract Cost** — One standard contract controls 100 shares, so your total cost is "
                "the premium multiplied by 100. A \\$0.45 premium costs \\$45 total per contract. "
                "That \\$45 is your maximum possible loss if the option expires worthless.\n\n"

                "**Expiry / DTE (Days to Expiration)** — The date the contract expires. After that date "
                "an out-of-the-money option is worth zero. Shorter DTE means cheaper premiums but faster "
                "time decay. 30 to 45 DTE is a common sweet spot for directional plays.\n\n"

                "**ITM / ATM / OTM** — In-the-money means the strike is already below the current stock "
                "price (for calls). At-the-money means the strike is close to the current price. "
                "Out-of-the-money means the strike is above the current price (for calls). OTM options "
                "cost less but need a bigger stock move to become profitable.\n\n"

                "**Break-even** — Strike price plus the premium you paid. The stock must close at or above "
                "this level at expiry for the trade to be profitable. Example: a \\$20 strike call bought "
                "for \\$0.45 breaks even at \\$20.45.\n\n"

                "**Delta** — How much the option price moves for every one dollar move in the stock. "
                "An at-the-money call has a delta of roughly 0.50, meaning a \\$1 stock move "
                "produces a \\$0.50 gain on the option, which is \\$50 per contract. Deep in-the-money "
                "options can have a delta near 1.0.\n\n"

                "**Theta (Time Decay)** — The daily dollar amount an option loses just from time passing, "
                "even if the stock does not move. Options bleed value every day and the decay "
                "accelerates sharply in the final two weeks before expiry.\n\n"

                "**IV (Implied Volatility)** — The market's expectation of future price swings baked into "
                "the premium. High IV makes options expensive. Avoid buying options right before earnings "
                "when IV is elevated — it collapses after the announcement (known as IV crush), "
                "which can make your option lose value even if the stock moves in your favor.\n\n"

                "**Buy Signal Score** — The bot's composite momentum score from 0 to 100. Watchlist stocks "
                "use rule-based technical indicators (40% weight) combined with ML confidence (60% weight). "
                "Screener-only stocks use a simplified score based on RSI, 5-day momentum, and volume. "
                "A score of 70 or above is strong, 40 to 69 is moderate, below 40 is speculative.\n\n"

                "**Source: Watchlist vs Screener** — Watchlist plays go through the full technical and ML "
                "scoring pipeline. Screener plays are additional candidates from the 200-ticker momentum "
                "scan that are not on your watchlist. Both sources are combined and ranked by signal score."
            )

    # ────────────────────────────────────────────────────────────────────────
    # C — Evaluate a Play
    # ────────────────────────────────────────────────────────────────────────

    with od_tab_c:
        st.markdown('<p style="font-size:0.95rem;font-weight:700;color:#1a1a1a;margin:0 0 4px;">Evaluate a Play</p>', unsafe_allow_html=True)
        st.caption("Enter any ticker and strike to get a live premium estimate, earnings check, and cost assessment.")

        ev_c1, ev_c2 = st.columns(2)
        ev_ticker = ev_c1.text_input("Ticker symbol", placeholder="e.g. SAIL", key="ev_ticker").strip().upper()
        ev_strike = ev_c2.number_input("Strike price ($)", min_value=0.01, value=10.0, step=0.50, format="%.2f", key="ev_strike")

        _DTE_OPTS = {
            "~2 weeks (14 DTE)": 14,
            "~1 month (30 DTE)": 30,
            "~45 days (45 DTE)": 45,
            "Custom date":       None,
        }
        ev_dte_label = st.selectbox("Expiry", list(_DTE_OPTS.keys()), key="ev_dte_sel")

        ev_expiry_date = None
        if _DTE_OPTS[ev_dte_label] is None:
            ev_expiry_date = st.date_input(
                "Custom expiry date",
                value=date.today() + timedelta(days=30),
                min_value=date.today() + timedelta(days=1),
                key="ev_custom_date",
            )

        if st.button("⚡  Evaluate", key="ev_evaluate_btn"):
            if not ev_ticker:
                st.warning("Enter a ticker symbol first.")
            elif ev_strike <= 0:
                st.warning("Enter a valid strike price.")
            else:
                dte_days = _DTE_OPTS[ev_dte_label]
                if dte_days is not None:
                    target = date.today() + timedelta(days=dte_days)
                    expiry = target + timedelta(days=(4 - target.weekday()) % 7)
                else:
                    expiry = ev_expiry_date
                with st.spinner(f"Fetching data for {ev_ticker}…"):
                    st.session_state["ev_result"] = _evaluate_play(ev_ticker, float(ev_strike), expiry)

        if "ev_result" in st.session_state:
            r = st.session_state["ev_result"]

            if "error" in r:
                st.error(r["error"])
            else:
                st.markdown("<br>", unsafe_allow_html=True)

                cost_ok  = r["cost"] <= 100
                cost_cls = "badge-green" if cost_ok else "badge-red"
                cost_lbl = f"✅ ${r['cost']:.0f} — within budget" if cost_ok else f"⚠️ ${r['cost']:.0f} — over $100"
                iv_high  = r["hv_pct"] > 80

                # Earnings section — always a non-empty string so it never creates a blank line
                d2e = r["days_to_earn"]
                if r["earnings"] is None:
                    _earn_bg, _earn_fg = "#fffde7", "#e65100"
                    _earn_txt = "⚠️ Earnings date unknown — verify on your broker before entry"
                elif d2e < 0:
                    _earn_bg, _earn_fg = "#f3e5f5", "#6a1b9a"
                    _earn_txt = f"Earnings recently passed ({abs(d2e)}d ago) — IV may still be resetting"
                elif d2e <= 7:
                    _earn_bg, _earn_fg = "#ffebee", "#c62828"
                    _earn_txt = f"⚠️ Do not enter — earnings in {d2e}d. IV crush risk."
                else:
                    _earn_bg, _earn_fg = "#e8f5e9", "#2e7d32"
                    _earn_txt = f"✅ Earnings clear — safe entry window. Next: {r['earnings']} ({d2e}d away)"

                # IV warning — appended to same line as earnings div to avoid blank lines
                _iv_extra = ""
                if iv_high:
                    _iv_extra = f'<div style="margin-top:8px;padding:8px 12px;background:#fff3e0;border-radius:8px;color:#e65100;font-weight:600;font-size:0.85rem;">⚠️ Elevated historical vol ({r["hv_pct"]:.1f}%) — premiums are expensive. Size position down.</div>'

                st.markdown(f"""
<div class="options-card">
  <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
    <span class="ticker">{r['ticker']}</span>
    <span class="badge badge-blue">Long Call · {r['dte']}d</span>
    <span class="badge {cost_cls}">{cost_lbl}</span>
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-top:14px;">
    {_field("Stock Price",   f"${r['price']:.2f}")}
    {_field("Strike",        f"${r['strike']:.2f}")}
    {_field("Expiry",        r['expiry'])}
    {_field("DTE",           f"{r['dte']} days")}
    {_field("Est. HV (30d)", f"{r['hv_pct']:.1f}%", "#e65100" if iv_high else "#1a1a1a")}
    {_field("Est. Premium",  f"${r['premium']:.2f} / share")}
    {_field("Contract Cost", f"${r['cost']:.0f}  (premium × 100)", "#00c853" if cost_ok else "#ff1744")}
    {_field("Break-even",    f"${r['breakeven']:.2f}")}
  </div>
  <div style="margin-top:12px;padding:8px 12px;background:{_earn_bg};border-radius:8px;color:{_earn_fg};font-weight:600;font-size:0.85rem;">{_earn_txt}</div>{_iv_extra}
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Research
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<p style="font-size:1.1rem;font-weight:700;color:#1a1a1a;margin:0 0 4px;">AI Stock Research</p>', unsafe_allow_html=True)
    st.caption("Powered by Claude Opus 4.7 with adaptive thinking · Financial data via yfinance")

    ticker_in = st.text_input(
        "ticker",
        placeholder="e.g. PLTR, PYPL, AMD",
        label_visibility="collapsed",
        key="research_ticker",
    ).strip().upper()

    c1, c2 = st.columns(2)
    btn_full  = c1.button("📊  Full Deep Dive (9 sections)", use_container_width=True)
    btn_short = c2.button("📋  Short Report (sections 2–8)", use_container_width=True)

    if btn_full or btn_short:
        if not ticker_in:
            st.warning("Please enter a ticker symbol first.")
        else:
            short_mode   = btn_short and not btn_full
            report_label = (
                f"{ticker_in} — "
                + ("Short Report (sections 2–8)" if short_mode else "Full Deep Dive")
            )
            st.markdown(f'<p style="font-size:1rem;font-weight:700;color:#1a1a1a;margin:12px 0 4px;">{report_label}</p>', unsafe_allow_html=True)

            with st.spinner(f"Fetching financial data for {ticker_in}…"):
                company, fin_data = fetch_financials(ticker_in)

            template = SHORT_VERSION if short_mode else DEEP_DIVE
            prompt = (
                template
                .replace("[TICKER]",        ticker_in)
                .replace("[COMPANY]",        company)
                .replace("[FINANCIAL_DATA]", fin_data)
            )

            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            client  = anthropic.Anthropic(api_key=api_key)

            def _report_gen():
                with client.messages.stream(
                    model="claude-opus-4-7",
                    max_tokens=4096,
                    thinking={"type": "adaptive"},
                    messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    for chunk in stream.text_stream:
                        yield chunk

            result = st.write_stream(_report_gen())
            st.session_state["_last_report"]       = result
            st.session_state["_last_report_label"] = report_label

    elif st.session_state.get("_last_report"):
        st.markdown(f'<p style="font-size:1rem;font-weight:700;color:#1a1a1a;margin:12px 0 4px;">{st.session_state["_last_report_label"]}</p>', unsafe_allow_html=True)
        st.markdown(st.session_state["_last_report"])


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Trade Log
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    trades = _load_trades()
    closed = trades.dropna(subset=["P/L %"]) if not trades.empty else pd.DataFrame()

    n_total  = len(trades) if not trades.empty else 0
    n_closed = len(closed)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(metric_card("Total Trades", str(n_total)), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_card("Closed Trades", str(n_closed)), unsafe_allow_html=True)

    if not closed.empty:
        wins    = int((closed["P/L %"] > 0).sum())
        avg_ret = float(closed["P/L %"].mean())
        with m3:
            wr_cls = "gain" if wins / n_closed >= 0.5 else "loss"
            st.markdown(metric_card("Win Rate", f"{wins/n_closed:.0%}", f"{wins}W / {n_closed-wins}L", wr_cls), unsafe_allow_html=True)
        with m4:
            rc = "gain" if avg_ret > 0 else "loss"
            st.markdown(metric_card("Avg Return", f"{avg_ret:+.2f}%", "closed trades", rc), unsafe_allow_html=True)
    else:
        with m3:
            st.markdown(metric_card("Win Rate", "—"), unsafe_allow_html=True)
        with m4:
            st.markdown(metric_card("Avg Return", "—"), unsafe_allow_html=True)

    st.divider()

    st.markdown('<p style="font-size:0.95rem;font-weight:700;color:#1a1a1a;margin:0 0 10px;">Log a Trade</p>', unsafe_allow_html=True)

    with st.form("trade_form", clear_on_submit=True):
        fc1, fc2 = st.columns(2)
        f_ticker = fc1.text_input("Ticker *", placeholder="PLTR")
        f_date   = fc2.date_input("Trade Date *", value=date.today())

        fc3, fc4 = st.columns(2)
        f_entry = fc3.number_input("Entry Price * ($)", min_value=0.01, step=0.01, format="%.2f")
        f_exit  = fc4.number_input(
            "Exit Price ($)  [leave 0 if open]",
            min_value=0.0, step=0.01, format="%.2f", value=0.0,
        )
        f_notes  = st.text_area("Notes", placeholder="Setup, strategy, outcome…", height=80)
        save_btn = st.form_submit_button("💾  Save Trade", use_container_width=True)

    if save_btn:
        if not f_ticker.strip():
            st.error("Ticker is required.")
        else:
            _save_trade(f_ticker.strip(), f_date, f_entry,
                        f_exit if f_exit > 0 else None, f_notes)
            st.success(f"✅ Trade saved — {f_ticker.upper()} @ ${f_entry:.2f}")
            st.rerun()

    st.divider()

    st.markdown('<p style="font-size:0.95rem;font-weight:700;color:#1a1a1a;margin:0 0 10px;">Trade History</p>', unsafe_allow_html=True)

    if trades.empty:
        st.markdown('<div class="card"><span style="color:#aaa;">No trades logged yet.</span></div>', unsafe_allow_html=True)
    else:
        disp = trades.drop(columns=["id"]).rename(columns={
            "ticker":      "Ticker",
            "trade_date":  "Date",
            "entry_price": "Entry $",
            "exit_price":  "Exit $",
            "notes":       "Notes",
        })

        def _pnl_color(v):
            try:
                if pd.isna(v):
                    return ""
                return "color:#00c853;font-weight:700" if float(v) > 0 else "color:#ff1744;font-weight:700"
            except Exception:
                return ""

        try:
            sty = disp.style.map(_pnl_color, subset=["P/L %"])
        except AttributeError:
            sty = disp.style.applymap(_pnl_color, subset=["P/L %"])

        sty = sty.format({
            "Entry $": "${:.2f}",
            "Exit $":  lambda v: f"${v:.2f}" if pd.notna(v) else "—",
            "P/L %":   lambda v: f"{v:+.2f}%" if pd.notna(v) else "Open",
        })
        st.dataframe(sty, use_container_width=True, hide_index=True)
