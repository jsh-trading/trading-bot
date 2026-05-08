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
from datetime import date, datetime, timedelta, timezone
import time as _time

import json
import pandas as pd
import yfinance as yf
import streamlit as st
import anthropic
from streamlit_autorefresh import st_autorefresh

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
_BP_PATH        = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'buying_power.txt'))
_POSITIONS_BACKUP_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'positions_backup.json'))
# CHALLENGE_CURRENT and BUYING_POWER are loaded after Supabase initializes (below page config)


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Trading Desk",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st_autorefresh(interval=60000, key="autorefresh")

# ── Supabase REST helpers (no client library — pure requests) ─────────────────
import requests as _requests

_has_supabase = (
    hasattr(st, "secrets")
    and "SUPABASE_URL" in st.secrets
    and "SUPABASE_KEY" in st.secrets
)

if _has_supabase:
    _SB_URL = st.secrets["SUPABASE_URL"].rstrip("/") + "/rest/v1/"
    _SB_KEY = st.secrets["SUPABASE_KEY"]
    _SB_HEADERS = {
        "apikey":        _SB_KEY,
        "Authorization": "Bearer " + _SB_KEY,
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",
    }
    st.sidebar.success("✅ Supabase connected")
else:
    _SB_URL = _SB_KEY = _SB_HEADERS = None
    st.sidebar.warning("⚠️ Using local SQLite — positions will reset on reboot.")


def _sb_get(table: str, params: dict | None = None) -> list:
    r = _requests.get(_SB_URL + table, headers=_SB_HEADERS, params=params, timeout=10)
    if not r.ok:
        print(f"[SB ERROR] GET {table} → {r.status_code}: {r.text}", flush=True)
    r.raise_for_status()
    return r.json()


def _sb_post(table: str, payload: dict) -> dict:
    r = _requests.post(_SB_URL + table, headers=_SB_HEADERS, json=payload, timeout=10)
    if not r.ok:
        print(f"[SB ERROR] POST {table} → {r.status_code}: {r.text}", flush=True)
    r.raise_for_status()
    data = r.json()
    return data[0] if isinstance(data, list) else data


def _sb_patch(table: str, params: dict, payload: dict) -> None:
    r = _requests.patch(_SB_URL + table, headers=_SB_HEADERS, params=params, json=payload, timeout=10)
    if not r.ok:
        print(f"[SB ERROR] PATCH {table} → {r.status_code}: {r.text}", flush=True)
    r.raise_for_status()


def _sb_delete(table: str, params: dict) -> None:
    r = _requests.delete(_SB_URL + table, headers=_SB_HEADERS, params=params, timeout=10)
    if not r.ok:
        print(f"[SB ERROR] DELETE {table} → {r.status_code}: {r.text}", flush=True)
    r.raise_for_status()


def _load_setting(key: str, default: float) -> float:
    if _has_supabase:
        try:
            rows = _sb_get("app_settings", {"key": f"eq.{key}", "select": "value"})
            if rows:
                return float(rows[0]["value"])
        except Exception:
            pass
    path = _BALANCE_PATH if key == "balance" else _BP_PATH
    try:
        return float(open(path).read().strip())
    except Exception:
        return default


CHALLENGE_CURRENT = _load_setting("balance", 325.75)
BUYING_POWER      = _load_setting("buying_power", CHALLENGE_CURRENT)

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


@st.cache_data(ttl=3600, show_spinner=False)
def _prev_close(ticker: str) -> float | None:
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        if len(hist) >= 2:
            return round(float(hist["Close"].iloc[-2]), 2)
    except Exception:
        pass
    return None


_STRIP_TICKERS = [("SPY", "SPY"), ("QQQ", "QQQ"), ("VIX", "^VIX"), ("BTC", "BTC-USD")]


@st.cache_data(ttl=120, show_spinner=False)
def _fetch_ticker_strip() -> list:
    out = []
    for label, sym in _STRIP_TICKERS:
        price = _live_price(sym)
        prev  = _prev_close(sym)
        pct   = ((price - prev) / prev * 100) if (price and prev and prev != 0) else None
        out.append({"label": label, "price": price, "pct": pct})
    return out


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_market_data() -> dict:
    """Fetch VIX level and SPY price / 200-day MA / RSI for Market Intel tab."""
    out = {"vix": None, "spy_price": None, "spy_ma200": None, "spy_rsi": None}
    try:
        vix_h = yf.Ticker("^VIX").history(period="2d")
        if not vix_h.empty:
            out["vix"] = round(float(vix_h["Close"].iloc[-1]), 2)
    except Exception:
        pass
    try:
        spy_h = yf.Ticker("SPY").history(period="1y")
        if len(spy_h) >= 50:
            closes = spy_h["Close"]
            out["spy_price"] = round(float(closes.iloc[-1]), 2)
            if len(closes) >= 200:
                out["spy_ma200"] = round(float(closes.tail(200).mean()), 2)
            delta = closes.diff().dropna()
            g_avg = delta.clip(lower=0).rolling(14).mean().iloc[-1]
            l_avg = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
            if float(l_avg) == 0:
                out["spy_rsi"] = 100.0
            else:
                out["spy_rsi"] = round(100 - 100 / (1 + float(g_avg) / float(l_avg)), 1)
    except Exception:
        pass
    return out


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
                notes         TEXT,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # add notes column to existing DBs that predate the schema change
        try:
            conn.execute("ALTER TABLE options_positions ADD COLUMN notes TEXT")
            conn.commit()
        except Exception:
            pass  # column already exists — safe to ignore
        try:
            conn.execute("ALTER TABLE options_positions ADD COLUMN live_option_price REAL")
            conn.commit()
        except Exception:
            pass

        # restore from JSON backup if table is empty (e.g. after Streamlit Cloud reboot)
        row_count = conn.execute("SELECT COUNT(*) FROM options_positions").fetchone()[0]
        if row_count == 0 and os.path.exists(_POSITIONS_BACKUP_PATH):
            try:
                with open(_POSITIONS_BACKUP_PATH) as f:
                    rows = json.load(f)
                for r in rows:
                    conn.execute("""
                        INSERT INTO options_positions
                          (ticker, type, expiry, earnings_date, strike, qty, entry_price,
                           stop_loss, target1, target2, target3, status, notes, created_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        r.get("ticker"), r.get("type"), r.get("expiry"),
                        r.get("earnings_date"), r.get("strike"), r.get("qty"),
                        r.get("entry_price"), r.get("stop_loss"), r.get("target1"),
                        r.get("target2"), r.get("target3"),
                        r.get("status", "Open"), r.get("notes"), r.get("created_at"),
                    ))
                conn.commit()
            except Exception:
                pass


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
    if _has_supabase:
        try:
            result = _sb_post("options_positions", {
                "ticker":        data["ticker"],
                "type":          data["type"],
                "expiry":        data["expiry"],
                "earnings_date": data.get("earnings_date"),
                "strike":        data["strike"],
                "qty":           data["qty"],
                "entry_price":   data["entry_price"],
                "stop_loss":     data.get("stop_loss"),
                "target1":       data.get("target1"),
                "target2":       data.get("target2"),
                "target3":       data.get("target3"),
                "status":        "Open",
                "notes":         data.get("notes"),
            })
            if result and result.get("id"):
                return  # confirmed saved with a real id
            print(f"[SB WARNING] Save returned no id — result: {result}", flush=True)
        except Exception as _e:
            print(f"[SB ERROR] _save_options_position failed: {_e}", flush=True)
    # Fallback: SQLite
    print("[DB] Falling back to SQLite for position save", flush=True)
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO options_positions
              (ticker, type, expiry, earnings_date, strike, qty, entry_price,
               stop_loss, target1, target2, target3, status, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            data["ticker"], data["type"], data["expiry"], data.get("earnings_date"),
            data["strike"], data["qty"], data["entry_price"],
            data.get("stop_loss"), data.get("target1"),
            data.get("target2"), data.get("target3"), "Open",
            data.get("notes"),
        ))
    _backup_positions()


def _load_options_positions() -> pd.DataFrame:
    if _has_supabase:
        try:
            rows = _sb_get("options_positions", {"order": "created_at.desc"})
            if rows is not None:  # empty list is valid — Supabase reachable, zero rows
                return pd.DataFrame(rows) if rows else pd.DataFrame()
        except Exception as _e:
            print(f"[SB ERROR] _load_options_positions failed: {_e} — falling back to SQLite", flush=True)
    with get_db_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM options_positions ORDER BY created_at DESC", conn)
    return df


def _backup_positions():
    if _has_supabase:
        return  # Supabase is persistent — local JSON backup not needed
    try:
        df = _load_options_positions()
        rows = df.to_dict(orient="records") if not df.empty else []
        with open(_POSITIONS_BACKUP_PATH, "w") as f:
            json.dump(rows, f, default=str)
    except Exception:
        pass


def _delete_options_position(pos_id: int):
    if _has_supabase:
        try:
            _sb_delete("options_positions", {"id": f"eq.{pos_id}"})
            return
        except Exception:
            pass
    with get_db_connection() as conn:
        conn.execute("DELETE FROM options_positions WHERE id=?", (pos_id,))
    _backup_positions()


def _update_live_option_price(pos_id: int, price) -> None:
    if _has_supabase:
        try:
            _sb_patch("options_positions", {"id": f"eq.{pos_id}"}, {"live_option_price": price})
            return
        except Exception:
            pass
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE options_positions SET live_option_price=? WHERE id=?",
            (price, pos_id),
        )
    _backup_positions()


def _on_live_opt_change(pos_id: int, key: str) -> None:
    val = st.session_state.get(key)
    try:
        price = float(val) if val is not None and float(val) > 0 else None
    except Exception:
        price = None
    _update_live_option_price(pos_id, price)


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


def _score_bar_html(val: int, color: str) -> str:
    return (
        f'<div style="background:#f2f2f2;border-radius:100px;height:5px;overflow:hidden;">'
        f'<div style="background:{color};border-radius:100px;height:5px;width:{val}%;"></div>'
        f'</div>'
    )


# ── persistence helpers ───────────────────────────────────────────────────────

def _save_setting(key: str, value: float) -> None:
    if _has_supabase:
        try:
            # upsert via POST with Prefer: resolution=merge-duplicates
            hdrs = {**_SB_HEADERS, "Prefer": "resolution=merge-duplicates,return=representation"}
            _requests.post(
                _SB_URL + "app_settings",
                headers=hdrs,
                json={"key": key, "value": str(value)},
                timeout=10,
            ).raise_for_status()
            return
        except Exception:
            pass
    path = _BALANCE_PATH if key == "balance" else _BP_PATH
    try:
        with open(path, "w") as f:
            f.write(f"{value:.2f}")
    except Exception:
        pass


def _save_balance(balance: float) -> None:
    _save_setting("balance", balance)


def _save_buying_power(bp: float) -> None:
    _save_setting("buying_power", bp)


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


# ── scenario engine data ──────────────────────────────────────────────────────

_SCENARIOS: dict = {
    "Fed Raises Rates": {
        "summary": "Higher rates pressure growth valuations and expand net interest margins for banks.",
        "sectors": [
            ("Banks & Financials",   "positive", "Net interest margin expands — core business improves"),
            ("Tech / High-Growth",   "negative", "Discount rate rises → valuations compress sharply"),
            ("Crypto & Mining",      "negative", "Risk-off rotation — liquidity tightens, speculative assets dump"),
            ("Real Estate / REITs",  "negative", "Mortgage rates rise, demand falls"),
            ("Telecom (bond proxy)", "negative", "Sells off with rising rates like a bond"),
            ("Auto / Industrials",   "neutral",  "Higher financing costs offset by steady demand"),
        ],
        "vulnerable": ["PLTR", "RBLX", "DKNG", "SQ", "SOFI", "MARA", "RIOT", "COIN", "SNAP"],
        "benefiting": ["BAC", "WFC", "KEY"],
        "action":     "Reduce tech and crypto exposure. Hold or add banks. Avoid high-multiple growth names.",
    },
    "Market Drops 10%": {
        "summary": "Broad risk-off sell-off. High-beta names lead the decline.",
        "sectors": [
            ("Crypto / Bitcoin Mining", "negative", "Highest beta — dumps hardest in risk-off"),
            ("Speculative Tech",        "negative", "High-multiple names correct sharply"),
            ("Banks",                   "negative", "Credit risk fears weigh, less than growth names"),
            ("Auto",                    "negative", "Consumer spending fears hurt cyclicals"),
            ("Telecom (dividend)",      "neutral",  "Defensive dividend offers relative stability"),
        ],
        "vulnerable": ["MARA", "RIOT", "COIN", "RBLX", "DKNG", "HOOD", "SQ", "PLTR"],
        "benefiting": ["T", "KEY"],
        "action":     "Close speculative call positions. Hold dividend payers. Wait for stabilization before new entries.",
    },
    "VIX Spikes Above 35": {
        "summary": "Panic mode. Options premiums explode. Long calls get IV-crushed after the spike.",
        "sectors": [
            ("Options (long calls)",  "negative", "IV crush follows spike — even winning trades can lose value"),
            ("Crypto",                "negative", "Correlates with equity fear — dumps hard"),
            ("Defensive Dividend",    "positive", "Flight to safety benefits stable income names"),
            ("Growth Tech",           "negative", "Risk-off rotation hits growth hardest"),
        ],
        "vulnerable": ["MARA", "RIOT", "COIN", "PLTR", "RBLX", "SNAP", "DKNG"],
        "benefiting": ["T", "BAC", "WFC", "KEY"],
        "action":     "Do NOT buy options when VIX > 35. Close open longs. Wait for VIX to drop below 25.",
    },
    "S&P Breaks 200-Day MA": {
        "summary": "Bear market signal. Momentum reverses. Technical selling accelerates across the board.",
        "sectors": [
            ("All Sectors",         "negative", "200-day break triggers broad technical selling"),
            ("High Beta / Crypto",  "negative", "Speculative names see outsized declines"),
            ("Defensive Dividend",  "neutral",  "Less damage but not immune"),
        ],
        "vulnerable": ["PLTR", "RBLX", "MARA", "RIOT", "COIN", "DKNG", "SQ", "HOOD", "SNAP"],
        "benefiting": [],
        "action":     "Stop all new entries. Tighten stops on existing positions. Close anything above breakeven.",
    },
    "Strong Jobs Report": {
        "summary": "Fed stays hawkish longer. Bond yields rise. Consumer spending outlook improves.",
        "sectors": [
            ("Banks",                   "positive", "Higher-for-longer rates boost net interest margin"),
            ("Consumer Discretionary",  "positive", "Employment strength drives consumer spending"),
            ("Tech / High-Growth",      "negative", "Rate expectations rise, multiples compress"),
            ("Crypto",                  "neutral",  "Mixed — strong economy vs hawkish Fed pressure"),
        ],
        "vulnerable": ["PLTR", "RBLX", "SOFI", "SQ", "SNAP"],
        "benefiting": ["BAC", "WFC", "KEY", "F", "GM", "UBER"],
        "action":     "Hold banks and consumer-facing plays. Reduce high-multiple tech positions.",
    },
    "Inflation Rises": {
        "summary": "Sticky prices keep Fed hawkish. Future cash flows worth less at higher discount rates.",
        "sectors": [
            ("Banks",                 "positive", "Rate hike expectations boost NIM outlook"),
            ("Tech / High P/E",       "negative", "Future cash flows discounted at higher rates"),
            ("Telecom (bond proxy)",  "negative", "Underperforms in rising-rate environment"),
            ("Crypto",                "negative", "Acts risk-off in practice — not a reliable inflation hedge"),
            ("Auto / Industrials",    "neutral",  "Input costs rise but pricing power partially offsets"),
        ],
        "vulnerable": ["PLTR", "RBLX", "SNAP", "DKNG", "T", "MARA", "RIOT", "COIN"],
        "benefiting": ["BAC", "WFC", "KEY", "F", "GM"],
        "action":     "Reduce growth and crypto exposure. Add bank names on dip. Shorten options duration.",
    },
}


# ── session state init ────────────────────────────────────────────────────────

if "wl_tickers" not in st.session_state:
    from data.watchlist import WATCHLIST as _WL_INIT
    st.session_state.wl_tickers = list(_WL_INIT)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

ET = timezone(timedelta(hours=-4))  # EDT (UTC-4 in summer)
now_et = datetime.now(ET)
st.markdown(
    '<div class="page-header">'
    '<span class="logo">📈</span>'
    '<span class="title">Trading Desk</span>'
    f'<span class="timestamp">{now_et.strftime("%A, %b %d %I:%M %p ET")}</span>'
    '</div>',
    unsafe_allow_html=True,
)

# ── ticker strip ─────────────────────────────────────────────────────────────
_strip_data = _fetch_ticker_strip()
_strip_parts = []
for _si in _strip_data:
    _sp = f"${_si['price']:,.2f}" if _si["price"] else "—"
    if _si["pct"] is not None:
        _sc = "#00e676" if _si["pct"] >= 0 else "#ff5252"
        _spct = f'<span style="color:{_sc}">{_si["pct"]:+.2f}%</span>'
    else:
        _spct = '<span style="color:#666">—</span>'
    _strip_parts.append(
        f'<span style="display:inline-flex;align-items:center;gap:7px;">'
        f'<b style="color:#fff;letter-spacing:.03em">{_si["label"]}</b>'
        f'<span style="color:#bbb">{_sp}</span>{_spct}</span>'
    )
st.markdown(
    '<div style="background:#1a1a2e;padding:7px 20px;display:flex;flex-wrap:wrap;'
    'gap:24px;align-items:center;font-size:0.82rem;border-radius:8px;margin-bottom:14px;">'
    + "  ·  ".join(_strip_parts)
    + "</div>",
    unsafe_allow_html=True,
)

# ── challenge tracker (5-card layout) ────────────────────────────────────────

# — pre-compute all values —
_ch_pct_overall = min(1.0, (CHALLENGE_CURRENT - CHALLENGE_START) / (CHALLENGE_GOAL - CHALLENGE_START))
_ch_gain        = CHALLENGE_CURRENT - CHALLENGE_START
_ch_ret_pct     = (_ch_gain / CHALLENGE_START) * 100
_ch_gain_color  = "#00c853" if _ch_gain >= 0 else "#ff1744"
_ch_gain_sign   = "+" if _ch_gain >= 0 else ""
_ch_prog_w      = f"{_ch_pct_overall * 100:.1f}"
_max_trade      = round(BUYING_POWER * 0.50)
_m1_pct         = (400 - CHALLENGE_START) / (CHALLENGE_GOAL - CHALLENGE_START) * 100
_m2_pct         = (700 - CHALLENGE_START) / (CHALLENGE_GOAL - CHALLENGE_START) * 100

_tracker_pos_df = _load_options_positions()
_open_pos_df    = _tracker_pos_df[_tracker_pos_df["status"] == "Open"] if not _tracker_pos_df.empty else pd.DataFrame()
_cap_at_risk    = float((_open_pos_df["entry_price"] * _open_pos_df["qty"] * 100).sum()) if not _open_pos_df.empty else 0.0
_risk_raw       = (_cap_at_risk / CHALLENGE_CURRENT * 100) if CHALLENGE_CURRENT > 0 else 0
_risk_score     = min(100, max(0, round(_risk_raw)))
_risk_label     = "Low" if _risk_score < 30 else ("Moderate" if _risk_score < 60 else "Elevated")
_risk_color     = "#00c853" if _risk_score < 30 else ("#ff9800" if _risk_score < 60 else "#ff1744")

# — position badges for card 4 —
_pos_badges_html = ""
_today_ch = date.today()
for _, _pr in _open_pos_df.iterrows():
    try:
        _exp_d  = datetime.strptime(str(_pr["expiry"]), "%Y-%m-%d").date()
        _dte_ch = (_exp_d - _today_ch).days
        _dte_s  = f"{_dte_ch}d"
    except Exception:
        _dte_s = "?"
    _pos_badges_html += (
        f'<span style="display:inline-flex;align-items:center;gap:4px;background:#1a1a1a;'
        f'color:#fff;border-radius:5px;padding:2px 7px;font-size:0.7rem;font-weight:700;margin:2px 2px 0 0;">'
        f'{_pr["ticker"]} <span style="color:#aaa;font-weight:400;">${float(_pr["strike"]):.0f} · {_dte_s}</span></span>'
    )
if not _pos_badges_html:
    _pos_badges_html = '<span style="color:#aaa;font-size:0.8rem;">No open positions</span>'

# — risk gradient marker bar —
_risk_bar_html = (
    f'<div style="position:relative;background:linear-gradient(to right,#00c853,#ffeb3b,#ff1744);'
    f'border-radius:100px;height:6px;margin-top:8px;">'
    f'<div style="position:absolute;top:-3px;left:calc({_risk_score}% - 6px);'
    f'width:12px;height:12px;background:#fff;border:2px solid {_risk_color};border-radius:50%;box-shadow:0 0 3px rgba(0,0,0,.2);"></div>'
    f'</div>'
)

_ch_tracker_col, _ch_btn_col = st.columns([11, 1])
with _ch_tracker_col:
    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:6px;">'

        # Card 1 — Total Value
        f'<div style="background:#fff;border:1px solid #f0f0f0;border-radius:12px;padding:16px 16px;box-shadow:0 1px 8px rgba(0,0,0,0.06);">'
        f'<div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:.09em;font-weight:600;color:#333;">Total Value</div>'
        f'<div style="font-size:1.9rem;font-weight:900;color:#00c853;line-height:1.1;letter-spacing:-.03em;margin-top:4px;">${CHALLENGE_CURRENT:,.2f}</div>'
        f'<div style="font-size:0.68rem;color:#555;font-weight:500;margin-top:4px;">Started: ${CHALLENGE_START:,.2f} · Goal: ${CHALLENGE_GOAL:,.0f}</div>'
        f'<div style="background:#f2f2f2;border-radius:100px;height:5px;overflow:hidden;margin-top:8px;">'
        f'<div style="background:#00c853;border-radius:100px;height:5px;width:{_ch_prog_w}%;"></div></div>'
        f'<div style="font-size:0.68rem;color:#555;font-weight:500;margin-top:4px;">{_ch_pct_overall*100:.1f}% to goal</div>'
        f'</div>'

        # Card 2 — Total Return
        f'<div style="background:#fff;border:1px solid #f0f0f0;border-radius:12px;padding:16px 16px;box-shadow:0 1px 8px rgba(0,0,0,0.06);">'
        f'<div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:.09em;font-weight:600;color:#333;">Total Return</div>'
        f'<div style="font-size:1.9rem;font-weight:900;color:{_ch_gain_color};line-height:1.1;letter-spacing:-.03em;margin-top:4px;">{_ch_gain_sign}${_ch_gain:,.2f}</div>'
        f'<div style="font-size:0.88rem;font-weight:700;color:{_ch_gain_color};margin-top:4px;">{"▲" if _ch_gain >= 0 else "▼"} {_ch_ret_pct:+.1f}%</div>'
        f'<div style="font-size:0.68rem;color:#555;font-weight:500;margin-top:4px;">From ${CHALLENGE_START:,.2f} baseline</div>'
        f'</div>'

        # Card 3 — Buying Power
        f'<div style="background:#fff;border:1px solid #f0f0f0;border-radius:12px;padding:16px 16px;box-shadow:0 1px 8px rgba(0,0,0,0.06);">'
        f'<div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:.09em;font-weight:600;color:#333;">Buying Power</div>'
        f'<div style="font-size:1.9rem;font-weight:900;color:#1565c0;line-height:1.1;letter-spacing:-.03em;margin-top:4px;">${BUYING_POWER:,.2f}</div>'
        f'<div style="font-size:0.68rem;color:#555;font-weight:500;margin-top:4px;">Available to deploy</div>'
        f'<div style="display:inline-block;background:#e8f5e9;color:#2e7d32;font-size:0.68rem;font-weight:700;border-radius:5px;padding:2px 8px;margin-top:8px;">Max 1 trade: ${_max_trade}</div>'
        f'</div>'

        # Card 4 — Active Positions
        f'<div style="background:#fff;border:1px solid #f0f0f0;border-radius:12px;padding:16px 16px;box-shadow:0 1px 8px rgba(0,0,0,0.06);">'
        f'<div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:.09em;font-weight:600;color:#333;">Active Positions</div>'
        f'<div style="font-size:1.9rem;font-weight:900;color:#000;line-height:1.1;letter-spacing:-.03em;margin-top:4px;">{len(_open_pos_df)}</div>'
        f'<div style="margin-top:6px;display:flex;flex-wrap:wrap;">{_pos_badges_html}</div>'
        f'</div>'

        # Card 5 — Risk Score
        f'<div style="background:#fff;border:1px solid #f0f0f0;border-radius:12px;padding:16px 16px;box-shadow:0 1px 8px rgba(0,0,0,0.06);">'
        f'<div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:.09em;font-weight:600;color:#333;">Risk Score</div>'
        f'<div style="font-size:1.9rem;font-weight:900;color:{_risk_color};line-height:1.1;letter-spacing:-.03em;margin-top:4px;">{_risk_score}/100</div>'
        f'<div style="font-size:0.82rem;font-weight:700;color:{_risk_color};margin-top:2px;">{_risk_label}</div>'
        f'{_risk_bar_html}'
        f'<div style="font-size:0.68rem;color:#555;font-weight:500;margin-top:6px;">Capital at risk: ${_cap_at_risk:,.0f}</div>'
        f'</div>'

        '</div>',
        unsafe_allow_html=True,
    )
with _ch_btn_col:
    st.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)
    if st.button("✏️", key="ch_edit_btn", help="Update balance / buying power", use_container_width=True):
        st.session_state["ch_editing"] = not st.session_state.get("ch_editing", False)

# Full-width goal progress bar — outside column context
st.markdown(
    f'<div style="background:#fff;border:1px solid #f0f0f0;border-radius:12px;'
    f'padding:14px 18px 28px 18px;box-shadow:0 1px 8px rgba(0,0,0,0.06);margin-bottom:6px;">'
    f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;">'
    f'<span style="font-size:0.68rem;color:#aaa;">${CHALLENGE_START:,.2f} start</span>'
    f'<span style="font-size:0.82rem;font-weight:700;color:#00c853;">{_ch_pct_overall*100:.1f}% to goal</span>'
    f'<span style="font-size:0.68rem;color:#aaa;">${CHALLENGE_GOAL:,.0f} goal</span>'
    f'</div>'
    f'<div style="position:relative;">'
    f'<div style="background:#f2f2f2;border-radius:100px;height:10px;overflow:hidden;">'
    f'<div style="background:#00c853;border-radius:100px;height:10px;width:{_ch_prog_w}%;"></div>'
    f'</div>'
    f'<div style="position:absolute;top:0;left:{_m1_pct:.1f}%;width:2px;height:10px;background:#1565c0;"></div>'
    f'<div style="position:absolute;top:0;left:{_m2_pct:.1f}%;width:2px;height:10px;background:#ff9800;"></div>'
    f'<div style="position:absolute;top:14px;left:{_m1_pct:.1f}%;transform:translateX(-50%);'
    f'font-size:0.6rem;color:#1565c0;font-weight:700;white-space:nowrap;">$400 · First Double</div>'
    f'<div style="position:absolute;top:14px;left:{_m2_pct:.1f}%;transform:translateX(-50%);'
    f'font-size:0.6rem;color:#ff9800;font-weight:700;white-space:nowrap;">$700 · Next Milestone</div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)

if st.session_state.get("ch_editing", False):
    _bv1, _bv2, _bv3, _bv4 = st.columns([3, 3, 1, 4])
    _new_bal = _bv1.number_input(
        "Balance ($)",
        value=float(CHALLENGE_CURRENT),
        min_value=0.01,
        step=0.01,
        format="%.2f",
        key="ch_bal_input",
    )
    _new_bp = _bv2.number_input(
        "Buying Power ($)",
        value=float(BUYING_POWER),
        min_value=0.0,
        step=0.01,
        format="%.2f",
        key="ch_bp_input",
    )
    if _bv3.button("Save", key="ch_bal_save", use_container_width=True):
        _save_balance(float(_new_bal))
        _save_buying_power(float(_new_bp))
        st.session_state["ch_editing"] = False
        st.rerun()

# ── session reminder banner ──────────────────────────────────────────────────
_h, _m = now_et.hour, now_et.minute
_now_mins = _h * 60 + _m
if 9 * 60 <= _now_mins < 9 * 60 + 45:
    st.markdown(
        '<div style="background:#e8f5e9;border:1.5px solid #00c853;border-radius:10px;'
        'padding:10px 16px;margin-bottom:10px;font-size:0.9rem;font-weight:600;color:#1b5e20;">'
        '🟢 MORNING SESSION — Check positions, run scanner, set alerts before 9:30 open.'
        '</div>',
        unsafe_allow_html=True,
    )
elif 15 * 60 + 30 <= _now_mins < 16 * 60:
    st.markdown(
        '<div style="background:#fffde7;border:1.5px solid #f9a825;border-radius:10px;'
        'padding:10px 16px;margin-bottom:10px;font-size:0.9rem;font-weight:600;color:#e65100;">'
        '🟡 EOD SESSION — Update SAIL price, update balance, review any open positions before close.'
        '</div>',
        unsafe_allow_html=True,
    )

# ── tabs (Options Desk is first / default) ────────────────────────────────────
tab2, tab1, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "⚡  Options Desk",
    "📊  Signals",
    "🔍  Research",
    "📒  Trade Log",
    "🌍  Market Intel",
    "🎯  Stock Scorer",
    "⚡  Scenario Engine",
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
    od_tab_a, od_tab_b, od_tab_c, od_tab_d = st.tabs(["📋  Active Positions", "🔎  Options Scanner", "🎯  Evaluate a Play", "📊  Sector Watch"])

    # ────────────────────────────────────────────────────────────────────────
    # A — Active Positions
    # ────────────────────────────────────────────────────────────────────────

    with od_tab_a:
        positions = _load_options_positions()

        if not positions.empty:
            open_pos = positions[positions["status"] == "Open"]
            total_invested = float((open_pos["entry_price"] * open_pos["qty"] * 100).sum())
            today = date.today()
            def _earn_within_week(val):
                try:
                    d = (datetime.strptime(str(val), "%Y-%m-%d").date() - today).days
                    return 0 <= d <= 7
                except Exception:
                    return False
            n_warn = sum(
                1 for _, r in open_pos.iterrows()
                if r["earnings_date"] and _earn_within_week(r["earnings_date"])
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
            for _pos_num, (_, row) in enumerate(positions.iterrows(), start=1):
                live = _live_price(row["ticker"])
                live_str = f"${live:.2f}" if live is not None else "—"

                earn_warn = ""
                _hard_sell_banner = ""
                if row["earnings_date"]:
                    try:
                        ed = datetime.strptime(str(row["earnings_date"]), "%Y-%m-%d").date()
                        d = (ed - today).days
                        if 0 <= d <= 7:
                            earn_warn = f'<span class="badge badge-yellow">⚠ Earnings in {d}d</span>'
                            _hard_sell_banner = (
                                f'<div style="background:#fff3e0;border:1.5px solid #ff6d00;border-radius:7px;'
                                f'padding:8px 12px;margin-top:10px;font-size:0.85rem;font-weight:700;color:#bf360c;">'
                                f'⚠️ HARD SELL — Exit by 9:45 AM on {row["earnings_date"]} to avoid IV crush.'
                                f'</div>'
                            )
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
                _notes_val = row.get("notes")
                notes_row = (
                    f'<div style="margin-top:6px;font-size:0.82rem;color:#424242;">📝 {_notes_val}</div>'
                    if _notes_val and str(_notes_val).strip() and str(_notes_val).strip().lower() != "nan"
                    else ""
                )

                # P&L from live option price
                _lop = row.get("live_option_price")
                try:
                    _lop_valid = _lop is not None and str(_lop) not in ("nan", "None", "") and float(_lop) > 0
                except Exception:
                    _lop_valid = False
                if _lop_valid:
                    _lop_f   = float(_lop)
                    _pnl_d   = (_lop_f - float(row["entry_price"])) * int(row["qty"]) * 100
                    _pnl_p   = ((_lop_f - float(row["entry_price"])) / float(row["entry_price"])) * 100
                    _pc      = "#00c853" if _pnl_d >= 0 else "#ff1744"
                    _pnl_d_str = f'<span style="color:{_pc};font-weight:700;">${_pnl_d:+,.2f}</span>'
                    _pnl_p_str = f'<span style="color:{_pc};font-weight:700;">{_pnl_p:+.1f}%</span>'
                else:
                    _pnl_d_str = "—"
                    _pnl_p_str = "—"

                # Estimated Greeks
                _greeks_html = ""
                try:
                    _dte_g = (datetime.strptime(str(row["expiry"]), "%Y-%m-%d").date() - today).days
                    if _dte_g > 0 and live is not None and float(row["strike"]) > 0 and float(row["entry_price"]) > 0:
                        _S      = float(live)
                        _K      = float(row["strike"])
                        _opt_p  = float(row["entry_price"])
                        _iv_est = (_opt_p / _S) * math.sqrt(365 / _dte_g) * 4
                        _delta  = max(0.05, min(0.95, 0.5 + (_S - _K) / (2 * _K)))
                        _theta  = -_opt_p * 0.05 / _dte_g
                        _pop    = _delta * 100
                        _dc     = "#00c853" if _delta > 0.5 else "#888"
                        _pc_pop = "#00c853" if _pop >= 50 else ("#ff9800" if _pop >= 30 else "#ff1744")
                        def _gf(lbl, val, col="#222"):
                            return (
                                f'<div><div style="color:#aaa;font-size:0.6rem;text-transform:uppercase;'
                                f'letter-spacing:.07em;font-weight:600;">{lbl}</div>'
                                f'<div style="color:{col};font-weight:700;font-size:0.88rem;">{val}</div></div>'
                            )
                        _greeks_html = (
                            '<div style="margin-top:12px;border-top:1px solid #f5f5f5;padding-top:10px;">'
                            '<div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:.09em;'
                            'font-weight:600;color:#bbb;margin-bottom:8px;">Greeks (Est.)</div>'
                            '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;">'
                            + _gf("Delta",    f"{_delta:.2f}",              _dc)
                            + _gf("Theta",    f"-${abs(_theta):.3f}/day",   "#ff1744")
                            + _gf("IV (Est)", f"{_iv_est*100:.0f}%",        "#424242")
                            + _gf("P(Profit)",f"{_pop:.0f}%",               _pc_pop)
                            + '</div></div>'
                        )
                except Exception:
                    _greeks_html = ""

                _ew_inline = (" " + earn_warn) if earn_warn else ""
                st.markdown(f"""
<div class="options-card">
  <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
    <span class="ticker">{row['ticker']}</span>
    <span class="badge {type_cls}">{row['type']}</span>
    <span class="badge {status_cls}">{row['status']}</span>{_ew_inline}
    <span style="margin-left:auto;color:#ccc;font-size:0.78rem;">#{_pos_num}</span>
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(115px,1fr));gap:10px;margin-top:14px;">
    {_field("Strike",    f"${row['strike']:.2f}")}
    {_field("Expiry",    row['expiry'])}
    {_field("Qty",       f"{int(row['qty'])} contract{'s' if row['qty']!=1 else ''}")}
    {_field("Entry",     f"${row['entry_price']:.2f}")}
    {_field("Stop Loss", f"${row['stop_loss']:.2f}" if row['stop_loss'] else "—", "#ff1744")}
    {_field("Live $",    live_str)}
  </div>
  <div style="margin-top:10px;">{targets_html}</div>{earn_row}{notes_row}{_hard_sell_banner}{_greeks_html}
</div>
""", unsafe_allow_html=True)

                # Current Option $ input + P&L — visually connected to card
                _lop_key = f"live_opt_{int(row['id'])}"
                _lop_default = float(_lop) if _lop_valid else 0.0
                _inp_col, _pnl_col = st.columns([1, 2])
                with _inp_col:
                    st.markdown(
                        '<div style="background:#e3f2fd;border-radius:8px;padding:6px 10px 2px 10px;'
                        'margin-top:-4px;border:1.5px solid #90caf9;">',
                        unsafe_allow_html=True,
                    )
                    st.number_input(
                        "Current Option $",
                        min_value=0.0,
                        value=_lop_default,
                        step=0.01,
                        format="%.2f",
                        key=_lop_key,
                        on_change=_on_live_opt_change,
                        args=(int(row["id"]), _lop_key),
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                with _pnl_col:
                    if _lop_valid:
                        st.markdown(
                            f'<div style="padding:10px 0 0 8px;">'
                            f'<span style="font-size:1.4rem;font-weight:900;color:{_pc};">'
                            f'${_pnl_d:+,.2f}</span>'
                            f'<span style="font-size:1.05rem;font-weight:700;color:{_pc};margin-left:10px;">'
                            f'{_pnl_p:+.1f}%</span>'
                            f'<span style="font-size:0.75rem;color:#999;margin-left:8px;">P&amp;L</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div style="padding:14px 0 0 8px;color:#bbb;font-size:0.85rem;">'
                            'Enter current option price to see P&amp;L'
                            '</div>',
                            unsafe_allow_html=True,
                        )

                if st.button(f"🗑 Remove #{_pos_num}", key=f"del_{_pos_num}"):
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

            f_pos_notes = st.text_area("Notes (optional)", placeholder="Optional: entry reason, setup notes...", height=80, key="pos_notes_input")

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
                    "notes":         f_pos_notes.strip() or None,
                })
                st.success(f"✅ Saved — {f_ticker.upper()} ${f_strike:.2f} {f_type} exp {f_expiry}")
                st.rerun()

    # ────────────────────────────────────────────────────────────────────────
    # B — Options Scanner
    # ────────────────────────────────────────────────────────────────────────

    with od_tab_b:
        _hide_movers = st.checkbox("🚫 Hide stocks that moved 5%+ at open (already repriced)", value=True)

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

        st.markdown(
            '<p style="color:#666;font-size:0.88rem;margin:0 0 8px 0;">'
            'Momentum plays from watchlist + 200-ticker screener universe — call contracts ≤ $100 total cost.'
            '</p>',
            unsafe_allow_html=True,
        )
        SECTORS = {
            "All": [],
            "Quantum":       ["RGTI","QBTS","QUBT","IONQ"],
            "Defense AI":    ["BBAI","PLTR"],
            "Space":         ["RKLB","ASTS","LUNR"],
            "Chips":         ["AMD","NVDA","MU","SMCI"],
            "Crypto":        ["MARA","RIOT","COIN"],
            "Momentum":      ["SNAP","SOFI","DKNG","TOST","INTC","RBLX"],
            "eVTOL":         ["ACHR","JOBY"],
            "Cybersecurity": ["CRWD","S","PANW"],
        }
        _sector = st.selectbox("Sector filter", list(SECTORS.keys()), key="sector_filter")

        if st.button("↺ Scan", use_container_width=False, key="scan_options"):
            _scan_options_candidates.clear()
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        with st.spinner("Scanning watchlist + screener universe…"):
            candidates = _scan_options_candidates()

        # Filter 1: hide any play with earnings within 7 days
        _today = date.today()
        _after_earn = []
        _earn_hidden = 0
        for c in candidates:
            _ed = _fetch_earnings_date(c["ticker"])
            if _ed is not None and 0 <= (_ed - _today).days <= 7:
                _earn_hidden += 1
            else:
                _after_earn.append((c, _ed))

        # Filter 2: hide plays where stock already moved ≥5% from prev close
        _visible = []
        _move_hidden = 0
        for c, _ed in _after_earn:
            if _hide_movers:
                _live  = _live_price(c["ticker"])
                _pc    = _prev_close(c["ticker"])
                if _live is not None and _pc is not None and _pc != 0:
                    if abs((_live - _pc) / _pc) >= 0.05:
                        _move_hidden += 1
                        continue
            _visible.append((c, _ed))

        # Sector filter
        _sector_tickers = SECTORS.get(_sector, [])
        if _sector_tickers:
            _visible = [(c, ed) for c, ed in _visible if c["ticker"] in _sector_tickers]

        if not _visible:
            _no_msg = (
                'No setups found today. Check back after market open or lower your signal threshold.'
                if not candidates else
                f'All candidates hidden — earnings within 7 days or stock already moved 5%+. '
                'Use the <b>Evaluate a Play</b> tab to check them individually.'
            )
            st.markdown(f'<div class="card"><span style="color:#aaa;">{_no_msg}</span></div>', unsafe_allow_html=True)
        else:
            # ── Top 3 Buy / Watch / Skip ──────────────────────────────────
            _buy   = [c for c, _ in _visible if c["signal"] >= 70][:3]
            _watch = [c for c, _ in _visible if 50 <= c["signal"] < 70][:3]
            _skip  = [c for c, _ in _visible if c["signal"] < 50][:3]

            def _mini_card(c, color, label):
                return f"""<div style="background:#fff;border:1.5px solid {color};border-radius:10px;padding:10px 14px;margin-bottom:8px;">
  <span style="font-weight:800;font-size:1rem;">{c['ticker']}</span>
  <span style="background:{color};color:#fff;border-radius:6px;padding:2px 8px;font-size:0.75rem;margin-left:8px;">{label}</span>
  <span style="float:right;font-weight:700;color:{color};">${c['cost']:.0f}</span>
  <div style="color:#666;font-size:0.78rem;margin-top:4px;">Signal {c['signal']:.0f} · Strike ${c['strike']:.2f} · Exp {c['expiry']}</div>
</div>"""

            st.markdown("### 🎯 Top Plays Today")
            _t1, _t2, _t3 = st.columns(3)
            with _t1:
                st.markdown("**✅ Buy**")
                if _buy:
                    for c in _buy:
                        st.markdown(_mini_card(c, "#00c853", "BUY"), unsafe_allow_html=True)
                else:
                    st.markdown('<div style="color:#aaa;font-size:0.85rem;">No high-confidence plays</div>', unsafe_allow_html=True)
            with _t2:
                st.markdown("**👀 Watch**")
                if _watch:
                    for c in _watch:
                        st.markdown(_mini_card(c, "#ff9800", "WATCH"), unsafe_allow_html=True)
                else:
                    st.markdown('<div style="color:#aaa;font-size:0.85rem;">No watch setups</div>', unsafe_allow_html=True)
            with _t3:
                st.markdown("**⛔ Skip**")
                if _skip:
                    for c in _skip:
                        st.markdown(_mini_card(c, "#ff1744", "SKIP"), unsafe_allow_html=True)
                else:
                    st.markdown('<div style="color:#aaa;font-size:0.85rem;">No skips today</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📋 All Candidates")

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
        if _move_hidden > 0:
            st.caption(
                f"🚫 {_move_hidden} play{'s' if _move_hidden != 1 else ''} hidden — "
                "stock already moved 5%+ at open."
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


    # ────────────────────────────────────────────────────────────────────────
    # D — Sector Watch
    # ────────────────────────────────────────────────────────────────────────

    _SECTOR_WATCH = {
        "⚛️ Quantum":    ["RGTI", "QBTS", "QUBT", "IONQ"],
        "🛡️ Defense/AI": ["BBAI", "PLTR"],
        "🚀 Space":      ["RKLB", "ASTS", "LUNR"],
        "💾 Chips":      ["AMD", "NVDA", "MU"],
        "₿ Crypto":     ["MARA", "RIOT", "COIN"],
        "⚡ Momentum":   ["SOFI", "DKNG", "SNAP", "HOOD"],
    }

    with od_tab_d:
        st.markdown('<p style="font-size:0.95rem;font-weight:700;color:#1a1a1a;margin:0 0 4px;">Sector Watch</p>', unsafe_allow_html=True)
        st.caption("Live prices per sector — IV%, Conviction, Catalyst and Rating are manually updated.")

        with st.expander("📖 How to use Sector Watch with the Options Scanner"):
            st.markdown("""
**Morning workflow (9:15 AM)**

1. **Check Sector Watch first** — scan each sector tab for tickers showing big % moves (green = up, red = down)
2. **Spot the moving sector** — if Quantum is up across the board after an earnings reaction, that's your sector
3. **Switch to Options Scanner** — use the Sector filter dropdown to select that sector
4. **Hit Scan** — the scanner will score only tickers in that sector and show plays under $100
5. **Check earnings dates** — the scanner auto-hides anything within 7 days of earnings
6. **Enter at open or not at all** — post-earnings sector reactions must be bought within the first 30 minutes or the move is gone

**What each column means**
- **Change %** — today's move vs yesterday's close. Anything ±3% is worth watching
- **IV%** — implied volatility (manually updated). Above 80% = elevated risk, above 100% = skip unless intentional
- **Conviction** — your personal confidence level in the setup (manually updated)
- **Catalyst** — the reason the sector is moving (earnings reaction, news, macro)
- **Rating** — your overall grade for the setup (manually updated)

**Key rule** — never chase a ticker that already moved 5%+ at open. The option has repriced and you are buying at the top.
            """)

        _sw_sector_tabs = st.tabs(list(_SECTOR_WATCH.keys()))
        for _sw_tab, (_sw_sector, _sw_tickers) in zip(_sw_sector_tabs, _SECTOR_WATCH.items()):
            with _sw_tab:
                _sw_rows = []
                for _sw_t in _sw_tickers:
                    _sw_price = _live_price(_sw_t)
                    _sw_prev  = _prev_close(_sw_t)
                    if _sw_price and _sw_prev and _sw_prev != 0:
                        _sw_chg = (_sw_price - _sw_prev) / _sw_prev * 100
                        _sw_chg_str = f"{_sw_chg:+.2f}%"
                    else:
                        _sw_chg_str = "—"
                    _sw_rows.append({
                        "Ticker":     _sw_t,
                        "Price":      f"${_sw_price:.2f}" if _sw_price else "—",
                        "Change %":   _sw_chg_str,
                        "IV%":        "—",
                        "Conviction": "—",
                        "Earnings":   "—",
                        "Catalyst":   "—",
                        "Rating":     "—",
                    })
                st.dataframe(
                    pd.DataFrame(_sw_rows),
                    use_container_width=True,
                    hide_index=True,
                )


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
        st.dataframe(sty, use_container_width=True, hide_index=True)  # noqa


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Market Intel
# ═══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown('<p style="font-size:1.1rem;font-weight:700;color:#000;margin:0 0 14px;">Market Intel</p>', unsafe_allow_html=True)

    _mi_today = date.today()
    _fed_date  = date(2026, 6, 17)
    _cpi_date  = date(2026, 5, 13)
    _fed_days  = (_fed_date - _mi_today).days
    _cpi_days  = (_cpi_date - _mi_today).days
    _fed_disp  = f"{_fed_days}d" if _fed_days > 0 else ("Today" if _fed_days == 0 else "Past")
    _cpi_disp  = f"{_cpi_days}d" if _cpi_days > 0 else ("Today" if _cpi_days == 0 else "Past")

    with st.spinner("Loading market data…"):
        _mkt = _fetch_market_data()

    # ── macro calendar ────────────────────────────────────────────────────────
    _mi_c1, _mi_c2, _mi_c3 = st.columns(3)
    with _mi_c1:
        st.markdown(metric_card("Next Fed Meeting", _fed_disp, "June 17–18, 2026"), unsafe_allow_html=True)
    with _mi_c2:
        st.markdown(metric_card("Next CPI Report", _cpi_disp, "May 13, 2026"), unsafe_allow_html=True)
    with _mi_c3:
        _vix = _mkt["vix"]
        if _vix is not None:
            if _vix > 35:
                _vcls, _vlbl = "loss", "⚠ Danger Zone"
            elif _vix > 25:
                _vcls, _vlbl = "loss", "Elevated"
            elif _vix > 20:
                _vcls, _vlbl = "warn", "Cautious"
            else:
                _vcls, _vlbl = "gain", "Calm"
            st.markdown(metric_card("VIX (Fear Index)", f"{_vix:.1f}", _vlbl, _vcls), unsafe_allow_html=True)
        else:
            st.markdown(metric_card("VIX (Fear Index)", "—", "unavailable"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── market mode row ───────────────────────────────────────────────────────
    _mi_c4, _mi_c5, _mi_c6 = st.columns(3)
    with _mi_c4:
        _sp = _mkt["spy_price"]
        _ma = _mkt["spy_ma200"]
        if _sp and _ma:
            _bull  = _sp > _ma
            _mcls  = "gain" if _bull else "loss"
            _mlbl  = "🐂 Bull Market" if _bull else "🐻 Bear Market"
            _dpct  = round((_sp / _ma - 1) * 100, 1)
            _dsub  = f"SPY {'+' if _dpct >= 0 else ''}{_dpct}% vs MA"
            st.markdown(metric_card("SPY vs 200-Day MA", _mlbl, _dsub, _mcls), unsafe_allow_html=True)
        else:
            st.markdown(metric_card("SPY vs 200-Day MA", "—", "unavailable"), unsafe_allow_html=True)
    with _mi_c5:
        _spy_rsi = _mkt["spy_rsi"]
        if _spy_rsi is not None:
            if _spy_rsi > 70:
                _scls, _slbl = "loss", "Extreme Greed"
            elif _spy_rsi > 60:
                _scls, _slbl = "warn", "Greed"
            elif _spy_rsi >= 40:
                _scls, _slbl = "flat", "Neutral"
            elif _spy_rsi >= 30:
                _scls, _slbl = "warn", "Fear"
            else:
                _scls, _slbl = "gain", "Extreme Fear"
            st.markdown(metric_card("Market Sentiment", _slbl, f"SPY RSI {_spy_rsi:.0f}", _scls), unsafe_allow_html=True)
        else:
            st.markdown(metric_card("Market Sentiment", "—", "unavailable"), unsafe_allow_html=True)
    with _mi_c6:
        _sp_disp = f"${_mkt['spy_price']:,.2f}" if _mkt["spy_price"] else "—"
        _ma_disp = f"${_mkt['spy_ma200']:,.2f}" if _mkt["spy_ma200"] else "—"
        st.markdown(metric_card("SPY Price", _sp_disp, f"200-MA: {_ma_disp}"), unsafe_allow_html=True)

    st.divider()

    # ── earnings this week ────────────────────────────────────────────────────
    st.markdown('<p style="font-size:1.0rem;font-weight:700;color:#000;margin:0 0 10px;">Earnings This Week</p>', unsafe_allow_html=True)
    _earn_sched = [
        ("RKLB", "Tonight"), ("MARA", "Tonight"), ("ASTS", "May 11"),
        ("RGTI", "May 11"),  ("ACHR", "May 11"),  ("QBTS", "May 12"), ("OKLO", "May 12"),
    ]
    _mi_wl = set(st.session_state.wl_tickers)
    for _erow in [_earn_sched[i:i+4] for i in range(0, len(_earn_sched), 4)]:
        _ecols = st.columns(len(_erow))
        for _ec, (_etk, _ewhen) in zip(_ecols, _erow):
            with _ec:
                _in_wl    = _etk in _mi_wl
                _tonight  = "Tonight" in _ewhen
                _ebcls    = "badge-red" if _tonight else "badge-yellow"
                _ewtxt    = "Tonight — Avoid!" if _tonight else _ewhen
                _wlbadge  = '<span class="badge badge-blue" style="margin-left:4px;">WL</span>' if _in_wl else ""
                _cborder  = "border-color:#ffcdd2;" if _tonight else ""
                st.markdown(f'<div class="card" style="text-align:center;padding:14px 12px;{_cborder}"><div style="font-size:1.1rem;font-weight:800;color:#000;">{_etk}</div><div style="margin:6px 0;"><span class="badge {_ebcls}">{_ewtxt}</span>{_wlbadge}</div></div>', unsafe_allow_html=True)

    st.divider()

    # ── macro notes ────────────────────────────────────────────────────────────
    st.markdown('<p style="font-size:1.0rem;font-weight:700;color:#000;margin:0 0 10px;">Macro Context</p>', unsafe_allow_html=True)
    st.markdown('<div class="card"><div style="font-size:0.72rem;font-weight:700;color:#424242;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;">Current Environment</div><div style="font-size:0.93rem;color:#000;line-height:1.75;">Fed on hold, watching inflation data closely. Next CPI: <b>May 13</b>. Next Fed meeting: <b>June 17–18</b>. Market is in data-dependent mode — strong jobs or sticky inflation could push the Fed to stay higher for longer. Monitor the SPY 200-day MA as the key bull/bear dividing line. Avoid buying options into any earnings on the schedule above.</div></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Congressional Trades (inside Market Intel)
# ═══════════════════════════════════════════════════════════════════════════════

    st.divider()
    st.markdown('<p style="font-size:1.0rem;font-weight:700;color:#000;margin:0 0 10px;">🏛️ Congressional Trades</p>', unsafe_allow_html=True)
    st.caption("Recent Senate & House stock disclosures via Quiver Quantitative")

    @st.cache_data(ttl=3600)
    def _fetch_congress_trades():
        try:
            import requests as _req
            r = _req.get("https://api.quiverquant.com/beta/live/congresstrading",
                headers={"accept": "application/json"}, timeout=10)
            if r.status_code == 200:
                return r.json()
        except:
            pass
        return []

    _ct_data = _fetch_congress_trades()

    if not _ct_data:
        st.markdown('<div class="card"><span style="color:#aaa;">Congressional trade data unavailable — API may require a free key at quiverquant.com</span></div>', unsafe_allow_html=True)
    else:
        # Filter buttons
        _ct_filter = st.radio("Filter", ["All", "Buys", "Sales", "Senate", "House"],
            horizontal=True, key="ct_filter", label_visibility="collapsed")

        # Your tickers section
        _ct_watchlist = set(st.session_state.get("wl_tickers", []))
        _ct_your = [t for t in _ct_data if t.get("Ticker","").upper() in _ct_watchlist]

        if _ct_your:
            st.markdown("**⭐ Your Tickers** — Congressional trades on stocks you're watching")
            for t in _ct_your[:5]:
                _ct_type = t.get("Transaction","")
                _ct_color = "#00c853" if "Purchase" in _ct_type else "#ff1744"
                _ct_label = "BUY" if "Purchase" in _ct_type else "SALE"
                _ct_chamber = t.get("Chamber", "")
                st.markdown(f"""<div style="background:#fff;border:1.5px solid {_ct_color};border-radius:10px;padding:12px 16px;margin-bottom:8px;">
  <span style="font-weight:800;font-size:1.05rem;">{t.get('Ticker','')}</span>
  <span style="background:{_ct_color};color:#fff;border-radius:6px;padding:2px 8px;font-size:0.75rem;margin-left:8px;">{_ct_label}</span>
  <span style="background:#424242;color:#fff;border-radius:6px;padding:2px 8px;font-size:0.75rem;margin-left:4px;">🏛 {_ct_chamber}</span>
  <div style="color:#333;font-size:0.85rem;margin-top:6px;font-weight:600;">{t.get('Representative','')}</div>
  <div style="color:#666;font-size:0.78rem;margin-top:2px;">
    💰 {t.get('Range', t.get('Amount','—'))} &nbsp;·&nbsp;
    📅 Traded: {t.get('TransactionDate','—')} &nbsp;·&nbsp;
    📋 Disclosed: {t.get('DisclosureDate','—')}
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown("**📋 Full Feed** — All recent Senate & House disclosures")

        _ct_filtered = _ct_data
        if _ct_filter == "Buys":
            _ct_filtered = [t for t in _ct_data if "Purchase" in t.get("Transaction","")]
        elif _ct_filter == "Sales":
            _ct_filtered = [t for t in _ct_data if "Sale" in t.get("Transaction","")]
        elif _ct_filter == "Senate":
            _ct_filtered = [t for t in _ct_data if t.get("Chamber","") == "Senate"]
        elif _ct_filter == "House":
            _ct_filtered = [t for t in _ct_data if t.get("Chamber","") == "House"]

        for t in _ct_filtered[:20]:
            _ct_type = t.get("Transaction","")
            _ct_color = "#00c853" if "Purchase" in _ct_type else "#ff1744"
            _ct_label = "BUY" if "Purchase" in _ct_type else "SALE"
            _ct_chamber = t.get("Chamber", "")
            _ct_in_wl = "⭐ Watched &nbsp;·&nbsp;" if t.get("Ticker","").upper() in _ct_watchlist else ""
            st.markdown(f"""<div style="background:#fff;border:1px solid #e0e0e0;border-radius:10px;padding:12px 16px;margin-bottom:8px;">
  <span style="font-weight:800;font-size:1.05rem;">{t.get('Ticker','')}</span>
  <span style="background:{_ct_color};color:#fff;border-radius:6px;padding:2px 8px;font-size:0.75rem;margin-left:8px;">{_ct_label}</span>
  <span style="background:#424242;color:#fff;border-radius:6px;padding:2px 8px;font-size:0.75rem;margin-left:4px;">🏛 {_ct_chamber}</span>
  <div style="color:#333;font-size:0.85rem;margin-top:6px;font-weight:600;">{t.get('Representative','')}</div>
  <div style="color:#666;font-size:0.78rem;margin-top:2px;">
    {_ct_in_wl}💰 {t.get('Range', t.get('Amount','—'))} &nbsp;·&nbsp;
    📅 Traded: {t.get('TransactionDate','—')} &nbsp;·&nbsp;
    📋 Disclosed: {t.get('DisclosureDate','—')}
  </div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 6 — Stock Scorer
# ═══════════════════════════════════════════════════════════════════════════════

with tab6:
    st.markdown('<p style="font-size:1.1rem;font-weight:700;color:#000;margin:0 0 4px;">Stock Scorer</p>', unsafe_allow_html=True)
    st.caption("Opportunity, risk, and volatility scores derived from signal strength, RSI position, and volume ratio.")

    with st.spinner("Scoring watchlist…"):
        _ss_df = _scored_stocks()

    if _ss_df.empty:
        st.markdown('<div class="card"><span style="color:#aaa;">No data available. Run python3 data/market_data.py first.</span></div>', unsafe_allow_html=True)
    else:
        _ss_rows = []
        for _, _r in _ss_df.iterrows():
            _rsi  = float(_r["RSI"])        if pd.notna(_r["RSI"])        else 50.0
            _sig  = float(_r["Buy Signal"]) if pd.notna(_r["Buy Signal"]) else 0.0
            _vrat = float(_r["Vol Ratio"])  if pd.notna(_r["Vol Ratio"])  else 1.0

            _risk  = min(int(abs(_rsi - 50) * 2), 100)
            _opp   = min(int(_sig), 100)
            _volsc = min(int(_vrat * 25), 100)

            if _opp >= 70 and _risk < 40:
                _rtg, _rcls = "Strong Buy", "badge-green"
            elif _opp >= 55 and _risk < 55:
                _rtg, _rcls = "Buy", "badge-blue"
            elif _opp >= 40:
                _rtg, _rcls = "Hold", "badge-yellow"
            elif _opp >= 25:
                _rtg, _rcls = "Watch", "badge-orange"
            else:
                _rtg, _rcls = "Avoid", "badge-red"

            if _opp >= 60 and _risk < 50:
                _zone, _zcol = "Buy Zone", "#00c853"
            elif _risk >= 60 or _opp < 25:
                _zone, _zcol = "Danger Zone", "#ff1744"
            elif _opp >= 35:
                _zone, _zcol = "Hold Zone", "#e65100"
            else:
                _zone, _zcol = "Watch Zone", "#999"

            _ss_rows.append({
                "ticker": str(_r["Ticker"]),
                "price":  float(_r["Close"]),
                "opp":    _opp,
                "risk":   _risk,
                "volsc":  _volsc,
                "rtg":    _rtg,
                "rcls":   _rcls,
                "zone":   _zone,
                "zcol":   _zcol,
            })

        _rtg_order = {"Strong Buy": 0, "Buy": 1, "Hold": 2, "Watch": 3, "Avoid": 4}
        _ss_rows.sort(key=lambda x: (_rtg_order.get(x["rtg"], 5), -x["opp"]))

        for _si in range(0, len(_ss_rows), 2):
            _pair  = _ss_rows[_si:_si + 2]
            _scols = st.columns(len(_pair))
            for _scorer_col, _sr in zip(_scols, _pair):
                with _scorer_col:
                    _obar = _score_bar_html(_sr["opp"],   "#00c853")
                    _rbar = _score_bar_html(_sr["risk"],  "#ff1744")
                    _vbar = _score_bar_html(_sr["volsc"], "#1565c0")
                    _risk_col = "#ff1744" if _sr["risk"] > 60 else "#000"
                    st.markdown(f"""
<div class="options-card">
  <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:12px;">
    <span class="ticker">{_sr['ticker']}</span>
    <span style="color:#999;font-size:0.85rem;font-weight:600;">${_sr['price']:.2f}</span>
    <span class="badge {_sr['rcls']}" style="margin-left:auto;">{_sr['rtg']}</span>
  </div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px;">
    {_field("Opportunity", str(_sr['opp']) + "/100")}
    {_field("Risk", str(_sr['risk']) + "/100", _risk_col)}
    {_field("Vol Activity", str(_sr['volsc']) + "/100")}
  </div>
  <div style="font-size:0.7rem;color:#999;text-transform:uppercase;font-weight:600;letter-spacing:.05em;margin-bottom:3px;">Opportunity</div>{_obar}
  <div style="font-size:0.7rem;color:#999;text-transform:uppercase;font-weight:600;letter-spacing:.05em;margin:8px 0 3px;">Risk</div>{_rbar}
  <div style="font-size:0.7rem;color:#999;text-transform:uppercase;font-weight:600;letter-spacing:.05em;margin:8px 0 3px;">Volume Activity</div>{_vbar}
  <div style="margin-top:12px;padding:6px 10px;background:#f8f9fa;border-radius:8px;font-size:0.82rem;font-weight:700;color:{_sr['zcol']};text-align:center;">{_sr['zone']}</div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 7 — Scenario Engine
# ═══════════════════════════════════════════════════════════════════════════════

with tab7:
    st.markdown('<p style="font-size:1.1rem;font-weight:700;color:#000;margin:0 0 4px;">Scenario Engine</p>', unsafe_allow_html=True)
    st.caption("Select a macro scenario to see projected sector impacts and which watchlist tickers are affected.")

    _se_scenario = st.selectbox(
        "Select scenario",
        list(_SCENARIOS.keys()),
        key="se_scenario_sel",
        label_visibility="collapsed",
    )

    _sc_data = _SCENARIOS[_se_scenario]
    _se_wl   = set(st.session_state.wl_tickers)

    st.markdown("<br>", unsafe_allow_html=True)

    # Summary
    st.markdown(f'<div class="card"><div style="font-size:0.72rem;font-weight:700;color:#424242;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">Scenario</div><div style="font-size:1.0rem;font-weight:700;color:#000;margin-bottom:6px;">{_se_scenario}</div><div style="font-size:0.92rem;color:#424242;line-height:1.7;">{_sc_data["summary"]}</div></div>', unsafe_allow_html=True)

    # Sector impact table
    st.markdown('<p style="font-size:0.95rem;font-weight:700;color:#000;margin:14px 0 8px;">Sector Impact</p>', unsafe_allow_html=True)
    _sec_html = ""
    for _sname, _simpact, _snote in _sc_data["sectors"]:
        if _simpact == "positive":
            _scolor, _sicon = "#00c853", "↑ Positive"
        elif _simpact == "negative":
            _scolor, _sicon = "#ff1744", "↓ Negative"
        else:
            _scolor, _sicon = "#424242", "→ Neutral"
        _sec_html += (
            f'<div style="display:grid;grid-template-columns:180px 100px 1fr;align-items:start;'
            f'gap:12px;padding:9px 0;border-bottom:1px solid #f2f2f2;">'
            f'<div style="font-weight:700;color:#000;font-size:0.87rem;">{_sname}</div>'
            f'<div style="font-weight:700;color:{_scolor};font-size:0.85rem;">{_sicon}</div>'
            f'<div style="color:#424242;font-size:0.82rem;line-height:1.5;">{_snote}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="card" style="padding:16px 22px;">{_sec_html}</div>', unsafe_allow_html=True)

    # Vulnerable / Benefiting columns
    _se_vc, _se_bc = st.columns(2)

    with _se_vc:
        st.markdown('<p style="font-size:0.9rem;font-weight:700;color:#ff1744;margin:14px 0 8px;">⚠ Vulnerable Tickers</p>', unsafe_allow_html=True)
        _vuln = _sc_data["vulnerable"]
        if _vuln:
            _vhtml = ""
            for _vt in sorted(_vuln, key=lambda t: (0 if t in _se_wl else 1, t)):
                _vcls = "badge-red" if _vt in _se_wl else "badge-yellow"
                _vsuf = " ★" if _vt in _se_wl else ""
                _vhtml += f'<span class="badge {_vcls}" style="margin:3px 4px 3px 0;">{_vt}{_vsuf}</span>'
            st.markdown(f'<div class="card" style="padding:14px 18px;">{_vhtml}<div style="margin-top:10px;font-size:0.72rem;color:#999;">★ on your watchlist</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><span style="color:#aaa;font-size:0.85rem;">None identified for this scenario.</span></div>', unsafe_allow_html=True)

    with _se_bc:
        st.markdown('<p style="font-size:0.9rem;font-weight:700;color:#00c853;margin:14px 0 8px;">✓ Potential Beneficiaries</p>', unsafe_allow_html=True)
        _bene = _sc_data["benefiting"]
        if _bene:
            _bhtml = ""
            for _bt in sorted(_bene, key=lambda t: (0 if t in _se_wl else 1, t)):
                _bcls = "badge-green" if _bt in _se_wl else "badge-blue"
                _bsuf = " ★" if _bt in _se_wl else ""
                _bhtml += f'<span class="badge {_bcls}" style="margin:3px 4px 3px 0;">{_bt}{_bsuf}</span>'
            st.markdown(f'<div class="card" style="padding:14px 18px;">{_bhtml}<div style="margin-top:10px;font-size:0.72rem;color:#999;">★ on your watchlist</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><span style="color:#aaa;font-size:0.85rem;">None identified — consider going defensive or cash.</span></div>', unsafe_allow_html=True)

    # Suggested action
    st.markdown('<p style="font-size:0.95rem;font-weight:700;color:#000;margin:14px 0 8px;">Suggested Action</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="card" style="background:#f8f9fa;border-left:4px solid #000;"><div style="font-size:0.93rem;color:#000;font-weight:600;line-height:1.7;">{_sc_data["action"]}</div></div>', unsafe_allow_html=True)
