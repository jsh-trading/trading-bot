"""
research/analyst.py

AI-powered stock research engine.

Fetches 3 years of financial data using yfinance (revenue, margins, FCF,
debt/equity, PE, PS), formats it, then sends a structured prompt to Claude
(claude-opus-4-7 with adaptive thinking) and streams the full research brief.

If FMP_API_KEY is set and the plan supports it, FMP is tried first; yfinance
is used as the primary/fallback source.

Usage:
    python3 research/analyst.py NVDA          # full 9-section deep dive
    python3 research/analyst.py NVDA short    # sections 2-8 only
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import warnings
import logging
warnings.filterwarnings("ignore")
for _log in ("yfinance", "yfinance.base", "urllib3"):
    logging.getLogger(_log).setLevel(logging.CRITICAL)

import numpy as np
import yfinance as yf
import anthropic

from research.prompts import DEEP_DIVE, SHORT_VERSION


# ── env loader ────────────────────────────────────────────────────────────────

def _load_env():
    """Parse .env from the project root into os.environ (no dotenv dependency)."""
    env_path = os.path.join(_PROJECT_ROOT, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

_load_env()


# ── financial data via yfinance ───────────────────────────────────────────────

def _safe(val, scale=1, fmt=".2f"):
    try:
        v = float(val)
        if np.isnan(v):
            return "n/a"
        return format(v / scale, fmt)
    except (TypeError, ValueError):
        return "n/a"


def fetch_financials(ticker: str) -> tuple[str, str]:
    """
    Pull key financials for *ticker* via yfinance.

    Returns (company_name, formatted_data_string).
    """
    ticker = ticker.upper()
    t = yf.Ticker(ticker)

    info = t.info or {}
    company = info.get("longName") or info.get("shortName") or ticker

    lines = [f"Financial data for {ticker} ({company}) — last 3 fiscal years\n"]

    # ── Income statement (annual) ─────────────────────────────────────────────
    try:
        inc = t.financials   # columns = fiscal year end dates, rows = line items
        if inc is not None and not inc.empty:
            cols = inc.columns[:3]   # most recent 3 years
            lines.append("Income Statement:")
            for col in cols:
                yr  = str(col)[:4]
                rev = inc.loc["Total Revenue", col]         if "Total Revenue"         in inc.index else None
                gp  = inc.loc["Gross Profit", col]          if "Gross Profit"          in inc.index else None
                oi  = inc.loc["Operating Income", col]      if "Operating Income"      in inc.index else None
                gm  = (float(gp) / float(rev)) if gp is not None and rev else None
                om  = (float(oi) / float(rev)) if oi is not None and rev else None
                lines.append(
                    f"  {yr}  Revenue: ${_safe(rev, 1e9)}B"
                    f"  |  Gross margin: {_safe(gm, fmt='.1%') if gm is not None else 'n/a'}"
                    f"  |  Operating margin: {_safe(om, fmt='.1%') if om is not None else 'n/a'}"
                )
    except Exception:
        pass

    # ── Cash flow (annual) ────────────────────────────────────────────────────
    try:
        cf = t.cashflow
        if cf is not None and not cf.empty:
            cols = cf.columns[:3]
            lines.append("Free Cash Flow:")
            for col in cols:
                yr  = str(col)[:4]
                fcf = cf.loc["Free Cash Flow", col] if "Free Cash Flow" in cf.index else None
                lines.append(f"  {yr}  FCF: ${_safe(fcf, 1e9)}B")
    except Exception:
        pass

    # ── Valuation & leverage from info ────────────────────────────────────────
    pe   = info.get("trailingPE") or info.get("forwardPE")
    ps   = info.get("priceToSalesTrailing12Months")
    de   = info.get("debtToEquity")
    if pe or ps or de:
        lines.append("Valuation & Leverage (current):")
        pe_str = f"{float(pe):.1f}x" if pe else "n/a"
        ps_str = f"{float(ps):.1f}x" if ps else "n/a"
        de_str = f"{float(de)/100:.2f}" if de else "n/a"   # yfinance reports D/E as %
        lines.append(f"  PE: {pe_str}  |  PS: {ps_str}  |  D/E: {de_str}")

    if len(lines) == 1:
        lines.append("  [No financial data available for this ticker]")

    return company, "\n".join(lines)


# ── Claude streaming call ─────────────────────────────────────────────────────

def run_analysis(ticker: str, short: bool = False):
    ticker  = ticker.upper()
    company, fin_data = fetch_financials(ticker)

    template = SHORT_VERSION if short else DEEP_DIVE
    prompt   = (
        template
        .replace("[TICKER]",        ticker)
        .replace("[COMPANY]",       company)
        .replace("[FINANCIAL_DATA]", fin_data)
    )

    mode_label = "SHORT VERSION (sections 2-8)" if short else "DEEP DIVE (full 9 sections)"
    W = 66
    print("=" * W)
    print(f"  RESEARCH REPORT — {ticker}  ({company})")
    print(f"  Mode: {mode_label}")
    print("=" * W)
    print()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

    print("\n")
    print("=" * W)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: python3 research/analyst.py <TICKER> [short]")
        sys.exit(1)

    ticker_arg = args[0]
    short_mode = len(args) > 1 and args[1].lower() == "short"
    run_analysis(ticker_arg, short=short_mode)
