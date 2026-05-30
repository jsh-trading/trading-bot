--- /tmp/trading-bot/dashboard/app.py	2026-05-30 18:30:40.989002082 +0000
+++ /sessions/nifty-pensive-mccarthy/mnt/outputs/app.py	2026-05-30 18:30:19.015684118 +0000
@@ -857,6 +857,61 @@
         print(f"[SEED WARN] watchlist seed failed: {_e}", flush=True)
 
 
+# ── Ticker → sector classifier ────────────────────────────────────────────────
+# When the auto-append job adds a brand-new ticker, this picks which sector
+# bucket it lands in. Manual mapping first (the user's known watchlist names),
+# yfinance industry heuristic second, ⚡ Momentum as catch-all.
+
+_TICKER_SECTOR_MAP = {
+    # ⚛️ Quantum
+    "RGTI": "⚛️ Quantum", "QBTS": "⚛️ Quantum", "QUBT": "⚛️ Quantum", "IONQ": "⚛️ Quantum",
+    # 🛡️ Defense / AI
+    "BBAI": "🛡️ Defense/AI", "PLTR": "🛡️ Defense/AI",
+    # 🚀 Space (incl. eVTOL)
+    "RKLB": "🚀 Space", "ASTS": "🚀 Space", "LUNR": "🚀 Space",
+    "ACHR": "🚀 Space", "JOBY": "🚀 Space",
+    # 💾 Chips
+    "AMD": "💾 Chips", "NVDA": "💾 Chips", "MU": "💾 Chips", "SMCI": "💾 Chips",
+    "INTC": "💾 Chips", "TSM": "💾 Chips", "AVGO": "💾 Chips",
+    # ₿ Crypto
+    "MARA": "₿ Crypto", "RIOT": "₿ Crypto", "COIN": "₿ Crypto", "HOOD": "₿ Crypto",
+    # ⚡ Momentum (general)
+    "SOFI": "⚡ Momentum", "DKNG": "⚡ Momentum", "SNAP": "⚡ Momentum",
+    "RBLX": "⚡ Momentum", "TOST": "⚡ Momentum", "UBER": "⚡ Momentum",
+}
+
+
+@st.cache_data(ttl=86400, show_spinner=False)  # 24h — industry classification rarely changes
+def _classify_ticker(ticker: str) -> str:
+    """Pick the Sector Watch bucket for a ticker. Returns one of the keys in
+    _SECTOR_WATCH_DEFAULTS. Falls back to ⚡ Momentum (the project's catch-all).
+    """
+    t = (ticker or "").strip().upper()
+    if not t:
+        return "⚡ Momentum"
+    # 1) Known mapping wins
+    if t in _TICKER_SECTOR_MAP:
+        return _TICKER_SECTOR_MAP[t]
+    # 2) yfinance industry heuristic
+    try:
+        info = yf.Ticker(t).info or {}
+        industry = (info.get("industry") or "").lower()
+        sector   = (info.get("sector")   or "").lower()
+        haystack = industry + " " + sector
+        if "semiconductor" in haystack or "chip" in haystack:
+            return "💾 Chips"
+        if "aerospace" in haystack or "space" in haystack:
+            return "🚀 Space"
+        if "defense" in haystack or "military" in haystack:
+            return "🛡️ Defense/AI"
+        if "cryptocurrency" in haystack or "bitcoin" in haystack or "blockchain" in haystack:
+            return "₿ Crypto"
+    except Exception as _e:
+        print(f"[YF WARN] _classify_ticker({t}) industry lookup failed: {_e}", flush=True)
+    # 3) Fallback
+    return "⚡ Momentum"
+
+
 def _update_options_position(pos_id: int, data: dict) -> None:
     """Update editable fields on an existing position. Mirrors _save_options_position
     for the user-editable columns (status/created_at/live_option_price untouched)."""
@@ -1303,6 +1358,59 @@
     return candidates[:15]
 
 
+# ── Auto-append job (Bundle B2) ───────────────────────────────────────────────
+# Runs from the bootstrap on first render after 9:30 AM and 10:30 AM ET each
+# weekday, and on demand from the "🔄 Run auto-pick now" button.
+
+def _run_auto_append(force: bool = False, slot: str = "manual") -> list:
+    """Find qualifying candidates and add new ones to the watchlist + their
+    classified sector. Returns a list of (ticker, sector) tuples actually
+    added. Empty list if nothing qualified or everything was already there.
+
+    Inclusion criteria (per user's design answers 2026-05-11):
+      - Signal ≥ 70
+      - Optionable on a major exchange (handled by scanner pipeline)
+      - Contract cost ≤ $100
+    """
+    added: list = []
+    try:
+        candidates = _scan_options_candidates() or []
+        existing_wl = set(_load_watchlist_from_db())
+        existing_sectors = _load_sector_tickers() or {}
+        existing_in_sectors = set()
+        for _tickers in existing_sectors.values():
+            existing_in_sectors.update(_tickers)
+
+        for c in candidates:
+            sig  = float(c.get("signal", 0) or 0)
+            cost = float(c.get("cost",   999) or 999)
+            tk   = (c.get("ticker", "") or "").strip().upper()
+            if not tk:
+                continue
+            if sig < 70:
+                continue
+            if cost > 100:
+                continue
+            # Already in BOTH watchlist and a sector → nothing to do
+            if tk in existing_wl and tk in existing_in_sectors:
+                continue
+            sector = _classify_ticker(tk)
+            new_to_wl     = tk not in existing_wl
+            new_to_sector = tk not in existing_in_sectors
+            if new_to_wl:
+                _add_to_watchlist(tk, source="auto")
+                existing_wl.add(tk)
+            if new_to_sector:
+                _add_sector_ticker(sector, tk)
+                existing_in_sectors.add(tk)
+                added.append((tk, sector))
+        if added:
+            print(f"[AUTO {slot}] added {len(added)}: {added}", flush=True)
+    except Exception as _e:
+        print(f"[AUTO ERROR] _run_auto_append({slot}) failed: {_e}", flush=True)
+    return added
+
+
 # ── styling helpers ───────────────────────────────────────────────────────────
 
 def _style_signals_table(row):
@@ -1391,6 +1499,36 @@
         pass
 
 
+# ── String variants ─────────────────────────────────────────────────────
+# Same Supabase table, but no float cast on load and no float format on save.
+# Used by the daily auto-append job to persist last-run dates (e.g. "2026-05-30").
+
+def _load_setting_str(key: str, default: str = "") -> str:
+    if _has_supabase:
+        try:
+            rows = _sb_get("app_settings", {"key": f"eq.{key}", "select": "value"})
+            if rows:
+                return str(rows[0]["value"])
+        except Exception as _e:
+            print(f"[SB WARN] _load_setting_str({key}) failed: {_e}", flush=True)
+    return default
+
+
+def _save_setting_str(key: str, value: str) -> bool:
+    if not _has_supabase:
+        return False
+    try:
+        try:
+            _sb_delete("app_settings", {"key": f"eq.{key}"})
+        except Exception:
+            pass
+        _sb_post("app_settings", {"key": key, "value": str(value)})
+        return True
+    except Exception as _e:
+        print(f"[SB ERROR] _save_setting_str({key}={value}) failed: {_e}", flush=True)
+        return False
+
+
 def _save_balance(balance: float) -> None:
     _save_setting("balance", balance)
 
@@ -1516,6 +1654,37 @@
     unsafe_allow_html=True,
 )
 
+# ── Auto-append time-based trigger (Bundle B2) ───────────────────────────────
+# Runs once per slot per weekday on the user's first visit after the slot time.
+# Slot timestamps persist in app_settings so subsequent visits no-op.
+try:
+    if now_et.weekday() < 5:  # Mon=0 ... Fri=4 only
+        _today_str = now_et.strftime("%Y-%m-%d")
+        _open_dt   = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
+        _settle_dt = now_et.replace(hour=10, minute=30, second=0, microsecond=0)
+
+        def _maybe_auto_append(_slot_name: str, _slot_dt, _setting_key: str):
+            if now_et < _slot_dt:
+                return
+            _last = _load_setting_str(_setting_key, "")
+            if _last == _today_str:
+                return
+            _added = _run_auto_append(force=True, slot=_slot_name)
+            _save_setting_str(_setting_key, _today_str)
+            if _added:
+                _msg = ", ".join(f"{_t}→{_s}" for _t, _s in _added[:5])
+                if len(_added) > 5:
+                    _msg += f" (+{len(_added) - 5} more)"
+                try:
+                    st.toast(f"🤖 Auto-added {len(_added)} ({_slot_name}): {_msg}", icon="✨")
+                except Exception:
+                    pass
+
+        _maybe_auto_append("9:30 AM open",     _open_dt,   "last_auto_append_open")
+        _maybe_auto_append("10:30 AM settle",  _settle_dt, "last_auto_append_settle")
+except Exception as _e:
+    print(f"[BOOT WARN] auto-append trigger failed: {_e}", flush=True)
+
 # ── ticker strip ─────────────────────────────────────────────────────────────
 _strip_data = _fetch_ticker_strip()
 _strip_parts = []
@@ -2704,6 +2873,49 @@
         st.markdown('<p style="font-size:0.95rem;font-weight:700;color:#1a1a1a;margin:0 0 4px;">Sector Watch</p>', unsafe_allow_html=True)
         st.caption("Live prices per sector — IV%, Conviction, Catalyst and Rating are manually updated.")
 
+        # ── Bot Auto-Pick Status panel ─────────────────────────────────────
+        # Shows last run, recent auto-adds, and a manual refresh button.
+        _last_open    = _load_setting_str("last_auto_append_open",   "")
+        _last_settle  = _load_setting_str("last_auto_append_settle", "")
+        _wl_meta      = _load_watchlist_meta()
+        _auto_rows    = sorted(
+            [(t, m.get("added_at", "")) for t, m in _wl_meta.items() if m.get("source") == "auto"],
+            key=lambda x: x[1] or "",
+            reverse=True,
+        )
+        _recent_auto  = [t for t, _ in _auto_rows[:5]]
+
+        _st_col, _btn_col = st.columns([5, 1])
+        with _st_col:
+            _parts = []
+            if _last_open:
+                _parts.append(f"open {_last_open}")
+            if _last_settle:
+                _parts.append(f"settle {_last_settle}")
+            if _parts:
+                _last_txt = "Last auto-pick: " + " · ".join(_parts)
+            else:
+                _last_txt = "No auto-picks yet — runs 9:30 AM & 10:30 AM ET weekdays."
+            if _recent_auto:
+                _last_txt += f"  ·  Recently auto-added: {', '.join(_recent_auto)}"
+            st.markdown(
+                f'<div style="background:#f5f8ff;border:1px solid #d6e4ff;border-radius:8px;'
+                f'padding:8px 12px;font-size:0.82rem;color:#1565c0;">🤖 {_last_txt}</div>',
+                unsafe_allow_html=True,
+            )
+        with _btn_col:
+            if st.button("🔄 Run now", key="manual_auto_pick", use_container_width=True,
+                         help="Force an auto-pick run regardless of time-of-day. Bypasses the 9:30 / 10:30 schedule."):
+                _added_now = _run_auto_append(force=True, slot="manual")
+                if _added_now:
+                    _msg = ", ".join(f"{_t}→{_s}" for _t, _s in _added_now[:5])
+                    if len(_added_now) > 5:
+                        _msg += f" (+{len(_added_now) - 5} more)"
+                    st.toast(f"🤖 Added {len(_added_now)}: {_msg}", icon="✨")
+                else:
+                    st.toast("No new qualifying plays right now.", icon="🔍")
+                st.rerun()
+
         with st.expander("📖 How to use Sector Watch with the Options Scanner"):
             st.markdown("""
 **Morning workflow (9:15 AM)**
