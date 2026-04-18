"""
TheStrat Scanner — Streamlit web app.

Run with:  streamlit run app.py
"""

import base64
import json
import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

from data_fetcher import compute_ftfc, compute_tfc_score, fetch_batch
from signals import classify_bar, compute_volume_ratio, scan_dataframe

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TheStrat Scanner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] { min-width: 280px; max-width: 320px; }
        .stDataFrame thead th { background-color: #1e2329 !important; }
        div[data-testid="metric-container"] { background: #1e2329; border-radius: 8px; padding: 8px 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Ticker persistence
# ---------------------------------------------------------------------------
TICKERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tickers.json")

ALL_SIGNALS    = ["Shooter", "Hammer", "2U Red", "2D Green", "Inside Bar", "Double Inside (1-1)", "Outside Bar", "RevStrat 1-2U", "RevStrat 1-2D", "3-2U", "3-2D"]
ALL_TIMEFRAMES = ["Daily", "Weekly", "Monthly", "Quarterly"]

# Higher timeframe for each scanning timeframe — used for target levels
HTF_MAP: dict[str, str | None] = {
    "Daily":     "Weekly",
    "2-Day":     "Weekly",
    "Weekly":    "Monthly",
    "Monthly":   "Quarterly",
    "Quarterly": None,
}

# Mapping from yfinance sector strings → our internal sector names
_YF_SECTOR_MAP = {
    "Technology":             "Technology",
    "Financial Services":     "Financials",
    "Healthcare":             "Healthcare",
    "Consumer Cyclical":      "Consumer Discretionary",
    "Consumer Defensive":     "Consumer Staples",
    "Energy":                 "Energy",
    "Industrials":            "Industrials",
    "Basic Materials":        "Materials",
    "Communication Services": "Communication",
    "Utilities":              "Utilities",
    "Real Estate":            "Real Estate",
}


def lookup_sector(ticker: str) -> str:
    """Fetch sector for a ticker via yfinance. Returns 'Other' on failure."""
    try:
        info = yf.Ticker(ticker).info
        yf_sector = info.get("sector", "")
        return _YF_SECTOR_MAP.get(yf_sector, "Other")
    except Exception:
        return "Other"


def get_effective_sector_map(tickers_data: dict) -> dict:
    """Merge hardcoded SECTOR_MAP with any custom sectors stored in tickers.json."""
    return {**SECTOR_MAP, **tickers_data.get("sectors", {})}

# ---------------------------------------------------------------------------
# Sector map — used to add a Sector column to scan results
# ---------------------------------------------------------------------------
SECTOR_MAP: dict[str, str] = {
    # --- Broad Market ETFs ---
    "SPY": "Broad Market", "QQQ": "Broad Market", "IWM": "Broad Market",

    # --- Sector ETFs ---
    "XLK": "Technology",   "SMH": "Semiconductors",
    "XLF": "Financials",   "XLE": "Energy",
    "XLV": "Healthcare",   "XBI": "Biotech",
    "XLI": "Industrials",  "XLP": "Consumer Staples",
    "XLU": "Utilities",    "XHB": "Homebuilders",
    "XME": "Metals & Mining", "OIH": "Oil Services",
    "GDX": "Gold Miners",  "JETS": "Airlines",

    # --- Mega-cap Tech ---
    "AAPL": "Technology",  "MSFT": "Technology",
    "AMZN": "Technology",  "GOOGL": "Technology",
    "META": "Technology",

    # --- Semiconductors ---
    "NVDA": "Semiconductors", "AMD": "Semiconductors",
    "AVGO": "Semiconductors", "INTC": "Semiconductors",
    "MRVL": "Semiconductors", "QCOM": "Semiconductors",
    "TSM":  "Semiconductors", "ARM":  "Semiconductors",
    "SMCI": "Semiconductors", "MU":   "Semiconductors",

    # --- Communication / Streaming ---
    "NFLX": "Communication", "DIS": "Communication",
    "WBD":  "Communication", "PINS": "Communication",
    "T":    "Communication",

    # --- Financials ---
    "JPM": "Financials", "BAC": "Financials", "GS":   "Financials",
    "MS":  "Financials", "WFC": "Financials", "C":    "Financials",
    "V":   "Financials", "MA":  "Financials", "SOFI": "Financials",
    "PYPL": "Financials", "HOOD": "Financials",

    # --- Crypto ---
    "COIN": "Crypto", "MARA": "Crypto", "MSTR": "Crypto", "IREN": "Crypto",

    # --- Energy ---
    "XOM": "Energy", "CVX": "Energy", "OXY": "Energy",
    "SLB": "Energy", "HAL": "Energy", "RIG": "Energy", "DVN": "Energy",

    # --- Airlines ---
    "AAL": "Airlines", "DAL": "Airlines", "UAL": "Airlines",

    # --- EV / Auto ---
    "RIVN": "EV / Auto", "F": "EV / Auto", "UBER": "Transportation",

    # --- Industrials ---
    "GE": "Industrials", "BA": "Industrials",

    # --- Defense ---
    "LMT": "Defense", "NOC": "Defense", "GD": "Defense", "DFEN": "Defense",

    # --- Healthcare ---
    "LLY": "Healthcare", "PFE": "Healthcare",
    "UNH": "Healthcare", "JNJ": "Healthcare", "ONDS": "Healthcare",

    # --- Consumer Staples ---
    "WMT": "Consumer Staples", "COST": "Consumer Staples",

    # --- Consumer Discretionary ---
    "HD":   "Consumer Discretionary", "LOW":  "Consumer Discretionary",
    "TGT":  "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "NKE":  "Consumer Discretionary", "ABNB": "Consumer Discretionary",
    "DKNG": "Consumer Discretionary", "CCL":  "Consumer Discretionary",

    # --- Software / Cloud ---
    "PLTR": "Software",  "SHOP": "Software",  "CRM":  "Software",
    "ADBE": "Software",  "ORCL": "Software",  "SNOW": "Software",
    "ZM":   "Software",  "U":    "Software",

    # --- Cybersecurity ---
    "PANW": "Cybersecurity", "CRWD": "Cybersecurity", "CSCO": "Cybersecurity",

    # --- Utilities ---
    "NEE": "Utilities",

    # --- Commodities ---
    "GLD": "Commodities", "SLV": "Commodities",

    # --- TSX: Financials ---
    "RY.TO": "Financials", "TD.TO": "Financials", "BNS.TO": "Financials",
    "BMO.TO": "Financials", "CM.TO": "Financials",

    # --- TSX: Energy ---
    "ENB.TO": "Energy", "CNQ.TO": "Energy", "SU.TO": "Energy", "CVE.TO": "Energy",

    # --- TSX: Industrials ---
    "CNR.TO": "Industrials", "CP.TO": "Industrials",

    # --- TSX: Technology ---
    "SHOP.TO": "Technology", "ATD.TO": "Consumer Discretionary",

    # --- TSX: Materials / Gold ---
    "WPM.TO": "Gold & Silver", "ABX.TO": "Gold & Silver", "AEM.TO": "Gold & Silver",
}


def to_tradingview_format(ticker: str) -> str:
    """Convert yfinance ticker format to TradingView format for watchlist pasting."""
    if ticker.endswith(".TO"):
        return "TSX:" + ticker[:-3]
    if ticker.endswith(".V"):
        return "TSXV:" + ticker[:-2]
    return ticker


def _use_github() -> bool:
    """True when running on Streamlit Cloud with GitHub secrets configured."""
    return "github" in st.secrets


def _gh_headers() -> dict:
    return {
        "Authorization": f"token {st.secrets['github']['token']}",
        "Accept": "application/vnd.github.v3+json",
    }


def _gh_url() -> str:
    owner = st.secrets["github"]["owner"]
    repo  = st.secrets["github"]["repo"]
    return f"https://api.github.com/repos/{owner}/{repo}/contents/tickers.json"


def load_tickers() -> dict:
    if _use_github():
        resp = requests.get(_gh_url(), headers=_gh_headers())
        resp.raise_for_status()
        return json.loads(base64.b64decode(resp.json()["content"]).decode())
    with open(TICKERS_FILE, "r") as f:
        return json.load(f)


def save_tickers(data: dict, message: str = "Update tickers") -> None:
    if _use_github():
        # Need the current file's SHA to update it
        resp = requests.get(_gh_url(), headers=_gh_headers())
        resp.raise_for_status()
        sha = resp.json()["sha"]
        payload = {
            "message": message,
            "content": base64.b64encode(
                json.dumps(data, indent=2).encode()
            ).decode(),
            "sha": sha,
        }
        requests.put(_gh_url(), headers=_gh_headers(), json=payload).raise_for_status()
    else:
        with open(TICKERS_FILE, "w") as f:
            json.dump(data, f, indent=2)


def flat_list(data: dict) -> list[str]:
    return (
        data.get("etfs", []) +
        data.get("stocks", []) +
        data.get("tsx", []) +
        data.get("custom", [])
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 TheStrat Scanner")
    st.divider()

    # -- Timeframes --
    st.subheader("Timeframes")
    tf_checks = {
        "Daily":     st.checkbox("Daily",     value=True),
        "2-Day":     st.checkbox("2-Day",     value=False),
        "Weekly":    st.checkbox("Weekly",    value=True),
        "Monthly":   st.checkbox("Monthly",   value=False),
        "Quarterly": st.checkbox("Quarterly", value=False),
    }
    selected_tfs = [tf for tf, on in tf_checks.items() if on]

    st.divider()

    # -- Signal filters --
    st.subheader("Signal Filters")
    selected_signals = st.multiselect(
        "Show signals:",
        ALL_SIGNALS,
        default=ALL_SIGNALS,
        label_visibility="collapsed",
    )
    direction_filter = st.radio(
        "Direction:",
        ["All", "Bullish", "Bearish", "Neutral", "FTFC Bullish", "FTFC Bearish"],
        horizontal=True,
    )

    st.divider()

    # -- Ticker management --
    st.subheader("Tickers")
    tickers_data = load_tickers()
    all_t = flat_list(tickers_data)

    _eff_map    = get_effective_sector_map(tickers_data)
    all_sectors = sorted(set(_eff_map.values())) + ["Other"]
    sector_filter = st.multiselect(
        "Sectors:",
        all_sectors,
        default=all_sectors,
        label_visibility="collapsed",
        placeholder="Filter by sector…",
    )

    with st.expander(f"Ticker list ({len(all_t)} total)"):
        st.caption("**ETFs:** " + ", ".join(tickers_data.get("etfs", [])))
        st.caption("**Stocks:** " + ", ".join(tickers_data.get("stocks", [])))
        if tickers_data.get("tsx"):
            st.caption("**TSX:** " + ", ".join(tickers_data.get("tsx", [])))
        if tickers_data.get("custom"):
            st.caption("**Custom:** " + ", ".join(tickers_data["custom"]))

    # Add ticker
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        new_t = st.text_input(
            "Add ticker", placeholder="e.g. TSLA", label_visibility="collapsed"
        ).upper().strip()
    with col_btn:
        st.write("")  # vertical align
        add_clicked = st.button("Add", use_container_width=True)

    if add_clicked and new_t:
        if new_t in flat_list(tickers_data):
            st.warning(f"{new_t} already in list.")
        else:
            with st.spinner(f"Looking up sector for {new_t}…"):
                sector = lookup_sector(new_t)
            tickers_data.setdefault("custom", []).append(new_t)
            tickers_data.setdefault("sectors", {})[new_t] = sector
            save_tickers(tickers_data)
            st.success(f"Added {new_t} → {sector}.")
            st.rerun()

    # Remove ticker (custom only)
    custom = tickers_data.get("custom", [])
    if custom:
        remove_t = st.selectbox("Remove custom ticker:", ["—"] + custom)
        if st.button("Remove", use_container_width=True) and remove_t != "—":
            tickers_data["custom"].remove(remove_t)
            save_tickers(tickers_data)
            st.success(f"Removed {remove_t}.")
            st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.header("TheStrat Scanner")
st.caption(
    "Signals are based on **fully closed bars**. "
    "For Daily scans run after market close to avoid partial-bar signals."
)

tickers_data = load_tickers()
ticker_list  = flat_list(tickers_data)

# Summary metrics
m1, m2, m3, _ = st.columns([1, 1, 1, 3])
m1.metric("Tickers",    len(ticker_list))
m2.metric("Timeframes", len(selected_tfs))
m3.metric("Total scans", len(ticker_list) * len(selected_tfs))

st.divider()

if not selected_tfs:
    st.warning("Select at least one timeframe in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Run scan
# ---------------------------------------------------------------------------
if st.button("▶  Run Scan", type="primary"):
    results: list[dict] = []
    errors:  list[str]  = []

    progress = st.progress(0.0, text="Computing FTFC…")

    # Always fetch daily data first — needed for FTFC regardless of selected timeframes
    daily_batch = fetch_batch(ticker_list, "Daily")

    # Pre-fetch all needed batches (selected TFs + two levels of HTFs for MTF context).
    # Weekly and Monthly are ALWAYS fetched — required for true FTFC classification
    # even if the user didn't select those timeframes for scanning.
    tfs_to_fetch: set[str] = set(selected_tfs) | {"Weekly", "Monthly"}
    for tf in selected_tfs:
        htf1 = HTF_MAP.get(tf)
        if htf1:
            tfs_to_fetch.add(htf1)
            htf2 = HTF_MAP.get(htf1)
            if htf2:
                tfs_to_fetch.add(htf2)
    tfs_to_fetch.discard("Daily")  # already have daily_batch

    tf_batches: dict[str, dict] = {"Daily": daily_batch}
    fetch_list = sorted(tfs_to_fetch)
    for i, tf in enumerate(fetch_list, start=1):
        progress.progress(0.1 * i / max(len(fetch_list), 1), text=f"Fetching **{tf}** data…")
        tf_batches[tf] = fetch_batch(ticker_list, tf)

    # Now that Daily, Weekly, Monthly batches are all in memory, compute true FTFC.
    weekly_batch  = tf_batches.get("Weekly", {})
    monthly_batch = tf_batches.get("Monthly", {})
    ftfc_map = {
        ticker: compute_ftfc(
            daily_batch.get(ticker),
            weekly_batch.get(ticker),
            monthly_batch.get(ticker),
        )
        for ticker in ticker_list
    }

    # TFC Score — graded alignment across up to 5 timeframes (uses any batches we have).
    quarterly_batch = tf_batches.get("Quarterly", {})
    twoday_batch    = tf_batches.get("2-Day", {})
    tfc_map = {
        ticker: compute_tfc_score(
            daily_batch.get(ticker),
            weekly_batch.get(ticker),
            monthly_batch.get(ticker),
            quarterly_batch.get(ticker),
            twoday_batch.get(ticker),
        )
        for ticker in ticker_list
    }

    total_steps = len(selected_tfs)

    for step, tf in enumerate(selected_tfs, start=1):
        progress.progress(0.1 + 0.9 * (step - 0.5) / total_steps, text=f"Scanning **{tf}**…")

        batch    = tf_batches.get(tf, {})
        htf_tf   = HTF_MAP.get(tf)
        htf_batch: dict = tf_batches.get(htf_tf, {}) if htf_tf else {}

        for ticker, df in batch.items():
            if df is None:
                errors.append(f"{ticker} ({tf}): no data / insufficient bars")
                continue

            ticker_ftfc = ftfc_map.get(ticker, "")

            # Apply FTFC direction filters at the ticker level
            if direction_filter == "FTFC Bullish" and ticker_ftfc != "Bullish":
                continue
            if direction_filter == "FTFC Bearish" and ticker_ftfc != "Bearish":
                continue

            try:
                sigs = scan_dataframe(df)
            except Exception as exc:
                errors.append(f"{ticker} ({tf}): {exc}")
                continue

            vol_ratio, vol_signal = compute_volume_ratio(df)

            # Higher timeframe targets for this ticker
            htf_high = htf_low = None
            htf_df = htf_batch.get(ticker)
            if htf_df is not None and len(htf_df) >= 1:
                htf_bar  = htf_df.iloc[-1]
                htf_high = round(float(htf_bar["High"]), 2)
                htf_low  = round(float(htf_bar["Low"]),  2)

            for sig in sigs:
                # Apply standard direction filter
                if direction_filter in ("Bullish", "Bearish", "Neutral"):
                    if sig["direction"] != direction_filter:
                        continue
                # Apply signal-name filter
                if sig["signal"] not in selected_signals:
                    continue

                last = df.iloc[-1]
                prev = df.iloc[-2]
                date_str = (
                    last.name.strftime("%Y-%m-%d")
                    if hasattr(last.name, "strftime")
                    else str(last.name)[:10]
                )

                # Continuation: signal direction matches FTFC; Reversal: goes against it
                if sig["direction"] == "Neutral":
                    setup_type = "Neutral"
                elif ticker_ftfc in ("Bullish", "Bearish"):
                    setup_type = "Continuation" if sig["direction"] == ticker_ftfc else "Reversal"
                else:
                    setup_type = "—"

                results.append(
                    {
                        "Ticker":      ticker,
                        "Timeframe":   tf,
                        "Signal":      sig["signal"],
                        "Direction":   sig["direction"],
                        "Setup Type":  setup_type,
                        "Bar Type":    sig["bar_type"],
                        "FTFC":        ticker_ftfc,
                        "TFC Score":   tfc_map.get(ticker, ""),
                        "Vol Ratio":   vol_ratio,
                        "Vol Signal":  vol_signal,
                        "Date":        date_str,
                        "Open":        round(float(last["Open"]),  2),
                        "High":        round(float(last["High"]),  2),
                        "Low":         round(float(last["Low"]),   2),
                        "Close":       round(float(last["Close"]), 2),
                        "Prev High":   round(float(prev["High"]),  2),
                        "Prev Low":    round(float(prev["Low"]),   2),
                        "HTF":         htf_tf or "—",
                        "HTF High":    htf_high,
                        "HTF Low":     htf_low,
                    }
                )

        progress.progress(0.1 + 0.9 * step / total_steps, text=f"Done: {tf}")

    progress.empty()

    # Scan HTF batches for MTF view context — data already in memory, no extra network calls.
    # Only run for tickers that produced at least one signal in the primary scan.
    mtf_context: list[dict] = []
    scanned_tickers = {r["Ticker"] for r in results}
    htf_only_tfs = [tf for tf in tf_batches if tf not in selected_tfs and tf != "Daily"]

    for tf in htf_only_tfs:
        batch    = tf_batches[tf]
        htf_tf   = HTF_MAP.get(tf)
        htf_batch: dict = tf_batches.get(htf_tf, {}) if htf_tf else {}

        for ticker, df in batch.items():
            if df is None or ticker not in scanned_tickers:
                continue
            ticker_ftfc = ftfc_map.get(ticker, "")
            try:
                sigs = scan_dataframe(df)
            except Exception:
                continue

            vol_ratio, vol_signal = compute_volume_ratio(df)

            htf_high = htf_low = None
            htf_df = htf_batch.get(ticker)
            if htf_df is not None and len(htf_df) >= 1:
                htf_bar  = htf_df.iloc[-1]
                htf_high = round(float(htf_bar["High"]), 2)
                htf_low  = round(float(htf_bar["Low"]),  2)

            for sig in sigs:
                last = df.iloc[-1]
                prev = df.iloc[-2]
                date_str = (
                    last.name.strftime("%Y-%m-%d")
                    if hasattr(last.name, "strftime")
                    else str(last.name)[:10]
                )
                if sig["direction"] == "Neutral":
                    setup_type = "Neutral"
                elif ticker_ftfc in ("Bullish", "Bearish"):
                    setup_type = "Continuation" if sig["direction"] == ticker_ftfc else "Reversal"
                else:
                    setup_type = "—"

                mtf_context.append({
                    "Ticker":     ticker,
                    "Timeframe":  tf,
                    "Signal":     sig["signal"],
                    "Direction":  sig["direction"],
                    "Setup Type": setup_type,
                    "Bar Type":   sig["bar_type"],
                    "FTFC":       ticker_ftfc,
                    "TFC Score":  tfc_map.get(ticker, ""),
                    "Vol Ratio":  vol_ratio,
                    "Vol Signal": vol_signal,
                    "Date":       date_str,
                    "Open":       round(float(last["Open"]),  2),
                    "High":       round(float(last["High"]),  2),
                    "Low":        round(float(last["Low"]),   2),
                    "Close":      round(float(last["Close"]), 2),
                    "Prev High":  round(float(prev["High"]),  2),
                    "Prev Low":   round(float(prev["Low"]),   2),
                    "HTF":        htf_tf or "—",
                    "HTF High":   htf_high,
                    "HTF Low":    htf_low,
                })

    st.session_state["results"]     = results
    st.session_state["mtf_context"] = mtf_context
    st.session_state["errors"]      = errors
    st.session_state["scan_time"]   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
if "results" not in st.session_state:
    st.stop()

results     = st.session_state["results"]
mtf_context = st.session_state.get("mtf_context", [])
errors      = st.session_state["errors"]
scan_time   = st.session_state["scan_time"]

st.caption(f"Last scan: {scan_time}")

if errors:
    with st.expander(f"⚠️ {len(errors)} ticker(s) had issues (click to expand)"):
        for e in errors:
            st.caption(e)

if not results:
    st.info("No signals found. Try adding more timeframes or adjusting filters.")
    st.stop()

# Apply sector filter
_eff_map = get_effective_sector_map(tickers_data)
if sector_filter:
    results = [r for r in results if _eff_map.get(r["Ticker"], "Other") in sector_filter]

if not results:
    st.info("No signals found for the selected sectors.")
    st.stop()

# -- Summary metrics --
bullish      = sum(1 for r in results if r["Direction"] == "Bullish")
bearish      = sum(1 for r in results if r["Direction"] == "Bearish")
neutral      = sum(1 for r in results if r["Direction"] == "Neutral")
ftfc_bull    = len({r["Ticker"] for r in results if r["FTFC"] == "Bullish"})
ftfc_bear    = len({r["Ticker"] for r in results if r["FTFC"] == "Bearish"})

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Signals",        len(results))
c2.metric("🟢 Bullish",     bullish)
c3.metric("🔴 Bearish",     bearish)
c4.metric("🟡 Neutral",     neutral)
c5.metric("🟢 FTFC Bull",   ftfc_bull)
c6.metric("🔴 FTFC Bear",   ftfc_bear)

# -- Build + sort DataFrame --
df_res = pd.DataFrame(results)

# Add Sector column
df_res["Sector"] = df_res["Ticker"].map(lambda t: _eff_map.get(t, "Other"))

tf_order = {"Quarterly": 0, "Monthly": 1, "Weekly": 2, "2-Day": 3, "Daily": 4}
df_res["_tf_sort"] = df_res["Timeframe"].map(tf_order)
df_res = (
    df_res
    .sort_values(["_tf_sort", "Sector", "Ticker"])
    .drop(columns=["_tf_sort"])
    .reset_index(drop=True)
)

# Reorder columns so Sector appears early
col_order = ["Ticker", "Sector", "Timeframe", "Signal", "Direction", "Setup Type", "FTFC",
             "TFC Score", "Vol Ratio", "Vol Signal", "Bar Type", "Date", "Open", "High",
             "Low", "Close", "Prev High", "Prev Low", "HTF", "HTF High", "HTF Low"]
df_res = df_res[[c for c in col_order if c in df_res.columns]]

# -- Styling --
def _style_direction(val: str) -> str:
    if val == "Bullish":
        return "color: #00c853; font-weight: 600"
    if val == "Bearish":
        return "color: #ff1744; font-weight: 600"
    return "color: #ffd600; font-weight: 600"

def _style_signal(val: str) -> str:
    bullish_sigs = {"Hammer", "RevStrat 1-2D", "3-2D", "2D Green"}
    bearish_sigs = {"Shooter", "RevStrat 1-2U", "3-2U", "2U Red"}
    if val in bullish_sigs:
        return "color: #00c853"
    if val in bearish_sigs:
        return "color: #ff1744"
    return "color: #ffd600"

def _style_ftfc(val: str) -> str:
    if val == "Bullish":
        return "color: #00c853; font-weight: 600"
    if val == "Bearish":
        return "color: #ff1744; font-weight: 600"
    return "color: #888888"

def _style_tfc_score(val: str) -> str:
    if not isinstance(val, str) or not val:
        return "color: #888888"
    if val.endswith("Bullish"):
        return "color: #00c853; font-weight: 600"
    if val.endswith("Bearish"):
        return "color: #ff1744; font-weight: 600"
    if val.endswith("Mixed"):
        return "color: #ffd600; font-weight: 600"
    return "color: #888888"

def _style_vol_signal(val: str) -> str:
    if val == "High":
        return "color: #00c853; font-weight: 600"
    if val == "Low":
        return "color: #ff1744; font-weight: 600"
    if val == "Normal":
        return "color: #888888"
    return "color: #555555"  # N/A

def _style_setup_type(val: str) -> str:
    if val == "Continuation":
        return "color: #00c853; font-weight: 600"
    if val == "Reversal":
        return "color: #ff9800; font-weight: 600"
    return "color: #888888"


def build_mtf_pivot(results: list[dict], scan_tfs: list[str], sector_map: dict) -> pd.DataFrame:
    """
    Pivot flat scan results into one row per ticker, one column per timeframe.
    Each TF cell shows signal(s) prefixed with a direction emoji.
    Confluence counts how many timeframes have an active signal.
    """
    DIR_EMOJI   = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡"}
    TF_PRIORITY = {"Daily": 0, "2-Day": 1, "Weekly": 2, "Monthly": 3, "Quarterly": 4}

    mtf: dict[str, dict] = {}

    for r in results:
        ticker = r["Ticker"]
        tf     = r["Timeframe"]

        if ticker not in mtf:
            mtf[ticker] = {
                "Ticker":    ticker,
                "Sector":    sector_map.get(ticker, "Other"),
                "FTFC":      r["FTFC"],
                "TFC Score": r.get("TFC Score", ""),
                "_htf_pri":  99,
            }

        # Format signal cell: emoji + signal name; join multiples with separator
        cell = f"{DIR_EMOJI.get(r['Direction'], '')} {r['Signal']}"
        if tf in mtf[ticker]:
            mtf[ticker][tf] += f"  /  {cell}"
        else:
            mtf[ticker][tf] = cell

        # HTF targets — keep from the most granular TF that has them
        pri = TF_PRIORITY.get(tf, 99)
        if pri < mtf[ticker]["_htf_pri"] and r.get("HTF High") is not None:
            mtf[ticker]["_htf_pri"] = pri
            mtf[ticker]["HTF"]      = r.get("HTF", "—")
            mtf[ticker]["HTF High"] = r["HTF High"]
            mtf[ticker]["HTF Low"]  = r["HTF Low"]

    # Assemble output rows
    rows = []
    for ticker, row in mtf.items():
        tf_count = sum(1 for tf in scan_tfs if tf in row)
        out: dict = {
            "Ticker":    row["Ticker"],
            "Sector":    row["Sector"],
            "FTFC":      row["FTFC"],
            "TFC Score": row.get("TFC Score", ""),
        }
        for tf in scan_tfs:
            out[tf] = row.get(tf, "—")
        out["Confluence"] = f"{tf_count}/{len(scan_tfs)}"
        out["HTF"]        = row.get("HTF", "—")
        out["HTF High"]   = row.get("HTF High")
        out["HTF Low"]    = row.get("HTF Low")
        rows.append(out)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Sort: highest confluence first, then sector, then ticker
    df["_conf_sort"] = df["Confluence"].str.split("/").str[0].astype(int)
    df = (
        df.sort_values(["_conf_sort", "Sector", "Ticker"], ascending=[False, True, True])
          .drop(columns=["_conf_sort"])
          .reset_index(drop=True)
    )
    return df


styled = (
    df_res.style
    .map(_style_direction,   subset=["Direction"])
    .map(_style_signal,      subset=["Signal"])
    .map(_style_ftfc,        subset=["FTFC"])
    .map(_style_tfc_score,   subset=["TFC Score"])
    .map(_style_vol_signal,  subset=["Vol Signal"])
    .map(_style_setup_type,  subset=["Setup Type"])
)

tab1, tab2 = st.tabs(["📋 Flat View", "🔀 MTF View"])

with tab1:
    st.dataframe(styled, use_container_width=True, hide_index=True, height=520)

    # -- Buttons row --
    btn_col1, btn_col2, _ = st.columns([1, 1.4, 4])

    # Download CSV
    csv = df_res.to_csv(index=False)
    with btn_col1:
        st.download_button(
            label="⬇  Download CSV",
            data=csv,
            file_name=f"strat_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

    # Copy Tickers to clipboard (deduped, TradingView-formatted, comma-separated)
    with btn_col2:
        unique_tickers = ",".join(
            to_tradingview_format(t) for t in sorted(df_res["Ticker"].unique().tolist())
        )
        components.html(
            f"""
            <script>
            function copyTickers() {{
                const text = "{unique_tickers}";
                const btn = document.getElementById("copy-btn");
                navigator.clipboard.writeText(text).then(function() {{
                    btn.textContent = "✓ Copied!";
                    btn.style.borderColor = "#00c853";
                    btn.style.color = "#00c853";
                    setTimeout(() => {{
                        btn.textContent = "📋 Copy Tickers";
                        btn.style.borderColor = "#555";
                        btn.style.color = "white";
                    }}, 2000);
                }}, function() {{
                    // execCommand fallback for older browsers
                    const el = document.createElement("textarea");
                    el.value = text;
                    el.style.position = "fixed";
                    el.style.opacity = "0";
                    document.body.appendChild(el);
                    el.select();
                    document.execCommand("copy");
                    document.body.removeChild(el);
                    btn.textContent = "✓ Copied!";
                    btn.style.borderColor = "#00c853";
                    btn.style.color = "#00c853";
                    setTimeout(() => {{
                        btn.textContent = "📋 Copy Tickers";
                        btn.style.borderColor = "#555";
                        btn.style.color = "white";
                    }}, 2000);
                }});
            }}
            </script>
            <button id="copy-btn" onclick="copyTickers()" style="
                background-color: #262730;
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px 16px;
                font-size: 14px;
                cursor: pointer;
                font-family: sans-serif;
                width: 100%;
                height: 38px;
                transition: color 0.2s, border-color 0.2s;
            ">📋 Copy Tickers</button>
            """,
            height=45,
        )

with tab2:
    st.caption(
        "One row per ticker. Each timeframe column shows the signal(s) that fired. "
        "**Confluence** = how many timeframes have an active signal. "
        "Sorted highest confluence first."
    )

    # Filter MTF context to only tickers that passed the sector filter
    result_tickers   = {r["Ticker"] for r in results}
    context_filtered = [r for r in mtf_context if r["Ticker"] in result_tickers]
    mtf_all          = results + context_filtered

    # TF columns in logical display order — includes both scanned TFs and HTF context TFs
    tf_display_order = ["Daily", "2-Day", "Weekly", "Monthly", "Quarterly"]
    scan_tfs = [tf for tf in tf_display_order if any(r["Timeframe"] == tf for r in mtf_all)]

    df_mtf = build_mtf_pivot(mtf_all, scan_tfs, _eff_map)

    if df_mtf.empty:
        st.info("No MTF data to display.")
    else:
        def _style_confluence(val: str) -> str:
            try:
                num, den = val.split("/")
                if int(num) == int(den) and int(den) > 1:
                    return "color: #00c853; font-weight: 700"   # all TFs aligned
                if int(num) > 1:
                    return "color: #ffd600; font-weight: 600"   # partial confluence
            except Exception:
                pass
            return "color: #888888"

        styled_mtf = (
            df_mtf.style
            .map(_style_ftfc,       subset=["FTFC"])
            .map(_style_tfc_score,  subset=["TFC Score"])
            .map(_style_confluence, subset=["Confluence"])
        )
        st.dataframe(styled_mtf, use_container_width=True, hide_index=True, height=520)

        mtf_csv = df_mtf.to_csv(index=False)
        st.download_button(
            label="⬇  Download MTF CSV",
            data=mtf_csv,
            file_name=f"strat_mtf_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )
