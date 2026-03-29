"""
TheStrat Scanner — Streamlit web app.

Run with:  streamlit run app.py
"""

import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from data_fetcher import fetch_batch
from signals import classify_bar, scan_dataframe

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


def to_tradingview_format(ticker: str) -> str:
    """Convert yfinance ticker format to TradingView format for watchlist pasting."""
    if ticker.endswith(".TO"):
        return "TSX:" + ticker[:-3]
    if ticker.endswith(".V"):
        return "TSXV:" + ticker[:-2]
    return ticker


def load_tickers() -> dict:
    with open(TICKERS_FILE, "r") as f:
        return json.load(f)


def save_tickers(data: dict) -> None:
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
        ["All", "Bullish", "Bearish", "Neutral"],
        horizontal=True,
    )

    st.divider()

    # -- Ticker management --
    st.subheader("Tickers")
    tickers_data = load_tickers()
    all_t = flat_list(tickers_data)

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
            tickers_data.setdefault("custom", []).append(new_t)
            save_tickers(tickers_data)
            st.success(f"Added {new_t}.")
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

    progress = st.progress(0.0, text="Initializing…")
    total_steps = len(selected_tfs)

    for step, tf in enumerate(selected_tfs, start=1):
        progress.progress((step - 0.5) / total_steps, text=f"Fetching **{tf}** data…")

        batch = fetch_batch(ticker_list, tf)

        for ticker, df in batch.items():
            if df is None:
                errors.append(f"{ticker} ({tf}): no data / insufficient bars")
                continue
            try:
                sigs = scan_dataframe(df)
            except Exception as exc:
                errors.append(f"{ticker} ({tf}): {exc}")
                continue

            for sig in sigs:
                # Apply direction filter
                if direction_filter != "All" and sig["direction"] != direction_filter:
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

                results.append(
                    {
                        "Ticker":     ticker,
                        "Timeframe":  tf,
                        "Signal":     sig["signal"],
                        "Direction":  sig["direction"],
                        "Bar Type":   sig["bar_type"],
                        "Date":       date_str,
                        "Open":       round(float(last["Open"]),  2),
                        "High":       round(float(last["High"]),  2),
                        "Low":        round(float(last["Low"]),   2),
                        "Close":      round(float(last["Close"]), 2),
                        "Prev High":  round(float(prev["High"]),  2),
                        "Prev Low":   round(float(prev["Low"]),   2),
                    }
                )

        progress.progress(step / total_steps, text=f"Done: {tf}")

    progress.empty()

    st.session_state["results"]   = results
    st.session_state["errors"]    = errors
    st.session_state["scan_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
if "results" not in st.session_state:
    st.stop()

results   = st.session_state["results"]
errors    = st.session_state["errors"]
scan_time = st.session_state["scan_time"]

st.caption(f"Last scan: {scan_time}")

if errors:
    with st.expander(f"⚠️ {len(errors)} ticker(s) had issues (click to expand)"):
        for e in errors:
            st.caption(e)

if not results:
    st.info("No signals found. Try adding more timeframes or adjusting filters.")
    st.stop()

# -- Summary metrics --
bullish = sum(1 for r in results if r["Direction"] == "Bullish")
bearish = sum(1 for r in results if r["Direction"] == "Bearish")
neutral = sum(1 for r in results if r["Direction"] == "Neutral")

c1, c2, c3, c4, _ = st.columns([1, 1, 1, 1, 2])
c1.metric("Signals",  len(results))
c2.metric("🟢 Bullish", bullish)
c3.metric("🔴 Bearish", bearish)
c4.metric("🟡 Neutral", neutral)

# -- Build + sort DataFrame --
df_res = pd.DataFrame(results)

tf_order = {"Quarterly": 0, "Monthly": 1, "Weekly": 2, "2-Day": 3, "Daily": 4}
df_res["_tf_sort"] = df_res["Timeframe"].map(tf_order)
df_res = (
    df_res
    .sort_values(["_tf_sort", "Ticker"])
    .drop(columns=["_tf_sort"])
    .reset_index(drop=True)
)

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

styled = (
    df_res.style
    .map(_style_direction, subset=["Direction"])
    .map(_style_signal,    subset=["Signal"])
)

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
