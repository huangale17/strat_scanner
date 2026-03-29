"""
OHLC data fetching via yfinance.

Supports Daily, 2-Day, Weekly, Monthly, and Quarterly timeframes.
2-Day and Quarterly are derived by resampling finer data.
Incomplete current-period bars are automatically dropped so that
all signals are based on fully closed bars.
"""

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Timeframe config
# ---------------------------------------------------------------------------
_TF_CONFIG = {
    "Daily":     {"period": "1y",  "interval": "1d"},
    "2-Day":     {"period": "2y",  "interval": "1d"},    # resampled below
    "Weekly":    {"period": "3y",  "interval": "1wk"},
    "Monthly":   {"period": "10y", "interval": "1mo"},
    "Quarterly": {"period": "10y", "interval": "1mo"},   # resampled below
}

_OHLCV = ["Open", "High", "Low", "Close", "Volume"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resample_2day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily OHLCV into 2-trading-day bars with a globally fixed
    grouping that matches TradingView's 2-day bar boundaries.

    Grouping is determined by counting weekdays (Mon-Fri) from a fixed
    reference date (Jan 3 2000, first trading day of Y2K) and doing
    integer division by 2.  This is independent of how many bars were
    fetched, so the pairings are always consistent:

        group = busday_count('2000-01-03', date) // 2

    Verified correct pairings (user-confirmed):
        Mar 25 + Mar 26  → same group  (last complete 2-day bar)
        Mar 27 + Mar 30  → same group  (current active, crosses weekend)

    Groups that contain only 1 trading day (e.g. a day whose weekday
    partner was a market holiday) are dropped to preserve the
    full-bar-close rule.
    """
    if df.empty:
        return df

    df = df.copy()
    _REF = np.datetime64("2000-01-03")

    def _group(idx):
        d = idx.date() if hasattr(idx, "date") else idx
        n = int(np.busday_count(_REF, np.datetime64(d.strftime("%Y-%m-%d"))))
        return n // 2

    df["_g"] = [_group(idx) for idx in df.index]

    # Only keep groups with exactly 2 trading days (complete pairs)
    group_sizes   = df.groupby("_g").size()
    complete_grps = set(group_sizes[group_sizes == 2].index)
    df_complete   = df[df["_g"].isin(complete_grps)].copy()

    if df_complete.empty:
        return pd.DataFrame(columns=_OHLCV)

    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    result = df_complete.groupby("_g").agg(agg)

    # Label each bar with the first trading day's timestamp
    first_dates = {g: grp.index[0] for g, grp in df_complete.groupby("_g")}
    result.index = [first_dates[g] for g in result.index]
    result.index.name = "Date"

    return result


def _resample_quarterly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample a monthly OHLCV DataFrame to calendar-quarter bars."""
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    # pandas ≥ 2.2 uses 'QE'; older versions use 'Q'
    for freq in ("QE", "Q"):
        try:
            return df.resample(freq).agg(agg).dropna(how="all")
        except ValueError:
            continue
    return df  # fallback: return monthly if resample fails


def _drop_incomplete_bar(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Remove the last bar when it represents the still-open current period,
    ensuring signals are only fired on fully closed bars.
    """
    if df.empty:
        return df

    today = date.today()
    last = df.index[-1]
    last_date = last.date() if hasattr(last, "date") else last

    if timeframe == "Daily":
        if last_date >= today:
            df = df.iloc[:-1]

    elif timeframe == "Weekly":
        # Only drop if today is a weekday — on weekends the prior week is fully closed.
        # today.weekday(): 0=Mon … 4=Fri, 5=Sat, 6=Sun
        if today.weekday() < 5:
            lw = last_date.isocalendar()
            tw = today.isocalendar()
            if lw[0] == tw[0] and lw[1] == tw[1]:
                df = df.iloc[:-1]

    elif timeframe == "Monthly":
        if last_date.year == today.year and last_date.month == today.month:
            df = df.iloc[:-1]

    elif timeframe == "Quarterly":
        lq = (last_date.month - 1) // 3
        tq = (today.month - 1) // 3
        if last_date.year == today.year and lq == tq:
            df = df.iloc[:-1]

    return df


def _fetch_one(ticker: str, timeframe: str) -> tuple[str, pd.DataFrame | None]:
    """Fetch OHLC for a single ticker / timeframe. Returns (ticker, df | None)."""
    cfg = _TF_CONFIG[timeframe]
    try:
        t = yf.Ticker(ticker)
        df = t.history(
            period=cfg["period"],
            interval=cfg["interval"],
            auto_adjust=True,
            actions=False,
        )
        if df.empty:
            return ticker, None

        df = df[_OHLCV].dropna()

        if timeframe == "2-Day":
            df = _resample_2day(df)
        elif timeframe == "Quarterly":
            df = _resample_quarterly(df)

        df = _drop_incomplete_bar(df, timeframe)

        return ticker, (df if len(df) >= 3 else None)
    except Exception:
        return ticker, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_batch(tickers: list[str], timeframe: str, max_workers: int = 12) -> dict[str, pd.DataFrame | None]:
    """
    Fetch OHLC data for multiple tickers in parallel.

    Returns a dict mapping ticker → DataFrame (or None if fetch failed / insufficient data).
    """
    results: dict[str, pd.DataFrame | None] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, t, timeframe): t for t in tickers}
        for future in as_completed(futures):
            ticker, df = future.result()
            results[ticker] = df
    return results
