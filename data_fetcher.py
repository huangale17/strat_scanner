"""
OHLC data fetching via yfinance.

Supports Daily, 2-Day, Weekly, Monthly, and Quarterly timeframes.
2-Day and Quarterly are derived by resampling finer data.
Incomplete current-period bars are automatically dropped so that
all signals are based on fully closed bars.
"""

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

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


def _last_bday_of_month(year: int, month: int) -> date:
    """Return the last Mon–Fri calendar day of the given month."""
    if month == 12:
        last_day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)
    bday = np.busday_offset(last_day.strftime("%Y-%m-%d"), 0, roll="preceding")
    return pd.Timestamp(bday).date()


def _drop_incomplete_bar(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Remove the last bar when it represents a still-open current period,
    ensuring signals are only fired on fully closed bars.

    Every bar closes at 4 PM ET on the last trading day of its period.
    We keep the bar once that moment has passed.
    """
    if df.empty:
        return df

    today     = date.today()
    last      = df.index[-1]
    last_date = last.date() if hasattr(last, "date") else last

    # True once the US market has closed for the day (4:00 PM ET or later)
    now_et        = datetime.now(ZoneInfo("America/New_York"))
    market_closed = now_et.hour >= 16

    _REF = np.datetime64("2000-01-03")

    def _bday_group(d: date) -> int:
        """Busday pair-group used for 2-Day resampling."""
        return int(np.busday_count(_REF, np.datetime64(d.strftime("%Y-%m-%d")))) // 2

    # ------------------------------------------------------------------ Daily
    if timeframe == "Daily":
        # Drop if today's bar is present but the session hasn't closed yet.
        if last_date >= today and not market_closed:
            df = df.iloc[:-1]

    # ----------------------------------------------------------------- 2-Day
    elif timeframe == "2-Day":
        # Resampled bars are labelled with the FIRST day of the pair.
        # The bar is complete only after the SECOND day closes at 4 PM ET.
        # If today is a weekday and falls in the same pair as the last bar
        # (and is therefore the 2nd day), drop while market is still open.
        if today.weekday() < 5:
            if _bday_group(last_date) == _bday_group(today) and last_date != today:
                if not market_closed:
                    df = df.iloc[:-1]

    # ---------------------------------------------------------------- Weekly
    elif timeframe == "Weekly":
        if today.weekday() < 5:                          # weekdays only
            lw = last_date.isocalendar()
            tw = today.isocalendar()
            if lw[0] == tw[0] and lw[1] == tw[1]:       # same ISO week
                # Last trading day of this week (usually Friday, but
                # rolls back to Thursday on holiday Fridays, etc.)
                friday     = today + timedelta(days=(4 - today.weekday()))
                last_bday  = pd.Timestamp(
                    np.busday_offset(friday.strftime("%Y-%m-%d"), 0, roll="preceding")
                ).date()

                if today < last_bday:
                    # More trading days still to come this week → always drop
                    df = df.iloc[:-1]
                elif today == last_bday and not market_closed:
                    # Last trading day of the week but session still open → drop
                    df = df.iloc[:-1]
                # else: last trading day is done → keep the bar

    # --------------------------------------------------------------- Monthly
    elif timeframe == "Monthly":
        if last_date.year == today.year and last_date.month == today.month:
            last_bday = _last_bday_of_month(today.year, today.month)
            if today < last_bday:
                df = df.iloc[:-1]
            elif today == last_bday and not market_closed:
                df = df.iloc[:-1]
            # else: month's last session is done → keep

    # ------------------------------------------------------------- Quarterly
    elif timeframe == "Quarterly":
        lq = (last_date.month - 1) // 3
        tq = (today.month - 1) // 3
        if last_date.year == today.year and lq == tq:
            q_last_month = (tq + 1) * 3          # 3, 6, 9, or 12
            last_bday    = _last_bday_of_month(today.year, q_last_month)
            if today < last_bday:
                df = df.iloc[:-1]
            elif today == last_bday and not market_closed:
                df = df.iloc[:-1]
            # else: quarter's last session is done → keep

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
# FTFC (Full Timeframe Continuity)
# ---------------------------------------------------------------------------

def compute_ftfc(
    daily_df: pd.DataFrame | None,
    weekly_df: pd.DataFrame | None,
    monthly_df: pd.DataFrame | None,
) -> str:
    """
    Compute Full Timeframe Continuity using #theStrat bar classification.

    A timeframe is considered bullish if its most recent *closed* bar is a
    2U bar (broke prior high, did not break prior low), and bearish if it is
    a 2D bar (broke prior low, did not break prior high).

    Full Bullish FTFC: Monthly, Weekly, and Daily are all 2U.
    Full Bearish FTFC: Monthly, Weekly, and Daily are all 2D.

    Any missing DataFrame (None or <2 bars) causes the function to return "".
    """
    from signals import classify_bar

    def _last_type(df: pd.DataFrame | None) -> str | None:
        if df is None or len(df) < 2:
            return None
        last = df.iloc[-1]
        prev = df.iloc[-2]
        return classify_bar(
            float(last["High"]), float(last["Low"]),
            float(prev["High"]), float(prev["Low"]),
        )

    d = _last_type(daily_df)
    w = _last_type(weekly_df)
    m = _last_type(monthly_df)

    if d is None or w is None or m is None:
        return ""

    if d == "2U" and w == "2U" and m == "2U":
        return "Bullish"
    if d == "2D" and w == "2D" and m == "2D":
        return "Bearish"
    return ""


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


# ---------------------------------------------------------------------------
# TFC Score (graded Timeframe Continuity)
# ---------------------------------------------------------------------------

def compute_tfc_score(
    daily_df: pd.DataFrame | None,
    weekly_df: pd.DataFrame | None,
    monthly_df: pd.DataFrame | None,
    quarterly_df: pd.DataFrame | None = None,
    tf_2day_df: pd.DataFrame | None = None,
) -> str:
    """
    Compute a graded Timeframe Continuity score across up to 5 timeframes.

    For each non-None DataFrame with at least 2 bars, classify the last bar
    using classify_bar. Count how many are 2U (bullish) and 2D (bearish).

    Returns:
        "{bullish}/{total} Bullish"  if bullish > bearish
        "{bearish}/{total} Bearish"  if bearish > bullish
        "{bullish}/{total} Mixed"    if tied (includes all-1 / all-3 cases)
        ""                           if fewer than 2 TFs have usable data
    """
    from signals import classify_bar

    dfs = [daily_df, weekly_df, monthly_df, quarterly_df, tf_2day_df]

    bullish = 0
    bearish = 0
    total = 0
    for df in dfs:
        if df is None or len(df) < 2:
            continue
        total += 1
        last = df.iloc[-1]
        prev = df.iloc[-2]
        bar_type = classify_bar(
            float(last["High"]), float(last["Low"]),
            float(prev["High"]), float(prev["Low"]),
        )
        if bar_type == "2U":
            bullish += 1
        elif bar_type == "2D":
            bearish += 1

    if total < 2:
        return ""

    if bullish > bearish:
        return f"{bullish}/{total} Bullish"
    if bearish > bullish:
        return f"{bearish}/{total} Bearish"
    return f"{bullish}/{total} Mixed"


def compute_combo(
    monthly_df: pd.DataFrame | None,
    weekly_df: pd.DataFrame | None,
    daily_df: pd.DataFrame | None,
) -> tuple[str, str]:
    """
    Compute the Monthly-Weekly-Daily bar type combo string and detect named patterns.

    Returns (combo_str, named_pattern):
        combo_str:     e.g. "2U-1-2U", or "—" if any TF has insufficient data
        named_pattern: one of "1-2U", "1-2D", "3-2U", "3-2D", "1-1", or ""
                       Weekly-Daily pair takes priority over Monthly-Weekly.
    """
    from signals import classify_bar

    def _bar_type(df: pd.DataFrame | None) -> str | None:
        if df is None or len(df) < 2:
            return None
        last, prev = df.iloc[-1], df.iloc[-2]
        return classify_bar(
            float(last["High"]), float(last["Low"]),
            float(prev["High"]), float(prev["Low"]),
        )

    m = _bar_type(monthly_df)
    w = _bar_type(weekly_df)
    d = _bar_type(daily_df)

    if m is None or w is None or d is None:
        return "—", ""

    combo_str = f"{m}-{w}-{d}"

    _NAMED = {"1-2U", "1-2D", "3-2U", "3-2D", "1-1"}

    wd = f"{w}-{d}"
    if wd in _NAMED:
        return combo_str, wd

    mw = f"{m}-{w}"
    if mw in _NAMED:
        return combo_str, mw

    return combo_str, ""
