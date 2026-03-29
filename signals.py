"""
TheStrat signal detection logic.

Bar type classification:
  1  = Inside bar  (high <= prev_high AND low >= prev_low)
  2U = Up bar      (high > prev_high AND low >= prev_low)
  2D = Down bar    (low < prev_low  AND high <= prev_high)
  3  = Outside bar (high > prev_high AND low < prev_low)
"""

import pandas as pd

# Wick/body thresholds for Shooters and Hammers.
# User spec: wick ~66% of range, body ~33%. We allow ±10% tolerance.
WICK_MIN = 0.56   # dominant wick must be at least this fraction of total range
BODY_MAX = 0.40   # body must be no more than this fraction of total range

# Reversal close threshold for RevStrat 1-2U/1-2D
REV_CLOSE_THRESHOLD = 0.45   # 2U bar: close must be in bottom 45% of range
                              # 2D bar: close must be in top 45% (i.e. >= 55%)


def classify_bar(h: float, l: float, ph: float, pl: float) -> str:
    """Return TheStrat bar type relative to the previous bar (ph/pl)."""
    breaks_high = h > ph
    breaks_low = l < pl
    if breaks_high and breaks_low:
        return "3"
    if breaks_high:
        return "2U"
    if breaks_low:
        return "2D"
    return "1"


def _wick_body_ratio(o, h, l, c):
    """
    Returns (upper_wick_pct, lower_wick_pct, body_pct) as fractions of total range.
    Returns (0, 0, 0) if range is zero.
    """
    total = h - l
    if total <= 0:
        return 0.0, 0.0, 0.0
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    return upper_wick / total, lower_wick / total, body / total


def check_shooter(row: pd.Series, prev: pd.Series) -> bool:
    """
    Shooter (bearish): 2U bar where the upper wick dominates.
    - Breaks prior high  (2U)
    - Upper wick >= WICK_MIN of total range
    - Body <= BODY_MAX of total range
    """
    o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
    if classify_bar(h, l, prev["High"], prev["Low"]) != "2U":
        return False
    uw, _, body = _wick_body_ratio(o, h, l, c)
    return uw >= WICK_MIN and body <= BODY_MAX


def check_hammer(row: pd.Series, prev: pd.Series) -> bool:
    """
    Hammer (bullish): 2D bar where the lower wick dominates.
    - Breaks prior low  (2D)
    - Lower wick >= WICK_MIN of total range
    - Body <= BODY_MAX of total range
    """
    o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
    if classify_bar(h, l, prev["High"], prev["Low"]) != "2D":
        return False
    _, lw, body = _wick_body_ratio(o, h, l, c)
    return lw >= WICK_MIN and body <= BODY_MAX


def check_rev_strat_1_2u(df: pd.DataFrame) -> bool:
    """
    RevStrat 1-2U (bearish setup):
      bar[-3] → bar[-2] = Inside bar (1)
      bar[-2] → bar[-1] = 2U, but close is in the bottom 45% of bar's range
                           (breakout above inside bar that failed — bears taking over)
    """
    if len(df) < 3:
        return False
    b3, b2, b1 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

    if classify_bar(b2["High"], b2["Low"], b3["High"], b3["Low"]) != "1":
        return False
    if classify_bar(b1["High"], b1["Low"], b2["High"], b2["Low"]) != "2U":
        return False

    total = b1["High"] - b1["Low"]
    if total <= 0:
        return False
    close_pos = (b1["Close"] - b1["Low"]) / total
    return close_pos <= REV_CLOSE_THRESHOLD


def check_rev_strat_1_2d(df: pd.DataFrame) -> bool:
    """
    RevStrat 1-2D (bullish setup):
      bar[-3] → bar[-2] = Inside bar (1)
      bar[-2] → bar[-1] = 2D, but close is in the top 45% of bar's range
                           (breakdown below inside bar that failed — bulls taking over)
    """
    if len(df) < 3:
        return False
    b3, b2, b1 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

    if classify_bar(b2["High"], b2["Low"], b3["High"], b3["Low"]) != "1":
        return False
    if classify_bar(b1["High"], b1["Low"], b2["High"], b2["Low"]) != "2D":
        return False

    total = b1["High"] - b1["Low"]
    if total <= 0:
        return False
    close_pos = (b1["Close"] - b1["Low"]) / total
    return close_pos >= (1.0 - REV_CLOSE_THRESHOLD)


def check_3_2u(df: pd.DataFrame) -> bool:
    """
    3-2U (bearish setup):
      bar[-3] → bar[-2] = Outside bar (3)
      bar[-2] → bar[-1] = 2U  (breaks above the 3 bar's high)
      bar[-1].Close is back WITHIN the 3 bar's range (≤ bar[-2].High)
      → failed breakout above the outside bar; bears stepped in.
    """
    if len(df) < 3:
        return False
    b3, b2, b1 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

    if classify_bar(b2["High"], b2["Low"], b3["High"], b3["Low"]) != "3":
        return False
    if classify_bar(b1["High"], b1["Low"], b2["High"], b2["Low"]) != "2U":
        return False

    # Close must fall back inside the 3 bar's range
    return b1["Close"] <= b2["High"] and b1["Close"] >= b2["Low"]


def check_3_2d(df: pd.DataFrame) -> bool:
    """
    3-2D (bullish setup):
      bar[-3] → bar[-2] = Outside bar (3)
      bar[-2] → bar[-1] = 2D  (breaks below the 3 bar's low)
      bar[-1].Close is back WITHIN the 3 bar's range (≥ bar[-2].Low)
      → failed breakdown below the outside bar; bulls stepped in.
    """
    if len(df) < 3:
        return False
    b3, b2, b1 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

    if classify_bar(b2["High"], b2["Low"], b3["High"], b3["Low"]) != "3":
        return False
    if classify_bar(b1["High"], b1["Low"], b2["High"], b2["Low"]) != "2D":
        return False

    # Close must fall back inside the 3 bar's range
    return b1["Close"] >= b2["Low"] and b1["Close"] <= b2["High"]


def check_double_inside(df: pd.DataFrame) -> bool:
    """
    1-1 (Double Inside Bar):
      bar[-3] → bar[-2] = Inside bar (1)
      bar[-2] → bar[-1] = Inside bar (1)
      Two consecutive inside bars — price coiling, breakout imminent.
    """
    if len(df) < 3:
        return False
    b3, b2, b1 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

    if classify_bar(b2["High"], b2["Low"], b3["High"], b3["Low"]) != "1":
        return False
    return classify_bar(b1["High"], b1["Low"], b2["High"], b2["Low"]) == "1"


def check_2u_red(row: pd.Series, prev: pd.Series) -> bool:
    """
    2U Red (bearish): breaks above the prior high (2U) but closes
    below its open — red candle body despite the breakout.
    Bears overwhelmed the initial push through the prior high.
    """
    if classify_bar(row["High"], row["Low"], prev["High"], prev["Low"]) != "2U":
        return False
    return row["Close"] < row["Open"]


def check_2d_green(row: pd.Series, prev: pd.Series) -> bool:
    """
    2D Green (bullish): breaks below the prior low (2D) but closes
    above its open — green candle body despite the breakdown.
    Bulls overwhelmed the initial push through the prior low.
    """
    if classify_bar(row["High"], row["Low"], prev["High"], prev["Low"]) != "2D":
        return False
    return row["Close"] > row["Open"]


def scan_dataframe(df: pd.DataFrame) -> list[dict]:
    """
    Scan the last completed bar of df for all TheStrat signals.
    Returns a list of dicts: {signal, bar_type, direction}
    An empty list means no signals on the last bar.
    """
    if len(df) < 3:
        return []

    last = df.iloc[-1]
    prev = df.iloc[-2]

    bar_type = classify_bar(last["High"], last["Low"], prev["High"], prev["Low"])
    signals = []

    # --- Inside Bar ---
    if bar_type == "1":
        signals.append({"signal": "Inside Bar", "bar_type": "1", "direction": "Neutral"})

    # --- Double Inside Bar (1-1) ---
    if check_double_inside(df):
        signals.append({"signal": "Double Inside (1-1)", "bar_type": "1-1", "direction": "Neutral"})

    # --- Outside Bar ---
    if bar_type == "3":
        signals.append({"signal": "Outside Bar", "bar_type": "3", "direction": "Neutral"})

    # --- Shooter ---
    if check_shooter(last, prev):
        signals.append({"signal": "Shooter", "bar_type": "2U", "direction": "Bearish"})

    # --- Hammer ---
    if check_hammer(last, prev):
        signals.append({"signal": "Hammer", "bar_type": "2D", "direction": "Bullish"})

    # --- RevStrat 1-2U ---
    if check_rev_strat_1_2u(df):
        signals.append({"signal": "RevStrat 1-2U", "bar_type": "1-2U", "direction": "Bearish"})

    # --- RevStrat 1-2D ---
    if check_rev_strat_1_2d(df):
        signals.append({"signal": "RevStrat 1-2D", "bar_type": "1-2D", "direction": "Bullish"})

    # --- 3-2U ---
    if check_3_2u(df):
        signals.append({"signal": "3-2U", "bar_type": "3-2U", "direction": "Bearish"})

    # --- 3-2D ---
    if check_3_2d(df):
        signals.append({"signal": "3-2D", "bar_type": "3-2D", "direction": "Bullish"})

    # --- 2U Red ---
    if check_2u_red(last, prev):
        signals.append({"signal": "2U Red", "bar_type": "2U", "direction": "Bearish"})

    # --- 2D Green ---
    if check_2d_green(last, prev):
        signals.append({"signal": "2D Green", "bar_type": "2D", "direction": "Bullish"})

    return signals
