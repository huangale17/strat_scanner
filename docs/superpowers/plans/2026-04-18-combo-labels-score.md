# Combo Labels + Signal Strength Score Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Monthly-Weekly-Daily combo label column and a 0–5 signal strength score column to both the flat and MTF scan views, with the flat view sorted score-descending by default.

**Architecture:** Two new pure functions (`compute_combo` in `data_fetcher.py`, `compute_signal_score` in `signals.py`) follow the existing pattern of `compute_ftfc`/`compute_tfc_score`. A `combo_map` is pre-computed per-ticker after the FTFC/TFC maps; score is computed per-signal row. Both are stored in session state and wired into flat view and MTF pivot.

**Tech Stack:** Python 3.12+, pandas, Streamlit — no new dependencies.

---

## File Map

| File | Change |
|------|--------|
| `data_fetcher.py` | Add `compute_combo()` |
| `signals.py` | Add `compute_signal_score()` |
| `app.py` | Import new functions; add `combo_map` to session state; add Score/Combo/Pattern to result dicts; update col order, sort, styling, MTF pivot call |
| `tests/test_combo.py` | New — tests for `compute_combo` |
| `tests/test_score.py` | New — tests for `compute_signal_score` |

---

## Task 1: `compute_combo` in `data_fetcher.py`

**Files:**
- Modify: `data_fetcher.py`
- Create: `tests/test_combo.py`

- [ ] **Step 1.1: Write the failing tests**

Create `tests/test_combo.py`:

```python
import pandas as pd
import pytest
from data_fetcher import compute_combo


def _make_df(highs, lows):
    """Build a minimal OHLCV DataFrame from high/low lists."""
    return pd.DataFrame({
        "Open":   highs,
        "High":   highs,
        "Low":    lows,
        "Close":  highs,
        "Volume": [1000] * len(highs),
    })


# --- bar type helpers: last bar vs second-to-last ---
# 2U: last high > prev high, last low >= prev low
_2U = _make_df([10, 12], [5, 6])   # last breaks high only
_2D = _make_df([10, 9],  [5, 3])   # last breaks low only
_1  = _make_df([10, 9],  [5, 6])   # inside bar (both within)
_3  = _make_df([10, 12], [5, 3])   # outside bar (breaks both)


class TestComboStr:
    def test_all_2u_returns_correct_string(self):
        combo_str, _ = compute_combo(_2U, _2U, _2U)
        assert combo_str == "2U-2U-2U"

    def test_mixed_types_builds_correct_string(self):
        combo_str, _ = compute_combo(_2U, _1, _2U)
        assert combo_str == "2U-1-2U"

    def test_none_monthly_returns_dash(self):
        combo_str, pattern = compute_combo(None, _2U, _2U)
        assert combo_str == "—"
        assert pattern == ""

    def test_none_weekly_returns_dash(self):
        combo_str, pattern = compute_combo(_2U, None, _2U)
        assert combo_str == "—"
        assert pattern == ""

    def test_none_daily_returns_dash(self):
        combo_str, pattern = compute_combo(_2U, _2U, None)
        assert combo_str == "—"
        assert pattern == ""

    def test_insufficient_bars_returns_dash(self):
        single_row = _make_df([10], [5])
        combo_str, pattern = compute_combo(single_row, _2U, _2U)
        assert combo_str == "—"
        assert pattern == ""


class TestNamedPattern:
    def test_weekly_daily_1_2u_detected(self):
        # Weekly=1, Daily=2U → Weekly-Daily pair = "1-2U"
        _, pattern = compute_combo(_2U, _1, _2U)
        assert pattern == "1-2U"

    def test_weekly_daily_1_2d_detected(self):
        _, pattern = compute_combo(_2U, _1, _2D)
        assert pattern == "1-2D"

    def test_weekly_daily_3_2u_detected(self):
        _, pattern = compute_combo(_2U, _3, _2U)
        assert pattern == "3-2U"

    def test_weekly_daily_3_2d_detected(self):
        _, pattern = compute_combo(_2U, _3, _2D)
        assert pattern == "3-2D"

    def test_weekly_daily_1_1_detected(self):
        _, pattern = compute_combo(_2U, _1, _1)
        assert pattern == "1-1"

    def test_monthly_weekly_1_2u_detected_when_daily_not_named(self):
        # Monthly=1, Weekly=2U, Daily=2D → wd="2U-2D" not named; mw="1-2U" named
        _, pattern = compute_combo(_1, _2U, _2D)
        assert pattern == "1-2U"

    def test_weekly_daily_wins_over_monthly_weekly(self):
        # Both pairs named: Monthly=1, Weekly=1, Daily=2U
        # mw="1-1" (named), wd="1-2U" (named) → Weekly-Daily wins
        _, pattern = compute_combo(_1, _1, _2U)
        assert pattern == "1-2U"

    def test_no_named_pattern_returns_empty_string(self):
        # 2U-2U-2U: neither "2U-2U" pair is named
        _, pattern = compute_combo(_2U, _2U, _2U)
        assert pattern == ""

    def test_no_named_pattern_returns_dash_combo_str(self):
        combo_str, _ = compute_combo(_2U, _2U, _2U)
        assert combo_str == "2U-2U-2U"
```

- [ ] **Step 1.2: Run tests to verify they fail**

```
cd C:\Users\alexa\Strat_Scanner
python -m pytest tests/test_combo.py -v
```

Expected: `ImportError` or `AttributeError: module 'data_fetcher' has no attribute 'compute_combo'`

- [ ] **Step 1.3: Implement `compute_combo` in `data_fetcher.py`**

Add after the `compute_tfc_score` function (end of file):

```python
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
```

- [ ] **Step 1.4: Run tests to verify they pass**

```
python -m pytest tests/test_combo.py -v
```

Expected: all 15 tests PASS

- [ ] **Step 1.5: Commit**

```
git add data_fetcher.py tests/test_combo.py
git commit -m "feat: add compute_combo for Monthly-Weekly-Daily bar type labels"
```

---

## Task 2: `compute_signal_score` in `signals.py`

**Files:**
- Modify: `signals.py`
- Create: `tests/test_score.py`

- [ ] **Step 2.1: Write the failing tests**

Create `tests/test_score.py`:

```python
import pytest
from signals import compute_signal_score


class TestComputeSignalScore:
    def test_perfect_bullish_score_is_5(self):
        # TFC 4/5 Bullish + direction Bullish = +2
        # Vol High = +1, Weekly = +1, Hammer = +1
        score = compute_signal_score("Bullish", "4/5 Bullish", "High", "Weekly", "Hammer")
        assert score == 5

    def test_perfect_bearish_score_is_5(self):
        score = compute_signal_score("Bearish", "4/5 Bearish", "High", "Monthly", "Shooter")
        assert score == 5

    def test_score_zero_when_all_factors_miss(self):
        score = compute_signal_score("Bullish", "2/3 Bullish", "Normal", "Daily", "Inside Bar")
        assert score == 0

    def test_tfc_direction_mismatch_gives_zero_tfc_points(self):
        # TFC 4/5 Bearish but signal is Bullish → TFC factor = 0
        score = compute_signal_score("Bullish", "4/5 Bearish", "Normal", "Daily", "Inside Bar")
        assert score == 0

    def test_tfc_numerator_below_4_gives_zero_tfc_points(self):
        # 3/5 Bullish + Bullish signal → TFC factor = 0 (threshold is >= 4)
        score = compute_signal_score("Bullish", "3/5 Bullish", "Normal", "Daily", "Inside Bar")
        assert score == 0

    def test_tfc_5_of_5_matching_gives_2_points(self):
        score = compute_signal_score("Bullish", "5/5 Bullish", "Normal", "Daily", "Inside Bar")
        assert score == 2

    def test_empty_tfc_score_gives_zero_tfc_points(self):
        score = compute_signal_score("Bullish", "", "Normal", "Daily", "Inside Bar")
        assert score == 0

    def test_malformed_tfc_score_gives_zero_tfc_points(self):
        score = compute_signal_score("Bullish", "invalid", "Normal", "Daily", "Inside Bar")
        assert score == 0

    def test_high_volume_adds_1(self):
        score = compute_signal_score("Bullish", "", "High", "Daily", "Inside Bar")
        assert score == 1

    def test_normal_volume_adds_0(self):
        score = compute_signal_score("Bullish", "", "Normal", "Daily", "Inside Bar")
        assert score == 0

    def test_low_volume_adds_0(self):
        score = compute_signal_score("Bullish", "", "Low", "Daily", "Inside Bar")
        assert score == 0

    def test_weekly_timeframe_adds_1(self):
        score = compute_signal_score("Bullish", "", "Normal", "Weekly", "Inside Bar")
        assert score == 1

    def test_monthly_timeframe_adds_1(self):
        score = compute_signal_score("Bullish", "", "Normal", "Monthly", "Inside Bar")
        assert score == 1

    def test_daily_timeframe_adds_0(self):
        score = compute_signal_score("Bullish", "", "Normal", "Daily", "Inside Bar")
        assert score == 0

    def test_quarterly_timeframe_adds_0(self):
        score = compute_signal_score("Bullish", "", "Normal", "Quarterly", "Inside Bar")
        assert score == 0

    def test_shooter_signal_adds_1(self):
        score = compute_signal_score("Bearish", "", "Normal", "Daily", "Shooter")
        assert score == 1

    def test_hammer_signal_adds_1(self):
        score = compute_signal_score("Bullish", "", "Normal", "Daily", "Hammer")
        assert score == 1

    def test_plain_2u_signal_adds_0(self):
        score = compute_signal_score("Bullish", "", "Normal", "Daily", "2D Green")
        assert score == 0

    def test_neutral_direction_with_bullish_tfc_gives_zero_tfc_points(self):
        # Neutral direction can never match a directional TFC
        score = compute_signal_score("Neutral", "4/5 Bullish", "Normal", "Daily", "Inside Bar")
        assert score == 0

    def test_combined_vol_plus_weekly_plus_shooter_is_3(self):
        score = compute_signal_score("Bearish", "", "High", "Weekly", "Shooter")
        assert score == 3
```

- [ ] **Step 2.2: Run tests to verify they fail**

```
python -m pytest tests/test_score.py -v
```

Expected: `ImportError` or `AttributeError: module 'signals' has no attribute 'compute_signal_score'`

- [ ] **Step 2.3: Implement `compute_signal_score` in `signals.py`**

Add at the end of `signals.py`, after `compute_volume_ratio`:

```python
def compute_signal_score(
    sig_direction: str,
    tfc_score: str,
    vol_signal: str,
    timeframe: str,
    signal_name: str,
) -> int:
    """
    Compute a 0–5 signal strength score for a single scan result row.

    Factors:
        +2  TFC numerator >= 4 AND TFC direction matches sig_direction
        +1  vol_signal == "High"
        +1  timeframe in {"Weekly", "Monthly"}
        +1  signal_name in {"Shooter", "Hammer"}
    """
    score = 0

    if tfc_score:
        try:
            parts = tfc_score.split()          # ["4/5", "Bullish"]
            num = int(parts[0].split("/")[0])
            tfc_dir = parts[1]
            if num >= 4 and tfc_dir == sig_direction:
                score += 2
        except (IndexError, ValueError):
            pass

    if vol_signal == "High":
        score += 1

    if timeframe in {"Weekly", "Monthly"}:
        score += 1

    if signal_name in {"Shooter", "Hammer"}:
        score += 1

    return score
```

- [ ] **Step 2.4: Run tests to verify they pass**

```
python -m pytest tests/test_score.py -v
```

Expected: all 20 tests PASS

- [ ] **Step 2.5: Run the full test suite**

```
python -m pytest tests/ -v
```

Expected: all tests PASS (combo + score + previously written ftfc/tfc/volume tests)

- [ ] **Step 2.6: Commit**

```
git add signals.py tests/test_score.py
git commit -m "feat: add compute_signal_score for 0-5 signal strength ranking"
```

---

## Task 3: Wire Combo + Score into scan results in `app.py`

**Files:**
- Modify: `app.py`

This task adds Score, Combo, and Pattern to every result dict (both `results` and `mtf_context`) and stores `combo_map` in session state.

- [ ] **Step 3.1: Update imports at top of `app.py`**

Change line 18–19 from:
```python
from data_fetcher import compute_ftfc, compute_tfc_score, fetch_batch
from signals import classify_bar, compute_volume_ratio, scan_dataframe
```
To:
```python
from data_fetcher import compute_combo, compute_ftfc, compute_tfc_score, fetch_batch
from signals import classify_bar, compute_signal_score, compute_volume_ratio, scan_dataframe
```

- [ ] **Step 3.2: Add `combo_map` computation after `tfc_map` block**

Find this block (around line 414–427):
```python
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
```

Add immediately after it:
```python
    combo_map = {
        ticker: compute_combo(
            monthly_batch.get(ticker),
            weekly_batch.get(ticker),
            daily_batch.get(ticker),
        )
        for ticker in ticker_list
    }
```

- [ ] **Step 3.3: Add Score/Combo/Pattern to each row in the main `results` loop**

Find the block inside the `for sig in sigs:` loop that computes `vol_ratio, vol_signal` and `htf_high, htf_low` (around lines 456–475). After that block and before `results.append(...)`, add:

```python
                combo_str, pattern = combo_map.get(ticker, ("—", ""))
                score = compute_signal_score(
                    sig["direction"],
                    tfc_map.get(ticker, ""),
                    vol_signal,
                    tf,
                    sig["signal"],
                )
```

Then in the `results.append({...})` dict, add these three fields after `"Vol Signal": vol_signal,`:
```python
                        "Score":   score,
                        "Combo":   combo_str,
                        "Pattern": pattern or "—",
```

- [ ] **Step 3.4: Add Score/Combo/Pattern to each row in the `mtf_context` loop**

Find the `mtf_context.append({...})` dict (around line 564). Apply the same pattern:

Before the `mtf_context.append(...)` call, add:
```python
                combo_str, pattern = combo_map.get(ticker, ("—", ""))
                mtf_score = compute_signal_score(
                    sig["direction"],
                    tfc_map.get(ticker, ""),
                    vol_signal,
                    tf,
                    sig["signal"],
                )
```

Then add to `mtf_context.append({...})` after `"Vol Signal": vol_signal,`:
```python
                    "Score":   mtf_score,
                    "Combo":   combo_str,
                    "Pattern": pattern or "—",
```

- [ ] **Step 3.5: Store `combo_map` in session state**

Find the session state block (around line 587):
```python
    st.session_state["results"]     = results
    st.session_state["mtf_context"] = mtf_context
    st.session_state["errors"]      = errors
    st.session_state["scan_time"]   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

Add:
```python
    st.session_state["combo_map"]   = combo_map
```

- [ ] **Step 3.6: Read `combo_map` from session state in display section**

Find where `results`, `mtf_context`, `errors`, `scan_time` are read from session state (around line 598–602):
```python
results     = st.session_state["results"]
mtf_context = st.session_state.get("mtf_context", [])
errors      = st.session_state["errors"]
scan_time   = st.session_state["scan_time"]
```

Add:
```python
combo_map   = st.session_state.get("combo_map", {})
```

- [ ] **Step 3.7: Commit**

```
git add app.py
git commit -m "feat: wire Combo, Pattern, Score into scan result rows and session state"
```

---

## Task 4: Update flat view — column order, sort, and styling

**Files:**
- Modify: `app.py`

- [ ] **Step 4.1: Update `col_order` to include new columns**

Find (around line 654):
```python
col_order = ["Ticker", "Sector", "Timeframe", "Signal", "Direction", "Setup Type", "FTFC",
             "TFC Score", "Vol Ratio", "Vol Signal", "Bar Type", "Date", "Open", "High",
             "Low", "Close", "Prev High", "Prev Low", "HTF", "HTF High", "HTF Low"]
```

Replace with:
```python
col_order = ["Ticker", "Sector", "Timeframe", "Signal", "Direction", "Setup Type", "FTFC",
             "TFC Score", "Score", "Combo", "Pattern",
             "Vol Ratio", "Vol Signal", "Bar Type", "Date", "Open", "High",
             "Low", "Close", "Prev High", "Prev Low", "HTF", "HTF High", "HTF Low"]
```

- [ ] **Step 4.2: Change default flat view sort to Score-descending**

Find (around line 644):
```python
df_res = (
    df_res
    .sort_values(["_tf_sort", "Sector", "Ticker"])
    .drop(columns=["_tf_sort"])
    .reset_index(drop=True)
)
```

Replace with:
```python
df_res = (
    df_res
    .sort_values(["Score", "_tf_sort", "Sector", "Ticker"], ascending=[False, True, True, True])
    .drop(columns=["_tf_sort"])
    .reset_index(drop=True)
)
```

- [ ] **Step 4.3: Add `_style_score` and `_style_pattern` styling functions**

Add after the existing `_style_setup_type` function (around line 708):

```python
def _style_score(val) -> str:
    try:
        v = int(val)
    except (TypeError, ValueError):
        return "color: #555555"
    if v == 5:
        return "color: #00c853; font-weight: 700"
    if v >= 3:
        return "color: #ffd600; font-weight: 600"
    if v >= 1:
        return "color: #aaaaaa"
    return "color: #555555"


def _style_pattern(val: str) -> str:
    if val in {"1-2U", "3-2U"}:
        return "color: #00c853; font-weight: 600"
    if val in {"1-2D", "3-2D"}:
        return "color: #ff1744; font-weight: 600"
    if val == "1-1":
        return "color: #ffd600; font-weight: 600"
    return "color: #888888"
```

- [ ] **Step 4.4: Apply new stylers to the flat view `styled` object**

Find (around line 783):
```python
styled = (
    df_res.style
    .map(_style_direction,   subset=["Direction"])
    .map(_style_signal,      subset=["Signal"])
    .map(_style_ftfc,        subset=["FTFC"])
    .map(_style_tfc_score,   subset=["TFC Score"])
    .map(_style_vol_signal,  subset=["Vol Signal"])
    .map(_style_setup_type,  subset=["Setup Type"])
)
```

Replace with:
```python
styled = (
    df_res.style
    .map(_style_direction,   subset=["Direction"])
    .map(_style_signal,      subset=["Signal"])
    .map(_style_ftfc,        subset=["FTFC"])
    .map(_style_tfc_score,   subset=["TFC Score"])
    .map(_style_vol_signal,  subset=["Vol Signal"])
    .map(_style_setup_type,  subset=["Setup Type"])
    .map(_style_score,       subset=["Score"])
    .map(_style_pattern,     subset=["Pattern"])
)
```

- [ ] **Step 4.5: Commit**

```
git add app.py
git commit -m "feat: flat view — Score-descending sort, Combo/Pattern/Score columns with styling"
```

---

## Task 5: Update MTF pivot for Score, Combo, Pattern

**Files:**
- Modify: `app.py` — `build_mtf_pivot` function and its call site + MTF styled block

- [ ] **Step 5.1: Add `combo_map` parameter to `build_mtf_pivot`**

Find the function signature (around line 711):
```python
def build_mtf_pivot(results: list[dict], scan_tfs: list[str], sector_map: dict) -> pd.DataFrame:
```

Replace with:
```python
def build_mtf_pivot(results: list[dict], scan_tfs: list[str], sector_map: dict, combo_map: dict) -> pd.DataFrame:
```

- [ ] **Step 5.2: Pre-compute max score per ticker inside `build_mtf_pivot`**

Find the line inside the function that initializes `mtf: dict[str, dict] = {}` (around line 721). Add immediately after it:

```python
    max_score: dict[str, int] = {}
    for r in results:
        t = r["Ticker"]
        max_score[t] = max(max_score.get(t, 0), r.get("Score", 0))
```

- [ ] **Step 5.3: Add Score, Combo, Pattern to each MTF output row**

Find the output row building block inside the `for ticker, row in mtf.items():` loop (around line 752):
```python
        out: dict = {
            "Ticker":    row["Ticker"],
            "Sector":    row["Sector"],
            "FTFC":      row["FTFC"],
            "TFC Score": row.get("TFC Score", ""),
        }
```

Replace with:
```python
        combo_str, pattern = combo_map.get(ticker, ("—", ""))
        out: dict = {
            "Ticker":    row["Ticker"],
            "Sector":    row["Sector"],
            "FTFC":      row["FTFC"],
            "TFC Score": row.get("TFC Score", ""),
            "Score":     max_score.get(ticker, 0),
            "Combo":     combo_str,
            "Pattern":   pattern or "—",
        }
```

- [ ] **Step 5.4: Update the `build_mtf_pivot` call site**

Find (around line 885):
```python
    df_mtf = build_mtf_pivot(mtf_all, scan_tfs, _eff_map)
```

Replace with:
```python
    df_mtf = build_mtf_pivot(mtf_all, scan_tfs, _eff_map, combo_map)
```

- [ ] **Step 5.5: Apply Score and Pattern styling to MTF styled object**

Find (around line 901):
```python
        styled_mtf = (
            df_mtf.style
            .map(_style_ftfc,       subset=["FTFC"])
            .map(_style_tfc_score,  subset=["TFC Score"])
            .map(_style_confluence, subset=["Confluence"])
        )
```

Replace with:
```python
        styled_mtf = (
            df_mtf.style
            .map(_style_ftfc,       subset=["FTFC"])
            .map(_style_tfc_score,  subset=["TFC Score"])
            .map(_style_score,      subset=["Score"])
            .map(_style_pattern,    subset=["Pattern"])
            .map(_style_confluence, subset=["Confluence"])
        )
```

- [ ] **Step 5.6: Commit**

```
git add app.py
git commit -m "feat: MTF pivot — add Score, Combo, Pattern columns"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** `compute_combo` ✓ | `compute_signal_score` ✓ | `combo_map` pre-computed ✓ | session state ✓ | flat view columns ✓ | flat sort Score-desc ✓ | Score/Pattern styling ✓ | MTF pivot Score/Combo/Pattern ✓ | MTF styling ✓
- [x] **No placeholders:** All steps contain complete code.
- [x] **Type consistency:** `compute_combo` returns `tuple[str, str]` — used as `combo_str, pattern = combo_map.get(ticker, ("—", ""))` in Tasks 3 and 5. `compute_signal_score` returns `int` — stored as `"Score": score` throughout. `build_mtf_pivot` new signature used consistently in Task 5.
- [x] **Edge case covered:** `combo_map.get(ticker, ("—", ""))` default handles tickers that had no data. `pattern or "—"` converts empty string to dash for display.
