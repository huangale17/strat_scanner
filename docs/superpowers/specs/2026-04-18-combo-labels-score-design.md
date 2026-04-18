# Design: Combo Labels + Signal Strength Score

**Date:** 2026-04-18  
**Features:** Feature 3 (True Combo Labels) + Feature 4 (Signal Strength Score)  
**Approach:** Option B — helper functions in existing modules

---

## Overview

Two new columns added to both the flat view and MTF view of the Strat Scanner:

- **Combo** — Monthly-Weekly-Daily bar type string (e.g. `"2U-1-2U"`)
- **Pattern** — named high-probability combo if detected (e.g. `"1-2U"`, `"3-2D"`, or `"—"`)
- **Score** — signal strength integer 0–5; flat view sorted Score-descending by default

All data is already in memory at scan time — no additional network calls required.

---

## Feature 3: Combo Labels

### New function: `data_fetcher.compute_combo`

```python
def compute_combo(
    monthly_df: pd.DataFrame | None,
    weekly_df: pd.DataFrame | None,
    daily_df: pd.DataFrame | None,
) -> tuple[str, str]:
```

**Returns:** `(combo_str, named_pattern)`

**Logic:**
1. Classify the last closed bar of each TF using `classify_bar(last, prev)`.
2. If any TF has fewer than 2 bars, return `("—", "")`.
3. `combo_str = f"{monthly_type}-{weekly_type}-{daily_type}"` (e.g. `"2U-1-2U"`).
4. Scan both adjacent pairs for named patterns:
   - Pair 1: Monthly-Weekly → `f"{monthly_type}-{weekly_type}"`
   - Pair 2: Weekly-Daily → `f"{weekly_type}-{daily_type}"`
5. Named patterns (in priority order): `1-2U`, `1-2D`, `3-2U`, `3-2D`, `1-1`
6. If both pairs match, the lower TF pair (Weekly-Daily) wins.
7. `named_pattern` is the matched string, or `""` if none match.

### In `app.py`

Pre-compute after `ftfc_map` and `tfc_map`:

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

Each result row gets:
- `"Combo"`: `combo_map[ticker][0]` (the full string, e.g. `"2U-1-2U"`)
- `"Pattern"`: `combo_map[ticker][1]` or `"—"` if empty

### Styling

`_style_pattern(val)` applied to the `"Pattern"` column:

| Pattern | Color |
|---------|-------|
| `1-2U`, `3-2U` | `#00c853` green (bullish) |
| `1-2D`, `3-2D` | `#ff1744` red (bearish) |
| `1-1` | `#ffd600` yellow (neutral/coiling) |
| `"—"` | `#888888` grey |

---

## Feature 4: Signal Strength Score

### New function: `signals.compute_signal_score`

```python
def compute_signal_score(
    sig_direction: str,   # "Bullish", "Bearish", or "Neutral"
    tfc_score: str,       # e.g. "4/5 Bullish" or ""
    vol_signal: str,      # "High", "Normal", "Low", "N/A"
    timeframe: str,       # e.g. "Daily", "Weekly", "Monthly"
    signal_name: str,     # e.g. "Shooter", "Hammer", "Inside Bar"
) -> int:
```

**Returns:** integer 0–5

**Scoring table:**

| Factor | Points | Condition |
|--------|--------|-----------|
| Strong TFC alignment | +2 | TFC numerator ≥ 4 **and** TFC direction word matches `sig_direction` |
| High volume | +1 | `vol_signal == "High"` |
| Higher timeframe signal | +1 | `timeframe in {"Weekly", "Monthly"}` |
| Quality signal type | +1 | `signal_name in {"Shooter", "Hammer"}` |

**TFC parsing:** extract numerator and direction from `"N/total Direction"` string:
```python
parts = tfc_score.split()   # ["4/5", "Bullish"]
num = int(parts[0].split("/")[0])
direction = parts[1]
```
If string is empty or malformed, TFC factor scores 0.

### In `app.py`

Called per-signal row after vol and htf are computed:

```python
score = compute_signal_score(
    sig["direction"], tfc_map.get(ticker, ""), vol_signal, tf, sig["signal"]
)
```

Result row gets `"Score": score`.

### Flat view sort change

Old:
```python
df_res.sort_values(["_tf_sort", "Sector", "Ticker"])
```

New:
```python
df_res.sort_values(["Score", "_tf_sort", "Sector", "Ticker"], ascending=[False, True, True, True])
```

### Score styling

`_style_score(val)` applied to the `"Score"` column:

| Score | Style |
|-------|-------|
| 5 | `color: #00c853; font-weight: 700` (bright green bold) |
| 3–4 | `color: #ffd600; font-weight: 600` (yellow) |
| 1–2 | `color: #aaaaaa` (dim) |
| 0 | `color: #555555` (grey) |

---

## Column Layout

### Flat view (`col_order`)

```
Ticker | Sector | Timeframe | Signal | Direction | Setup Type | FTFC | TFC Score |
Score | Combo | Pattern | Vol Ratio | Vol Signal | Bar Type | Date |
Open | High | Low | Close | Prev High | Prev Low | HTF | HTF High | HTF Low
```

### MTF view (`build_mtf_pivot`)

- `combo_map` passed as a parameter to `build_mtf_pivot`.
- Each ticker row gets `"Combo"` and `"Pattern"` from `combo_map`.
- `"Score"` on each MTF row = max score across all signals for that ticker (already in results list).

```
Ticker | Sector | FTFC | TFC Score | Score | Combo | Pattern |
[Daily] | [Weekly] | ... | Confluence | HTF | HTF High | HTF Low
```

---

## Files Changed

| File | Change |
|------|--------|
| `data_fetcher.py` | Add `compute_combo()` |
| `signals.py` | Add `compute_signal_score()` |
| `app.py` | Import new functions; add `combo_map`; add Score/Combo/Pattern to result rows; update column order, sort, styling, MTF pivot call |

---

## Out of Scope

- Features 1 (Trade Plan) and 2 (HTF Alerts) — separate session
- No changes to data fetching, sidebar filters, CSV download, or copy-tickers button
- No new test files in this pass (existing 33 tests unchanged)
