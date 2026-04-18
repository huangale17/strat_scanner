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


# bar type fixtures — last bar vs second-to-last
_2U = _make_df([10, 12], [5, 6])   # breaks high only
_2D = _make_df([10, 9],  [5, 3])   # breaks low only
_1  = _make_df([10, 9],  [5, 6])   # inside bar
_3  = _make_df([10, 12], [5, 3])   # outside bar


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
        # Monthly=1, Weekly=1, Daily=2U → wd="1-2U" named, mw="1-1" named — wd wins
        _, pattern = compute_combo(_1, _1, _2U)
        assert pattern == "1-2U"

    def test_no_named_pattern_returns_empty_string(self):
        _, pattern = compute_combo(_2U, _2U, _2U)
        assert pattern == ""

    def test_no_named_pattern_combo_str_still_correct(self):
        combo_str, _ = compute_combo(_2U, _2U, _2U)
        assert combo_str == "2U-2U-2U"
