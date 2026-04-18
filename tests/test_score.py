import pytest
from signals import compute_signal_score


class TestComputeSignalScore:
    def test_perfect_bullish_score_is_5(self):
        score = compute_signal_score("Bullish", "4/5 Bullish", "High", "Weekly", "Hammer")
        assert score == 5

    def test_perfect_bearish_score_is_5(self):
        score = compute_signal_score("Bearish", "4/5 Bearish", "High", "Monthly", "Shooter")
        assert score == 5

    def test_score_zero_when_all_factors_miss(self):
        score = compute_signal_score("Bullish", "2/3 Bullish", "Normal", "Daily", "Inside Bar")
        assert score == 0

    def test_tfc_direction_mismatch_gives_zero_tfc_points(self):
        score = compute_signal_score("Bullish", "4/5 Bearish", "Normal", "Daily", "Inside Bar")
        assert score == 0

    def test_tfc_numerator_below_4_gives_zero_tfc_points(self):
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

    def test_plain_2d_green_signal_adds_0(self):
        score = compute_signal_score("Bullish", "", "Normal", "Daily", "2D Green")
        assert score == 0

    def test_neutral_direction_with_bullish_tfc_gives_zero_tfc_points(self):
        score = compute_signal_score("Neutral", "4/5 Bullish", "Normal", "Daily", "Inside Bar")
        assert score == 0

    def test_combined_vol_plus_weekly_plus_shooter_is_3(self):
        score = compute_signal_score("Bearish", "", "High", "Weekly", "Shooter")
        assert score == 3
