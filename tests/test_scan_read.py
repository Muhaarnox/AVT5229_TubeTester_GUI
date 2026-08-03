"""Tests for _read_measurement_point, _robust_average, and outlier detection."""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from scan_test_helpers import (
    _make_mock_client,
    _get_param_default,
    CalibrationData,
    ScanRange, ScanSettings, run_scan,
    _read_measurement_point, _robust_average,
)


class TestReadMeasurementPoint(unittest.TestCase):
    """Tests for _read_measurement_point with Ia/Ig2 averaging."""

    @patch("time.sleep")
    def test_single_sample(self, _):
        """With ia_samples=1, Ia and Ig2 each read once."""
        client = _make_mock_client()
        # Interleaved: Ia, Ig2, then Ua, Ug1, Ug2, Uh, Ih, Er
        client.get_param.side_effect = [500, 10, 250, 600, 250, 63, 30, 0]
        point = _read_measurement_point(client, calibration=CalibrationData(), ia_samples=1)
        self.assertAlmostEqual(point["ia"], 5.0)  # 500 * 0.01
        self.assertAlmostEqual(point["ig2"], 0.10)  # 10 / 100
        self.assertAlmostEqual(point["ua"], 250.0)
        self.assertEqual(point["er"], 0)
        self.assertEqual(client.get_param.call_count, 8)

    @patch("time.sleep")
    def test_averaging_3_samples(self, _):
        """With ia_samples=3, robust averaging (median) is used."""
        client = _make_mock_client()
        # 3 full reads: (Ia, Ig2, Ua, Ug1, Ug2, Uh, Ih) x 3, then Er
        client.get_param.side_effect = [
            490, 8, 250, 600, 250, 63, 30,   # sample 1
            500, 10, 250, 600, 250, 63, 30,  # sample 2
            510, 12, 250, 600, 250, 63, 30,  # sample 3
            0,                                # Er
        ]
        point = _read_measurement_point(client, calibration=CalibrationData(), ia_samples=3)
        self.assertAlmostEqual(point["ia"], 5.0)    # median(490,500,510)*0.01 = 500*0.01
        self.assertAlmostEqual(point["ig2"], 0.10)   # median(8,10,12)/100 = 10/100
        # 3 × 7 params + 1 Er = 22 calls
        self.assertEqual(client.get_param.call_count, 22)

    @patch("time.sleep")
    def test_ia_samples_zero_treated_as_one(self, _):
        """ia_samples=0 should behave as ia_samples=1 (safety clamp)."""
        client = _make_mock_client()
        client.get_param.side_effect = [500, 10, 250, 600, 250, 63, 30, 0]
        point = _read_measurement_point(client, calibration=CalibrationData(), ia_samples=0)
        self.assertAlmostEqual(point["ia"], 5.0)
        self.assertAlmostEqual(point["ig2"], 0.10)
        # max(0, 1) = 1 → same as single sample: 7 + 1 Er = 8 calls
        self.assertEqual(client.get_param.call_count, 8)


class TestRobustAverage(unittest.TestCase):
    """Tests for _robust_average helper."""

    def test_single_value(self):
        self.assertAlmostEqual(_robust_average([5.0]), 5.0)

    def test_two_values_mean(self):
        self.assertAlmostEqual(_robust_average([4.0, 6.0]), 5.0)

    def test_three_values_median(self):
        """3 values: median (middle element)."""
        self.assertAlmostEqual(_robust_average([1.0, 5.0, 100.0]), 5.0)

    def test_empty_returns_zero(self):
        """Empty list should return 0.0 (defensive guard)."""
        self.assertAlmostEqual(_robust_average([]), 0.0)

    def test_four_values_median(self):
        """4 values: true median of [1,5,6,100] = (5+6)/2 = 5.5."""
        self.assertAlmostEqual(_robust_average([1.0, 5.0, 6.0, 100.0]), 5.5)

    def test_five_values_trimmed_mean(self):
        """5 values: drop min+max, mean of middle 3."""
        vals = [1.0, 4.9, 5.0, 5.1, 100.0]
        self.assertAlmostEqual(_robust_average(vals), 5.0)

    def test_seven_values_trimmed_mean(self):
        """7 values: drop min+max, mean of middle 5."""
        vals = [0.1, 4.8, 4.9, 5.0, 5.1, 5.2, 99.0]
        expected = (4.8 + 4.9 + 5.0 + 5.1 + 5.2) / 5.0
        self.assertAlmostEqual(_robust_average(vals), expected)


class TestReadMeasurementPointOutlier(unittest.TestCase):
    """Tests for outlier detection and re-read in _read_measurement_point."""

    @patch("time.sleep")
    def test_outlier_triggers_reread(self, _):
        """Spike in Ia (range-switch) triggers N extra reads; result uses robust avg."""
        client = _make_mock_client()
        # 3 initial reads: one has a spike (12530 raw → 125.3 mA)
        # Then 3 re-reads (normal values), then Er
        client.get_param.side_effect = [
            1710, 300, 270, 1210, 270, 63, 30,    # sample 1: ia=17.10
            12530, 300, 270, 1210, 270, 63, 30,   # sample 2: ia=125.30 SPIKE
            1740, 300, 270, 1210, 270, 63, 30,    # sample 3: ia=17.40
            # outlier detected: max/min = 125.3/17.1 > 2.0 → re-read 3 more
            1720, 300, 270, 1210, 270, 63, 30,    # sample 4: ia=17.20
            1730, 300, 270, 1210, 270, 63, 30,    # sample 5: ia=17.30
            1725, 300, 270, 1210, 270, 63, 30,    # sample 6: ia=17.25
            0,                                      # Er
        ]
        point = _read_measurement_point(
            client, calibration=CalibrationData(), ia_samples=3,
            ia_outlier_ratio=2.0)
        # Pool [17.10, 125.30, 17.40, 17.20, 17.30, 17.25]: MAD rejection
        # drops only the spike → mean of the five survivors = 17.25
        self.assertAlmostEqual(point["ia"], 17.25, places=3)
        self.assertEqual(client.get_param.call_count, 6 * 7 + 1)

    @patch("time.sleep")
    def test_no_outlier_no_reread(self, _):
        """Normal readings: no re-read, still uses robust averaging (median for 3)."""
        client = _make_mock_client()
        client.get_param.side_effect = [
            1710, 300, 270, 1210, 270, 63, 30,
            1720, 300, 270, 1210, 270, 63, 30,
            1730, 300, 270, 1210, 270, 63, 30,
            0,                                     # Er
        ]
        point = _read_measurement_point(
            client, calibration=CalibrationData(), ia_samples=3,
            ia_outlier_ratio=2.0)
        # Median of [17.10, 17.20, 17.30] = 17.20
        self.assertAlmostEqual(point["ia"], 17.20, places=2)
        self.assertEqual(client.get_param.call_count, 3 * 7 + 1)

    @patch("time.sleep")
    def test_low_current_no_outlier_check(self, _):
        """Near-zero Ia: max/min ratio is high but max is below floor → no re-read."""
        client = _make_mock_client()
        client.get_param.side_effect = [
            1, 0, 50, 2000, 50, 63, 30,     # ia=0.01
            3, 0, 50, 2000, 50, 63, 30,     # ia=0.03
            2, 0, 50, 2000, 50, 63, 30,     # ia=0.02
            0,                                # Er
        ]
        point = _read_measurement_point(
            client, calibration=CalibrationData(), ia_samples=3,
            ia_outlier_ratio=2.0)
        # max(0.01, 0.03, 0.02) = 0.03 < floor (0.5) → no re-read
        self.assertEqual(client.get_param.call_count, 3 * 7 + 1)
        # Median of [0.01, 0.02, 0.03] = 0.02
        self.assertAlmostEqual(point["ia"], 0.02, places=3)

    @patch("time.sleep")
    def test_collapsed_sample_detected(self, _):
        """A sample collapsing far below the floor (bad contact) is an outlier.

        The floor gates on max, not min: gating on min would skip the check
        entirely here and let the worst instability pass unreported.
        """
        client = _make_mock_client()
        client.get_param.side_effect = [
            2000, 300, 270, 1210, 270, 63, 30,   # sample 1: ia=20.00
            20, 300, 270, 1210, 270, 63, 30,     # sample 2: ia=0.20 COLLAPSE
            2010, 300, 270, 1210, 270, 63, 30,   # sample 3: ia=20.10
            # max/min = 100 > 2.0 → re-read 3 more
            2020, 300, 270, 1210, 270, 63, 30,   # sample 4: ia=20.20
            2030, 300, 270, 1210, 270, 63, 30,   # sample 5: ia=20.30
            2040, 300, 270, 1210, 270, 63, 30,   # sample 6: ia=20.40
            0,                                     # Er
        ]
        point = _read_measurement_point(
            client, calibration=CalibrationData(), ia_samples=3,
            ia_outlier_ratio=2.0)
        self.assertEqual(client.get_param.call_count, 6 * 7 + 1)
        # Pool [20.00, 0.20, 20.10, 20.20, 20.30, 20.40]: only the collapsed
        # sample is far from the median → mean of the rest = 20.20
        self.assertAlmostEqual(point["ia"], 20.20, places=3)

    @patch("time.sleep")
    def test_zero_sample_detected_without_zero_division(self, _):
        """Ia reading exactly 0 → undefined ratio, treated as an outlier."""
        client = _make_mock_client()
        client.get_param.side_effect = [
            1500, 300, 270, 1210, 270, 63, 30,   # sample 1: ia=15.00
            0, 300, 270, 1210, 270, 63, 30,      # sample 2: ia=0.00
            1510, 300, 270, 1210, 270, 63, 30,   # sample 3: ia=15.10
            1520, 300, 270, 1210, 270, 63, 30,   # re-read 1
            1530, 300, 270, 1210, 270, 63, 30,   # re-read 2
            1540, 300, 270, 1210, 270, 63, 30,   # re-read 3
            0,                                     # Er
        ]
        point = _read_measurement_point(
            client, calibration=CalibrationData(), ia_samples=3,
            ia_outlier_ratio=2.0)
        self.assertEqual(client.get_param.call_count, 6 * 7 + 1)
        # Pool [15.00, 0.00, 15.10, 15.20, 15.30, 15.40]: the zero is
        # rejected, the four good highs are not → mean = 15.20
        self.assertAlmostEqual(point["ia"], 15.20, places=3)

    @patch("time.sleep")
    def test_collapse_below_floor_not_checked(self, _):
        """Whole point below the floor: near-zero noise, ratio means nothing."""
        client = _make_mock_client()
        client.get_param.side_effect = [
            40, 0, 50, 2000, 50, 63, 30,    # ia=0.40 (below floor)
            5, 0, 50, 2000, 50, 63, 30,     # ia=0.05
            40, 0, 50, 2000, 50, 63, 30,    # ia=0.40
            0,                                # Er
        ]
        point = _read_measurement_point(
            client, calibration=CalibrationData(), ia_samples=3,
            ia_outlier_ratio=2.0)
        self.assertEqual(client.get_param.call_count, 3 * 7 + 1)
        self.assertAlmostEqual(point["ia"], 0.40, places=3)

    @patch("time.sleep")
    def test_ratio_exactly_at_threshold_does_not_trigger(self, _):
        """spread == ratio is not an outlier (strict >), one LSB more is."""
        at_threshold = _make_mock_client()
        at_threshold.get_param.side_effect = [
            1000, 0, 250, 1210, 175, 63, 30,    # ia=10.00
            2000, 0, 250, 1210, 175, 63, 30,    # ia=20.00 → ratio exactly 2.0
            1000, 0, 250, 1210, 175, 63, 30,
            0,                                    # Er
        ]
        _read_measurement_point(
            at_threshold, calibration=CalibrationData(), ia_samples=3,
            ia_outlier_ratio=2.0)
        self.assertEqual(at_threshold.get_param.call_count, 3 * 7 + 1)

        above = _make_mock_client()
        above.get_param.side_effect = [
            1000, 0, 250, 1210, 175, 63, 30,    # ia=10.00
            2001, 0, 250, 1210, 175, 63, 30,    # ia=20.01 → just above
            1000, 0, 250, 1210, 175, 63, 30,
            1000, 0, 250, 1210, 175, 63, 30,
            1000, 0, 250, 1210, 175, 63, 30,
            1000, 0, 250, 1210, 175, 63, 30,
            0,                                    # Er
        ]
        _read_measurement_point(
            above, calibration=CalibrationData(), ia_samples=3,
            ia_outlier_ratio=2.0)
        self.assertEqual(above.get_param.call_count, 6 * 7 + 1)

    @patch("time.sleep")
    def test_reread_batch_size_is_honoured(self, _):
        """The extra batch is ia_outlier_reread_samples, not ia_samples."""
        client = _make_mock_client()
        client.get_param.side_effect = [
            1710, 300, 270, 1210, 270, 63, 30,    # ia=17.10
            12530, 300, 270, 1210, 270, 63, 30,   # ia=125.30 SPIKE
            1740, 300, 270, 1210, 270, 63, 30,    # ia=17.40
            1720, 300, 270, 1210, 270, 63, 30,    # single re-read
            0,                                      # Er
        ]
        point = _read_measurement_point(
            client, calibration=CalibrationData(), ia_samples=3,
            ia_outlier_ratio=2.0, ia_outlier_reread_samples=1)
        self.assertEqual(client.get_param.call_count, 4 * 7 + 1)
        # Pool of 4 → spike rejected, mean of [17.10, 17.40, 17.20] = 17.233
        self.assertAlmostEqual(point["ia"], 17.2333, places=3)

    @patch("time.sleep")
    def test_reread_zero_warns_without_extra_reads(self, _):
        """0 = diagnostics only: no extra batch, robust average of what we have."""
        client = _make_mock_client()
        client.get_param.side_effect = [
            1710, 300, 270, 1210, 270, 63, 30,
            12530, 300, 270, 1210, 270, 63, 30,
            1740, 300, 270, 1210, 270, 63, 30,
            0,                                      # Er
        ]
        stats = {}
        point = _read_measurement_point(
            client, calibration=CalibrationData(), ia_samples=3,
            ia_outlier_ratio=2.0, ia_outlier_reread_samples=0, stats=stats)
        self.assertEqual(client.get_param.call_count, 3 * 7 + 1)
        # Counter still fires — the point is still flagged as unstable.
        # No "unstable" counter though: a pool of 3 gives rejection nothing
        # to work with, so that verdict would just restate the detection.
        self.assertEqual(stats, {"ia_outlier_rereads": 1})
        # Nothing rejected → median of [17.10, 125.30, 17.40] = 17.40
        self.assertAlmostEqual(point["ia"], 17.40, places=2)

    @patch("time.sleep")
    def test_floor_holds_for_degenerate_ratio(self, _):
        """ratio < 1 fires on every armed point, but the floor still gates.

        max/min is >= 1 by construction, so any ratio below 1 means "always".
        Below the floor there is nothing to judge, so it must stay silent.
        """
        client = _make_mock_client()
        client.get_param.side_effect = [
            40, 0, 50, 2000, 50, 63, 30,    # ia=0.40 (below floor)
            10, 0, 50, 2000, 50, 63, 30,    # ia=0.10
            40, 0, 50, 2000, 50, 63, 30,    # ia=0.40
            0,                                # Er
        ]
        _read_measurement_point(
            client, calibration=CalibrationData(), ia_samples=3,
            ia_outlier_ratio=0.5)
        self.assertEqual(client.get_param.call_count, 3 * 7 + 1)

    @patch("time.sleep")
    def test_floor_boundary_is_strict(self, _):
        """max == floor does not arm the check; one LSB above it does."""
        at_floor = _make_mock_client()
        at_floor.get_param.side_effect = [
            50, 0, 50, 2000, 50, 63, 30,    # ia=0.50 == floor
            5, 0, 50, 2000, 50, 63, 30,     # ia=0.05
            50, 0, 50, 2000, 50, 63, 30,    # ia=0.50
            0,                                # Er
        ]
        _read_measurement_point(
            at_floor, calibration=CalibrationData(), ia_samples=3,
            ia_outlier_ratio=2.0)
        self.assertEqual(at_floor.get_param.call_count, 3 * 7 + 1)

        above = _make_mock_client()
        above.get_param.side_effect = [
            51, 0, 50, 2000, 50, 63, 30,    # ia=0.51 > floor
            5, 0, 50, 2000, 50, 63, 30,     # ia=0.05
            51, 0, 50, 2000, 50, 63, 30,    # ia=0.51
            51, 0, 50, 2000, 50, 63, 30,    # re-read 1
            51, 0, 50, 2000, 50, 63, 30,    # re-read 2
            51, 0, 50, 2000, 50, 63, 30,    # re-read 3
            0,                                # Er
        ]
        _read_measurement_point(
            above, calibration=CalibrationData(), ia_samples=3,
            ia_outlier_ratio=2.0)
        self.assertEqual(above.get_param.call_count, 6 * 7 + 1)

    @patch("time.sleep")
    def test_outlier_disabled(self, _):
        """ia_outlier_ratio=0: no re-read even with spike, but still robust avg for n>=3."""
        client = _make_mock_client()
        client.get_param.side_effect = [
            1710, 300, 270, 1210, 270, 63, 30,
            12530, 300, 270, 1210, 270, 63, 30,
            1740, 300, 270, 1210, 270, 63, 30,
            0,                                     # Er
        ]
        point = _read_measurement_point(
            client, calibration=CalibrationData(), ia_samples=3,
            ia_outlier_ratio=0.0)
        # No re-read, but n=3 → median of [17.10, 125.30, 17.40] = 17.40
        self.assertAlmostEqual(point["ia"], 17.40, places=2)
        self.assertEqual(client.get_param.call_count, 3 * 7 + 1)

    @patch("time.sleep")
    def test_two_samples_no_outlier_check(self, _):
        """ia_samples=2: outlier check skipped (needs >= 3), plain mean used."""
        client = _make_mock_client()
        client.get_param.side_effect = [
            1710, 300, 270, 1210, 270, 63, 30,
            12530, 300, 270, 1210, 270, 63, 30,
            0,                                     # Er
        ]
        point = _read_measurement_point(
            client, calibration=CalibrationData(), ia_samples=2,
            ia_outlier_ratio=2.0)
        # n=2 → plain mean: (17.10 + 125.30) / 2 = 71.20
        self.assertAlmostEqual(point["ia"], 71.20, places=2)
        self.assertEqual(client.get_param.call_count, 2 * 7 + 1)


class TestOutlierRejection(unittest.TestCase):
    """MAD rejection of the pool + the control pass after it."""

    def test_rejects_the_spike_not_the_scatter(self):
        from lm19.scan.io import _reject_outlier_indices
        # One spike among samples that scatter by a few LSB: the spike must
        # go and nothing else, or the value drifts toward the median.
        vals = [17.10, 125.30, 17.40, 17.20, 17.30, 17.25]
        kept, rejected = _reject_outlier_indices(vals)
        self.assertEqual(rejected, [1])
        self.assertEqual(kept, [0, 2, 3, 4, 5])

    def test_rejects_two_correlated_spikes(self):
        """Two spikes on the same side — the case a trimmed mean cannot fix."""
        from lm19.scan.io import _reject_outlier_indices
        vals = [5.00, 40.00, 5.10, 41.00, 5.05, 5.00]
        kept, rejected = _reject_outlier_indices(vals)
        self.assertEqual(rejected, [1, 3])
        self.assertAlmostEqual(
            sum(vals[i] for i in kept) / len(kept), 5.0375, places=4)

    def test_zero_mad_keeps_quantization_scatter(self):
        """MAD = 0 must not turn one LSB of scatter into an outlier.

        A pool of near-identical samples collapses MAD to zero, and with a
        bare k*MAD tolerance every sample but the median would be thrown
        away — that is what the absolute floor is for.
        """
        from lm19.scan.io import _reject_outlier_indices
        kept, rejected = _reject_outlier_indices([7.0, 7.0, 7.0, 7.02, 7.0])
        self.assertEqual(rejected, [])
        self.assertEqual(len(kept), 5)
        # ... while a real spike in the same zero-MAD pool still goes
        kept, rejected = _reject_outlier_indices([7.0, 7.0, 7.0, 7.02, 90.0])
        self.assertEqual(rejected, [4])

    def test_majority_always_survives(self):
        """The rule can never strip the pool down to a couple of samples.

        Half the deviations are <= MAD by definition of a median, and the
        tolerance is >= 3 MADs — so no distribution, however wild, leaves
        a minority to average. Pinned on pools built to break it.
        """
        from lm19.scan.io import _reject_outlier_indices
        wild_pools = [
            [5.0, 5.02, 5.04, 80.0, 200.0, 400.0],   # half far away
            [5.0, 5.0, 5.0, 100.0, 200.0, 400.0],    # MAD = 0, half far
            [0.0, 0.0, 900.0, 900.0],                # two clusters
            [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],        # no cluster at all
        ]
        for vals in wild_pools:
            kept, _rejected = _reject_outlier_indices(vals)
            self.assertGreaterEqual(len(kept), (len(vals) + 1) // 2, vals)

    def test_small_pool_is_left_alone(self):
        """Below the keep floor there is nothing to reject from."""
        from lm19.scan.io import _reject_outlier_indices
        kept, rejected = _reject_outlier_indices([1.0, 9.0, 1.0])
        self.assertEqual(rejected, [])
        self.assertEqual(kept, [0, 1, 2])

    @patch("time.sleep")
    def test_ig2_follows_the_rejected_ia_positions(self, _):
        """A transient corrupts the whole sample, not the Ia channel alone."""
        client = _make_mock_client()
        client.get_param.side_effect = [
            1000, 100, 250, 1210, 175, 63, 30,    # ia=10.00 ig2=1.00
            9000, 900, 250, 1210, 175, 63, 30,    # SPIKE: ia=90.00 ig2=9.00
            1000, 100, 250, 1210, 175, 63, 30,
            1000, 100, 250, 1210, 175, 63, 30,
            1000, 100, 250, 1210, 175, 63, 30,
            1000, 100, 250, 1210, 175, 63, 30,
            0,                                      # Er
        ]
        point = _read_measurement_point(
            client, calibration=CalibrationData(), ia_samples=3,
            ia_outlier_ratio=2.0)
        self.assertAlmostEqual(point["ia"], 10.00, places=3)
        # Ig2 averaged over the same five surviving samples — 9.00 dropped
        self.assertAlmostEqual(point["ig2"], 1.00, places=3)

    @patch("time.sleep")
    def test_unstable_point_counted_after_rejection(self, _):
        """Scatter that rejection cannot clean is reported, not hidden."""
        client = _make_mock_client()
        client.get_param.side_effect = [
            500, 0, 250, 1210, 175, 63, 30,     # 5.00
            1500, 0, 250, 1210, 175, 63, 30,    # 15.00
            500, 0, 250, 1210, 175, 63, 30,     # 5.00
            1500, 0, 250, 1210, 175, 63, 30,    # 15.00
            500, 0, 250, 1210, 175, 63, 30,     # 5.00
            1500, 0, 250, 1210, 175, 63, 30,    # 15.00
            0,                                    # Er
        ]
        stats = {}
        point = _read_measurement_point(
            client, calibration=CalibrationData(), ia_samples=3,
            ia_outlier_ratio=2.0, stats=stats)
        # Nothing stands out from this scatter → nothing rejected → median
        self.assertAlmostEqual(point["ia"], 10.00, places=3)
        self.assertEqual(stats,
                         {"ia_outlier_rereads": 1, "ia_unstable_points": 1})

    @patch("time.sleep")
    def test_cleaned_point_is_not_counted_unstable(self, _):
        """The discriminator: one spike, cleanly removed → no unstable flag."""
        client = _make_mock_client()
        client.get_param.side_effect = [
            1000, 0, 250, 1210, 175, 63, 30,
            9000, 0, 250, 1210, 175, 63, 30,    # single spike
            1000, 0, 250, 1210, 175, 63, 30,
            1000, 0, 250, 1210, 175, 63, 30,
            1000, 0, 250, 1210, 175, 63, 30,
            1000, 0, 250, 1210, 175, 63, 30,
            0,                                    # Er
        ]
        stats = {}
        _read_measurement_point(
            client, calibration=CalibrationData(), ia_samples=3,
            ia_outlier_ratio=2.0, stats=stats)
        self.assertEqual(stats, {"ia_outlier_rereads": 1})


class TestScanSummaryCompleteness(unittest.TestCase):
    """The emitted scan_summary carries exactly the declared fields.

    Key set derived from the TypedDict, not hand-listed: a counter added
    to ``_ScanSummary`` but never emitted fails here.
    """

    @patch("time.sleep")
    def test_summary_event_matches_typed_dict(self, _):
        from lm19.scan.events import _ScanSummary
        client = _make_mock_client()
        client.get_param.side_effect = _get_param_default(100)
        events = []
        settings = ScanSettings(
            ua=ScanRange(100, 100, 0),
            ug1=ScanRange(-2.0, -2.0, 0),
            ug2=ScanRange(0, 0, 0),
            uh=6.3, ih=0.3, is_triode=True,
            calibration=CalibrationData(),
        )
        run_scan(client, settings, progress=events.append)
        summary = [e for e in events if e.get("event") == "scan_summary"]
        self.assertEqual(len(summary), 1)
        self.assertEqual(set(summary[0]), set(_ScanSummary.__annotations__))


class TestScanForwardsOutlierSettings(unittest.TestCase):
    """run_scan must hand its own outlier settings to the read helper.

    The unit tests above prove the helper honours its arguments; only a
    call-site spy proves the caller supplies them.
    """

    @patch("time.sleep")
    def test_run_scan_forwards_ratio_and_reread(self, _):
        client = _make_mock_client()
        client.get_param.side_effect = _get_param_default(100)
        seen = []

        def _spy(*args, **kwargs):
            seen.append(kwargs)
            return {"ua": 100.0, "ug1": -2.0, "ug2": 0.0, "ia": 5.0,
                    "ig2": 0.0, "uh": 6.3, "ih": 0.3, "er": 0}

        settings = ScanSettings(
            ua=ScanRange(100, 100, 0),
            ug1=ScanRange(-2.0, -2.0, 0),
            ug2=ScanRange(0, 0, 0),
            uh=6.3, ih=0.3, is_triode=True,
            calibration=CalibrationData(),
            ia_samples=3,
            ia_outlier_ratio=1.7,          # all three differ from each
            ia_outlier_reread_samples=7,   # other and from the defaults
        )
        with patch("lm19.scan.runner._read_measurement_point", _spy):
            run_scan(client, settings)

        self.assertTrue(seen, "read helper was never called")
        self.assertEqual(seen[0]["ia_outlier_ratio"], 1.7)
        self.assertEqual(seen[0]["ia_outlier_reread_samples"], 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
