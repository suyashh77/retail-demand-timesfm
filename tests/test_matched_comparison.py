"""Material checks for equal local windows, channel exclusion and calibration."""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import matched_comparison as m


class MatchedInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = m.load_panel(ROOT / 'data/retail_panel.csv')
        cls.origin = m.TEST_ORIGINS[0]

    def test_excluded_covariates_cannot_change_features(self):
        observed, y, promo, traffic, calendar = self.panel
        for mode in ('sales', 'calendar'):
            before = m.design(self.origin, *self.panel, mode)
            excluded_calendar = calendar * np.nan if mode == 'sales' else calendar
            after = m.design(self.origin, observed, y, promo * np.nan,
                             traffic * np.nan, excluded_calendar, mode)
            for field in ('x', 'labels', 'future'):
                np.testing.assert_array_equal(before[field], after[field])

    def test_outside_window_and_future_targets_and_traffic_are_excluded(self):
        observed, y, promo, traffic, calendar = [x.copy() for x in self.panel]
        start = self.origin - pd.Timedelta(days=m.CONTEXT)
        changed = [observed, y, promo, traffic, calendar]
        for channel in changed:
            channel.loc[channel.index < start] = 1e9
        for channel in (observed, y, traffic):
            channel.loc[channel.index >= self.origin] = 1e9
        for mode in m.MODES:
            before = m.design(self.origin, *self.panel, mode)
            after = m.design(self.origin, *changed, mode)
            for field in ('x', 'labels', 'future'):
                np.testing.assert_array_equal(before[field], after[field])
            self.assertEqual(before['history_start'], start)
            self.assertEqual(before['history_end'], self.origin - pd.Timedelta(days=1))
            self.assertLess(before['last_label'], self.origin)
            self.assertGreaterEqual(before['example_origins'].min(), 28)
            self.assertLess(before['target_indices'].max(), m.CONTEXT)

    def test_all_related_sales_groups_are_exposed(self):
        before = m.design(self.origin, *self.panel, 'sales')
        observed, y, promo, traffic, calendar = self.panel
        changed = y.copy()
        changed.loc[self.origin - pd.Timedelta(days=28):self.origin - pd.Timedelta(days=1), 'DAIRY'] += 500
        after = m.design(self.origin, observed, changed, promo, traffic, calendar, 'sales')
        dairy = [i for i, name in enumerate(before['feature_names']) if name.startswith('DAIRY:')]
        self.assertTrue(dairy)
        self.assertFalse(np.array_equal(before['future'][:, dairy], after['future'][:, dairy]))
        self.assertEqual(before['labels'].shape[1], 4)

    def test_interval_offsets_ignore_test_outcomes(self):
        rows = []
        for split, residuals in [('calibration', [-3, -1, 0, 1, 4]), ('test', [10, 20])]:
            for residual in residuals:
                rows.append(dict(model='test', family='test', split=split,
                                 actual=100 + residual, prediction=100.))
        points = pd.DataFrame(rows)
        before = m.attach_quantiles(points)
        points.loc[points.split == 'test', 'actual'] = 1e9
        after = m.attach_quantiles(points)
        columns = [f'q{q}' for q in range(10, 100, 10)]
        np.testing.assert_array_equal(before[columns], after[columns])


if __name__ == '__main__':
    unittest.main()
