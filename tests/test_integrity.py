"""Tests for material risks: leakage, marginal-quantile aggregation, stock balance."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import experiment as e


class ExperimentIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observed,cls.y,cls.promo,cls.traffic,cls.calendar=e.load_panel(ROOT/'data/retail_panel.csv')

    def test_future_sales_and_traffic_do_not_enter_inputs(self):
        origin=e.TEST_ORIGINS[0]
        original=e.make_inputs(origin,self.y,self.promo,self.traffic,self.calendar)
        y=self.y.copy()
        traffic=self.traffic.copy()
        y.loc[origin:]=1e9
        traffic.loc[origin:]=1e9
        changed=e.make_inputs(origin,y,self.promo,traffic,self.calendar)
        for a,b in zip(original,changed):
            np.testing.assert_array_equal(a,b)

    def test_causal_imputation_does_not_use_later_values(self):
        sample=pd.DataFrame({'y':np.arange(21,dtype=float)})
        sample.loc[10,'y']=np.nan
        a=e.causal_week_fill(sample)
        sample.loc[11:,'y']=1e9
        b=e.causal_week_fill(sample)
        self.assertEqual(a.loc[10,'y'],3.)
        self.assertEqual(a.loc[10,'y'],b.loc[10,'y'])

    def test_calibration_labels_end_before_test(self):
        self.assertLess(e.CAL_ORIGINS.max()+pd.Timedelta(days=13),e.TEST_ORIGINS.min())

    def test_conventional_features_ignore_future_sales_and_traffic(self):
        i=self.y.index.get_loc(e.TEST_ORIGINS[0])
        y=self.y.iloc[:,0].to_numpy().copy()
        p=self.promo.iloc[:,0].to_numpy()
        traffic=self.traffic.to_numpy().copy()
        calendar=self.calendar.to_numpy()
        original=e.direct_features(i,13,y,p,traffic,calendar)
        y[i:]=1e9
        traffic[i:]=1e9
        np.testing.assert_array_equal(original,e.direct_features(i,13,y,p,traffic,calendar))

    def test_saved_predictions_and_aggregate_buffer(self):
        p=pd.read_csv(ROOT/'results/forecasts.csv')
        keys=['model','origin','family','horizon']
        self.assertFalse(p.duplicated(keys).any())
        self.assertEqual(len(p),6*24*4*14)
        self.assertTrue(np.isfinite(p.prediction).all())
        self.assertTrue((p.prediction>=0).all())
        qcols=[f'q{int(q*100):02d}' for q in e.Q]
        self.assertTrue((np.diff(p[qcols],axis=1)>=-1e-5).all())
        before=e.aggregate_buffers(p)
        p.loc[p.split=='test','actual']=1e9
        p.loc[:,qcols]=1e9
        after=e.aggregate_buffers(p)
        pd.testing.assert_frame_equal(before,after)

    def test_inventory_mass_balance_and_fill_rate(self):
        daily=pd.read_csv(ROOT/'results/inventory_daily.csv')
        for (_,family),g in daily.groupby(['model','family']):
            initial=self.observed.loc['2017-05-10':'2017-05-16',family].sum()*2
            previous=np.r_[initial,g.ending_inventory.to_numpy()[:-1]]
            np.testing.assert_allclose(previous+g.arrival-g.served,g.ending_inventory,atol=1e-7)
            np.testing.assert_allclose(g.served+g.lost,g.demand,atol=1e-7)
            self.assertTrue((g.ending_inventory>=0).all())
        summary=pd.read_csv(ROOT/'results/inventory_metrics.csv')
        self.assertTrue(summary.fill_rate.between(0,1).all())

if __name__=='__main__':
    unittest.main(verbosity=2)
