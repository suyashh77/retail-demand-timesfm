"""Prepare a fixed Favorita slice without inventing missing sales or prices."""
import argparse
import json
from pathlib import Path
import pandas as pd

FAMILIES = ['BEVERAGES', 'BREAD/BAKERY', 'DAIRY', 'GROCERY I']
STORE = 1

def prepare(raw: Path, destination: Path):
    parts = []
    for chunk in pd.read_csv(raw / 'train.csv', chunksize=250_000):
        parts.append(chunk.loc[(chunk.store_nbr == STORE) & chunk.family.isin(FAMILIES)])
    sales = pd.concat(parts, ignore_index=True)
    sales['date'] = pd.to_datetime(sales.date)
    assert not sales.duplicated(['date', 'family']).any()
    sales = sales[sales.date >= '2015-01-01'].copy()
    # Christmas is absent from the source, rather than a verified zero-sales day.
    # It remains missing in the exported panel. Context imputation is causal in experiment.py.
    dates = pd.date_range(sales.date.min(), sales.date.max(), freq='D')
    grid = pd.MultiIndex.from_product([dates, FAMILIES], names=['date', 'family'])
    panel = sales.set_index(['date', 'family']).reindex(grid).reset_index()
    panel['store_nbr'] = STORE
    trans = pd.read_csv(raw / 'transactions.csv', parse_dates=['date'])
    panel = panel.merge(trans.loc[trans.store_nbr == STORE, ['date', 'transactions']], on='date', how='left')
    holidays = pd.read_csv(raw / 'holidays_events.csv', parse_dates=['date'])
    applicable = ((holidays.locale == 'National') |
                  ((holidays.locale == 'Regional') & (holidays.locale_name == 'Pichincha')) |
                  ((holidays.locale == 'Local') & (holidays.locale_name == 'Quito')))
    scheduled = holidays.type.isin(['Holiday', 'Transfer', 'Bridge', 'Additional'])
    active = holidays.loc[applicable & scheduled & ~holidays.transferred.astype(bool)]
    panel['holiday'] = panel.date.isin(active.date).astype(int)
    panel['dow_sin'] = __import__('numpy').sin(2 * __import__('numpy').pi * panel.date.dt.dayofweek / 7)
    panel['dow_cos'] = __import__('numpy').cos(2 * __import__('numpy').pi * panel.date.dt.dayofweek / 7)
    panel['month_sin'] = __import__('numpy').sin(2 * __import__('numpy').pi * panel.date.dt.dayofyear / 365.25)
    panel['month_cos'] = __import__('numpy').cos(2 * __import__('numpy').pi * panel.date.dt.dayofyear / 365.25)
    panel['payday'] = ((panel.date.dt.day == 15) | panel.date.dt.is_month_end).astype(int)
    destination.mkdir(parents=True, exist_ok=True)
    panel.to_csv(destination / 'retail_panel.csv', index=False)
    metadata = {'store': STORE, 'city': 'Quito', 'families': FAMILIES,
                'first_date': str(panel.date.min().date()), 'last_date': str(panel.date.max().date()),
                'rows': len(panel), 'observed_sales_rows': int(panel.sales.notna().sum()),
                'missing_sales_rows': int(panel.sales.isna().sum()),
                'target': 'Observed product-family sales; fractional sales possible; not latent uncensored demand',
                'price_available': False, 'future_promotion_availability': 'Assumed planned; source has no plan-version timestamps',
                'holiday_handling': 'National plus Quito/Pichincha; transferred dates excluded; Transfer dates included; unscheduled Events excluded'}
    (destination / 'dataset_card.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    args = parser.parse_args()
    prepare(args.raw_dir, args.out_dir)
