"""Land embodied in the oil exchange and in the output option, 2035 projected request.

Oil yields (t oil per ha per year): palm 2.9-3.3, rapeseed and sunflower 0.7-0.8, groundnut 0.14-0.2, coconut 0.27-0.5,
soybean 0.45-0.55; cottonseed, maize, rice-bran and palm-kernel oils are by-products and carry no additional land.
Fixed supply: the change in embodied land is an accounting identity (sum over oils of change in use / yield).
Output: results/land_accounting.csv
"""
import importlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
X = importlib.import_module('20_relocation_options')
YLD = {'palm': (2.9, 3.3), 'rapeseed': (0.7, 0.8), 'sunflower': (0.7, 0.8), 'groundnut': (0.14, 0.2),
       'coconut': (0.27, 0.5), 'soy': (0.45, 0.55)}
dt = pd.read_csv('../results/options_detail_m1.csv.gz').fillna(0)
S = X.future()
rows = []


def land(series, bound):
    """Mha; bound 0 -> low land (high yield), 1 -> high land (low yield)"""
    return sum(series.get(k, 0.0) / YLD[k][1 - bound] for k in YLD) / 1.0   # Mt / (t/ha) = Mha


for name in ['strict_d0', 'strict_d0.25', 'strict_d0.5', 'full_d0']:
    b = dt[dt.setting == name]
    fuel = {k: b['fuel_' + k].sum() for k in YLD if 'fuel_' + k in b}
    food = {k: b['food_' + k].sum() for k in YLD if 'food_' + k in b}
    rows.append(dict(setting=name, fuel_land_low=land(fuel, 0), fuel_land_high=land(fuel, 1),
                     food_land_low=land(food, 0), food_land_high=land(food, 1)))
d = pd.DataFrame(rows).set_index('setting')
for c in ['fuel_land_low', 'fuel_land_high', 'food_land_low', 'food_land_high']:
    d[c.replace('land', 'dland')] = d[c] - d.loc['strict_d0', c]
d['total_dland_low'] = d.fuel_dland_low + d.food_dland_low
d['total_dland_high'] = d.fuel_dland_high + d.food_dland_high
supply = {k: S.s[:, S.ki[k]].sum() / 1000 for k in YLD}
out = {'supply_Mt_' + k: v for k, v in supply.items()}
sup10 = {k: 0.1 * v for k, v in supply.items()}
extra = dict(output10_land_low=land(sup10, 0), output10_land_high=land(sup10, 1), palm_area_low=supply['palm'] / YLD['palm'][1],
             palm_area_high=supply['palm'] / YLD['palm'][0],
             swap_fuel_palm_Mt=(dt[dt.setting == 'strict_d0.5'].fuel_palm.sum() - dt[dt.setting == 'strict_d0'].fuel_palm.sum()))
extra['upper_bound_new_palm_land_low'] = extra['swap_fuel_palm_Mt'] / YLD['palm'][1]
extra['upper_bound_new_palm_land_high'] = extra['swap_fuel_palm_Mt'] / YLD['palm'][0]
d.round(3).to_csv('../results/land_accounting.csv')
pd.Series({**out, **extra}).round(3).to_csv('../results/land_accounting_summary.csv', header=['value'])
print(d.round(2).T.to_string()); print(pd.Series({**out, **extra}).round(2).to_string())
