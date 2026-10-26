"""Supplementary Data 1-3 from the result tables."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import paths
import pandas as pd
R = paths.REPO + 'results/'
O = paths.REPO + 'source_data/supplementary_data/'
x = pd.read_csv(R + 'external_feedstock_benchmarks.csv').assign(table='external_feedstock_benchmarks')
m = pd.read_csv(R + 'historical_EU_mix_sensitivity.csv').assign(table='EU_mix_reweighting')
pd.concat([x, m], ignore_index=True, sort=False).to_csv(O + 'Supplementary_Data_1_feedstock_checks.csv', index=False)
f = pd.read_csv(R + 'fig4_frontier.csv').query('year == 2035').assign(table='frontier', gap_Mt=lambda d: d.gap,
                                                                      cut_Mt=lambda d: d.cut, unmet_Mt=lambda d: d.fuel_unmet,
                                                                      requested_Mt=lambda d: d.E)
f = f[['table', 'target', 'access', 'm', 'requested_Mt', 'unmet_Mt', 'cut_Mt', 'gap_Mt']]
o = pd.read_csv(R + 'options_frontier.csv').assign(table='eligibility_relocation_options', target='eat_quality')
s = pd.read_csv(R + 'revision_scenarios.csv').assign(table='scenarios')
pd.concat([f, o, s], ignore_index=True, sort=False).to_csv(O + 'Supplementary_Data_2_delivery_and_cuts.csv', index=False)
c = pd.read_csv(R + 'revision_country_allocations.csv').rename(columns={'shortfall_Mt': 'gap_Mt'})
c.to_csv(O + 'Supplementary_Data_3_country_allocations.csv', index=False)
print(c.scenario.unique())

l = pd.read_csv(R + 'land_accounting.csv').assign(table='land_embodied_by_setting')
b = pd.read_csv(R + 'food_basis_headline.csv').assign(table='food_use_basis_headline')
a = pd.read_csv(R + 'allocation_rules.csv').assign(table='regional_additional_gap_by_allocation')
pd.concat([l.reset_index(drop=True), b, a], ignore_index=True, sort=False).to_csv(O + 'Supplementary_Data_4_land_food_basis_allocations.csv', index=False)
