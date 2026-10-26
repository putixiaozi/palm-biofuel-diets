"""Every number quoted in the manuscript, from the result tables. Output: results/key_numbers_text.csv"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import model as M

R = M.ROOT + 'results/'
K = []


def add(name, value, source):
    K.append(dict(name=name, value=value, source=source))


# history and supply
sm = pd.read_csv(R + 'fig1_summary_by_year.csv').set_index('year')
for y in [2010, 2023]:
    for c in sm.columns:
        add(f'{c}_{y}', sm.loc[y, c], 'fig1_summary_by_year.csv')
h = pd.read_csv(R + 'history_fuel_cost.csv')
for t in ['eat_qty', 'eat_quality', 'adequacy']:
    for y in [2010, 2016, 2017, 2023]:
        add(f'hist_additional_{t}_{y}', h[(h.target == t) & (h.year == y)].biofuel_cost.iloc[0], 'history_fuel_cost.csv')

# 2035 frontier
f = pd.read_csv(R + 'fig4_frontier.csv').query('year == 2035')
for t in ['eat_qty', 'eat_quality', 'adequacy']:
    for acc in ['constrained', 'pool']:
        q = f[(f.target == t) & (f.access == acc)].set_index('m')
        for m in [0.0, 1.0, 1.5, 2.0]:
            add(f'gap_{t}_{acc}_m{m:g}', q.loc[m, 'gap'], 'fig4_frontier.csv')
            add(f'unmet_{t}_{acc}_m{m:g}', q.loc[m, 'fuel_unmet'], 'fig4_frontier.csv')
            add(f'cut_{t}_{acc}_m{m:g}', q.loc[m, 'cut'], 'fig4_frontier.csv')
        add(f'additional_{t}_{acc}_m1', q.loc[1.0, 'gap'] - q.loc[0.0, 'gap'], 'fig4_frontier.csv')
        zero = q[q.gap < 0.01]
        add(f'zero_gap_max_E_{t}_{acc}', zero.E.max(), 'fig4_frontier.csv (largest grid request with gap < 0.01 Mt)')
add('E1_2035', f[(f.target == 'eat_quality') & (f.access == 'constrained') & (f.m == 1.0)].E.iloc[0], 'fig4_frontier.csv')

# layers
at = pd.read_csv(R + 'fig3_layer_shapley_attrib.csv').query("target == 'eat_quality'").set_index('year')
for y in [2023, 2035]:
    for l in ['L1', 'L2', 'L3', 'L4', 'L5', 'total']:
        add(f'shapley_attrib_{l}_{y}', at.loc[y, l], 'fig3_layer_shapley_attrib.csv')
    for l in ['L1', 'L2', 'L3', 'L4', 'L5']:
        add(f'single_attrib_{l}_{y}', at.loc[y, 'single_' + l], 'fig3_layer_shapley_attrib.csv')

# reference level
rf = pd.read_csv(R + 'reference_target_diagnostic.csv')
for _, r in rf.iterrows():
    add(f'ref_additional_{int(r.year)}_{int(r.reference_g_per_day)}g', r.attributable_Mt, 'reference_target_diagnostic.csv')

# eligibility x relocation, options, sensitivities
cv = pd.read_csv(R + 'options_compatible.csv')
for _, r in cv.iterrows():
    add(f'compatible_{r.setting}_tol{r.tolerance_Mt:g}', r.compatible_Mt, 'options_compatible.csv')
fr = pd.read_csv(R + 'options_frontier.csv')
ref0 = fr[(fr.setting == 'strict_d0') & (fr.m == 0)].gap_Mt.iloc[0]
for s, g in fr.groupby('setting'):
    q = g.set_index('m')
    add(f'gap_m1_{s}', q.loc[1.0, 'gap_Mt'], 'options_frontier.csv')
    add(f'additional_m1_{s}', q.loc[1.0, 'gap_Mt'] - ref0, 'options_frontier.csv (minus current no-fuel gap)')
dt = pd.read_csv(R + 'options_detail_m1.csv.gz').fillna(0)
PT = {'palm', 'palmkernel', 'coconut'}
for s, b in dt.groupby('setting'):
    for use in ['fuel', 'food']:
        cc = [c for c in b.columns if c.startswith(use + '_') and c != 'fuel_base_Mt']
        add(f'{use}_palmtype_{s}', sum(b[c].sum() for c in cc if c.split('_', 1)[1] in PT), 'options_detail_m1.csv.gz')
        add(f'{use}_unsat_{s}', sum(b[c].sum() for c in cc if c.split('_', 1)[1] not in PT), 'options_detail_m1.csv.gz')
    fc = [c for c in b.columns if c.startswith('fuel_') and c != 'fuel_base_Mt']
    t = b.set_index('iso3c')[fc].sum(axis=1) - b.set_index('iso3c').fuel_base_Mt
    for iso in ['IDN', 'USA', 'BRA', 'MYS', 'THA', 'DEU', 'ARG']:
        add(f'fuel_change_{iso}_{s}', t.get(iso, 0.0), 'options_detail_m1.csv.gz (minus national request)')
dg = pd.read_csv(R + 'options_diagnostic.csv')
for _, r in dg.iterrows():
    add(f'diag80_{r.relaxation}', r.gap_Mt, 'options_diagnostic.csv (full_d0.5 at 80 Mt)')

# levers at the projected request
rs = pd.read_csv(R + 'runs_summary.csv')
st = rs[(rs.block == 'strategy') & (rs.target == 'eat_quality') & (rs.m == 1.0)].copy()
st['strategies'] = st.strategies.fillna('none')
for k, g in zip(st.strategies, st.gap / 1e6):
    add(f'lever_gap_{k}', g, 'runs_summary.csv strategy block')
sh = pd.read_csv(R + 'fig5_strategy_shapley.csv').query("target == 'eat_quality' and m == 1.0")
for s in ['S1', 'S2', 'S3', 'S4']:
    add(f'lever_shapley_{s}', sh[s].iloc[0], 'fig5_strategy_shapley.csv')

# countries
c = pd.read_csv(R + 'fig6_country.csv').query("target == 'eat_quality'")
has = c[c.u1 > 1e4]
add('countries_with_gap', len(has), 'fig6_country.csv (gap > 10 kt)')
add('population_with_gap_bn', has['pop'].sum() / 1e9, 'fig6_country.csv')
for d in ['L2', 'L34']:
    g = has[has.dominant == d]
    add(f'n_{d}', len(g), 'fig6_country.csv'); add(f'pop_{d}_bn', g['pop'].sum() / 1e9, 'fig6_country.csv')
    add(f'gap_{d}', g.u1.sum() / 1e6, 'fig6_country.csv'); add(f'additional_{d}', g.attrib.sum() / 1e6, 'fig6_country.csv')
reg = M.REG.set_index('iso3c').region
c['freg'] = c.iso3c.map(reg)
tot_add = c.attrib.sum()
for r_ in ['South Asia', 'Western Africa', 'Eastern Africa', 'Middle East', 'South East Asia', 'Northern Africa']:
    add(f'additional_share_{r_}', c[c.freg == r_].attrib.sum() / tot_add, 'fig6_country.csv')
for iso in ['IND', 'PAK', 'BGD', 'EGY', 'NGA']:
    add(f'additional_{iso}', c[c.iso3c == iso].attrib.sum() / 1e6, 'fig6_country.csv')
add('additional_total_country_sum', tot_add / 1e6, 'fig6_country.csv')

# pcap alternative and FAOSTAT basis
for fn in ['revision_scenarios.csv']:
    s = pd.read_csv(R + fn)
    for _, r in s[s.scenario.isin(['pcap_alt', 'grouped_saturated_cap', 'baseline']) & (s.year == 2035)].iterrows():
        add(f'{r.scenario}_m{r.m:g}_gap', r.gap_Mt, fn)
    for _, r in s[s.scenario.str.startswith(('palm_share_', 'uco_domestic_', 'output_')) & (s.m == 1.0)].iterrows():
        add(f'{r.scenario}_m1_gap', r.gap_Mt, fn)
cp = pd.read_csv(R + 'compatible_volume_pcap.csv').query('year == 2035')
for _, r in cp.iterrows():
    add(f'pcap_compatible_{r.access}_tol{r.tol:g}', r.E_c / 1e6, 'compatible_volume_pcap.csv')


# food-use bases, land accounting, allocation rules
fb = pd.read_csv(R + 'food_basis_headline.csv')
for _, r in fb.iterrows():
    add(f'foodbasis_additional_{r.basis}_{r.setting}', r.additional, 'food_basis_headline.csv')
    add(f'foodbasis_fuelpalm_change_{r.basis}_{r.setting}', r.fuel_palm_change, 'food_basis_headline.csv')
la = pd.read_csv(R + 'land_accounting_summary.csv').set_index('Unnamed: 0').value
for k, v in la.items():
    add('land_' + k, v, 'land_accounting_summary.csv')
ld = pd.read_csv(R + 'land_accounting.csv').set_index('setting')
for st in ld.index:
    for c in ['fuel_dland_low', 'fuel_dland_high', 'food_dland_low', 'food_dland_high']:
        add(f'land_{c}_{st}', ld.loc[st, c], 'land_accounting.csv')
ar = pd.read_csv(R + 'allocation_rules.csv').pivot(index='draw', columns='region', values='additional')
sh = (ar['South Asia'] + ar['Sub-Saharan Africa']) / ar.sum(axis=1)
for q in [0, .25, .5, .75, 1]:
    add(f'SA_SSA_share_q{q}', sh.quantile(q), 'allocation_rules.csv')
add('SA_SSA_share_gt50_draws', int((sh > .5).sum()), 'allocation_rules.csv')
add('SA_gt_SSA_draws', int((ar['South Asia'] > ar['Sub-Saharan Africa']).sum()), 'allocation_rules.csv')
for r_ in ar.columns:
    add(f'alloc_additional_min_{r_}', ar[r_].min(), 'allocation_rules.csv'); add(f'alloc_additional_max_{r_}', ar[r_].max(), 'allocation_rules.csv')

out = pd.DataFrame(K)
out.to_csv(R + 'key_numbers_text.csv', index=False)
print(out.assign(value=out.value.round(3)).to_string())
