"""Unsaturated-oil reference of 20, 30, 40 and 50 g per person per day (central specification), and the comparison
of reconstructed food use with FAOSTAT vegetable-oil fat supply. Changing the unsaturated-oil reference also changes total oil reference by the
same amount, retaining the source model's 6.8 g palm allowance and all other rules.
"""
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as M
OUT = Path(M.ROOT) / 'results'
DD = Path(M.DD)

base = M.System(M.load_year(2023))
future = M.project(base)
rows = []
for system in [base, future]:
    links, caps = M.build_links(system)
    for reference in [20., 30., 40., 50.]:
        M.EAT_UNSAT = reference
        M.EAT_TOTAL = reference + M.EAT_PALM
        target = system.targets('eat_quality')
        results = [M.frontier_point(system, target, links, caps, system.eligibility(), m)
                   for m in [0., 1.]]
        row = dict(year=system.year, reference_g_per_day=reference,
                   total_reference_g_per_day=M.EAT_TOTAL,
                   gap0_Mt=results[0]['gap']/1e6, gap1_Mt=results[1]['gap']/1e6,
                   attributable_Mt=(results[1]['gap']-results[0]['gap'])/1e6,
                   cut0_Mt=results[0]['cut']/1e6, cut1_Mt=results[1]['cut']/1e6)
        rows.append(row)
        print(row, flush=True)
        pd.DataFrame(rows).to_csv(OUT/'reference_target_diagnostic.csv', index=False)
M.EAT_UNSAT, M.EAT_TOTAL = 40., 46.8

use = pd.read_csv(DD/'base_use_by_country_oil_2010_2023.csv')
pop = pd.read_csv(DD/'base_population_fat_2010_2023.csv')
comparison = use.query('year == 2023').groupby('iso3c').food.sum().to_frame()
comparison = comparison.join(pop.query('year == 2023').set_index('iso3c'))
comparison['model_food_g_per_day'] = comparison.food / comparison['pop'] / 365 * 1e6
comparison['ratio_to_FBS_oil_fat'] = comparison.model_food_g_per_day / comparison.oilfat_fbs
comparison.to_csv(OUT/'food_accounting_comparison_2023.csv')
