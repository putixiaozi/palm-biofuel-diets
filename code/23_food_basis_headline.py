"""Headline results of Figs 2-3 under three food-use bases: central (reconstructed), FAOSTAT food supply with the
difference held as other use, and FAOSTAT food supply with the difference released to allocation.
Output: results/food_basis_headline.csv"""
import importlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
X = importlib.import_module('20_relocation_options')
PT = {'palm', 'palmkernel', 'coconut'}
rows = []
for basis, pre in [('central', ''), ('faostat_other', 'faostat_other_'), ('faostat_released', 'faostat_released_')]:
    for lab, key in [('fixed_strict', 'strict_d0'), ('reloc25_strict', 'strict_d0.25'), ('reloc50_strict', 'strict_d0.5'),
                     ('fixed_full', 'full_d0')]:
        name = pre + key
        r0 = X.evaluate(name, 0.0)
        r1, z, S = X.evaluate(name, 1.0, detail=True)
        fuel_pt = sum(z['B'][:, S.ki[k]].sum() for k in PT if k in S.ki) / 1e6
        food_pt = sum(z['F'][:, S.ki[k]].sum() for k in PT if k in S.ki) / 1e6
        rows.append(dict(basis=basis, setting=lab, gap0=r0['gap_Mt'], gap1=r1['gap_Mt'], additional=r1['gap_Mt'] - r0['gap_Mt'],
                         cut1=r1['cut_Mt'], unmet1=r1['unmet_Mt'], fuel_palmtype=fuel_pt, food_palmtype=food_pt))
        print(rows[-1], flush=True)
d = pd.DataFrame(rows)
ref = d[d.setting == 'fixed_strict'].set_index('basis')
d['fuel_palm_change'] = d.fuel_palmtype - d.basis.map(ref.fuel_palmtype)
d['food_palm_change'] = d.food_palmtype - d.basis.map(ref.food_palmtype)
d.to_csv('../results/food_basis_headline.csv', index=False)
print(d.round(2).to_string())
