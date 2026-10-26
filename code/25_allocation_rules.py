"""Regional additional gaps under many efficient allocations: random country weights break the tie among allocations
that minimise the total gap (2035, projected request, central specification). Output: results/allocation_rules.csv"""
import importlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from multiprocessing import Pool
import model as M
X = importlib.import_module('20_relocation_options')
RG = {"South Asia": "South Asia", "Western Africa": "Sub-Saharan Africa", "Eastern Africa": "Sub-Saharan Africa",
      "Southern Africa": "Sub-Saharan Africa", "Northern Africa": "N. Africa & Middle East",
      "Middle East": "N. Africa & Middle East", "South East Asia": "East & SE Asia", "East Asia": "East & SE Asia",
      "Central America": "Latin America", "South America": "Latin America"}
S0 = X.future()
reg = M.REG.set_index('iso3c').region
group = np.array([RG.get(reg.get(c), 'Other') for c in S0.C])


def run(k):
    rng = np.random.default_rng(k)
    S = S0.copy(); tg = S.targets('eat_quality'); L, cap = M.build_links(S)
    w = np.exp(rng.normal(0, 1.5, len(S.C))) if k >= 0 else np.ones(len(S.C))
    u = {}
    for m in (0.0, 1.0):
        z = M.solve(S, tg, L, cap, S.fuel_base * m, S.eligibility(), fuel_min=m * S.fuel_base.sum() * (1 - 1e-7),
                    mode='weighted', gap_tol=1e-4, tie_weights=w)
        u[m] = z['u'] / 1e6
    out = []
    for g in np.unique(group):
        sel = group == g
        out.append(dict(draw=k, region=g, gap=u[1.0][sel].sum(), additional=(u[1.0] - u[0.0])[sel].sum(),
                        total_gap=u[1.0].sum()))
    return out


if __name__ == '__main__':
    with Pool(6) as p:
        rows = [r for x in p.map(run, range(40)) for r in x]
    d = pd.DataFrame(rows); d.to_csv('../results/allocation_rules.csv', index=False)
    t = d.pivot(index='draw', columns='region', values='additional')
    t['SA+SSA share'] = (t['South Asia'] + t['Sub-Saharan Africa']) / t[[c for c in t.columns if c not in ('SA+SSA share',)]].sum(axis=1)
    print(t.describe().round(2).T.to_string())
