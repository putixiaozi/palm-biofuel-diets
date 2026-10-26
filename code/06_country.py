"""Country-level allocations (Figs 3 and 6, Supplementary figures).

Country shortfalls are not unique at the minimum total shortfall, so every run here reports the efficient and fair
allocation: total shortfall held at its minimum (gap_tol 1e-4), then relative shortfalls u_j / D_j filled
lexicographically from the worst-off country down (mode 'waterfill').

Blocks
  detail   2023 and 2035, m = 0 and 1, constrained and pooled: country x oil food and biofuel, country shortfall
  layers   2035, m = 1, 32 relaxation subsets: country shortfall for country-level Shapley values
  fair     2035, m = 1: efficiency-equity curve (gap_tol from 0 to fairness first)
Outputs: results/country_detail.csv.gz, results/country_alloc.csv.gz, results/country_layers.csv.gz,
         results/fair_curve.csv
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from multiprocessing import Pool

import model as M

OUT = M.ROOT + 'results/'
TARGETS = ['adequacy', 'eat_qty', 'eat_quality']
LAYERS = ['L1', 'L2', 'L3', 'L4', 'L5']
EFF = 1e-4
_cache = {}


def system(year):
    if year not in _cache:
        _cache[year] = M.System(M.load_year(year)) if year <= 2023 else M.project(system(2023), year)
    return _cache[year]


def run(spec):
    S = system(spec['year'])
    relax = set(spec.get('relax', ()))
    tg = S.targets(spec['target'], flexible='L5' in relax)
    L, ic = M.build_links(S, 'pooled' if 'L2' in relax else 'historical')
    el = S.eligibility(policy='L4' not in relax)
    m = spec['m']
    E = m * S.fuel_base.sum()
    cap = None if 'L3' in relax else S.fuel_base * m
    res = M.solve(S, tg, L, ic, cap, el, fuel_min=E * (1 - 1e-7), mode='waterfill', gap_tol=spec.get('tol', EFF),
                  supply_scale=1.10 if 'L1' in relax else 1.0, detail=spec.get('detail', False))
    tag = {k: v for k, v in spec.items() if k not in ('relax', 'detail')}
    tag['relax'] = '+'.join(sorted(relax)) or 'none'
    if not res['feasible']:
        return tag, None, None
    D = tg['D'] * M.UNIT
    cty = pd.DataFrame({'iso3c': S.C, 'pop': S.pop, 'T': tg['T'] * M.UNIT, 'H': tg['H'] * M.UNIT, 'D': D,
                        'obs': tg['obs'] * M.UNIT, 'u': res['u'], 'v': res['v'], 'fuel_base': S.fuel_base * M.UNIT * m})
    alloc = None
    if spec.get('detail'):
        F = pd.DataFrame(res['F'], index=S.C, columns=S.K).stack().rename('food')
        B = pd.DataFrame(res['B'], index=S.C, columns=S.K).stack().rename('biofuel')
        alloc = pd.concat([F, B], axis=1).query('food > 1 or biofuel > 1').rename_axis(['iso3c', 'oil']).reset_index()
    tag.update(gap=res['gap'], cut=res['cut'], fuel=res['fuel'], fuel_unmet=res['fuel_unmet'])
    return tag, cty, alloc


def specs():
    out = []
    for y, tg, m, pool in itertools.product([2023, 2035], TARGETS, [0.0, 1.0], [False, True]):
        out.append(dict(block='detail', year=y, target=tg, m=m, relax=('L2', 'L3', 'L4') if pool else (),
                        detail=True))
    for tg in TARGETS:
        for r in range(6):
            for sub in itertools.combinations(LAYERS, r):
                if sub and sub != ('L2', 'L3', 'L4'):
                    out.append(dict(block='layers', year=2035, target=tg, m=1.0, relax=sub))
        for tol in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
            out.append(dict(block='fair', year=2035, target=tg, m=1.0, relax=(), tol=tol))
        out.append(dict(block='fair', year=2035, target=tg, m=1.0, relax=(), tol=None))
    return out


def main():
    sp = specs()
    sp.sort(key=lambda s: 'L2' in s.get('relax', ()))
    for n, s in enumerate(sp):
        s['run_id'] = n
    print(len(sp), 'runs', flush=True)
    rows, ctys, allocs = [], [], []
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as pool:
        for k, (tag, cty, alloc) in enumerate(pool.imap_unordered(run, sp, chunksize=1)):
            rows.append(tag)
            if cty is not None:
                ctys.append(cty.assign(run_id=tag['run_id']))
            if alloc is not None:
                allocs.append(alloc.assign(run_id=tag['run_id']))
            print(k, tag['block'], tag['year'], tag['target'], tag['m'], tag['relax'], tag.get('tol'),
                  'gap %.2f' % (tag.get('gap', np.nan) / 1e6), flush=True)
    runs = pd.DataFrame(rows).sort_values('run_id')
    runs.to_csv(OUT + 'country_runs.csv', index=False)
    pd.concat(ctys).to_csv(OUT + 'country_detail.csv.gz', index=False)
    pd.concat(allocs).to_csv(OUT + 'country_alloc.csv.gz', index=False)
    print('done', flush=True)


if __name__ == '__main__':
    main()
