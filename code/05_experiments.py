"""All LP experiments behind Figs 2-6 and the Supplementary figures.

Blocks
  frontier   shortfall vs biofuel volume (national volumes scaled by m), constrained vs aggregate pool
  layers     all 32 relaxation subsets of the five constraint layers at the scenario biofuel volume (m = 1)
  structural one-at-a-time relaxations at zero biofuel (m = 0)
  strategy   16 subsets of four levers (S1 used cooking oil, S2 output, S3 import access, S4 diet convergence)
  sens       parameter sensitivity at 2035, m = 1
  history    2010-2023, m = 0 and m = 1
Outputs: results/runs_summary.csv, results/runs_country.csv.gz, results/targets_by_country.csv
"""
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from multiprocessing import Pool

import model as M

OUT = M.ROOT + 'results/'
TARGETS = ['adequacy', 'eat_qty', 'eat_quality']
LAYERS = ['L1', 'L2', 'L3', 'L4', 'L5']      # production, access, conversion, eligibility, demand
UCO_SHARE = 0.05                            # additional used cooking oil collected, share of food oil use
_cache = {}


def system(year, floor='percap'):
    key = (year, floor)
    if key not in _cache:
        if year <= 2023:
            _cache[key] = M.System(M.load_year(year))
        else:
            _cache[key] = M.project(system(2023), year, floor=floor)
    return _cache[key]


def run(spec):
    """spec: dict(block, year, target, m, relax=set(), delta, uco, waste, fat_share, floor, mode, access)."""
    t0 = time.time()
    relax = set(spec.get('relax', ()))
    S = system(spec['year'], spec.get('floor', 'percap'))
    if spec.get('uco', 0) > 0:
        S = S.copy()
        S.s[:, S.uco] = spec['uco'] * S.food0[:, S.virgin].sum(1)
    tg = S.targets(spec['target'], waste=spec.get('waste', 0.0), fat_share=spec.get('fat_share', M.FAT_SHARE),
                   flexible='L5' in relax, palm_cap=spec.get('palm_cap', False))
    access = 'pooled' if 'L2' in relax else spec.get('access', 'historical')
    L, ic = M.build_links(S, access, spec.get('delta', 0.0))
    el = S.eligibility(policy='L4' not in relax)
    m = spec['m']
    E = m * S.fuel_base.sum()
    cap = None if 'L3' in relax else S.fuel_base * m
    res = M.solve(S, tg, L, ic, cap, el, fuel_min=E * (1 - 1e-7), mode=spec.get('mode', 'min_gap'),
                  supply_scale=1.10 if 'L1' in relax else 1.0)
    row = {k: v for k, v in spec.items() if k != 'relax'}
    row.update(relax='+'.join(sorted(relax)) or 'none', E=E * M.UNIT, feasible=res['feasible'],
               gap=res.get('gap', np.nan), cut=res.get('cut', np.nan), fuel_unmet=res.get('fuel_unmet', np.nan),
               fuel=res.get('fuel', np.nan), t_eq=res.get('t_eq', np.nan), gap_obs=tg['gap_obs'],
               secs=round(time.time() - t0, 1))
    cty = None
    if res['feasible']:
        nz = (res['u'] > 1) | (res['v'] > 1)
        cty = pd.DataFrame({'iso3c': np.array(S.C)[nz], 'u': res['u'][nz], 'v': res['v'][nz]})
    return row, cty


def specs():
    out = []
    grid = np.round(np.arange(0, 2.01, 0.1), 2)
    for y, tg in itertools.product([2023, 2035], TARGETS):
        for m in grid:
            out.append(dict(block='frontier', year=y, target=tg, m=m, relax=()))
            out.append(dict(block='frontier_pool', year=y, target=tg, m=m, relax=('L2', 'L3', 'L4')))
        for r in range(6):
            for sub in itertools.combinations(LAYERS, r):
                out.append(dict(block='layers', year=y, target=tg, m=1.0, relax=sub))
        for sub in [()] + [(l,) for l in LAYERS] + [tuple(LAYERS)]:
            out.append(dict(block='structural', year=y, target=tg, m=0.0, relax=sub))
    for tg, m in itertools.product(TARGETS, [1.0, 1.5, 2.0]):
        for r in range(5):
            for sub in itertools.combinations(['S1', 'S2', 'S3', 'S4'], r):
                relax = (['L1'] if 'S2' in sub else []) + (['L5'] if 'S4' in sub else [])
                out.append(dict(block='strategy', year=2035, target=tg, m=m, relax=tuple(relax), strategies='+'.join(sub) or 'none',
                                uco=UCO_SHARE if 'S1' in sub else 0.0, delta=0.5 if 'S3' in sub else 0.0))
    for tg in TARGETS:
        for d in [0.25, 0.5, 1.0]:
            out.append(dict(block='sens', year=2035, target=tg, m=1.0, relax=(), delta=d, param=f'delta={d}'))
        out.append(dict(block='sens', year=2035, target=tg, m=1.0, relax=(), waste=0.1, param='waste=0.1'))
        out.append(dict(block='sens', year=2035, target=tg, m=1.0, relax=(), floor='oecd', param='floor=oecd'))
        out.append(dict(block='sens', year=2035, target=tg, m=1.0, relax=(), access='complete', param='links=complete'))
        for u in [0.025, 0.075]:
            out.append(dict(block='sens', year=2035, target=tg, m=1.0, relax=(), uco=u, param=f'uco={u}'))
        for y in range(2010, 2024):
            for m in [0.0, 1.0]:
                out.append(dict(block='history', year=y, target=tg, m=m, relax=()))
    for fs in [0.15, 0.25]:
        out.append(dict(block='sens', year=2035, target='adequacy', m=1.0, relax=(), fat_share=fs, param=f'fat={fs}'))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    sp = specs()
    # cheap runs first so partial output is useful; pooled runs are slowest
    sp.sort(key=lambda s: ('L2' in s.get('relax', ()), s['block'] != 'frontier'))
    for n, s in enumerate(sp):
        s['run_id'] = n
    print(len(sp), 'runs', flush=True)
    rows, ctys = [], []
    with Pool(6) as pool:
        for k, (row, cty) in enumerate(pool.imap_unordered(run, sp, chunksize=1)):
            rows.append(row)
            if cty is not None:
                ctys.append(cty.assign(run_id=row['run_id']))
            if k % 25 == 0 or k == len(sp) - 1:
                pd.DataFrame(rows).sort_values('run_id').to_csv(OUT + 'runs_summary.csv', index=False)
                print(k, row['block'], row['year'], row['target'], row['m'], row['relax'],
                      'gap %.2f cut %.2f' % (row['gap'] / 1e6, row['cut'] / 1e6), row['secs'], flush=True)
    pd.DataFrame(rows).sort_values('run_id').to_csv(OUT + 'runs_summary.csv', index=False)
    pd.concat(ctys).to_csv(OUT + 'runs_country.csv.gz', index=False)
    # country targets for reporting
    T = []
    for y in [2023, 2035]:
        S = system(y)
        for tg in TARGETS:
            t = S.targets(tg)
            T.append(pd.DataFrame({'year': y, 'target': tg, 'iso3c': S.C, 'pop': S.pop, 'H': t['H'] * M.UNIT,
                                   'D': t['D'] * M.UNIT, 'T': t['T'] * M.UNIT, 'obs': t['obs'] * M.UNIT,
                                   'fuel_base': S.fuel_base * M.UNIT}))
    pd.concat(T).to_csv(OUT + 'targets_by_country.csv', index=False)
    print('done', flush=True)


if __name__ == '__main__':
    main()
