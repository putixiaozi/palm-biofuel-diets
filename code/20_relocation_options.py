"""Feedstock eligibility x bounded relocation, unrestricted allocation and single options, 2035, unsaturated target.

Compatible volume: the largest request m * sum(b_j) at which the total shortfall and the food-use cut stay within a
tolerance of the central no-fuel values (fixed national shares, palm exclusion; the same for every setting that keeps
historical trade access, because location and eligibility do not act at zero fuel), with unmet fuel below 0.001 Mt.
The FAOSTAT food-basis sensitivities use their own no-fuel values because they change the food floors.

Bounded relocation: incumbent producers deliver between (1 - delta) m b_j and (1 + delta) m b_j at the same global
requirement; countries without baseline fuel stay non-producers.
Outputs: results/options_frontier.csv, options_compatible.csv, options_detail_m1.csv, options_diagnostic.csv
"""
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import model as M

OUT = M.ROOT + 'results/'
TOLS = [0.1, 1.0]
HI, STEPS = 2.5, 12
_S = {}


def future(basis='model'):
    if basis not in _S:
        S = M.project(M.System(M.load_year(2023)))
        if basis.startswith('faostat'):
            # food floors at FAOSTAT vegetable-oil food supply (countries with population data); the difference is
            # either held as non-food use ('faostat_other') or released to allocation ('faostat_released')
            food = S.food0[:, S.virgin].sum(1)
            r = np.where((food > 0) & (S.pop > 0), np.minimum(1.0, S.fbs_food / np.maximum(food, 1e-12)), 1.0)
            moved = S.food0 * (1 - r[:, None])
            S.food0 = S.food0 - moved
            if basis == 'faostat_other':
                S.nonfuel0 = S.nonfuel0 + moved
        _S[basis] = S
    return _S[basis]


SETTINGS = {}
for palm in ['strict', 'full']:
    for d in [0.0, 0.1, 0.25, 0.5]:
        SETTINGS[f'{palm}_d{d:g}'] = dict(palm=palm, delta=d)
SETTINGS['unrestricted'] = dict(palm='full', access='pooled', free_location=True)
SETTINGS['opt_uco5_traded'] = dict(palm='strict', uco=0.05)
SETTINGS['opt_uco5_domestic'] = dict(palm='strict', uco=0.05, domestic_uco=True)
SETTINGS['opt_output10'] = dict(palm='strict', supply=1.1)
SETTINGS['opt_imports50'] = dict(palm='strict', imp=0.5)
SETTINGS['opt_diet'] = dict(palm='strict', flexible=True)
for basis in ['faostat_other', 'faostat_released']:
    for d in [0.0, 0.25, 0.5]:
        SETTINGS[f'{basis}_strict_d{d:g}'] = dict(palm='strict', delta=d, basis=basis)
    SETTINGS[f'{basis}_full_d0'] = dict(palm='full', delta=0.0, basis=basis)


def evaluate(name, m, detail=False, extra=None):
    c = dict(SETTINGS[name], **(extra or {}))
    S = future(c.get('basis', 'model')).copy()
    if c.get('uco'):
        S.s[:, S.uco] = c['uco'] * S.food0[:, S.virgin].sum(1)
    tg = S.targets('eat_quality', flexible=c.get('flexible', False))
    L, cap = M.build_links(S, c.get('access', 'historical'), c.get('imp', 0.0))
    if c.get('domestic_uco'):
        L = L[(L[:, 2] != S.uco) | (L[:, 0] == L[:, 1])]
    d = c.get('delta', 0.0)
    fuel_cap = None if c.get('free_location') else S.fuel_base * m * (1 + d)
    fuel_lower = S.fuel_base * m * (1 - d) if d and not c.get('free_location') else None
    z = M.solve(S, tg, L, cap, fuel_cap, S.eligibility(policy=c['palm'] == 'strict'),
                fuel_min=m * S.fuel_base.sum() * (1 - 1e-7), fuel_lower=fuel_lower,
                supply_scale=c.get('supply', 1.0), detail=detail)
    if not z['feasible']:                    # national lower bounds cannot all be met when supply runs short
        assert not detail, (name, m)
        return dict(setting=name, m=m, requested_Mt=m * S.fuel_base.sum() / 1000, feasible=False)
    row = dict(setting=name, m=m, feasible=True, requested_Mt=m * S.fuel_base.sum() / 1000, delivered_Mt=z['fuel'] / 1e6,
               unmet_Mt=z['fuel_unmet'] / 1e6, cut_Mt=z['cut'] / 1e6, gap_Mt=z['gap'] / 1e6)
    return (row, z, S) if detail else row


def frontier(name):
    return [evaluate(name, m) for m in np.round(np.arange(0, 2.01, 0.1), 2)]


def boundary(args):
    name, ref = args
    rows, out = [], []
    for tol in TOLS:
        lo, hi = 0.0, HI
        for _ in range(STEPS):
            mid = (lo + hi) / 2
            r = evaluate(name, mid); rows.append(r)
            ok = r['feasible'] and r['gap_Mt'] - ref['gap_Mt'] <= tol and r['cut_Mt'] - ref['cut_Mt'] <= tol \
                and r['unmet_Mt'] < 1e-3
            lo, hi = (mid, hi) if ok else (lo, mid)
        E1 = future(SETTINGS[name].get('basis', 'model')).fuel_base.sum() / 1000
        out.append(dict(setting=name, tolerance_Mt=tol, compatible_Mt=lo * E1, interval_width_Mt=(hi - lo) * E1,
                        ref_gap_Mt=ref['gap_Mt'], ref_cut_Mt=ref['cut_Mt'], at_search_limit=hi == HI))
    return out, rows


def detail(name):
    row, z, S = evaluate(name, 1.0, detail=True)
    B = pd.DataFrame(z['B'] / 1e6, index=S.C, columns=S.K).add_prefix('fuel_')
    F = pd.DataFrame(z['F'] / 1e6, index=S.C, columns=S.K).add_prefix('food_')
    F0 = pd.DataFrame(S.food0 / 1e3, index=S.C, columns=S.K).add_prefix('food0_')
    out = pd.concat([B, F, F0], axis=1)
    out = out.loc[:, out.abs().sum() > 1e-9]
    return out.assign(setting=name, fuel_base_Mt=S.fuel_base / 1000, pop=S.pop).rename_axis('iso3c').reset_index()


def diagnostic(extra_name):
    """Which restriction binds when full eligibility and +-50% relocation stop adding volume (request 80 Mt)."""
    m = 80.0 / (future().fuel_base.sum() / 1000)
    extra = {'none': {}, 'imports_uncapped': dict(access='complete', imp=1e6), 'supply+10%': dict(supply=1.1),
             'delta=1.0': dict(delta=1.0), 'location_free': dict(free_location=True),
             'links_and_location_free': dict(access='pooled', free_location=True)}[extra_name]
    r = evaluate('full_d0.5', m, extra=extra)
    return dict(r, relaxation=extra_name)


def main():
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    ref = evaluate('strict_d0', 0.0)
    refs = {b: evaluate(f'{b}_strict_d0', 0.0) for b in ['faostat_other', 'faostat_released']}
    print('reference', ref, refs, flush=True)
    with Pool(nproc) as pool:
        names = [n for n in SETTINGS if n.startswith(('strict', 'full', 'unrestricted'))]
        fr = pool.map_async(frontier, names, chunksize=1)
        bd = pool.map_async(boundary, [(n, refs[SETTINGS[n]['basis']] if 'basis' in SETTINGS[n] else ref) for n in SETTINGS], chunksize=1)
        dt = pool.map_async(detail, ['strict_d0', 'strict_d0.25', 'strict_d0.5', 'full_d0', 'full_d0.25', 'full_d0.5'])
        dg = pool.map_async(diagnostic, ['none', 'imports_uncapped', 'supply+10%', 'delta=1.0', 'location_free',
                                         'links_and_location_free'])
        pd.DataFrame([r for x in fr.get() for r in x]).to_csv(OUT + 'options_frontier.csv', index=False)
        print('frontier done', flush=True)
        b = bd.get()
        pd.DataFrame([r for x, _ in b for r in x]).to_csv(OUT + 'options_compatible.csv', index=False)
        pd.DataFrame([r for _, y in b for r in y]).to_csv(OUT + 'options_bisection_runs.csv', index=False)
        print('boundaries done', flush=True)
        pd.concat(dt.get()).to_csv(OUT + 'options_detail_m1.csv.gz', index=False)
        pd.DataFrame(dg.get()).to_csv(OUT + 'options_diagnostic.csv', index=False)
    print(pd.read_csv(OUT + 'options_compatible.csv').round(2).to_string())
    print(pd.read_csv(OUT + 'options_diagnostic.csv').round(3).to_string())
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
