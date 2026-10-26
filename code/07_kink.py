"""Compatible biofuel volume E_c: the largest uniform scaling m of national biofuel volumes at which the
biofuel-attributable shortfall S(E) - S(0) stays within a tolerance (0.1, 0.5 or 1 Mt; bisection on m), and the
average marginal shortfall per tonne of biofuel over the next 0.1 E1. Output: results/compatible_volume.csv
The central quality target protects each country's total oil food use, so a volume is compatible only if neither the
shortfall nor the food-use cut rises by more than the tolerance. With argument pcap the quality target instead caps palm
in food (EAT-Lancet 6.8 g) without the total-use floor. Output: results/compatible_volume_pcap.csv
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
TOLS = [1e5, 5e5, 1e6]      # tonnes
_cache = {}


def system(year):
    if year not in _cache:
        _cache[year] = M.System(M.load_year(year)) if year <= 2023 else M.project(system(2023), year)
    return _cache[year]


def kink(spec):
    year, target, pool, palm_cap = spec
    S = system(year)
    tg = S.targets(target, palm_cap=palm_cap)
    L, ic = M.build_links(S, 'pooled' if pool else 'historical')
    el = S.eligibility(policy=not pool)

    def harm(m):
        cap = None if pool else S.fuel_base * m
        E = m * S.fuel_base.sum()
        r = M.solve(S, tg, L, ic, cap, el, fuel_min=E * (1 - 1e-7))
        return r['gap'], r['cut']

    g0, c0 = harm(0.0)

    def excess(m):
        g, c = harm(m)
        return g - g0 if palm_cap else max(g - g0, c - c0)

    E1 = S.fuel_base.sum() * M.UNIT
    rows = []
    for tol in TOLS:
        lo, hi = 0.0, 3.0
        if excess(hi) <= tol:
            lo = hi
        else:
            for _ in range(14):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if excess(mid) <= tol else (lo, mid)
        slope = (harm(lo + 0.1)[0] - harm(lo)[0]) / (0.1 * E1) if lo < 3.0 else np.nan
        rows.append(dict(year=year, target=target, access='pool' if pool else 'constrained', tol=tol, S0=g0, cut0=c0,
                         m_c=lo, E_c=lo * E1, E1=E1, slope_above=slope))
    return rows


def main():
    alt = len(sys.argv) > 2 and sys.argv[2] == 'pcap'   # alternative: EAT-Lancet palm cap without total-use floor
    targets = ['eat_quality'] if alt else ['adequacy', 'eat_qty', 'eat_quality']
    sp = [(y, t, p, alt) for y, t, p in itertools.product([2023, 2035], targets, [False, True])]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as pool:
        rows = pool.map(kink, sp, chunksize=1)
    df = pd.DataFrame([x for r in rows for x in r])
    df.to_csv(OUT + ('compatible_volume_pcap.csv' if alt else 'compatible_volume.csv'), index=False)
    print(df.assign(S0=df.S0 / 1e6, E_c=df.E_c / 1e6, E1=df.E1 / 1e6).round(3).to_string())


if __name__ == '__main__':
    main()
