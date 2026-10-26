"""Robustness runs (Supplementary figures).

Blocks
  sens0      the parameter sensitivities of the main experiments at zero biofuel (m = 0), so the biofuel-attributable
             shortfall can be reported under each alternative parameter
  pcap_*     the quality target with the EAT-Lancet palm cap in food instead of total-use protection:
             frontier (constrained and pooled), the 32 layer subsets, the four-lever strategies and 2010-2023 history
Output: results/robustness_runs.csv
"""
import importlib
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from multiprocessing import Pool

X = importlib.import_module('05_experiments')
OUT = X.OUT


def specs():
    out = [dict(s, m=0.0, block='sens0') for s in X.specs() if s['block'] == 'sens']
    q = dict(target='eat_quality', palm_cap=True)
    grid = np.round(np.arange(0, 2.01, 0.1), 2)
    for y in [2023, 2035]:
        for m in grid:
            out.append(dict(q, block='pcap_frontier', year=y, m=m, relax=()))
            out.append(dict(q, block='pcap_frontier_pool', year=y, m=m, relax=('L2', 'L3', 'L4')))
        for r in range(6):
            for sub in itertools.combinations(X.LAYERS, r):
                out.append(dict(q, block='pcap_layers', year=y, m=1.0, relax=sub))
    for r in range(5):
        for sub in itertools.combinations(['S1', 'S2', 'S3', 'S4'], r):
            relax = (['L1'] if 'S2' in sub else []) + (['L5'] if 'S4' in sub else [])
            out.append(dict(q, block='pcap_strategy', year=2035, m=1.0, relax=tuple(relax), strategies='+'.join(sub) or 'none',
                            uco=X.UCO_SHARE if 'S1' in sub else 0.0, delta=0.5 if 'S3' in sub else 0.0))
    for y in range(2010, 2024):
        for m in [0.0, 1.0]:
            out.append(dict(q, block='pcap_history', year=y, m=m, relax=()))
    return out


def main():
    sp = specs()
    sp.sort(key=lambda s: 'L2' in s.get('relax', ()))
    for n, s in enumerate(sp):
        s['run_id'] = n
    print(len(sp), 'runs', flush=True)
    rows = []
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as pool:
        for k, (row, _) in enumerate(pool.imap_unordered(X.run, sp, chunksize=1)):
            rows.append(row)
            if k % 25 == 0 or k == len(sp) - 1:
                pd.DataFrame(rows).sort_values('run_id').to_csv(OUT + 'robustness_runs.csv', index=False)
                print(k, row['block'], row['year'], row['target'], row['m'], row['relax'],
                      'gap %.2f cut %.2f' % (row['gap'] / 1e6, row['cut'] / 1e6), flush=True)
    pd.DataFrame(rows).sort_values('run_id').to_csv(OUT + 'robustness_runs.csv', index=False)
    print('done', flush=True)


if __name__ == '__main__':
    main()
