"""Zero-biofuel shortfall for the layer subsets that the structural block does not cover, so that layer Shapley values
can be computed for the biofuel-attributable shortfall S(E1) - S(0). At zero biofuel the conversion-location (L3) and
eligibility (L4) layers have no effect, so the eight subsets of {L1, L2, L5} span the zero-biofuel game.
Output: results/layers_m0_runs.csv
"""
import importlib
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from multiprocessing import Pool

X = importlib.import_module('05_experiments')


def main():
    sp = [dict(block='layers_m0', year=y, target=t, m=0.0, relax=sub)
          for y, t in itertools.product([2023, 2035], X.TARGETS)
          for sub in [('L1', 'L2'), ('L1', 'L5'), ('L2', 'L5')]]
    for n, s in enumerate(sp):
        s['run_id'] = n
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as pool:
        rows = [r for r, _ in pool.imap_unordered(X.run, sp, chunksize=1)]
    df = pd.DataFrame(rows).sort_values('run_id')
    df.to_csv(X.OUT + 'layers_m0_runs.csv', index=False)
    print(df[['year', 'target', 'relax', 'gap', 'secs']].assign(gap=df.gap / 1e6).round(2).to_string())


if __name__ == '__main__':
    main()
