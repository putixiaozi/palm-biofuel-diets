"""Country tables from the efficient-and-fair allocations (06_country.py).

Outputs (results/):
  fig3d_feedstock_region.csv  biofuel feedstock by producing region and oil group, E1, constrained vs pooled
  fig6_country.csv            2035 E1, per country: target, shortfall at m = 1 and m = 0, biofuel-attributable part,
                              Shapley values of the shortfall over the five layers and the dominant layer
  fair_check.csv              total and largest relative shortfall when fairness is imposed first or within a slack
                              on total shortfall
"""
import importlib
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

H = importlib.import_module('04_history')
A = importlib.import_module('08_analysis')

OUT = A.OUT
r = pd.read_csv(OUT + 'country_runs.csv')
r['relax'] = r.relax.fillna('none')
d = pd.read_csv(OUT + 'country_detail.csv.gz')
al = pd.read_csv(OUT + 'country_alloc.csv.gz')


def rid(**kw):
    q = r.copy()
    for k, v in kw.items():
        q = q[q[k] == v]
    assert len(q) == 1, kw
    return q.run_id.iloc[0]


def feedstock():
    rows = []
    for y in [2023, 2035]:
        for access, relax in [('constrained', 'none'), ('pool', 'L2+L3+L4')]:
            a = al[al.run_id == rid(block='detail', year=y, target='eat_quality', m=1.0, relax=relax)]
            g = a.assign(region=a.iso3c.map(H.region), group=a.oil.map(H.oil_group)) \
                .groupby(['region', 'group']).biofuel.sum().div(1e6).rename('Mt').reset_index()
            rows.append(g[g.Mt > 0].assign(year=y, access=access))
    out = pd.concat(rows)
    out.to_csv(OUT + 'fig3d_feedstock_region.csv', index=False)
    return out


def country():
    U = d.pivot(index='iso3c', columns='run_id', values='u')
    base = d[d.run_id == rid(block='detail', year=2035, target='eat_quality', m=1.0, relax='none')].set_index('iso3c')
    rows = []
    for tg in ['adequacy', 'eat_qty', 'eat_quality']:
        q = r[(r.year == 2035) & (r.target == tg) & (r.m == 1.0) & r.block.isin(['layers', 'detail'])]
        ids = dict(zip(q.relax, q.run_id))
        v = {k: U[ids[k]] for k in ids}
        sh = A.shapley(v, A.LAYERS)
        b1 = d[d.run_id == ids['none']].set_index('iso3c')
        b0 = d[d.run_id == rid(block='detail', year=2035, target=tg, m=0.0, relax='none')].set_index('iso3c')
        t = pd.DataFrame({'pop': b1['pop'], 'D': b1.D, 'u1': b1.u, 'u0': b0.u.reindex(b1.index)})
        t['attrib'] = t.u1 - t.u0
        for l in A.LAYERS:
            t['sh_' + l] = sh[l]
        rows.append(t.reset_index().assign(target=tg))
    c = pd.concat(rows)
    c['rel1'] = np.where(c.D > 0, c.u1 / c.D, np.nan)
    c['g_pc_day'] = c.u1 / c['pop'] / 365 * 1e6
    S = c[['sh_' + l for l in A.LAYERS]].copy()
    S['sh_L34'] = S.pop('sh_L3') + S.pop('sh_L4')
    c['dominant'] = np.where(c.u1 > 1e4, S.idxmax(axis=1).str.replace('sh_', ''), 'none')
    c['region'] = c.iso3c.map(H.region)
    c.to_csv(OUT + 'fig6_country.csv', index=False)
    return c


def fair():
    q = r[r.block == 'fair'].copy()
    q['tol'] = q.tol.fillna(-1)                      # -1: fairness first
    rows = []
    for _, x in q.iterrows():
        z = d[d.run_id == x.run_id]
        rows.append(dict(target=x.target, tol=x.tol, gap=x.gap / 1e6,
                         max_rel=(z.u / z.D.where(z.D > 0)).max(), n_short=(z.u > 1e4).sum()))
    f = pd.DataFrame(rows)
    f.to_csv(OUT + 'fair_check.csv', index=False)
    return f


if __name__ == '__main__':
    pd.set_option('display.width', 250)
    fs = feedstock()
    print(fs.pivot_table(index=['year', 'access', 'region'], columns='group', values='Mt', aggfunc='sum').round(2).fillna(0).to_string())
    c = country()
    q = c[c.target == 'eat_quality']
    print(q.sort_values('u1', ascending=False).head(15)[['iso3c', 'pop', 'D', 'u1', 'u0', 'attrib', 'rel1', 'g_pc_day',
                                                         'sh_L1', 'sh_L2', 'sh_L3', 'sh_L4', 'sh_L5', 'dominant']].round(3).to_string())
    print(q.sort_values('attrib', ascending=False).head(12)[['iso3c', 'u1', 'u0', 'attrib']].assign(
        u1=lambda x: x.u1 / 1e6, u0=lambda x: x.u0 / 1e6, attrib=lambda x: x.attrib / 1e6).round(3).to_string())
    print('attrib < 0 sum (Mt):', q.attrib[q.attrib < 0].sum() / 1e6, ' >0 sum:', q.attrib[q.attrib > 0].sum() / 1e6)
    print(q.groupby('dominant').agg(n=('iso3c', 'size'), pop=('pop', 'sum'), u1=('u1', 'sum')).assign(
        pop=lambda x: x['pop'] / 1e6, u1=lambda x: x.u1 / 1e6).round(2).to_string())
    print(fair().round(4).to_string())
