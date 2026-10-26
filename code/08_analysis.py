"""Figure tables and key numbers from the LP runs (shortfall = u, which already includes any cut below current use).

Outputs (results/): fig2_decomposition.csv, fig3_layer_shapley.csv, fig4_frontier.csv, fig5_strategy_shapley.csv,
                    history_fuel_cost.csv, key_numbers.csv; from the robustness runs: si_sensitivity_attrib.csv,
                    si_pcap_frontier.csv, si_pcap_layers.csv, si_pcap_history.csv, si_pcap_strategy.csv; layer Shapley values of
                    the biofuel-attributable shortfall S(E1) - S(0): fig3_layer_shapley_attrib.csv
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import paths
import itertools
import math

import numpy as np
import pandas as pd

ROOT = paths.REPO
OUT = ROOT + 'results/'
LAYERS = ['L1', 'L2', 'L3', 'L4', 'L5']
LEVERS = ['S1', 'S2', 'S3', 'S4']


def key(sub):
    return '+'.join(sorted(sub)) or 'none'


def shapley(v, players):
    """v: dict subset-key -> shortfall; returns each player's average reduction."""
    n, out = len(players), {}
    for p in players:
        others = [q for q in players if q != p]
        out[p] = sum(math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n) * (v[key(s)] - v[key(s + (p,))])
                     for k in range(n) for s in itertools.combinations(others, k))
    return out


def main():
    r = pd.read_csv(OUT + 'runs_summary.csv')
    r['relax'] = r.relax.fillna('none')
    for c in ['gap', 'cut', 'E', 'gap_obs', 'fuel_unmet']:
        r[c] = r[c] / 1e6
    K = []
    # frontier
    f = r[r.block.isin(['frontier', 'frontier_pool'])].copy()
    f['access'] = np.where(f.block == 'frontier', 'constrained', 'pool')
    f[['year', 'target', 'access', 'm', 'E', 'gap', 'cut', 'fuel_unmet', 'gap_obs']].sort_values(
        ['year', 'target', 'access', 'm']).to_csv(OUT + 'fig4_frontier.csv', index=False)
    G = f.set_index(['year', 'target', 'access', 'm']).gap
    # two-factor decomposition: network (constrained vs pool) x biofuel (m vs 0), plus reallocation vs baseline
    rows = []
    for (y, tg), _ in f.groupby(['year', 'target']):
        base = f[(f.year == y) & (f.target == tg)].gap_obs.iloc[0]
        for m in [1.0, 1.5, 2.0]:
            p0, pm, c0, cm = G[y, tg, 'pool', 0.0], G[y, tg, 'pool', m], G[y, tg, 'constrained', 0.0], G[y, tg, 'constrained', m]
            rows.append(dict(year=y, target=tg, m=m, E=m * f[(f.year == y) & (f.m == 1.0)].E.iloc[0],
                             scarcity_no_fuel=p0, network=0.5 * ((c0 - p0) + (cm - pm)),
                             biofuel=0.5 * ((pm - p0) + (cm - c0)), total=cm,
                             baseline=base if m == 1.0 else np.nan,
                             reallocation=base - cm if m == 1.0 else np.nan,
                             biofuel_pool=pm - p0, biofuel_constrained=cm - c0))
    dec = pd.DataFrame(rows)
    dec.to_csv(OUT + 'fig2_decomposition.csv', index=False)
    # five constraint layers
    rows = []
    for (y, tg), g in r[r.block == 'layers'].groupby(['year', 'target']):
        v = dict(zip(g.relax, g.gap))
        sh = shapley(v, LAYERS)
        rows.append(dict(year=y, target=tg, total=v['none'], residual=v[key(LAYERS)], **sh,
                         **{'single_' + l: v[l] for l in LAYERS}, L3_L4=v['L3+L4'], L2_L3_L4=v['L2+L3+L4']))
    pd.DataFrame(rows).to_csv(OUT + 'fig3_layer_shapley.csv', index=False)
    # strategies
    rows = []
    for (tg, m), g in r[r.block == 'strategy'].groupby(['target', 'm']):
        v = dict(zip(g.strategies.fillna('none'), g.gap))
        rows.append(dict(target=tg, m=m, none=v['none'], combined=v[key(LEVERS)], **shapley(v, LEVERS),
                         **{'alone_' + s: v[s] for s in LEVERS}))
    pd.DataFrame(rows).to_csv(OUT + 'fig5_strategy_shapley.csv', index=False)
    # history: biofuel opportunity cost under the constrained network
    h = r[r.block == 'history'].pivot_table(index=['year', 'target'], columns='m', values='gap').reset_index()
    h.columns = ['year', 'target', 'S0', 'S1']
    obs = r[(r.block == 'history') & (r.m == 1.0)].set_index(['year', 'target'])
    h['observed'] = obs.gap_obs.reindex(pd.MultiIndex.from_frame(h[['year', 'target']])).values
    h['E'] = obs.E.reindex(pd.MultiIndex.from_frame(h[['year', 'target']])).values
    h['biofuel_cost'] = h.S1 - h.S0
    h['reallocation'] = h.observed - h.S1
    h.to_csv(OUT + 'history_fuel_cost.csv', index=False)
    # sensitivity
    s = r[r.block == 'sens'][['target', 'param', 'gap']]
    s.to_csv(OUT + 'sensitivity.csv', index=False)
    # key numbers
    for _, d in dec.iterrows():
        for c in ['scarcity_no_fuel', 'network', 'biofuel', 'total', 'baseline', 'reallocation', 'biofuel_pool',
                  'biofuel_constrained']:
            K.append((f'dec_{d.year}_{d.target}_m{d.m}_{c}', d[c]))
    for _, d in h.iterrows():
        for c in ['S0', 'S1', 'observed', 'biofuel_cost', 'reallocation', 'E']:
            K.append((f'hist_{d.year}_{d.target}_{c}', d[c]))
    pd.DataFrame(K, columns=['name', 'Mt']).to_csv(OUT + 'key_numbers.csv', index=False)
    pd.set_option('display.width', 250)
    print(dec.round(2).to_string())
    print(pd.read_csv(OUT + 'fig3_layer_shapley.csv').round(2).to_string())
    print(pd.read_csv(OUT + 'fig5_strategy_shapley.csv').round(2).to_string())
    print(h[h.year.isin([2010, 2015, 2023])].round(2).to_string())


def robustness():
    """Biofuel-attributable shortfall under each parameter sensitivity, and the main tables for the variant that
    caps palm in food (pcap) instead of protecting total oil food use."""
    r = pd.read_csv(OUT + 'robustness_runs.csv')
    m = pd.read_csv(OUT + 'runs_summary.csv')
    for d in (r, m):
        d['relax'] = d.relax.fillna('none')
        for c in ['gap', 'cut', 'E', 'gap_obs', 'fuel_unmet']:
            d[c] = d[c] / 1e6
    s1 = m[m.block == 'sens'].set_index(['target', 'param'])[['gap', 'cut']]
    s0 = r[r.block == 'sens0'].set_index(['target', 'param'])[['gap', 'cut']]
    t = s1.join(s0, lsuffix='_m1', rsuffix='_m0')
    t['attrib'] = t.gap_m1 - t.gap_m0
    t.reset_index().to_csv(OUT + 'si_sensitivity_attrib.csv', index=False)
    f = r[r.block.isin(['pcap_frontier', 'pcap_frontier_pool'])].copy()
    f['access'] = np.where(f.block == 'pcap_frontier', 'constrained', 'pool')
    f[['year', 'access', 'm', 'E', 'gap', 'cut', 'fuel_unmet', 'gap_obs']].sort_values(['year', 'access', 'm']) \
        .to_csv(OUT + 'si_pcap_frontier.csv', index=False)
    rows = []
    for y, g in r[r.block == 'pcap_layers'].groupby('year'):
        v = dict(zip(g.relax, g.gap))
        rows.append(dict(year=y, total=v['none'], residual=v[key(LAYERS)], **shapley(v, LAYERS),
                         **{'single_' + l: v[l] for l in LAYERS}))
    pd.DataFrame(rows).to_csv(OUT + 'si_pcap_layers.csv', index=False)
    h = r[r.block == 'pcap_history'].pivot_table(index='year', columns='m', values=['gap', 'cut'])
    h.columns = [f'{a}_m{b:g}' for a, b in h.columns]
    h['biofuel_cost'] = h.gap_m1 - h.gap_m0
    h.reset_index().to_csv(OUT + 'si_pcap_history.csv', index=False)
    g = r[r.block == 'pcap_strategy']
    v = dict(zip(g.strategies.fillna('none'), g.gap))
    pd.DataFrame([dict(target='eat_quality_pcap', m=1.0, none=v['none'], combined=v[key(LEVERS)], **shapley(v, LEVERS),
                       **{'alone_' + s: v[s] for s in LEVERS})]).to_csv(OUT + 'si_pcap_strategy.csv', index=False)
    for n in ['si_sensitivity_attrib', 'si_pcap_layers', 'si_pcap_history', 'si_pcap_strategy']:
        print(pd.read_csv(OUT + n + '.csv').round(2).to_string())


def layers_attrib():
    """Shapley values of S(E1) - S(0) over the five layers. At zero biofuel L3 and L4 do nothing, so S(0) of a subset is
    S(0) of its intersection with {L1, L2, L5} (structural and layers_m0 runs)."""
    r = pd.concat([pd.read_csv(OUT + 'runs_summary.csv'), pd.read_csv(OUT + 'layers_m0_runs.csv')])
    r['relax'] = r.relax.fillna('none')
    r['gap'] = r.gap / 1e6
    rows = []
    for (y, tg), g in r[r.block == 'layers'].groupby(['year', 'target']):
        v1 = dict(zip(g.relax, g.gap))
        z = r[r.block.isin(['structural', 'layers_m0']) & (r.year == y) & (r.target == tg) & (r.m == 0.0)]
        v0 = dict(zip(z.relax, z.gap))
        v0[key(('L1', 'L2', 'L5'))] = v0[key(LAYERS)]
        va = {}
        for n in range(6):
            for sub in itertools.combinations(LAYERS, n):
                va[key(sub)] = v1[key(sub)] - v0[key([l for l in sub if l in ('L1', 'L2', 'L5')])]
        rows.append(dict(year=y, target=tg, total=va['none'], residual=va[key(LAYERS)], **shapley(va, LAYERS),
                         **{'single_' + l: va[l] for l in LAYERS}))
    pd.DataFrame(rows).to_csv(OUT + 'fig3_layer_shapley_attrib.csv', index=False)
    print(pd.DataFrame(rows).round(2).to_string())


if __name__ == '__main__':
    main()
    robustness()
    layers_attrib()
