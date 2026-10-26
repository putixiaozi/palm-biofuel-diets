"""Inputs for Supplementary Figs 1-2: three-source reconciliation (2010-2023) and the 2035 projection inputs.

Outputs (results/): si_reconciliation.csv (global totals by source), si_production_country_2023.csv (FABIO vs PSD),
si_food_country_2023.csv (food oil per person,
this study vs FAOSTAT FBS), si_projection.csv (2023 vs 2035 supply by oil, biofuel by country, targets by region)
Run with python3 -I (reads the downloaded PSD file).
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import model as M
H = importlib.import_module('04_history')
B = importlib.import_module('03_build_base')

OUT = M.ROOT + 'results/'
PSD = M.RAW + 'usda_psd/psd_oilseeds.csv'
OILS = ['Oil, Coconut', 'Oil, Cottonseed', 'Oil, Olive', 'Oil, Palm', 'Oil, Palm Kernel', 'Oil, Peanut',
        'Oil, Rapeseed', 'Oil, Soybean', 'Oil, Sunflowerseed']


def reconciliation():
    p = pd.read_csv(PSD, usecols=['Commodity_Description', 'Country_Name', 'Market_Year', 'Attribute_Description', 'Value'])
    p = p[p.Commodity_Description.isin(OILS) & p.Market_Year.between(2010, 2023)]
    attr = {'Production': 'production', 'Food Use Dom. Cons.': 'food', 'Industrial Dom. Cons.': 'industrial'}
    psd = p[p.Attribute_Description.isin(attr)].groupby(['Market_Year', 'Attribute_Description']).Value.sum() \
        .div(1e3).rename('Mt').reset_index().rename(columns={'Market_Year': 'year'})
    psd['series'] = psd.Attribute_Description.map(attr)
    rows = [psd.assign(source='USDA PSD')[['year', 'series', 'source', 'Mt']]]
    prod = pd.read_csv(M.DD + 'base_production_by_country_oil_2010_2023.csv')
    rows.append(prod.groupby('year').production.sum().div(1e6).rename('Mt').reset_index()
                .assign(series='production', source='This study (FABIO)'))
    use = pd.read_csv(M.DD + 'base_use_by_country_oil_2010_2023.csv').fillna(0)
    for c, lab in [('food', 'food'), ('biofuel', 'biofuel'), ('industrial', 'industrial')]:
        rows.append(use.groupby('year')[c].sum().div(1e6).rename('Mt').reset_index()
                    .assign(series=lab, source='This study (FABIO x PSD shares; OECD-FAO biofuel)'))
    pop = pd.read_csv(M.DD + 'base_population_fat_2010_2023.csv')
    fbs = (pop.oilfat_fbs * pop['pop'] * M.G2T).groupby(pop.year).sum().div(1e6).rename('Mt').reset_index()
    rows.append(fbs.assign(series='food', source='FAOSTAT FBS'))
    out = pd.concat(rows)
    out.to_csv(OUT + 'si_reconciliation.csv', index=False)
    f = use[use.year == 2023].groupby('iso3c').food.sum()
    q = pop[pop.year == 2023].set_index('iso3c')
    c = pd.DataFrame({'pop': q['pop'], 'study_g_d': f.reindex(q.index) / q['pop'] / M.G2T, 'fbs_g_d': q.oilfat_fbs})
    c.dropna().query('pop > 1e6').reset_index().to_csv(OUT + 'si_food_country_2023.csv', index=False)
    # country production, 2023: FABIO vs PSD (EU as one producer)
    q = p[(p.Market_Year == 2023) & (p.Attribute_Description == 'Production')].copy()
    q['iso3c'] = q.Country_Name.map(B.psd_iso)
    q.loc[q.Country_Name == 'European Union', 'iso3c'] = 'EU'
    ps = q.dropna(subset=['iso3c']).groupby('iso3c').Value.sum().div(1e3)
    fb = prod[prod.year == 2023].assign(iso3c=lambda d: d.iso3c.where(~d.iso3c.isin(B.EU27), 'EU'))
    fb = fb[fb.oil.isin(B.PSD_OIL.values())].groupby('iso3c').production.sum().div(1e6)
    pc = pd.DataFrame({'fabio_Mt': fb, 'psd_Mt': ps}).fillna(0)
    pc = pc[pc.max(axis=1) > 0.05].reset_index().rename(columns={'index': 'iso3c'})
    pc.to_csv(OUT + 'si_production_country_2023.csv', index=False)
    return out


def projection():
    S23 = M.System(M.load_year(2023))
    S35 = M.project(S23, 2035)
    rows = []
    for yr, S in [(2023, S23), (2035, S35)]:
        sup = pd.Series(S.s.sum(0) * M.UNIT / 1e6, index=S.K).drop('uco')
        g = sup.groupby(sup.index.map(H.oil_group)).sum()
        rows.append(pd.DataFrame({'year': yr, 'panel': 'supply', 'key': g.index, 'Mt': g.values}))
        fb = pd.Series(S.fuel_base * M.UNIT / 1e6, index=S.C)
        fr = fb.groupby(fb.index.map(H.region)).sum()
        rows.append(pd.DataFrame({'year': yr, 'panel': 'biofuel', 'key': fr.index, 'Mt': fr.values}))
    t = pd.read_csv(OUT + 'targets_by_country.csv').query("target == 'eat_quality'")
    t['region'] = t.iso3c.map(H.region)
    g = t.groupby(['year', 'region'])[['D', 'obs', 'pop']].sum().reset_index()
    for c, lab in [('D', 'unsaturated target'), ('obs', 'unsaturated food, observed or projected'), ('pop', 'population')]:
        rows.append(g.assign(panel='targets', key=g.region + ' | ' + lab, Mt=g[c] / 1e6)[['year', 'panel', 'key', 'Mt']])
    out = pd.concat(rows)
    out.to_csv(OUT + 'si_projection.csv', index=False)
    return out


if __name__ == '__main__':
    pd.set_option('display.width', 250)
    r = reconciliation()
    print(r.pivot_table(index='year', columns=['series', 'source'], values='Mt').round(1).loc[[2010, 2015, 2020, 2023]].T.to_string())
    pr = projection()
    print(pr[pr.panel != 'targets'].pivot_table(index=['panel', 'key'], columns='year', values='Mt').round(2).to_string())
    print(pr[pr.panel == 'targets'].pivot_table(index='key', columns='year', values='Mt').round(1).to_string())
