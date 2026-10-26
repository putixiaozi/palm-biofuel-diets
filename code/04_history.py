"""Historical material-flow facts, 2010-2023 (Fig. 1 and Supplementary figures)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import paths
import numpy as np
import pandas as pd

DD, OUT = paths.REPO + 'data_derived/', paths.REPO + 'results/'
REG = pd.read_csv(paths.FABIO + 'regions.csv')
EU27 = set(REG.loc[REG.EU27, 'iso3c'])
CONT = REG.set_index('iso3c').continent.to_dict()
TROPICAL = ['palm', 'palmkernel', 'coconut']
GROUP = {'palm': 'Palm', 'palmkernel': 'Palm kernel & coconut', 'coconut': 'Palm kernel & coconut', 'soy': 'Soybean',
         'rapeseed': 'Rapeseed', 'sunflower': 'Sunflower'}


def oil_group(o):
    return GROUP.get(o, 'Other unsaturated')


def region(iso):
    named = {'CHN': 'China', 'IND': 'India', 'USA': 'United States', 'BRA': 'Brazil', 'ARG': 'Argentina',
             'IDN': 'Indonesia', 'MYS': 'Malaysia', 'THA': 'Thailand'}
    if iso in named:
        return named[iso]
    if iso in EU27:
        return 'EU27'
    if CONT.get(iso) == 'AFR':
        return 'Africa'
    return 'Rest of world'


def main():
    use = pd.read_csv(DD + 'base_use_by_country_oil_2010_2023.csv').fillna(0)
    prod = pd.read_csv(DD + 'base_production_by_country_oil_2010_2023.csv')
    flows = pd.read_csv(DD + 'base_bilateral_flows_2010_2023.csv.gz')
    pop = pd.read_csv(DD + 'base_population_fat_2010_2023.csv')
    # a. supply by oil group
    prod['group'] = prod.oil.map(oil_group)
    prod.groupby(['year', 'group']).production.sum().div(1e6).rename('Mt').reset_index() \
        .to_csv(OUT + 'fig1a_supply_by_oil.csv', index=False)
    # b. use by end use and oil class
    use['class'] = np.where(use.oil.isin(TROPICAL), 'Palm-type', 'Unsaturated')
    u = use.melt(id_vars=['year', 'iso3c', 'oil', 'class'], value_vars=['food', 'biofuel', 'nonfuel', 'stock'],
                 var_name='use', value_name='t')
    ub = u.groupby(['year', 'class', 'use']).t.sum().div(1e6).rename('Mt').reset_index()
    ub.to_csv(OUT + 'fig1b_use_by_class.csv', index=False)
    # c. biofuel feedstock by producing region and oil group
    use['region'] = use.iso3c.map(region)
    use['group'] = use.oil.map(oil_group)
    use.groupby(['year', 'region', 'group']).biofuel.sum().div(1e6).rename('Mt').reset_index() \
        .query('Mt > 0').to_csv(OUT + 'fig1c_biofuel_by_region_oil.csv', index=False)
    # d. co-production in crushing (2023, global)
    c = pd.read_csv(DD + 'fabio_crushing_coproduction_2010_2023.csv')
    c = c[(c.year == 2023) & (c.flow == 'output')]
    names = {'p067': 'Soybean', 'p069': 'Sunflower seed', 'p070': 'Rapeseed', 'p071': 'Cottonseed',
             'p072': 'Palm kernel', 'p073': 'Oil palm fruit', 'p068': 'Groundnut'}
    oil_items = {'c068', 'c069', 'c070', 'c071', 'c072', 'c073', 'c074'}
    c = c[c.proc.isin(names)]
    c['part'] = np.where(c.item.isin(oil_items), 'oil', 'co-product')
    cp = c.groupby(['proc', 'part']).value.sum().unstack().div(1e6)
    cp['oil_share'] = cp.oil / cp.sum(axis=1)
    cp.index = cp.index.map(names)
    cp.reset_index().rename(columns={'proc': 'crop'}).to_csv(OUT + 'fig1d_coproduction_2023.csv', index=False)
    # e. import dependence of food oil, 2023, and per-capita food oil
    f = flows[flows.year == 2023]
    imp = f[f['from'] != f['to']].groupby('to').t.sum()
    tot = f.groupby('to').t.sum()
    y = use[use.year == 2023].groupby('iso3c')[['food', 'biofuel']].sum()
    p = pop[pop.year == 2023].set_index('iso3c')
    y['pop'] = p['pop']
    y['food_gpd'] = y.food / (y['pop'] * 365e-6)
    y['fbs_oil_gpd'] = p.oilfat_fbs
    y['import_share'] = (imp / tot).reindex(y.index)
    y['region'] = y.index.map(region)
    y.reset_index().to_csv(OUT + 'fig1e_country_2023.csv', index=False)
    # summary numbers
    w = ub.pivot_table(index='year', columns='use', values='Mt', aggfunc='sum')
    w['supply'] = prod.groupby('year').production.sum() / 1e6
    w['biofuel_share'] = w.biofuel / w.supply
    un = ub[ub['class'] == 'Unsaturated'].pivot_table(index='year', columns='use', values='Mt')
    w['unsat_biofuel_share'] = un.biofuel / un.sum(axis=1)
    w.round(3).to_csv(OUT + 'fig1_summary_by_year.csv')
    print(w.round(2).loc[[2010, 2015, 2020, 2023]].to_string())
    print(cp.round(3).to_string())


if __name__ == '__main__':
    main()
