"""Build the observed country x oil allocation (2010-2023).

Backbone: FABIO v2 production and bilateral origin->destination flows (physically balanced).
Food vs industrial split of domestic use: USDA PSD (country x oil, market-year shares).
Biofuel use: OECD-FAO Agricultural Outlook 2026-2035 (vegetable oil 'Biofuel use').
Population and non-oil fat supply: FAOSTAT Food Balance Sheets.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import paths
import numpy as np
import pandas as pd
import pycountry

ROOT = paths.REPO
FAB = paths.FABIO
DD = ROOT + 'data_derived/'

OILS = {'c068': 'soy', 'c069': 'groundnut', 'c070': 'sunflower', 'c071': 'rapeseed', 'c072': 'cottonseed',
        'c073': 'palmkernel', 'c074': 'palm', 'c075': 'coconut', 'c076': 'sesame', 'c077': 'olive',
        'c078': 'ricebran', 'c079': 'maize'}            # c080 'other oilcrop oils' excluded (see SI Note 1)
PSD_OIL = {'Oil, Soybean': 'soy', 'Oil, Peanut': 'groundnut', 'Oil, Sunflowerseed': 'sunflower',
           'Oil, Rapeseed': 'rapeseed', 'Oil, Cottonseed': 'cottonseed', 'Oil, Palm Kernel': 'palmkernel',
           'Oil, Palm': 'palm', 'Oil, Coconut': 'coconut', 'Oil, Olive': 'olive'}
FUEL_OILS = ['soy', 'rapeseed', 'sunflower', 'palm', 'cottonseed', 'groundnut', 'maize']
PSD_FIX = {'Burma': 'MMR', 'Congo (Kinshasa)': 'COD', 'Gambia, The': 'GMB', 'Korea, North': 'PRK',
           'Korea, South': 'KOR', 'Turkey': 'TUR', 'Russia': 'RUS', 'Iran': 'IRN', 'Syria': 'SYR',
           'Taiwan': 'TWN', 'Vietnam': 'VNM', 'Tanzania': 'TZA', 'Bolivia': 'BOL', 'Venezuela': 'VEN',
           'Moldova': 'MDA', "Cote d'Ivoire": 'CIV', 'Laos': 'LAO', 'Czech Republic': 'CZE',
           'Belgium-Luxembourg': 'BEL', 'Hong Kong': 'HKG', 'North Macedonia': 'MKD'}

reg = pd.read_csv(FAB + 'regions.csv')
EU27 = set(reg.loc[reg.EU27, 'iso3c'])


def psd_iso(name):
    if name in PSD_FIX:
        return PSD_FIX[name]
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None


def load_psd():
    df = pd.read_csv(ROOT + 'data_raw/usda_psd/psd_oilseeds.csv')
    df = df[df.Commodity_Description.isin(PSD_OIL) & (df.Market_Year.between(2009, 2024))]
    df = df[df.Attribute_Description.isin(['Food Use Dom. Cons.', 'Industrial Dom. Cons.', 'Feed Waste Dom. Cons.'])]
    df['oil'] = df.Commodity_Description.map(PSD_OIL)
    df['iso3c'] = df.Country_Name.map(psd_iso)
    df.loc[df.Country_Name == 'European Union', 'iso3c'] = 'EU'
    df = df.dropna(subset=['iso3c'])
    p = df.pivot_table(index=['iso3c', 'oil', 'Market_Year'], columns='Attribute_Description',
                       values='Value', aggfunc='sum').fillna(0).reset_index()
    p.columns = ['iso3c', 'oil', 'year', 'feedwaste', 'food', 'industrial']
    return p


def psd_food_share(psd, year):
    """Food share of domestic use, 3-market-year mean centred on `year` (shrinks year-to-year noise)."""
    w = psd[psd.year.between(year - 1, year + 1)].groupby(['iso3c', 'oil'])[['food', 'industrial', 'feedwaste']].sum()
    tot = w.sum(axis=1)
    return (w.food / tot.replace(0, np.nan)).dropna().rename('phi')


def load_oecd_bf():
    df = pd.read_csv(ROOT + 'data_raw/oecd_fao/oecd_fao_outlook_2026_2035_oilcomplex.csv', low_memory=False)
    v = df[(df.COMMODITY == 'CPC_216') & (df.MEASURE == 'BF')]
    return v.pivot_table(index='REF_AREA', columns='TIME_PERIOD', values='OBS_VALUE') * 1000  # tonnes


def load_fbs():
    f = pd.read_csv(DD + 'fbs_oil_fat_2010_2023.csv')
    f = f[f['Area Code'] != 351]
    f['iso3c'] = f['Area Code'].map(reg.set_index('area_code').iso3c)
    return f.dropna(subset=['iso3c'])


def build_year(year, bil, out, psd, bf, fbs):
    b = bil[(bil.year == year) & bil.item.isin(OILS)].copy()
    b['oil'] = b.item.map(OILS)
    b['use'] = np.where(b.use.str.startswith('int_'), 'domestic', b.use)
    b.loc[b.use.isin(['food', 'other', 'tourist']), 'use'] = 'domestic'
    flows = b.groupby(['from', 'to', 'oil']).value.sum().rename('t').reset_index()       # all uses, for trade network
    dest = b.groupby(['to', 'oil', 'use']).value.sum().unstack(fill_value=0)
    dest.index.names = ['iso3c', 'oil']
    tab = pd.DataFrame(index=dest.index)
    tab['stock'] = dest.get('stock_addition', 0)
    tab['dom'] = dest['domestic']                     # food + industrial + feed (+ tourist) at destination
    # FABIO's own food share, fallback where PSD has no record
    fab_food = b[b.use.eq('domestic') & bil.loc[b.index, 'use'].isin(['food', 'tourist'])].groupby(['to', 'oil']).value.sum()
    fab_food.index.names = ['iso3c', 'oil']
    tab['phi_fabio'] = (fab_food / tab.dom.replace(0, np.nan)).reindex(tab.index)
    phi = psd_food_share(psd, year)
    key = [('EU' if i in EU27 else i, o) for i, o in tab.index]
    tab['phi_psd'] = phi.reindex(pd.MultiIndex.from_tuples(key)).values
    tab['phi'] = tab.phi_psd.fillna(tab.phi_fabio).fillna(1.0).clip(0, 1)
    tab['phi_src'] = np.where(tab.phi_psd.notna(), 'PSD', 'FABIO')
    tab['food'] = tab.phi * tab.dom
    tab['industrial'] = tab.dom - tab.food
    # biofuel: OECD country totals; EU split by member-state industrial use of fuel oils
    tab['biofuel'] = 0.0
    ind = tab.industrial.unstack(fill_value=0)
    fuel_ind = ind[[c for c in FUEL_OILS if c in ind]].copy()
    if 'coconut' in ind:
        fuel_ind.loc[fuel_ind.index == 'PHL', 'coconut'] = ind.loc[ind.index == 'PHL', 'coconut']
    fuel_ind = fuel_ind.fillna(0)
    if 'USA' in fuel_ind.index:                       # palm-based fuel does not qualify under the US RFS
        fuel_ind.loc['USA', 'palm'] = 0.0
    bf_y = bf[year].dropna() if year in bf else pd.Series(dtype=float)
    alloc = {}
    for iso, val in bf_y.items():
        if iso == 'EU':
            members = [m for m in fuel_ind.index if m in EU27]
            w = fuel_ind.loc[members].sum(axis=1)
            for m in members:
                alloc[m] = val * w[m] / w.sum()
        elif len(iso) == 3 and iso in fuel_ind.index:
            alloc[iso] = val
    shortfall = 0.0
    for iso, val in alloc.items():
        row = fuel_ind.loc[iso]
        cap = row.sum()
        use = min(val, cap)
        for oil, x in row.items():
            if cap > 0 and x > 0:
                tab.loc[(iso, oil), 'biofuel'] = use * x / cap
        if val > cap:
            # OECD biofuel exceeds the PSD-based industrial use: move the excess from food use of fuel oils
            ok = [o for o in row.index if (o != 'coconut' or iso == 'PHL') and (o != 'palm' or iso != 'USA')]
            fd = tab.loc[iso, 'food'].reindex(ok).fillna(0)
            extra = min(val - cap, fd.sum())
            shortfall += val - cap - extra
            for oil, x in fd.items():
                if x > 0:
                    d = extra * x / fd.sum()
                    tab.loc[(iso, oil), ['biofuel', 'food', 'industrial']] += [d, -d, d]
    tab['nonfuel'] = tab.industrial - tab.biofuel
    tab = tab.reset_index()
    tab['year'] = year
    prod = out[(out.year == year) & out.item.isin(OILS)].assign(oil=lambda d: d.item.map(OILS))
    prod = prod.groupby(['iso3c', 'oil']).X.sum().rename('production').reset_index()
    prod['year'] = year
    flows['year'] = year
    # population and fat supply
    y = fbs[fbs.year == year]
    g = lambda it, el: y[(y['Item Code'] == it) & (y['Element Code'] == el)].groupby('iso3c').value.sum()
    pop = pd.DataFrame({'pop': g(2501, 511) * 1000, 'kcal': g(2901, 664), 'fat': g(2901, 684),
                        'oilfat_fbs': g(2914, 684), 'oilkcal_fbs': g(2914, 664)})
    pop['year'] = year
    return tab, prod, flows, pop.reset_index(), shortfall


def main():
    bil = pd.read_csv(DD + 'fabio_oil_bilateral_use_2010_2023.csv.gz')
    out = pd.read_csv(DD + 'fabio_oil_seed_cake_output_2010_2023.csv')
    psd, bf, fbs = load_psd(), load_oecd_bf(), load_fbs()
    T, P, F, POP = [], [], [], []
    for yr in range(2010, 2024):
        tab, prod, flows, pop, sf = build_year(yr, bil, out, psd, bf, fbs)
        T.append(tab); P.append(prod); F.append(flows); POP.append(pop)
        print(yr, 'biofuel allocated %.1f Mt (unallocated %.2f Mt)' % (tab.biofuel.sum() / 1e6, sf / 1e6))
    pd.concat(T).to_csv(DD + 'base_use_by_country_oil_2010_2023.csv', index=False)
    pd.concat(P).to_csv(DD + 'base_production_by_country_oil_2010_2023.csv', index=False)
    pd.concat(F).to_csv(DD + 'base_bilateral_flows_2010_2023.csv.gz', index=False)
    pd.concat(POP).to_csv(DD + 'base_population_fat_2010_2023.csv', index=False)


if __name__ == '__main__':
    main()
