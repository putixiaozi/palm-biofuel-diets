"""External feedstock benchmarks and a volume-preserving EU mix sensitivity.

FAS values are assessment estimates, not direct measurements. EIA withheld values
remain missing. Benchmark coverage is explicitly reported and not extrapolated.
EU sensitivity keeps each member's OECD-derived total, changes oil preferences to
FAS EU shares, respects available domestic-use quantities, and conserves mass.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import model as M

ROOT=Path(M.ROOT);OUT=ROOT/'results';RAW=Path(M.RAW)/'revision_sources'
u=pd.read_csv(Path(M.DD)/'base_use_by_country_oil_2010_2023.csv')
records=[]
eu={'rapeseed':[6300,5850,6300,6100,5950,5800,6075,6200,6375],
    'soy':[500,550,700,1200,1070,900,780,950,1090],
    'sunflower':[210,250,230,250,270,240,225,300,260],
    'palm':[2000,2020,2300,2250,2600,1500,890,380,240]}
for oil,values in eu.items():
    for year,v in zip(range(2015,2024),values):
        records.append(dict(region='EU',year=year,oil=oil,external_Mt=v/1000,source='FAS EU 2024 Table 9',coverage='FAS assessment'))
for year,v in zip(range(2015,2024),[1104,3220,2576,5152,7084,7820,8786,10028,11592]):
    records.append(dict(region='IDN',year=year,oil='palm',external_Mt=v/1000,source='FAS Indonesia 2024',coverage='FAS assessment'))
for oil,values in {'soy':[2300,2760,2615,2714,3406,3609,4159,4445,5010,4687],
                   'palm':[0,0,0,32,66,112,155,158,147,171]}.items():
    for year,v in zip(range(2014,2024),values):
        records.append(dict(region='BRA',year=year,oil=oil,external_Mt=v/1000,source='FAS Brazil 2024 Table 13',coverage='Other feedstock not disaggregated'))
e=pd.read_excel(RAW/'EIA_feedstocks_monthly.xls',sheet_name='Data 3',header=2)
for word,oil in [('Total Canola','rapeseed'),('Total Corn','maize'),('Total Soybean','soy')]:
    col=next(c for c in e.columns if c.startswith(word))
    for year,g in e.groupby(e.Date.dt.year):
        if year>2023:continue
        vals=pd.to_numeric(g[col],errors='coerce')
        records.append(dict(region='USA',year=year,oil=oil,external_Mt=vals.sum(min_count=12)*0.00045359237,
                            source='EIA monthly feedstocks',coverage=f'{vals.count()} of 12 months reported'))
external=pd.DataFrame(records)
u['region']=u.iso3c.where(~u.iso3c.isin(M.EU27),'EU')
calc=u.groupby(['region','year','oil']).biofuel.sum().div(1e6).rename('reconstructed_Mt')
external=external.join(calc,on=['region','year','oil'])
external['difference_Mt']=external.reconstructed_Mt-external.external_Mt
external.to_csv(OUT/'external_feedstock_benchmarks.csv',index=False)
rows=[]
for year in range(2015,2024):
    S=M.System(M.load_year(year));pref=np.zeros(len(S.K))
    for oil,values in eu.items():pref[S.ki[oil]]=values[year-2015]
    for c in M.EU27:
        if c not in S.ci:continue
        j=S.ci[c];total=S.fuel_base[j]
        available=S.food0[j]+S.bio0[j]+S.nonfuel0[j]
        new=np.minimum(total*pref/pref.sum(),available)
        for _ in range(20):
            remainder=total-new.sum()
            if remainder<1e-9:break
            room=np.maximum(available-new,0)*(pref>0)
            if room.sum()<remainder:
                room=np.maximum(available-new,0)*S.eligibility()[j]
            assert room.sum()>=remainder-1e-8,(c,year)
            new+=remainder*room/room.sum()
        newfood=np.minimum(S.food0[j],np.maximum(available-new,0))
        S.food0[j]=newfood;S.bio0[j]=new;S.nonfuel0[j]=available-new-newfood
        assert abs(new.sum()-total)<1e-6
        assert S.nonfuel0[j].min()>-1e-7
    L,ic=M.build_links(S)
    for target in ['eat_qty','eat_quality']:
        tg=S.targets(target)
        z=[M.frontier_point(S,tg,L,ic,S.eligibility(),m) for m in [0.,1.]]
        assert all(x['feasible'] for x in z)
        row=dict(year=year,target=target,gap0_Mt=z[0]['gap']/1e6,gap1_Mt=z[1]['gap']/1e6,
                 attributable_Mt=(z[1]['gap']-z[0]['gap'])/1e6)
        rows.append(row);pd.DataFrame(rows).to_csv(OUT/'historical_EU_mix_sensitivity.csv',index=False)
        print(row,flush=True)
