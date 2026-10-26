"""Revision analyses. All alternative settings are explicitly scenario assumptions.

Partial palm eligibility caps palm at a scenario fraction of national requirements
in the USA and EU, without treating this fraction as a legal allowance.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import model as M

ROOT = Path(M.ROOT)
OUT = ROOT/'results'
BASE = M.System(M.load_year(2023))
FUTURE = M.project(BASE)
ROWS, COUNTRY = [], []


def evaluate(name, m, year=2035, relocation=0., palm_share=0., uco=0.,
             domestic_uco=False, supply=1., target='eat_quality', pcap=False,
             grouped_cap=False, fair=False):
    S = (FUTURE if year == 2035 else BASE).copy()
    if uco:
        S.s[:, S.uco] = uco*S.food0[:, S.virgin].sum(axis=1)
    tg = S.targets(target, palm_cap=pcap)
    L, cap = M.build_links(S)
    if domestic_uco:
        L = L[(L[:,2] != S.uco) | (L[:,0] == L[:,1])]
    fuel_cap = S.fuel_base*m*(1+relocation)
    fuel_lower = S.fuel_base*m*(1-relocation) if relocation else None
    eligible = S.eligibility(policy=palm_share == 0)
    pc = None
    if palm_share:
        pc = np.full(len(S.C), np.inf)
        for c in M.NO_PALM_FUEL:
            if c in S.ci:
                j = S.ci[c];pc[j] = palm_share*S.fuel_base[j]*m
    food_cap = None
    if grouped_cap:
        food_cap = np.where(S.pop>0, M.EAT_PALM*M.G2T*S.pop/M.UNIT, np.inf)
    z = M.solve(S,tg,L,cap,fuel_cap,eligible,fuel_min=m*S.fuel_base.sum()*(1-1e-7),
                supply_scale=supply, fuel_lower=fuel_lower, palm_fuel_cap=pc,
                palm_type_food_cap=food_cap, mode='waterfill' if fair else 'min_gap',
                gap_tol=1e-4 if fair else None, detail=True)
    assert z['feasible'], (name, m, z)
    if relocation and z['fuel_unmet'] < 1:
        delivered = z['B'].sum(axis=1)/M.UNIT
        assert np.max(delivered-fuel_cap) < 1e-4
        assert np.min(delivered-fuel_lower) > -1e-4
    if pc is not None:
        assert np.max(z['B'][:,S.palm]/M.UNIT-pc) < 1e-4
    row = dict(scenario=name,year=year,target=target,m=m,requested_Mt=m*S.fuel_base.sum()/1000,
               delivered_Mt=z['fuel']/1e6,unmet_Mt=z['fuel_unmet']/1e6,
               cut_Mt=z['cut']/1e6,gap_Mt=z['gap']/1e6,relocation=relocation,
               palm_share=palm_share,uco=uco,supply_scale=supply)
    ROWS.append(row)
    pd.DataFrame(ROWS).to_csv(OUT/'revision_scenarios.csv',index=False)
    if fair:
        c = pd.DataFrame(dict(scenario=name,m=m,iso3c=S.C,shortfall_Mt=z['u']/1e6,
                              population=S.pop,target_Mt=tg['D']/1000))
        COUNTRY.append(c)
        pd.concat(COUNTRY).to_csv(OUT/'revision_country_allocations.csv',index=False)
    print(row,flush=True)
    return row


# Bounded relocation, full eligibility and the option comparison are in 20_relocation_options.py.
a=evaluate('baseline',0); b=evaluate('baseline',1)
for share in [.1,.25,.5,1.]:
    for m in [0.,1.]:evaluate(f'palm_share_{share:g}',m,palm_share=share)
for uco in [.025,.05,.075]:
    for m in [0.,1.]:evaluate(f'uco_domestic_{uco:g}',m,uco=uco,domestic_uco=True)
for supply in [1.05,1.10,1.20]:
    for m in [0.,1.]:evaluate(f'output_{supply:g}',m,supply=supply)
for year in [2023,2035]:
    for m in [0.,1.]:
        evaluate('pcap_alt',m,year=year,pcap=True)
        evaluate('grouped_saturated_cap',m,year=year,grouped_cap=True)
print('REVISION CHECKS COMPLETE',flush=True)
