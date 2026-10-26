"""Allocation model for vegetable oils: food targets versus biofuel use under physical and policy constraints.

Constraint layers (each can be relaxed in experiments):
  L1 production   oil output by country and oil type is fixed (relaxation: +10% output)
  L2 trade access each country's oil imports stay within its 2010-2023 maximum (x import growth for 2035)
  L3 conversion   biofuel is made only where biofuel is made today (2035: OECD-FAO projection), within that volume
  L4 eligibility  palm oil is not an eligible feedstock in the EU and the US
  L5 demand       no country consumes less of the target oil group than it does now (rigid demand)

Food targets are soft (shortfall u_j >= 0). If the hard layers cannot all hold at the requested fuel volume,
the model reports the smallest cut v_j below current consumption that restores feasibility.
The LP works in kt; results are reported in tonnes.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import paths
import copy
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog

DATA_ROOT = paths.REPO               # inputs (read only)
ROOT = paths.REPO                    # outputs
DD = DATA_ROOT + 'data_derived/'
RAW = DATA_ROOT + 'data_raw/'
UNSAT = ['soy', 'groundnut', 'sunflower', 'rapeseed', 'cottonseed', 'sesame', 'olive', 'ricebran', 'maize']
FUEL_OILS = ['soy', 'rapeseed', 'sunflower', 'palm', 'cottonseed', 'groundnut', 'maize', 'coconut']
EAT_UNSAT, EAT_PALM = 40.0, 6.8          # g/person/day, Willett et al. 2019 reference intakes
EAT_TOTAL = EAT_UNSAT + EAT_PALM
FAT_SHARE = 0.20                          # minimum share of dietary energy from fat (FAO 2010)
G2T = 365e-6                              # g/person/day x persons -> tonnes/yr
UNIT = 1e3                                # tonnes per LP unit (kt)
REG = pd.read_csv(paths.FABIO + 'regions.csv')
EU27 = set(REG.loc[REG.EU27, 'iso3c'])
CONT = REG.set_index('iso3c').continent.to_dict()
OECD_CONT = {'AFR': 'F', 'ASI': 'S', 'LAM': 'A8', 'EU': 'EU', 'EUR': 'E', 'OCE': 'O', 'NAM': 'A2', 'ROW': 'W'}
NO_PALM_FUEL = EU27 | {'USA'}


def load_year(year, hist=(2010, 2023)):
    use = pd.read_csv(DD + 'base_use_by_country_oil_2010_2023.csv')
    prod = pd.read_csv(DD + 'base_production_by_country_oil_2010_2023.csv')
    flows = pd.read_csv(DD + 'base_bilateral_flows_2010_2023.csv.gz')
    pop = pd.read_csv(DD + 'base_population_fat_2010_2023.csv')
    fl_h = flows[flows.year.between(*hist)]
    links = fl_h[fl_h.t > 0].groupby(['from', 'to', 'oil']).t.max()
    imp = fl_h[fl_h['from'] != fl_h['to']].groupby(['year', 'to']).t.sum().unstack(0)
    return dict(year=year, use=use[use.year == year].fillna(0), prod=prod[prod.year == year], links=links,
                imp_max=imp.max(axis=1), x0=flows[flows.year == year].set_index(['from', 'to', 'oil']).t,
                pop=pop[pop.year == year].set_index('iso3c'))


class System:
    """Data of one allocation problem (a year or a projection), indexed by country x oil (kt)."""

    def __init__(self, d):
        u, pr = d['use'], d['prod']
        self.year = d['year']
        self.C = sorted(set(u.iso3c) | set(pr.iso3c))
        self.K = sorted(u.oil.unique()) + ['uco']           # 'uco': additional used cooking oil (fuel only)
        self.ci = {c: n for n, c in enumerate(self.C)}
        self.ki = {k: n for n, k in enumerate(self.K)}
        nC, nK = len(self.C), len(self.K)
        self.s = np.zeros((nC, nK))
        for r in pr.itertuples():
            self.s[self.ci[r.iso3c], self.ki[r.oil]] += r.production / UNIT
        U = lambda col: (u.pivot_table(index='iso3c', columns='oil', values=col, aggfunc='sum')
                         .reindex(index=self.C, columns=self.K).fillna(0).values / UNIT)
        self.food0, self.bio0, self.nonfuel0, self.stock0 = U('food'), U('biofuel'), U('nonfuel'), U('stock')
        self.fuel_base = self.bio0.sum(1)
        self.links = d['links'] / UNIT
        self.imp_max = d['imp_max'].reindex(self.C).fillna(0).values / UNIT
        x0 = d['x0'][d['x0'].index.get_level_values(0) != d['x0'].index.get_level_values(1)]
        self.imp_obs = x0.groupby(level=1).sum().reindex(self.C).fillna(0).values / UNIT
        p = d['pop'].reindex(self.C)
        self.pop = p['pop'].fillna(0).values
        self.kcal, self.fat = p.kcal.fillna(0).values, p.fat.fillna(0).values
        self.oilfat_fbs, self.oilkcal_fbs = p.oilfat_fbs.fillna(0).values, p.oilkcal_fbs.fillna(0).values
        self.fbs_food = self.oilfat_fbs * G2T * self.pop / UNIT     # FAOSTAT vegetable-oil food supply, kt
        self.virgin = np.array([k != 'uco' for k in self.K])
        self.unsat_idx = [self.ki[k] for k in UNSAT if k in self.ki]
        self.palm, self.uco = self.ki['palm'], self.ki['uco']

    def copy(self):
        return copy.deepcopy(self)

    def targets(self, kind, waste=0.0, fat_share=FAT_SHARE, flexible=False, palm_cap=False):
        """Food target of each country (kt).

        Returns mask (oil group counted), H (hard floor = current use of the group, or the converged level when
        flexible), D (target = max(current, reference need)), P (palm cap in food), T (reference need).
        Flexible demand: countries above the EAT-Lancet reference may converge to it; the reducible amount is
        bounded by the lower of model food use and FAOSTAT food supply, so oil processed for re-export is not
        credited as a dietary saving.
        eat_quality with palm_cap=False drops the EAT-Lancet palm limit and instead protects each country's total
        vegetable-oil intake (H2), so palm leaves a diet only when unsaturated oil replaces it.
        """
        g = G2T * self.pop / UNIT
        has = self.pop > 0
        sc = 1.0 / (1.0 - waste)
        food = self.food0[:, self.virgin].sum(1)
        unsat = self.food0[:, self.unsat_idx].sum(1)
        palm = self.food0[:, self.palm]
        mask = self.virgin.copy()
        P = np.full(len(self.C), np.inf)
        R = EAT_TOTAL * sc * g
        obs, fbs = food, self.fbs_food
        if kind == 'adequacy':
            oil_gd = np.where(has, food / np.maximum(g, 1e-12), 0)
            kcal_adj = self.kcal - self.oilkcal_fbs + oil_gd * 8.84
            need = np.maximum(0, fat_share * kcal_adj / 9 - (self.fat - self.oilfat_fbs)) * sc
            T = np.where(has, need * g, 0)
        elif kind == 'eat_qty':
            T = np.where(has, R, 0)
        elif kind == 'eat_quality':
            if palm_cap:
                P = np.where(has, EAT_PALM * sc * g, np.inf)
            T = np.where(has, np.maximum(EAT_UNSAT * sc * g, R - np.minimum(palm, P)), 0)
            H2 = food.copy()
            if flexible:
                H2 = np.where(has, food - np.maximum(0, np.minimum(food, self.fbs_food) - R), food)
            obs = unsat
            fbs = self.fbs_food * np.where(food > 0, unsat / np.maximum(food, 1e-12), 0)
            mask = np.zeros(len(self.K), bool); mask[self.unsat_idx] = True
            R = EAT_UNSAT * sc * g
        else:
            raise ValueError(kind)
        H = obs.copy()
        if flexible:
            excess = np.maximum(0, np.minimum(obs, fbs) - np.maximum(R, T))
            H = np.where(has, obs - excess, obs)
        D = np.maximum(H, T)
        out = dict(kind=kind, mask=mask, H=H, D=D, P=P, T=T, obs=obs, gap_obs=np.maximum(0, T - obs).sum() * UNIT)
        if kind == 'eat_quality' and not palm_cap:
            out['H2'] = H2
        return out

    def eligibility(self, policy=True):
        """Feedstock eligibility. policy=True applies the rules in force in the system year: palm excluded in
        the US (RFS) throughout, and in the EU from 2030 (Delegated Regulation 2019/807 phase-out)."""
        e = np.zeros((len(self.C), len(self.K)), bool)
        for k in FUEL_OILS + ['uco']:
            e[:, self.ki[k]] = True
        e[:, self.ki['coconut']] = False                     # coconut methyl ester only in the Philippines
        if 'PHL' in self.ci:
            e[self.ci['PHL'], self.ki['coconut']] = True
        if policy:
            for c in (NO_PALM_FUEL if self.year >= 2030 else {'USA'}):
                if c in self.ci:
                    e[self.ci[c], self.palm] = False
        return e

    def fuel_cap(self, E=None, mode='footprint'):
        """Country biofuel capacity (kt). footprint: today's (or projected) producers, scaled to total E."""
        if mode == 'anywhere':
            return None
        base = self.fuel_base
        return base * (1.0 if E is None else E / base.sum())


# ------------------------------------------------------------------------------------------------ projection
def load_oecd():
    df = pd.read_csv(RAW + 'oecd_fao/oecd_fao_outlook_2026_2035_oilcomplex.csv', low_memory=False,
                     usecols=['REF_AREA', 'COMMODITY', 'MEASURE', 'TIME_PERIOD', 'OBS_VALUE'])
    v = df[df.COMMODITY == 'CPC_216']
    return v.pivot_table(index=['REF_AREA', 'MEASURE'], columns='TIME_PERIOD', values='OBS_VALUE')


def project(S, year=2035, oecd=None, floor='percap'):
    """Scale a base-year system with OECD-FAO Outlook growth (country if covered, else EU or continent).

    Production and imports follow the country's OECD-FAO growth; other industrial use follows world growth;
    food floors keep base-year use per person (floor='percap') or follow OECD-FAO food projections ('oecd').
    Import ceilings never fall below the 2010-2023 maximum.
    """
    o = load_oecd() if oecd is None else oecd
    base = S.year
    areas3 = {a for a, _ in o.index if len(a) == 3}

    def growth(meas, areas):
        for a in areas:
            if (a, meas) in o.index:
                x0, x1 = o.loc[(a, meas), base], o.loc[(a, meas), year]
                if x0 > 1.0 and np.isfinite(x1):            # values in kt; ignore negligible bases
                    return x1 / x0
        return np.nan

    P = S.copy()
    G = {}
    for c in S.C:
        cont = OECD_CONT.get(CONT.get(c, 'ROW'), 'W')
        chain = ([c] if c in areas3 else []) + (['EU'] if c in EU27 else []) + [cont, 'W']
        G[c] = {m: growth(m, chain) for m in ['QP', 'FO', 'OU', 'IM']}
        G[c]['POP'] = growth('FO', chain) / growth('FO_PC', chain)
    g = pd.DataFrame(G).T.reindex(S.C)
    g = g.fillna(g.loc[:, :].median())
    P.s[:, P.virgin] *= g.QP.values[:, None]
    gf = (g.POP if floor == 'percap' else g.FO).values
    P.food0 *= gf[:, None]
    P.nonfuel0 *= growth('OU', ['W'])
    P.stock0[:] = 0.0
    P.pop = S.pop * g.POP.values
    P.fbs_food = S.fbs_food * gf
    P.imp_max = np.maximum(S.imp_max, S.imp_obs * g.IM.values)
    # biofuel: OECD country volumes; the EU total split by member states' base-year shares
    bf = np.zeros(len(S.C))
    for a in areas3 | {'EU'}:
        if (a, 'BF') not in o.index:
            continue
        val = o.loc[(a, 'BF'), year] / 1.0                      # kt
        if not np.isfinite(val) or val < 1.0:
            continue
        if a == 'EU':
            mem = [S.ci[m] for m in EU27 if m in S.ci]
            w = S.fuel_base[mem]
            bf[mem] += val * w / w.sum()
        elif a in S.ci:
            bf[S.ci[a]] += val
    P.fuel_base = bf
    mix = np.where(S.fuel_base[:, None] > 0, S.bio0 / np.maximum(S.fuel_base[:, None], 1e-12), 0)
    P.bio0 = mix * bf[:, None]
    P.year = year
    P.growth = g
    return P


# ------------------------------------------------------------------------------------------------ LP
def build_links(S, access='historical', delta=0.0):
    """Shipment variables (i, j, k) and per-importer import caps (None = no cap)."""
    nC, nK = len(S.C), len(S.K)
    L = set()
    if access == 'historical':
        for (a, b, k) in S.links.index:
            if a in S.ci and b in S.ci and k in S.ki:
                L.add((S.ci[a], S.ci[b], S.ki[k]))
    else:                                                   # 'complete' or 'pooled'
        for i, k in zip(*np.nonzero(S.s[:, S.virgin] > 0)):
            k = np.where(S.virgin)[0][k]
            L.update((i, j, k) for j in range(nC))
    for i, k in zip(*np.nonzero(S.s > 0)):
        L.add((i, i, k))
        if k == S.uco:                                      # used cooking oil trades freely to fuel producers
            L.update((i, j, k) for j in range(nC))
    L = np.array(sorted(L))
    imp_cap = None if access == 'pooled' else S.imp_max * (1 + delta) * (1 + 1e-6) + 1e-3
    return L, imp_cap


def solve(S, tgt, L, imp_cap=None, fuel_cap=None, eligible=None, fuel_min=0.0, mode='min_gap',
          supply_scale=1.0, detail=False, gap_tol=None, fuel_lower=None,
          palm_fuel_cap=None, palm_type_food_cap=None, tie_weights=None):
    """Lexicographic LP.

    mode 'min_gap' : minimise total shortfall at fuel >= fuel_min (kt)
    mode 'max_fuel': minimise shortfall at fuel >= fuel_min, then maximise fuel at that shortfall
    mode 'equity'  : minimise the largest relative shortfall, then total shortfall
    mode 'weighted': total shortfall within (1 + gap_tol) of its minimum, then sum_j w_j u_j minimised (tie_weights)
    mode 'waterfill': lexicographic max-min of relative shortfall u_j / D_j (progressive filling); with gap_tol
                      the total shortfall is first held within (1 + gap_tol) of its minimum (efficient and fair),
                      without it fairness comes first
    If no allocation satisfies all hard layers, the model first minimises biofuel that cannot be delivered at
    all (fuel_unmet), then the consumption cut below the hard floors, then the shortfall.
    """
    nC, nK = len(S.C), len(S.K)
    nL, nF = len(L), nC * nK
    oF, oB, oW, oU, oV = nL, nL + nF, nL + 2 * nF, nL + 3 * nF, nL + 3 * nF + nC
    oT = oV + nC
    oZ = oT + 1
    nv = oZ + 1
    if eligible is None:
        eligible = S.eligibility(True)
    if fuel_cap is None:
        fuel_cap = np.full(nC, np.inf)
    ub = np.full(nv, np.inf)
    ub[oB:oB + nF] = np.where((eligible & (fuel_cap[:, None] > 0)).ravel(), np.inf, 0.0)
    Fub = np.full((nC, nK), np.inf)
    Fub[:, S.uco] = 0.0
    Fub[:, S.palm] = tgt['P']
    ub[oF:oF + nF] = Fub.ravel()
    ub[oU:oU + nC] = tgt['D']
    ub[oV:oV + nC] = 0.0
    ub[oT] = 1.0
    ub[oZ] = 0.0
    i, j, k = L[:, 0], L[:, 1], L[:, 2]
    jk = np.arange(nF)
    rows = np.concatenate([i * nK + k, nF + j * nK + k, nF + jk, nF + jk, nF + jk])
    cols = np.concatenate([np.arange(nL), np.arange(nL), oF + jk, oB + jk, oW + jk])
    vals = np.concatenate([np.ones(nL), np.ones(nL), -np.ones(3 * nF)])
    A_eq = sp.csr_matrix((vals, (rows, cols)), shape=(2 * nF, nv))
    s = S.s.copy()
    s[:, S.virgin] *= supply_scale
    b_eq = np.concatenate([s.ravel(), (S.nonfuel0 + S.stock0).ravel()])
    R, Cc, V, b = [], [], [], []
    r = 0
    m = np.where(tgt['mask'])[0]
    jj, kk = np.repeat(np.arange(nC), len(m)), np.tile(m, nC)
    ar = np.arange(nC)
    for slack, rhs in ((oV, tgt['H']), (oU, tgt['D'])):     # hard floor (cut v), soft target (shortfall u)
        R += [r + jj, r + ar]; Cc += [oF + jj * nK + kk, slack + ar]; V += [-np.ones(len(jj)), -np.ones(nC)]
        b.append(-rhs); r += nC
    if 'H2' in tgt:                                        # total-intake floor, shares the cut variable
        jv, kv = np.repeat(np.arange(nC), nK - 1), np.tile(np.where(S.virgin)[0], nC)
        R += [r + jv, r + ar]; Cc += [oF + jv * nK + kv, oV + ar]; V += [-np.ones(len(jv)), -np.ones(nC)]
        b.append(-tgt['H2']); r += nC
    fin = np.where(np.isfinite(fuel_cap))[0]
    R.append(r + np.repeat(np.arange(len(fin)), nK)); Cc.append(oB + (np.repeat(fin, nK) * nK + np.tile(np.arange(nK), len(fin))))
    V.append(np.ones(len(fin) * nK)); b.append(fuel_cap[fin]); r += len(fin)
    if fuel_lower is not None:
        countries = np.where(fuel_lower > 0)[0]
        R.append(r + np.repeat(np.arange(len(countries)), nK))
        Cc.append(oB + np.repeat(countries, nK) * nK + np.tile(np.arange(nK), len(countries)))
        V.append(-np.ones(len(countries) * nK)); b.append(-fuel_lower[countries]); r += len(countries)
    if palm_fuel_cap is not None:
        countries = np.where(np.isfinite(palm_fuel_cap))[0]
        R.append(r + np.arange(len(countries))); Cc.append(oB + countries * nK + S.palm)
        V.append(np.ones(len(countries))); b.append(palm_fuel_cap[countries]); r += len(countries)
    if palm_type_food_cap is not None:
        countries = np.where(np.isfinite(palm_type_food_cap))[0]
        oils = [S.ki[k] for k in ['palm', 'palmkernel', 'coconut']]
        R.append(r + np.repeat(np.arange(len(countries)), len(oils)))
        Cc.append(oF + np.repeat(countries, len(oils)) * nK + np.tile(oils, len(countries)))
        V.append(np.ones(len(countries) * len(oils))); b.append(palm_type_food_cap[countries]); r += len(countries)
    if imp_cap is not None:
        f = (i != j) & (k != S.uco)
        R.append(r + j[f]); Cc.append(np.where(f)[0]); V.append(np.ones(f.sum())); b.append(imp_cap); r += nC
    r_fuel, r_gap, r_cut = r, r + 1, r + 2
    R += [np.full(nF + 1, r_fuel), np.full(nC, r_gap), np.full(nC, r_cut)]
    Cc += [np.append(oB + jk, oZ), oU + ar, oV + ar]
    V += [-np.ones(nF + 1), np.ones(nC), np.ones(nC)]
    b.append(np.array([-fuel_min, tgt['D'].sum() + 1.0, np.maximum(tgt['H'], tgt.get('H2', 0)).sum() + 1.0])); r += 3
    if mode in ('equity', 'waterfill'):
        pos = np.where(tgt['D'] > 1e-9)[0]
        r_eq, t_at = r, sum(len(v) for v in V) + len(pos)      # first equity row; position of its t coefficients
        R += [r + np.arange(len(pos))] * 2; Cc += [oU + pos, np.full(len(pos), oT)]
        V += [np.ones(len(pos)), -tgt['D'][pos]]; b.append(np.zeros(len(pos))); r += len(pos)
    A_ub = sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(Cc))), shape=(r, nv))
    b_ub = np.concatenate(b).astype(float)

    def lp(c):
        return linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                       bounds=np.column_stack([np.zeros(nv), ub]), method='highs')

    def obj(*pairs):
        c = np.zeros(nv)
        c[oW:oW + nF] = 1e-7                                # tie-break: do not discard oil needlessly
        for lo, n, w in pairs:
            c[lo:lo + n] = w
        return c

    c_first = obj((oT, 1, 1.0)) if mode == 'equity' or (mode == 'waterfill' and gap_tol is None) else obj((oU, nC, 1.0))
    res = lp(c_first)
    if res.status == 2:
        ub[oV:oV + nC] = np.maximum(tgt['H'], tgt.get('H2', 0))
        ub[oZ] = fuel_min
        rz = lp(obj((oZ, 1, 1.0)))
        if rz.status != 0:
            return {'feasible': False, 'status': rz.status}
        ub[oZ] = rz.x[oZ] * (1 + 1e-6) + 1e-6
        r0 = lp(obj((oV, nC, 1.0)))
        if r0.status != 0:
            return {'feasible': False, 'status': r0.status}
        b_ub[r_cut] = r0.x[oV:oV + nC].sum() * (1 + 1e-6) + 1e-3
        res = lp(c_first)
    if res.status != 0:
        return {'feasible': False, 'status': res.status}
    t_eq = res.x[oT] if mode == 'equity' else np.nan
    if mode == 'equity':
        ub[oT] = t_eq * (1 + 1e-6) + 1e-9
        res = lp(obj((oU, nC, 1.0)))
    if mode == 'weighted':                                  # any efficient allocation: total held at its minimum, then
        b_ub[r_gap] = res.x[oU:oU + nC].sum() * (1 + (gap_tol or 1e-4)) + 1e-3     # weighted country shortfalls minimised
        res = lp(obj((oU, nC, tie_weights)))
    if mode == 'waterfill':
        if gap_tol is not None:
            b_ub[r_gap] = res.x[oU:oU + nC].sum() * (1 + gap_tol) + 1e-3
        Rv, Cv, Vv = np.concatenate(R), np.concatenate(Cc), np.concatenate(V)
        fixed, t_eq = np.zeros(len(pos), bool), 1.0
        for _ in range(100):
            rt = lp(obj((oT, 1, 1.0)))
            if rt.status != 0:
                break
            t_eq = rt.x[oT]
            block = ~fixed & (rt.ineqlin.marginals[r_eq:r_eq + len(pos)] < -1e-9)
            if t_eq < 1e-9 or not block.any():
                break
            fixed |= block                                  # these countries cannot fall below t_eq: freeze them
            lvl = t_eq * tgt['D'][pos[block]] * (1 + 1e-7) + 1e-9
            ub[oU + pos[block]] = lvl
            Vv[t_at + np.where(block)[0]] = 0.0              # the row becomes u_j <= frozen level
            b_ub[r_eq + np.where(block)[0]] = lvl
            A_ub = sp.csr_matrix((Vv, (Rv, Cv)), shape=(r, nv))
        ub[oT] = t_eq * (1 + 1e-6) + 1e-9
        rf = lp(obj((oU, nC, 1.0)))
        res = rf if rf.status == 0 else res
    if mode == 'max_fuel':
        b_ub[r_gap] = res.x[oU:oU + nC].sum() * (1 + 1e-6) + 1e-3
        res2 = lp(obj((oB, nF, -1.0)))
        res = res2 if res2.status == 0 else res
    x = res.x
    out = dict(feasible=True, fuel=x[oB:oB + nF].sum() * UNIT, gap=x[oU:oU + nC].sum() * UNIT,
               cut=x[oV:oV + nC].sum() * UNIT, fuel_unmet=x[oZ] * UNIT, u=x[oU:oU + nC] * UNIT, v=x[oV:oV + nC] * UNIT, t_eq=t_eq)
    if detail:
        out.update(B=x[oB:oB + nF].reshape(nC, nK) * UNIT, F=x[oF:oF + nF].reshape(nC, nK) * UNIT,
                   w=x[oW:oW + nF].reshape(nC, nK) * UNIT, x=x[:nL] * UNIT)
    return out


def frontier_point(S, tgt, L, imp_cap, eligible, m, conversion='footprint', detail=False):
    """Minimal shortfall when every country's biofuel volume is m x its scenario volume (national mandates).

    conversion='anywhere' keeps the world total m x E but lets the location of biofuel production move.
    """
    E = m * S.fuel_base.sum()
    cap = S.fuel_base * m if conversion == 'footprint' else None
    return solve(S, tgt, L, imp_cap, cap, eligible, fuel_min=E * (1 - 1e-7), detail=detail)
