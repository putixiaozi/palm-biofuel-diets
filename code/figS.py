"""Supplementary Figs 1-13 (same spec as the main figures). Usage: python figS.py [1 2 ...] (default all)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from figstyle import (RES, TGT, TGT_LAB, MM, INK, USE, OIL, OIL_ORDER, CLASS, LAYER, LAYER_LAB, LEVER, LEVER_LAB,
                      panel, save, table)

SUB = "SI"
TARGETS = ["adequacy", "eat_qty", "eat_quality"]
LAB = {"adequacy": "Fat availability", "eat_qty": "Total oil", "eat_quality": "Unsaturated oil"}
GREY, LIGHT = "#636363", "#BDBDBD"
H2C = "#8C6BB1"                                   # alternative quality target: palm cap in food
G2T = 365e-6


def legend(ax, handles, **kw):
    kw.setdefault("loc", "upper left")
    return ax.legend(handles=handles, **kw)


# ------------------------------------------------------------------ S1 data reconciliation
def s1():
    r = pd.read_csv(RES + "si_reconciliation.csv")
    pc = pd.read_csv(RES + "si_production_country_2023.csv")
    S = lambda s, src: r[(r.series == s) & r.source.str.startswith(src)].set_index("year").Mt.sort_index()
    fig = plt.figure(figsize=(183 * MM, 150 * MM))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.38, left=0.1, right=0.97, top=0.96, bottom=0.09)
    yrs = range(2010, 2024)
    # a production
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(S("production", "This"), color=INK, lw=2, label="FABIO (this study)")
    ax.plot(S("production", "USDA"), color=GREY, lw=1.6, ls="--", label="USDA PSD")
    ax.set_ylabel("Production (Mt)"); ax.set_ylim(0, 250)
    legend(ax, None, loc="lower right"); panel(ax, "a", x=-0.2)
    # b food use
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(S("food", "This"), color=USE["food"], lw=2, label="This study")
    ax.plot(S("food", "USDA"), color=GREY, lw=1.6, ls="--", label="USDA PSD")
    ax.plot(S("food", "FAOSTAT"), color=GREY, lw=1.6, ls=":", label="FAOSTAT FBS fat")
    ax.set_ylabel("Food use (Mt)"); ax.set_ylim(0, 180)
    legend(ax, None, loc="lower right"); panel(ax, "b", x=-0.2)
    # c biofuel and industrial use
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(S("industrial", "This"), color=INK, lw=2, label="Industrial, this study")
    ax.plot(S("industrial", "USDA"), color=GREY, lw=1.6, ls="--", label="Industrial, USDA PSD")
    ax.plot(S("biofuel", "This"), color=USE["biofuel"], lw=2, label="Biofuel, OECD-FAO")
    ax.set_ylabel("Non-food use (Mt)"); ax.set_ylim(0, 80)
    legend(ax, None, loc="upper left"); panel(ax, "c", x=-0.2)
    for a in fig.axes:
        a.set_xticks([2010, 2015, 2020]); a.set_xlim(2009.5, 2023.5)
    # d country production 2023
    ax = fig.add_subplot(gs[1, 1])
    lim = (0.05, 80)
    ax.plot(lim, lim, color=LIGHT, lw=1, zorder=0)
    pc = pc[(pc.psd_Mt >= 0.05) & (pc.fabio_Mt >= 0.05)]
    ax.scatter(pc.psd_Mt, pc.fabio_Mt, s=22, color=INK, alpha=0.7, lw=0)
    lab = {"IDN": (12, 60), "CHN": (55, 9), "MYS": (2.6, 32), "EU": (2.0, 16), "ARG": (25, 2.6)}
    for iso, xy in lab.items():
        q = pc[pc.iso3c == iso].iloc[0]
        ax.annotate(iso, xy=(q.psd_Mt, q.fabio_Mt), xytext=xy, fontsize=12, ha="center", va="center", color=GREY,
                    arrowprops=dict(arrowstyle="-", color=LIGHT, lw=0.8, shrinkA=1, shrinkB=3))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("USDA PSD production, 2023 (Mt)"); ax.set_ylabel("FABIO production,\n2023 (Mt)")
    panel(ax, "d", x=-0.2)
    rr = r.pivot_table(index="year", columns=["series", "source"], values="Mt")
    rr.columns = [f"{a} | {b}" for a, b in rr.columns]
    table(rr.reset_index().merge(pc.assign(year=2023), how="outer", on="year"), "FigS1", SUB)
    save(fig, "FigS1", SUB)


# ------------------------------------------------------------------ S2 projection inputs
def s2():
    p = pd.read_csv(RES + "si_projection.csv")
    fig = plt.figure(figsize=(183 * MM, 165 * MM))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.95, left=0.2, right=0.97, top=0.9, bottom=0.08)

    def paired(ax, d, keys, colors, xlabel):
        for i, k in enumerate(keys):
            for y, off, al in [(2035, -0.18, 1.0), (2023, 0.18, 0.45)]:
                v = d[(d.key == k) & (d.year == y)].Mt.sum()
                ax.barh(i + off, v, height=0.34, color=colors[k] if isinstance(colors, dict) else colors, alpha=al, lw=0)
        ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys); ax.set_ylim(len(keys) - 0.5, -0.5)
        ax.set_xlabel(xlabel); ax.tick_params(axis="y", length=0)

    # a supply by oil group
    ax = fig.add_subplot(gs[0, 0])
    paired(ax, p[p.panel == "supply"], OIL_ORDER, OIL, "Supply (Mt)")
    ax.set_yticklabels(["Soybean", "Rapeseed", "Sunflower", "Other unsat.", "Palm", "Palm kernel\n& coconut"])
    ax.legend(handles=[Patch(color=GREY, label="2035"), Patch(color=GREY, alpha=0.45, label="2023")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    panel(ax, "a", x=-0.5)
    # b biofuel by region
    ax = fig.add_subplot(gs[0, 1])
    b = p[p.panel == "biofuel"]
    keys = b[b.year == 2035].sort_values("Mt", ascending=False).key.tolist()
    keys = [k for k in keys if b[b.key == k].Mt.max() > 0.05]
    paired(ax, b, keys, USE["biofuel"], "Biofuel feedstock (Mt)")
    panel(ax, "b", x=-0.5)
    # c, d unsaturated-oil target by region, 2035
    t = p[p.panel == "targets"].copy()
    t[["region", "what"]] = t.key.str.split(" \\| ", expand=True)
    w = t.pivot_table(index=["year", "region"], columns="what", values="Mt").reset_index()
    w["gap"] = w["unsaturated target"] - w["unsaturated food, observed or projected"]
    w["gpd"] = w.gap / (w.population * 1e6 * G2T) * 1e6
    w35 = w[w.year == 2035].sort_values("unsaturated target", ascending=False)
    regs = w35.region.tolist()
    ax = fig.add_subplot(gs[1, 0])
    for i, rg in enumerate(regs):
        q = w35[w35.region == rg].iloc[0]
        ax.barh(i, q["unsaturated target"], height=0.62, color=TGT["eat_quality"], alpha=0.35, lw=0)
        ax.barh(i, q["unsaturated food, observed or projected"], height=0.62, color=TGT["eat_quality"], lw=0)
    ax.set_yticks(range(len(regs))); ax.set_yticklabels(regs); ax.set_ylim(len(regs) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0); ax.set_xlabel("Unsaturated oil, 2035 (Mt)")
    ax.legend(handles=[Patch(color=TGT["eat_quality"], label="Projected food use"),
                       Patch(color=TGT["eat_quality"], alpha=0.35, label="Target")], loc="lower right")
    panel(ax, "c", x=-0.5)
    ax = fig.add_subplot(gs[1, 1])
    for i, rg in enumerate(regs):
        for y, off, al in [(2035, -0.18, 1.0), (2023, 0.18, 0.45)]:
            v = w[(w.year == y) & (w.region == rg)].gpd.iloc[0]
            ax.barh(i + off, v, height=0.34, color=TGT["eat_quality"], alpha=al, lw=0)
    ax.set_yticks(range(len(regs))); ax.set_yticklabels(regs); ax.set_ylim(len(regs) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0); ax.set_xlabel("Gap to target\n(g per person per day)")
    panel(ax, "d", x=-0.5)
    table(pd.concat([p[p.panel != "targets"], w.melt(id_vars=["year", "region"], var_name="key", value_name="value")
                     .assign(panel="targets")]), "FigS2", SUB)
    save(fig, "FigS2", SUB)


# ------------------------------------------------------------------ S3 frontier details
def s3():
    f = pd.read_csv(RES + "fig4_frontier.csv")
    f[["E", "gap"]] = f[["E", "gap"]]
    f["attrib"] = f.gap - f.groupby(["year", "target", "access"]).gap.transform(lambda s: s.iloc[0])
    cv = pd.read_csv(RES + "compatible_volume.csv"); cv[["E_c", "E1"]] /= 1e6
    fig = plt.figure(figsize=(183 * MM, 160 * MM))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.42, left=0.1, right=0.97, top=0.96, bottom=0.09)
    hand = [Line2D([], [], color=TGT[t], lw=2, label=LAB[t]) for t in TARGETS] + \
           [Line2D([], [], color=GREY, lw=1.6, label="Constrained"), Line2D([], [], color=GREY, lw=1.6, ls="--", label="Unrestricted")]
    for k, (yr, col, lab, xl) in enumerate([(2023, "attrib", "Additional reference gap\nwith biofuel (Mt)", 85),
                                            (2035, "gap", "Total gap (Mt)", 128)]):
        ax = fig.add_subplot(gs[0, k])
        d = f[f.year == yr]
        e1 = d[d.m == 1.0].E.iloc[0]
        ax.axvline(e1, color=LIGHT, lw=0.8, ls=":", zorder=0)
        for t in TARGETS:
            for a, ls in [("constrained", "-"), ("pool", "--")]:
                g = d[(d.target == t) & (d.access == a)]
                ax.plot(g.E, g[col], ls=ls, color=TGT[t], lw=2 if t == "eat_quality" else 1.5)
        ax.set_xlim(0, xl); ax.set_ylim(0, None)
        ax.set_xlabel(f"Requested feedstock, {yr} (Mt)"); ax.set_ylabel(lab)
        if k == 0:
            legend(ax, hand, loc="upper left")
        panel(ax, "ab"[k], x=-0.2)
    # c E_c against tolerance
    ax = fig.add_subplot(gs[1, 0])
    for t in TARGETS:
        for y, mk in [(2035, "o"), (2023, "s")]:
            q = cv[(cv.year == y) & (cv.target == t) & (cv.access == "constrained")].sort_values("tol")
            ax.plot(q.tol / 1e6, q.E_c, marker=mk, color=TGT[t], lw=1.4, ms=6, mfc=TGT[t] if y == 2035 else "white", mew=1.3)
    ax.set_xticks([0.1, 0.5, 1.0]); ax.set_xlim(0, 1.1); ax.set_ylim(0, 70)
    ax.set_xlabel("Tolerance (Mt)"); ax.set_ylabel("Compatible volume,\nconstrained (Mt)")
    legend(ax, [Line2D([], [], color=GREY, marker="o", lw=0, label="2035"),
                Line2D([], [], color=GREY, marker="s", mfc="white", lw=0, label="2023")], loc="lower right",
           bbox_to_anchor=(1.0, 0.1))
    panel(ax, "c", x=-0.2)
    # d marginal gap per extra Mt above E_c
    ax = fig.add_subplot(gs[1, 1])
    rows = [(y, a) for y in [2035, 2023] for a in ["constrained", "pool"]]
    for i, (y, a) in enumerate(rows):
        for j, t in enumerate(TARGETS):
            q = cv[(cv.year == y) & (cv.target == t) & (cv.access == a) & (cv.tol == 1e5)].slope_above.iloc[0]
            ax.barh(i + (j - 1) * 0.26, q, height=0.24, color=TGT[t], lw=0, alpha=1 if a == "constrained" else 0.55)
    ax.set_yticks(range(4)); ax.set_yticklabels([f"{y}, {'constr.' if a == 'constrained' else 'unrestr.'}" for y, a in rows])
    ax.set_ylim(3.6, -0.6); ax.set_xlim(0, 1.05); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Added gap per added Mt\nabove the compatible volume")
    panel(ax, "d", x=-0.48)
    table(f.merge(cv, how="outer", on=["year", "target", "access"]), "FigS3", SUB)
    save(fig, "FigS3", SUB)


# ------------------------------------------------------------------ S4 constraint layers, total gap
def s4():
    sh = pd.read_csv(RES + "fig3_layer_shapley.csv")
    L = ["L1", "L2", "L3", "L4", "L5"]
    fig = plt.figure(figsize=(183 * MM, 150 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.75, 1], hspace=0.5, wspace=0.15, left=0.3, right=0.97, top=0.85, bottom=0.17)
    ax = fig.add_subplot(gs[0, :])
    for i, t in enumerate(TARGETS):
        for y, off, al in [(2035, -0.19, 1.0), (2023, 0.19, 0.45)]:
            q = sh[(sh.year == y) & (sh.target == t)].iloc[0]; x0 = 0
            for l in L:
                v = max(q[l], 0)
                ax.barh(i + off, v, left=x0, height=0.34, color=LAYER[l], alpha=al, lw=0); x0 += v
            ax.text(x0 + 0.3, i + off, f"{q.total:.1f}", va="center", fontsize=12, color=GREY)
    ax.set_yticks(range(3)); ax.set_yticklabels([LAB[t] for t in TARGETS]); ax.set_ylim(2.55, -0.55)
    ax.tick_params(axis="y", length=0); ax.set_xlabel("Total gap at actual or projected biofuel use (Mt)")
    ax.set_xlim(0, 30)
    ax.legend(handles=[Patch(color=LAYER[l], label=LAYER_LAB[l]) for l in L] +
              [Patch(color=GREY, label="2035"), Patch(color=GREY, alpha=0.45, label="2023")], loc="lower left",
              bbox_to_anchor=(-0.02, 1.0), ncol=3, columnspacing=0.8, handlelength=1.0)
    panel(ax, "a", x=-0.2, y=1.25)
    rows = ["none", "L1", "L2", "L3", "L4", "L5", "L3_L4", "L2_L3_L4"]
    rlab = ["All constraints", "Relax production", "Relax trade access", "Relax conversion location",
            "Relax feedstock eligibility", "Relax rigid demand", "Relax location + eligibility", "Unrestricted allocation"]
    for k, y in enumerate([2035, 2023]):
        ax = fig.add_subplot(gs[1, k])
        for j, t in enumerate(TARGETS):
            q = sh[(sh.year == y) & (sh.target == t)].iloc[0]
            vals = [q.total] + [q[f"single_{l}"] for l in L] + [q.L3_L4, q.L2_L3_L4]
            ax.plot(vals, np.arange(len(rows)) + (j - 1) * 0.2, "o", color=TGT[t], ms=6,
                    mfc=TGT[t] if y == 2035 else "white", mew=1.3, label=LAB[t])
        ax.set_yticks(range(len(rows))); ax.set_ylim(len(rows) - 0.5, -0.5); ax.tick_params(axis="y", length=0)
        ax.set_yticklabels(rlab if k == 0 else [""] * len(rows)); ax.set_xlim(-1, 30)
        ax.set_xlabel(f"Gap, {y} (Mt)")
        for i in range(len(rows)):
            ax.axhline(i, color="#F0F0F0", lw=0.6, zorder=0)
        if k == 1:
            ax.legend(handles=[Line2D([], [], color=TGT[t], marker="o", lw=0, ms=6, label=LAB[t]) for t in TARGETS],
                      loc="upper center", bbox_to_anchor=(-0.08, -0.25), ncol=3, handletextpad=0.1, columnspacing=1.0)
        panel(ax, "bc"[k], x=-0.9 if k == 0 else -0.04)
    table(sh, "FigS4", SUB)
    save(fig, "FigS4", SUB)


# ------------------------------------------------------------------ S5 history 2010-2023
def s5():
    h = pd.read_csv(RES + "history_fuel_cost.csv")
    fig = plt.figure(figsize=(183 * MM, 150 * MM))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.4, left=0.1, right=0.97, top=0.86, bottom=0.08)
    for k, t in enumerate(TARGETS):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        q = h[h.target == t].sort_values("year")
        ax.fill_between(q.year, q.S0, q.S1, color=USE["biofuel"], alpha=0.18, lw=0)
        ax.plot(q.year, q.S0, color=GREY, lw=1.6, ls="--")
        ax.plot(q.year, q.S1, color=TGT[t], lw=2.2)
        ax.plot(q.year, q.observed, color=INK, lw=1.2, ls=":")
        ax.set_ylim(0, None); ax.set_xticks([2010, 2015, 2020]); ax.set_xlim(2009.5, 2023.5)
        ax.set_ylabel(f"{LAB[t]}\ngap (Mt)")
        panel(ax, "abc"[k], x=-0.2)
    fig.legend(handles=[Line2D([], [], color=GREY, lw=1.6, ls="--", label="Optimum without biofuel"),
                        Line2D([], [], color=GREY, lw=2.2, label="Optimum with reconstructed biofuel"),
                        Line2D([], [], color=INK, lw=1.2, ls=":", label="Baseline food use"),
                        Patch(color=USE["biofuel"], alpha=0.18, label="Additional gap")],
               loc="lower center", bbox_to_anchor=(0.53, 0.9), ncol=2, columnspacing=1.2)
    ax = fig.add_subplot(gs[1, 1])
    for t in TARGETS:
        q = h[h.target == t].sort_values("year")
        ax.plot(q.E, q.biofuel_cost, "-o", color=TGT[t], ms=4, lw=1.2, label=LAB[t])
    ax.plot([0, 42], [0, 42], color=LIGHT, lw=0.9, zorder=0)
    ax.text(30, 33, "1:1", color=GREY, fontsize=12)
    ax.set_xlim(14, 42); ax.set_ylim(0, 42)
    ax.set_xlabel("Biofuel feedstock use (Mt)"); ax.set_ylabel("Additional reference gap\nwith biofuel (Mt)")
    ax.legend(loc="upper left", handletextpad=0.3)
    panel(ax, "d", x=-0.2)
    table(h, "FigS5", SUB)
    save(fig, "FigS5", SUB)


# ------------------------------------------------------------------ S6 H2 robustness
def s6():
    hf = pd.read_csv(RES + "si_pcap_frontier.csv"); hf["target"] = "h2"
    f = pd.read_csv(RES + "fig4_frontier.csv"); f = f[f.target == "eat_quality"].assign(target="main")
    fr = pd.concat([f, hf])
    fr["attrib"] = fr.gap - fr.groupby(["year", "target", "access"]).gap.transform(lambda s: s.iloc[0])
    hl = pd.read_csv(RES + "si_pcap_layers.csv")
    ml = pd.read_csv(RES + "fig3_layer_shapley.csv").query("target == 'eat_quality'")
    hh = pd.read_csv(RES + "si_pcap_history.csv")
    mh = pd.read_csv(RES + "history_fuel_cost.csv").query("target == 'eat_quality'")
    hs = pd.read_csv(RES + "si_pcap_strategy.csv")
    ms = pd.read_csv(RES + "fig5_strategy_shapley.csv").query("target == 'eat_quality' and m == 1.0")
    fig = plt.figure(figsize=(183 * MM, 160 * MM))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.55, left=0.1, right=0.97, top=0.95, bottom=0.09)
    # a frontier 2035
    ax = fig.add_subplot(gs[0, 0])
    for tg, c in [("main", TGT["eat_quality"]), ("h2", H2C)]:
        for a, ls in [("constrained", "-"), ("pool", "--")]:
            g = fr[(fr.year == 2035) & (fr.target == tg) & (fr.access == a)]
            ax.plot(g.E / 1e6 if g.E.max() > 1e3 else g.E, g.attrib / 1e6 if g.attrib.max() > 1e3 else g.attrib,
                    ls=ls, color=c, lw=2 if tg == "main" else 1.6)
    ax.set_xlim(0, 128); ax.set_ylim(0, None)
    ax.set_xlabel("Requested feedstock, 2035 (Mt)"); ax.set_ylabel("Additional reference gap\nwith biofuel (Mt)")
    legend(ax, [Line2D([], [], color=TGT["eat_quality"], lw=2, label="Central specification"),
                Line2D([], [], color=H2C, lw=1.6, label="Palm cap in food"),
                Line2D([], [], color=GREY, lw=1.4, label="Constrained"),
                Line2D([], [], color=GREY, lw=1.4, ls="--", label="Unrestricted")])
    panel(ax, "a", x=-0.2)
    # b layer Shapley 2035 and 2023
    ax = fig.add_subplot(gs[0, 1])
    L = ["L1", "L2", "L3", "L4", "L5"]
    for i, l in enumerate(L):
        for y, off in [(2035, -0.17), (2023, 0.17)]:
            m_ = ml[ml.year == y][l].iloc[0]; h_ = hl[hl.year == y][l].iloc[0]
            ax.plot([m_, h_], [i + off] * 2, color=LIGHT, lw=1, zorder=0)
            ax.plot(m_, i + off, "o", color=LAYER[l], ms=7, mfc=LAYER[l] if y == 2035 else "white", mew=1.4)
            ax.plot(h_, i + off, "D", color=H2C, ms=5.5, mfc=H2C if y == 2035 else "white", mew=1.2)
    ax.set_yticks(range(5)); ax.set_yticklabels([LAYER_LAB[l] for l in L]); ax.set_ylim(4.6, -0.6)
    for t, l in zip(ax.get_yticklabels(), L):
        t.set_color(LAYER[l])
    ax.tick_params(axis="y", length=0); ax.set_xlabel("Layer contribution to\ntotal gap (Mt)"); ax.set_xlim(0, 21)
    legend(ax, [Line2D([], [], color=GREY, marker="o", lw=0, label="Central"),
                Line2D([], [], color=H2C, marker="D", lw=0, ms=5.5, label="Palm cap"),
                Line2D([], [], color=GREY, marker="o", lw=0, label="2035"),
                Line2D([], [], color=GREY, marker="o", mfc="white", mew=1.3, lw=0, label="2023")],
           loc="center right", bbox_to_anchor=(1.02, 0.36), ncol=1, handletextpad=0.1)
    panel(ax, "b", x=-0.75)
    # c history
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(mh.year, mh.biofuel_cost, color=TGT["eat_quality"], lw=2, label="Central specification")
    ax.plot(hh.year, hh.biofuel_cost, color=H2C, lw=1.6, ls="-.", label="Palm cap in food")
    ax.set_xticks([2010, 2015, 2020]); ax.set_xlim(2009.5, 2023.5); ax.set_ylim(0, 18)
    ax.set_ylabel("Additional reference gap\nwith biofuel (Mt)"); ax.legend(loc="upper left")
    panel(ax, "c", x=-0.2)
    # d strategy Shapley
    ax = fig.add_subplot(gs[1, 1])
    S = ["S1", "S2", "S3", "S4"]
    for i, s in enumerate(S):
        ax.barh(i - 0.19, ms[s].iloc[0], height=0.34, color=LEVER[s], lw=0)
        ax.barh(i + 0.19, hs[s].iloc[0], height=0.34, color=LEVER[s], lw=0, alpha=0.45, hatch="////", edgecolor="white")
    ax.set_yticks(range(4)); ax.set_yticklabels([LEVER_LAB[s] for s in S]); ax.set_ylim(3.6, -0.6)
    ax.tick_params(axis="y", length=0); ax.set_xlabel("Lever contribution to\ngap reduction (Mt)"); ax.set_xlim(0, 9.5)
    legend(ax, [Patch(color=GREY, label="Central"), Patch(facecolor=GREY, alpha=0.45, hatch="////",
                                                            edgecolor="white", label="Palm cap")], loc="lower right")
    panel(ax, "d", x=-0.75)
    table(pd.concat([fr.assign(panel="a"), hl.assign(panel="b", target="h2"), ml.assign(panel="b", target="main"),
                     hh.assign(panel="c", target="h2"), mh.assign(panel="c", target="main"),
                     hs.assign(panel="d"), ms.assign(panel="d")]), "FigS6", SUB)
    save(fig, "FigS6", SUB)


# ------------------------------------------------------------------ S7 parameter sensitivity
PAR = {"main": "Central specification", "delta=0.25": "Import capacity +25%", "delta=0.5": "Import capacity +50%",
       "delta=1.0": "Import capacity +100%", "links=complete": "All producer–importer links",
       "floor=oecd": "Food floors from OECD-FAO", "waste=0.1": "Food waste 10%", "uco=0.025": "UCO 2.5% of food use",
       "uco=0.075": "UCO 7.5% of food use", "fat=0.15": "Fat energy share 15%", "fat=0.25": "Fat energy share 25%"}


def s7():
    s = pd.read_csv(RES + "si_sensitivity_attrib.csv")
    sh = pd.read_csv(RES + "fig3_layer_shapley.csv").query("year == 2035")
    a = pd.read_csv(RES + "fig3_layer_shapley_attrib.csv").query("year == 2035") if os.path.exists(
        RES + "fig3_layer_shapley_attrib.csv") else None
    main = {t: (sh[sh.target == t].total.iloc[0]) for t in TARGETS}
    cvf = pd.read_csv(RES + "fig4_frontier.csv").query("year == 2035 and access == 'constrained'")
    main_att = {t: cvf[(cvf.target == t) & (cvf.m == 1.0)].gap.iloc[0] - cvf[(cvf.target == t) & (cvf.m == 0.0)].gap.iloc[0]
                for t in TARGETS}
    rows = list(PAR)
    fig = plt.figure(figsize=(183 * MM, 120 * MM))
    gs = fig.add_gridspec(1, 2, wspace=0.12, left=0.3, right=0.97, top=0.95, bottom=0.13)
    for k, (col, xl, ref) in enumerate([("attrib", "Additional gap (Mt)", main_att),
                                        ("gap_m1", "Total gap (Mt)", main)]):
        ax = fig.add_subplot(gs[0, k])
        for j, t in enumerate(TARGETS):
            q = s[s.target == t].set_index("param")[col]
            q["main"] = ref[t]
            ys = [i + (j - 1) * 0.22 for i, r in enumerate(rows) if r in q.index]
            ax.plot([q[r] for r in rows if r in q.index], ys, "o", color=TGT[t], ms=6, label=LAB[t])
            ax.axvline(ref[t], color=TGT[t], lw=0.8, ls=":", zorder=0)
        ax.set_yticks(range(len(rows))); ax.set_ylim(len(rows) - 0.5, -0.5); ax.tick_params(axis="y", length=0)
        ax.set_yticklabels([PAR[r] for r in rows] if k == 0 else [""] * len(rows))
        for i in range(len(rows)):
            ax.axhline(i, color="#F0F0F0", lw=0.6, zorder=0)
        ax.set_xlabel(xl); ax.set_xlim(-0.5, 37)
        if k == 0:
            ax.legend(loc="lower right", handletextpad=0.2)
        panel(ax, "ab"[k], x=-0.03 if k else -0.03 - 0.0, y=1.0)
    fig.axes[0].texts[0].set_position((-0.82, 1.0))
    table(s.assign(main_attrib=s.target.map(main_att), main_total=s.target.map(main)), "FigS7", SUB)
    save(fig, "FigS7", SUB)


# ------------------------------------------------------------------ S8 levers at higher biofuel volumes
def s8():
    st = pd.read_csv(RES + "fig5_strategy_shapley.csv")
    r = pd.read_csv(RES + "runs_summary.csv").query("block == 'strategy'")
    S = ["S1", "S2", "S3", "S4"]
    fig = plt.figure(figsize=(183 * MM, 150 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.7, wspace=0.12, left=0.27, right=0.97, top=0.86, bottom=0.17)
    ax = fig.add_subplot(gs[0, :])
    rows = [(t, m) for t in TARGETS for m in [1.0, 1.5, 2.0]]
    ypos = [i + (i // 3) * 0.6 for i in range(len(rows))]
    for (t, m), y in zip(rows, ypos):
        q = st[(st.target == t) & (st.m == m)].iloc[0]; pos, neg = 0, 0
        for s in S:
            v = q[s]
            if v >= 0:
                ax.barh(y, v, left=pos, height=0.7, color=LEVER[s], lw=0); pos += v
            else:
                ax.barh(y, v, left=neg, height=0.7, color=LEVER[s], lw=0, hatch="////", edgecolor="white"); neg += v
        ax.text(pos + 0.5, y, f"{q.none:.1f} → {q.combined:.1f}", va="center", fontsize=12, color=GREY)
    ax.set_yticks(ypos); ax.set_yticklabels([f"{LAB[t]}, { {1.0: '61.7', 1.5: '92.5', 2.0: '123.4'}[m]} Mt" for t, m in rows])
    ax.set_yticklabels([f"{LAB[t]}, { {1.0: '61.7', 1.5: '92.5', 2.0: '123.4'}[m] } Mt" for t, m in rows])
    ax.set_ylim(max(ypos) + 0.6, -0.6); ax.tick_params(axis="y", length=0); ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("Lever contribution to gap reduction, 2035 (Mt)"); ax.set_xlim(-3, 52)
    ax.legend(handles=[Patch(color=LEVER[s], label=LEVER_LAB[s]) for s in S], loc="lower left",
              bbox_to_anchor=(-0.02, 1.0), ncol=2, columnspacing=1.0, handlelength=1.0)
    panel(ax, "a", x=-0.36, y=1.15)
    # b gap at E3 under single levers; c biofuel demand left unmet (identical across targets)
    q = r[(r.m == 2.0) & r.strategies.isin(["none"] + S)].copy()
    order = ["none"] + S
    ax = fig.add_subplot(gs[1, 0])
    for j, t in enumerate(TARGETS):
        qq = q[q.target == t].set_index("strategies").gap.reindex(order) / 1e6
        ax.barh(np.arange(5) + (j - 1) * 0.26, qq.values, height=0.24, color=TGT[t], lw=0, label=LAB[t])
    ax.set_yticks(range(5)); ax.set_ylim(4.6, -0.6); ax.tick_params(axis="y", length=0)
    ax.set_yticklabels(["No lever"] + [LEVER_LAB[s] for s in S]); ax.set_xlabel("Gap at 123.4 Mt (Mt)")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.62, -0.32), ncol=3, columnspacing=1.0, handlelength=1.0)
    panel(ax, "b", x=-0.62)
    ax = fig.add_subplot(gs[1, 1])
    fu = q[q.target == "eat_quality"].set_index("strategies").fuel_unmet.reindex(order) / 1e6
    ax.barh(np.arange(5), fu.values, height=0.5, color=USE["biofuel"], lw=0)
    for i, v in enumerate(fu.values):
        ax.text(v + 0.08, i, f"{v:.2f}", va="center", fontsize=12, color=GREY)
    ax.set_yticks(range(5)); ax.set_yticklabels([""] * 5); ax.set_ylim(4.6, -0.6); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 4.8); ax.set_xlabel("Biofuel demand unmet\nat 123.4 Mt (Mt)")
    panel(ax, "c", x=-0.06)
    table(pd.concat([st.assign(panel="a"), q[["target", "m", "strategies", "gap", "cut", "fuel_unmet"]].assign(panel="b,c")]),
          "FigS8", SUB)
    save(fig, "FigS8", SUB)


# ------------------------------------------------------------------ S9 countries: layer attribution
def s9():
    c = pd.read_csv(RES + "fig6_country.csv")
    L = ["L1", "L2", "L3", "L4", "L5"]
    fig = plt.figure(figsize=(183 * MM, 175 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.42, wspace=0.42, left=0.1, right=0.97, top=0.87, bottom=0.08)
    spec = [("eat_quality", 20, gs[0, :], "a"), ("eat_qty", 10, gs[1, 0], "b"), ("adequacy", 10, gs[1, 1], "c")]
    keep = []
    for t, n, g, let in spec:
        ax = fig.add_subplot(g)
        q = c[c.target == t].sort_values("u1", ascending=False).head(n)
        keep.append(q)
        x = np.arange(len(q))
        pos, neg = np.zeros(len(q)), np.zeros(len(q))
        for l in L:
            v = q[f"sh_L{l[1]}"].values / 1e6
            ax.bar(x, np.clip(v, 0, None), bottom=pos, width=0.72, color=LAYER[l], lw=0); pos += np.clip(v, 0, None)
            ax.bar(x, np.clip(v, None, 0), bottom=neg, width=0.72, color=LAYER[l], lw=0); neg += np.clip(v, None, 0)
        ax.plot(x, q.u1 / 1e6, "_", color=INK, ms=11, mew=1.6)
        ax.set_xticks(x); ax.set_xticklabels(q.iso3c, rotation=90 if n > 12 else 0, fontsize=12)
        ax.axhline(0, color=INK, lw=0.6); ax.set_xlim(-0.6, len(q) - 0.4)
        ax.set_ylabel(f"{LAB[t]}\ngap (Mt)")
        if let == "a":
            ax.legend(handles=[Patch(color=LAYER[l], label=LAYER_LAB[l]) for l in L] +
                      [Line2D([], [], color=INK, marker="_", lw=0, ms=11, mew=1.6, label="Gap")],
                      loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3, columnspacing=0.9, handlelength=1.0,
                      handletextpad=0.4)
        if t != "eat_quality":
            ax.tick_params(axis="x", labelrotation=90)
        panel(ax, let, x=-0.07 if let == "a" else -0.2, y=1.0 if let != "a" else 1.2)
    table(pd.concat(keep), "FigS9", SUB)
    save(fig, "FigS9", SUB)


# ------------------------------------------------------------------ S10 fairness
def s10():
    fc = pd.read_csv(RES + "fair_check.csv")
    c = pd.read_csv(RES + "fig6_country.csv")
    fig = plt.figure(figsize=(183 * MM, 80 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.5, left=0.09, right=0.97, top=0.95, bottom=0.2)
    tols = sorted(fc.tol.unique(), key=lambda v: (v < 0, v))
    for k, (col, yl, sc) in enumerate([("gap", "Total gap (Mt)", 1), ("max_rel", "Largest relative\ngap (%)", 100)]):
        ax = fig.add_subplot(gs[0, k])
        for t in TARGETS:
            q = fc[fc.target == t].set_index("tol")[col].reindex(tols)
            ax.plot(range(len(tols)), q.values * sc, "-o", color=TGT[t], ms=4.5, lw=1.4, label=LAB[t])
        ax.set_xticks(range(len(tols))); ax.set_xticklabels(["F" if v < 0 else (f"{v * 100:g}" if v in (0, 0.05, 0.2, 0.5) else "") for v in tols])
        ax.set_xlabel("Efficiency tolerance (%)"); ax.set_ylabel(yl); ax.set_ylim(0, 32 if k == 0 else 105)
        if k == 0:
            ax.legend(loc="center", bbox_to_anchor=(0.5, 0.6), handlelength=1.2)
        panel(ax, "ab"[k], x=-0.26)
    ax = fig.add_subplot(gs[0, 2])
    out = []
    for t in TARGETS:
        q = c[(c.target == t) & (c.u1 > 1e3)].sort_values("rel1", ascending=False)
        cum = np.r_[0, q["pop"].cumsum().values] / 1e9
        rel = np.r_[q.rel1.values[:1], q.rel1.values] * 100
        ax.step(cum, rel, where="pre", color=TGT[t], lw=2 if t == "eat_quality" else 1.5)
        ax.plot([cum[-1]] * 2, [rel[-1], 0], color=TGT[t], lw=2 if t == "eat_quality" else 1.5)
        out.append(q.assign(cum_pop_bn=cum[1:]))
    ax.set_xlabel("Population (bn)"); ax.set_ylabel("Relative gap (%)")
    ax.set_xlim(0, 5.5); ax.set_ylim(0, 105)
    panel(ax, "c", x=-0.24)
    table(pd.concat([fc.assign(panel="a,b"), pd.concat(out)[["target", "iso3c", "pop", "rel1", "cum_pop_bn"]].assign(panel="c")]),
          "FigS10", SUB)
    save(fig, "FigS10", SUB)


# ------------------------------------------------------------------ S11 reference and implementation sensitivities
def s11():
    r = pd.read_csv(RES + "reference_target_diagnostic.csv")
    s = pd.read_csv(RES + "revision_scenarios.csv")
    b = s[(s.scenario == "baseline") & (s.m == 1.0)].gap_Mt.iloc[0]
    fig = plt.figure(figsize=(183 * MM, 122 * MM))
    gs = fig.add_gridspec(1, 3, wspace=0.68, left=0.095, right=0.975, top=0.96, bottom=0.2)
    # a dietary reference sensitivity
    ax = fig.add_subplot(gs[0, 0])
    for y, mk, mfc in [(2023, "s", "white"), (2035, "o", TGT["eat_quality"])]:
        q = r[r.year == y]
        ax.plot(q.reference_g_per_day, q.attributable_Mt, color=TGT["eat_quality"], marker=mk,
                mfc=mfc, mew=1.4, lw=1.7, ms=6.5, label=str(y))
    ax.set_xticks([20, 30, 40, 50]); ax.set_xlim(17, 53); ax.set_ylim(-0.8, 25)
    ax.set_xlabel("Unsaturated-oil reference\n(g per person per day)")
    ax.set_ylabel("Additional reference gap\nwith biofuel (Mt)")
    ax.legend(loc="upper left")
    panel(ax, "a", x=-0.24)
    # b partial palm eligibility
    ax = fig.add_subplot(gs[0, 1])
    q = s[(s.scenario.str.startswith("palm_share_")) & (s.m == 1.0)].copy()
    q["share"] = q.scenario.str.extract(r"([0-9.]+)$").astype(float) * 100
    q = q.sort_values("share")
    ax.plot(np.r_[0, q.share], np.r_[b, q.gap_Mt], "-o", color=OIL["Palm"], lw=2, ms=6)
    ax.axhline(b, color=GREY, lw=0.9, ls=":")
    ax.set_xticks([0, 25, 50, 100]); ax.set_xlim(-3, 104); ax.set_ylim(13.5, 28)
    ax.set_xlabel("Palm share of US and EU\nnational requests (%)")
    ax.set_ylabel("Total unsaturated-oil gap\nat 61.7 Mt (Mt)")
    ax.text(2, b + 0.45, "Strict exclusion", fontsize=12, color=GREY, va="bottom")
    panel(ax, "b", x=-0.24)
    # c domestic UCO
    ax = fig.add_subplot(gs[0, 2])
    q = s[(s.scenario.str.startswith("uco_domestic_")) & (s.m == 1.0)].copy()
    q["pct"] = q.scenario.str.extract(r"([0-9.]+)$").astype(float) * 100
    q = q.sort_values("pct")
    ax.plot(np.r_[0, q.pct], np.r_[b, q.gap_Mt], "-o", color=OIL["UCO"], lw=2, ms=6)
    ax.axhline(b, color=GREY, lw=0.9, ls=":")
    ax.set_xticks([0, 2.5, 5, 7.5]); ax.set_xlim(-0.4, 8.2); ax.set_ylim(22.5, 28)
    ax.set_xticklabels(["0", "2.5", "5", "7.5"], fontsize=11)
    ax.set_xlabel("Domestic recovery\n(% of food use)", fontsize=12)
    ax.set_ylabel("Total unsaturated-oil gap\nat 61.7 Mt (Mt)")
    panel(ax, "c", x=-0.24)
    table(pd.concat([r.assign(panel="a"), s[(s.scenario.str.startswith(("palm_share_", "uco_domestic_"))) |
                                          (s.scenario == "baseline")].assign(panel="b,c")]), "FigS11", SUB)
    save(fig, "FigS11", SUB)


# ------------------------------------------------------------------ S12 external feedstock checks
def s12():
    x = pd.read_csv(RES + "external_feedstock_benchmarks.csv")
    m = pd.read_csv(RES + "historical_EU_mix_sensitivity.csv")
    fig = plt.figure(figsize=(183 * MM, 142 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.74, wspace=0.52,
                          left=0.105, right=0.97, top=0.90, bottom=0.12)
    # a external validation panels
    for k, (reg, oils, title) in enumerate([
        ("EU", ["palm", "rapeseed", "soy", "sunflower"], "EU"),
        ("USA", ["maize", "rapeseed", "soy"], "USA")]):
        ax = fig.add_subplot(gs[0, k])
        q = x[(x.region == reg) & x.oil.isin(oils)]
        cols = {"palm": OIL["Palm"], "rapeseed": OIL["Rapeseed"], "soy": OIL["Soybean"],
                "sunflower": OIL["Sunflower"], "maize": "#FDB863"}
        for i, oil in enumerate(oils):
            z = q[q.oil == oil].dropna(subset=["external_Mt"])
            z = z[z.year == z.year.max()]
            if z.empty:
                continue
            ax.plot(z.external_Mt, z.reconstructed_Mt, "o", color=cols[oil], ms=8, mew=0, label=oil.capitalize())
        top = max(q.external_Mt.max(), q.reconstructed_Mt.max()) * 1.18
        ax.plot([0, top], [0, top], color=LIGHT, lw=0.9, zorder=0)
        ax.set_xlim(0, top); ax.set_ylim(0, top)
        ax.set_xlabel("External estimate (Mt)"); ax.set_ylabel("Reconstructed feedstock (Mt)")
        ax.legend(loc="upper left", handletextpad=0.1)
        panel(ax, "ab"[k], x=-0.24)
    # b country checks across years for Indonesia/Brazil
    ax = fig.add_subplot(gs[1, 0])
    spec = [("IDN", "palm", OIL["Palm"], "Indonesia palm"), ("BRA", "soy", OIL["Soybean"], "Brazil soy"),
            ("BRA", "palm", OIL["Palm"], "Brazil palm")]
    for reg, oil, c, lab in spec:
        q = x[(x.region == reg) & (x.oil == oil)]
        ax.plot(q.year, q.reconstructed_Mt - q.external_Mt, color=c, lw=1.8,
                ls="-" if lab != "Brazil palm" else "--", label=lab)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xlim(2014, 2023.5); ax.set_ylim(-1.6, 0.75)
    ax.set_xticks([2015, 2018, 2021, 2023])
    ax.set_xlabel("Year"); ax.set_ylabel("Reconstruction minus\nexternal estimate (Mt)")
    ax.set_xticks([2015, 2017, 2019, 2021, 2023])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, ncol=2, fontsize=11, columnspacing=0.8,
              handlelength=1.6, borderaxespad=0.2)
    panel(ax, "c", x=-0.24, y=1.3)
    # c EU mix sensitivity
    ax = fig.add_subplot(gs[1, 1])
    for t, c, lab in [("eat_qty", TGT["eat_qty"], "Total oil"),
                      ("eat_quality", TGT["eat_quality"], "Unsaturated oil")]:
        q = m[m.target == t]
        ax.plot(q.year, q.attributable_Mt, "-o", color=c, ms=4.5, lw=1.8, label=lab)
    ax.set_xticks([2015, 2017, 2019, 2021, 2023]); ax.set_xlim(2014.5, 2023.5); ax.set_ylim(3, 15)
    ax.set_xlabel("Year"); ax.set_ylabel("Additional gap with\nEU mix reweighting (Mt)")
    ax.legend(loc="upper left")
    panel(ax, "d", x=-0.24, y=1.04)
    table(pd.concat([x.assign(panel="a-c"), m.assign(panel="d")]), "FigS12", SUB)
    save(fig, "FigS12", SUB)


# ------------------------------------------------------------------ S13 food-use basis and crushing co-production
def s13():
    cv = pd.read_csv(RES + "options_compatible.csv")
    cop = pd.read_csv(RES + "fig1d_coproduction_2023.csv").sort_values("oil_share", ascending=False)
    fig = plt.figure(figsize=(183 * MM, 150 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 0.85], wspace=0.75, hspace=0.95, left=0.2, right=0.97, top=0.9, bottom=0.09)
    # a compatible volumes under three food-use bases
    ax = fig.add_subplot(gs[0, 0])
    rows = [("Fixed shares", "strict_d0"), ("±25% by producer", "strict_d0.25"), ("±50% by producer", "strict_d0.5"),
            ("Palm eligible", "full_d0")]
    bases = [("", "Central (reconstructed food use)", TGT["eat_quality"]),
             ("faostat_other_", "FAOSTAT, difference held as other use", GREY),
             ("faostat_released_", "FAOSTAT, difference released", "#1B9E77")]
    for i, (lab, key) in enumerate(rows):
        for k, (pre, blab, col) in enumerate(bases):
            q = cv[cv.setting == pre + key]
            y = i + (k - 1) * 0.24
            ax.plot([q.compatible_Mt.min(), q.compatible_Mt.max()], [y, y], color=col, lw=4, solid_capstyle="butt")
            ax.plot(q.compatible_Mt.min(), y, "o", color=col, ms=5)
    ax.axvline(61.675, color=LIGHT, lw=1, zorder=0)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.tick_params(axis="y", length=0); ax.set_xlim(0, 110); ax.set_xlabel("Compatible volume, 2035 (Mt)")
    legend(ax, [Line2D([], [], color=c, lw=4, label=l) for _, l, c in bases], loc="lower center",
           bbox_to_anchor=(0.45, 1.0), ncol=1, fontsize=11, borderaxespad=0.2)
    panel(ax, "a", x=-0.5, y=1.33)
    # b additional gap at the projected request under the three food-use bases
    hb = pd.read_csv(RES + "food_basis_headline.csv")
    ax = fig.add_subplot(gs[0, 1])
    labs = [("fixed_strict", "Fixed"), ("reloc25_strict", "±25%"), ("reloc50_strict", "±50%"), ("fixed_full", "Palm eligible")]
    for i, (key, lab) in enumerate(labs):
        for k, (pre, blab, col) in enumerate(bases):
            v = hb[(hb.setting == key) & (hb.basis == {"": "central", "faostat_other_": "faostat_other", "faostat_released_": "faostat_released"}[pre])].additional.iloc[0]
            ax.barh(i + (k - 1) * 0.26, v, height=0.24, color=col)
            ax.text(v + 0.5, i + (k - 1) * 0.26, f"{v:.1f}", va="center", fontsize=10, color=col)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels([l[1] for l in labs]); ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 33); ax.set_xlabel("Additional gap (Mt)")
    panel(ax, "b", x=-0.5, y=1.12)
    # c oil share of crush output (descriptive; supply is exogenous)
    ax = fig.add_subplot(gs[1, 0])
    cls = {"Oil palm fruit": "Palm", "Palm kernel": "Palm kernel & coconut"}
    y = np.arange(len(cop))
    ax.barh(y, cop.oil_share * 100, color=[OIL[cls.get(c, "Soybean")] for c in cop.crop], height=0.68)
    ax.barh(y, (1 - cop.oil_share) * 100, left=cop.oil_share * 100, color="#E0E0E0", height=0.68)
    for i, v in enumerate(cop.oil_share):
        ax.text(v * 100 + 2, i, f"{v * 100:.0f}%", va="center", ha="left", fontsize=12, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(cop.crop.str.replace(" seed", "")); ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.set_xlabel("Oil share of crush output,\n2023 (%)"); ax.tick_params(axis="y", length=0)
    legend(ax, [Patch(color="#E0E0E0", label="Meal and cake")], loc="lower center", bbox_to_anchor=(0.5, 1.0),
           borderaxespad=0.2)
    panel(ax, "c", x=-0.5, y=1.12)
    table(pd.concat([cv[cv.setting.str.contains("strict_d0|full_d0$|faostat")].assign(panel="a"),
                     cop.assign(panel="c"), pd.read_csv(RES + "food_basis_headline.csv").assign(panel="b")], ignore_index=True, sort=False), "FigS13", SUB)
    save(fig, "FigS13", SUB)


FIGS = {1: s1, 2: s2, 3: s3, 4: s4, 5: s5, 6: s6, 7: s7, 8: s8, 9: s9, 10: s10, 11: s11, 12: s12, 13: s13}
if __name__ == "__main__":
    for k in ([int(a) for a in sys.argv[1:]] or FIGS):
        FIGS[k]()
