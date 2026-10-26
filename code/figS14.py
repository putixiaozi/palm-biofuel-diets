"""Supplementary Fig. 12 | Country classification under the fair allocation (illustration) (2035, projected request,
constrained network, efficient-and-fair allocation).
a, countries by the dominant layer of their gap (country Shapley values): import-limited (trade access) and
biofuel-exposed (conversion location and feedstock eligibility); shading of import-limited countries = relative gap;
b, the 15 largest country gaps split into the gap without biofuel and the additional gap with biofuel;
c, regional totals of the same two parts under the fair allocation and under the minimum-gap solution without a
fairness rule."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from figstyle import RES, LAYER, USE, DEC, MM, INK, panel, save, table, world_map

q = pd.read_csv(RES + "fig6_country.csv").query("target == 'eat_quality'").set_index("iso3c")
fair = pd.read_csv(RES + "fair_check.csv").query("target == 'eat_quality'")
TYPE = {"L2": "Import-limited", "L34": "Biofuel-exposed"}
TC = {"L2": LAYER["L2"], "L34": LAYER["L3"]}
ramp = mpl.colors.LinearSegmentedColormap.from_list("access", ["#FCEBB0", LAYER["L2"], "#7A5A00"])
norm = mpl.colors.Normalize(0.2, 1.0)

fig = plt.figure(figsize=(183 * MM, 168 * MM))
gs = fig.add_gridspec(3, 2, height_ratios=[0.9, 0.1, 0.95], width_ratios=[1, 1.1], hspace=0.1, wspace=0.95,
                      left=0.08, right=0.98, top=0.99, bottom=0.08)

# a map
ax = fig.add_subplot(gs[0, :])
M = world_map(); cs = M["countries"]
ax.fill(*M["outline"], color="#F7F9FB", lw=0, zorder=0)
dom, rel = cs.iso3c.map(q.dominant), cs.iso3c.map(q.rel1)
fc = [ramp(norm(r)) if d == "L2" else mpl.colors.to_rgba(TC["L34"]) if d == "L34" else
      mpl.colors.to_rgba("#E3E3E3") if d == "none" else mpl.colors.to_rgba("white") for d, r in zip(dom, rel)]
cs.plot(ax=ax, color=fc, edgecolor="white", linewidth=0.15, zorder=1)
cs[dom.isna().values].plot(ax=ax, facecolor="white", edgecolor="#BDBDBD", linewidth=0.3, zorder=1)
M["disputed"].plot(ax=ax, color="#4D4D4D", linewidth=0.6, zorder=3)
ax.plot(*M["outline"], color="#BDBDBD", lw=0.6, zorder=0)
ax.set_xlim(-2.72, 2.72); ax.set_ylim(-1.0, 1.32); ax.set_aspect("equal"); ax.axis("off")
pt = cs.set_index("iso3c").representative_point()
LBL = {"IND": (1.45, -0.28), "COD": (0.0, -0.3), "PAK": (0.80, 0.90), "BGD": (1.58, 0.72), "EGY": (0.22, 0.74),
       "NGA": (-0.30, -0.12), "TZA": (0.80, -0.10), "ETH": (0.86, 0.12), "MDG": (1.05, -0.34), "IDN": (2.05, -0.42),
       "VNM": (2.02, 0.56), "PHL": (2.25, 0.30)}
for iso, xy in LBL.items():
    x, y = pt[iso].x, pt[iso].y
    if xy is None:
        ax.text(x, y, iso, fontsize=12, ha="center", va="center", color="white")
    else:
        ax.annotate(iso, (x, y), xy, fontsize=12, ha="center", va="center", color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, shrinkA=4, shrinkB=0))
# legend row under the map
lg = fig.add_subplot(gs[1, :]); lg.axis("off")
cax = lg.inset_axes([0.03, 0.15, 0.2, 0.3])
cb = fig.colorbar(mpl.cm.ScalarMappable(norm, ramp), cax=cax, orientation="horizontal", ticks=[0.2, 0.6, 1.0])
cb.ax.set_xticklabels(["20", "60", "100"]); cb.outline.set_visible(False); cb.ax.tick_params(length=2)
cax.set_title("Import-limited (% of target)", fontsize=12, loc="left", pad=4)
lg.legend(handles=[Patch(color=TC["L34"], label="Biofuel-exposed"),
                   Patch(color="#E3E3E3", label="Target met"),
                   Patch(facecolor="white", edgecolor="#BDBDBD", label="Not modelled")],
          loc="center left", bbox_to_anchor=(0.4, 0.42), ncol=3, handlelength=1.0, columnspacing=0.9, borderaxespad=0)
# South China Sea inset; the disputed boundary geometry comes from the supplied standard map.
inset = ax.inset_axes([0.675, 0.01, 0.13, 0.22])   # southern Indian Ocean, clear of land and labels
inset.fill(*M["outline"], color="#F7F9FB", lw=0, zorder=0)
inset_cs = M["countries"]
inset_fc = [ramp(norm(r)) if d == "L2" else mpl.colors.to_rgba(TC["L34"]) if d == "L34" else
            mpl.colors.to_rgba("#E3E3E3") if d == "none" else mpl.colors.to_rgba("white") for d, r in zip(dom, rel)]
inset_cs.plot(ax=inset, color=inset_fc, edgecolor="white", linewidth=0.12, zorder=1)
M["disputed"].plot(ax=inset, color="#4D4D4D", linewidth=0.7, zorder=3)
inset.set_xlim(1.40, 1.92); inset.set_ylim(-0.26, 0.50); inset.set_aspect("equal"); inset.set_xticks([]); inset.set_yticks([])
for sp in inset.spines.values(): sp.set_visible(True); sp.set_color("#777777"); sp.set_linewidth(0.7)
inset.text(0.03, 0.03, "South China Sea", transform=inset.transAxes, fontsize=9, color=INK, va="bottom")
panel(ax, "a", x=0.0, y=0.93)

# b largest country shortfalls
ax = fig.add_subplot(gs[2, 0])
top = q.sort_values("u1", ascending=False).head(15)
yy = np.arange(len(top))
ax.barh(yy, top.u0 / 1e6, color=DEC["scarcity"], height=0.68, label="Without biofuel")
ax.barh(yy, top.attrib / 1e6, left=top.u0 / 1e6, color=USE["biofuel"], height=0.68, label="Additional")
ax.set_yticks(yy, top.index); ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
for tl, iso in zip(ax.get_yticklabels(), top.index):
    tl.set_color(TC[q.dominant[iso]])
xm = np.ceil((top.u1.max() / 1e6) + 0.3)
ax.set_xlim(0, xm); ax.set_xlabel("Unsaturated-oil gap, 2035 (Mt)")
ax.legend(loc="lower right", handlelength=1.0, borderaxespad=0)
panel(ax, "b", x=-0.2)

# c regional totals under the fair allocation and under the minimum-gap solution without a fairness rule
ax = fig.add_subplot(gs[2, 1])
rr = pd.read_csv(RES + "region_rule_comparison.csv").set_index("iso3c").drop("Total")
rr = rr.loc[(rr.gap_fair + rr.gap_mingap).sort_values(ascending=False).index]
yy = np.arange(len(rr))
for k, (rule, alpha, lab) in enumerate([("fair", 1.0, "Fair allocation"), ("mingap", 0.45, "Minimum gap only")]):
    off = (k - 0.5) * 0.38
    ax.barh(yy + off, rr["gap_" + rule] - rr["add_" + rule], color=DEC["scarcity"], height=0.36, alpha=alpha)
    ax.barh(yy + off, rr["add_" + rule], left=rr["gap_" + rule] - rr["add_" + rule], color=USE["biofuel"],
            height=0.36, alpha=alpha)
ax.set_yticks(yy, rr.index); ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
ax.set_xlim(0, np.ceil(rr[["gap_fair", "gap_mingap"]].max().max() + 1)); ax.set_xlabel("Unsaturated-oil gap, 2035 (Mt)")
ax.legend(handles=[Patch(color="#636363", label="Fair allocation"), Patch(color="#636363", alpha=0.45,
                                                                           label="Minimum gap only")],
          loc="lower right", handlelength=1.0, borderaxespad=0)
panel(ax, "c", x=-0.62)
save(fig, "FigS14", "SI")

out = q[q.dominant != "none"].reset_index()[["iso3c", "region", "dominant", "pop", "D", "u1", "u0", "attrib", "rel1"]]
out["type"] = out.dominant.map(TYPE)
table(out.assign(u1=out.u1 / 1e6, u0=out.u0 / 1e6, attrib=out.attrib / 1e6, D=out.D / 1e6)
      .rename(columns={"D": "target_Mt", "u1": "gap_Mt", "u0": "gap_no_biofuel_Mt",
                       "attrib": "additional_gap_Mt", "rel1": "relative_gap"}), "FigS14", "SI")
print(fair[["tol", "gap", "max_rel"]].to_string())
