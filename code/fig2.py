"""Fig. 2 | Projected 2035 supply covers fuel and reference diets in aggregate but not by oil type.
a, total reference gaps against the requested biofuel feedstock, constrained network vs unrestricted allocation, for the
unsaturated-oil and total-oil references; b, Shapley values of the additional unsaturated-oil gap at the scenario
request over five constraint layers, 2035 and 2023; c, additional unsaturated-oil gap against the reference level."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from figstyle import RES, TGT, LAYER, LAYER_LAB, MM, INK, panel, save, table

E35 = 61.675
fr = pd.read_csv(RES + "fig4_frontier.csv").query("year == 2035 and target in ['eat_quality', 'eat_qty']")
at = pd.read_csv(RES + "fig3_layer_shapley_attrib.csv").query("target == 'eat_quality'").set_index("year")
ref = pd.read_csv(RES + "reference_target_diagnostic.csv")
LY = ["L1", "L2", "L3", "L4", "L5"]
LAB = {"eat_quality": "Unsaturated oil", "eat_qty": "Total oil"}
ALLOC = {"constrained": ("-", "Constrained network"), "pool": ((0, (4, 2)), "Unrestricted allocation")}

fig = plt.figure(figsize=(183 * MM, 118 * MM))
gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], hspace=0.75, wspace=0.95, left=0.1, right=0.97, top=0.9,
                      bottom=0.12)

# a total gaps against request (main panel)
ax = fig.add_subplot(gs[:, 0])
for t in ["eat_quality", "eat_qty"]:
    for acc, (ls, _) in ALLOC.items():
        q = fr[(fr.target == t) & (fr.access == acc)].sort_values("E")
        ax.plot(q.E, q.gap, color=TGT[t], ls=ls, lw=2.2 if t == "eat_quality" else 1.6)
ax.axvline(E35, color="#BDBDBD", lw=1, zorder=0)
ax.text(E35 + 1.2, 45, "2035 request", color="#636363", fontsize=12, va="center", rotation=90)
c = fr[(fr.target == "eat_quality") & (fr.access == "constrained")].set_index("m").gap
ax.annotate(f"{c[1.0]:.1f}", (E35, c[1.0]), (E35 - 15, c[1.0] + 6), fontsize=12, color=TGT["eat_quality"],
            arrowprops=dict(arrowstyle="-", color=TGT["eat_quality"], lw=0.7))
ax.text(1.5, c[0.0] + 0.8, f"{c[0.0]:.1f} without biofuel", fontsize=12, color=TGT["eat_quality"], va="bottom")
ax.set(xlim=(0, 125), ylim=(-0.5, 75), xlabel="Requested biofuel feedstock (Mt)", ylabel="Total reference gap (Mt)")
ax.legend(handles=[Line2D([], [], color=TGT[t], lw=2, label=LAB[t]) for t in ["eat_quality", "eat_qty"]] +
          [Line2D([], [], color="#636363", ls=ls, lw=1.6, label=l) for ls, l in ALLOC.values()],
          loc="upper left", ncol=1, handlelength=2.2, borderaxespad=0.2, labelspacing=0.3)
panel(ax, "a", x=-0.17, y=1.0)

# b Shapley values of the additional unsaturated-oil gap
ax = fig.add_subplot(gs[0, 1])
y = np.arange(5)
off = {2035: -0.19, 2023: 0.19}
for yr, alpha in [(2035, 1.0), (2023, 0.45)]:
    v = at.loc[yr, LY].values
    ax.barh(y + off[yr], v, height=0.38, color=[LAYER[l] for l in LY], alpha=alpha)
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(y, [LAYER_LAB[l] for l in LY]); ax.invert_yaxis()
for tl, l in zip(ax.get_yticklabels(), LY):
    tl.set_color(LAYER[l])
ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
lim = np.ceil(max(at.loc[[2023, 2035], LY].abs().max().max(), 4) + 0.5)
ax.set_xlim(-lim / 2, lim); ax.set_xlabel("Shapley value (Mt)")
ax.legend(handles=[Patch(color="#636363", label="2035"), Patch(color="#636363", alpha=0.45, label="2023")],
          loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.0, borderaxespad=0.2)
panel(ax, "b", x=-0.95, y=1.12)

# c reference level
ax = fig.add_subplot(gs[1, 1])
for year, marker, color in [(2023, "o", "#969696"), (2035, "s", TGT["eat_quality"])]:
    q = ref[ref.year == year]
    ax.plot(q.reference_g_per_day, q.attributable_Mt, marker=marker, color=color, lw=1.8, ms=6, label=str(year))
ax.axvline(40, color="#BDBDBD", lw=0.8, ls=":")
ax.set(xticks=[20, 30, 40, 50], xlabel="Unsaturated-oil reference\n(g per person per day)",
       ylabel="Additional gap (Mt)")
ax.set_ylim(bottom=-0.5)
ax.legend(loc="upper left", handlelength=1.4, borderaxespad=0)
panel(ax, "c", x=-0.42, y=1.02)
save(fig, "Fig2")

table(pd.concat([
    fr[["target", "access", "m", "E", "gap", "cut", "fuel_unmet"]].assign(panel="a", year=2035),
    at[LY].reset_index().melt(id_vars="year", var_name="layer", value_name="shapley_Mt").assign(panel="b"),
    ref.assign(panel="c")], ignore_index=True, sort=False), "Fig2")
