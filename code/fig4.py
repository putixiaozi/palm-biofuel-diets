"""Fig. 4 | South Asia and sub-Saharan Africa hold the largest part of the additional unsaturated-oil gap in 2035.
a, regional additional gaps in 40 efficient allocations (random country weights break ties among allocations with the
same minimal total gap), the fair allocation and one further minimum-gap solution; b, combined share of South Asia and
sub-Saharan Africa in the additional gap across the same allocations."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figstyle import RES, USE, MM, INK, panel, save, table

d = pd.read_csv(RES + "allocation_rules.csv")
rr = pd.read_csv(RES + "region_rule_comparison.csv").set_index("iso3c").drop("Total")
wide = d.pivot(index="draw", columns="region", values="additional")
order = wide.median().sort_values(ascending=False).index.tolist()
share = (wide["South Asia"] + wide["Sub-Saharan Africa"]) / wide.sum(axis=1)
fair_share = (rr.loc["South Asia", "add_fair"] + rr.loc["Sub-Saharan Africa", "add_fair"]) / rr.add_fair.sum()
min_share = (rr.loc["South Asia", "add_mingap"] + rr.loc["Sub-Saharan Africa", "add_mingap"]) / rr.add_mingap.sum()

fig = plt.figure(figsize=(183 * MM, 82 * MM))
gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.5, left=0.2, right=0.96, top=0.84, bottom=0.17)
rng = np.random.default_rng(3)

ax = fig.add_subplot(gs[0, 0])
for i, r in enumerate(order):
    v = wide[r].values
    ax.scatter(v, i + rng.uniform(-0.2, 0.2, len(v)), s=14, color="#BDBDBD", edgecolor="none", zorder=2)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    ax.plot([q1, q3], [i, i], color="#636363", lw=4, solid_capstyle="butt", zorder=3)
    ax.plot([med, med], [i - 0.28, i + 0.28], color=INK, lw=1.6, zorder=4)
    ax.plot(rr.loc[r, "add_fair"], i, "D", color=USE["biofuel"], ms=7, zorder=5)
    ax.plot(rr.loc[r, "add_mingap"], i, "D", mfc="white", mec=USE["biofuel"], mew=1.5, ms=7, zorder=5)
ax.set_yticks(range(len(order)), order); ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
ax.set_xlabel("Additional unsaturated-oil gap, 2035 (Mt)"); ax.set_xlim(left=0)
ax.legend(handles=[Line2D([], [], marker="o", color="#BDBDBD", ls="none", ms=5, label="40 efficient allocations"),
                   Line2D([], [], color="#636363", lw=4, label="Interquartile range"),
                   Line2D([], [], marker="D", color=USE["biofuel"], ls="none", ms=6, label="Fair allocation"),
                   Line2D([], [], marker="D", mfc="white", mec=USE["biofuel"], ls="none", ms=6, label="Other minimum-gap solution")],
          loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=11, columnspacing=0.8, handletextpad=0.3,
          borderaxespad=0.2)
panel(ax, "a", x=-0.62, y=1.2)

ax = fig.add_subplot(gs[0, 1])
ax.scatter(share * 100, rng.uniform(-0.18, 0.18, len(share)), s=18, color="#BDBDBD", edgecolor="none", zorder=2)
ax.plot([np.percentile(share, 25) * 100, np.percentile(share, 75) * 100], [0, 0], color="#636363", lw=6,
        solid_capstyle="butt", zorder=3)
ax.plot(fair_share * 100, 0, "D", color=USE["biofuel"], ms=8, zorder=5)
ax.plot(min_share * 100, 0, "D", mfc="white", mec=USE["biofuel"], mew=1.5, ms=8, zorder=5)
ax.axvline(50, color="#BDBDBD", lw=0.9, ls=(0, (3, 2)), zorder=0)
ax.set_xlim(30, 100); ax.set_xticks([40, 60, 80, 100]); ax.set_ylim(-0.6, 0.6); ax.set_yticks([])
for sp in ("left",):
    ax.spines[sp].set_visible(False)
ax.set_xlabel("South Asia + sub-Saharan\nAfrica share (%)")
panel(ax, "b", x=-0.12, y=1.2)
save(fig, "Fig4")
table(pd.concat([d.assign(panel="a"), rr.reset_index().assign(panel="a,b")], ignore_index=True, sort=False), "Fig4")
print("share across draws: min %.2f median %.2f max %.2f; >50%% in %d of %d draws; fair %.2f; other %.2f" %
      (share.min(), share.median(), share.max(), (share > 0.5).sum(), len(share), fair_share, min_share))
