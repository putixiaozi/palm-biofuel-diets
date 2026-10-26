"""Fig. 3 | Palm eligibility and production location are substitute routes to the same oil swap, 2035.
a, total unsaturated-oil gap against the requested feedstock under palm exclusion or full eligibility, with fixed
national shares or incumbent producers varying by +-25% or +-50%, and unrestricted allocation; b, compatible volumes
(gap within 0.1-1 Mt of the no-biofuel gap of the current system); c, change in national feedstock use by oil at the
projected request when producers may vary by +-50% under palm exclusion; d, the oil swap between fuel and food relative
to fixed shares under palm exclusion."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from figstyle import RES, OIL, OIL_ORDER, CLASS, MM, INK, panel, save, table

E35 = 61.675
PAL = {"strict": "#2166AC", "full": "#D95F02", "unrestricted": "#4D4D4D"}
LS = {0.0: "-", 0.25: "-.", 0.5: ":"}
fr = pd.read_csv(RES + "options_frontier.csv")
cv = pd.read_csv(RES + "options_compatible.csv")
dt = pd.read_csv(RES + "options_detail_m1.csv.gz").fillna(0)
GROUP = {"soy": "Soybean", "rapeseed": "Rapeseed", "sunflower": "Sunflower", "palm": "Palm",
         "palmkernel": "Palm kernel & coconut", "coconut": "Palm kernel & coconut"}
PALMTYPE = {"palm", "palmkernel", "coconut"}
OILS = [k for k in ["coconut", "cottonseed", "groundnut", "maize", "olive", "palm", "palmkernel", "rapeseed",
                    "ricebran", "sesame", "soy", "sunflower"]]


def cols(prefix, d):
    return [f"{prefix}_{k}" for k in OILS if f"{prefix}_{k}" in d.columns]


ref0 = fr[(fr.setting == "strict_d0") & (fr.m == 0)].gap_Mt.iloc[0]

fig = plt.figure(figsize=(183 * MM, 172 * MM))
gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], height_ratios=[1.1, 1], hspace=0.75, wspace=0.62, left=0.1,
                      right=0.97, top=0.86, bottom=0.08)

# a total unsaturated-oil gap against request (main panel)
ax = fig.add_subplot(gs[0, 0])
for palm in ["strict", "full"]:
    for d, ls in LS.items():
        q = fr[fr.setting == f"{palm}_d{d:g}"].sort_values("m")
        ax.plot(q.requested_Mt, q.gap_Mt, color=PAL[palm], ls=ls, lw=2)
q = fr[fr.setting == "unrestricted"].sort_values("m")
ax.plot(q.requested_Mt, q.gap_Mt, color=PAL["unrestricted"], lw=1.6)
ax.axhline(ref0, color="#BDBDBD", lw=0.9, ls=(0, (2, 2)), zorder=0)
ax.text(3, ref0 - 1.2, "No biofuel, current system", fontsize=12, color="#636363", va="top", ha="left")
ax.axvline(E35, color="#BDBDBD", lw=1, zorder=0)
ax.set(xlim=(0, 125), ylim=(-0.5, 45), xlabel="Requested biofuel feedstock (Mt)",
       ylabel="Unsaturated-oil gap (Mt)")
fig.legend(handles=[Line2D([], [], color=PAL["strict"], lw=2, label="US–EU palm exclusion"),
                   Line2D([], [], color=PAL["full"], lw=2, label="Palm eligible"),
                   Line2D([], [], color=PAL["unrestricted"], lw=1.6, label="Unrestricted allocation"),
                   Line2D([], [], color="#636363", ls="-", lw=1.6, label="Fixed national shares"),
                   Line2D([], [], color="#636363", ls="-.", lw=1.6, label="±25% by producer"),
                   Line2D([], [], color="#636363", ls=":", lw=1.6, label="±50% by producer")],
          loc="upper center", bbox_to_anchor=(0.53, 1.0), ncol=3, columnspacing=1.0, handlelength=2.2,
          borderaxespad=0.2)
panel(ax, "a", x=-0.17, y=1.02)

# b compatible volumes
ax = fig.add_subplot(gs[0, 1])
rows = [("Fixed", 0.0), ("±10%", 0.1), ("±25%", 0.25), ("±50%", 0.5)]
for i, (lab, d) in enumerate(rows):
    for palm, dy in [("strict", -0.15), ("full", 0.15)]:
        q = cv[cv.setting == f"{palm}_d{d:g}"]
        lo, hi = q.compatible_Mt.min(), q.compatible_Mt.max()
        ax.plot([lo, hi], [i + dy] * 2, color=PAL[palm], lw=4, solid_capstyle="butt")
        ax.plot(lo, i + dy, "o", color=PAL[palm], ms=5)
        ax.text(hi + 1.5, i + dy, f"{lo:.0f}–{hi:.0f}", va="center", fontsize=11, color=PAL[palm])
q = cv[cv.setting == "unrestricted"]
ax.plot([q.compatible_Mt.min(), q.compatible_Mt.max()], [4, 4], color=PAL["unrestricted"], lw=4)
ax.plot(q.compatible_Mt.min(), 4, "o", color=PAL["unrestricted"], ms=5)
ax.text(q.compatible_Mt.max() + 1.5, 4, f"{q.compatible_Mt.min():.0f}–{q.compatible_Mt.max():.0f}", va="center",
        fontsize=11, color=PAL["unrestricted"])
ax.axvline(E35, color="#BDBDBD", lw=1, zorder=0)
ax.set_yticks(range(5), [r[0] for r in rows] + ["Unrestricted"]); ax.set_ylim(4.6, -0.6)
ax.tick_params(axis="y", length=0)
ax.set_xlim(30, 115); ax.set_xticks([40, 60, 80, 100]); ax.set_xlabel("Compatible volume (Mt)")
panel(ax, "b", x=-0.38, y=1.02)

# c national feedstock change, +-50% under palm exclusion vs fixed shares
ax = fig.add_subplot(gs[1, 0])
b0 = dt[dt.setting == "strict_d0"].set_index("iso3c")
b1 = dt[dt.setting == "strict_d0.5"].set_index("iso3c")
fc = cols("fuel", b0)
delta = (b1[fc] - b0[fc]).rename(columns=lambda c: c.replace("fuel_", ""))
g = delta.T.groupby(lambda k: GROUP.get(k, "Other unsaturated")).sum().T.reindex(columns=OIL_ORDER).fillna(0)
tot = g.sum(axis=1)
top = tot.abs().sort_values(ascending=False).head(6).index
g = pd.concat([g.loc[top], g.drop(top).sum().rename("Others").to_frame().T])
g = g.loc[g.sum(axis=1).sort_values(ascending=False).index]
yy = np.arange(len(g))
pos, neg = np.zeros(len(g)), np.zeros(len(g))
for k in OIL_ORDER:
    v = g[k].values
    ax.barh(yy, np.where(v > 0, v, 0), left=pos, color=OIL[k], height=0.66, edgecolor="white", linewidth=0.3)
    ax.barh(yy, np.where(v < 0, v, 0), left=neg, color=OIL[k], height=0.66, edgecolor="white", linewidth=0.3)
    pos += np.where(v > 0, v, 0); neg += np.where(v < 0, v, 0)
for i, t in enumerate(g.sum(axis=1)):
    if abs(t) >= 0.05:
        ax.text((pos[i] if t >= 0 else neg[i]) + (0.4 if t >= 0 else -0.4), i, f"{t:+.1f}", va="center",
                ha="left" if t >= 0 else "right", fontsize=12)
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(yy, g.index); ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
lim = np.ceil(max(pos.max(), -neg.min()) + 4)
ax.set_xlim(-lim, lim); ax.set_xlabel("Change in biofuel feedstock (Mt)")
ax.legend(handles=[Patch(color=OIL[k], label=k) for k in ["Soybean", "Rapeseed", "Sunflower", "Other unsaturated", "Palm"]],
          loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=11, handlelength=0.9, columnspacing=0.8,
          borderaxespad=0.2)
panel(ax, "c", x=-0.17, y=1.22)

# d oil swap between fuel and food
ax = fig.add_subplot(gs[1, 1])


def swap(name):
    b = dt[dt.setting == name]
    out = {}
    for use in ["fuel", "food"]:
        cc = cols(use, b)
        pt = sum(b[c].sum() for c in cc if c.split("_", 1)[1] in PALMTYPE)
        un = sum(b[c].sum() for c in cc if c.split("_", 1)[1] not in PALMTYPE)
        out[use] = (pt, un)
    return out


s0 = swap("strict_d0")
items = [("Fuel", "fuel"), ("Food", "food")]
cases = [("±50% by producer", "strict_d0.5", PAL["strict"]), ("Palm eligible", "full_d0", PAL["full"])]
labels, vals = [], []
for lab, use in items:
    for cls, k in [("palm-type", 0), ("unsaturated", 1)]:
        labels.append(f"{lab}, {cls}")
yy = np.arange(4)
for j, (cl, name, col) in enumerate(cases):
    s = swap(name)
    v = [s[u][k] - s0[u][k] for _, u in items for k in [0, 1]]
    vals.append(v)
    ax.barh(yy + (j - 0.5) * 0.36, v, height=0.34, color=col, label=cl)
    for i, x in enumerate(v):
        ax.text(x + (0.4 if x >= 0 else -0.4), i + (j - 0.5) * 0.36, f"{x:+.1f}", va="center",
                ha="left" if x >= 0 else "right", fontsize=11)
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(yy, labels); ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
for tl, cls in zip(ax.get_yticklabels(), ["Palm-type", "Unsaturated"] * 2):
    tl.set_color(CLASS[cls])
lim = np.ceil(np.abs(vals).max() + 6)
ax.set_xlim(-lim, lim); ax.set_xlabel("Change relative to fixed\nshares with exclusion (Mt)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, handlelength=1.0, borderaxespad=0.2)
panel(ax, "d", x=-0.62, y=1.22)
save(fig, "Fig3")

table(pd.concat([
    fr.assign(panel="a"), cv.assign(panel="b"),
    g.reset_index().rename(columns={"index": "iso3c"}).melt(id_vars="iso3c", var_name="oil_group", value_name="change_Mt")
    .assign(panel="c", setting="strict_d0.5 minus strict_d0"),
    pd.DataFrame([dict(panel="d", setting=name, series=l, change_Mt=x) for (_, name, _), v in zip(cases, vals)
                  for l, x in zip(labels, v)])], ignore_index=True, sort=False), "Fig3")
