"""Fig. 5 | Options compared on one scale: compatible biofuel feedstock in 2035.
a, compatible volume (unsaturated-oil gap within 0.1-1 Mt of the no-biofuel gap of the current system) under fixed
national shares and palm exclusion, with each option applied alone; b, total unsaturated-oil gap at the projected
request for all 16 combinations of four options (used cooking oil 5% of food use, oil output +10%, import capacity
+50%, convergence of above-reference food use)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from figstyle import RES, TGT, LEVER, LEVER_LAB, MM, INK, panel, save, table

E35 = 61.675
cv = pd.read_csv(RES + "options_compatible.csv")
OPT = [("strict_d0", "Fixed shares, palm excluded", "#2166AC"),
       ("opt_uco5_domestic", "Used cooking oil, domestic use", LEVER["S1"]),
       ("opt_uco5_traded", "Used cooking oil, traded", LEVER["S1"]),
       ("opt_output10", "Oil output +10%", LEVER["S2"]),
       ("opt_imports50", "Import capacity +50%", LEVER["S3"]),
       ("opt_diet", "Diet convergence", LEVER["S4"]),
       ("strict_d0.25", "±25% by producer", "#2166AC"),
       ("strict_d0.5", "±50% by producer", "#2166AC"),
       ("full_d0", "Palm eligible", "#D95F02"),
       ("unrestricted", "Unrestricted allocation", "#4D4D4D")]
r = pd.read_csv(RES + "runs_summary.csv")
st = r[(r.block == "strategy") & (r.target == "eat_quality") & (r.m == 1.0)].copy()
st["gap"] /= 1e6
st["strategies"] = st.strategies.fillna("none")
q = st.set_index("strategies").gap.sort_values(ascending=False)
S0 = r[(r.block == "structural") & (r.year == 2035) & (r.target == "eat_quality") &
       (r.relax.fillna("none") == "none")].gap.iloc[0] / 1e6
LV = ["S1", "S2", "S3", "S4"]

fig = plt.figure(figsize=(183 * MM, 190 * MM))
gs = fig.add_gridspec(3, 1, height_ratios=[1.45, 1.0, 0.55], hspace=0.34, left=0.33, right=0.95, top=0.98, bottom=0.04)

# a compatible volumes, one option at a time (main panel)
ax = fig.add_subplot(gs[0])
base = cv[cv.setting == "strict_d0"].compatible_Mt
for i, (name, lab, col) in enumerate(OPT):
    v = cv[cv.setting == name].compatible_Mt
    lo, hi = v.min(), v.max()
    ax.plot([lo, hi], [i, i], color=col, lw=5, solid_capstyle="butt")
    ax.plot(lo, i, "o", color=col, ms=6)
    if i:
        ax.text(hi + 1.5, i, f"+{lo - base.min():.0f}", va="center", fontsize=12, color=INK)
ax.axvline(E35, color="#BDBDBD", lw=1, zorder=0)
ax.text(E35 + 0.8, -0.55, "2035 request", fontsize=12, color="#636363", va="bottom")
ax.axhline(5.5, color="#BDBDBD", lw=0.8, ls=(0, (3, 2)))
ax.axvspan(base.min(), base.max(), color="#EEF3F9", zorder=0, lw=0)
ax.set_yticks(range(len(OPT)), [o[1] for o in OPT]); ax.set_ylim(len(OPT) - 0.4, -1.0)
for tl, o in zip(ax.get_yticklabels(), OPT):
    tl.set_color(o[2])
ax.tick_params(axis="y", length=0)
ax.set_xlim(30, 90); ax.set_xticks([40, 50, 60, 70, 80, 90]); ax.set_xlabel("Compatible volume (Mt)")
panel(ax, "a", x=-0.47, y=1.0)

# b all combinations at the projected request
ax = fig.add_subplot(gs[1])
x = np.arange(len(q))
ax.bar(x, q.values, color=TGT["eat_quality"], width=0.66)
ax.axhline(S0, color="#636363", lw=1, ls="--")
ax.text(len(q) - 0.5, S0 + 0.6, "No biofuel, current system", ha="right", va="bottom", fontsize=12, color="#636363")
for i in [0, len(q) - 1]:
    ax.text(x[i], q.values[i] + 0.5, f"{q.values[i]:.1f}", ha="center", va="bottom", fontsize=12)
ax.set_xlim(-0.6, len(q) - 0.4); ax.set_ylim(0, np.ceil(q.max() / 5) * 5 + 3); ax.set_xticks([])
ax.set_ylabel("Unsaturated-oil gap\nat 61.7 Mt (Mt)")
ax.spines["bottom"].set_visible(False)
panel(ax, "b", x=-0.47, y=1.0)
mx = fig.add_subplot(gs[2], sharex=ax)
for j, s in enumerate(LV):
    on = np.array([s in k.split("+") for k in q.index])
    mx.scatter(x[~on], [j] * (~on).sum(), s=34, color="#E0E0E0", zorder=2)
    mx.scatter(x[on], [j] * on.sum(), s=40, color=LEVER[s], zorder=3)
for i, k in enumerate(q.index):
    js = [j for j, s in enumerate(LV) if s in k.split("+")]
    if len(js) > 1:
        mx.plot([i, i], [min(js), max(js)], color="#636363", lw=1.2, zorder=1)
mx.set_yticks(range(4), ["Used cooking oil", "Oil output +10%", "Import capacity +50%", "Diet convergence"])
mx.set_ylim(3.6, -0.6)
for t, s in zip(mx.get_yticklabels(), LV):
    t.set_color(LEVER[s])
mx.tick_params(length=0); mx.tick_params(axis="x", labelbottom=False)
for sp in mx.spines.values():
    sp.set_visible(False)
save(fig, "Fig5")

table(pd.concat([cv[cv.setting.isin([o[0] for o in OPT])].assign(panel="a"),
                 q.rename("gap_Mt").reset_index().assign(panel="b", m=1.0)], ignore_index=True, sort=False), "Fig5")
