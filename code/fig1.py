"""Fig. 1 | Biofuel growth moved food-fuel competition from oil quantity to oil type, 2010-2023."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from figstyle import RES, OIL, OIL_ORDER, CLASS, TGT, USE, MM, INK, panel, save, table

sup = pd.read_csv(RES + "fig1a_supply_by_oil.csv").pivot(index="year", columns="group", values="Mt")[OIL_ORDER]
use = pd.read_csv(RES + "fig1b_use_by_class.csv")
fuel = use[use.use == "biofuel"].pivot(index="year", columns="class", values="Mt")[["Unsaturated", "Palm-type"]]
summ = pd.read_csv(RES + "fig1_summary_by_year.csv").set_index("year")
reg = pd.read_csv(RES + "fig1c_biofuel_by_region_oil.csv")
reg = reg[reg.year == 2023].copy()
reg["region"] = reg.region.replace({"India": "Rest of world", "Africa": "Rest of world"})
reg = reg.pivot_table(index="region", columns="group", values="Mt", aggfunc="sum").reindex(columns=OIL_ORDER).fillna(0)
order = reg.drop("Rest of world").sum(axis=1).sort_values().index.tolist()
reg = reg.loc[["Rest of world"] + order]
hist = pd.read_csv(RES + "history_fuel_cost.csv").pivot(index="year", columns="target", values="biofuel_cost")
E = pd.read_csv(RES + "history_fuel_cost.csv").query("target == 'eat_qty'").set_index("year").E

fig = plt.figure(figsize=(183 * MM, 160 * MM))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.45, wspace=0.95, left=0.1, right=0.97, top=0.88, bottom=0.08)

# a supply by oil type
ax = fig.add_subplot(gs[0, 0])
yrs = sup.index.values
ax.stackplot(yrs, sup.T.values, colors=[OIL[g] for g in OIL_ORDER], edgecolor="white", linewidth=0.4)
for y, ha in [(2010, "left"), (2023, "right")]:
    ax.text(y + (0.3 if ha == "left" else -0.3), sup.loc[y].sum() + 10, f"{sup.loc[y].sum():.0f}", ha=ha, va="bottom", fontsize=12, color=INK)
ax.set_xlim(2010, 2023); ax.set_ylim(0, 260); ax.set_xticks([2010, 2016, 2022])
ax.set_ylabel("Supply (Mt)")
panel(ax, "a", x=-0.36)

# b biofuel feedstock by oil class
ax = fig.add_subplot(gs[0, 1])
ax.stackplot(yrs, fuel.T.values, colors=[CLASS[c] for c in fuel.columns], edgecolor="white", linewidth=0.4)
ax.text(2010.3, 54, f"{summ.biofuel_share[2010] * 100:.0f}% \u2192 {summ.biofuel_share[2023] * 100:.0f}%\nof supply",
        ha="left", va="top", fontsize=12)
ax.text(2016.5, fuel.loc[2016, "Unsaturated"] * 0.42, "Unsaturated", color="white", fontsize=12, ha="center")
ax.text(2018.0, 21.5, "Palm-type", color="white",
        fontsize=12, ha="center", va="center")
ax.set_xlim(2010, 2023); ax.set_ylim(0, 55); ax.set_xticks([2010, 2016, 2022])
ax.set_ylabel("Biofuel feedstock (Mt)")
panel(ax, "b", x=-0.36)

# c 2023 biofuel feedstock by producing region and oil
ax = fig.add_subplot(gs[0, 2])
left = np.zeros(len(reg))
for g in OIL_ORDER:
    ax.barh(range(len(reg)), reg[g], left=left, color=OIL[g], height=0.68, edgecolor="white", linewidth=0.4)
    left += reg[g].values
for i, t in enumerate(left):
    ax.text(t + 0.3, i, f"{t:.1f}", va="center", fontsize=12)
ax.set_yticks(range(len(reg)), [r.replace("United States", "USA").replace("Rest of world", "Others") for r in reg.index])
ax.set_xlim(0, 14); ax.set_xticks([0, 5, 10]); ax.set_xlabel("Feedstock, 2023 (Mt)")
ax.tick_params(axis="y", length=0)
panel(ax, "c", x=-0.62)

# d additional reference gaps under the constrained network (main panel)
ax = fig.add_subplot(gs[1, :])
ax.plot(E.index, E.values, color=USE["biofuel"], lw=1.2, ls=(0, (4, 2)))
for t, mk in [("eat_qty", "s"), ("eat_quality", "o")]:
    ax.plot(hist.index, hist[t], color=TGT[t], lw=2.4, marker=mk, ms=6)
lab = {"eat_quality": "Unsaturated oil", "eat_qty": "Total oil"}
for t in ["eat_quality", "eat_qty"]:
    ax.text(2023.4, hist.loc[2023, t], f"{lab[t]}: {hist.loc[2010, t]:.1f} \u2192 {hist.loc[2023, t]:.1f}",
            color=TGT[t], va="center", fontsize=12)
ax.text(2023.4, E[2023], f"Biofuel feedstock: {E[2010]:.1f} \u2192 {E[2023]:.1f}", color=USE["biofuel"], va="center",
        fontsize=12)
ax.set_xlim(2009.6, 2030.5); ax.set_ylim(0, 42); ax.set_xticks(range(2010, 2024, 2))
ax.set_ylabel("Additional gap with\nbiofuel (Mt)")
ax.spines["bottom"].set_bounds(2010, 2023)
panel(ax, "d", x=-0.075, y=1.0)

# shared oil legend above a-c
fig.legend(handles=[Patch(color=OIL[g], label=g) for g in OIL_ORDER], loc="upper center", ncol=3,
           bbox_to_anchor=(0.54, 0.995), columnspacing=1.2)
save(fig, "Fig1")

rows = [sup.reset_index().melt(id_vars="year", var_name="series", value_name="value").assign(panel="a", unit="Mt"),
        fuel.reset_index().melt(id_vars="year", var_name="series", value_name="value").assign(panel="b", unit="Mt"),
        summ.biofuel_share.rename("value").reset_index().assign(series="biofuel share of supply", panel="b", unit="fraction"),
        reg.reset_index().melt(id_vars="region", var_name="series", value_name="value").rename(columns={"region": "category"})
        .assign(panel="c", year=2023, unit="Mt"),
        hist[["eat_qty", "eat_quality"]].reset_index().melt(id_vars="year", var_name="series", value_name="value")
        .assign(panel="d", unit="Mt"),
        E.rename("value").reset_index().assign(series="biofuel feedstock use", panel="d", unit="Mt")]
table(pd.concat(rows)[["panel", "series", "category", "year", "value", "unit"]], "Fig1")
