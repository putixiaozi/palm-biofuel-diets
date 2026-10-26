"""Fig. 1 | Analytical framework. a, schematic of fuel producers with and without palm eligibility, palm producers and
importing countries, with the two policy levers and the exchange of oils; b, the four questions and the sections that
answer them, with the exchange identity that holds at fixed supply."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Ellipse, Rectangle, Wedge
from figstyle import RES, MM, INK, save, table

BLUE, ORANGE, RED, GREY, LAND = "#2166AC", "#D95F02", "#C0392B", "#BDBDBD", "#F3EEE3"
FS = 13


def drop(ax, x, y, s, c):
    ax.add_patch(Circle((x, y), s, fc=c, ec="none", zorder=5))
    ax.add_patch(Polygon([(x - 0.93 * s, y + 0.37 * s), (x + 0.93 * s, y + 0.37 * s), (x, y + 2.1 * s)], fc=c, ec="none", zorder=5))


def drum(ax, x, y, w=0.9, h=1.15, c="#7F7F7F"):
    ax.add_patch(Rectangle((x - w / 2, y), w, h, fc=c, ec="none", zorder=4))
    ax.add_patch(Ellipse((x, y), w, 0.32, fc=c, ec="none", zorder=4))
    ax.add_patch(Ellipse((x, y + h), w, 0.32, fc="#A6A6A6", ec=c, lw=1, zorder=5))
    for f in (0.33, 0.66):
        ax.plot([x - w / 2, x + w / 2], [y + f * h, y + f * h], color="#5F5F5F", lw=1, zorder=5)


def bowl(ax, x, y, r=0.85):
    ax.add_patch(Wedge((x, y), r, 180, 360, fc="#E8E2D6", ec="#8C7B63", lw=1.4, zorder=4))
    ax.plot([x - r * 1.08, x + r * 1.08], [y, y], color="#8C7B63", lw=1.8, zorder=5)


def marker(ax, x, y, n, c=INK, fc="white"):
    ax.add_patch(Circle((x, y), 0.32, fc=fc, ec=c, lw=1.6, zorder=8))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=FS, color=c, fontweight="bold", zorder=9)


def tile(ax, x0, y0, w, h, fc=LAND):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.35", fc=fc, ec="#CFC6B5", lw=1.2, zorder=1))


def arrow(ax, p, q, c, ls="-", rad=0.0, lw=2.2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=17, color=c, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=6, shrinkA=2, shrinkB=2))


fig = plt.figure(figsize=(183 * MM, 178 * MM))
S = 12                                                # small labels

# ---------------------------------------------------------------- a schematic (full width)
ax = fig.add_axes([0.0, 0.5, 1.0, 0.46]); ax.set_xlim(0, 20); ax.set_ylim(0, 9.0); ax.set_aspect("equal"); ax.axis("off")
fig.text(0.005, 0.995, "a", fontsize=16, fontweight="bold", va="top")
tile(ax, 0.3, 4.1, 6.4, 4.8); tile(ax, 13.3, 4.1, 6.4, 4.8); tile(ax, 6.2, 0.15, 7.6, 3.75)
ax.text(3.5, 8.65, "United States and\nEuropean Union", ha="center", va="top", fontsize=FS, color=INK)
ax.text(3.5, 7.35, "palm excluded", ha="center", va="top", fontsize=S, color=RED, style="italic")
ax.text(16.5, 8.65, "Indonesia, Malaysia\nand Thailand", ha="center", va="top", fontsize=FS, color=INK)
ax.text(10.0, 3.7, "Importers in South Asia\nand sub-Saharan Africa", ha="center", va="top", fontsize=FS, color=INK)
drum(ax, 1.9, 4.75, w=1.0, h=1.35); ax.text(1.9, 4.5, "Fuel", ha="center", va="top", fontsize=S, color="#636363")
for dx in (0, 0.6):
    drop(ax, 4.35 + dx, 5.25, 0.22, BLUE)
ax.text(4.65, 4.85, "Soy, rapeseed", ha="center", va="top", fontsize=S, color=BLUE)
arrow(ax, (4.0, 5.55), (2.55, 5.55), BLUE)
drum(ax, 18.1, 4.75, w=1.0, h=1.35); ax.text(18.1, 4.5, "Fuel", ha="center", va="top", fontsize=S, color="#636363")
for dx in (0, 0.6):
    drop(ax, 15.05 + dx, 5.25, 0.22, ORANGE)
ax.text(15.35, 4.85, "Palm", ha="center", va="top", fontsize=S, color=ORANGE)
arrow(ax, (16.0, 5.55), (17.45, 5.55), ORANGE)
bowl(ax, 10.0, 1.75, r=0.95)
drop(ax, 9.55, 1.95, 0.2, BLUE); drop(ax, 10.45, 1.95, 0.2, ORANGE)
ax.text(10.0, 0.6, "Diets", ha="center", va="top", fontsize=S, color="#636363")
arrow(ax, (6.9, 7.45), (13.1, 7.45), RED, ls=(0, (4, 2)), lw=2.0)
ax.text(10.0, 7.65, "Relocate fuel production (±δ)", ha="center", va="bottom", fontsize=S, color=RED)
arrow(ax, (13.1, 6.15), (6.9, 6.15), RED, ls=(0, (4, 2)), lw=2.0)
ax.text(10.0, 5.95, "Admit palm in US and EU fuels", ha="center", va="top", fontsize=S, color=RED)
arrow(ax, (5.3, 4.0), (8.8, 1.95), BLUE, rad=0.2)
ax.text(4.1, 2.3, "Unsaturated oil\nmoves to diets", ha="center", va="center", fontsize=S, color=BLUE)
arrow(ax, (11.2, 1.95), (14.7, 4.0), ORANGE, rad=0.2)
ax.text(15.9, 2.3, "Palm moves\nto fuel", ha="center", va="center", fontsize=S, color=ORANGE)
marker(ax, 0.75, 6.4, 1); marker(ax, 11.6, 1.05, 2); marker(ax, 10.0, 6.8, 3, c=RED); marker(ax, 6.75, 3.35, 4)

# ---------------------------------------------------------------- b question cards (2 x 2) and exchange identity
bx = fig.add_axes([0.0, 0.0, 1.0, 0.47]); bx.set_xlim(0, 20); bx.set_ylim(0, 9.15); bx.set_aspect("equal"); bx.axis("off")
fig.text(0.005, 0.48, "b", fontsize=16, fontweight="bold", va="top")
cards = [(1, INK, "Fuel growth by oil type", "Additional gaps,\n2010–2023 ·\nFig. 2"),
         (2, INK, "The 2035 gap and its causes", "Gaps with and without biofuel;\nconstraint layers · Fig. 3"),
         (3, RED, "Two levers, one exchange", "Palm eligibility and ±δ relocation;\nfood-use data, land · Fig. 4"),
         (4, INK, "Who bears the gap", "Regional shares across allocations;\nother options · Figs. 5 and 6")]
pos = [(0.1, 6.35), (10.1, 6.35), (0.1, 3.55), (10.1, 3.55)]
for (n, c, head, body), (x0, y0) in zip(cards, pos):
    bx.add_patch(FancyBboxPatch((x0, y0), 9.8, 2.6, boxstyle="round,pad=0.02,rounding_size=0.25", fc="white", ec=GREY, lw=1.3))
    marker(bx, x0 + 0.55, y0 + 2.08, n, c=c)
    bx.text(x0 + 1.1, y0 + 2.08, head, ha="left", va="center", fontsize=FS, fontweight="bold", color=c)
    bx.text(x0 + 1.1, y0 + 1.55, body, ha="left", va="top", fontsize=S, color=INK, linespacing=1.2)
h = pd.read_csv(RES + "history_fuel_cost.csv").pivot(index="year", columns="target", values="biofuel_cost")
ix = bx.inset_axes([5.15, 6.6, 1.85, 1.25], transform=bx.transData)
ix.plot(h.index, h.eat_qty, color="#4D4D4D", lw=1.8); ix.plot(h.index, h.eat_quality, color=BLUE, lw=1.8)
ix.set_xticks([]); ix.set_yticks([]); ix.spines["top"].set_visible(False); ix.spines["right"].set_visible(False)
ix.text(1.04, 0.9, "Unsaturated", transform=ix.transAxes, fontsize=S, color=BLUE, va="center")
ix.text(1.04, 0.25, "Total oil", transform=ix.transAxes, fontsize=S, color="#4D4D4D", va="center")
bx.add_patch(FancyArrowPatch((10.0, 3.5), (10.0, 2.95), arrowstyle="-|>", mutation_scale=16, color=RED, lw=2))
bx.add_patch(FancyBboxPatch((0.1, 0.1), 19.8, 2.75, boxstyle="round,pad=0.02,rounding_size=0.25", fc="#FBE9E7", ec=RED, lw=1.6))
bx.text(0.6, 2.25, "Exchange at fixed supply", ha="left", va="center", fontsize=FS, fontweight="bold", color=RED)
bx.text(0.6, 1.3, r"$\Delta U_{\mathrm{food}} = -\Delta U_{\mathrm{fuel}} = \Delta P_{\mathrm{fuel}} = -\Delta P_{\mathrm{food}}$",
        ha="left", va="center", fontsize=FS + 1, color=INK)
bx.text(10.6, 1.3, "Unsaturated oil gained by diets equals\nthe palm that fuel takes from them",
        ha="left", va="center", fontsize=S, color=INK, linespacing=1.2)
save(fig, "Fig0")
table(h.reset_index()[["year", "eat_qty", "eat_quality"]].rename(columns={"eat_qty": "additional_gap_total_oil_Mt",
      "eat_quality": "additional_gap_unsaturated_Mt"}).assign(panel="b, card 1"), "Fig0")
