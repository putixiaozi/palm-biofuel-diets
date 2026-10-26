"""Master style for the vegetable-oil figures: Arial, >=13 pt, no titles, bold lowercase panel letters, one semantic
palette (oil types, uses, targets, decomposition terms, constraint layers, levers)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import paths
import os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

for f in ["Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf"]:
    p = "/System/Library/Fonts/Supplemental/" + f
    if os.path.exists(p): fm.fontManager.addfont(p)

ROOT = paths.REPO
DATA_ROOT = paths.REPO
RES = ROOT + "results/"
OUT = ROOT + "figures"
DATA = OUT + "/figdata"
MM = 1 / 25.4
W2 = 183 * MM                      # double-column width

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12, "legend.frameon": False, "legend.handlelength": 1.4, "legend.borderaxespad": 0.3,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8, "axes.titlesize": 13,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "axes.grid": False, "grid.color": "#E3E3E3", "grid.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "mathtext.fontset": "custom",
    "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.1,   # wide pad: tight bbox can miss ~1.5 mm of text
})

# oil types: unsaturated = blues, palm-type = oranges, recovered = teal
OIL = {"Soybean": "#2166AC", "Rapeseed": "#67A9CF", "Sunflower": "#92C5DE", "Other unsaturated": "#C6DBEF",
       "Palm": "#D95F02", "Palm kernel & coconut": "#FDB863", "UCO": "#17BECF"}
OIL_ORDER = ["Soybean", "Rapeseed", "Sunflower", "Other unsaturated", "Palm", "Palm kernel & coconut"]
CLASS = {"Unsaturated": "#2166AC", "Palm-type": "#D95F02"}
USE = {"food": "#4D9221", "biofuel": "#762A83", "nonfuel": "#BABABA"}
# food targets: quantity = greys, quality = blue
TGT = {"adequacy": "#969696", "eat_qty": "#4D4D4D", "eat_quality": "#2166AC"}
TGT_LAB = {"adequacy": "Fat availability", "eat_qty": "EAT oil quantity", "eat_quality": "EAT unsaturated oil"}
ACCESS_LS = {"constrained": "-", "pool": "--"}
DEC = {"scarcity": "#BDBDBD", "network": "#F4A582", "biofuel": "#762A83", "interaction": "#C2A5CF"}
DEC_LAB = {"scarcity": "Scarcity without biofuel", "network": "Network constraints", "biofuel": "Biofuel alone",
           "interaction": "Biofuel \u00d7 network"}
LAYER = {"L1": "#A6761D", "L2": "#E6AB02", "L3": "#7570B3", "L4": "#E7298A", "L5": "#66A61E"}
LAYER_LAB = {"L1": "Output +10%", "L2": "Trade access", "L3": "Conversion location", "L4": "Feedstock eligibility",
             "L5": "Rigid demand"}
LEVER = {"S1": "#17BECF", "S2": "#A6761D", "S3": "#E6AB02", "S4": "#66A61E"}
LEVER_LAB = {"S1": "Used cooking oil", "S2": "Output +10%", "S3": "Trade access", "S4": "Diet convergence"}
INK = "#222222"


def panel(ax, letter, x=-0.02, y=1.0):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="right")


def fit_width(fig, target=W2, n=3):
    """rescale the canvas width so the tight bounding box is exactly the journal column width (font sizes unchanged)"""
    for _ in range(n):
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        w, h = fig.get_size_inches()
        fig.set_size_inches(w + (target - (bb.width + 2 * mpl.rcParams["savefig.pad_inches"])), h)


def save(fig, name, sub="", width=W2):
    out = OUT + ("/" + sub if sub else ""); os.makedirs(out + "/check", exist_ok=True); fit_width(fig, width)
    bb, pad = fig.get_tightbbox(fig.canvas.get_renderer()), 2 * mpl.rcParams["savefig.pad_inches"]
    print(f"{name}: {(bb.width + pad) * 25.4:.0f} x {(bb.height + pad) * 25.4:.0f} mm")
    for ext, kw in ([] if os.environ.get("FIGFAST") else [("png", dict(dpi=1000)), ("pdf", {}), ("svg", {})]):   # FIGFAST: proof only
        fig.savefig(f"{out}/{name}.{ext}", **kw)
    fig.savefig(f"{out}/check/{name}_check.png", dpi=150)   # 100%-zoom proof at screen resolution
    plt.close(fig)


def table(df, name, sub=""):
    d = DATA + ("/" + sub if sub else ""); os.makedirs(d, exist_ok=True)
    df.to_csv(f"{d}/{name}.csv", index=False)


# ---------------------------------------------------------------- world map (Equal Earth; standard-map boundaries incl. disputed lines)
MAPCACHE = DATA_ROOT + "data_derived/map/world_equal_earth.pkl"   # built from nature1/China_and_World_Map_shapefiles


def world_map():
    import pickle
    return pickle.load(open(MAPCACHE, "rb"))


def draw_world(ax, values, cmap, norm, M=None, missing="#E6E6E6", edge="white"):
    """values: Series indexed by ISO3. Countries without data are light grey; Antarctica is not drawn."""
    import pandas as pd
    M = M or world_map()
    ax.fill(*M["outline"], color="#F7F9FB", lw=0, zorder=0)
    cs = M["countries"]; v = cs.iso3c.map(values)
    fc = [cmap(norm(x)) if pd.notna(x) else mpl.colors.to_rgba(missing) for x in v]
    cs.plot(ax=ax, color=fc, edgecolor=edge, linewidth=0.15, zorder=1)
    M["disputed"].plot(ax=ax, color="#4D4D4D", linewidth=0.6, zorder=3)
    ax.plot(*M["outline"], color="#BDBDBD", lw=0.6, zorder=0)
    ax.set_xlim(-2.72, 2.72); ax.set_ylim(-1.0, 1.32); ax.set_aspect("equal"); ax.axis("off")
    return M
