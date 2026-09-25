"""
Paper 2 (MBEC) - Figures 1-5 and graphical abstract, regenerated from the result tables.

Inputs (all in DATA):
  audit_cells_full.csv, gt_shape.csv, sickle_fraction_error.csv, sickle_fraction_perslide.csv,
  bm_ridge_ratio.csv, bm_alpha_summary.csv
Outputs (in OUT): FigN.pdf / .eps / .tiff (600 dpi) / .png, Graphical_abstract.*

Run locally or on Kaggle:  python paper2_figures.py  (edit DATA / OUT below)
Fig. 6 needs the TIFF fields and masks, so it is produced by the separate Kaggle cell.
"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import stats
from scipy.ndimage import distance_transform_edt
from skimage.measure import label as sklabel
from skimage.segmentation import watershed
from PIL import Image
import statsmodels.api as sm

DATA = os.environ.get("P2_DATA", "/kaggle/working")
OUT = os.environ.get("P2_OUT", "/kaggle/working/paper2_figures")
os.makedirs(OUT, exist_ok=True)

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "legend.frameon": False})
# Okabe-Ito
C = dict(rel="#D55E00", fa="#E69F00", fb="#0072B2", fc="#009E73", cp="#000000",
         neg="#0072B2", pos="#D55E00", grey="#999999", iso="#BBBBBB", touch="#D55E00")
MLAB = {"original": "Released", "fix_A": "fix$_A$", "fix_B": "fix$_B$",
        "fix_C": "fix$_C$", "cellpose_sam": "Cellpose-SAM"}

def save(fig, name):
    for ext in ("pdf", "eps"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", dpi=600, bbox_inches="tight")
    Image.open(f"{OUT}/{name}.png").convert("RGB").save(
        f"{OUT}/{name}.tiff", compression="tiff_lzw", dpi=(600, 600))
    plt.close(fig); print("wrote", name)

def panel(ax, l): ax.set_title(l, loc="left", fontweight="bold", fontsize=10)

cells = pd.read_csv(f"{DATA}/audit_cells_full.csv")
shape = pd.read_csv(f"{DATA}/gt_shape.csv")
m = cells.merge(shape, on=["slide", "inst"], suffixes=("", "_s"))

# =====================================================================
# FIG 1  rationale (a) and illustrative mechanism (b)
# =====================================================================
def box(ax, x, y, w, h, txt, fc="white", ec="#555555", fs=7, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.25)

def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                                 lw=0.9, color="#333333"))

fig = plt.figure(figsize=(7.2, 5.6))
gs = fig.add_gridspec(3, 4, height_ratios=[0.62, 1, 1], hspace=0.38, wspace=0.42)
axA = fig.add_subplot(gs[0, :]); axA.set_xlim(0, 10); axA.set_ylim(0.1, 2.0); axA.axis("off")
panel(axA, "a")
xs = [0.05, 2.05, 4.35, 6.25, 8.3]; w = [1.7, 2.0, 1.6, 1.75, 1.65]
labels = ["Thin blood film\nfield", "Segmentation\n(distance transform,\nseeds > 0.3 × field max,\nwatershed, area filter)",
          "Crop & rescale\n128 × 128 px", "Classifier\nsickled / normal", "Sickle fraction\n(diagnostic output)"]
fcs = ["#F4F4F4", "#FBE3D6", "#F4F4F4", "#DCEBF7", "#F4F4F4"]
for x, ww, t, f in zip(xs, w, labels, fcs):
    box(axA, x, 0.55, ww, 1.25, t, fc=f)
for i in range(4):
    arrow(axA, xs[i] + w[i] + 0.02, 1.17, xs[i + 1] - 0.03, 1.17)
axA.text(xs[1] + w[1] / 2, 0.25, "audited here: never evaluated at instance level",
         ha="center", fontsize=6.5, color=C["rel"], style="italic")
axA.text(xs[3] + w[3] / 2, 0.25, "published: precision 0.86, recall 0.69",
         ha="center", fontsize=6.5, color=C["fb"], style="italic")

# illustrative synthetic fields
H, W = 140, 260
yy, xx = np.mgrid[0:H, 0:W]
def disk(cx, cy, r): return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
def ell(cx, cy, a, b): return ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1
rows = [("round pair", disk(88, 70, 40) | disk(166, 70, 40)),
        ("elongated + round", ell(80, 70, 66, 22) | disk(180, 70, 40))]
FIELD_MAX = 110.0; T = 0.3 * FIELD_MAX        # illustrative field-wide maximum elsewhere in the image
CEIL = 1.6 * np.pi * 40 ** 2                  # illustrative area ceiling (1.6 × one round cell)
for r, (name, mask) in enumerate(rows, start=1):
    dt = distance_transform_edt(mask)
    seeds = sklabel(dt > T)
    lab = watershed(-dt, seeds, mask=mask)
    ax = fig.add_subplot(gs[r, 0]); ax.imshow(mask, cmap="Greys", vmin=0, vmax=1.6); ax.axis("off")
    ax.text(-8, H / 2, name, rotation=90, ha="right", va="center", fontsize=7)
    if r == 1: ax.set_title("cells", fontsize=7.5, pad=2)
    ax = fig.add_subplot(gs[r, 1]); ax.imshow(dt, cmap="magma"); ax.axis("off")
    ax.axhline(70, color="w", lw=0.6, ls="--")
    if r == 1: ax.set_title("distance transform", fontsize=7.5, pad=2)
    ax = fig.add_subplot(gs[r, 2]); ax.plot(dt[70], color="k", lw=1)
    ax.axhline(T, color=C["rel"], lw=1, ls="--"); ax.set_ylim(0, 50); ax.set_xticks([]); ax.tick_params(labelsize=6.5)
    ax.text(2, T + 1.5, "0.3 × field max", ha="left", fontsize=6, color=C["rel"])
    ax.set_ylabel("DT (px)", fontsize=7)
    if r == 1: ax.set_title("profile through centres", fontsize=7.5, pad=2)
    ax = fig.add_subplot(gs[r, 3]); ax.axis("off")
    rgb = np.ones((H, W, 3))
    n_seed = seeds.max(); regions = [lab == i for i in range(1, lab.max() + 1)]
    kept_any = False
    for reg in regions:
        a = reg.sum()
        col = (0.0, 0.62, 0.45) if a < CEIL else (0.84, 0.37, 0.0)
        rgb[reg] = col; kept_any |= a < CEIL
    rgb[seeds > 0] = (1, 1, 1)
    ax.imshow(rgb)
    fate = ("2 seeds: both cells retained" if n_seed >= 2
            else "1 seed: basins merge, region > area\nceiling, both cells deleted")
    ax.text(W / 2, H + 18, fate, ha="center", va="top", fontsize=6.5,
            color=(C["fc"] if n_seed >= 2 else C["rel"]))
    if r == 1: ax.set_title("seeds and fate", fontsize=7.5, pad=2)
fig.text(0.075, 0.655, "b", fontweight="bold", fontsize=10)
fig.text(0.99, 0.01, "Panel b is illustrative, not measured", ha="right", fontsize=6, color="grey")
save(fig, "Fig1")

# =====================================================================
# FIG 2  recall by contact
# =====================================================================
iso_touch = {}
for meth in ["original", "fix_A", "fix_B", "cellpose_sam"]:
    g = cells[cells.method == meth]
    iso_touch[meth] = (100 * g[g.contact <= 0.01]["rec@0.5"].mean(),
                       100 * g[g.contact > 0.01]["rec@0.5"].mean())
fixc_file = f"{DATA}/fixC_h9_percell.csv"
if os.path.exists(fixc_file) and "contact" in pd.read_csv(fixc_file, nrows=1).columns:
    fcx = pd.read_csv(fixc_file)
    iso_touch["fix_C"] = (100 * fcx[fcx.contact <= 0.01]["rec@0.5"].mean(),
                          100 * fcx[fcx.contact > 0.01]["rec@0.5"].mean())
else:
    iso_touch["fix_C"] = (86.8, 51.2)       # h = 9, recorded 3 Sep 2026
order = ["original", "fix_A", "fix_B", "fix_C", "cellpose_sam"]
fig, ax = plt.subplots(figsize=(4.2, 2.6))
x = np.arange(len(order)); wd = 0.36
iv = [iso_touch[k][0] for k in order]; tv = [iso_touch[k][1] for k in order]
b1 = ax.bar(x - wd / 2, iv, wd, color=C["iso"], ec="k", lw=0.5, label="Isolated (n = 3,039)")
b2 = ax.bar(x + wd / 2, tv, wd, color=C["touch"], ec="k", lw=0.5, label="Touching (n = 723)")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{b.get_height():.0f}",
                ha="center", fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels([MLAB[k] for k in order], fontsize=7.5)
ax.set_ylabel("Recall at IoU 0.5 (%)"); ax.set_ylim(0, 110); ax.legend(fontsize=7, loc="upper left", ncol=2)
save(fig, "Fig2")

# =====================================================================
# FIG 3  discarded vs retained cells (released pipeline)
# =====================================================================
o = m[(m.method == "original") & m.fate.isin(["deleted_oversize", "matched"])]
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.4))
for ax, col, lab, l in zip(axs, ["area", "ecc", "extent"],
                           ["Area (px)", "Eccentricity", "Extent"], "abc"):
    d0 = o[o.fate == "matched"][col]; d1 = o[o.fate == "deleted_oversize"][col]
    vp = ax.violinplot([d0, d1], showextrema=False, widths=0.8)
    for body, cc in zip(vp["bodies"], [C["fb"], C["rel"]]):
        body.set_facecolor(cc); body.set_edgecolor("k"); body.set_linewidth(0.4); body.set_alpha(0.6)
    for i, d in enumerate([d0, d1], 1):
        ax.hlines(np.median(d), i - 0.22, i + 0.22, color="k", lw=1.5)
    per = o.groupby(["slide", "fate"])[col].median().unstack().dropna()
    p = stats.wilcoxon(per["deleted_oversize"], per["matched"]).pvalue
    ptxt = f"$P$ = {p:.2f}" if p >= 0.01 else f"$P$ = {p:.1e}".replace("e-0", "×10$^{-").replace("e-", "×10$^{-") + "}$"
    ax.set_title(ptxt, fontsize=7.5)
    ax.set_xticks([1, 2]); ax.set_xticklabels([f"retained\n(n = {len(d0):,})", f"deleted oversize\n(n = {len(d1):,})"], fontsize=6.8)
    ax.set_ylabel(lab); panel(ax, l); ax.title.set_position((0.5, 1.0))
    if col == "area":
        ax.axhline(17000, color="grey", ls=":", lw=1); ax.text(0.45, 17400, "17,000 px ceiling", ha="left", fontsize=6, color="grey")
axs[0].set_title("a", loc="left", fontweight="bold", fontsize=10)
axs[1].set_title("b", loc="left", fontweight="bold", fontsize=10)
axs[2].set_title("c", loc="left", fontweight="bold", fontsize=10)
fig.tight_layout(w_pad=1.5)
save(fig, "Fig3")

# =====================================================================
# FIG 4  sickle-fraction error
# =====================================================================
se = pd.read_csv(f"{DATA}/sickle_fraction_error.csv")
ps = pd.read_csv(f"{DATA}/sickle_fraction_perslide.csv")
fixc_err = {0.65: -3.24, 0.70: -3.37, 0.75: -3.02}     # h = 9, run of 25 Sep 2026
fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
ax = axs[0]
for meth, cc in [("original", C["rel"]), ("fix_B", C["fb"]), ("cellpose_sam", C["cp"])]:
    g = se[se.method == meth]; ax.plot(g.thr, g.err, "-o", color=cc, ms=3.5, lw=1.2, label=MLAB[meth])
ax.plot(list(fixc_err), list(fixc_err.values()), "-o", color=C["fc"], ms=3.5, lw=1.2, label="fix$_C$ ($h$ = 9)")
ax.axhline(0, color="grey", lw=0.8); ax.set_xlabel("Eccentricity threshold for 'sickled'")
ax.set_ylabel("Error in sickle fraction (pp)"); ax.legend(fontsize=6.5, loc="upper center", ncol=2); panel(ax, "a")
ax.set_ylim(-7.2, 1.9)
ax = axs[1]
ax.scatter(ps.true, ps.rep, s=14, color=C["rel"], zorder=3)
lim = [10, 55]; ax.plot(lim, lim, "k--", lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("True sickle fraction (%)"); ax.set_ylabel("Reported, released pipeline (%)")
ax.text(12, 51, f"{(ps.d < 0).sum()} of {len(ps)} slides under-reported\nmean {ps.d.mean():+.1f} pp", fontsize=6.5)
panel(ax, "b"); fig.tight_layout(w_pad=2)
save(fig, "Fig4")

# =====================================================================
# FIG 5  phenotype-aligned disparity and mechanism
# =====================================================================
d = pd.read_csv(f"{DATA}/bm_ridge_ratio.csv"); a = pd.read_csv(f"{DATA}/bm_alpha_summary.csv")
fig, ax = plt.subplots(1, 4, figsize=(7.5, 2.35), gridspec_kw=dict(width_ratios=[1, 1.1, 1.1, 0.9]))
rng = np.random.default_rng(3)
for xpos, (k, cc) in enumerate([(0, C["neg"]), (1, C["pos"])]):
    y = d[d.pos == k].recall
    ax[0].scatter(xpos + rng.uniform(-.12, .12, len(y)), y, s=16, color=cc, zorder=3)
    ax[0].hlines(y.mean(), xpos - .22, xpos + .22, color=cc, lw=2)
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["SCD−\n(n = 9)", "SCD+\n(n = 15)"], fontsize=7)
ax[0].set_xlim(-.5, 1.5); ax[0].set_ylabel("Per-slide recall at IoU 0.5 (%)")
ax[0].text(.35, 79, "$P$ = 0.020", fontsize=7)
fit = sm.OLS(d.recall, sm.add_constant(d.ratio)).fit(); xx_ = np.linspace(.25, .82, 50)
ax[1].plot(xx_, fit.params.iloc[0] + fit.params.iloc[1] * xx_, "k", lw=1)
for k, cc, l in [(0, C["neg"], "SCD−"), (1, C["pos"], "SCD+")]:
    s = d[d.pos == k]; ax[1].scatter(s.ratio, s.recall, s=12, color=cc, label=l, zorder=3)
ax[1].axvline(.30, color="grey", ls=":", lw=1); ax[1].text(.31, 15.5, "seed thr.", fontsize=6, color="grey")
ax[1].text(.26, 73, f"$R^2$ = {fit.rsquared:.2f}\nclass $P$ = 0.35", fontsize=7)
ax[1].set_xlabel("Ridge / field max"); ax[1].set_ylabel("Per-slide recall (%)"); ax[1].legend(fontsize=6.5, loc="lower right")
ax[2].plot(a.alpha, a.neg, "-o", color=C["neg"], ms=3.5, label="SCD−")
ax[2].plot(a.alpha, a.pos, "-o", color=C["pos"], ms=3.5, label="SCD+")
ax[2].axvline(.30, color="grey", ls=":", lw=1); ax[2].text(.305, 21, "released", fontsize=6, color="grey")
ax[2].set_xlabel("Seed coefficient $\\alpha$"); ax[2].set_ylabel("Recall (%)"); ax[2].legend(fontsize=6.5, loc="lower left")
vals, pv = [18.8, 5.4, 4.6], ["0.020", "0.37", "0.12"]
ax[3].bar([0, 1, 2], vals, color=[C["rel"], C["fb"], C["fc"]], width=.65)
for i, (v, p) in enumerate(zip(vals, pv)): ax[3].text(i, v + .4, f"{v}\n$P$ = {p}", ha="center", fontsize=6.5)
ax[3].set_xticks([0, 1, 2]); ax[3].set_xticklabels(["Released", "fix$_B$", "fix$_C$"], fontsize=7)
ax[3].set_ylim(0, 24); ax[3].set_ylabel("SCD− minus SCD+ (pp)")
for i, l in enumerate("abcd"): panel(ax[i], l)
fig.tight_layout(w_pad=1.2)
save(fig, "Fig5")

# =====================================================================
# GRAPHICAL ABSTRACT  (aspect 32.93 : 37.63, width : height; vector PDF scales freely)
# =====================================================================
fig = plt.figure(figsize=(3.293 * 1.6, 3.763 * 1.6))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 11.43); ax.axis("off")
ax.text(5, 11.02, "Segmentation errors that track the disease", ha="center", fontsize=10.5, fontweight="bold")
def gbox(y, h, title, fc):
    ax.add_patch(FancyBboxPatch((0.3, y), 9.4, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec="#777777", lw=0.8))
    ax.text(0.6, y + h - 0.28, title, fontsize=8.5, fontweight="bold", va="top")
def ins(x, y, w, h): return ax.inset_axes([x, y, w, h], transform=ax.transData)
# 1 -- pipeline
gbox(8.25, 2.4, "1  Segmentation is the unevaluated first stage", "#F4F4F4")
for k, (xx, t, fc) in enumerate([(0.7, "blood film\nfield", "white"), (3.55, "segmentation", "#FBE3D6"),
                                 (6.4, "classifier", "#DCEBF7")]):
    ax.add_patch(FancyBboxPatch((xx, 8.75), 2.3, 1.05, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec="#555555", lw=0.7))
    ax.text(xx + 1.15, 9.27, t, ha="center", va="center", fontsize=7.5)
    if k < 2: ax.add_patch(FancyArrowPatch((xx + 2.33, 9.27), (xx + 2.82, 9.27), arrowstyle="-|>", mutation_scale=8, color="#333333"))
ax.text(4.7, 8.45, "cells it misses never reach the classifier", ha="center", fontsize=6.8, style="italic", color=C["rel"])
# 2 -- disparity
gbox(4.45, 3.55, "2  It loses more cells from sickle cell films", "#FBE3D6")
b = ins(1.2, 5.25, 4.0, 2.0)
b.bar([0, 1], [62.3, 43.5], color=[C["neg"], C["pos"]], width=0.6)
for i2, v in enumerate([62.3, 43.5]): b.text(i2, v + 3, f"{v}%", ha="center", fontsize=7.5, fontweight="bold")
b.set_xticks([0, 1]); b.set_xticklabels(["control", "sickle cell"], fontsize=7); b.set_ylim(0, 82)
b.set_yticks([]); b.spines["left"].set_visible(False); b.patch.set_alpha(0)
b.set_title("cells recovered", fontsize=7, pad=1)
ax.text(5.6, 6.75, "Cause:", fontsize=7.5, fontweight="bold")
ax.text(5.6, 6.35, "seed threshold scaled\nto the largest structure\nin the whole field", fontsize=7.2, va="top", linespacing=1.3)
ax.text(5.6, 5.2, "explains ~2/3 of the gap", fontsize=7, style="italic")
# 3 -- consequence and repair
gbox(0.3, 3.85, "3  Consequence and repair", "#DDF1EA")
c1 = ins(1.0, 1.35, 3.4, 1.85)
c1.bar([0, 1], [30.9, 25.6], color=[C["grey"], C["rel"]], width=0.6)
for i2, v in enumerate([30.9, 25.6]): c1.text(i2, v + 1.5, f"{v}%", ha="center", fontsize=7.5, fontweight="bold")
c1.set_xticks([0, 1]); c1.set_xticklabels(["true", "reported"], fontsize=7); c1.set_ylim(0, 40)
c1.set_yticks([]); c1.spines["left"].set_visible(False); c1.patch.set_alpha(0)
c1.set_title("sickle fraction: 17% under-count", fontsize=7, pad=1)
c2 = ins(5.6, 1.35, 3.4, 1.85)
c2.bar([0, 1], [18.8, 4.6], color=[C["rel"], C["fc"]], width=0.6)
for i2, v in enumerate([18.8, 4.6]): c2.text(i2, v + 0.8, f"{v}", ha="center", fontsize=7.5, fontweight="bold")
c2.set_xticks([0, 1]); c2.set_xticklabels(["released", "fixed seeding"], fontsize=7); c2.set_ylim(0, 24)
c2.set_yticks([]); c2.spines["left"].set_visible(False); c2.patch.set_alpha(0)
c2.set_title("class gap (points)", fontsize=7, pad=1)
ax.text(5, 0.5, "Evaluate segmentation for attribute-independent error, not aggregate recall",
        ha="center", fontsize=6.6, style="italic")
for y0, y1 in [(8.22, 8.03), (4.42, 4.2)]:
    ax.add_patch(FancyArrowPatch((5, y0), (5, y1), arrowstyle="-|>", mutation_scale=10, color="#333333"))
save(fig, "Graphical_abstract")
print("done ->", OUT)
