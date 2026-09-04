#!/usr/bin/env python3

from pathlib import Path
import glob
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(".")
OUT = ROOT / "fig" / "paperA_final_v2"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save(fig, name, dpi=500):
    fig.tight_layout()
    pdf = OUT / f"{name}.pdf"
    png = OUT / f"{name}.png"

    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print("WROTE:", pdf)
    print("WROTE:", png)


# ============================================================
# FIGURE 1
# CROSS-DATASET STRUCTURAL CORRELATION HEATMAP
# ============================================================

DATASETS = {
    "WN18RR": ROOT / "runs/wn18rr/correlation_summary.csv",
    "FB15k-237": ROOT / "runs/fb15k237/correlation_summary.csv",
    "YAGO11k": ROOT / "runs/yago/correlation_summary.csv",
    "WIKIDATA12k": ROOT / "runs/wikidata/correlation_summary.csv",
    "ICEWS14": ROOT / "runs/icews14/correlation_summary.csv",
    "GDELT": ROOT / "runs/gdelt/correlation_summary.csv",
}

DISPLAY = {
    "forman_ricci": "Forman–Ricci",
    "ollivier_rc_exact": "Ollivier–Ricci (exact)",
    "ollivier_rc": "Ollivier–Ricci (approx.)",
    "growth_rate": "Growth rate",
    "saturation_r": "Saturation radius",
    "branch_persist": "Branch persistence",
    "static_fanout": "Static fan-out",
    "temporal_fanout": "Temporal fan-out",
    "tail_fanin": "Tail fan-in",
    "grc": "Global reaching centrality",
    "longest_path": "Longest path",
    "reciprocity": "Reciprocity",
    "clustering": "Clustering",
    "beta1": "Cycle rank",
    "delta_hyp": r"$\delta$-hyperbolicity",
    "head_tail_ratio": "Head/tail ratio",
    "tail_entropy": "Tail entropy",
    "one_to_one_frac": "1-to-1 fraction",
    "powerlaw_gamma": "Power-law exponent",
}

ORDER = [
    "forman_ricci",
    "ollivier_rc_exact",
    "ollivier_rc",
    "growth_rate",
    "saturation_r",
    "branch_persist",
    "static_fanout",
    "temporal_fanout",
    "tail_fanin",
    "grc",
    "longest_path",
    "reciprocity",
    "clustering",
    "beta1",
    "delta_hyp",
    "head_tail_ratio",
    "tail_entropy",
    "one_to_one_frac",
    "powerlaw_gamma",
]

matrix = pd.DataFrame(
    index=ORDER,
    columns=list(DATASETS.keys()),
    dtype=float
)

for ds, path in DATASETS.items():

    if not path.exists():
        print("MISSING:", path)
        continue

    df = pd.read_csv(path)

    for _, row in df.iterrows():

        cov = row["covariate"]

        if cov in matrix.index:
            matrix.loc[cov, ds] = row["rho_pooled"]


fig, ax = plt.subplots(figsize=(9.4, 8.5))

arr = matrix.values.astype(float)
masked = np.ma.masked_invalid(arr)

cmap = plt.get_cmap("coolwarm").copy()
cmap.set_bad("0.90")

im = ax.imshow(
    masked,
    aspect="auto",
    vmin=-1,
    vmax=1,
    cmap=cmap
)

ax.set_xticks(np.arange(len(matrix.columns)))
ax.set_xticklabels(
    matrix.columns,
    rotation=35,
    ha="right"
)

ax.set_yticks(np.arange(len(matrix.index)))
ax.set_yticklabels([
    DISPLAY.get(x, x)
    for x in matrix.index
])

for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):

        if np.isfinite(arr[i, j]):

            ax.text(
                j,
                i,
                f"{arr[i,j]:.2f}",
                ha="center",
                va="center",
                fontsize=7
            )

        else:
            ax.text(
                j,
                i,
                "NA",
                ha="center",
                va="center",
                fontsize=7
            )

cbar = fig.colorbar(
    im,
    ax=ax,
    shrink=0.82
)

cbar.set_label(r"Spearman $\rho$")

ax.set_title(
    "Structural correlates of learned relation-specific curvature"
)

save(fig, "fig1_cross_dataset_heatmap")


# ============================================================
# FIGURE 2
# CROSS-SEED REPRODUCIBILITY
# ============================================================

def load_seed_curvatures(run_dir):

    vectors = {}

    for fp in sorted(
        Path(run_dir).glob("curvatures_seed*.csv")
    ):

        m = re.search(
            r"seed(\d+)",
            fp.name
        )

        if not m:
            continue

        seed = int(m.group(1))

        df = pd.read_csv(fp)

        if (
            "relation_id" not in df.columns
            or "curvature" not in df.columns
        ):
            continue

        vectors[seed] = dict(
            zip(
                df["relation_id"].astype(int),
                df["curvature"].astype(float)
            )
        )

    return vectors


def pairwise_seed_rho(run_dir):

    vecs = load_seed_curvatures(run_dir)

    seeds = sorted(vecs)
    vals = []

    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):

            a = seeds[i]
            b = seeds[j]

            common = sorted(
                set(vecs[a]) &
                set(vecs[b])
            )

            if len(common) < 3:
                continue

            x = [
                vecs[a][r]
                for r in common
            ]

            y = [
                vecs[b][r]
                for r in common
            ]

            rho, _ = spearmanr(x, y)

            vals.append(rho)

    return vals


RUNS = {
    "WN18RR": "runs/wn18rr",
    "FB15k-237": "runs/fb15k237",
    "YAGO11k": "runs/yago",
    "WIKIDATA12k": "runs/wikidata",
    "ICEWS14": "runs/icews14",
    "GDELT": "runs/gdelt",
}

rows = []

for ds, run in RUNS.items():

    vals = pairwise_seed_rho(run)

    if not vals:
        continue

    rows.append({
        "dataset": ds,
        "mean": np.mean(vals),
        "min": np.min(vals),
        "max": np.max(vals),
        "n_pairs": len(vals),
    })

rep = pd.DataFrame(rows)

fig, ax = plt.subplots(
    figsize=(7.6, 4.5)
)

x = np.arange(len(rep))

lower = rep["mean"] - rep["min"]
upper = rep["max"] - rep["mean"]

ax.errorbar(
    x,
    rep["mean"],
    yerr=[
        lower,
        upper
    ],
    fmt="o",
    markersize=6,
    capsize=4,
    linewidth=1.5
)

ax.set_xticks(x)
ax.set_xticklabels(
    rep["dataset"],
    rotation=30,
    ha="right"
)

ax.set_ylim(0, 1.03)

ax.set_ylabel(
    r"Pairwise seed Spearman $\rho$"
)

ax.set_title(
    "Learned curvature is reproducible across random seeds"
)

for i, v in enumerate(rep["mean"]):

    ax.text(
        i,
        min(v + 0.045, 0.995),
        f"{v:.3f}",
        ha="center",
        fontsize=8
    )

ax.text(
    0.01,
    0.03,
    "Error bars: minimum–maximum pairwise seed agreement",
    transform=ax.transAxes,
    fontsize=8
)

save(fig, "fig2_curvature_reproducibility")


# ============================================================
# FIGURE 3
# SYNTHETIC STRUCTURAL DISSOCIATION
# ============================================================

profile = pd.read_csv(
    ROOT / "data/synthetic/relation_profile.csv"
)

seed_files = sorted(
    glob.glob(
        "runs/synthetic/curvatures_seed*.csv"
    )
)

curv_rows = []

for fp in seed_files:

    m = re.search(
        r"seed(\d+)",
        fp
    )

    seed = int(m.group(1))

    df = pd.read_csv(fp)[[
        "relation_id",
        "curvature"
    ]].copy()

    df["seed"] = seed

    curv_rows.append(df)

curv = pd.concat(
    curv_rows,
    ignore_index=True
)

syn = curv.merge(
    profile,
    on="relation_id"
)

agg = (
    syn.groupby(
        [
            "relation_id",
            "family",
            "branching",
            "depth",
            "cross_frac"
        ],
        as_index=False
    )
    .agg(
        mean_curvature=(
            "curvature",
            "mean"
        ),
        sd_curvature=(
            "curvature",
            "std"
        )
    )
)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(11.5, 3.9),
    sharey=True
)

COMMON_YMIN = 0.55
COMMON_YMAX = 1.38


# Panel A — branching

z = (
    agg[
        agg["family"]
        .str.startswith("branch_b")
    ]
    .sort_values("branching")
)

axes[0].errorbar(
    z["branching"],
    z["mean_curvature"],
    yerr=z["sd_curvature"],
    marker="o",
    capsize=3
)

axes[0].set_xscale(
    "log",
    base=2
)

axes[0].set_xticks([
    1,
    2,
    4,
    8,
    16
])

axes[0].set_xticklabels([
    "1",
    "2",
    "4",
    "8",
    "16"
])

axes[0].set_xlabel(
    "Branching factor"
)

axes[0].set_ylabel(
    "Learned curvature"
)

axes[0].set_title(
    "(a) Branching varied"
)


# Panel B — depth

z = (
    agg[
        agg["family"]
        .str.startswith("depth_d")
    ]
    .sort_values("depth")
)

axes[1].errorbar(
    z["depth"],
    z["mean_curvature"],
    yerr=z["sd_curvature"],
    marker="o",
    capsize=3
)

axes[1].set_xscale(
    "log",
    base=2
)

axes[1].set_xticks([
    2,
    4,
    8,
    16
])

axes[1].set_xticklabels([
    "2",
    "4",
    "8",
    "16"
])

axes[1].set_xlabel(
    "Chain depth"
)

axes[1].set_title(
    "(b) Depth varied"
)


# Panel C — cross-edge contamination

z = (
    agg[
        agg["family"]
        .str.startswith("loopy_p")
    ]
    .sort_values("cross_frac")
)

axes[2].errorbar(
    z["cross_frac"],
    z["mean_curvature"],
    yerr=z["sd_curvature"],
    marker="o",
    capsize=3
)

axes[2].set_xlabel(
    "Cross-edge fraction"
)

axes[2].set_title(
    "(c) Cross-edge contamination varied"
)


for ax in axes:

    ax.set_ylim(
        COMMON_YMIN,
        COMMON_YMAX
    )

    ax.grid(
        axis="y",
        alpha=0.20
    )


fig.suptitle(
    "Controlled structural dissociation",
    y=1.03
)

save(fig, "fig3_structural_dissociation")


# ============================================================
# FIGURE 4
# PLANTED HIERARCHY UNDER STRUCTURAL NOISE
# ============================================================

with open(
    "runs/planted_hierarchy/summary.json"
) as f:

    planted = json.load(f)

rows = []

for _, obj in planted.items():

    row = {
        "noise": obj["noise_rate"],
        "density": obj["density"]
    }

    for feat in [
        "forman_ricci",
        "growth_rate",
        "saturation_r",
        "branch_persist"
    ]:

        v = obj.get(
            feat,
            {}
        )

        row[feat] = (
            v.get("rho")
            if isinstance(v, dict)
            else np.nan
        )

    rows.append(row)

ph = (
    pd.DataFrame(rows)
    .sort_values("noise")
)

labels = {
    "forman_ricci": "Forman–Ricci",
    "growth_rate": "Growth rate",
    "saturation_r": "Saturation radius",
    "branch_persist": "Branch persistence",
}

fig, ax = plt.subplots(
    figsize=(7.6, 4.8)
)

for feat, label in labels.items():

    ax.plot(
        ph["noise"],
        ph[feat],
        marker="o",
        linewidth=1.8,
        label=label
    )

ax.axhline(
    0,
    linewidth=1,
    linestyle="--"
)

ax.set_xscale(
    "symlog",
    linthresh=0.1
)

noise_ticks = [
    0,
    0.1,
    0.25,
    0.5,
    1,
    2,
    4,
    8,
    16
]

ax.set_xticks(noise_ticks)

ax.set_xticklabels([
    "0",
    "0.1",
    "0.25",
    "0.5",
    "1",
    "2",
    "4",
    "8",
    "16"
])

ax.set_xlabel(
    "Noise edges / tree edges"
)

ax.set_ylabel(
    r"Spearman $\rho$ with planted depth"
)

ax.set_ylim(
    -1.08,
    1.08
)

ax.legend(
    frameon=False,
    ncol=2
)

ax.set_title(
    "Structural proxies diverge under hierarchy corruption"
)

save(fig, "fig4_planted_hierarchy_noise")


# ============================================================
# FIGURE 5
# DEGREE-PRESERVING STRUCTURAL NULL
# ============================================================

REWIRE_FEATURES = [
    "forman_ricci",
    "growth_rate",
    "branch_persist",
    "tail_fanin",
    "static_fanout",
]

REWIRE_LABELS = {
    "forman_ricci": "Forman",
    "growth_rate": "Growth",
    "branch_persist": "Branch persistence",
    "tail_fanin": "Tail fan-in",
    "static_fanout": "Static fan-out",
}

real_paths = {
    "WN18RR":
        "runs/wn18rr/correlation_summary.csv",

    "YAGO11k":
        "runs/yago/correlation_summary.csv",
}

rewired_paths = {
    "WN18RR":
        "runs/wn18rr_rewired/correlation_summary.csv",

    "YAGO11k":
        "runs/yago_rewired/correlation_summary.csv",
}

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.7, 4.6),
    sharey=True
)

for ax, ds in zip(
    axes,
    ["WN18RR", "YAGO11k"]
):

    real = (
        pd.read_csv(
            real_paths[ds]
        )
        .set_index("covariate")
    )

    rew = (
        pd.read_csv(
            rewired_paths[ds]
        )
        .set_index("covariate")
    )

    original = [
        real.loc[f, "rho_pooled"]
        for f in REWIRE_FEATURES
    ]

    null = [
        rew.loc[f, "rho_pooled"]
        for f in REWIRE_FEATURES
    ]

    x = np.arange(
        len(REWIRE_FEATURES)
    )

    width = 0.36

    ax.bar(
        x - width / 2,
        original,
        width,
        label="Original"
    )

    ax.bar(
        x + width / 2,
        null,
        width,
        label="Degree-preserving rewired"
    )

    ax.axhline(
        0,
        linewidth=1
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            REWIRE_LABELS[f]
            for f in REWIRE_FEATURES
        ],
        rotation=35,
        ha="right"
    )

    ax.set_title(ds)

    ax.set_ylim(
        -1,
        1
    )

axes[0].set_ylabel(
    r"Spearman $\rho$"
)

axes[1].legend(
    frameon=False
)

fig.suptitle(
    "Degree-preserving rewiring changes structural correspondence",
    y=1.03
)

save(fig, "fig5_degree_preserving_null")


# ============================================================
# LOAD DENSITY SWEEP
# ============================================================

with open(
    "runs/density_sweep/summary.json"
) as f:

    density = json.load(f)

er_rows = []

for key, obj in density.items():

    m = re.fullmatch(
        r"er_p([0-9.]+)",
        key
    )

    if not m:
        continue

    p = float(
        m.group(1)
    )

    row = {
        "p": p
    }

    for feat in [
        "forman_ricci",
        "growth_rate",
        "saturation_r",
        "branch_persist"
    ]:

        z = obj.get(
            feat,
            {}
        )

        row[f"{feat}_rho"] = (
            z.get("rho")
            if isinstance(z, dict)
            else np.nan
        )

        row[f"{feat}_var"] = (
            z.get("var_feature")
            if isinstance(z, dict)
            else np.nan
        )

    er_rows.append(row)

er = (
    pd.DataFrame(er_rows)
    .sort_values("p")
)


# ============================================================
# APPENDIX FIGURE A1
# DENSITY SWEEP — CORRELATION
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.5, 4.7)
)

for feat, label in labels.items():

    ax.plot(
        er["p"],
        er[f"{feat}_rho"],
        marker="o",
        linewidth=1.7,
        label=label
    )

ax.axhline(
    0,
    linewidth=1,
    linestyle="--"
)

ax.set_xlim(
    0,
    1.03
)

ax.set_xticks([
    0.02,
    0.05,
    0.10,
    0.20,
    0.40,
    0.70,
    1.00
])

ax.set_xticklabels([
    "0.02",
    "0.05",
    "0.10",
    "0.20",
    "0.40",
    "0.70",
    "1.00"
])

ax.set_xlabel(
    r"Erdős–Rényi edge probability $p$"
)

ax.set_ylabel(
    r"Spearman $\rho$"
)

ax.set_ylim(
    -1.08,
    1.08
)

ax.legend(
    frameon=False,
    ncol=2
)

ax.set_title(
    "Structural associations across graph density"
)

ax.axvline(
    1.0,
    linestyle=":",
    linewidth=1
)

ax.text(
    0.985,
    -0.92,
    "complete graph:\nall measures degenerate",
    ha="right",
    va="bottom",
    fontsize=8
)

save(fig, "figA1_density_correlations")


# ============================================================
# APPENDIX FIGURE A2
# DENSITY SWEEP — TRUE VARIANCE COLLAPSE
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.5, 4.7)
)

degenerate_y = 1e-9

for feat, label in labels.items():

    y = (
        er[f"{feat}_var"]
        .astype(float)
        .values
    )

    x = (
        er["p"]
        .astype(float)
        .values
    )

    positive = (
        np.isfinite(y)
        & (y > 0)
    )

    zero = (
        np.isfinite(y)
        & (y == 0)
    )

    ax.plot(
        x[positive],
        y[positive],
        marker="o",
        linewidth=1.7,
        label=label
    )

    if zero.any():

        ax.scatter(
            x[zero],
            np.full(
                zero.sum(),
                degenerate_y
            ),
            marker="x",
            s=55
        )

ax.set_yscale(
    "log"
)

ax.set_xlim(
    0,
    1.03
)

ax.set_ylim(
    5e-10,
    None
)

ax.set_xticks([
    0.02,
    0.05,
    0.10,
    0.20,
    0.40,
    0.70,
    1.00
])

ax.set_xticklabels([
    "0.02",
    "0.05",
    "0.10",
    "0.20",
    "0.40",
    "0.70",
    "1.00"
])

ax.set_xlabel(
    r"Erdős–Rényi edge probability $p$"
)

ax.set_ylabel(
    "Variance of structural statistic"
)

ax.set_title(
    "Structural measurements become degenerate at high density"
)

ax.legend(
    frameon=False,
    ncol=2
)

ax.text(
    0.02,
    0.03,
    "× at lower boundary = exact zero variance",
    transform=ax.transAxes,
    fontsize=8
)

save(fig, "figA2_density_variance_collapse")


# ============================================================
# APPENDIX FIGURE A3
# CONTROLLED GRAPH-GROWTH REGIMES
# ============================================================

tree_profile = pd.read_csv(
    "data/trees/relation_profile.csv"
)

tree_seed_files = sorted(
    glob.glob(
        "runs/trees/curvatures_seed*.csv"
    )
)

rows = []

for fp in tree_seed_files:

    m = re.search(
        r"seed(\d+)",
        fp
    )

    seed = int(
        m.group(1)
    )

    z = pd.read_csv(fp)[[
        "relation_id",
        "curvature"
    ]].copy()

    z["seed"] = seed

    rows.append(z)

tc = pd.concat(
    rows,
    ignore_index=True
)

tc = tc.merge(
    tree_profile,
    on="relation_id"
)

reg = (
    tc.groupby(
        [
            "relation_id",
            "family",
            "regime",
            "branching",
            "depth"
        ],
        as_index=False
    )
    .agg(
        mean_curvature=(
            "curvature",
            "mean"
        ),
        sd_curvature=(
            "curvature",
            "std"
        )
    )
)

desired_order = [
    "tree_b2_d3",
    "tree_b2_d6",
    "tree_b3_d4",
    "tree_b4_d3",
    "tree_b5_d3",
    "star_b8",
    "star_b32",
    "chain_d8",
    "chain_d16",
    "grid_8",
    "grid_12",
    "clique_6",
    "clique_10",
]

pretty = {
    "tree_b2_d3": "Tree\n2×3",
    "tree_b2_d6": "Tree\n2×6",
    "tree_b3_d4": "Tree\n3×4",
    "tree_b4_d3": "Tree\n4×3",
    "tree_b5_d3": "Tree\n5×3",
    "star_b8": "Star\n8",
    "star_b32": "Star\n32",
    "chain_d8": "Chain\n8",
    "chain_d16": "Chain\n16",
    "grid_8": "Grid\n8",
    "grid_12": "Grid\n12",
    "clique_6": "Clique\n6",
    "clique_10": "Clique\n10",
}

reg["order"] = (
    reg["family"]
    .map({
        name: i
        for i, name
        in enumerate(desired_order)
    })
)

reg = reg.sort_values(
    "order"
)

fig, ax = plt.subplots(
    figsize=(9.4, 5.1)
)

x = np.arange(
    len(reg)
)

ax.errorbar(
    x,
    reg["mean_curvature"],
    yerr=reg["sd_curvature"],
    fmt="o",
    capsize=3,
    markersize=6
)

ax.set_xticks(x)

ax.set_xticklabels([
    pretty.get(
        f,
        f
    )
    for f in reg["family"]
])

ax.set_ylabel(
    "Learned curvature"
)

ax.set_title(
    "Learned curvature across controlled graph-growth regimes"
)

# regime separators
for xpos in [
    4.5,
    6.5,
    8.5,
    10.5
]:
    ax.axvline(
        xpos,
        linestyle=":",
        linewidth=0.8
    )

ax.text(
    2,
    ax.get_ylim()[1],
    "Sustained exponential",
    ha="center",
    va="bottom",
    fontsize=8
)

ax.text(
    5.5,
    ax.get_ylim()[1],
    "Burst",
    ha="center",
    va="bottom",
    fontsize=8
)

ax.text(
    7.5,
    ax.get_ylim()[1],
    "Linear",
    ha="center",
    va="bottom",
    fontsize=8
)

ax.text(
    9.5,
    ax.get_ylim()[1],
    "Polynomial",
    ha="center",
    va="bottom",
    fontsize=8
)

ax.text(
    11.5,
    ax.get_ylim()[1],
    "Bounded",
    ha="center",
    va="bottom",
    fontsize=8
)

save(fig, "figA3_growth_regimes")


# ============================================================
# DONE
# ============================================================

print()
print("=" * 80)
print("ALL PAPER A FIGURES COMPLETE")
print("OUTPUT:", OUT)
print("=" * 80)
