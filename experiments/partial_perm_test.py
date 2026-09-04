#!/usr/bin/env python3

import os
import glob
import csv
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

N_PERM = 10000
SEED = 42
rng = np.random.default_rng(SEED)

RUNS = [
    ("Dissociation", "runs/synthetic", 10),
    ("Trees",        "runs/trees", 20),
    ("YAGO11k",      "runs/yago", 20),
    ("WN18RR",       "runs/wn18rr", 20),
    ("WIKIDATA12k",  "runs/wikidata", 20),
    ("FB15k-237",    "runs/fb15k237", 20),
    ("ICEWS14",      "runs/icews14", 50),
    ("GDELT",        "runs/gdelt", 50),
]

COVARIATE = "forman_ricci"


def residualize_rank(y, z):
    """
    Partial Spearman:
    rank-transform y and z, then regress rank(y) on [1, rank(z)]
    and return residuals.
    """
    y = rankdata(y, method="average")
    z = rankdata(z, method="average")

    X = np.column_stack([np.ones(len(z)), z])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_spearman(x, y, z):
    """
    Spearman partial correlation rho(x,y | z).
    """
    rx = residualize_rank(x, z)
    ry = residualize_rank(y, z)
    return np.corrcoef(rx, ry)[0, 1]


def load_curvatures(run_dir):
    """
    Load all seed curvature files and return:
      relation_id, seed, curvature

    Tries common file patterns.
    """
    patterns = [
        os.path.join(run_dir, "curvatures_seed*.csv"),
        os.path.join(run_dir, "*curvature*seed*.csv"),
        os.path.join(run_dir, "curvature_seed*.csv"),
    ]

    files = []
    for p in patterns:
        files.extend(glob.glob(p))

    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(
            f"No per-seed curvature CSVs found in {run_dir}"
        )

    dfs = []

    for i, f in enumerate(files):
        df = pd.read_csv(f)

        # relation id
        rid_candidates = [
            "relation_id", "rel_id", "relation", "rel"
        ]
        rid_col = next((c for c in rid_candidates if c in df.columns), None)

        # curvature
        curv_candidates = [
            "curvature", "c", "c_r", "learned_curvature"
        ]
        curv_col = next((c for c in curv_candidates if c in df.columns), None)

        if rid_col is None or curv_col is None:
            raise ValueError(
                f"{f}: could not identify relation/curvature columns. "
                f"Columns are {list(df.columns)}"
            )

        seed = i
        base = os.path.basename(f)
        if "seed" in base:
            try:
                seed = int(base.split("seed")[1].split(".")[0].split("_")[0])
            except Exception:
                pass

        tmp = df[[rid_col, curv_col]].copy()
        tmp.columns = ["relation_id", "curvature"]
        tmp["seed"] = seed
        dfs.append(tmp)

    return pd.concat(dfs, ignore_index=True)


def load_stats(run_dir, min_triples):
    path = os.path.join(run_dir, "relation_stats.csv")
    df = pd.read_csv(path)

    if "relation_id" not in df.columns:
        raise ValueError(f"{path}: missing relation_id")

    if "n_triples" not in df.columns:
        raise ValueError(f"{path}: missing n_triples")

    if COVARIATE not in df.columns:
        raise ValueError(f"{path}: missing {COVARIATE}")

    df = df[df["n_triples"] >= min_triples].copy()

    df["logfreq"] = np.log(df["n_triples"].astype(float))

    return df[
        ["relation_id", "n_triples", "logfreq", COVARIATE]
    ]


def test_dataset(name, run_dir, min_triples):
    curv = load_curvatures(run_dir)
    stats = load_stats(run_dir, min_triples)

    dat = curv.merge(stats, on="relation_id", how="inner")
    dat = dat.dropna(
        subset=["curvature", COVARIATE, "logfreq"]
    ).copy()

    # Important:
    # The structural covariate is fixed per relation, but curvature is repeated
    # over seeds. We retain this seed structure in the statistic.
    #
    # Observed partial rho:
    observed = partial_spearman(
        dat["curvature"].to_numpy(),
        dat[COVARIATE].to_numpy(),
        dat["logfreq"].to_numpy()
    )

    # Permute structural covariate at the RELATION level.
    # All seed rows belonging to the same relation receive the same permuted
    # covariate. This avoids treating seeds as independent structural samples.
    rel = (
        stats[["relation_id", COVARIATE, "logfreq"]]
        .dropna()
        .drop_duplicates("relation_id")
        .copy()
    )

    # only relations actually present in merged data
    used_ids = set(dat["relation_id"].unique())
    rel = rel[rel["relation_id"].isin(used_ids)].copy()

    rel_ids = rel["relation_id"].to_numpy()
    cov_values = rel[COVARIATE].to_numpy()

    permuted = np.empty(N_PERM)

    for b in range(N_PERM):
        shuffled = rng.permutation(cov_values)

        mapping = dict(zip(rel_ids, shuffled))
        x_perm = dat["relation_id"].map(mapping).to_numpy()

        permuted[b] = partial_spearman(
            dat["curvature"].to_numpy(),
            x_perm,
            dat["logfreq"].to_numpy()
        )

    # two-sided permutation p-value with +1 correction
    p = (
        np.sum(np.abs(permuted) >= abs(observed)) + 1
    ) / (N_PERM + 1)

    return {
        "dataset": name,
        "n_rel": dat["relation_id"].nunique(),
        "n_seed_rows": len(dat),
        "partial_rho": observed,
        "partial_perm_p": p,
    }


results = []

for name, run_dir, min_triples in RUNS:
    try:
        r = test_dataset(name, run_dir, min_triples)
        results.append(r)
    except Exception as e:
        print(f"[ERROR] {name}: {e}")

print()
print(
    f"{'Dataset':<16} "
    f"{'Nrel':>6} "
    f"{'Nrows':>7} "
    f"{'partial rho':>12} "
    f"{'perm p':>11} "
    f"{'sig?':>7}"
)
print("-" * 67)

for r in results:
    print(
        f"{r['dataset']:<16} "
        f"{r['n_rel']:>6d} "
        f"{r['n_seed_rows']:>7d} "
        f"{r['partial_rho']:>+12.3f} "
        f"{r['partial_perm_p']:>11.5g} "
        f"{str(r['partial_perm_p'] < 0.05):>7}"
    )

pd.DataFrame(results).to_csv(
    "partial_permutation_summary.csv",
    index=False
)

print("\nwrote partial_permutation_summary.csv")
