#!/usr/bin/env python3

import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import rankdata

N_PERM = 10000
SEED = 42

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


def residuals(y, z):
    """Residuals of rank(y) after linear adjustment for rank(z)."""
    ry = rankdata(y, method="average")
    rz = rankdata(z, method="average")

    X = np.column_stack([np.ones(len(rz)), rz])
    beta, *_ = np.linalg.lstsq(X, ry, rcond=None)

    return ry - X @ beta


def partial_spearman(x, y, z):
    rx = residuals(x, z)
    ry = residuals(y, z)
    return np.corrcoef(rx, ry)[0, 1]


def load_curvatures(run_dir):
    patterns = [
        os.path.join(run_dir, "curvatures_seed*.csv"),
        os.path.join(run_dir, "*curvature*seed*.csv"),
        os.path.join(run_dir, "curvature_seed*.csv"),
    ]

    files = sorted(set(
        f for pattern in patterns
        for f in glob.glob(pattern)
    ))

    if not files:
        raise FileNotFoundError(
            f"No curvature CSV files found in {run_dir}"
        )

    dfs = []

    for i, f in enumerate(files):
        df = pd.read_csv(f)

        rid_candidates = [
            "relation_id", "rel_id", "relation", "rel"
        ]
        curv_candidates = [
            "curvature", "c", "c_r", "learned_curvature"
        ]

        rid = next(
            (x for x in rid_candidates if x in df.columns), None
        )
        curv = next(
            (x for x in curv_candidates if x in df.columns), None
        )

        if rid is None or curv is None:
            raise ValueError(
                f"Cannot identify columns in {f}: {list(df.columns)}"
            )

        tmp = df[[rid, curv]].copy()
        tmp.columns = ["relation_id", "curvature"]
        tmp["seed"] = i

        dfs.append(tmp)

    return pd.concat(dfs, ignore_index=True)


def test_dataset(name, run_dir, min_triples, seed):
    rng = np.random.default_rng(seed)

    stats_path = os.path.join(run_dir, "relation_stats.csv")
    stats = pd.read_csv(stats_path)

    stats = stats[
        stats["n_triples"] >= min_triples
    ].copy()

    stats["logfreq"] = np.log(
        stats["n_triples"].astype(float)
    )

    curv = load_curvatures(run_dir)

    dat = curv.merge(
        stats[
            ["relation_id", "logfreq", COVARIATE]
        ],
        on="relation_id",
        how="inner"
    )

    dat = dat.dropna(
        subset=["curvature", COVARIATE, "logfreq"]
    ).copy()

    # ---------------------------------------------------------
    # Observed pooled partial Spearman
    # ---------------------------------------------------------

    observed = partial_spearman(
        dat["curvature"].to_numpy(),
        dat[COVARIATE].to_numpy(),
        dat["logfreq"].to_numpy()
    )

    # ---------------------------------------------------------
    # Residualize at relation level.
    #
    # This preserves the relationship between Forman-Ricci and
    # log frequency rather than destroying it by directly
    # permuting Forman-Ricci.
    # ---------------------------------------------------------

    rel = (
        dat[
            ["relation_id", COVARIATE, "logfreq"]
        ]
        .drop_duplicates("relation_id")
        .sort_values("relation_id")
        .reset_index(drop=True)
    )

    x = rel[COVARIATE].to_numpy()
    z = rel["logfreq"].to_numpy()

    rx = residuals(x, z)

    # curvature residuals remain seed-specific
    y_rank = rankdata(
        dat["curvature"].to_numpy(),
        method="average"
    )
    z_rows = rankdata(
        dat["logfreq"].to_numpy(),
        method="average"
    )

    Xz = np.column_stack(
        [np.ones(len(z_rows)), z_rows]
    )

    beta_y, *_ = np.linalg.lstsq(
        Xz, y_rank, rcond=None
    )

    ry = y_rank - Xz @ beta_y

    # Map relation -> residualized structural value
    rel_to_rx = dict(
        zip(rel["relation_id"], rx)
    )

    xres_rows = (
        dat["relation_id"]
        .map(rel_to_rx)
        .to_numpy()
    )

    obs_resid = np.corrcoef(
        xres_rows, ry
    )[0, 1]

    # ---------------------------------------------------------
    # Cluster permutation:
    # permute structural residuals across RELATIONS.
    #
    # All seed observations belonging to the same relation
    # receive the same permuted residual.
    # ---------------------------------------------------------

    rel_ids = rel["relation_id"].to_numpy()

    extreme = 0

    for _ in range(N_PERM):
        perm_rx = rng.permutation(rx)

        mapping = dict(
            zip(rel_ids, perm_rx)
        )

        xp = (
            dat["relation_id"]
            .map(mapping)
            .to_numpy()
        )

        rho_perm = np.corrcoef(
            xp, ry
        )[0, 1]

        if abs(rho_perm) >= abs(obs_resid):
            extreme += 1

    p = (extreme + 1) / (N_PERM + 1)

    return {
        "dataset": name,
        "n_rel": rel["relation_id"].nunique(),
        "n_rows": len(dat),
        "partial_rho": observed,
        "residual_rho": obs_resid,
        "partial_perm_p": p,
    }


results = []

for i, (name, run, m) in enumerate(RUNS):
    try:
        results.append(
            test_dataset(
                name,
                run,
                m,
                SEED + i
            )
        )
    except Exception as e:
        print(f"[ERROR] {name}: {e}")


print()
print(
    f"{'Dataset':<16}"
    f"{'Nrel':>7}"
    f"{'rho':>10}"
    f"{'perm p':>12}"
    f"{'sig?':>8}"
)

print("-" * 53)

for r in results:
    print(
        f"{r['dataset']:<16}"
        f"{r['n_rel']:>7d}"
        f"{r['partial_rho']:>+10.3f}"
        f"{r['partial_perm_p']:>12.5g}"
        f"{str(r['partial_perm_p'] < .05):>8}"
    )

pd.DataFrame(results).to_csv(
    "partial_permutation_summary_v2.csv",
    index=False
)

print("\nwrote partial_permutation_summary_v2.csv")
