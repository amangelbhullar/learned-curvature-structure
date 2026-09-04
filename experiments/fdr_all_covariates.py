#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests

RUNS = [
    ("Dissociation", "runs/synthetic"),
    ("Trees",        "runs/trees"),
    ("YAGO11k",      "runs/yago"),
    ("WN18RR",       "runs/wn18rr"),
    ("WIKIDATA12k",  "runs/wikidata"),
    ("FB15k-237",    "runs/fb15k237"),
    ("ICEWS14",      "runs/icews14"),
    ("GDELT",        "runs/gdelt"),
]

all_rows = []

for dataset, run in RUNS:

    p = os.path.join(
        run,
        "correlation_summary.csv"
    )

    d = pd.read_csv(p)

    for _, r in d.iterrows():

        if pd.isna(r.get("perm_p")):
            continue

        all_rows.append({
            "dataset": dataset,
            "covariate": r["covariate"],
            "rho": r["rho_pooled"],
            "p": r["perm_p"],
        })

df = pd.DataFrame(all_rows)

# ------------------------------------------------------------
# A. FDR separately within each dataset
# ------------------------------------------------------------

within = []

for dataset, d in df.groupby("dataset"):

    reject, q, _, _ = multipletests(
        d["p"],
        alpha=0.05,
        method="fdr_bh"
    )

    z = d.copy()
    z["q_within_dataset"] = q
    z["fdr_sig_within"] = reject

    within.append(z)

within = pd.concat(
    within,
    ignore_index=True
)

# ------------------------------------------------------------
# B. Global FDR across entire exploratory battery
# ------------------------------------------------------------

reject, q, _, _ = multipletests(
    df["p"],
    alpha=0.05,
    method="fdr_bh"
)

df["q_global"] = q
df["fdr_sig_global"] = reject

out = within.merge(
    df[
        [
            "dataset",
            "covariate",
            "q_global",
            "fdr_sig_global",
        ]
    ],
    on=["dataset", "covariate"]
)

print("\nFULL FDR TABLE")
print(
    out.sort_values(
        ["dataset", "q_within_dataset"]
    ).to_string(index=False)
)

print("\nSUMMARY")
for dataset, d in out.groupby("dataset"):

    print(
        f"{dataset:<16} "
        f"raw p<.05={sum(d.p < .05):>2}  "
        f"within-FDR={sum(d.fdr_sig_within):>2}  "
        f"global-FDR={sum(d.fdr_sig_global):>2}"
    )

out.to_csv(
    "runs/fdr_covariate_results.csv",
    index=False
)

print("\nwrote runs/fdr_covariate_results.csv")
