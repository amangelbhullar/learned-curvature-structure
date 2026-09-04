#!/usr/bin/env python3
"""
Planted-hierarchy noise sweep.

Design: fix a ground-truth balanced tree with known node depths (the
target -- its variance NEVER degenerates). Then add random "noise"
edges at increasing rates, from pure tree up to near-complete. Measure
how well each geometric feature still recovers planted depth as a
function of noise. Because the target is external and fixed, any decay
in rho is attributable to FEATURE degradation alone -- unlike the ER
density sweep, where target and features degenerate together.

Expected result (hypothesis): a robustness ordering, e.g.
    saturation_r dies first, then branch_persist, then growth_rate,
    with forman_ricci most robust.

Output format matches runs/ so it plugs into the same parsing.
Place this file next to density_sweep.py (it imports the features and
stats helpers from there).

Usage:
    python planted_hierarchy.py [--branching 3] [--depth 5] \
        [--seeds 0 1 2 3 4] [--out runs/planted_hierarchy]
"""

import argparse
import json
import os

import networkx as nx
import numpy as np
from scipy import stats

from density_sweep import (
    FEATURES,
    bootstrap_ci,
    log_nbr_size,
    partial_spearman,
    perm_pvalue,
    spearman,
)


# ---------------------------------------------------------------------------
# Planted hierarchy generator
# ---------------------------------------------------------------------------

def planted_tree(branching, depth):
    """Balanced tree + dict of true depths (the fixed target)."""
    T = nx.balanced_tree(branching, depth)
    depths = nx.single_source_shortest_path_length(T, 0)
    return T, depths


def add_noise_edges(T, rate, rng):
    """Add `rate` * n_tree_edges random non-edges. rate=0 -> pure tree.
    rate is unbounded above; it saturates when the graph completes."""
    G = T.copy()
    n = G.number_of_nodes()
    m_add = int(round(rate * T.number_of_edges()))
    max_extra = n * (n - 1) // 2 - G.number_of_edges()
    m_add = min(m_add, max_extra)
    added = 0
    nodes = list(G.nodes())
    # rejection-sample; fine because we cap at max_extra
    while added < m_add:
        u, v = rng.choice(nodes, 2, replace=False)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            added += 1
    return G


# noise rates: 0 = pure tree; 1.0 = as many noise edges as tree edges; etc.
NOISE_RATES = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branching", type=int, default=3)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n_perm", type=int, default=10_000)
    ap.add_argument("--n_boot", type=int, default=2_000)
    ap.add_argument("--out", default="runs/planted_hierarchy")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    T, depths = planted_tree(args.branching, args.depth)
    n = T.number_of_nodes()
    m_tree = T.number_of_edges()
    max_density = 2 * m_tree / (n * (n - 1))
    print(f"# planted tree: branching={args.branching} depth={args.depth} "
          f"n={n} tree_edges={m_tree} base_density={max_density:.4f}")
    print(f"# target = planted depth (var={np.var(list(depths.values())):.3f}, "
          f"fixed across all noise levels)\n")

    nodes = list(T.nodes())
    y = np.array([depths[u] for u in nodes], dtype=float)

    summary = {}
    stat_rng = np.random.default_rng(12345)

    for rate in NOISE_RATES:
        tag = f"noise_{rate:g}"
        print(f"=== {args.out}/{tag} ===")
        summary[tag] = {"noise_rate": rate}

        pooled = {f: {"x": [], "y": [], "z": []} for f in FEATURES}
        per_seed = {f: [] for f in FEATURES}
        varlog = {f: [] for f in FEATURES}
        densities = []

        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            G = add_noise_edges(T, rate, rng)
            densities.append(nx.density(G))
            lz = log_nbr_size(G)
            z = np.array([lz[u] for u in nodes])
            for fname, fn in FEATURES.items():
                fv = fn(G)
                x = np.array([fv[u] for u in nodes])
                varlog[fname].append(float(np.var(x)))
                per_seed[fname].append(spearman(x, y))
                pooled[fname]["x"].extend(x)
                pooled[fname]["y"].extend(y)
                pooled[fname]["z"].extend(z)

        summary[tag]["density"] = float(np.mean(densities))
        for fname in FEATURES:
            x = np.array(pooled[fname]["x"])
            yy = np.array(pooled[fname]["y"])
            z = np.array(pooled[fname]["z"])
            var_x = float(np.mean(varlog[fname]))
            if np.std(x) == 0:
                print(f"{fname:>18}: DEGENERATE (var(feature)={var_x:.3g}, "
                      f"var(target)={np.var(yy):.3g}) -- rho undefined")
                summary[tag][fname] = {"rho": None, "var_feature": var_x}
                continue
            rho, p = perm_pvalue(x, yy, stat_rng, args.n_perm)
            lo, hi = bootstrap_ci(x, yy, stat_rng, args.n_boot)
            pr = partial_spearman(x, yy, z)
            ps = ";".join(f"{r:.3f}" if not np.isnan(r) else "nan"
                          for r in per_seed[fname])
            p_str = ("p<1e-4" if p <= 1 / (args.n_perm + 1) + 1e-12
                     else f"p={p:.4g}")
            print(f"{fname:>18}: rho={rho:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  "
                  f"{p_str}  partial(|log n)={pr:+.3f}  "
                  f"var(feat)={var_x:.3g}  per-seed=({ps})")
            summary[tag][fname] = {
                "rho": rho, "ci": [lo, hi], "p": p, "partial": pr,
                "var_feature": var_x, "per_seed": per_seed[fname],
            }
        print()

    # robustness ordering: noise rate at which |rho| first drops below 0.5
    # (and below 0.2), per feature
    print("--- robustness ordering (|rho| vs planted depth) ---")
    half_life = {}
    for fname in FEATURES:
        r50 = r20 = None
        for rate in NOISE_RATES:
            e = summary[f"noise_{rate:g}"].get(fname, {})
            r = e.get("rho")
            a = abs(r) if r is not None else 0.0
            if r50 is None and a < 0.5:
                r50 = rate
            if r20 is None and a < 0.2:
                r20 = rate
        half_life[fname] = (r50, r20)
        print(f"{fname:>18}: |rho| first <0.5 at noise={r50}, "
              f"first <0.2 at noise={r20}"
              f"{'  (never)' if r50 is None else ''}")
    print("\nfeatures that survive higher noise are more robust probes "
          "of planted hierarchy.")

    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"saved: {args.out}/summary.json")


if __name__ == "__main__":
    main()
