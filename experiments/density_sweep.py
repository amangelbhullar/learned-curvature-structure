#!/usr/bin/env python3
"""
Density-sweep sanity check for curvature-probe features.

Hypothesis: on complete / near-complete graphs the local features
(forman_ricci, growth_rate, saturation_r, branch_persist) become
(near-)constant across nodes -> variance collapses -> correlation with
any node-level target degenerates. This script generates a suite of
synthetic graphs from tree-like to complete, computes the same four
features, reports per-graph feature variance AND Spearman rho against a
"ground-truth" node property (hyperbolicity proxy: negative distance-
to-periphery / eccentricity-based depth), with permutation p-values,
bootstrap CIs, partial correlation controlling log n(eighborhood size),
and per-seed breakdown -- printed in the same format as your runs/
output so it plugs into the same downstream parsing.

Usage:
    python density_sweep.py [--n 200] [--seeds 0 1 2 3 4] [--out runs/density_sweep]

If your pipeline computes the target differently, swap out
`target_property()` -- everything else stands alone.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Graph suite
# ---------------------------------------------------------------------------

def make_suite(n, seed):
    """Sparse -> dense sweep plus structured baselines."""
    g = {}
    for p in (0.02, 0.05, 0.10, 0.20, 0.40, 0.70, 1.00):
        G = nx.erdos_renyi_graph(n, p, seed=seed)
        g[f"er_p{p:.2f}"] = G
    # tree (maximally sparse, hyperbolic-like)
    try:
        g["tree"] = nx.random_labeled_tree(n, seed=seed)
    except AttributeError:  # older networkx
        g["tree"] = nx.random_tree(n, seed=seed)
    g["ba_m3"] = nx.barabasi_albert_graph(n, 3, seed=seed)
    g["complete"] = nx.complete_graph(n)
    # dense cliques + sparse bridges (KG-like)
    k = max(3, n // 10)
    g["clique_chain"] = nx.connected_caveman_graph(k, 10)
    # keep only the largest connected component of everything
    for name, G in list(g.items()):
        if not nx.is_connected(G):
            cc = max(nx.connected_components(G), key=len)
            g[name] = G.subgraph(cc).copy()
    return g


# ---------------------------------------------------------------------------
# Features (node-level; edge features averaged onto incident nodes)
# ---------------------------------------------------------------------------

def forman_ricci_node(G):
    """Unweighted combinatorial Forman curvature per edge:
    F(u,v) = 4 - deg(u) - deg(v)  (triangle-free form; standard baseline).
    Node value = mean over incident edges."""
    deg = dict(G.degree())
    acc = defaultdict(list)
    for u, v in G.edges():
        f = 4 - deg[u] - deg[v]
        acc[u].append(f)
        acc[v].append(f)
    return {u: (np.mean(acc[u]) if acc[u] else 0.0) for u in G.nodes()}


def bfs_layers(G, src, max_r):
    sizes = []
    seen = {src}
    frontier = {src}
    for _ in range(max_r):
        nxt = set()
        for u in frontier:
            for w in G.neighbors(u):
                if w not in seen:
                    nxt.add(w)
        seen |= nxt
        sizes.append(len(nxt))
        frontier = nxt
        if not nxt:
            break
    return sizes, len(seen)


def growth_rate_node(G, max_r=4):
    """Log-slope of ball-size growth over first max_r hops."""
    out = {}
    for u in G.nodes():
        sizes, _ = bfs_layers(G, u, max_r)
        cum = np.cumsum([1] + sizes)  # |B_r|
        r = np.arange(len(cum))
        if len(cum) < 3:
            out[u] = 0.0
            continue
        slope, *_ = np.polyfit(r, np.log(cum), 1)
        out[u] = slope
    return out


def saturation_r_node(G, frac=0.9, max_r=None):
    """Smallest radius whose ball covers `frac` of the component."""
    N = G.number_of_nodes()
    if max_r is None:
        max_r = N
    out = {}
    for u in G.nodes():
        sizes, _ = bfs_layers(G, u, max_r)
        cum = 1
        r_sat = len(sizes)
        for r, s in enumerate(sizes, start=1):
            cum += s
            if cum >= frac * N:
                r_sat = r
                break
        out[u] = r_sat
    return out


def branch_persist_node(G, max_r=4):
    """How long distinct branches from a node stay disconnected.
    For each node: remove it, count components among its neighbors'
    induced subgraph within radius max_r; persistence = mean over radii of
    (#components / deg). Tree -> stays high; clique -> collapses to ~1/deg."""
    out = {}
    for u in G.nodes():
        nbrs = list(G.neighbors(u))
        d = len(nbrs)
        if d <= 1:
            out[u] = 1.0
            continue
        H = G.subgraph([x for x in G.nodes() if x != u])
        # grow balls around each neighbor jointly, track merging via union-find
        parent = {v: v for v in nbrs}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        owner = {v: v for v in nbrs}  # node -> which branch reached it first
        frontier = {v: {v} for v in nbrs}
        persist = []
        for _ in range(max_r):
            new_frontier = {v: set() for v in nbrs}
            for b in nbrs:
                for x in frontier[b]:
                    for w in H.neighbors(x):
                        if w in owner:
                            ra, rb = find(owner[w]), find(b)
                            if ra != rb:
                                parent[ra] = rb
                        else:
                            owner[w] = b
                            new_frontier[b].add(w)
            frontier = new_frontier
            ncomp = len({find(b) for b in nbrs})
            persist.append(ncomp / d)
            if all(not f for f in frontier.values()):
                # exhausted; components frozen hereafter
                persist += [ncomp / d] * (max_r - len(persist))
                break
        out[u] = float(np.mean(persist))
    return out


FEATURES = {
    "forman_ricci": forman_ricci_node,
    "growth_rate": growth_rate_node,
    "saturation_r": saturation_r_node,
    "branch_persist": branch_persist_node,
}


# ---------------------------------------------------------------------------
# Target property + neighborhood-size covariate
# ---------------------------------------------------------------------------

def target_property(G):
    """Ground-truth node score the features are supposed to track.
    Proxy for 'tree-depth / peripheralness': eccentricity minus radius,
    negated closeness centrality blend. Swap this for your pipeline's
    actual target if different."""
    ecc = nx.eccentricity(G)
    clo = nx.closeness_centrality(G)
    return {u: ecc[u] - 2.0 * clo[u] for u in G.nodes()}


def log_nbr_size(G, r=2):
    out = {}
    for u in G.nodes():
        sizes, tot = bfs_layers(G, u, r)
        out[u] = np.log(max(tot, 2))
    return out


# ---------------------------------------------------------------------------
# Stats: rho, permutation p, bootstrap CI, partial correlation
# ---------------------------------------------------------------------------

def spearman(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return stats.spearmanr(x, y).statistic


def perm_pvalue(x, y, rng, n_perm=10_000):
    obs = spearman(x, y)
    if np.isnan(obs):
        return np.nan, np.nan
    cnt = 0
    y = np.asarray(y)
    for _ in range(n_perm):
        r = spearman(x, rng.permutation(y))
        if abs(r) >= abs(obs):
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def bootstrap_ci(x, y, rng, n_boot=2_000, alpha=0.05):
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r = spearman(x[idx], y[idx])
        if not np.isnan(r):
            vals.append(r)
    if not vals:
        return np.nan, np.nan
    return (np.percentile(vals, 100 * alpha / 2),
            np.percentile(vals, 100 * (1 - alpha / 2)))


def partial_spearman(x, y, z):
    """Spearman partial correlation of x,y controlling z (rank-residual method)."""
    if np.std(x) == 0 or np.std(y) == 0 or np.std(z) == 0:
        return np.nan
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))

    def resid(a, b):
        beta = np.polyfit(b, a, 1)
        return a - np.polyval(beta, b)

    return stats.pearsonr(resid(rx, rz), resid(ry, rz)).statistic


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n_perm", type=int, default=10_000)
    ap.add_argument("--n_boot", type=int, default=2_000)
    ap.add_argument("--out", default="runs/density_sweep")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    graph_names = list(make_suite(args.n, args.seeds[0]).keys())

    results = {}  # graph -> feature -> dict
    for gname in graph_names:
        results[gname] = {f: {"per_seed": [], "pooled_x": defaultdict(list)}
                          for f in FEATURES}

    pooled = {g: {f: {"x": [], "y": [], "z": []} for f in FEATURES}
              for g in graph_names}
    varlog = {g: {f: [] for f in FEATURES} for g in graph_names}

    for seed in args.seeds:
        suite = make_suite(args.n, seed)
        for gname, G in suite.items():
            tgt = target_property(G)
            lz = log_nbr_size(G)
            nodes = list(G.nodes())
            y = np.array([tgt[u] for u in nodes])
            z = np.array([lz[u] for u in nodes])
            for fname, fn in FEATURES.items():
                fv = fn(G)
                x = np.array([fv[u] for u in nodes])
                varlog[gname][fname].append(float(np.var(x)))
                r = spearman(x, y)
                results[gname][fname]["per_seed"].append(r)
                pooled[gname][fname]["x"].extend(x)
                pooled[gname][fname]["y"].extend(y)
                pooled[gname][fname]["z"].extend(z)

    rng = np.random.default_rng(12345)
    summary = {}
    for gname in graph_names:
        print(f"=== {args.out}/{gname} ===")
        summary[gname] = {}
        for fname in FEATURES:
            x = np.array(pooled[gname][fname]["x"])
            y = np.array(pooled[gname][fname]["y"])
            z = np.array(pooled[gname][fname]["z"])
            var_x = float(np.mean(varlog[gname][fname]))
            if np.std(x) == 0 or np.std(y) == 0:
                print(f"{fname:>18}: DEGENERATE (var(feature)={var_x:.3g}, "
                      f"var(target)={np.var(y):.3g}) -- rho undefined")
                summary[gname][fname] = {"rho": None, "var_feature": var_x}
                continue
            # NOTE: bootstrap resamples the SAME pooled (x,y) pairs the point
            # estimate uses -- avoids the CI/point mismatch seen in fb15k237.
            rho, p = perm_pvalue(x, y, rng, args.n_perm)
            lo, hi = bootstrap_ci(x, y, rng, args.n_boot)
            pr = partial_spearman(x, y, z)
            per_seed = ";".join(f"{r:.3f}" if not np.isnan(r) else "nan"
                                for r in results[gname][fname]["per_seed"])
            p_str = "p<1e-4" if p <= 1 / (args.n_perm + 1) + 1e-12 else f"p={p:.4g}"
            print(f"{fname:>18}: rho={rho:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  "
                  f"{p_str}  partial(|log n)={pr:+.3f}  "
                  f"var(feat)={var_x:.3g}  per-seed=({per_seed})")
            summary[gname][fname] = {
                "rho": rho, "ci": [lo, hi], "p": p, "partial": pr,
                "var_feature": var_x,
                "per_seed": results[gname][fname]["per_seed"],
            }
        print()

    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    # quick verdict on the variance-collapse hypothesis
    print("--- variance-collapse check (er sweep) ---")
    er = [g for g in graph_names if g.startswith("er_p")]
    for fname in FEATURES:
        vs = [summary[g][fname]["var_feature"] for g in er]
        rs = [summary[g][fname].get("rho") for g in er]
        rs_abs = [abs(r) if r is not None else 0.0 for r in rs]
        if np.std(vs) > 0 and np.std(rs_abs) > 0:
            track = stats.spearmanr(vs, rs_abs).statistic
            print(f"{fname:>18}: corr(var(feat), |rho|) across density = {track:+.3f}")
        else:
            print(f"{fname:>18}: degenerate across sweep")
    print(f"\nsaved: {args.out}/summary.json")


if __name__ == "__main__":
    main()
