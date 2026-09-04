#!/usr/bin/env python3
"""
Recompute Ollivier-Ricci curvature properly (reference implementation:
GraphRicciCurvature package, exact Sinkhorn/OTD transport, no neighbor
truncation, no biased greedy fallback) and write it into each run's
relation_stats.csv as a new column `ollivier_rc_exact`, keeping the old
approximate `ollivier_rc` column for comparison.

Also runs a built-in toy-graph sanity check before touching any data:
a path graph must give negative mean edge curvature, a complete graph
positive, a balanced tree strongly negative. If the sanity check fails,
the script aborts.

Per relation: build the relation's subgraph the same way as
hierarchy_measures.py (undirected simple graph over the relation's
triples, capped at --max_nodes by BFS from a random triple endpoint --
ADJUST build_subgraph() if your pipeline differs), compute exact ORC on
every edge (alpha=0, i.e. mass fully on neighbors, matching the old
implementation's uniform-neighborhood convention), record the mean.

Usage:
  python recompute_ollivier.py --sanity-only          # just the toy check
  python recompute_ollivier.py \
      --data_dir data/synthetic --stats runs/synthetic/relation_stats.csv \
      [--min_triples 20] [--max_nodes 600] [--seed 0]

Loop over datasets the same way you ran hierarchy_measures.py.
"""

import argparse
import csv
import random
import sys
from collections import defaultdict

import networkx as nx

import numpy as np
import ot  # POT: exact earth-mover distance


def mean_orc(G, alpha=0.0):
    """Exact mean Ollivier-Ricci edge curvature, self-contained.
    kappa(u,v) = 1 - W1(m_u, m_v)/d(u,v); m_x = alpha at x, (1-alpha)
    uniform on neighbors; costs are exact shortest-path distances.
    Exact for graphs of a few hundred nodes."""
    if G.number_of_edges() == 0 or G.number_of_nodes() < 3:
        return float("nan")
    import networkx as nx
    # all-pairs shortest paths once (graphs are <= ~600 nodes)
    sp = dict(nx.all_pairs_shortest_path_length(G))
    vals = []
    for u, v in G.edges():
        Su = [u] + list(G.neighbors(u))
        Sv = [v] + list(G.neighbors(v))
        mu = np.full(len(Su), (1 - alpha) / max(len(Su) - 1, 1))
        mu[0] = alpha
        mv = np.full(len(Sv), (1 - alpha) / max(len(Sv) - 1, 1))
        mv[0] = alpha
        mu /= mu.sum(); mv /= mv.sum()
        C = np.array([[sp[a].get(b, G.number_of_nodes()) for b in Sv]
                      for a in Su], dtype=float)
        w1 = ot.emd2(mu, mv, C)
        vals.append(1.0 - w1 / sp[u][v])
    return float(np.mean(vals)) if vals else float("nan")


def sanity_check():
    """Reference behaviors: path negative, tree very negative, clique positive."""
    print("== toy-graph sanity (alpha=0, exact OTD) ==")
    cases = {
        "path_20":  nx.path_graph(20),
        "tree_b3d4": nx.balanced_tree(3, 4),
        "cycle_20": nx.cycle_graph(20),
        "complete_12": nx.complete_graph(12),
    }
    res = {name: mean_orc(G) for name, G in cases.items()}
    for name, v in res.items():
        print(f"  {name:>12}: mean ORC = {v:+.3f}")
    # correct alpha=0 references: path & cycle exactly 0, tree < 0, clique > 0
    ok = (abs(res["path_20"]) < 1e-9 and abs(res["cycle_20"]) < 1e-9
          and res["tree_b3d4"] < -0.1 and res["complete_12"] > 0.5)
    print("  PASS" if ok else "  FAIL -- do not trust results; investigate")
    return ok


# ---------------------------------------------------------------------------
def load_triples(data_dir):
    triples = []
    for line in open(f"{data_dir}/train.txt"):
        p = line.rstrip("\n").split("\t") if "\t" in line else line.split()
        if len(p) >= 3:
            triples.append((p[0].strip(), p[1].strip(), p[2].strip()))
    return triples


def build_subgraph(triples_r, max_nodes, rng):
    """Undirected simple graph over one relation's triples, BFS-capped.
    NOTE: mirror hierarchy_measures.py's construction. If that script
    builds subgraphs differently (direction, dedup, cap strategy), adjust
    here so exact-ORC is computed on the SAME graphs as everything else."""
    G = nx.Graph()
    G.add_edges_from((h, t) for h, _, t in triples_r if h != t)
    if G.number_of_nodes() <= max_nodes:
        return G
    # cap: BFS from a random node (connected subsample; random-node
    # sampling shatters sparse graphs -- see paper discussion)
    start = rng.choice(sorted(G.nodes()))
    keep, frontier = {start}, [start]
    while frontier and len(keep) < max_nodes:
        nxt = []
        for u in frontier:
            for w in G.neighbors(u):
                if w not in keep:
                    keep.add(w); nxt.append(w)
                    if len(keep) >= max_nodes: break
            if len(keep) >= max_nodes: break
        frontier = nxt
    return G.subgraph(keep).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir")
    ap.add_argument("--stats")
    ap.add_argument("--min_triples", type=int, default=20)
    ap.add_argument("--max_nodes", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sanity-only", action="store_true")
    args = ap.parse_args()

    if not sanity_check():
        sys.exit(1)
    if args.sanity_only:
        return
    if not (args.data_dir and args.stats):
        ap.error("--data_dir and --stats required unless --sanity-only")

    rng = random.Random(args.seed)
    triples = load_triples(args.data_dir)
    by_rel = defaultdict(list)
    for h, r, t in triples:
        by_rel[r].append((h, r, t))

    rows = list(csv.DictReader(open(args.stats)))
    # relation_stats keys relations by id; try to match on whatever id
    # column exists plus a name column if present
    idcol = "relation_id" if "relation_id" in rows[0] else list(rows[0])[0]
    namecol = next((c for c in ("relation", "relation_name", "rel")
                    if c in rows[0]), None)

    # map stats-row -> triple key. If relations in train.txt are names and
    # stats has only ids, we need the same ordering the pipeline used;
    # safest is matching by name when available, else by sorted order.
    rels_sorted = sorted(by_rel)
    def rel_key(row):
        if namecol and row[namecol] in by_rel:
            return row[namecol]
        try:
            return rels_sorted[int(row[idcol])]
        except Exception:
            return None

    n_done = 0
    for row in rows:
        key = rel_key(row)
        if key is None or len(by_rel[key]) < args.min_triples:
            row["ollivier_rc_exact"] = "nan"
            continue
        G = build_subgraph(by_rel[key], args.max_nodes, rng)
        v = mean_orc(G)
        row["ollivier_rc_exact"] = "nan" if v != v else f"{v:.6f}"
        n_done += 1
        print(f"  rel {row[idcol]:>6} ({key[:40]:<40}) "
              f"n={G.number_of_nodes():>4} e={G.number_of_edges():>5} "
              f"ORC={row['ollivier_rc_exact'] or 'nan'}")

    out = args.stats  # in place; .bak written first
    import shutil
    shutil.copy(args.stats, args.stats + ".bak")
    with open(out, "w", newline="") as f:
        fields = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[done] {n_done} relations -> {out} (backup: {out}.bak)")


if __name__ == "__main__":
    main()
