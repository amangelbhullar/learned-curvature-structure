#!/usr/bin/env python3
"""Canonical hierarchy measures per relation, training-free.

Adds to the covariate battery the two measures the geometry community
actually uses:

  delta_hyp    : sampled Gromov 4-point delta (tree-likeness; 0 = tree).
                 Scale-normalised by mean pairwise distance so it is
                 comparable across relations of different diameter.
  ollivier_rc  : mean Ollivier-Ricci curvature over sampled edges, using
                 uniform neighbourhood measures and exact optimal transport
                 via scipy.optimize.linear_sum_assignment when the supports
                 are equal size, otherwise a greedy transport bound.
  longest_path : longest shortest-path (approx. eccentricity-based depth)
  reciprocity  : fraction of edges whose reverse is also present

Joins onto an existing relation_stats.csv (adds columns) or writes its own.

  python hierarchy_measures.py --data_dir data/wn18rr \
      --stats runs/wn18rr/relation_stats.csv --min_triples 20
"""

import argparse, csv, math, os, random
from collections import defaultdict, deque


def read_quads(path):
    q = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 3:
                continue
            try:
                h, r, t = int(p[0]), int(p[1]), int(p[2])
            except ValueError:
                continue
            q.append((h, r, t))
    return q


def bfs(adj, src, cap=12):
    dist = {src: 0}
    dq = deque([src])
    while dq:
        u = dq.popleft()
        if dist[u] >= cap:
            continue
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist


def gromov_delta(adj, nodes, rng, n_quads=400, n_src=60):
    """Sampled 4-point delta, normalised by mean pairwise distance.
    Samples quadruples from within one large connected component so that
    all six pairwise distances are defined."""
    if len(nodes) < 4:
        return float("nan")
    # find the largest connected component (capped BFS from random seeds)
    best = []
    seen = set()
    for s in nodes:
        if s in seen:
            continue
        comp = list(bfs(adj, s, cap=50).keys())
        seen.update(comp)
        if len(comp) > len(best):
            best = comp
        if len(best) > 0.5 * len(nodes):
            break
    if len(best) < 4:
        return float("nan")
    srcs = rng.sample(best, min(n_src, len(best)))
    D = {s: bfs(adj, s, cap=50) for s in srcs}

    def d(a, b):
        if a in D and b in D[a]:
            return D[a][b]
        if b in D and a in D[b]:
            return D[b][a]
        return None

    deltas, dists = [], []
    for _ in range(n_quads):
        w, x, y, z = rng.sample(srcs, 4) if len(srcs) >= 4 else (None,)*4
        if w is None:
            break
        pairs = [d(w, x), d(y, z), d(w, y), d(x, z), d(w, z), d(x, y)]
        if any(p is None for p in pairs):
            continue
        s1, s2, s3 = pairs[0] + pairs[1], pairs[2] + pairs[3], pairs[4] + pairs[5]
        top2 = sorted([s1, s2, s3])[-2:]
        deltas.append((top2[1] - top2[0]) / 2.0)
        dists.extend(pairs)
    if not deltas or not dists:
        return float("nan")
    mean_d = sum(dists) / len(dists)
    return (sum(deltas) / len(deltas)) / mean_d if mean_d > 0 else float("nan")


def ollivier_ricci(adj, edges, rng, n_edges=200, max_nbr=20):
    """Mean Ollivier-Ricci curvature over sampled edges.
    kappa(u,v) = 1 - W1(m_u, m_v) / d(u,v), with d(u,v)=1 for edges and
    uniform neighbourhood measures. W1 computed by a greedy transport on
    the neighbour-pair distance matrix (exact for equal-size supports via
    Hungarian if scipy is available)."""
    try:
        from scipy.optimize import linear_sum_assignment
        have_scipy = True
    except Exception:
        have_scipy = False

    es = edges if len(edges) <= n_edges else rng.sample(edges, n_edges)
    vals = []
    for u, v in es:
        Nu = list(adj[u])[:max_nbr] or [u]
        Nv = list(adj[v])[:max_nbr] or [v]
        # pairwise distances between neighbourhoods (BFS-capped, cheap)
        Du = {a: bfs(adj, a, cap=4) for a in Nu}
        C = [[Du[a].get(b, 5) for b in Nv] for a in Nu]
        if have_scipy and len(Nu) == len(Nv):
            import numpy as np
            r, c = linear_sum_assignment(np.array(C, dtype=float))
            w1 = sum(C[i][j] for i, j in zip(r, c)) / len(Nu)
        else:                                   # greedy upper bound
            w1 = sum(min(row) for row in C) / len(Nu)
        vals.append(1.0 - w1)                   # d(u,v)=1 for an edge
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--stats", required=True,
                    help="relation_stats.csv to augment (modified in place)")
    ap.add_argument("--min_triples", type=int, default=20)
    ap.add_argument("--max_nodes", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    quads = read_quads(os.path.join(args.data_dir, "train.txt"))
    und = defaultdict(lambda: defaultdict(set))
    directed = defaultdict(set)
    nodes = defaultdict(set)
    cnt = defaultdict(int)
    for h, r, t in quads:
        und[r][h].add(t); und[r][t].add(h)
        directed[r].add((h, t))
        nodes[r].update((h, t)); cnt[r] += 1

    rows = list(csv.DictReader(open(args.stats)))
    out_rows = []
    for row in rows:
        r = int(row["relation_id"])
        if cnt.get(r, 0) < args.min_triples:
            row.update(delta_hyp="", ollivier_rc="", longest_path="",
                       reciprocity="")
            out_rows.append(row); continue
        nl = sorted(nodes[r])
        if len(nl) > args.max_nodes:
            nl = sorted(rng.sample(nl, args.max_nodes))
        sub = {u: (und[r][u] & set(nl)) for u in nl}
        edges = [(u, v) for u in nl for v in sub[u] if u < v]

        dh = gromov_delta(sub, nl, rng)
        orc = ollivier_ricci(sub, edges, rng) if edges else float("nan")
        # depth: max eccentricity over a few BFS roots
        depths = [max(bfs(sub, s).values()) for s in rng.sample(nl, min(15, len(nl)))]
        lp = max(depths) if depths else float("nan")
        fwd = directed[r]
        recip = (sum(1 for (a, b) in fwd if (b, a) in fwd) / len(fwd)) if fwd else float("nan")

        row.update(delta_hyp=f"{dh:.6f}", ollivier_rc=f"{orc:.6f}",
                   longest_path=str(lp), reciprocity=f"{recip:.6f}")
        out_rows.append(row)
        print(f"[rel {r:4d}] delta={dh:.4f} ollivier={orc:+.4f} "
              f"depth={lp} recip={recip:.3f}", flush=True)

    fields = list(out_rows[0].keys())
    with open(args.stats, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(out_rows)
    print(f"[done] augmented {args.stats}")


if __name__ == "__main__":
    main()
