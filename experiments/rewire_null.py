#!/usr/bin/env python3
"""Degree-preserving rewiring null.

THE OBJECTION THIS ANSWERS
--------------------------
Our headline covariate, Forman-Ricci curvature, is 4 - deg(u) - deg(v): a
function of endpoint degrees. The strongest deflationary reading of the whole
paper is therefore "you have shown that learned curvature correlates with
degree, which is unsurprising."

THE TEST
--------
Rewire each relation subgraph with a configuration model: the degree sequence is
preserved EXACTLY, while higher-order structure (paths, cycles, volume growth,
tree-likeness, community structure) is destroyed. Retrain the probe on the
rewired graph and recompute every correlation.

  If the correlation SURVIVES rewiring, learned curvature is a degree statistic
  and nothing more. The paper says so plainly.

  If the correlation WEAKENS, curvature reads structure beyond degree, the
  Ollivier-Ricci null becomes a genuine puzzle, and the growth-rate result
  is load-bearing rather than incidental.

Either outcome is publishable; the point is that the paper should not leave the
question open.

WHAT IS PRESERVED / DESTROYED
-----------------------------
  preserved : per-node degree within each relation, relation sizes,
              entity vocabulary, number of triples
  destroyed : which specific nodes are connected, and hence all path
              structure, volume growth, cycles, clustering

USAGE
-----
  # 1. build the rewired dataset
  python rewire_null.py --data_dir data/wn18rr --out data/wn18rr_rewired \
      --seed 0 --report

  # 2. train the probe on it exactly as on the real data
  for S in 0 1 2; do
    python atth_probe.py --data_dir data/wn18rr_rewired \
      --out_dir runs/wn18rr_rewired --seed $S --epochs 150 --dim 32
  done

  # 3. recompute covariates ON THE REWIRED GRAPH and correlate
  python compute_fanout.py --data_dir data/wn18rr_rewired \
      --out runs/wn18rr_rewired/relation_stats.csv
  python volume_growth.py --data_dir data/wn18rr_rewired \
      --stats runs/wn18rr_rewired/relation_stats.csv --min_triples 20
  python analyze_correlation.py --run_dir runs/wn18rr_rewired \
      --stats runs/wn18rr_rewired/relation_stats.csv --min_triples 20

  # 4. compare real vs rewired side by side
  python rewire_null.py --compare real:runs/wn18rr rewired:runs/wn18rr_rewired
"""

import argparse, csv, math, os, random, statistics
from collections import defaultdict


# ---------------------------------------------------------------------
# configuration-model rewiring
# ---------------------------------------------------------------------

def read_quads(path):
    q = []
    for line in open(path):
        p = line.split()
        if len(p) < 3:
            continue
        try:
            h, r, t = int(p[0]), int(p[1]), int(p[2])
        except ValueError:
            continue
        tau = p[3] if len(p) > 3 else "0"
        q.append((h, r, t, tau))
    return q


def rewire_relation(edges, rng, max_tries=200):
    """Double-edge swap on a directed edge list.

    Repeatedly pick two edges (a->b), (c->d) and rewrite them as
    (a->d), (c->b). This preserves every node's out-degree and in-degree
    exactly while randomising who connects to whom. Swaps that would create
    a self-loop or a duplicate edge are rejected.
    """
    E = list(edges)
    present = set(E)
    n_swaps = 10 * len(E)          # standard mixing heuristic
    done = 0
    for _ in range(n_swaps * 3):
        if done >= n_swaps:
            break
        i, j = rng.randrange(len(E)), rng.randrange(len(E))
        if i == j:
            continue
        (a, b), (c, d) = E[i], E[j]
        if a == d or c == b:                  # would self-loop
            continue
        if (a, d) in present or (c, b) in present:   # would duplicate
            continue
        present.discard((a, b)); present.discard((c, d))
        present.add((a, d)); present.add((c, b))
        E[i], E[j] = (a, d), (c, b)
        done += 1
    return E, done


def degree_profile(edges):
    out, inn = defaultdict(int), defaultdict(int)
    for a, b in edges:
        out[a] += 1; inn[b] += 1
    return out, inn


def build(args):
    rng = random.Random(args.seed)
    quads = read_quads(os.path.join(args.data_dir, "train.txt"))

    by_rel = defaultdict(list)
    taus_by_rel = defaultdict(list)
    for h, r, t, tau in quads:
        by_rel[r].append((h, t))
        taus_by_rel[r].append(tau)

    new_quads = []
    print(f"{'rel':>5s} {'edges':>8s} {'swaps':>8s} {'deg ok':>7s} "
          f"{'clust before':>13s} {'clust after':>12s}")
    for r in sorted(by_rel):
        edges = by_rel[r]
        if len(edges) < 4:
            for (h, t), tau in zip(edges, taus_by_rel[r]):
                new_quads.append((h, r, t, tau))
            continue

        o0, i0 = degree_profile(edges)
        new_edges, n_swaps = rewire_relation(edges, rng)
        o1, i1 = degree_profile(new_edges)
        deg_ok = (o0 == o1) and (i0 == i1)

        if args.report:
            c0 = _clustering(edges)
            c1 = _clustering(new_edges)
            print(f"{r:5d} {len(edges):8d} {n_swaps:8d} {str(deg_ok):>7s} "
                  f"{c0:13.4f} {c1:12.4f}")
        else:
            print(f"{r:5d} {len(edges):8d} {n_swaps:8d} {str(deg_ok):>7s}")

        if not deg_ok:
            raise SystemExit(f"degree sequence changed for relation {r} "
                             "-- this is a bug, do not use the output")

        for (h, t), tau in zip(new_edges, taus_by_rel[r]):
            new_quads.append((h, r, t, tau))

    # keep valid/test as-is so the eval protocol is unchanged in shape;
    # they are not used for any correlation, only for early stopping.
    os.makedirs(args.out, exist_ok=True)
    rng.shuffle(new_quads)
    with open(os.path.join(args.out, "train.txt"), "w") as f:
        for q in new_quads:
            f.write("\t".join(map(str, q)) + "\n")
    for split in ("valid.txt", "test.txt"):
        src = os.path.join(args.data_dir, split)
        if os.path.exists(src):
            with open(src) as a, open(os.path.join(args.out, split), "w") as b:
                b.write(a.read())
    for extra in ("relation2id.txt", "entity2id.txt"):
        src = os.path.join(args.data_dir, extra)
        if os.path.exists(src):
            with open(src) as a, open(os.path.join(args.out, extra), "w") as b:
                b.write(a.read())
    print(f"\n[done] {len(new_quads)} rewired triples -> {args.out}")
    print("Degree sequences preserved exactly; path structure randomised.")


def _clustering(edges):
    """undirected clustering coefficient, for the before/after report"""
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    tot, n = 0.0, 0
    for v, nb in adj.items():
        k = len(nb)
        if k < 2:
            continue
        links = sum(1 for x in nb for y in nb if x < y and y in adj[x])
        tot += 2.0 * links / (k * (k - 1)); n += 1
    return tot / n if n else 0.0


# ---------------------------------------------------------------------
# side-by-side comparison
# ---------------------------------------------------------------------

def compare(specs):
    runs = {}
    for spec in specs:
        name, path = spec.split(":", 1)
        p = os.path.join(path, "correlation_summary.csv")
        if not os.path.exists(p):
            print(f"[skip] {name}: no correlation_summary.csv"); continue
        d = {}
        for r in csv.DictReader(open(p)):
            try:
                d[r["covariate"]] = (float(r.get("rho_pooled", r.get("rho_mean_of_seeds", r.get("rho_mean","nan")))),
                                     float(r["perm_p"]))
            except (ValueError, KeyError):
                continue
        runs[name] = d
    if len(runs) < 2:
        print("need at least two runs to compare"); return

    covs = ["forman_ricci", "growth_rate", "saturation_r", "branch_persist",
            "static_fanout", "tail_fanin", "beta1", "clustering",
            "longest_path", "reciprocity"]
    names = list(runs)
    hdr = f"{'covariate':>16s}" + "".join(f"{n:>14s}" for n in names) + f"{'delta':>10s}"
    print(hdr); print("-" * len(hdr))
    for c in covs:
        if not all(c in runs[n] for n in names):
            continue
        vals = [runs[n][c][0] for n in names]
        line = f"{c:>16s}" + "".join(
            f"{runs[n][c][0]:+13.3f}{'*' if runs[n][c][1] < 0.05 else ' '}"
            for n in names)
        line += f"{abs(vals[0]) - abs(vals[-1]):+10.3f}"
        print(line)
    print("\ndelta = |rho on real| - |rho on rewired|.")
    print("Large positive delta => the association depends on structure beyond")
    print("degree. Near zero => learned curvature is a degree statistic.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir")
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", action="store_true",
                    help="also print clustering before/after per relation")
    ap.add_argument("--compare", nargs="*", default=None,
                    help="name:run_dir pairs to compare instead of building")
    a = ap.parse_args()
    if a.compare:
        compare(a.compare)
    elif a.data_dir and a.out:
        build(a)
    else:
        ap.error("give --data_dir and --out, or --compare")
