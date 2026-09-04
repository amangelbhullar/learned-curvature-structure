#!/usr/bin/env python3
"""Synthetic graphs where BRANCHING and DEPTH are combined, not traded off.

The earlier benchmark varied one factor at a time, so no relation had sustained
exponential volume growth (b^d). Sarkar's theorem needs both. Here every
relation is a genuine b-ary tree of depth d, so volume growth is
exponential and sustained -- the regime where hyperbolic geometry is actually
justified.

  tree_b{B}_d{D}  : complete B-ary tree of depth D
                    volume growth ~ B^r, sustained to radius D
  star_b{B}       : B-ary tree of depth 1 (high degree, no sustained growth)
  chain_d{D}      : depth D, branching 1 (sustained but not exponential)
  grid_{K}        : K x K lattice (polynomial growth, flat geometry)
  clique_{K}      : K-clique (bounded diameter, spherical-ish)

Ground truth in relation_profile.csv, including the theoretical growth rate
log(B) for trees.
"""

import argparse, math, os, random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--copies", type=int, default=6,
                    help="independent components per relation")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    quads, prof = [], []
    nxt = 0
    rid = 0

    def new(k):
        nonlocal nxt
        ids = list(range(nxt, nxt + k)); nxt += k; return ids

    def add_tree(r, B, D):
        """complete B-ary tree of depth D"""
        root = new(1)[0]
        level = [root]
        for _ in range(D):
            nxt_level = []
            for u in level:
                kids = new(B)
                for v in kids:
                    quads.append((u, r, v, 0))
                nxt_level += kids
            level = nxt_level

    # --- sustained exponential growth: real trees ---
    for B, D in [(2, 6), (3, 4), (4, 3), (2, 3), (5, 3)]:
        r = rid; rid += 1
        prof.append((r, f"tree_b{B}_d{D}", "sustained_exp", B, D,
                     round(math.log(B), 4)))
        for _ in range(a.copies):
            add_tree(r, B, D)

    # --- high degree, no sustained growth: stars ---
    for B in [8, 32]:
        r = rid; rid += 1
        prof.append((r, f"star_b{B}", "burst_then_flat", B, 1,
                     round(math.log(B), 4)))
        for _ in range(a.copies * 3):
            add_tree(r, B, 1)

    # --- sustained but linear: chains ---
    for D in [8, 16]:
        r = rid; rid += 1
        prof.append((r, f"chain_d{D}", "linear", 1, D, 0.0))
        for _ in range(a.copies * 3):
            add_tree(r, 1, D)

    # --- polynomial growth: grids (flat) ---
    for K in [8, 12]:
        r = rid; rid += 1
        prof.append((r, f"grid_{K}", "polynomial", 4, K, 0.0))
        for _ in range(max(a.copies // 2, 1)):
            g = new(K * K)
            idx = lambda i, j: g[i * K + j]
            for i in range(K):
                for j in range(K):
                    if i + 1 < K: quads.append((idx(i, j), r, idx(i + 1, j), 0))
                    if j + 1 < K: quads.append((idx(i, j), r, idx(i, j + 1), 0))

    # --- bounded diameter: cliques (spherical-ish) ---
    for K in [6, 10]:
        r = rid; rid += 1
        prof.append((r, f"clique_{K}", "bounded", K - 1, 1, 0.0))
        for _ in range(a.copies * 2):
            cl = new(K)
            for i in range(K):
                for j in range(K):
                    if i != j: quads.append((cl[i], r, cl[j], 0))

    rng.shuffle(quads)
    n = len(quads)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "train.txt"), "w") as f:
        for q in quads[:int(.9 * n)]:
            f.write("\t".join(map(str, q)) + "\n")
    with open(os.path.join(a.out, "test.txt"), "w") as f:
        for q in quads[int(.9 * n):]:
            f.write("\t".join(map(str, q)) + "\n")
    with open(os.path.join(a.out, "relation_profile.csv"), "w") as f:
        f.write("relation_id,family,regime,branching,depth,log_branching\n")
        for row in prof:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"[done] {nxt} entities, {rid} relations, {n} edges -> {a.out}")
    for p in prof:
        print(f"  {p[0]:2d} {p[1]:14s} {p[2]:16s} b={p[3]:3d} d={p[4]:2d} "
              f"log(b)={p[5]}")


if __name__ == "__main__":
    main()
