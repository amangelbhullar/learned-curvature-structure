#!/usr/bin/env python3
"""
Controlled synthetic TKGs for the causal claim (Reviewer 2's missing experiment).

Design: one dataset, many relations, each relation generated with a KNOWN
structural profile so the three notions of "hierarchy" are dissociated by
construction:

  branch_b{B}   : per-(head,tau) branching factor exactly B, fixed depth,
                  perfectly tree-like (delta = 0). Sweep B in {1,2,4,8,16}.
                  -> varies BRANCHING, holds tree-likeness & depth fixed.
  depth_d{D}    : chains of depth D, branching 1, tree-like.
                  -> varies DEPTH, holds branching fixed at 1.
  loopy_p{P}    : branching 4 but with extra cross edges (fraction P) that
                  destroy tree-likeness (delta grows) while keeping the same
                  branching factor.
                  -> varies TREE-LIKENESS, holds branching fixed.

Prediction if the paper's thesis is right: learned curvature orders with B in
the branch family, is flat across the depth family, and is insensitive to P
in the loopy family.

Writes train/valid/test.txt + stat.txt + relation_profile.csv (ground truth)
in RE-GCN format, directly consumable by atth_probe.py / compute_fanout.py.

  python make_synthetic.py --out data/synthetic --n_time 20 --heads_per_rel 60
"""

import argparse, os, random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_time", type=int, default=20)
    ap.add_argument("--heads_per_rel", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    quads, profiles = [], []
    next_ent = 0
    rel_id = 0

    def new_ents(k):
        nonlocal next_ent
        ids = list(range(next_ent, next_ent + k))
        next_ent += k
        return ids

    # family 1: branching sweep (depth fixed at 1, tree-like)
    for B in (1, 2, 4, 8, 16):
        r = rel_id; rel_id += 1
        profiles.append((r, f"branch_b{B}", B, 1, 0.0))
        for _ in range(args.heads_per_rel):
            h = new_ents(1)[0]
            for tau in range(args.n_time):
                for t in new_ents(B):
                    quads.append((h, r, t, tau))

    # family 2: depth sweep (branching fixed at 1)
    for D in (2, 4, 8, 16):
        r = rel_id; rel_id += 1
        profiles.append((r, f"depth_d{D}", 1, D, 0.0))
        for _ in range(args.heads_per_rel):
            chain = new_ents(D + 1)
            for tau in range(args.n_time):
                for a, b in zip(chain, chain[1:]):
                    quads.append((a, r, b, tau))

    # family 3: tree-likeness sweep (branching fixed at 4, cross-edge fraction P)
    for P in (0.0, 0.2, 0.5, 1.0):
        r = rel_id; rel_id += 1
        profiles.append((r, f"loopy_p{P}", 4, 1, P))
        for _ in range(args.heads_per_rel):
            h = new_ents(1)[0]
            kids = new_ents(4)
            for tau in range(args.n_time):
                for t in kids:
                    quads.append((h, r, t, tau))
                n_cross = int(round(P * 4))
                for _ in range(n_cross):
                    a, b = rng.sample(kids, 2)
                    quads.append((a, r, b, tau))

    rng.shuffle(quads)
    n = len(quads)
    splits = {"train": quads[: int(0.8 * n)],
              "valid": quads[int(0.8 * n): int(0.9 * n)],
              "test": quads[int(0.9 * n):]}

    os.makedirs(args.out, exist_ok=True)
    for name, qs in splits.items():
        with open(os.path.join(args.out, f"{name}.txt"), "w") as f:
            for h, r, t, tau in qs:
                f.write(f"{h}\t{r}\t{t}\t{tau}\n")
    with open(os.path.join(args.out, "stat.txt"), "w") as f:
        f.write(f"{next_ent} {rel_id} {args.n_time}\n")
    with open(os.path.join(args.out, "relation_profile.csv"), "w") as f:
        f.write("relation_id,family,branching,depth,cross_frac\n")
        for row in profiles:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"[done] {next_ent} entities, {rel_id} relations, {n} quads -> {args.out}")


if __name__ == "__main__":
    main()
