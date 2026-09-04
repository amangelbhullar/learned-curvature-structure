#!/usr/bin/env python3
"""Volume growth: the graph quantity hyperbolic geometry actually exploits.

Hyperbolic space is characterised by exponential volume growth: in H^2 a disk of
radius r has area 2*pi*(cosh r - 1) ~ e^r. Sarkar's embedding theorem works
because a b-ary tree of depth d has b^d nodes at depth d -- BOTH branching and
depth enter, multiplicatively.

Degree-based proxies (fan-out, Forman-Ricci) capture only the branching term.
A star has enormous degree but its ball stops growing at r=1: no sustained
exponential growth, hence nothing for hyperbolic geometry to exploit. A deep
tree sustains it. This script measures the difference.

Per relation subgraph we compute:
  growth_rate      slope of log|B(v,r)| vs r over the usable radius range
                   (exponential growth => constant positive slope)
  growth_r2        R^2 of that log-linear fit (how exponential is it, really)
  ball_r1..r4      mean |B(v,r)| at radii 1..4
  saturation_r     radius at which growth stops (|B(v,r+1)| < 1.05*|B(v,r)|)
  branch_persist   Spearman(deg(v), mean deg of v's neighbours) -- does
                   branching recur, or is it a single hub layer?
  eff_diameter     90th-percentile shortest path (sampled)

Usage:
  python volume_growth.py --data_dir data/synthetic \
      --stats runs/synthetic/relation_stats.csv --min_triples 10
"""

import argparse, csv, math, os, random, statistics
from collections import defaultdict, deque

MAXR = 8


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
        q.append((h, r, t))
    return q


def ball_sizes(adj, src, maxr=MAXR):
    """|B(src, r)| for r = 0..maxr."""
    seen = {src}
    frontier = [src]
    sizes = [1]
    for _ in range(maxr):
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); nxt.append(v)
        frontier = nxt
        sizes.append(len(seen))
        if not frontier:
            sizes += [len(seen)] * (maxr - len(sizes) + 1)
            break
    return sizes[:maxr + 1]


def loglinear_slope(xs, ys):
    """least-squares slope and R^2 of ys ~ a + b*xs."""
    n = len(xs)
    if n < 3:
        return float("nan"), float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return float("nan"), float("nan")
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return b, r2


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); rk = [0.0]*len(v); i = 0
        while i < len(order):
            j = i
            while j+1 < len(order) and v[order[j+1]] == v[order[i]]: j += 1
            avg = (i+j)/2.0+1.0
            for k in range(i, j+1): rk[order[k]] = avg
            i = j+1
        return rk
    if len(x) < 3:
        return float("nan")
    rx, ry = rank(x), rank(y); n = len(x)
    mx, my = sum(rx)/n, sum(ry)/n
    num = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))
    return num/den if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--stats", required=True, help="relation_stats.csv to augment")
    ap.add_argument("--min_triples", type=int, default=20)
    ap.add_argument("--n_roots", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    quads = read_quads(os.path.join(args.data_dir, "train.txt"))
    adj = defaultdict(lambda: defaultdict(set))
    nodes = defaultdict(set)
    cnt = defaultdict(int)
    for h, r, t in quads:
        adj[r][h].add(t); adj[r][t].add(h)
        nodes[r].update((h, t)); cnt[r] += 1

    rows = list(csv.DictReader(open(args.stats)))
    out = []
    for row in rows:
        r = int(row["relation_id"])
        if cnt.get(r, 0) < args.min_triples:
            row.update(growth_rate="", growth_r2="", sustained_growth="", ball_r1="", ball_r2="",
                       ball_r3="", ball_r4="", saturation_r="",
                       branch_persist="", eff_diameter="")
            out.append(row); continue

        nl = sorted(nodes[r])
        roots = nl if len(nl) <= args.n_roots else rng.sample(nl, args.n_roots)
        curves = [ball_sizes(adj[r], s) for s in roots]
        mean_ball = [statistics.mean(c[i] for c in curves) for i in range(MAXR + 1)]

        # exponential fit over radii where the ball is still growing.
        # Always use at least r=1..2 so saturating families (stars) get a
        # finite, comparable value rather than NaN.
        sat = MAXR
        for i in range(1, MAXR):
            if mean_ball[i + 1] < 1.05 * mean_ball[i]:
                sat = i; break
        fit_hi = max(sat, 3)
        xs = list(range(1, fit_hi + 1))
        ys = [math.log(max(mean_ball[i], 1.0)) for i in xs]
        slope, r2 = loglinear_slope(xs, ys)
        # sustained growth: slope measured only where the ball is still growing
        sus_hi = max(sat, 2)
        sus_xs = list(range(1, sus_hi + 1))
        sus_ys = [math.log(max(mean_ball[i], 1.0)) for i in sus_xs]
        sus_slope, _ = loglinear_slope(sus_xs + [sus_hi + 1],
                                       sus_ys + [math.log(max(mean_ball[min(sus_hi + 1, MAXR)], 1.0))])

        deg = {v: len(adj[r][v]) for v in nl}
        sample = nl if len(nl) <= 400 else rng.sample(nl, 400)
        nbr_deg = [statistics.mean([deg[u] for u in adj[r][v]]) if adj[r][v] else 0.0
                   for v in sample]
        persist = spearman([deg[v] for v in sample], nbr_deg)

        ecc = [max(i for i, s in enumerate(ball_sizes(adj[r], s0))
                   if s == ball_sizes(adj[r], s0)[-1]) for s0 in roots[:30]]
        effd = statistics.median(ecc) if ecc else float("nan")

        row.update(growth_rate=f"{slope:.6f}", growth_r2=f"{r2:.6f}",
                   sustained_growth=f"{sus_slope:.6f}",
                   ball_r1=f"{mean_ball[1]:.3f}", ball_r2=f"{mean_ball[2]:.3f}",
                   ball_r3=f"{mean_ball[3]:.3f}", ball_r4=f"{mean_ball[4]:.3f}",
                   saturation_r=str(sat), branch_persist=f"{persist:.6f}",
                   eff_diameter=f"{effd:.2f}")
        out.append(row)
        print(f"[rel {r:4d}] growth={slope:+.3f} (R2={r2:.2f}) "
              f"balls={mean_ball[1]:.1f}/{mean_ball[2]:.1f}/{mean_ball[3]:.1f} "
              f"sat_r={sat} persist={persist:+.3f}", flush=True)

    with open(args.stats, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"[done] augmented {args.stats}")


if __name__ == "__main__":
    main()
