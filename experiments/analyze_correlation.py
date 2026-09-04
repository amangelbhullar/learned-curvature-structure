#!/usr/bin/env python3
"""Curvature <-> structure correlation: Spearman rho per seed, pooled bootstrap
95% CI, permutation p, and partial rho controlling log(n_triples)."""

import argparse, csv, glob, math, os, random


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else 0.0


def partial_spearman(x, y, z):
    rxy, rxz, ryz = spearman(x, y), spearman(x, z), spearman(y, z)
    den = math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return (rxy - rxz * ryz) / den if den > 0 else 0.0


def bootstrap_ci(x, y, reps=10000, seed=0):
    rng = random.Random(seed)
    n = len(x)
    vals = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(spearman([x[i] for i in idx], [y[i] for i in idx]))
    vals.sort()
    return vals[int(0.025 * reps)], vals[int(0.975 * reps)]


def perm_p(x, y, reps=10000, seed=0):
    rng = random.Random(seed)
    obs = abs(spearman(x, y))
    y2 = list(y)
    hits = 0
    for _ in range(reps):
        rng.shuffle(y2)
        if abs(spearman(x, y2)) >= obs:
            hits += 1
    return (hits + 1) / (reps + 1)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--min_triples", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stats = {int(r["relation_id"]): r for r in read_csv(args.stats)}
    seed_files = sorted(glob.glob(os.path.join(args.run_dir, "curvatures_seed*.csv")))
    if not seed_files:
        raise SystemExit(f"no curvatures_seed*.csv in {args.run_dir}")
    print(f"[info] {len(seed_files)} seed file(s): "
          f"{[os.path.basename(p) for p in seed_files]}")

    covariates = ["temporal_fanout", "static_fanout", "tail_fanin",
                  "tail_entropy", "head_tail_ratio", "one_to_one_frac",
                  "powerlaw_gamma", "forman_ricci", "grc",
                  "clustering", "beta1", "delta_hyp", "ollivier_rc", "ollivier_rc_exact", "longest_path", "reciprocity", "growth_rate", "saturation_r", "branch_persist"]
    rows_out = []
    curv_by_seed = {}
    for path in seed_files:
        seed = os.path.basename(path).split("seed")[1].split(".")[0]
        curv = {int(r["relation_id"]): float(r["curvature"]) for r in read_csv(path)}
        keep = [rid for rid in curv
                if rid in stats and int(stats[rid]["n_triples"]) >= args.min_triples]
        curv_by_seed[seed] = (keep, curv)
        print(f"[seed {seed}] {len(keep)} relations after min_triples filter")

    for cov in covariates:
        rhos, pooled_x, pooled_y, pooled_z = [], [], [], []
        for seed, (keep, curv) in sorted(curv_by_seed.items()):
            kept = [rid for rid in keep
                    if cov in stats[rid]
                    and not math.isnan(float(stats[rid][cov]))]
            if len(kept) < 4:
                continue
            x = [curv[rid] for rid in kept]
            y = [float(stats[rid][cov]) for rid in kept]
            z = [math.log(int(stats[rid]["n_triples"])) for rid in kept]
            rhos.append(spearman(x, y))
            pooled_x += x; pooled_y += y; pooled_z += z
        if len(pooled_x) < 4:
            print(f"{cov:>18}: skipped (insufficient valid data)")
            continue
        lo, hi = bootstrap_ci(pooled_x, pooled_y)
        p = perm_p(pooled_x, pooled_y)
        pr = partial_spearman(pooled_x, pooled_y, pooled_z)
        pooled_rho = spearman(pooled_x, pooled_y)
        mean_rho = sum(rhos) / len(rhos)
        rows_out.append({
            "covariate": cov,
            "rho_per_seed": ";".join(f"{r:.3f}" for r in rhos),
            "rho_pooled": f"{pooled_rho:.3f}",
            "rho_mean_of_seeds": f"{mean_rho:.3f}",
            "ci95": f"[{lo:.3f},{hi:.3f}]",
            "perm_p": f"{p:.4g}",
            "partial_rho_ctrl_logfreq": f"{pr:.3f}",
        })
        print(f"{cov:>18}: rho={pooled_rho:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  "
              f"p={p:.4g}  partial(|log n)={pr:+.3f}  per-seed=({rows_out[-1]['rho_per_seed']})")

    out = args.out or os.path.join(args.run_dir, "correlation_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
