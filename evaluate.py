"""
evaluate.py

Monte Carlo evaluation of the whole search pipeline, without SITL:

    lawnmower sweep -> estimate -> plan refine box (refine.plan_refine)
                    -> refine pass -> re-estimate (estimator.estimate)

It reuses the REAL planner and estimator, so a change to either shows up
here in minutes. What it does NOT model: flight dynamics (the drone is
assumed to hit every waypoint exactly and samples are evenly spaced along
the path), terrain, wind, or a beacon whose real behaviour differs from
the log-distance + Gaussian-fading model. Use sitl_batch for the dynamics,
and the props-off field test for the model itself.

Each trial draws a random beacon, runs the pipeline with fresh fading, and
records errors and the estimator's own 90% radius. Output is a table, a
per-trial CSV, and an error-CDF plot.

    python3 evaluate.py                       # default grid, 100 trials/cell
    python3 evaluate.py --trials 30 --fading 5 --placement inside
"""

import argparse
import csv
import math
import time
from multiprocessing import Pool

import numpy as np

import estimator
import refine
from patterns import lawnmower_waypoints, path_length

AREA = 100.0          # sweep box is AREA x AREA m
SWEEP_ALT = 15.0
SWEEP_SPACING = 20.0
SWEEP_SPEED = 5.0     # m/s, ArduCopter WPNAV_SPEED default
REFINE_SPEED = 2.0    # matches refine.py --refine-speed
SAMPLE_HZ = 5.0
NOISE_FLOOR_DBFS = -50.0   # readings below this are dropped by load_samples


def sample_path(vertices, step_m, beacon_ne, rng, fading_db, beacon_down=0.0):
    """Evenly spaced samples along a polyline, with modelled power and fading."""
    pts = []
    for a, b in zip(vertices, vertices[1:]):
        k = max(1, int(math.dist(a, b) / step_m))
        for t in np.linspace(0, 1, k, endpoint=False):
            pts.append([a[i] + t * (b[i] - a[i]) for i in range(3)])
    pts = np.array(pts)
    dist = np.sqrt((pts[:, 0] - beacon_ne[0]) ** 2 + (pts[:, 1] - beacon_ne[1]) ** 2
                   + (pts[:, 2] - beacon_down) ** 2)
    dist = np.maximum(dist, estimator.MIN_DIST_M)
    p = (estimator.REF_POWER_DBFS
         - 10 * estimator.PATH_LOSS_EXP * np.log10(dist / estimator.REF_DIST_M)
         + rng.normal(0, fading_db, len(pts)))
    keep = p >= NOISE_FLOOR_DBFS
    return pts[keep, 0], pts[keep, 1], pts[keep, 2], p[keep]


def draw_beacon(rng, placement):
    if placement == "inside":
        return (rng.uniform(10, AREA - 10), rng.uniform(10, AREA - 10))
    # "outside": 0-20 m beyond one of the four edges, anywhere along it
    off, along = rng.uniform(0, 20), rng.uniform(0, AREA)
    side = rng.integers(4)
    return [(-off, along), (AREA + off, along), (along, -off), (along, AREA + off)][side]


def run_trial(job):
    placement, fading, seed = job
    rng = np.random.default_rng(seed)
    beacon = draw_beacon(rng, placement)
    args = refine.parse_args([])

    # --- sweep ---
    wps = lawnmower_waypoints(AREA, AREA, SWEEP_SPACING, SWEEP_ALT)
    sweep_path = [(0.0, 0.0, -SWEEP_ALT)] + wps
    n, e, d, p = sample_path(sweep_path, SWEEP_SPEED / SAMPLE_HZ, beacon, rng, fading)
    out = dict(placement=placement, fading=fading, seed=seed,
               beacon_n=beacon[0], beacon_e=beacon[1], ok=1)
    if len(p) < 10:     # beacon never heard above the noise floor
        out.update(ok=0)
        return out
    est1 = estimator.estimate(n, e, d, p, sigma_db=fading)
    pk = estimator.peak_reading_position(n, e, p)

    # --- refine ---
    rwps, _ = refine.plan_refine(est1, n, e, args)
    rpath = [wps[-1]] + rwps
    n2, e2, d2, p2 = sample_path(rpath, REFINE_SPEED / SAMPLE_HZ, beacon, rng, fading)
    N, E, D, P = (np.concatenate(x) for x in ((n, n2), (e, e2), (d, d2), (p, p2)))
    est2 = estimator.estimate(N, E, D, P, sigma_db=fading)

    err = lambda ne: math.dist(ne, beacon)
    e1, e2_ = err((est1.n_m, est1.e_m)), err((est2.n_m, est2.e_m))
    flight_s = (path_length(sweep_path) / SWEEP_SPEED + path_length(rpath) / REFINE_SPEED)
    out.update(peak_err=err(pk), sweep_err=e1, sweep_r90=est1.r90_m,
               final_err=e2_, final_r90=est2.r90_m,
               sweep_cov=int(e1 <= est1.r90_m), final_cov=int(e2_ <= est2.r90_m),
               refine_path_m=path_length(rpath), flight_s=flight_s, n_samples=len(P))
    return out


# ---------------- reporting ----------------

def stats(x):
    x = np.asarray(x, float)
    return (f"med {np.median(x):5.1f}  mean {x.mean():5.1f}  "
            f"p90 {np.percentile(x, 90):5.1f}  max {x.max():5.1f}")


def report(rows):
    cells = sorted({(r["placement"], r["fading"]) for r in rows},
                   key=lambda c: (c[0] != "inside", c[1]))
    for placement, fading in cells:
        sel = [r for r in rows if (r["placement"], r["fading"]) == (placement, fading)]
        ok = [r for r in sel if r["ok"]]
        print(f"\n== beacon {placement}, fading {fading:g} dB  "
              f"({len(ok)}/{len(sel)} trials heard the beacon)")
        if not ok:
            continue
        g = lambda k: [r[k] for r in ok]
        print(f"  loudest-reading err (m): {stats(g('peak_err'))}")
        print(f"  after sweep         (m): {stats(g('sweep_err'))}")
        print(f"  after refine        (m): {stats(g('final_err'))}")
        fe = np.array(g("final_err"))
        print(f"  final err < 3 m: {np.mean(fe < 3) * 100:3.0f}%   < 5 m: {np.mean(fe < 5) * 100:3.0f}%   "
              f"< 10 m: {np.mean(fe < 10) * 100:3.0f}%")
        print(f"  refine helped (final < sweep): {np.mean(fe < np.array(g('sweep_err'))) * 100:3.0f}% of trials")
        print(f"  truth within r90: sweep {np.mean(g('sweep_cov')) * 100:3.0f}%  "
              f"final {np.mean(g('final_cov')) * 100:3.0f}%  (ideal 90%)  | "
              f"median final r90 {np.median(g('final_r90')):.1f} m")
        print(f"  flight time ~{np.median(g('flight_s')) / 60:.1f} min "
              f"(refine path {np.median(g('refine_path_m')):.0f} m)")


def plot_cdf(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = sorted({(r["placement"], r["fading"]) for r in rows},
                   key=lambda c: (c[0] != "inside", c[1]))
    fig, axes = plt.subplots(1, len(cells), figsize=(3.6 * len(cells), 3.8),
                             sharey=True, squeeze=False)
    for ax, (placement, fading) in zip(axes[0], cells):
        ok = [r for r in rows if (r["placement"], r["fading"]) == (placement, fading) and r["ok"]]
        for key, label, style in (("peak_err", "loudest reading", ":"),
                                  ("sweep_err", "after sweep", "--"),
                                  ("final_err", "after refine", "-")):
            v = np.sort([r[key] for r in ok])
            ax.step(v, np.arange(1, len(v) + 1) / len(v), style, where="post", label=label)
        ax.set_xlim(0, 25)
        ax.set_title(f"{placement}, {fading:g} dB fading", fontsize=9)
        ax.set_xlabel("position error (m)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("fraction of trials")
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--trials", type=int, default=100, help="trials per cell")
    p.add_argument("--fading", type=float, nargs="+", default=[3.0, 5.0, 8.0])
    p.add_argument("--placement", nargs="+", default=["inside", "outside"],
                   choices=["inside", "outside"])
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="evaluate", help="output prefix (.csv, _cdf.png)")
    a = p.parse_args()

    jobs = [(pl, f, a.seed * 100_000 + ci * 1_000 + t)
            for ci, (pl, f) in enumerate((pl, f) for pl in a.placement for f in a.fading)
            for t in range(a.trials)]
    print(f"{len(jobs)} trials, {a.workers} workers ...")
    t0, rows = time.time(), []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_trial, jobs, chunksize=2), 1):
            rows.append(r)
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)} ({time.time() - t0:.0f} s)")
    print(f"done in {time.time() - t0:.0f} s")

    keys = sorted({k for r in rows for k in r})
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    report(rows)
    plot_cdf(rows, a.out + "_cdf.png")
    print(f"\nper-trial data -> {a.out}.csv, plot -> {a.out}_cdf.png")


if __name__ == "__main__":
    main()
