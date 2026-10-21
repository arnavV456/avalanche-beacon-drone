"""
estimator.py

Where is the beacon? Takes the (position, power) samples from logger.py
and returns a probability map over the ground plus a best estimate.

WHY NOT JUST TAKE THE STRONGEST READING
  Fading swings individual readings by 10 dB or more, so the loudest
  sample is usually a lucky one, not the closest one. Instead, every
  sample votes on every candidate location:

    for a candidate beacon position, the path-loss model predicts what
    power the drone should have seen at each sample position; the closer
    those predictions are to what was actually measured, the more likely
    the candidate. All samples count, so fading averages out.

THE MODEL
    P(d) = ref_power - 10 * n * log10(d / ref_dist)  [+ unknown offset]
  d is the 3D drone-to-candidate distance (candidate sits at `beacon_down`
  below the home level, 0 = same level as the takeoff point). Residuals
  are assumed Gaussian in dB with std `sigma_db` -- set it to the fading
  you expect (4-6 dB is a sensible start). The posterior over a uniform
  grid is the heat map; no iterative optimiser, so no local minima.

THE OFFSET (fit_offset=True, default)
  The reference power (-15 dBFS at 4 m) only holds after the props-off
  calibration walk. Until then every reading may be shifted by an unknown
  constant, so the fit solves for that shift per candidate instead of
  trusting the constant. It costs a little accuracy; turn it off once
  `ref_power` is calibrated for the real site.

WHAT IT CANNOT FIX
  - A wrong path-loss exponent (`path_loss_exp`) biases range, not
    direction; test sensitivity before trusting metres.
  - Samples at the noise floor are not Gaussian: loading drops readings
    below `min_power_dbfs`. If the beacon is out of range for much of the
    sweep, you are throwing away information, but not inventing any.
  - r90_m is the honest number: 90% of the probability mass lies within
    that many metres of the estimate. Believe it over the point estimate.

Usage:
    python3 estimator.py samples.csv --plot map.png [--true N E]
    python3 estimator.py --selftest
"""

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass

import numpy as np

from patterns import lawnmower_waypoints, latlon_to_ned, ned_to_latlon

REF_POWER_DBFS = -15.0     # same calibration as search.py / fake_sdr.py
REF_DIST_M = 4.0
PATH_LOSS_EXP = 2.0
MIN_DIST_M = 0.5


@dataclass
class Estimate:
    n_m: float            # MAP estimate, local NED north
    e_m: float            # MAP estimate, local NED east
    mean_n_m: float       # posterior mean (more stable if the map is flat)
    mean_e_m: float
    r90_m: float          # 90% of posterior mass lies within this of (n_m, e_m)
    offset_db: float      # fitted constant shift of the readings at the MAP
    n_samples: int
    grid_n: np.ndarray    # 1-D axes of the posterior grid
    grid_e: np.ndarray
    posterior: np.ndarray  # shape (len(grid_n), len(grid_e)), sums to 1


# ---------------- loading ----------------

def load_samples(path, min_alt_m=3.0, min_power_dbfs=-50.0, modes=None):
    """
    Read logger.py's CSV. Keeps only rows where the drone was armed and at
    least min_alt_m up (drops ground sitting and takeoff), the reading is
    above the noise floor, and -- if `modes` is given -- the flight mode is
    in that set (e.g. {"AUTO"} to use the sweep only).
    Returns four float arrays: n, e, d, power.
    """
    n, e, d, p = [], [], [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["armed"] != "1" or -float(r["d_m"]) < min_alt_m:
                continue
            if modes is not None and r["mode"] not in modes:
                continue
            power = float(r["power_dbfs"])
            if power < min_power_dbfs:
                continue
            n.append(float(r["n_m"]))
            e.append(float(r["e_m"]))
            d.append(float(r["d_m"]))
            p.append(power)
    return tuple(np.array(a) for a in (n, e, d, p))


def load_origin(csv_path):
    """(lat0, lon0) from the .origin.json logger.py wrote beside the CSV, or None."""
    path = os.path.splitext(csv_path)[0] + ".origin.json"
    try:
        with open(path) as f:
            o = json.load(f)
        return o["lat0"], o["lon0"]
    except (OSError, KeyError, ValueError):
        return None


def load_samples_multi(paths, quiet=False, **kw):
    """
    Combine several logger CSVs (e.g. the sweep flight and the refine
    flight after a battery swap). The autopilot's local origin is reset on
    every power-up, so each file is shifted into the FIRST file's frame
    using its .origin.json. A file without one is assumed to share the
    first file's frame, with a warning.
    Returns (n, e, d, power, origin) where origin is the first file's
    (lat0, lon0) or None.
    """
    origin0 = load_origin(paths[0])
    parts = []
    for i, path in enumerate(paths):
        n, e, d, p = load_samples(path, **kw)
        if i > 0 and len(p):
            o = load_origin(path)
            if o is not None and origin0 is not None:
                dn, de = latlon_to_ned(origin0[0], origin0[1], o[0], o[1])
                n, e = n + dn, e + de
                if not quiet:
                    print(f"  {os.path.basename(path)}: shifted into first file's frame "
                          f"by N{dn:+.1f} E{de:+.1f} m")
            elif not quiet:
                print(f"  WARNING: no origin for {os.path.basename(path)}; assuming same "
                      f"frame as {os.path.basename(paths[0])} (wrong if the drone was "
                      f"power-cycled between flights)")
        parts.append((n, e, d, p))
    n, e, d, p = (np.concatenate([x[k] for x in parts]) for k in range(4))
    return n, e, d, p, origin0


# ---------------- estimation ----------------

def estimate(n, e, d, power, *, beacon_down=0.0, sigma_db=5.0,
             path_loss_exp=PATH_LOSS_EXP, ref_power=REF_POWER_DBFS,
             ref_dist=REF_DIST_M, fit_offset=True,
             margin_m=60.0, cell_m=1.0, bounds=None):
    """
    Posterior over beacon position on a grid, from samples at (n, e, d)
    with measured `power` (dBFS, gain-compensated).

    bounds = (n_min, n_max, e_min, e_max) to force the search area;
    default is the bounding box of the samples grown by margin_m, so a
    beacon just outside the swept box can still be found.
    """
    n, e, d, power = (np.asarray(a, float) for a in (n, e, d, power))
    if len(power) < 3:
        raise ValueError("need at least 3 samples")

    if bounds is None:
        n0, n1 = n.min() - margin_m, n.max() + margin_m
        e0, e1 = e.min() - margin_m, e.max() + margin_m
    else:
        n0, n1, e0, e1 = bounds
    grid_n = np.arange(n0, n1 + cell_m, cell_m)
    grid_e = np.arange(e0, e1 + cell_m, cell_m)

    dz2 = (d - beacon_down) ** 2
    logl = np.empty((len(grid_n), len(grid_e)))
    offset = np.empty_like(logl)
    for i, gn in enumerate(grid_n):
        # distances from every candidate in this row to every sample
        dist = np.sqrt((n[None, :] - gn) ** 2 + (e[None, :] - grid_e[:, None]) ** 2
                       + dz2[None, :])
        dist = np.maximum(dist, MIN_DIST_M)
        model = ref_power - 10 * path_loss_exp * np.log10(dist / ref_dist)
        resid = power[None, :] - model
        off = resid.mean(axis=1, keepdims=True) if fit_offset else 0.0
        resid = resid - off
        logl[i] = -np.sum(resid ** 2, axis=1) / (2 * sigma_db ** 2)
        offset[i] = off.ravel() if fit_offset else 0.0

    post = np.exp(logl - logl.max())
    post /= post.sum()

    i, j = np.unravel_index(np.argmax(post), post.shape)
    mean_n = float(np.sum(post.sum(axis=1) * grid_n))
    mean_e = float(np.sum(post.sum(axis=0) * grid_e))

    # radius around the MAP that contains 90% of the probability mass
    cell_dist = np.hypot(grid_n[:, None] - grid_n[i], grid_e[None, :] - grid_e[j]).ravel()
    order = np.argsort(cell_dist)
    cum = np.cumsum(post.ravel()[order])
    r90 = float(cell_dist[order][np.searchsorted(cum, 0.9)])

    return Estimate(float(grid_n[i]), float(grid_e[j]), mean_n, mean_e, r90,
                    float(offset[i, j]), len(power), grid_n, grid_e, post)


def peak_reading_position(n, e, power):
    """Baseline for comparison: where the single loudest sample was taken."""
    k = int(np.argmax(power))
    return float(n[k]), float(e[k])


# ---------------- plotting ----------------

def plot_map(est, n, e, power, path, true_ne=None, decades=4.0, pad_m=15.0):
    """
    Heat map of the posterior on a log scale, zoomed to the swept area.

    Colour is log10(probability / peak probability), floored at -`decades`,
    so a sharp peak still shows a readable gradient around it (a linear
    scale collapses to one bright dot). Because the scale is relative to
    the peak, it shows WHERE the beacon is likely, not how sure we are --
    the dashed r90 circle carries that. The view is the sample bounding
    box grown by pad_m, widened if needed to keep the r90 circle and the
    true beacon in frame.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rel = np.log10(np.maximum(est.posterior / est.posterior.max(), 10.0 ** -decades))

    xs = [e.min() - pad_m, e.max() + pad_m, est.e_m - est.r90_m, est.e_m + est.r90_m]
    ys = [n.min() - pad_m, n.max() + pad_m, est.n_m - est.r90_m, est.n_m + est.r90_m]
    if true_ne is not None:
        xs += [true_ne[1] - pad_m / 2, true_ne[1] + pad_m / 2]
        ys += [true_ne[0] - pad_m / 2, true_ne[0] + pad_m / 2]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    extent = (est.grid_e[0], est.grid_e[-1], est.grid_n[0], est.grid_n[-1])
    im = ax.imshow(rel, origin="lower", extent=extent, cmap="inferno",
                   vmin=-decades, vmax=0, aspect="equal")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("relative likelihood, log10 (0 = most likely)")

    ax.plot(e, n, "-", color="white", lw=0.5, alpha=0.35)
    ax.scatter(e, n, s=5, color="white", alpha=0.7, label="samples")
    ax.plot(est.e_m, est.n_m, "c+", ms=14, mew=2, label=f"estimate (r90 {est.r90_m:.1f} m)")
    ax.add_patch(plt.Circle((est.e_m, est.n_m), est.r90_m, fill=False, ec="c", ls="--"))
    if true_ne is not None:
        ax.plot(true_ne[1], true_ne[0], "x", color="lime", ms=12, mew=2, label="true beacon")

    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(min(ys), max(ys))
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Beacon location likelihood")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------- CLI / self-test ----------------

def _synthetic_sweep(rng, beacon_ne, *, sigma_fade=5.0, true_exp=2.0, true_offset=0.0,
                     length=100, width=100, spacing=15, alt=15, step=0.6, noise_floor=-50.0):
    """Samples along a lawnmower, generated from the model (not from FakeSdr)."""
    wps = lawnmower_waypoints(length, width, spacing, alt)
    pts = []
    for a, b in zip(wps, wps[1:]):
        k = max(1, int(math.dist(a, b) / step))
        for t in np.linspace(0, 1, k, endpoint=False):
            pts.append([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2]])
    pts = np.array(pts)
    dist = np.maximum(np.sqrt((pts[:, 0] - beacon_ne[0]) ** 2 + (pts[:, 1] - beacon_ne[1]) ** 2
                              + pts[:, 2] ** 2), MIN_DIST_M)
    p = (REF_POWER_DBFS - 10 * true_exp * np.log10(dist / REF_DIST_M) + true_offset
         + rng.normal(0, sigma_fade, len(pts)))
    keep = p >= noise_floor
    return pts[keep, 0], pts[keep, 1], pts[keep, 2], p[keep]


def selftest():
    rng = np.random.default_rng(7)

    def trial(beacon, **kw):
        n, e, d, p = _synthetic_sweep(rng, beacon, **kw)
        est = estimate(n, e, d, p, sigma_db=kw.get("sigma_fade", 5.0))
        mp = peak_reading_position(n, e, p)
        return (math.dist((est.n_m, est.e_m), beacon),
                math.dist(mp, beacon), est.r90_m)

    def report(label, beacons, **kw):
        res = np.array([trial(b, **kw) for b in beacons])
        print(f"  {label:<42} MAP err med {np.median(res[:, 0]):5.1f} m "
              f"(p90 {np.percentile(res[:, 0], 90):5.1f}) | "
              f"peak-reading err med {np.median(res[:, 1]):5.1f} m | "
              f"r90 med {np.median(res[:, 2]):4.1f} m")
        return res

    inside = [(rng.uniform(10, 90), rng.uniform(10, 90)) for _ in range(25)]
    print("25 trials/row (10 for outside); 100x100 m box, 7 lanes, 15 m alt, 0.6 m sample spacing")
    base = report("baseline (5 dB fading, model exact)", inside)
    report("heavy fading (8 dB)", inside, sigma_fade=8.0)
    report("hidden 4 dB offset (fit_offset on)", inside, true_offset=4.0)
    report("wrong exponent (truth 2.4, assumed 2.0)", inside, true_exp=2.4)
    outside = [(rng.uniform(-20, 0) if k % 2 else rng.uniform(100, 120), rng.uniform(10, 90))
               for k in range(10)]
    report("beacon 0-20 m OUTSIDE the box", outside)

    # Hard checks on the baseline condition.
    assert np.median(base[:, 0]) < 5.0, "median error should be a few metres"
    assert np.median(base[:, 0]) < np.median(base[:, 1]), "must beat the peak-reading baseline"
    # Calibration: r90 should cover the truth ~90% of the time (allow slack).
    covered = float(np.mean(base[:, 0] <= base[:, 2] * 1.0 + 1e-9))
    print(f"  truth inside r90 in {covered * 100:.0f}% of baseline trials (ideal ~90%)")
    assert covered > 0.7
    print("OK")


def main():
    p = argparse.ArgumentParser(description="Locate the beacon from logger.py samples.")
    p.add_argument("csv", nargs="*", help="one or more logger CSVs (combined)")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--sigma", type=float, default=5.0, help="expected fading std (dB)")
    p.add_argument("--exp", type=float, default=PATH_LOSS_EXP, help="path-loss exponent")
    p.add_argument("--no-offset", action="store_true", help="trust ref_power, don't fit offset")
    p.add_argument("--modes", nargs="*", help="only use samples in these flight modes")
    p.add_argument("--min-alt", type=float, default=3.0)
    p.add_argument("--min-power", type=float, default=-50.0)
    p.add_argument("--true", nargs=2, type=float, metavar=("N", "E"), help="known beacon (sim scoring)")
    p.add_argument("--plot", help="write the heat map PNG here")
    a = p.parse_args()

    if a.selftest:
        return selftest()
    if not a.csv:
        p.error("give a CSV or --selftest")

    n, e, d, pw, origin = load_samples_multi(
        a.csv, min_alt_m=a.min_alt, min_power_dbfs=a.min_power,
        modes=set(a.modes) if a.modes else None)
    print(f"{len(pw)} usable samples (power {pw.min():.1f}..{pw.max():.1f} dBFS)")
    est = estimate(n, e, d, pw, sigma_db=a.sigma, path_loss_exp=a.exp, fit_offset=not a.no_offset)
    print(f"MAP estimate  : N={est.n_m:.1f}  E={est.e_m:.1f}  (r90 = {est.r90_m:.1f} m)")
    if origin is not None:
        lat, lon = ned_to_latlon(origin[0], origin[1], est.n_m, est.e_m)
        print(f"map position  : {lat:.6f}, {lon:.6f}   (paste into a map app)")
    print(f"posterior mean: N={est.mean_n_m:.1f}  E={est.mean_e_m:.1f}")
    print(f"fitted offset : {est.offset_db:+.1f} dB")
    pk = peak_reading_position(n, e, pw)
    print(f"loudest sample: N={pk[0]:.1f}  E={pk[1]:.1f}  (baseline)")
    if a.true:
        print(f"error vs truth: MAP {math.dist((est.n_m, est.e_m), a.true):.1f} m | "
              f"loudest sample {math.dist(pk, a.true):.1f} m")
    if a.plot:
        plot_map(est, n, e, pw, a.plot, true_ne=a.true)
        print(f"heat map -> {a.plot}")


if __name__ == "__main__":
    main()
