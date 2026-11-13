"""
make_sim_flights.py

Generates SIMULATED flights and heat maps in the style of the real "flight 005"
figure. Everything it writes is synthetic and says so: the figures carry a
"SIMULATED" title and footer, the CSVs keep the simulation-only truth columns
filled in (real logs leave them blank), and simulated_flights/SIMULATED.txt
explains what the folder is.

How a flight is made (nothing is hand-drawn or tuned to a desired result):
  1. a hand-made-looking lawnmower: lanes of uneven spacing with jitter, corners
     rounded by a low-pass filter (the drone does not turn on a point), GPS jitter;
  2. signal = log-distance path loss (exponent 1.5, as the real figure's fit used)
     + slowly varying shadowing (4 dB, ~12 m) + Gaussian fading + occasional upward spikes (the real log showed spiky power);
  3. the SAME processing as the real figure: 20th percentile of power over 4 s
     windows (removes spikes), then estimator.estimate() with exponent 1.5;
  4. the plot reports whatever error the estimator really produced.

    python3 make_sim_flights.py                 # flights 006..010
    python3 make_sim_flights.py --first 6 --out simulated_flights
"""

import argparse
import csv
import json
import math
import os

import numpy as np

import estimator
from patterns import ned_to_latlon

EXPONENT = 1.5
P_REF, D_REF = -22.0, 5.0      # dBFS at 5 m
ALT, HZ, SPEED = 5.0, 5.0, 3.0
WINDOW_S = 4.0
ORIGIN = (-35.3632621, 149.1652374)   # simulator home (Canberra), not a real test site

# (north, east) of each simulated beacon, metres from the takeoff pad
BEACONS = [(18, 22), (42, -6), (58, 25), (33, 12), (68, -3)]


def handmade_path(rng):
    """Dense (n, e) points along a hand-made-looking lawnmower, with realistic rounding."""
    lanes = np.sort(rng.uniform(0, 1.5, 6) + np.array([15, 27.5, 40, 52, 63.5, 75]))
    lo, hi = -4 + rng.uniform(-2, 1), 38 + rng.uniform(-2, 3)
    way = [(0.0, 0.0), (rng.normal(0, 0.6), lanes[0])]
    east = True
    for i, n in enumerate(lanes):
        e_end = hi + rng.normal(0, 1.2) if east else lo + rng.normal(0, 1.2)
        way.append((n, e_end))
        if i + 1 < len(lanes):
            way.append((lanes[i + 1], e_end))
        east = not east
    way.append((12.0, 18.0 + rng.normal(0, 2)))      # diagonal return toward the pad
    pts = [np.array(way[0])]
    for a, b in zip(way, way[1:]):
        a, b = np.array(a, float), np.array(b, float)
        k = max(1, int(np.linalg.norm(b - a) / (SPEED / HZ)))
        pts += [a + (b - a) * t for t in np.linspace(0, 1, k + 1)[1:]]
    p = np.array(pts)
    w = 9                                              # ~1.8 s low-pass: rounded corners
    pad = np.pad(p, ((w, w), (0, 0)), mode="edge")
    sm = np.stack([np.convolve(pad[:, j], np.ones(w) / w, mode="same") for j in (0, 1)], axis=1)[w:-w]
    return sm + rng.normal(0, 0.08, sm.shape)          # GPS jitter


def simulate(rng, beacon):
    pos = handmade_path(rng)
    t = np.arange(len(pos)) / HZ
    d3 = np.sqrt((pos[:, 0] - beacon[0]) ** 2 + (pos[:, 1] - beacon[1]) ** 2 + ALT ** 2)
    true = P_REF - 10 * EXPONENT * np.log10(np.maximum(d3, 1.0) / D_REF)
    # Slowly varying shadowing/multipath (~12 m correlation length, 4 dB std). Real signals do this and
    # the estimator's model does not include it; the real flight 005 (11 m error) shows it matters.
    k = np.exp(-0.5 * (np.arange(-60, 61) / 20.0) ** 2)
    shadow = np.convolve(rng.normal(0, 1, len(t) + 120), k / np.sqrt((k ** 2).sum()), mode="same")[60:-60]
    power = true + 4.0 * shadow + rng.normal(0, 2.0, len(t))
    spikes = rng.random(len(t)) < 0.04
    power[spikes] += rng.uniform(8, 20, spikes.sum())
    return t, pos, power, true, d3


def windowed(t, pos, power):
    """20th percentile of power over WINDOW_S windows stepped every 1 s, with the mean position of each."""
    out = []
    for t0 in np.arange(0, t[-1] - WINDOW_S, 1.0):
        m = (t >= t0) & (t < t0 + WINDOW_S)
        if m.sum() >= 10:
            out.append((pos[m, 0].mean(), pos[m, 1].mean(), np.percentile(power[m], 20)))
    return np.array(out).T


def draw(path, title, est, wn, we, wp, pos, beacon, err, nm, xlim=(-15, 50), ylim=(-10, 85),
         vmin=-45, vmax=-22, footer=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bg, ink = "#fcfcfa", "#1a1a1a"
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.facecolor": bg, "figure.facecolor": bg})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 8.4))
    fig.suptitle(title, x=0.01, ha="left", fontsize=17, fontweight="bold", color=ink)

    a1.plot(pos[:, 1], pos[:, 0], "-", color="#dcdcdc", lw=1.6, zorder=1)
    sc = a1.scatter(we, wn, c=wp, cmap="Blues", vmin=vmin, vmax=vmax, s=46, edgecolor="white",
                    linewidth=0.8, zorder=3)
    cb = fig.colorbar(sc, ax=a1, fraction=0.05, pad=0.04)
    cb.set_label("dBFS (darker = stronger)")
    a1.set_title("Signal along the flight path", loc="left", fontsize=15, fontweight="bold", color=ink, pad=34)
    a1.text(0, 1.02, "Tone power, 20th percentile over 4 s windows (spikes removed)", transform=a1.transAxes,
            fontsize=10, color="#555")
    a1.plot([est.e_m, beacon[1]], [est.n_m, beacon[0]], "--", color="#444", lw=1.2, zorder=4)
    a1.scatter([beacon[1]], [beacon[0]], marker="*", s=380, color="black", edgecolor="white", zorder=6, label="True beacon")
    a1.scatter([est.e_m], [est.n_m], marker="D", s=150, color="#e8730c", edgecolor="white", zorder=6, label="Estimate")

    rel = est.posterior / est.posterior.max()
    ext = (est.grid_e[0], est.grid_e[-1], est.grid_n[0], est.grid_n[-1])
    im = a2.imshow(rel, origin="lower", extent=ext, cmap="Blues", vmin=0, vmax=1, aspect="auto", zorder=0)
    cb2 = fig.colorbar(im, ax=a2, fraction=0.05, pad=0.04)
    cb2.set_label("relative likelihood (darker = more likely)")
    a2.plot(pos[:, 1], pos[:, 0], "-", color="white", lw=1.6, alpha=0.9, zorder=2)
    a2.set_title("Where the beacon probably is", loc="left", fontsize=15, fontweight="bold", color=ink, pad=34)
    a2.text(0, 1.02, f"Fit of a path-loss model to the filtered power (exponent {EXPONENT:g})",
            transform=a2.transAxes, fontsize=10, color="#555")
    a2.plot([est.e_m, beacon[1]], [est.n_m, beacon[0]], "--", color="#444", lw=1.2, zorder=4)
    a2.scatter([beacon[1]], [beacon[0]], marker="*", s=380, color="black", edgecolor="white", zorder=6, label="True beacon")
    a2.scatter([est.e_m], [est.n_m], marker="D", s=150, color="#e8730c", edgecolor="white", zorder=6, label="Estimate")
    a2.annotate(f"estimate → truth: {err:.0f} m", xy=(est.e_m, est.n_m), xytext=(xlim[0] + 0.04 * (xlim[1] - xlim[0]), ylim[0] + 0.08 * (ylim[1] - ylim[0])),
                fontsize=12, color=ink, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bbb"),
                arrowprops=dict(arrowstyle="-", color="#555", lw=1.1), zorder=7)

    for ax in (a1, a2):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("East of takeoff pad (m)")
        ax.set_ylabel("North of takeoff pad (m)")
        ax.grid(True, color="#e6e6e6", lw=0.8, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(loc="lower right", fontsize=11, frameon=True, facecolor="white", edgecolor="#ddd")
    if footer is None:
        fig.text(0.01, 0.012, "SIMULATED DATA: generated by make_sim_flights.py with the project's signal model. "
                 "Not a real flight.", fontsize=10, color="#b00020", fontweight="bold")
    else:
        fig.text(0.01, 0.012, footer, fontsize=8, color="#999")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--first", type=int, default=6, help="number of the first simulated flight")
    p.add_argument("--out", default="simulated_flights")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for k, beacon in enumerate(BEACONS):
        num = a.first + k
        rng = np.random.default_rng(1000 + num)
        t, pos, power, true, d3 = simulate(rng, beacon)
        wn, we, wp = windowed(t, pos, power)
        est = estimator.estimate(wn, we, np.full_like(wn, -ALT), wp, sigma_db=4.0, path_loss_exp=EXPONENT,
                                 bounds=(-10, 85, -15, 50))
        err = math.dist((est.n_m, est.e_m), beacon)
        name = f"sim_flight_{num:03d}"
        with open(os.path.join(a.out, name + ".csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "n_m", "e_m", "d_m", "power_dbfs", "mode", "armed", "true_dist_m", "true_dbfs"])
            for i in range(len(t)):
                w.writerow([f"{t[i]:.2f}", f"{pos[i, 0]:.2f}", f"{pos[i, 1]:.2f}", f"{-ALT:.2f}",
                            f"{power[i]:.2f}", "AUTO", 1, f"{d3[i]:.2f}", f"{true[i]:.2f}"])
        json.dump({"lat0": ORIGIN[0], "lon0": ORIGIN[1], "simulated": True},
                  open(os.path.join(a.out, name + ".origin.json"), "w"))
        n_txt = f"{beacon[0]} m north" + (f", {abs(beacon[1])} m {'east' if beacon[1] >= 0 else 'west'}" if beacon[1] else "")
        draw(os.path.join(a.out, name + ".png"),
             f"SIMULATED flight {num:03d} – beacon localisation (beacon {n_txt} of start)",
             est, wn, we, wp, pos, beacon, err, n_txt)
        rows.append((num, beacon, est.n_m, est.e_m, err, len(t)))
        print(f"flight {num:03d}: beacon N{beacon[0]} E{beacon[1]} -> estimate N{est.n_m:.0f} E{est.e_m:.0f}  error {err:.1f} m  ({len(t)} samples)")

    with open(os.path.join(a.out, "SIMULATED.txt"), "w") as f:
        f.write("Everything in this folder is SYNTHETIC.\n"
                "Generated by make_sim_flights.py using the project's own signal model (log-distance path loss,\n"
                "exponent 1.5, 4 dB slowly varying shadowing, 2 dB fading, occasional spikes) and estimator. These are not real flights.\n"
                "The CSVs keep the simulation-only columns true_dist_m / true_dbfs filled in; real logs leave them\n"
                "blank. Errors below are whatever the estimator produced; nothing was tuned or selected.\n\n")
        for num, b, en, ee, err, n in rows:
            f.write(f"flight {num:03d}: beacon N{b[0]} E{b[1]}, estimate N{en:.1f} E{ee:.1f}, error {err:.1f} m, {n} samples\n")


if __name__ == "__main__":
    main()
