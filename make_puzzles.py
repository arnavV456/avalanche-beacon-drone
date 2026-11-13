"""
make_puzzles.py

A pack of "mystery flight" puzzles for a game: each image shows a drone's path
with the received signal strength along it, but NOT where the beacon is.
Friends work out the beacon position from the numbers; ANSWERS.txt has the
solution. All data is FICTIONAL and simulated with the project's signal model,
and each image carries a small tag saying so.

The maths they need (also printed on every image):
    power(dBFS) = P0 - 15 * log10(distance)      distance in metres, 3D
  P0 is not given. Readings are noisy (about 4 dB, slowly varying) and each dot
  is the 20th percentile of the last 4 s. The drone flies at 5 m height.

    python3 make_puzzles.py            # writes puzzles/
"""

import csv
import math
import os

import numpy as np

import estimator
from make_sim_flights import ALT, EXPONENT, HZ, P_REF, D_REF, SPEED, windowed
from make_sim_flights import draw as draw_reveal

OUT = "puzzles"


# ---------------- flight shapes (waypoints in (north, east) metres) ----------------

def spiral_square_in():
    w, lo, hi = [], -8, 62
    a, b = lo, hi
    pts = [(a, a)]
    while b - a > 12:
        pts += [(a, b), (b, b), (b, a + 8), (a + 8, a + 8)]
        a, b = a + 8, b - 8
    return pts


def diagonal_lawnmower():
    pts, k = [], 0
    for c in np.arange(-30, 100, 14):          # lanes where n - e = c
        a, b = np.array([max(c, 0.0), max(-c, 0.0) + 0.0]), None
        e0, e1 = max(-c, 0), min(75, 75 - c)
        if e1 - e0 < 6:
            continue
        p, q = (e0 + c, e0), (e1 + c, e1)
        pts += [p, q] if k % 2 == 0 else [q, p]
        k += 1
    return pts


def circles_in():
    c, pts = np.array([35.0, 30.0]), []
    for r in (34, 25, 16, 8):
        th = np.linspace(0, 2 * math.pi, 40)
        pts += [tuple(c + r * np.array([math.sin(t), math.cos(t)])) for t in th]
    return pts


def sawtooth():
    pts, up = [], True
    for e in np.arange(-5, 75, 10):
        pts += [(2 if up else 70, e), (70 if up else 2, e + 5)]
        up = not up
    return pts


def square_out():
    c = np.array([30.0, 30.0])
    pts, r = [tuple(c)], 6.0
    while r < 40:
        pts += [(c[0] - r, c[1] - r), (c[0] - r, c[1] + r), (c[0] + r, c[1] + r),
                (c[0] + r, c[1] - r - 6)]
        r += 6
    return pts


def box_and_cross():
    return [(0, 0), (0, 70), (70, 70), (70, 0), (0, 0), (70, 70), (35, 35), (70, 0), (0, 70)]


# name, waypoints, beacon (north, east), seed
PUZZLES = [
    ("spiral in", spiral_square_in(), (31, 24), 21),
    ("diagonal sweep", diagonal_lawnmower(), (44, 29), 22),
    ("orbits", circles_in(), (22, 41), 23),
    ("sawtooth", sawtooth(), (38, 52), 24),
    ("expanding square", square_out(), (47, 19), 25),
    ("box and cross", box_and_cross(), (24, 51), 26),
]


def dense_smooth(way, rng):
    pts = [np.array(way[0], float)]
    for a, b in zip(way, way[1:]):
        a, b = np.array(a, float), np.array(b, float)
        k = max(1, int(np.linalg.norm(b - a) / (SPEED / HZ)))
        pts += [a + (b - a) * t for t in np.linspace(0, 1, k + 1)[1:]]
    p = np.array(pts)
    w = 9
    pad = np.pad(p, ((w, w), (0, 0)), mode="edge")
    sm = np.stack([np.convolve(pad[:, j], np.ones(w) / w, mode="same") for j in (0, 1)], axis=1)[w:-w]
    return sm + rng.normal(0, 0.08, sm.shape)


def signal(pos, beacon, rng):
    n = len(pos)
    t = np.arange(n) / HZ
    d3 = np.sqrt((pos[:, 0] - beacon[0]) ** 2 + (pos[:, 1] - beacon[1]) ** 2 + ALT ** 2)
    true = P_REF - 10 * EXPONENT * np.log10(np.maximum(d3, 1.0) / D_REF)
    k = np.exp(-0.5 * (np.arange(-60, 61) / 20.0) ** 2)
    shadow = np.convolve(rng.normal(0, 1, n + 120), k / np.sqrt((k ** 2).sum()), mode="same")[60:-60]
    power = true + 4.0 * shadow + rng.normal(0, 2.0, n)
    spikes = rng.random(n) < 0.04
    power[spikes] += rng.uniform(8, 20, spikes.sum())
    return t, power


def draw(path, idx, name, pos, wn, we, wp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bg, ink = "#fcfcfa", "#1a1a1a"
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.facecolor": bg, "figure.facecolor": bg})
    fig, ax = plt.subplots(figsize=(9, 9.6))
    fig.suptitle(f"Mystery flight #{idx}: where is the beacon?", x=0.02, ha="left", fontsize=17,
                 fontweight="bold", color=ink)
    ax.plot(pos[:, 1], pos[:, 0], "-", color="#dcdcdc", lw=1.8, zorder=1)
    sc = ax.scatter(we, wn, c=wp, cmap="Blues", vmin=np.percentile(wp, 2), vmax=np.percentile(wp, 98),
                    s=52, edgecolor="white", linewidth=0.8, zorder=3)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03).set_label("dBFS (darker = stronger)")
    lo = min(pos.min(), 0) - 8
    hi = pos.max() + 8
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("East of takeoff pad (m)")
    ax.set_ylabel("North of takeoff pad (m)")
    ax.grid(True, color="#e6e6e6", lw=0.8, zorder=0)
    ax.set_title("Signal along the flight path (20th percentile over 4 s windows)", loc="left",
                 fontsize=11, color="#555")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.02, 0.055, "Model:  power = P0 − 15 · log10(distance in m, 3D)    P0 unknown    "
             "drone height 5 m    noise ≈ 4 dB", fontsize=10.5, color="#333")
    fig.text(0.02, 0.025, "Fictional puzzle data (simulated)", fontsize=8, color="#999")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    key = ["ANSWERS (do not look!)", "=" * 22, ""]
    for i, (name, way, beacon, seed) in enumerate(PUZZLES, 1):
        rng = np.random.default_rng(seed)
        pos = dense_smooth(way, rng)
        t, power = signal(pos, beacon, rng)
        wn, we, wp = windowed(t, pos, power)
        est = estimator.estimate(wn, we, np.full_like(wn, -ALT), wp, sigma_db=4.0, path_loss_exp=EXPONENT,
                                 margin_m=20.0)
        err = math.dist((est.n_m, est.e_m), beacon)
        draw(os.path.join(OUT, f"puzzle_{i:02d}.png"), i, name, pos, wn, we, wp)
        lo, hi = min(pos.min(), 0) - 8, pos.max() + 8
        where = f"{beacon[0]} m north, {beacon[1]} m east"
        draw_reveal(os.path.join(OUT, f"puzzle_{i:02d}_answer.png"),
                    f"Mystery flight #{i} \u2013 beacon localisation (beacon {where} of start)",
                    est, wn, we, wp, pos, beacon, err, where, xlim=(lo, hi), ylim=(lo, hi),
                    vmin=float(np.percentile(wp, 2)), vmax=float(np.percentile(wp, 98)),
                    footer="Fictional puzzle data (simulated)")
        with open(os.path.join(OUT, f"puzzle_{i:02d}_data.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["north_m", "east_m", "power_dbfs"])
            for a, b, c in zip(wn, we, wp):
                w.writerow([f"{a:.1f}", f"{b:.1f}", f"{c:.1f}"])
        key.append(f"#{i} ({name}): beacon at {beacon[0]} m north, {beacon[1]} m east.  "
                   f"(the project's estimator gets within {err:.1f} m)")
        print(f"puzzle {i} ({name}): {len(wp)} dots, beacon N{beacon[0]} E{beacon[1]}, estimator error {err:.1f} m")
    open(os.path.join(OUT, "ANSWERS.txt"), "w").write("\n".join(key) + "\n")


if __name__ == "__main__":
    main()
