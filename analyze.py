"""
analyze.py

One command after each flight: turn the logged flight(s) into a heat map,
an estimate of where the beacon is, and (optionally) the next mission.

    python3 analyze.py data/flight_001_*.csv --truth 12.971598 77.594562
    python3 analyze.py data/flight_001_*.csv data/flight_002_*.csv --truth LAT LON

  --truth LAT LON   where you put the beacon, read off your phone. The report
                    then shows how far the estimate is from it. Phone GPS is
                    good to roughly 3-5 m, so that is the floor on how well
                    "error" can mean anything; stand still ~30 s and use a GPS
                    app that shows accuracy. The drone's own GPS adds ~1-2 m.
  --plan FILE       the QGC mission you flew (File > Save in QGC Plan view);
                    drawn dashed so you can see planned vs flown.
  --make-plan       also write the tighter lawnmower around the estimate as
                    refine.plan / refine.waypoints (open in QGC). Skip this
                    flag if you would rather draw your own in QGC.

Outputs go to results/<first-flight>_<n>flights/ (or --out):
    heatmap.png   left: measured signal along the flown path; right: where
                  the beacon probably is (log scale, zoomed)
    map.html      the same on satellite imagery (open in a browser; needs
                  internet for the map tiles). Hover a dot to see its dBFS.
    result.txt / result.json   the numbers
    refine.plan, refine.waypoints   (with --make-plan)

Give it several CSVs after a battery swap: they are lined up using each
file's .origin.json (keep each pair together).
"""

import argparse
import base64
import io
import json
import math
import os

import numpy as np

import estimator
import make_plan
from patterns import latlon_to_ned, ned_to_latlon, refine_waypoints

RSSI_LO, RSSI_HI = -50.0, -10.0     # colour scale range for the signal dots


# ---------------- QGC plan overlay ----------------

def plan_latlons(path):
    """Waypoint (lat, lon) list from a QGC .plan; tolerant of survey items."""
    with open(path) as f:
        plan = json.load(f)
    pts = []

    def walk(item):
        if not isinstance(item, dict):
            return
        if item.get("type") == "SimpleItem":
            params = item.get("params") or []
            if len(params) >= 6 and params[4] is not None and params[5] is not None \
                    and (params[4] != 0 or params[5] != 0) and item.get("command") in (16, 82, 84, 85):
                pts.append((params[4], params[5]))
        elif item.get("type") == "ComplexItem":
            tsc = item.get("TransectStyleComplexItem", {})
            vis = tsc.get("VisualTransectPoints")
            if vis:
                pts.extend((p[0], p[1]) for p in vis)
            else:
                for sub in tsc.get("Items", []):
                    walk(sub)

    for it in plan.get("mission", {}).get("items", []):
        walk(it)
    return pts


# ---------------- figures ----------------

def _posterior_rel(est, decades=4.0):
    return np.log10(np.maximum(est.posterior / est.posterior.max(), 10.0 ** -decades))


def _zoom(ax, ns, es, est, truth_ne, pad=15.0):
    xs = [min(es) - pad, max(es) + pad, est.e_m - est.r90_m, est.e_m + est.r90_m]
    ys = [min(ns) - pad, max(ns) + pad, est.n_m - est.r90_m, est.n_m + est.r90_m]
    if truth_ne is not None:
        xs += [truth_ne[1] - pad / 2, truth_ne[1] + pad / 2]
        ys += [truth_ne[0] - pad / 2, truth_ne[0] + pad / 2]
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(min(ys), max(ys))


def draw_figure(path, est, track, usable, plan_ne, truth_ne, title, decades=4.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tn, te, tp = track
    un, ue, up = usable
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.4))

    # left: signal along the flown path
    a1.plot(te, tn, "-", color="0.75", lw=0.6, zorder=1)
    below = tp < RSSI_LO
    if below.any():
        a1.scatter(te[below], tn[below], s=8, color="0.55", zorder=2, label="below noise cutoff")
    sc = a1.scatter(ue, un, c=up, s=14, cmap="turbo", vmin=RSSI_LO, vmax=RSSI_HI, zorder=3)
    fig.colorbar(sc, ax=a1, fraction=0.046, pad=0.03, label="measured power (dBFS)")
    if plan_ne is not None and len(plan_ne):
        a1.plot(plan_ne[:, 1], plan_ne[:, 0], "--", color="k", lw=1, zorder=2, label="planned mission")
    a1.plot(est.e_m, est.n_m, "k+", ms=14, mew=2, zorder=5, label="estimate")
    if truth_ne is not None:
        a1.plot(truth_ne[1], truth_ne[0], "x", color="magenta", ms=12, mew=2.5, zorder=5,
                label="beacon (phone GPS)")
    a1.set_aspect("equal")
    a1.set_title("Signal along the flown path")
    a1.legend(loc="upper left", fontsize=8, framealpha=0.85)

    # right: posterior heat map
    rel = _posterior_rel(est, decades)
    extent = (est.grid_e[0], est.grid_e[-1], est.grid_n[0], est.grid_n[-1])
    im = a2.imshow(rel, origin="lower", extent=extent, cmap="inferno", vmin=-decades, vmax=0,
                   aspect="equal")
    fig.colorbar(im, ax=a2, fraction=0.046, pad=0.03, label="relative likelihood, log10 (0 = most likely)")
    a2.plot(te, tn, "-", color="white", lw=0.4, alpha=0.4)
    a2.plot(est.e_m, est.n_m, "c+", ms=14, mew=2, label=f"estimate (r90 {est.r90_m:.1f} m)")
    a2.add_patch(plt.Circle((est.e_m, est.n_m), est.r90_m, fill=False, ec="c", ls="--"))
    if truth_ne is not None:
        a2.plot(truth_ne[1], truth_ne[0], "x", color="lime", ms=12, mew=2.5, label="beacon (phone GPS)")
    a2.set_title("Where the beacon probably is")
    a2.legend(loc="upper left", fontsize=8, framealpha=0.85)

    for ax in (a1, a2):
        _zoom(ax, tn, te, est, truth_ne)
        ax.set_xlabel("East of takeoff pad (m)")
        ax.set_ylabel("North of takeoff pad (m)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def posterior_overlay(est, origin, decades=4.0, cutoff=3.5):
    """Transparent RGBA PNG (base64) of the posterior + its lat/lon bounds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rel = _posterior_rel(est, decades)
    keep = np.argwhere(rel > -cutoff)
    pad = 5
    i0, j0 = np.maximum(keep.min(axis=0) - pad, 0)
    i1, j1 = np.minimum(keep.max(axis=0) + pad, np.array(rel.shape) - 1)
    sub = rel[i0:i1 + 1, j0:j1 + 1]
    rgba = plt.get_cmap("inferno")((sub + decades) / decades)
    rgba[..., 3] = np.clip((sub + cutoff) / cutoff, 0, 1) * 0.8
    rgba = rgba[::-1]                       # image row 0 = north
    buf = io.BytesIO()
    plt.imsave(buf, rgba, format="png")
    cell = float(est.grid_n[1] - est.grid_n[0])
    s, w = ned_to_latlon(origin[0], origin[1], est.grid_n[i0] - cell / 2, est.grid_e[j0] - cell / 2)
    nn, ee = ned_to_latlon(origin[0], origin[1], est.grid_n[i1] + cell / 2, est.grid_e[j1] + cell / 2)
    return base64.b64encode(buf.getvalue()).decode(), [[s, w], [nn, ee]]


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Beacon search map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>html,body,#map{height:100%;margin:0} .box{position:absolute;z-index:1000;bottom:28px;left:10px;
background:#fffd;padding:8px 12px;font:13px system-ui;border-radius:6px;max-width:260px}</style>
</head><body><div id="map"></div><div class="box" id="box"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const D = __DATA__;
const sat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {maxZoom: 22, maxNativeZoom: 19, attribution: 'Tiles &copy; Esri'});
const osm = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom: 22, maxNativeZoom: 19,
  attribution: '&copy; OpenStreetMap contributors'});
const map = L.map('map', {layers: [sat]});
map.fitBounds(D.fit);   // a view must exist before other layers are added
const heat = L.imageOverlay('data:image/png;base64,' + D.overlay, D.bounds, {opacity: 1});
const colour = p => { const t = Math.min(1, Math.max(0, (p - D.lo) / (D.hi - D.lo)));
  return 'hsl(' + Math.round(240 - 240 * t) + ',90%,50%)'; };
const dots = L.layerGroup(D.samples.map(s => L.circleMarker([s[0], s[1]], {radius: 4, weight: 0.5, color: '#000',
  fillColor: colour(s[2]), fillOpacity: 0.95}).bindTooltip(s[2].toFixed(1) + ' dBFS')));
const track = L.polyline(D.track, {color: '#fff', weight: 1, opacity: 0.6});
const layers = {'Probability heat map': heat, 'Signal dots (dBFS)': dots, 'Flown path': track};
if (D.plan.length) layers['Planned mission'] = L.polyline(D.plan, {color: '#ff0', weight: 2, dashArray: '6 6'});
const est = L.layerGroup([L.circle(D.est, {radius: D.r90, color: '#0ff', weight: 2, fill: false}),
  L.marker(D.est).bindTooltip('estimate (90% within ' + D.r90.toFixed(1) + ' m)')]);
layers['Estimate'] = est;
if (D.truth) layers['Beacon (phone GPS)'] = L.circleMarker(D.truth, {radius: 8, color: '#f0f', weight: 3, fill: false})
  .bindTooltip('beacon (phone GPS)');
Object.values(layers).forEach(l => l.addTo(map));
L.control.layers({'Satellite': sat, 'Street map': osm}, layers, {collapsed: false}).addTo(map);
document.getElementById('box').innerHTML = D.summary;
</script></body></html>
"""


def write_map_html(path, origin, est, usable_ll, track_ll, plan_ll, truth_ll, est_ll, summary):
    overlay, bounds = posterior_overlay(est, origin)
    lats = [p[0] for p in track_ll] + [est_ll[0]]
    lons = [p[1] for p in track_ll] + [est_ll[1]]
    data = {"overlay": overlay, "bounds": bounds, "samples": usable_ll, "track": track_ll,
            "plan": plan_ll, "est": list(est_ll), "r90": est.r90_m,
            "truth": list(truth_ll) if truth_ll else None, "lo": RSSI_LO, "hi": RSSI_HI,
            "fit": [[min(lats), min(lons)], [max(lats), max(lons)]], "summary": summary}
    with open(path, "w") as f:
        f.write(HTML.replace("__DATA__", json.dumps(data)))


# ---------------- main ----------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="+", help="flight CSV(s) from autolog.py, in flight order")
    p.add_argument("--truth", nargs=2, type=float, metavar=("LAT", "LON"), help="beacon position from your phone")
    p.add_argument("--plan", help="the QGC .plan you flew, to draw planned vs flown")
    p.add_argument("--make-plan", action="store_true", help="write a tighter lawnmower around the estimate")
    p.add_argument("--alt", type=float, default=5.0, help="refine altitude (m above takeoff)")
    p.add_argument("--spacing", type=float, default=5.0, help="refine lane spacing (m)")
    p.add_argument("--speed", type=float, default=3.0, help="refine mission speed (m/s)")
    p.add_argument("--sigma", type=float, default=5.0, help="expected fading std (dB)")
    p.add_argument("--exp", type=float, default=estimator.PATH_LOSS_EXP, help="path-loss exponent")
    p.add_argument("--min-alt", type=float, default=3.0)
    p.add_argument("--min-power", type=float, default=-50.0)
    p.add_argument("--origin", nargs=2, type=float, metavar=("LAT", "LON"),
                   help="override: lat/lon of the takeoff pad (if a .origin.json is missing)")
    p.add_argument("--out", help="output folder")
    p.add_argument("--no-html", action="store_true")
    a = p.parse_args()

    kw = dict(min_alt_m=a.min_alt)
    n, e, d, pw, origin = estimator.load_samples_multi(a.csv, min_power_dbfs=a.min_power, **kw)
    tn, te, _, tp, _ = estimator.load_samples_multi(a.csv, min_power_dbfs=-300.0, quiet=True, **kw)
    if a.origin:
        origin = tuple(a.origin)
    if origin is None:
        p.error("no origin: autolog.py writes <csv>.origin.json once it has a GPS fix; keep it next "
                "to the CSV, or pass --origin LAT LON (the takeoff pad)")
    if len(pw) < 10:
        p.error(f"only {len(pw)} usable samples (armed, >= {a.min_alt:g} m up, above {a.min_power:g} dBFS)")

    est = estimator.estimate(n, e, d, pw, sigma_db=a.sigma, path_loss_exp=a.exp)
    est_ll = ned_to_latlon(origin[0], origin[1], est.n_m, est.e_m)
    pk = estimator.peak_reading_position(n, e, pw)

    truth_ne = truth_ll = err = None
    if a.truth:
        truth_ll = tuple(a.truth)
        truth_ne = latlon_to_ned(origin[0], origin[1], *truth_ll)
        err = math.dist((est.n_m, est.e_m), truth_ne)
        err_pk = math.dist(pk, truth_ne)

    plan_ll = plan_ne = None
    if a.plan:
        plan_ll = plan_latlons(a.plan)
        if plan_ll:
            plan_ne = np.array([latlon_to_ned(origin[0], origin[1], la, lo) for la, lo in plan_ll])
        else:
            print(f"WARNING: found no waypoints in {a.plan}; skipping the planned-mission overlay")

    first = os.path.splitext(os.path.basename(a.csv[0]))[0]
    out = a.out or os.path.join("results", f"{first}_{len(a.csv)}flights")
    os.makedirs(out, exist_ok=True)

    lines = [f"Flights          : {len(a.csv)}  ({', '.join(os.path.basename(c) for c in a.csv)})",
             f"Usable samples   : {len(pw)}  (power {pw.min():.1f} .. {pw.max():.1f} dBFS)",
             f"Estimate         : {est.n_m:.1f} m north, {est.e_m:.1f} m east of the takeoff pad",
             f"Map position     : {est_ll[0]:.6f}, {est_ll[1]:.6f}",
             f"Uncertainty      : 90% inside {est.r90_m:.1f} m of that point",
             f"Fitted offset    : {est.offset_db:+.1f} dB (large = reference power is off; harmless for position)",
             f"Loudest reading  : {pk[0]:.1f} N, {pk[1]:.1f} E  (naive baseline)"]
    if err is not None:
        lines += [f"Beacon (phone)   : {truth_ll[0]:.6f}, {truth_ll[1]:.6f}  -> {truth_ne[0]:.1f} N, {truth_ne[1]:.1f} E",
                  f"ERROR            : {err:.1f} m  (loudest-reading baseline: {err_pk:.1f} m)",
                  "                   includes phone GPS (~3-5 m) and drone GPS (~1-2 m) error"]
    summary = "<b>" + "<br>".join(l.replace("  ", "&nbsp;&nbsp;") for l in [
        f"{len(a.csv)} flight(s), {len(pw)} samples",
        f"estimate {est_ll[0]:.6f}, {est_ll[1]:.6f}",
        f"90% within {est.r90_m:.1f} m"] + ([f"error vs phone: {err:.1f} m"] if err is not None else [])) + "</b>"

    title = (f"{', '.join(os.path.basename(c) for c in a.csv)}   estimate {est.n_m:.1f} N / {est.e_m:.1f} E, "
             f"r90 {est.r90_m:.1f} m" + (f", error {err:.1f} m" if err is not None else ""))
    draw_figure(os.path.join(out, "heatmap.png"), est, (tn, te, tp), (n, e, pw), plan_ne, truth_ne, title)

    if not a.no_html:
        usable_ll = [[*ned_to_latlon(origin[0], origin[1], ni, ei), float(pi)] for ni, ei, pi in zip(n, e, pw)]
        track_ll = [list(ned_to_latlon(origin[0], origin[1], ni, ei)) for ni, ei in zip(tn, te)]
        write_map_html(os.path.join(out, "map.html"), origin, est, usable_ll, track_ll,
                       [list(x) for x in (plan_ll or [])], truth_ll, est_ll, summary)

    if a.make_plan:
        wps, desc = refine_waypoints(n, e, est.n_m, est.e_m, est.r90_m, a.spacing, a.alt)
        ll = [ned_to_latlon(origin[0], origin[1], wn, we) for wn, we, _ in wps]
        with open(os.path.join(out, "refine.plan"), "w") as f:
            json.dump(make_plan.qgc_plan(ll, a.alt, a.speed, origin), f, indent=2)
        with open(os.path.join(out, "refine.waypoints"), "w") as f:
            f.write(make_plan.waypoints_file(ll, a.alt, a.speed, origin))
        lines += ["", f"Refine mission   : {desc}",
                  "Waypoints (lat, lon, alt above takeoff), in case QGC will not load the file:"]
        lines += [f"  {i:2d}  {la:.7f}  {lo:.7f}  {a.alt:g} m" for i, (la, lo) in enumerate(ll, 1)]

    text = "\n".join(lines)
    with open(os.path.join(out, "result.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump({"flights": [os.path.basename(c) for c in a.csv], "samples": int(len(pw)),
                   "origin": list(origin), "estimate": {"n_m": est.n_m, "e_m": est.e_m,
                   "lat": est_ll[0], "lon": est_ll[1], "r90_m": est.r90_m, "offset_db": est.offset_db},
                   "truth": ({"lat": truth_ll[0], "lon": truth_ll[1], "n_m": truth_ne[0], "e_m": truth_ne[1],
                              "error_m": err} if err is not None else None),
                   "params": {"sigma_db": a.sigma, "path_loss_exp": a.exp, "min_alt": a.min_alt,
                              "min_power": a.min_power}}, f, indent=2)

    print("\n" + text)
    print(f"\nWrote to {out}/: heatmap.png" + ("" if a.no_html else ", map.html")
          + ", result.txt, result.json" + (", refine.plan, refine.waypoints" if a.make_plan else ""))


if __name__ == "__main__":
    main()
