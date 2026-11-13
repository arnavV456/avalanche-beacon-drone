"""
dashboard.py

Interactive dashboard of logged flights: where the drone actually flew, the
received signal strength (RSSI) along the way and, where known, where the beacon
really was versus where it was detected. One HTML file, one tab per flight plus
an Overview tab comparing them.

    python3 dashboard.py data/flight_*.csv                         # all flights, tabs
    python3 dashboard.py data/flight_005_*.csv --beacon 22.1 -0.45 # one flight
    python3 dashboard.py data/flight_*.csv --beacons data/beacons.json

Reads the CSVs written by autolog.py (and the .origin.json beside each, for the
satellite view). Needs internet for the Plotly/Leaflet libraries and map tiles.

Per flight tab: stat cards; the top-down path coloured by RSSI (switch the
filter); detection with the likelihood heat map; with a known beacon, the REAL
and DETECTED positions, the error and signal against distance; RSSI over time
with flight modes shaded; altitude and speed; a 3-D path; the RSSI distribution;
and the path on satellite imagery.

Real beacon positions come from --beacons FILE (JSON: {"005": [22.1, -0.45]}, metres
north/east in that flight's log coordinates; default data/beacons.json if present)
or --beacon N E for a single flight. Flights without one still get a detection,
but no error.

Files that are not flight logs (missing the t_s, n_m, e_m, d_m, power_dbfs columns)
are skipped with a message.

DETECTION is the method from the project's robust_est.py / plot_heatmap.py:
20th percentile of power in sliding 20-sample windows (step 5), then a grid fit
of  P = A - 10*1.5*log10(distance)  with a Huber-robust cost and A fitted as the
median offset. "Detected" is the best-scoring cell; the 90% region is from the
likelihood surface.

"Low / high envelope" = 20th / 80th percentile of power over a 4 s window around
each sample. The 20th removes upward spikes but, if readings jump between two
levels, it keeps only the low level.

Simulated data (origin file says so, or the simulation-only columns are filled in)
gets a SIMULATED badge on its tab.
"""

import argparse
import csv
import json
import math
import os
import re
import time
from urllib.parse import quote, unquote

import numpy as np

CUTOFF_DBFS = -50.0
REQUIRED = {"t_s", "n_m", "e_m", "d_m", "power_dbfs"}
# detection parameters: identical to robust_est.py / plot_heatmap.py
PCT, WIN, STEP, EXP, SIGMA, MARGIN, RES = 0.20, 20, 5, 1.5, 5.0, 10.0, 1.0


def load(path):
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        raise ValueError("no rows")
    missing = REQUIRED - set(rows[0].keys())
    if missing:
        raise ValueError("not a flight log (missing columns: " + ", ".join(sorted(missing)) + ")")
    f = lambda k: np.array([float(r[k]) for r in rows])
    sim = any((r.get("true_dist_m") or "") != "" for r in rows)
    origin = None
    op = os.path.splitext(path)[0] + ".origin.json"
    if os.path.exists(op):
        o = json.load(open(op))
        origin = (o["lat0"], o["lon0"])
        sim = sim or bool(o.get("simulated"))
    return dict(t=f("t_s"), n=f("n_m"), e=f("e_m"), alt=-f("d_m"), p=f("power_dbfs"),
                mode=[r.get("mode", "") for r in rows], origin=origin, simulated=sim)


def derive(d):
    t, n, e, p = d["t"], d["n"], d["e"], d["p"]
    win = [p[(t >= ti - 2) & (t <= ti + 2)] for ti in t]
    filt = np.array([np.percentile(w, 20) for w in win])
    med = np.array([np.percentile(w, 50) for w in win])
    p80 = np.array([np.percentile(w, 80) for w in win])
    speed = np.zeros(len(t))
    for i, ti in enumerate(t):
        a, b = np.searchsorted(t, ti - 1.0), min(np.searchsorted(t, ti + 1.0), len(t) - 1)
        if b > a:
            speed[i] = np.hypot(n[b] - n[a], e[b] - e[a]) / max(t[b] - t[a], 1e-3)
    segs, start = [], 0
    for i in range(1, len(t) + 1):
        if i == len(t) or d["mode"][i] != d["mode"][start]:
            segs.append([float(t[start]), float(t[i - 1]), d["mode"][start]])
            start = i
    return filt, med, p80, speed, segs


def robust_locate(n, e, h, p):
    """The project's robust localisation (robust_est.py), reproduced exactly."""
    wn, we, wh, wp = [], [], [], []
    for i in range(0, len(p) - WIN, STEP):
        sel = slice(i, i + WIN)
        if h[sel].mean() < 1.0:
            continue
        wp.append(np.sort(p[sel])[int(PCT * (WIN - 1))])
        wn.append(n[sel].mean()); we.append(e[sel].mean()); wh.append(h[sel].mean())
    if len(wp) < 5:
        return None
    wn, we, wh, wp = map(np.array, (wn, we, wh, wp))
    gn = np.arange(n.min() - MARGIN, n.max() + MARGIN + RES, RES)
    ge = np.arange(e.min() - MARGIN, e.max() + MARGIN + RES, RES)
    cost = np.full((len(gn), len(ge)), np.inf)
    for a, bn in enumerate(gn):
        for b, be in enumerate(ge):
            dd = np.sqrt((wn - bn) ** 2 + (we - be) ** 2 + wh ** 2)
            model = -10 * EXP * np.log10(np.maximum(dd, 1.0))
            r = (wp - model - np.median(wp - model)) / SIGMA
            cost[a, b] = np.sum(np.where(np.abs(r) < 1.5, 0.5 * r ** 2, 1.5 * (np.abs(r) - 0.75)))
    post = np.exp(-(cost - cost.min()))
    post_sum = post / post.sum()
    a, b = np.unravel_index(np.argmin(cost), cost.shape)
    mn, me = float((post_sum.sum(1) * gn).sum()), float((post_sum.sum(0) * ge).sum())
    order = np.argsort(post_sum.ravel())[::-1]
    cells = order[: np.searchsorted(np.cumsum(post_sum.ravel()[order]), 0.9) + 1]
    cn, ce = np.unravel_index(cells, post_sum.shape)
    rad = float(np.max(np.hypot(gn[cn] - mn, ge[ce] - me)))
    return dict(best=(float(gn[a]), float(ge[b])), mean=(mn, me), rad=rad, gn=gn, ge=ge,
                post=post / post.max(), windows=int(len(wp)))


def flight_id(path):
    m = re.search(r"flight_(\d+)_(\d{8})-(\d{6})", os.path.basename(path))
    if m:
        d, t = m.group(2), m.group(3)
        return m.group(1), f"{t[0:2]}:{t[2:4]}"
    return os.path.splitext(os.path.basename(path))[0], ""


def build(d, name, when, filename, beacon=None, detect=True, hide_est=False, hide_tab=False, overview_graphs=False, hide_live_signal=False):
    filt, med, p80, speed, segs = derive(d)
    t, n, e, alt, p = d["t"], d["n"], d["e"], d["alt"], d["p"]
    horiz = float(np.hypot(np.diff(n), np.diff(e)).sum())
    best = int(np.argmax(filt))
    org = d["origin"]
    if org:
        from patterns import ned_to_latlon
        ll = [ned_to_latlon(org[0], org[1], a, b) for a, b in zip(n, e)]
        lat, lon = [x[0] for x in ll], [x[1] for x in ll]
    else:
        lat = lon = None
    r2 = lambda a: [round(float(x), 2) for x in a]

    note = None
    if len(t) < 60:
        note = f"Very short log ({len(t)} readings, {t[-1] - t[0]:.0f} s): the charts will be sparse and a beacon estimate may not be possible."
    elif not org:
        note = "No .origin.json beside this CSV, so there is no map position."

    data = {
        "name": name, "when": when, "file": filename, "simulated": d["simulated"], "cutoff": CUTOFF_DBFS, "note": note,
        "t": r2(t), "n": r2(n), "e": r2(e), "alt": r2(alt), "p": r2(p), "filt": r2(filt), "med": r2(med),
        "p80": r2(p80), "speed": r2(speed), "mode": d["mode"], "segs": segs,
        "lat": [round(x, 7) for x in lat] if lat else None, "lon": [round(x, 7) for x in lon] if lon else None,
        "stats": {
            "duration": round(float(t[-1] - t[0]), 1), "samples": int(len(t)), "distance": round(horiz),
            "maxalt": round(float(alt.max()), 1), "medspeed": round(float(np.median(speed)), 1),
            "maxspeed": round(float(speed.max()), 1),
            "pmin": round(float(p.min()), 1), "pmed": round(float(np.median(p)), 1), "pmax": round(float(p.max()), 1),
            "below": round(float((p < CUTOFF_DBFS).mean() * 100)),
            "best": {"filt": round(float(filt[best]), 1), "t": round(float(t[best]), 1),
                     "n": round(float(n[best]), 1), "e": round(float(e[best]), 1)},
        },
        "beacon": None, "est": None, "post": None, "dist": None, "hideEst": False, "hideTab": False, "overviewGraphs": False, "hideLiveSignal": False,
    }

    data["hideEst"] = bool(hide_est)
    data["hideTab"] = bool(hide_tab)
    data["overviewGraphs"] = bool(overview_graphs)
    data["hideLiveSignal"] = bool(hide_live_signal)
    loc = robust_locate(n, e, alt, p) if (beacon or detect) else None
    if loc:
        ll_of = (lambda a, b: [round(x, 7) for x in ned_to_latlon(org[0], org[1], a, b)]) if org else (lambda a, b: None)
        bn, be = loc["best"]
        data["est"] = {"n": round(bn, 1), "e": round(be, 1), "mean_n": round(loc["mean"][0], 1),
                       "mean_e": round(loc["mean"][1], 1), "rad": round(loc["rad"], 1),
                       "ll": ll_of(bn, be), "windows": loc["windows"]}
        data["post"] = {"gn": r2(loc["gn"]), "ge": r2(loc["ge"]),
                        "z": [[round(float(v), 3) for v in row] for row in loc["post"]]}
    if beacon:
        tn, te = beacon
        data["beacon"] = {"n": tn, "e": te,
                          "ll": [round(x, 7) for x in ned_to_latlon(org[0], org[1], tn, te)] if org else None}
        if loc:
            data["est"]["err"] = round(math.hypot(loc["best"][0] - tn, loc["best"][1] - te), 1)
            data["est"]["err_mean"] = round(math.hypot(loc["mean"][0] - tn, loc["mean"][1] - te), 1)
        dist = np.sqrt((n - tn) ** 2 + (e - te) ** 2 + alt ** 2)
        data["dist"] = r2(dist)
        rows = []
        for lo in np.arange(0, dist.max() + 5, 5.0):
            m = (dist >= lo) & (dist < lo + 5)
            if m.sum() >= 5:
                rows.append([round(float(lo + 2.5), 1), round(float(np.percentile(p[m], 20)), 1),
                             round(float(np.percentile(p[m], 50)), 1), round(float(np.percentile(p[m], 80)), 1)])
        data["distbins"] = rows
        # a flight that ends in LAND: how far from the beacon the drone came down (last logged position)
        if d["mode"][-1] == "LAND":
            data["landing"] = {"dist": round(math.hypot(n[-1] - tn, e[-1] - te), 1), "alt": round(float(alt[-1]), 1),
                               "n": round(float(n[-1]), 2), "e": round(float(e[-1]), 2),
                               "ll": [round(x, 7) for x in ned_to_latlon(org[0], org[1], n[-1], e[-1])] if org else None}
    return data


VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm"}
PHOTO_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def scan_media(media_dir, out_dir, exclude=()):
    """
    Videos are matched to a flight by the number in the file name ("flight005.mp4",
    "Flight006.mp4"). Every photo goes in one gallery, ordered by the time in its
    WhatsApp-style name. Paths are relative to the HTML file, so nothing is copied.
    Returns (videos {"005": url}, photos [{src, cap, date}], unused video names).
    """
    videos, photos, unused = {}, [], []
    for root, _, files in os.walk(media_dir):
        for fn in sorted(files):
            path, ext = os.path.join(root, fn), os.path.splitext(fn)[1].lower()
            rel = "/".join(quote(x) for x in os.path.relpath(path, out_dir).split(os.sep))
            if ext in VIDEO_EXT:
                m = re.search(r"flight\s*0*(\d+)", fn, re.I)
                if m:
                    videos[m.group(1).zfill(3)] = rel
                else:
                    unused.append(fn)
            elif ext in PHOTO_EXT and not any(x in fn for x in exclude):
                m = re.search(r"(\d{4})-(\d{2})-(\d{2}) at (\d{2})\.(\d{2})\.(\d{2})", fn)
                key = "".join(m.groups()) if m else time.strftime("%Y%m%d%H%M%S", time.localtime(os.path.getmtime(path)))
                photos.append((key, {"src": rel, "cap": f"{m.group(4)}:{m.group(5)}" if m else "",
                                     "date": f"{m.group(3)}/{m.group(2)}/{m.group(1)}" if m else ""}))
    photos.sort(key=lambda x: x[0])
    return videos, [p for _, p in photos], unused


def svg_search():
    """Illustration (not data): a drone flying a lawnmower grid, signal getting stronger near the beacon."""
    bx, by = 300, 150
    lanes = [50 + 72 * i for i in range(6)]
    pts, up = [(lanes[0], 255)], True
    for i, x in enumerate(lanes):
        pts += [(x, 45 if i % 2 == 0 else 255)]
        if i + 1 < len(lanes):
            pts += [(lanes[i + 1], 45 if i % 2 == 0 else 255)]
    path = " ".join(f"{x},{y}" for x, y in pts)
    dots = []
    for x in lanes:
        for y in range(60, 256, 19):
            s_ = math.exp(-math.hypot(x - bx, y - by) / 85.0)
            dots.append(f'<circle class="dot" cx="{x}" cy="{y}" r="{3.2 + 3.4 * s_:.1f}" style="opacity:{0.18 + 0.82 * s_:.2f}"/>')
    star = " ".join(f"{bx + r * math.sin(math.pi * k / 5):.1f},{by - r * math.cos(math.pi * k / 5):.1f}"
                    for k, r in enumerate([15, 6.5] * 5))
    star2 = " ".join(f"{640 + r * math.sin(math.pi * k / 5):.1f},{150 - r * math.cos(math.pi * k / 5):.1f}"
                     for k, r in enumerate([13, 5.6] * 5))
    return (
        '<svg viewBox="0 0 770 300" role="img" aria-label="Illustration: the drone flies a back-and-forth grid; readings get stronger near the beacon; the software draws a probability map">'
        '<defs><radialGradient id="blobGrad"><stop offset="0" style="stop-color:var(--accent);stop-opacity:.95"/>'
        '<stop offset="0.55" style="stop-color:var(--accent);stop-opacity:.35"/><stop offset="1" style="stop-color:var(--accent);stop-opacity:0"/></radialGradient></defs>'
        '<rect class="box" x="10" y="10" width="440" height="280" rx="18"/>'
        '<text class="m" x="28" y="34" font-size="13">Search area, seen from above</text>'
        f'<polyline class="lane" points="{path}"/>' + "".join(dots) +
        f'<polygon class="star" points="{star}"/><text class="t" x="{bx + 22}" y="{by - 14}" font-size="14">beacon</text>'
        '<path class="arrow" d="M466 150h46"/><path class="arrowhead" d="M505 142l10 8-10 8"/>'
        '<text class="m" x="490" y="132" text-anchor="middle" font-size="12">software</text>'
        '<rect class="box" x="530" y="10" width="230" height="280" rx="18"/>'
        '<text class="m" x="548" y="34" font-size="13">Probability map</text>'
        '<circle cx="640" cy="150" r="88" fill="url(#blobGrad)"/>'
        f'<polygon class="star" points="{star2}"/>'
        '<text class="m" x="645" y="268" text-anchor="middle" font-size="12.5">darker = more likely</text></svg>')


def load_story(path, photos):
    """story.html -> {"hero": html, "about": html}, with {{img:NAME}} and {{svg:search}} filled in."""
    raw = open(path).read()
    srcs = {unquote(p["src"].rsplit("/", 1)[-1]): p["src"] for p in photos}
    out = {}
    for key in ("HERO", "ABOUT"):
        m = re.search(rf"<!--{key}-->(.*?)<!--/{key}-->", raw, re.S)
        out[key.lower()] = m.group(1) if m else ""

    def img(m):
        name = m.group(1).strip()
        if name not in srcs:
            print(f"  story: photo not found in MEDIA, left blank: {name}")
            return ""
        return srcs[name]

    for k in out:
        out[k] = re.sub(r"\{\{img:([^}]*)\}\}", img, out[k]).replace("{{svg:search}}", svg_search())
    return out


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
:root{--bg:#f3f6fb;--card:#fff;--ink:#0f1b2d;--mute:#5a6b82;--line:#e1e8f2;--accent:#2563eb;--accent2:#0d9488;--warm:#f59e0b;
  --shadow:0 1px 2px rgba(15,27,45,.05),0 10px 30px rgba(15,27,45,.07)}
@media (prefers-color-scheme:dark){:root{--bg:#0a111e;--card:#121b2d;--ink:#e8eef8;--mute:#93a5bd;--line:#21304a;--accent:#6aa4ff;--accent2:#2dd4bf;--warm:#fbbf24;
  --shadow:0 1px 2px rgba(0,0,0,.45),0 10px 30px rgba(0,0,0,.35)}}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;-webkit-font-smoothing:antialiased}
h1,h2,h3{letter-spacing:-.02em}
header{padding:18px 24px 4px;display:flex;flex-wrap:wrap;gap:10px;align-items:baseline}
header h1{font-size:22px;margin:0}.sub{color:var(--mute)}
/* hero */
.hero{background:linear-gradient(105deg,rgba(6,14,32,.93) 0%,rgba(6,14,32,.78) 42%,rgba(6,14,32,.28) 100%),var(--hero) center/cover no-repeat;color:#fff}
.hero-in{max-width:1100px;margin:0 auto;padding:84px 24px 70px}
.hero h1{font-size:clamp(38px,6.4vw,68px);line-height:1.02;margin:10px 0 16px;letter-spacing:-.035em}
.hero-sub{font-size:clamp(16px,2vw,21px);max-width:46ch;color:#dbe6f8;margin:0 0 22px}
.hero-chips span{display:inline-block;border:1px solid rgba(255,255,255,.28);background:rgba(255,255,255,.1);backdrop-filter:blur(6px);padding:6px 14px;border-radius:99px;font-size:13px;margin:0 8px 8px 0}
.hero-btns{margin-top:14px}
.btn{display:inline-block;background:#fff;color:#0b1b3a;font-weight:650;padding:12px 22px;border-radius:12px;text-decoration:none;margin:6px 10px 0 0;transition:transform .15s,box-shadow .15s}
.btn:hover{transform:translateY(-1px);box-shadow:0 8px 20px rgba(0,0,0,.3)}
.btn.ghost{background:transparent;color:#fff;border:1px solid rgba(255,255,255,.55)}
.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:12px;font-weight:700;color:var(--accent)}
.eyebrow.light{color:#9cc2ff}
/* tabs */
.tabs{display:flex;flex-wrap:wrap;justify-content:center;gap:8px;padding:10px 16px;position:sticky;top:0;z-index:2000;background:var(--bg);border-bottom:1px solid var(--line)}
@supports (backdrop-filter:blur(8px)){.tabs{background:color-mix(in srgb,var(--bg) 82%,transparent);backdrop-filter:blur(12px)}}
.tabbtn{padding:8px 18px;border:1px solid var(--line);border-radius:99px;background:var(--card);color:var(--ink);cursor:pointer;font:inherit;font-weight:600;transition:all .15s}
.tabbtn:hover{border-color:var(--accent)}
.tabbtn.active{background:var(--accent);border-color:var(--accent);color:#fff;box-shadow:0 4px 14px color-mix(in srgb,var(--accent) 40%,transparent)}
.tabbtn small{opacity:.75;margin-left:6px}
.badge{background:#b00020;color:#fff;font-weight:700;padding:2px 10px;border-radius:99px;font-size:12px}
/* story */
.story{padding-bottom:40px}
.sec{max-width:1100px;margin:0 auto;padding:60px 24px 4px}
.sec h2{font-size:clamp(26px,3.6vw,40px);line-height:1.12;margin:6px 0 14px}
.sec h3{font-size:17px;margin:12px 0 6px}
.lede{font-size:18px;color:var(--mute);max-width:64ch;margin:0 0 12px}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:16px;margin-top:22px}
.icard{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:var(--shadow)}
.icard p{margin:0;color:var(--mute)}
.icon{width:46px;height:46px;border-radius:13px;display:grid;place-items:center;color:var(--accent);background:color-mix(in srgb,var(--accent) 12%,var(--card))}
.ill-wrap{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:14px;box-shadow:var(--shadow);margin:22px 0 8px}
.ill-wrap svg{width:100%;height:auto;display:block}
.ill-wrap.sys{margin-top:26px}
.figcap{color:var(--mute);font-size:13.5px;max-width:78ch;margin:0 0 6px}
.box{fill:var(--card);stroke:var(--line);stroke-width:1.5}.box.strong{stroke:var(--accent);stroke-width:2}
.pill{fill:color-mix(in srgb,var(--accent) 9%,var(--card));stroke:var(--line)}
.t{fill:var(--ink);font:600 15px ui-sans-serif,system-ui,sans-serif}.t.h{font-size:19px;font-weight:700}
.m{fill:var(--mute);font:14px ui-sans-serif,system-ui,sans-serif}
.wave path{fill:none;stroke:var(--accent2);stroke-width:3;stroke-linecap:round}
.arrow{stroke:var(--accent);stroke-width:3;fill:none;stroke-linecap:round}.arrowhead{stroke:var(--accent);stroke-width:3;fill:none;stroke-linecap:round;stroke-linejoin:round}
.lane{fill:none;stroke:var(--mute);stroke-width:2;stroke-dasharray:2 7;stroke-linecap:round;opacity:.7}
.dot{fill:var(--accent)}.star{fill:var(--warm);stroke:var(--card);stroke-width:2}
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin-top:20px}
.step{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px;box-shadow:var(--shadow)}
.step p{margin:0;color:var(--mute);font-size:14px}.step h3{margin-top:8px}
.n{display:inline-grid;place-items:center;width:30px;height:30px;border-radius:50%;background:var(--accent);color:#fff;font-weight:700;font-size:14px}
.split{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:24px;align-items:start;margin-top:22px}
@media (max-width:860px){.split{grid-template-columns:1fr}}
.photos{display:grid;gap:12px}.photos.two{grid-template-columns:1fr 1fr}.photos.three{grid-template-columns:repeat(3,minmax(0,1fr));margin-top:22px}
@media (max-width:760px){.photos.three{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:480px){.photos.three{grid-template-columns:1fr}}
.photos figure{margin:0}.photos img{width:100%;height:250px;object-fit:cover;border-radius:16px;display:block;box-shadow:var(--shadow);background:var(--line)}
.photos.two img{height:340px}
.photos figcaption{font-size:13px;color:var(--mute);padding:7px 3px 0}
.parts{display:grid;gap:10px}
.part{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:12px;padding:12px 16px}
.part b{display:block}.part span{color:var(--mute);font-size:14px}
.more{margin:18px 0 0}.more a{color:var(--accent);font-weight:650;text-decoration:none}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px;margin-top:20px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:var(--shadow)}
.stat b{display:block;font-size:34px;line-height:1.1;letter-spacing:-.03em;color:var(--accent);margin-bottom:8px}
.stat span{color:var(--mute);font-size:14.5px}
.callout{margin-top:18px;border-left:4px solid var(--accent2);background:color-mix(in srgb,var(--accent2) 9%,var(--card));border-radius:0 14px 14px 0;padding:16px 20px;max-width:80ch}
.next{padding-left:20px;max-width:70ch}.next li{margin:10px 0;color:var(--mute)}.next b{color:var(--ink)}
.tags span{display:inline-block;background:var(--card);border:1px solid var(--line);border-radius:99px;padding:6px 14px;margin:0 8px 8px 0;font-size:13.5px}
.fine{color:var(--mute);font-size:13px;margin-top:14px}.sec.last{padding-bottom:30px}
/* data pages */
.note{margin:10px 24px 0;padding:10px 14px;border-radius:12px;background:rgba(245,158,11,.13);border:1px solid rgba(245,158,11,.5)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;padding:16px 24px 6px;max-width:1400px;margin:0 auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:12px 14px;box-shadow:var(--shadow)}
.card.key{border-color:var(--accent)}
.card b{display:block;font-size:20px;letter-spacing:-.01em}.card span{color:var(--mute);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:16px;padding:12px 24px 32px;max-width:1400px;margin:0 auto}
.panel{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:12px;box-shadow:var(--shadow)}
.panel h2{font-size:16px;margin:6px 10px 0}.panel p{margin:2px 10px 8px;color:var(--mute);font-size:13px}
.bigcard{grid-column:1/-1;display:flex;align-items:center;gap:clamp(18px,4vw,44px);padding:clamp(22px,4vw,40px) clamp(22px,5vw,52px);border-radius:24px;color:#fff;
  background:radial-gradient(120% 160% at 0% 0%,#22c55e 0%,#0f766e 55%,#0b4f4a 100%);box-shadow:0 18px 50px rgba(13,148,136,.35)}
.bigcard .num{font-size:clamp(64px,13vw,150px);font-weight:800;letter-spacing:-.05em;line-height:.9;white-space:nowrap}
.bigcard .num small{font-size:.38em;font-weight:700;letter-spacing:-.02em;margin-left:.08em}
.bigcard .txt{min-width:0}.bigcard h2{margin:0 0 8px;font-size:clamp(22px,3.2vw,34px);line-height:1.15;color:#fff}
.bigcard p{margin:0;color:rgba(255,255,255,.88);font-size:clamp(14px,1.6vw,17px);max-width:52ch}
.bigcard .eyebrow{color:rgba(255,255,255,.8);margin-bottom:6px}
@media (max-width:640px){.bigcard{flex-direction:column;align-items:flex-start}}
.howto summary{cursor:pointer;font-weight:700;font-size:16px;padding:6px 10px;list-style:none}
.howto summary::before{content:"\25B8";display:inline-block;margin-right:8px;color:var(--accent);transition:transform .15s}
.howto[open] summary::before{transform:rotate(90deg)}
.howto dl{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px 22px;margin:8px 12px 10px}
.howto dt{font-weight:700}.howto dd{margin:2px 0 0;color:var(--mute);font-size:14px}
.wide{grid-column:1/-1}.plot{width:100%;height:420px}.map{height:560px;border-radius:12px}
table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--mute);font-weight:650;font-size:12px;text-transform:uppercase;letter-spacing:.06em}tr.row{cursor:pointer}tr.row:hover{background:color-mix(in srgb,var(--accent) 9%,transparent)}
.tablewrap{overflow-x:auto}
.vgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:12px;padding:6px 8px 8px}
.vcard video{width:100%;max-height:520px;border-radius:12px;background:#000;display:block}
.vcap{padding:6px 2px;color:var(--mute)}.vcap b{color:var(--ink)}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px;padding:6px 8px 8px}
.gallery figure{margin:0;cursor:zoom-in}.gallery img{width:100%;height:150px;object-fit:cover;border-radius:10px;display:block;background:var(--line);transition:transform .15s}
.gallery img:hover{transform:scale(1.02)}
.gallery figcaption{font-size:11px;color:var(--mute);padding-top:3px}
.live{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:14px;padding:6px 8px 8px;align-items:start}
.livevid video{width:100%;max-height:560px;border-radius:12px;background:#000;display:block}
.liveplots{display:flex;flex-direction:column;gap:6px;min-width:0}
.readout{font:12.5px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;padding:8px 10px;background:var(--bg);border:1px solid var(--line);border-radius:10px;min-height:3.2em}
@media (max-width:900px){.live{grid-template-columns:1fr}}
.lightbox{position:fixed;inset:0;background:rgba(0,0,0,.88);display:none;align-items:center;justify-content:center;z-index:5000;cursor:zoom-out}
.lightbox img{max-width:95vw;max-height:94vh;border-radius:8px}
@media (max-width:560px){.grid{grid-template-columns:1fr;padding:10px}.cards{padding:10px}.hero-in{padding:56px 18px 48px}.sec{padding:44px 18px 4px}.photos.two{grid-template-columns:1fr}.photos.two img{height:260px}}
</style></head><body>
<div id="hero"></div>
<header id="plainhead"><h1>__TITLE__</h1><span class="sub" id="sub"></span></header>
<div class="tabs" id="tabs"></div>
<div id="sections"></div>
<div class="lightbox" id="lb" onclick="this.style.display='none'"><img id="lbimg" alt=""></div>
<script>
const FL = __DATA__;
const MEDIA = __MEDIA__;
const STORY = __STORY__;
if (STORY.hero) { document.getElementById('hero').innerHTML = STORY.hero; document.getElementById('plainhead').style.display = 'none'; }
const cs = 'Viridis', estC = '#e8730c';
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
const ink = getComputedStyle(document.body).color;
const grid = dark ? '#2a3541' : '#e3e7ec', paper = 'rgba(0,0,0,0)', truthC = dark ? '#ffffff' : '#000000';
const base = {paper_bgcolor: paper, plot_bgcolor: paper, font: {color: ink}, margin: {l: 55, r: 20, t: 10, b: 45}};
const cfg = {responsive: true, displaylogo: false};
const PAL = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#0aa5a5', '#d6309e', '#8a9a00', '#a05a2c', '#555555'];
const mounted = {}, maps = {};
document.getElementById('sub').textContent = FL.length + (FL.length === 1 ? ' flight' : ' flights');

function tileLayers() {
  return {sat: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      {maxZoom: 22, maxNativeZoom: 17, attribution: 'Tiles &copy; Esri'}),
    osm: L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom: 22, maxNativeZoom: 19, attribution: '&copy; OpenStreetMap contributors'})};
}

// ---------- tabs ----------
const tabsEl = document.getElementById('tabs'), secEl = document.getElementById('sections');
const keys = (STORY.about ? ['about'] : []).concat(FL.length > 1 ? ['ov'] : []).concat(FL.map((_, i) => i).filter(i => !FL[i].hideTab));
keys.forEach(k => {
  const b = document.createElement('button'); b.className = 'tabbtn'; b.id = 'btn-' + k;
  b.innerHTML = k === 'about' ? 'The project' : k === 'ov' ? (STORY.about ? 'Flight data' : 'Overview') : 'Flight ' + FL[k].name + (FL[k].simulated ? '<small>SIM</small>' : '');
  b.onclick = () => show(k); tabsEl.appendChild(b);
  const s = document.createElement('section'); s.className = 'tab'; s.id = 'tab-' + k; s.hidden = true; if (k === 'about') s.innerHTML = STORY.about; secEl.appendChild(s);
});
function show(k) {
  keys.forEach(x => { document.getElementById('btn-' + x).classList.toggle('active', x === k);
    document.getElementById('tab-' + x).hidden = (x !== k); });
  if (!mounted[k]) { mounted[k] = true; if (k === 'ov') mountOverview(); else if (k !== 'about') mountFlight(k); }
  else { document.querySelectorAll('#tab-' + k + ' .js-plotly-plot').forEach(d => Plotly.Plots.resize(d));
         if (maps[k]) maps[k].invalidateSize(); }
  window.scrollTo({top: k === 'about' ? 0 : Math.max(0, document.getElementById('tabs').offsetTop - 1)});
}

// ---------- overview ----------
function drawHeat(divId, D) {
  const B = D.beacon, E = D.est, showE = !!(E && !D.hideEst);
  const truthTrace = () => ({x: [B.e], y: [B.n], mode: 'markers', name: 'Real beacon', hovertemplate: 'Real beacon<br>N %{y} E %{x}<extra></extra>',
    marker: {symbol: 'star', size: 22, color: truthC, line: {color: dark ? '#000' : '#fff', width: 2}}});
  const estTrace = () => ({x: [E.e], y: [E.n], mode: 'markers', name: 'Detected', hovertemplate: 'Detected<br>N %{y} E %{x}<extra></extra>',
    marker: {symbol: 'diamond', size: 15, color: estC, line: {color: '#fff', width: 2}}});
  const linkTrace = () => ({x: [B.e, E.e], y: [B.n, E.n], mode: 'lines', line: {color: '#888', width: 1.5, dash: 'dash'}, hoverinfo: 'skip', showlegend: false});
  const P = D.post, tr = [{type: 'heatmap', x: P.ge, y: P.gn, z: P.z, zsmooth: false, zmin: 0, zmax: 1,
      colorscale: [[0, '#eaf2fc'], [0.25, '#86b6ef'], [0.5, '#3987e5'], [0.75, '#1c5cab'], [1, '#0d366b']],
      colorbar: {title: {text: 'relative<br>likelihood'}, thickness: 14}, hovertemplate: 'N %{y} E %{x}<br>%{z:.2f}<extra></extra>'},
    {x: D.e, y: D.n, mode: 'lines', line: {color: 'rgba(255,255,255,0.95)', width: 7}, hoverinfo: 'skip', showlegend: false},
    {x: D.e, y: D.n, mode: 'lines', name: 'flight path', line: {color: '#0b1f3a', width: 3}, hoverinfo: 'skip'}];
  const ann = [];
  if (B && showE) { tr.push(linkTrace(), truthTrace(), estTrace());
    ann.push({x: E.e, y: E.n, ax: 70, ay: 60, text: 'estimate → truth: ' + E.err + ' m', showarrow: true, arrowcolor: '#666',
      bgcolor: dark ? '#18202a' : '#fff', bordercolor: grid, font: {color: ink}}); } else if (B) tr.push(truthTrace()); else if (showE) tr.push(estTrace());
  Plotly.newPlot(divId, tr, {...base, margin: {l: 60, r: 20, t: 10, b: 50},
    xaxis: {title: 'East of takeoff pad (m)', gridcolor: grid, zeroline: false},
    yaxis: {title: 'North of takeoff pad (m)', gridcolor: grid, zeroline: false, scaleanchor: 'x'},
    shapes: showE ? [{type: 'circle', xref: 'x', yref: 'y', x0: E.mean_e - E.rad, x1: E.mean_e + E.rad, y0: E.mean_n - E.rad, y1: E.mean_n + E.rad,
      line: {color: estC, width: 1.5, dash: 'dot'}}] : [], annotations: ann, legend: {orientation: 'h', y: -0.15}}, cfg);
}
function drawRssi(divId, D) {
  const shapes = D.segs.map((s, j) => ({type: 'rect', xref: 'x', yref: 'paper', x0: s[0], x1: s[1], y0: 0, y1: 1, line: {width: 0},
    fillcolor: ['rgba(31,111,235,0.10)', 'rgba(45,164,78,0.12)', 'rgba(207,34,46,0.10)', 'rgba(191,135,0,0.12)'][j % 4], layer: 'below'}));
  const notes = D.segs.map(s => ({x: (s[0] + s[1]) / 2, y: 1.0, yref: 'paper', text: s[2], showarrow: false, yanchor: 'bottom', font: {size: 11}}));
  Plotly.newPlot(divId, [
    {x: D.t, y: D.p, mode: 'lines+markers', name: 'raw', line: {color: '#8da2b8', width: 1}, marker: {size: 3}, opacity: 0.7},
    {x: D.t, y: D.filt, mode: 'lines', name: 'low envelope (20th pct, 4 s)', line: {color: '#1f6feb', width: 3}},
    {x: D.t, y: D.med, mode: 'lines', name: 'median (4 s)', line: {color: '#2da44e', width: 2}, visible: 'legendonly'},
    {x: D.t, y: D.p80, mode: 'lines', name: 'high envelope (80th pct, 4 s)', line: {color: '#bf8700', width: 2.5}}],
    {...base, margin: {l: 55, r: 20, t: 28, b: 45}, shapes: [...shapes,
      {type: 'line', xref: 'paper', x0: 0, x1: 1, y0: D.cutoff, y1: D.cutoff, line: {color: '#cf222e', width: 1.5, dash: 'dash'}}],
     annotations: notes, xaxis: {title: 'time since arming (s)', gridcolor: grid}, yaxis: {title: 'dBFS', gridcolor: grid},
     legend: {orientation: 'h', y: -0.28}, hovermode: 'x unified'}, cfg);
}
function graphsHTML() {
  return FL.filter(D => D.overviewGraphs).map(D =>
    (D.post ? '<div class="panel"><h2>Flight ' + D.name + ': where the beacon is' + (D.beacon ? ' (real vs detected)' : '') + '</h2><p>Likelihood surface from the detection method (darker = more likely).' +
      (D.hideEst ? '' : ' Dotted circle = 90% region.') + '</p><div id="ov-heat-' + D.name + '" class="plot" style="height:520px"></div></div>' : '') +
    '<div class="panel"><h2>Flight ' + D.name + ': signal strength over time</h2><p>Raw readings with the 20th and 80th percentile envelopes. The dashed line is the -50 dBFS cutoff the estimator treats as noise. Click a legend entry to show or hide a line.</p>' +
      '<div id="ov-rssi-' + D.name + '" class="plot" style="height:520px"></div></div>').join('');
}
function liveHTML(D, pid) {
  return '<div class="panel wide"><h2>Flight ' + D.name + ' · video with live data</h2>' +
    '<p>' + (D.hideLiveSignal ? 'The red marker follows the video.' : 'The red marker and cursor follow the video. Click the signal chart to jump the video to that moment.') + '</p><div class="live"><div class="livevid">' +
    '<video id="' + pid + '-v" controls preload="metadata" playsinline src="' + D.video + '"></video>' +
    '<div class="vcap"><b>Flight ' + D.name + '</b> · ' + D.when + '</div>' +
    '</div>' +
    '<div class="liveplots"><div id="' + pid + '-ro" class="readout"></div><div id="' + pid + '-lp" class="plot" style="height:330px"></div>' +
    (D.hideLiveSignal ? '' : '<div id="' + pid + '-ls" class="plot" style="height:270px"></div>') + '</div></div></div>';
}
function mountLive(D, pid) {
  const v = document.getElementById(pid + '-v'), ro = document.getElementById(pid + '-ro');
  const B = D.beacon, T = D.t, OFF = D.videoOffset || 0;   // the log starts at this video time (s); set in data/beacons.json
  const idx = tl => { if (tl <= T[0]) return 0; if (tl >= T[T.length - 1]) return T.length - 1;
    let lo = 0, hi = T.length - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (T[m] <= tl) lo = m; else hi = m; } return lo; };
  const lm = {...base, margin: {l: 50, r: 10, t: 8, b: 42}};
  Plotly.newPlot(pid + '-lp', [
    {x: D.e, y: D.n, mode: 'lines', line: {color: dark ? '#667' : '#c3cad3', width: 2}, hoverinfo: 'skip', showlegend: false},
    {x: [], y: [], mode: 'lines', name: 'flown so far', line: {color: '#1f6feb', width: 4}, hoverinfo: 'skip'},
    {x: [D.e[0]], y: [D.n[0]], mode: 'markers', name: 'drone', marker: {size: 17, color: '#cf222e', line: {color: '#fff', width: 2.5}}, hoverinfo: 'skip'}]
    .concat(B ? [{x: [B.e], y: [B.n], mode: 'markers', name: 'Real beacon', marker: {symbol: 'star', size: 20, color: truthC, line: {color: dark ? '#000' : '#fff', width: 2}}}] : [])
    .concat(D.landing ? [{x: [D.landing.e], y: [D.landing.n], mode: 'markers', name: 'Landed ' + D.landing.dist + ' m from the beacon',
      marker: {symbol: 'diamond-open', size: 20, color: '#16a34a', line: {width: 3, color: '#16a34a'}},
      hovertemplate: 'Landing spot<br>' + D.landing.dist + ' m from the real beacon<extra></extra>'}] : []),
    {...lm, xaxis: {title: 'East of takeoff pad (m)', gridcolor: grid, zeroline: false},
     yaxis: {title: 'North (m)', gridcolor: grid, zeroline: false, scaleanchor: 'x'}, legend: {orientation: 'h', y: -0.28}}, cfg);
  const showSig = !D.hideLiveSignal;
  if (showSig) Plotly.newPlot(pid + '-ls', [
    {x: T, y: D.p, mode: 'lines', name: 'raw', line: {color: '#8da2b8', width: 1}, opacity: 0.75},
    {x: T, y: D.filt, mode: 'lines', name: 'low envelope', line: {color: '#1f6feb', width: 2.5}},
    {x: T, y: D.p80, mode: 'lines', name: 'high envelope', line: {color: '#bf8700', width: 2.5}},
    {x: [T[0]], y: [D.p[0]], mode: 'markers', name: 'now', marker: {size: 11, color: '#cf222e', line: {color: '#fff', width: 2}}, hoverinfo: 'skip'}],
    {...lm, margin: {l: 50, r: 10, t: 8, b: 42}, xaxis: {title: 'log time (s)', gridcolor: grid}, yaxis: {title: 'signal (dBFS)', gridcolor: grid},
     shapes: [{type: 'line', xref: 'x', yref: 'paper', x0: T[0], x1: T[0], y0: 0, y1: 1, line: {color: '#cf222e', width: 2}}],
     legend: {orientation: 'h', y: -0.36}, hovermode: 'closest'}, cfg);
  let last = -1;
  function update(force) {
    const tl = v.currentTime - OFF, i = idx(tl);
    const note = tl < T[0] - 0.25 ? 'before the log starts · ' : (tl > T[T.length - 1] + 0.25 ? 'after the log ends · ' : '');
    ro.textContent = note + 'video ' + v.currentTime.toFixed(1) + ' s · log ' + T[i].toFixed(1) + ' s · signal ' + D.p[i] + ' dBFS (low ' + D.filt[i] +
      ' / high ' + D.p80[i] + ') · altitude ' + D.alt[i] + ' m · speed ' + D.speed[i] + ' m/s · ' + D.mode[i];
    if (i === last && !force) return;
    last = i;
    Plotly.restyle(pid + '-lp', {x: [D.e.slice(0, i + 1)], y: [D.n.slice(0, i + 1)]}, [1]);
    Plotly.restyle(pid + '-lp', {x: [[D.e[i]]], y: [[D.n[i]]]}, [2]);
    if (showSig) {
      Plotly.restyle(pid + '-ls', {x: [[T[i]]], y: [[D.p[i]]]}, [3]);
      Plotly.relayout(pid + '-ls', {'shapes[0].x0': T[i], 'shapes[0].x1': T[i]});
    }
  }
  const tick = () => { update(); if (!v.paused && !v.ended) requestAnimationFrame(tick); };
  v.addEventListener('play', tick);
  ['seeked', 'seeking', 'pause', 'loadedmetadata', 'timeupdate'].forEach(ev => v.addEventListener(ev, () => update()));
  if (showSig) document.getElementById(pid + '-ls').on('plotly_click', ev => { v.currentTime = Math.max(0, ev.points[0].x + OFF); update(true); });
  update(true);
}
function videosHTML() {
  return FL.filter(D => D.video).map(D => liveHTML(D, 'ov' + D.name)).join('');
}
let lbIndex = 0;
function lightbox(k) {
  lbIndex = (k + MEDIA.photos.length) % MEDIA.photos.length;
  document.getElementById('lbimg').src = MEDIA.photos[lbIndex].src; document.getElementById('lb').style.display = 'flex';
}
document.addEventListener('keydown', e => { const lb = document.getElementById('lb'); if (lb.style.display !== 'flex') return;
  if (e.key === 'Escape') lb.style.display = 'none'; if (e.key === 'ArrowRight') lightbox(lbIndex + 1); if (e.key === 'ArrowLeft') lightbox(lbIndex - 1); });
function galleryHTML() {
  if (!MEDIA.photos.length) return '';
  return '<div class="panel wide"><h2>Photos from the test day</h2><p>' + MEDIA.photos.length + ' photos. Click one to enlarge; arrow keys move between them.</p><div class="gallery">' +
    MEDIA.photos.map((p, k) => '<figure onclick="lightbox(' + k + ')"><img loading="lazy" src="' + p.src + '" alt="photo ' + (k + 1) + '"><figcaption>' +
      p.cap + '</figcaption></figure>').join('') + '</div></div>';
}
function landingCardHTML() {
  return FL.filter(D => D.landing).map(D =>
    '<div class="bigcard"><div class="num">' + D.landing.dist + '<small> m</small></div><div class="txt"><div class="eyebrow">Flight ' + D.name + ' \u00b7 final landing</div>' +
    '<h2>The drone landed only ' + D.landing.dist + ' metres from the actual beacon.</h2>' +
    '<p>The distance between the spot where the drone came down and the real position of the beacon, measured on the ground.</p></div></div>').join('');
}
function howToHTML() {
  return '<details class="panel wide howto" open><summary>How to read this page</summary><dl>' +
    '<div><dt>Signal strength (dBFS)</dt><dd>How loudly the drone’s receiver hears the beacon. Numbers closer to zero are stronger, so −20 is much stronger than −45. It rises as the drone gets closer.</dd></div>' +
    '<div><dt>The flight path</dt><dd>Each dot is one reading, placed where the drone was when it took it. Darker or yellower dots are stronger readings.</dd></div>' +
    '<div><dt>The heat map</dt><dd>The software’s guess at where the beacon is. The darker the blue, the more likely the beacon is at that spot.</dd></div>' +
    '<div><dt>Real vs detected</dt><dd>“Real” is where the beacon was, measured on the ground. “Detected” is where the software thought it was. The gap between them is the error.</dd></div>' +
    '<div><dt>Low and high envelopes</dt><dd>The raw readings jump between strong and weak, so we also show a smoothed lower line and upper line that follow the readings.</dd></div>' +
    '<div><dt>The videos</dt><dd>Footage of each flight. The red marker and cursor beside it follow the video, so you can see what the drone was hearing at each moment.</dd></div>' +
    '</dl></details>';
}
function mountOverview() {
  const root = document.getElementById('tab-ov');
  const fmt = (v, u) => v === null || v === undefined ? '–' : v + (u || '');
  const rows = FL.map((D, i) => {
    const S = D.stats, E = D.est, B = D.beacon;
    return '<tr' + (D.hideTab ? '' : ' class="row" onclick="show(' + i + ')"') + '><td><b style="color:' + PAL[i % PAL.length] + '">&#9632;</b> <b>' + D.name + '</b>' +
      (D.simulated ? ' <span class="badge">SIM</span>' : '') + '</td><td>' + S.duration + ' s</td><td>' + S.distance + ' m</td><td>' +
      S.maxalt + ' m</td><td>' + S.pmin + ' / ' + S.pmed + ' / ' + S.pmax + '</td><td>' +
      (E ? 'N ' + E.n + ', E ' + E.e : '–') + '</td><td>' +
      (E && E.err !== undefined ? '<b>' + E.err + ' m</b>' : '–') + '</td></tr>';
  }).join('');
  root.innerHTML = '<div class="grid">' + landingCardHTML() + howToHTML() + '<div class="panel wide"><h2>All flights</h2><p>Click a row to open that flight. Signal columns are min / median / max in dBFS. ' +
    'The error is the distance from the real beacon to the detected position, and is only shown where the real position is known.</p><div class="tablewrap"><table><thead><tr>' +
    '<th>Flight</th><th>Duration</th><th>Distance</th><th>Max alt</th><th>Signal (dBFS)</th><th>Detected</th><th>Error</th>' +
    '</tr></thead><tbody>' + rows + '</tbody></table></div></div>' + videosHTML() + graphsHTML() +
    '<div class="panel wide"><h2>All flights on the map</h2><p>Each flight is a different colour (same colour as its square in the table). Use the layer box to hide flights.' +
    (FL.some(D => D.landing) ? ' The green ring marks where a flight came down, with its distance from the real beacon.' : '') + '</p>' +
    '<div id="ov-map" class="map"></div></div>' + galleryHTML() + '</div>';
  FL.filter(D => D.video).forEach(D => mountLive(D, 'ov' + D.name));
  FL.filter(D => D.overviewGraphs).forEach(D => { if (D.post) drawHeat('ov-heat-' + D.name, D); drawRssi('ov-rssi-' + D.name, D); });
  const withLL = FL.map((D, i) => [D, i]).filter(x => x[0].lat);
  if (!withLL.length) { document.getElementById('ov-map').innerHTML = '<p style="padding:20px">No flight has a map position.</p>'; return; }
  const T = tileLayers(), map = L.map('ov-map', {layers: [T.sat], scrollWheelZoom: false});
  map.on('click', () => map.scrollWheelZoom.enable());
  const all = [], overlays = {};
  withLL.forEach(([D, i]) => {
    const g = L.layerGroup(), c = PAL[i % PAL.length], pts = D.lat.map((a, j) => [a, D.lon[j]]); all.push(...pts);
    L.polyline(pts, {color: c, weight: 3, opacity: 0.9}).bindTooltip('Flight ' + D.name).addTo(g);
    L.circleMarker(pts[0], {radius: 5, color: '#fff', weight: 1.5, fillColor: c, fillOpacity: 1}).bindTooltip('Flight ' + D.name + ' start').addTo(g);
    if (D.est && D.est.ll && !D.hideEst) { L.circleMarker(D.est.ll, {radius: 8, color: '#fff', weight: 2, fillColor: estC, fillOpacity: 1})
      .bindTooltip('Detected, flight ' + D.name, {direction: 'top'}).addTo(g); all.push(D.est.ll); }
    if (D.beacon && D.beacon.ll) { L.circleMarker(D.beacon.ll, {radius: 8, color: '#fff', weight: 2, fillColor: '#000', fillOpacity: 1})
      .bindTooltip('Real beacon, flight ' + D.name, {direction: 'top'}).addTo(g); all.push(D.beacon.ll); }
    if (D.landing && D.landing.ll) {
      if (D.beacon && D.beacon.ll) L.polyline([D.landing.ll, D.beacon.ll], {color: '#16a34a', weight: 3, dashArray: '4 4'}).addTo(g);
      L.circleMarker(D.landing.ll, {radius: 15, color: '#16a34a', weight: 4, fill: false})
        .bindTooltip('Flight ' + D.name + ' landed ' + D.landing.dist + ' m from the real beacon', {permanent: true, direction: 'bottom', offset: [0, 12]}).addTo(g);
      all.push(D.landing.ll);
    }
    g.addTo(map); overlays['<span style="color:' + c + '">&#9632;</span> Flight ' + D.name] = g;
  });
  map.fitBounds(L.latLngBounds(all).pad(0.3), {maxZoom: 19});
  L.control.layers({'Satellite': T.sat, 'Street map': T.osm}, overlays, {collapsed: false}).addTo(map);
  maps.ov = map;
}

// ---------- one flight ----------
function mountFlight(i) {
  const D = FL[i], S = D.stats, B = D.beacon, E = D.est, p = 'f' + i, id = s => p + '-' + s;
  const showE = !!(E && !D.hideEst);   // draw the orange detected marker (and its circle/link)?
  const root = document.getElementById('tab-' + i);
  const hasHeat = !!(E && D.post), hasDist = !!(B && D.dist);
  root.innerHTML =
    '<div class="cards" id="' + id('cards') + '"></div>' + (D.note ? '<div class="note">' + D.note + '</div>' : '') +
    '<div class="grid">' + (D.video ? liveHTML(D, p + 'v') : '') +
    '<div class="panel"><h2>Flight path, coloured by signal strength</h2><p>Top-down view. Each dot is one reading; darker/yellower = stronger. Buttons switch between the raw readings and filtered versions: the 20th percentile keeps the low readings, the 80th the high ones, the median is in between.</p><div id="' + id('path') + '" class="plot" style="height:560px"></div></div>' +
    '<div class="panel"><h2>On the map</h2><p>Satellite view of the same path, dots coloured by the median signal.</p><div id="' + id('map') + '" class="map"></div></div>' +
    (hasHeat ? '<div class="panel"><h2>' + (B ? 'Where the beacon is: real vs detected' : 'Where the beacon probably is') + '</h2><p>Likelihood surface from the detection method (darker = more likely). Dotted circle = 90% region.</p><div id="' + id('heat') + '" class="plot" style="height:520px"></div></div>' : '') +
    (hasDist ? '<div class="panel"><h2>Signal against distance to the real beacon</h2><p>Grey dots: raw readings. Lines: 20th / 50th / 80th percentile in 5 m bins. A beacon should get stronger as distance falls (axis runs far to near).</p><div id="' + id('dist') + '" class="plot" style="height:520px"></div></div>' : '') +
    '<div class="panel wide"><h2>Signal strength over time</h2><p>Shaded bands show the flight mode. The dashed line is the -50 dBFS cutoff the estimator treats as noise. Click a legend entry to show or hide a line.</p><div id="' + id('rssi') + '" class="plot" style="height:340px"></div></div>' +
    '<div class="panel"><h2>Altitude and speed</h2><p>Height above the takeoff point and ground speed.</p><div id="' + id('alt') + '" class="plot"></div></div>' +
    '<div class="panel"><h2>3-D path</h2><p>Drag to rotate. Colour = low-envelope signal strength.</p><div id="' + id('p3d') + '" class="plot"></div></div>' +
    '<div class="panel"><h2>Distribution of readings</h2><p>How often each signal level occurred.</p><div id="' + id('hist') + '" class="plot"></div></div></div>';

  if (D.video) mountLive(D, p + 'v');
  const sorted = [...D.p].sort((a, b) => a - b);
  const cmin = Math.floor(sorted[Math.floor(0.02 * sorted.length)]), cmax = Math.ceil(sorted[Math.floor(0.98 * sorted.length)]);
  const cards = [];
  if (D.simulated) cards.push(['Data', 'SIMULATED', 1]);
  if (B && E) cards.push(['Real beacon', 'N ' + B.n + ', E ' + B.e + ' m', 1], ['Detected', 'N ' + E.n + ', E ' + E.e + ' m', 1],
    ['Detection error', E.err + ' m', 1], ['90% region', 'radius ~' + E.rad + ' m', 1]);
  else if (E) cards.push(['Detected beacon', 'N ' + E.n + ', E ' + E.e + ' m', 1], ['90% region', 'radius ~' + E.rad + ' m', 1]);
  if (D.landing) cards.push(['Landed from the real beacon', D.landing.dist + ' m', 1]);
  cards.push(['Duration', S.duration + ' s'], ['Distance flown', S.distance + ' m'], ['Max altitude', S.maxalt + ' m'],
    ['Median / max speed', S.medspeed + ' / ' + S.maxspeed + ' m/s'], ['Signal range', S.pmin + ' to ' + S.pmax + ' dBFS'],
    ['Median signal', S.pmed + ' dBFS'], ['Strongest, low envelope', S.best.filt + ' dBFS'],
    ['…at', 'N ' + S.best.n + ', E ' + S.best.e + ' m (t=' + S.best.t + ' s)'], ['Below noise cutoff', S.below + ' %']);
  document.getElementById(id('cards')).innerHTML = cards.map(c => '<div class="card' + (c[2] ? ' key' : '') + '"><b>' + c[1] + '</b><span>' + c[0] + '</span></div>').join('');

  const hov = D.t.map((t, j) => [t, D.p[j], D.filt[j], D.alt[j], D.speed[j], D.mode[j]]);
  const tmpl = 't = %{customdata[0]} s<br>raw %{customdata[1]} dBFS<br>low envelope %{customdata[2]} dBFS<br>alt %{customdata[3]} m, speed %{customdata[4]} m/s<br>mode %{customdata[5]}<extra></extra>';
  const mk = c => ({color: c, colorscale: cs, cmin, cmax, size: 8, line: {color: 'white', width: 0.6}, colorbar: {title: {text: 'dBFS'}, thickness: 14}});
  const truthTrace = () => ({x: [B.e], y: [B.n], mode: 'markers', name: 'Real beacon', hovertemplate: 'Real beacon<br>N %{y} E %{x}<extra></extra>',
    marker: {symbol: 'star', size: 22, color: truthC, line: {color: dark ? '#000' : '#fff', width: 2}}});
  const estTrace = () => ({x: [E.e], y: [E.n], mode: 'markers', name: 'Detected', hovertemplate: 'Detected<br>N %{y} E %{x}<extra></extra>',
    marker: {symbol: 'diamond', size: 15, color: estC, line: {color: '#fff', width: 2}}});
  const linkTrace = () => ({x: [B.e, E.e], y: [B.n, E.n], mode: 'lines', line: {color: '#888', width: 1.5, dash: 'dash'}, hoverinfo: 'skip', showlegend: false});

  const pathTraces = [
    {x: D.e, y: D.n, mode: 'lines', line: {color: dark ? '#556' : '#c8ced6', width: 2}, hoverinfo: 'skip', showlegend: false},
    {x: D.e, y: D.n, mode: 'markers', marker: mk(D.filt), customdata: hov, hovertemplate: tmpl, name: 'readings', showlegend: false},
    {x: [D.e[0]], y: [D.n[0]], mode: 'markers+text', text: ['start'], textposition: 'top right',
     marker: {symbol: 'triangle-up', size: 13, color: '#2da44e', line: {color: 'white', width: 1}}, name: 'start', hoverinfo: 'skip'},
    {x: [D.e[D.e.length - 1]], y: [D.n[D.n.length - 1]], mode: 'markers+text', text: ['end'], textposition: 'bottom right',
     marker: {symbol: 'square', size: 11, color: '#cf222e', line: {color: 'white', width: 1}}, name: 'end', hoverinfo: 'skip'}];
  if (B && showE) pathTraces.push(linkTrace(), truthTrace(), estTrace()); else if (B) pathTraces.push(truthTrace()); else if (showE) pathTraces.push(estTrace());
  Plotly.newPlot(id('path'), pathTraces, {...base, margin: {l: 60, r: 20, t: 40, b: 50},
    xaxis: {title: 'East of takeoff pad (m)', gridcolor: grid, zeroline: false},
    yaxis: {title: 'North of takeoff pad (m)', gridcolor: grid, zeroline: false, scaleanchor: 'x'}, legend: {orientation: 'h', y: 1.08},
    updatemenus: [{type: 'buttons', direction: 'right', x: 0, y: 1.14, xanchor: 'left', buttons: [
      {label: 'Low (20th pct)', method: 'restyle', args: [{'marker.color': [D.filt]}, [1]]},
      {label: 'Median', method: 'restyle', args: [{'marker.color': [D.med]}, [1]]},
      {label: 'High (80th pct)', method: 'restyle', args: [{'marker.color': [D.p80]}, [1]]},
      {label: 'Raw', method: 'restyle', args: [{'marker.color': [D.p]}, [1]]}]}]}, cfg);

  if (hasHeat) drawHeat(id('heat'), D);
  if (hasDist) {
    const bins = D.distbins;
    Plotly.newPlot(id('dist'), [
      {x: D.dist, y: D.p, mode: 'markers', name: 'raw readings', marker: {size: 4, color: dark ? '#6b7a89' : '#aab4bf', opacity: 0.7}},
      {x: bins.map(b => b[0]), y: bins.map(b => b[3]), mode: 'lines+markers', name: '80th pct', line: {color: '#bf8700', width: 3}},
      {x: bins.map(b => b[0]), y: bins.map(b => b[2]), mode: 'lines+markers', name: 'median', line: {color: '#2da44e', width: 3}},
      {x: bins.map(b => b[0]), y: bins.map(b => b[1]), mode: 'lines+markers', name: '20th pct', line: {color: '#1f6feb', width: 3}}],
      {...base, xaxis: {title: 'distance from the drone to the real beacon (m, 3-D)', gridcolor: grid, autorange: 'reversed'},
       yaxis: {title: 'dBFS', gridcolor: grid}, legend: {orientation: 'h', y: -0.2}, hovermode: 'closest'}, cfg);
  }

  drawRssi(id('rssi'), D);

  Plotly.newPlot(id('alt'), [
    {x: D.t, y: D.alt, name: 'altitude (m)', mode: 'lines', line: {color: '#1f6feb', width: 2.5}},
    {x: D.t, y: D.speed, name: 'speed (m/s)', mode: 'lines', line: {color: '#bf8700', width: 2}, yaxis: 'y2'}],
    {...base, xaxis: {title: 'time since arming (s)', gridcolor: grid}, yaxis: {title: 'altitude (m)', gridcolor: grid},
     yaxis2: {title: 'speed (m/s)', overlaying: 'y', side: 'right', showgrid: false}, legend: {orientation: 'h', y: -0.25}, hovermode: 'x unified'}, cfg);

  const t3 = [{type: 'scatter3d', x: D.e, y: D.n, z: D.alt, mode: 'lines+markers', name: 'path',
    line: {color: D.filt, colorscale: cs, cmin, cmax, width: 5}, marker: {size: 3, color: D.filt, colorscale: cs, cmin, cmax}, customdata: hov, hovertemplate: tmpl}];
  if (B) t3.push({type: 'scatter3d', x: [B.e], y: [B.n], z: [0], mode: 'markers', name: 'Real beacon', marker: {size: 7, color: truthC, symbol: 'diamond'}});
  if (showE) t3.push({type: 'scatter3d', x: [E.e], y: [E.n], z: [0], mode: 'markers', name: 'Detected', marker: {size: 7, color: estC, symbol: 'diamond'}});
  Plotly.newPlot(id('p3d'), t3, {...base, margin: {l: 0, r: 0, t: 0, b: 0}, scene: {xaxis: {title: 'East (m)'}, yaxis: {title: 'North (m)'},
    zaxis: {title: 'Altitude (m)'}, aspectmode: 'data'}, showlegend: false}, cfg);

  Plotly.newPlot(id('hist'), [{type: 'histogram', x: D.p, xbins: {size: 1}, marker: {color: '#1f6feb'}, name: 'readings'}],
    {...base, xaxis: {title: 'dBFS', gridcolor: grid}, yaxis: {title: 'number of readings', gridcolor: grid},
     shapes: [{type: 'line', x0: D.cutoff, x1: D.cutoff, yref: 'paper', y0: 0, y1: 1, line: {color: '#cf222e', dash: 'dash', width: 2}},
              {type: 'line', x0: S.pmed, x1: S.pmed, yref: 'paper', y0: 0, y1: 1, line: {color: '#2da44e', width: 2}}],
     annotations: [{x: D.cutoff, yref: 'paper', y: 1, text: 'noise cutoff', showarrow: false, xanchor: 'left', font: {color: '#cf222e', size: 11}},
                   {x: S.pmed, yref: 'paper', y: 0.92, text: 'median', showarrow: false, xanchor: 'left', font: {color: '#2da44e', size: 11}}]}, cfg);

  if (D.lat) {
    const T = tileLayers(), map = L.map(id('map'), {layers: [T.sat], scrollWheelZoom: false});
    map.on('click', () => map.scrollWheelZoom.enable());   // page scrolls normally until you click the map
    const pts = D.lat.map((a, j) => [a, D.lon[j]]), fit = pts.slice();
    if (B && B.ll) fit.push(B.ll);
    if (showE && E.ll) fit.push(E.ll);
    map.fitBounds(L.latLngBounds(fit).pad(0.6), {maxZoom: 19});
    const colour = v => { const t = Math.min(1, Math.max(0, (v - cmin) / (cmax - cmin))); return 'hsl(' + Math.round(250 - 250 * t) + ',90%,50%)'; };
    L.polyline(pts, {color: '#fff', weight: 1.5, opacity: 0.7}).addTo(map);
    pts.forEach((q, j) => L.circleMarker(q, {radius: 4, weight: 0.5, color: '#000', fillColor: colour(D.med[j]), fillOpacity: 0.95})
      .bindTooltip(D.med[j] + ' dBFS (median), t=' + D.t[j] + ' s').addTo(map));
    L.marker(pts[0]).bindTooltip('start').addTo(map);
    L.circleMarker(pts[pts.length - 1], {radius: 8, color: '#f00', weight: 3, fill: false}).bindTooltip('end').addTo(map);
    if (showE && E.ll) {
      L.circle(E.ll, {radius: E.rad, color: estC, weight: 2, dashArray: '5 5', fill: false}).addTo(map);
      L.circleMarker(E.ll, {radius: 9, color: '#fff', weight: 2, fillColor: estC, fillOpacity: 1})
        .bindTooltip('Detected beacon (90% within ~' + E.rad + ' m)', {permanent: true, direction: 'right'}).addTo(map);
    }
    if (B && B.ll) {
      L.circleMarker(B.ll, {radius: 9, color: '#fff', weight: 2, fillColor: '#000', fillOpacity: 1})
        .bindTooltip('Real beacon', {permanent: true, direction: 'left'}).addTo(map);
      if (showE && E.ll) L.polyline([B.ll, E.ll], {color: '#fff', weight: 2, dashArray: '4 4'}).addTo(map);
    }
    L.control.layers({'Satellite': T.sat, 'Street map': T.osm}, {}, {collapsed: false}).addTo(map);
    maps[i] = map;
  } else document.getElementById(id('map')).innerHTML = '<p style="padding:20px">No .origin.json beside the CSV, so there is no map position.</p>';
}
show(keys[0]);
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("csv", nargs="+", help="flight CSV(s); several make one tab each")
    p.add_argument("--out", help="output HTML (default: dashboards/all_flights.html, or <flight>.html for one)")
    p.add_argument("--title", help="page title")
    p.add_argument("--beacon", nargs=2, type=float, metavar=("N", "E"),
                   help="real beacon position for a SINGLE flight (metres north/east in its log coordinates)")
    p.add_argument("--beacons", help="JSON {\"005\": [N, E], ...} of real beacon positions (default data/beacons.json)")
    p.add_argument("--media", help="folder of videos/photos to embed (default: MEDIA if it exists)")
    p.add_argument("--no-media", action="store_true", help="do not embed videos or photos")
    p.add_argument("--exclude-photo", nargs="+", default=[], metavar="TEXT",
                   help="leave out photos whose file name contains this text")
    p.add_argument("--story", help="story.html with the project text and photos (default: story.html if it exists)")
    p.add_argument("--no-story", action="store_true", help="plain dashboard without the project page and banner")
    p.add_argument("--no-detect", action="store_true", help="skip beacon detection")
    p.add_argument("--hide-live-signal", nargs="+", metavar="NNN", default=[],
                   help="flight numbers (e.g. 006) whose video panel has no live signal-strength chart")
    p.add_argument("--overview-graphs", nargs="+", metavar="NNN", default=[],
                   help="flight numbers (e.g. 005) whose signal and heat-map charts are also shown on the Overview")
    p.add_argument("--hide-tab", nargs="+", metavar="NNN", default=[],
                   help="flight numbers (e.g. 006) that get no tab of their own; they still appear in the Overview")
    p.add_argument("--hide-estimate-marker", nargs="+", metavar="NNN", default=[],
                   help="flight numbers (e.g. 006) whose orange detected marker is not drawn; the numbers stay in the cards")
    a = p.parse_args()

    known, hide, cfg_ = {}, {str(int(x)).zfill(3) for x in a.hide_estimate_marker}, {}
    hide_tabs = {str(int(x)).zfill(3) for x in a.hide_tab}
    ov_graphs = {str(int(x)).zfill(3) for x in a.overview_graphs}
    no_sig = {str(int(x)).zfill(3) for x in a.hide_live_signal}
    bf = a.beacons or ("data/beacons.json" if os.path.exists("data/beacons.json") else None)
    if bf:
        cfg_ = json.load(open(bf))
        known = {str(int(k)).zfill(3): tuple(v) for k, v in cfg_.items() if not k.startswith("_")}
        hide |= {str(int(x)).zfill(3) for x in cfg_.get("_hide_estimate_marker", [])}
        hide_tabs |= {str(int(x)).zfill(3) for x in cfg_.get("_hide_tab", [])}
        ov_graphs |= {str(int(x)).zfill(3) for x in cfg_.get("_overview_graphs", [])}
        no_sig |= {str(int(x)).zfill(3) for x in cfg_.get("_hide_live_signal", [])}

    flights, skipped = [], []
    for path in sorted(a.csv):
        try:
            d = load(path)
        except (ValueError, KeyError) as exc:
            skipped.append((path, str(exc)))
            continue
        num, when = flight_id(path)
        beacon = tuple(a.beacon) if (a.beacon and len(a.csv) == 1) else known.get(str(num).zfill(3))
        flights.append(build(d, num, when, os.path.basename(path), beacon, not a.no_detect, str(num).zfill(3) in hide,
                              str(num).zfill(3) in hide_tabs, str(num).zfill(3) in ov_graphs,
                              str(num).zfill(3) in no_sig))
    for path, why in skipped:
        print(f"SKIPPED {path}: {why}")
    if not flights:
        raise SystemExit("no usable flight logs")

    out0 = a.out or os.path.join("dashboards", "all_flights.html" if len(flights) > 1 else
                                 os.path.splitext(os.path.basename(a.csv[0]))[0] + ".html")
    media_dir = None if a.no_media else (a.media or ("MEDIA" if os.path.isdir("MEDIA") else None))
    mvideos, mphotos, munused = ({}, [], [])
    if media_dir:
        excl = list(a.exclude_photo) + list(cfg_.get("_exclude_photos", []))
        mvideos, mphotos, munused = scan_media(media_dir, os.path.dirname(out0) or ".", excl)
        offs = {str(int(k)).zfill(3): float(v) for k, v in cfg_.get("_video_offset", {}).items()}
        for D in flights:
            D["video"] = mvideos.get(str(D["name"]).zfill(3))
            D["videoOffset"] = offs.get(str(D["name"]).zfill(3), 0.0)
        print(f"media: {len([D for D in flights if D.get('video')])} flight video(s), {len(mphotos)} photo(s) from {media_dir}"
              + (f"; NOT used (no flight number in the name): {', '.join(munused)}" if munused else ""))
    story_path = None if a.no_story else (a.story or ("story.html" if os.path.exists("story.html") else None))
    story = load_story(story_path, mphotos) if story_path else {"hero": "", "about": ""}
    NAME = "Avalanche victim detection dashboard"
    title = a.title or (NAME + " \u00b7 Flight " + flights[0]["name"] if len(flights) == 1 else NAME)
    out = a.out or os.path.join("dashboards", "all_flights.html" if len(flights) > 1 else
                                os.path.splitext(os.path.basename(a.csv[0]))[0] + ".html")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(HTML.replace("__DATA__", json.dumps(flights)).replace("__MEDIA__", json.dumps({"photos": mphotos})).replace("__STORY__", json.dumps(story))
                .replace("__TITLE__", title))
    for D in flights:
        s, e = D["stats"], D["est"]
        line = f"  flight {D['name']}: {s['samples']} readings, {s['duration']} s, signal {s['pmin']}..{s['pmax']} dBFS"
        if e:
            line += f" | detected N {e['n']} E {e['e']}"
            if D["beacon"]:
                line += f" | real N {D['beacon']['n']} E {D['beacon']['e']}, error {e['err']} m"
        print(line + ("  [SIMULATED]" if D["simulated"] else ""))
    print(f"wrote {out}  ({len(flights)} flight tab{'s' if len(flights) != 1 else ''}"
          + (" + overview" if len(flights) > 1 else "") + ")")


if __name__ == "__main__":
    main()
