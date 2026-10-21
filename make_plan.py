"""
make_plan.py

After the sweep: estimate where the beacon is, and write a small
lawnmower mission around that spot for YOU to load into QGC and fly.
Nothing here talks to the drone.

    python3 make_plan.py flight1.csv

Reads the CSV (and flight1.origin.json beside it, written by logger.py),
prints the estimate with a real map position, and writes:

    flight1_refine.plan        open in QGC:  Plan -> File -> Open
    flight1_refine.waypoints   older QGC/Mission Planner format, same path
    flight1_refine_map.png     heat map of where the beacon probably is

The mission is waypoints only: lanes ACROSS the sweep lanes, over a box
sized from the estimate's uncertainty (2 x r90 each way, 16 to 60 m wide),
at --alt metres above the takeoff point, preceded by a speed command.
Take off and climb manually, then start the mission in AUTO.

Several CSVs (e.g. after a battery swap) are combined and aligned using
their origin files:  python3 make_plan.py sweep.csv more.csv

NOT TESTED IN QGC ITSELF: the .plan layout follows QGC's file format, but
I could not open it in QGC from here. Open it and LOOK at the map before
flying. If it will not load, try the .waypoints file, or type in the
latitude/longitude list printed at the end.
"""

import argparse
import json
import os

import estimator
from patterns import ned_to_latlon, refine_waypoints


def qgc_plan(wps_latlon, alt, speed, home):
    """Dict in QGC's .plan format: speed command then waypoints, relative alt."""
    items, jump = [], 1
    items.append({"autoContinue": True, "command": 178, "doJumpId": jump,
                  "frame": 2, "params": [1, speed, -1, 0, 0, 0, 0],
                  "type": "SimpleItem"})
    for lat, lon in wps_latlon:
        jump += 1
        items.append({"AMSLAltAboveTerrain": None, "Altitude": alt, "AltitudeMode": 1,
                      "autoContinue": True, "command": 16, "doJumpId": jump,
                      "frame": 3, "params": [0, 0, 0, None, lat, lon, alt],
                      "type": "SimpleItem"})
    return {
        "fileType": "Plan",
        "geoFence": {"circles": [], "polygons": [], "version": 2},
        "groundStation": "QGroundControl",
        "mission": {"cruiseSpeed": speed, "firmwareType": 3, "hoverSpeed": speed,
                    "items": items,
                    "plannedHomePosition": [home[0], home[1], 0],
                    "vehicleType": 2, "version": 2},
        "rallyPoints": {"points": [], "version": 2},
        "version": 1,
    }


def waypoints_file(wps_latlon, alt, speed, home):
    """Legacy 'QGC WPL 110' text: home, speed command, then waypoints."""
    rows = ["QGC WPL 110",
            f"0\t1\t0\t16\t0\t0\t0\t0\t{home[0]:.8f}\t{home[1]:.8f}\t0\t1",
            f"1\t0\t2\t178\t1\t{speed}\t-1\t0\t0\t0\t0\t1"]
    for i, (lat, lon) in enumerate(wps_latlon, start=2):
        rows.append(f"{i}\t0\t3\t16\t0\t0\t0\t0\t{lat:.8f}\t{lon:.8f}\t{alt}\t1")
    return "\n".join(rows) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("csv", nargs="+", help="logger CSV(s), first one defines the frame")
    p.add_argument("--alt", type=float, default=5.0, help="refine altitude above takeoff (m)")
    p.add_argument("--spacing", type=float, default=5.0, help="lane spacing (m)")
    p.add_argument("--speed", type=float, default=3.0, help="mission speed (m/s)")
    p.add_argument("--sigma", type=float, default=5.0, help="expected fading std (dB)")
    p.add_argument("--min-alt", type=float, default=3.0)
    p.add_argument("--min-power", type=float, default=-50.0)
    p.add_argument("--origin", nargs=2, type=float, metavar=("LAT", "LON"),
                   help="override: lat/lon of the local-NED origin (the takeoff pad)")
    p.add_argument("--out", help="output prefix (default: <first csv>_refine)")
    a = p.parse_args()

    prefix = a.out or (os.path.splitext(a.csv[0])[0] + "_refine")
    n, e, d, pw, origin = estimator.load_samples_multi(
        a.csv, min_alt_m=a.min_alt, min_power_dbfs=a.min_power)
    if a.origin:
        origin = tuple(a.origin)
    if origin is None:
        p.error("no origin: logger.py writes <csv>.origin.json once it has a GPS fix; "
                "pass --origin LAT LON (the takeoff pad) to override")
    if len(pw) < 10:
        p.error(f"only {len(pw)} usable samples (armed, >= {a.min_alt:g} m up, above "
                f"{a.min_power:g} dBFS) -- check the flight and the beacon")

    print(f"{len(pw)} usable samples, power {pw.min():.1f} .. {pw.max():.1f} dBFS")
    est = estimator.estimate(n, e, d, pw, sigma_db=a.sigma)
    lat, lon = ned_to_latlon(origin[0], origin[1], est.n_m, est.e_m)
    print(f"\nBeacon estimate : {est.n_m:.1f} m north, {est.e_m:.1f} m east of the takeoff pad")
    print(f"Map position    : {lat:.6f}, {lon:.6f}")
    print(f"Uncertainty     : 90% inside {est.r90_m:.1f} m of that point")

    wps, desc = refine_waypoints(n, e, est.n_m, est.e_m, est.r90_m, a.spacing, a.alt)
    ll = [ned_to_latlon(origin[0], origin[1], wn, we) for wn, we, _ in wps]
    print(f"\nRefine mission  : {desc}")

    with open(prefix + ".plan", "w") as f:
        json.dump(qgc_plan(ll, a.alt, a.speed, origin), f, indent=2)
    with open(prefix + ".waypoints", "w") as f:
        f.write(waypoints_file(ll, a.alt, a.speed, origin))
    estimator.plot_map(est, n, e, pw, prefix + "_map.png")

    print(f"\nWrote {prefix}.plan, {prefix}.waypoints, {prefix}_map.png")
    print("In QGC: Plan -> File -> Open, check the map, upload, take off manually, start in AUTO.")
    print("\nWaypoints (lat, lon, alt above takeoff):")
    for i, (la, lo) in enumerate(ll, 1):
        print(f"  {i:2d}  {la:.7f}  {lo:.7f}  {a.alt:g} m")


if __name__ == "__main__":
    main()
