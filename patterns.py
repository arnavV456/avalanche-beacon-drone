"""
patterns.py

Pure waypoint generators for the search phases. No MAVLink, no SDR --
just geometry, so it can be tested without SITL.

All coordinates are local NED metres relative to the point the drone
takes off from (north, east, down). Down is negative for altitude.
"""

import math


def lawnmower_waypoints(length_n, width_e, spacing, alt, origin_n=0.0, origin_e=0.0):
    """
    Back-and-forth sweep of a rectangle, lanes running north-south.

    The rectangle spans length_n metres north and width_e metres east of
    (origin_n, origin_e). `spacing` is the MAXIMUM lane spacing (set it to
    the camera ground footprint); the actual spacing is nudged down so an
    integer number of equally spaced lanes fits exactly, and lanes are
    centred in their strips so the outer lanes sit half a spacing in from
    the edge -- every point in the box is then within spacing/2 of a lane.

    Returns a list of (n, e, d) tuples, two per lane (lane start and end).
    """
    if min(length_n, width_e, spacing) <= 0:
        raise ValueError("length_n, width_e and spacing must be positive")

    lanes = math.ceil(width_e / spacing)
    step = width_e / lanes
    d = -abs(alt)

    waypoints = []
    for i in range(lanes):
        e = origin_e + (i + 0.5) * step
        south, north = origin_n, origin_n + length_n
        ends = (south, north) if i % 2 == 0 else (north, south)
        waypoints.append((ends[0], e, d))
        waypoints.append((ends[1], e, d))
    return waypoints


def path_length(waypoints, start=None):
    """Total straight-line horizontal+vertical distance through the waypoints."""
    pts = ([start] if start is not None else []) + list(waypoints)
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))


def refine_waypoints(n, e, est_n, est_e, r90, spacing, alt, min_half=8.0, max_half=30.0):
    """
    Small lawnmower centred on a beacon estimate, lanes ACROSS the sweep.

    n, e      positions the sweep covered (sequences of metres, local NED);
              used only to tell which way the sweep lanes ran and to keep
              the box centre within 20 m of the swept area.
    est_n/e   the estimate to centre on; r90 its 90% radius (m).
    Box half-width = 2 * r90, limited to [min_half, max_half]. Returns
    (waypoints, description) with waypoints as (n, e, d) tuples.
    """
    def clip(x, lo, hi):
        return max(lo, min(hi, x))

    # Whichever axis the drone moved along more is the sweep lane direction.
    dn = sum(abs(b - a) for a, b in zip(n, n[1:]))
    de = sum(abs(b - a) for a, b in zip(e, e[1:]))
    sweep_lanes_north = dn >= de

    half = clip(2.0 * r90, min_half, max_half)
    cn = clip(est_n, min(n) - 20, max(n) + 20)
    ce = clip(est_e, min(e) - 20, max(e) + 20)
    side = 2 * half

    if sweep_lanes_north:
        # lawnmower_waypoints makes north-south lanes; build with the axes
        # swapped, then swap back to get east-west lanes.
        raw = lawnmower_waypoints(side, side, spacing, alt,
                                  origin_n=ce - half, origin_e=cn - half)
        wps = [(b, a, d) for a, b, d in raw]
        lanes = "east-west"
    else:
        wps = lawnmower_waypoints(side, side, spacing, alt,
                                  origin_n=cn - half, origin_e=ce - half)
        lanes = "north-south"
    desc = (f"{side:.0f} x {side:.0f} m box at N={cn:.1f} E={ce:.1f}, "
            f"{lanes} lanes @ {alt:g} m alt, {len(wps) // 2} lanes, "
            f"{path_length(wps):.0f} m of path")
    return wps, desc


# Flat-earth conversion; fine for the few hundred metres we work over.
M_PER_DEG = 111_320.0


def ned_to_latlon(lat0, lon0, n, e):
    """Local north/east metres from an origin -> (lat, lon) degrees."""
    return (lat0 + n / M_PER_DEG,
            lon0 + e / (M_PER_DEG * math.cos(math.radians(lat0))))


def latlon_to_ned(lat0, lon0, lat, lon):
    """(lat, lon) degrees -> north/east metres from an origin."""
    return ((lat - lat0) * M_PER_DEG,
            (lon - lon0) * M_PER_DEG * math.cos(math.radians(lat0)))


if __name__ == "__main__":
    # Quick self-check: 100 x 100 m box, 15 m max spacing, 15 m altitude.
    wps = lawnmower_waypoints(100, 100, 15, 15)
    lanes = len(wps) // 2
    es = sorted({round(w[1], 3) for w in wps})
    print(f"{lanes} lanes, actual spacing {es[1] - es[0]:.2f} m, "
          f"outer lanes at {es[0]:.2f} and {es[-1]:.2f} m")
    print(f"path length {path_length(wps, start=(0, 0, -15)):.0f} m "
          f"-> {path_length(wps, start=(0, 0, -15)) / 3:.0f} s at 3 m/s")
    for w in wps:
        print(f"  N={w[0]:6.1f}  E={w[1]:6.2f}  D={w[2]:6.1f}")

    # Assertions: coverage, serpentine order, altitude.
    assert lanes == 7
    assert all(0 <= w[0] <= 100 and 0 <= w[1] <= 100 for w in wps)
    assert all(w[2] == -15 for w in wps)
    assert wps[1][0] == 100 and wps[2][0] == 100 and wps[3][0] == 0  # serpentine
    assert max(b[1] - a[1] for a, b in zip(wps[::2], wps[2::2])) <= 15 + 1e-9
    print("OK")
