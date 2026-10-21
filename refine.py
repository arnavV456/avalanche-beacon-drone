"""
refine.py

Second stage of the search: take the coarse sweep, work out roughly where
the beacon is, then fly a tight pattern around that spot to pin it down.

HOW IT RUNS (start it BEFORE the sweep):
  1. COLLECT  Logs (position, power) every tick while the sweep is flown
              by the autopilot (a QGC survey in AUTO). Sends NO commands.
  2. PLAN     When the pilot switches to GUIDED (and enough samples are in),
              estimates the beacon from the sweep, then builds a small
              lawnmower centred on the estimate. Lanes run PERPENDICULAR to
              the sweep lanes -- that crossing geometry is what shrinks the
              uncertainty blob. Box size follows the uncertainty (r90).
  3. REFINE   Flies it in GUIDED with position setpoints, still logging.
  4. CONFIRM  Re-estimates from ALL samples (sweep + refine), flies to the
              final estimate and holds there. Camera confirmation hooks in
              here later.

  Handoff to GUIDED is a deliberate pilot action, as in search.py. If the
  pilot leaves GUIDED at any time the script stops sending setpoints and
  exits -- the pilot always wins.

  This one loop owns `master` and `sdr`; nothing else may send setpoints
  (see INTEGRATING WITH OTHER SUBROUTINES in search.py).

KNOWN GAPS -- add before real flight:
  - No geofence or beacon-never-found failsafe. The refine box is clamped
    to within 20 m of the swept area, but that is a convenience, not a
    fence. Set a real one on the autopilot (FENCE_*).
  - Setpoints pause for ~1-2 s while the estimator runs; the drone just
    holds its last target, but don't run this on a Pi Zero without timing it.

SITL:
    python3 refine.py --beacon 65 35 0 --fading 5
  Real drone: omit --beacon (uses the Pluto) and pass --conn /dev/ttyAMA0.

Outputs (prefix = --out, default refine_<time>):
  <prefix>.csv         every sample, same columns as logger.py
  <prefix>_sweep.png   heat map after the sweep
  <prefix>_final.png   heat map after the refine pass
"""

import argparse
import csv
import math
import time

import numpy as np
from pymavlink import mavutil

import estimator
from fake_sdr import FakeSdr, local_position_from_master
from logger import COLUMNS
from patterns import refine_waypoints
from search import (BEACON_TONE_OFFSET_HZ, connect_pluto, read_tone_power_dbfs,
                    request_message_rate)

# Position only; ignore velocity, accel, yaw, yaw rate.
POSITION_ONLY_MASK = 0b0000111111111000


# ---------------- plumbing ----------------

def send_position_ned(master, n, e, d):
    master.mav.set_position_target_local_ned_send(
        0, master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, POSITION_ONLY_MASK,
        n, e, d, 0, 0, 0, 0, 0, 0, 0, 0)


def set_speed(master, speed_ms):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, 0,
        1, speed_ms, -1, 0, 0, 0, 0)   # 1 = ground speed


class Sampler:
    """One place that reads the SDR + position and appends a CSV row."""

    def __init__(self, master, sdr, csv_path, sim):
        self.master, self.sdr, self.sim = master, sdr, sim
        self.t0 = time.time()
        self.rows = 0
        self.usable = 0   # armed, >=3 m up, above the noise floor
        self._f = open(csv_path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(COLUMNS)

    def tick(self):
        """Drain MAVLink, take one sample, return (n, e, d) or None."""
        while self.master.recv_match(blocking=False) is not None:
            pass
        msg = self.master.messages.get("LOCAL_POSITION_NED")
        if msg is None:
            return None
        power = read_tone_power_dbfs(self.sdr)
        dist = self.sdr.last_distance_m if self.sim else None
        true = self.sdr.last_true_dbfs if self.sim else None
        self._w.writerow([
            f"{time.time() - self.t0:.2f}",
            f"{msg.x:.2f}", f"{msg.y:.2f}", f"{msg.z:.2f}", f"{power:.2f}",
            self.master.flightmode, int(bool(self.master.motors_armed())),
            f"{dist:.2f}" if dist is not None else "",
            f"{true:.2f}" if true is not None else "",
        ])
        self.rows += 1
        if self.master.motors_armed() and -msg.z >= 3.0 and power >= -50.0:
            self.usable += 1
        if self.rows % 25 == 0:
            self._f.flush()
        return (msg.x, msg.y, msg.z)

    def flush(self):
        self._f.flush()

    def close(self):
        self._f.close()


def pace(next_tick, period):
    next_tick += period
    time.sleep(max(0.0, next_tick - time.time()))
    return next_tick


# ---------------- planning ----------------

def plan_refine(est, n, e, args):
    """Waypoints for the refine pass; returns (waypoints, description)."""
    return refine_waypoints(n, e, est.n_m, est.e_m, est.r90_m, args.refine_spacing,
                            args.refine_alt, args.min_half, args.max_half)


# ---------------- flying ----------------

def fly_waypoints(master, sampler, wps, args, period):
    """Fly waypoints in GUIDED while sampling. False if the pilot took over."""
    set_speed(master, args.refine_speed)
    next_tick = time.time()
    for k, w in enumerate(wps, 1):
        t_wp = time.time()
        while True:
            pos = sampler.tick()
            if master.flightmode != "GUIDED":
                return False
            send_position_ned(master, *w)
            if pos is not None and math.dist(pos, w) < args.reach_tol:
                break
            if time.time() - t_wp > args.wp_timeout:
                print(f"  waypoint {k}/{len(wps)} not reached in {args.wp_timeout:.0f} s -- skipping")
                break
            next_tick = pace(next_tick, period)
        print(f"  waypoint {k}/{len(wps)} done  N={w[0]:.0f} E={w[1]:.0f}")
    return True


def hold_at(master, sampler, target, hold_s, period):
    """Hover at target, still sampling. Returns on timeout / pilot takeover / Ctrl+C."""
    t_end = time.time() + hold_s if hold_s > 0 else float("inf")
    next_tick = time.time()
    while time.time() < t_end:
        sampler.tick()
        if master.flightmode != "GUIDED":
            return False
        send_position_ned(master, *target)
        next_tick = pace(next_tick, period)
    return True


# ---------------- main ----------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--conn", default="udp:127.0.0.1:14551")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--out", default=time.strftime("refine_%Y%m%d_%H%M%S"), help="output prefix")
    p.add_argument("--hz", type=float, default=5.0)
    p.add_argument("--beacon", nargs=3, type=float, metavar=("N", "E", "D"),
                   help="SIM: hidden beacon, local NED m (omit to use the real Pluto)")
    p.add_argument("--fading", type=float, default=0.0, help="SIM: fading std (dB)")
    p.add_argument("--noise-floor", type=float, default=-60.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--sigma", type=float, default=5.0, help="estimator: expected fading std (dB)")
    p.add_argument("--min-samples", type=int, default=60, help="usable airborne samples needed before GUIDED triggers PLAN")
    p.add_argument("--refine-alt", type=float, default=8.0)
    p.add_argument("--refine-spacing", type=float, default=5.0)
    p.add_argument("--refine-speed", type=float, default=2.0)
    p.add_argument("--min-half", type=float, default=8.0, help="smallest refine half-width (m)")
    p.add_argument("--max-half", type=float, default=30.0, help="largest refine half-width (m)")
    p.add_argument("--confirm-alt", type=float, default=6.0)
    p.add_argument("--hold-s", type=float, default=20.0, help="hover time at the end (0 = until stopped)")
    p.add_argument("--reach-tol", type=float, default=2.0)
    p.add_argument("--wp-timeout", type=float, default=90.0)
    return p.parse_args(argv)


def run_estimate(sampler, args, label, png, true_ne):
    sampler.flush()
    n, e, d, pw = estimator.load_samples(sampler._f.name)
    if len(pw) < 10:
        print(f"{label}: only {len(pw)} usable samples -- cannot estimate")
        return None, None
    est = estimator.estimate(n, e, d, pw, sigma_db=args.sigma)
    line = (f"{label}: N={est.n_m:.1f} E={est.e_m:.1f}  r90={est.r90_m:.1f} m  "
            f"({len(pw)} samples, offset {est.offset_db:+.1f} dB)")
    if true_ne is not None:
        line += f"  | true error {math.dist((est.n_m, est.e_m), true_ne):.1f} m"
    print(line)
    estimator.plot_map(est, n, e, pw, png, true_ne=true_ne)
    return est, (n, e)


def main():
    args = parse_args()
    period = 1.0 / args.hz
    sim = args.beacon is not None
    true_ne = tuple(args.beacon[:2]) if sim else None

    print(f"Connecting to autopilot on {args.conn} ...")
    master = mavutil.mavlink_connection(args.conn, baud=args.baud)
    master.wait_heartbeat()
    print(f"Heartbeat OK (system {master.target_system})")
    request_message_rate(master, mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 10)

    if sim:
        sdr = FakeSdr(beacon_ned=args.beacon, get_position=local_position_from_master(master),
                      tone_hz=BEACON_TONE_OFFSET_HZ, noise_floor_dbfs=args.noise_floor,
                      fading_std_db=args.fading, seed=args.seed)
        print(f"SIM: hidden beacon at N={args.beacon[0]} E={args.beacon[1]} D={args.beacon[2]}")
    else:
        sdr = connect_pluto()

    sampler = Sampler(master, sdr, args.out + ".csv", sim)
    try:
        # ---- 1. COLLECT ----
        print("COLLECT: logging the sweep. Switch to GUIDED when it is finished.")
        next_tick = time.time()
        while True:
            sampler.tick()
            if master.flightmode == "GUIDED" and sampler.usable >= args.min_samples:
                break
            next_tick = pace(next_tick, period)
        print(f"GUIDED confirmed with {sampler.usable} usable samples -- taking over.")

        # ---- 2. PLAN ----
        est1, (n, e) = run_estimate(sampler, args, "SWEEP estimate", args.out + "_sweep.png", true_ne)
        if est1 is None:
            print("Not enough data to plan a refine pass; holding.")
            return
        wps, desc = plan_refine(est1, n, e, args)
        print(f"REFINE plan: {desc}")

        # ---- 3. REFINE ----
        if not fly_waypoints(master, sampler, wps, args, period):
            print("Pilot left GUIDED -- stopping.")
            return

        # ---- 4. CONFIRM ----
        est2, _ = run_estimate(sampler, args, "FINAL estimate", args.out + "_final.png", true_ne)
        if est2 is None:
            return
        target = (est2.n_m, est2.e_m, -abs(args.confirm_alt))
        print(f"CONFIRM: moving to N={target[0]:.1f} E={target[1]:.1f} and holding at "
              f"{args.confirm_alt:g} m ...")
        # fly there (single waypoint), then hover
        if fly_waypoints(master, sampler, [target], args, period):
            hold_at(master, sampler, target, args.hold_s, period)
        print("Done." if master.flightmode == "GUIDED" else "Pilot left GUIDED -- stopping.")
    except KeyboardInterrupt:
        print("\nInterrupted -- no further setpoints will be sent.")
    finally:
        sampler.close()
        print(f"Samples saved to {args.out}.csv")


if __name__ == "__main__":
    main()
