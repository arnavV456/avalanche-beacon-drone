"""
logger.py

Passive sample logger for the search sweep. Connects to the autopilot,
reads the tone power from the SDR at LOG_HZ and writes one CSV row per
sample with the drone's position beside it. It sends NO flight commands
(the only thing it transmits is a request for position messages), so it is
safe to run alongside a QGC/AUTO mission or any other controller.

The CSV is the input to the estimator: (n, e, d, power_dbfs) per row.

SITL (default):
    sim_vehicle.py -v ArduCopter --console --map     # outputs to 14550 & 14551
    python3 logger.py --beacon 60 40 0 --fading 4

  QGC listens on 14550, so this defaults to 14551 to avoid a port clash.

Real drone: leave out --beacon and it uses the real Pluto (adi.Pluto)
instead of FakeSdr. The true_* columns are then blank.

Also writes <out>.origin.json: the lat/lon of the autopilot's local-NED
origin (where its navigation started, near the takeoff pad). make_plan.py
and estimator.py use it to turn north/east metres into a real map position.
To get it the logger asks the autopilot for GLOBAL_POSITION_INT as well as
LOCAL_POSITION_NED (two telemetry-rate requests; no flight commands).

Columns:
  t_s            seconds since logging started
  n_m,e_m,d_m    drone position, local NED metres (d negative = up)
  power_dbfs     gain-compensated tone power (raw read, unsmoothed)
  mode, armed    autopilot flight mode / armed flag at that moment
  true_dist_m    (sim only) real distance to the hidden beacon
  true_dbfs      (sim only) noise-free power at that distance, incl. fading
"""

import argparse
import csv
import json
import math
import os
import time

from pymavlink import mavutil

from fake_sdr import FakeSdr, local_position_from_master
from search import (BEACON_TONE_OFFSET_HZ, connect_pluto, read_tone_power_dbfs,
                    request_message_rate)

COLUMNS = ["t_s", "n_m", "e_m", "d_m", "power_dbfs", "mode", "armed",
           "true_dist_m", "true_dbfs"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--conn", default="udp:127.0.0.1:14551",
                   help="MAVLink connection string (default: %(default)s)")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--out", default=None, help="CSV path (default: samples_<time>.csv)")
    p.add_argument("--hz", type=float, default=5.0, help="samples per second")
    p.add_argument("--beacon", nargs=3, type=float, metavar=("N", "E", "D"),
                   help="SIM: hidden beacon in local NED metres; omit to use the real Pluto")
    p.add_argument("--fading", type=float, default=0.0, help="SIM: fading std (dB)")
    p.add_argument("--noise-floor", type=float, default=-60.0, help="SIM: noise floor (dBFS)")
    p.add_argument("--seed", type=int, default=None, help="SIM: RNG seed")
    return p.parse_args()


def try_write_origin(master, pos_msg, path):
    """
    Local NED origin as lat/lon = (global position) - (local offset). Needs
    a GPS fix; returns True once written so we only do it once per run.
    """
    g = master.messages.get("GLOBAL_POSITION_INT")
    if g is None or (g.lat == 0 and g.lon == 0):
        return False
    lat = g.lat / 1e7 - pos_msg.x / 111_320.0
    lon = g.lon / 1e7 - pos_msg.y / (111_320.0 * math.cos(math.radians(g.lat / 1e7)))
    with open(path, "w") as f:
        json.dump({"lat0": lat, "lon0": lon}, f)
    print(f"Local-NED origin: {lat:.7f}, {lon:.7f} -> {path}")
    return True


def main():
    args = parse_args()
    out_path = args.out or time.strftime("samples_%Y%m%d_%H%M%S.csv")

    print(f"Connecting to autopilot on {args.conn} ...")
    master = mavutil.mavlink_connection(args.conn, baud=args.baud)
    master.wait_heartbeat()
    print(f"Heartbeat OK (system {master.target_system}, component {master.target_component})")
    request_message_rate(master, mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 10)
    request_message_rate(master, mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 5)

    sim = args.beacon is not None
    if sim:
        sdr = FakeSdr(beacon_ned=args.beacon,
                      get_position=local_position_from_master(master),
                      tone_hz=BEACON_TONE_OFFSET_HZ,
                      noise_floor_dbfs=args.noise_floor,
                      fading_std_db=args.fading,
                      seed=args.seed)
        print(f"SIM: hidden beacon at N={args.beacon[0]} E={args.beacon[1]} D={args.beacon[2]} m")
    else:
        sdr = connect_pluto()

    period = 1.0 / args.hz
    origin_path = os.path.splitext(out_path)[0] + ".origin.json"
    origin_done = False
    t0 = time.time()
    next_tick = t0
    rows = 0
    print(f"Logging to {out_path} at {args.hz:g} Hz (Ctrl+C to stop) ...")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        try:
            while True:
                # Drain pending messages so master.messages (position cache)
                # and master.flightmode stay current.
                while master.recv_match(blocking=False) is not None:
                    pass

                pos_msg = master.messages.get("LOCAL_POSITION_NED")
                if pos_msg is None:
                    # No position yet -- nothing useful to log against.
                    time.sleep(0.05)
                    continue

                if not origin_done:
                    origin_done = try_write_origin(master, pos_msg, origin_path)

                power = read_tone_power_dbfs(sdr)
                armed = bool(master.motors_armed())
                w.writerow([
                    f"{time.time() - t0:.2f}",
                    f"{pos_msg.x:.2f}", f"{pos_msg.y:.2f}", f"{pos_msg.z:.2f}",
                    f"{power:.2f}", master.flightmode, int(armed),
                    f"{sdr.last_distance_m:.2f}" if sim and sdr.last_distance_m is not None else "",
                    f"{sdr.last_true_dbfs:.2f}" if sim and sdr.last_true_dbfs is not None else "",
                ])
                rows += 1
                if rows % int(args.hz * 5) == 0:
                    f.flush()
                    print(f"  {rows} rows | N={pos_msg.x:6.1f} E={pos_msg.y:6.1f} "
                          f"D={pos_msg.z:6.1f} | {power:6.1f} dBFS | {master.flightmode}")

                next_tick += period
                time.sleep(max(0.0, next_tick - time.time()))
        except KeyboardInterrupt:
            print(f"\nStopped. {rows} rows written to {out_path}")


if __name__ == "__main__":
    main()
