"""
autolog.py

Always-on, hands-off logger for the Pi. Start it at boot (see
install_service.sh) and forget about it: no SSH, no tmux, no file names.

  * Finds the Pixhawk by probing serial ports for a MAVLink heartbeat. (The
    Pluto can also show up as a /dev/ttyACM* console, so a fixed name is not
    safe.)
  * Idle while the drone is disarmed. When it ARMS, opens a new numbered file
    data/flight_001_<time>.csv and logs (position, power) at 5 Hz; when it
    DISARMS, closes the file. One arm-to-disarm = one file, so the sweep and
    the second flight are simply flight_001 and flight_002.
  * Writes flight_NNN_<time>.origin.json beside each file (lat/lon of the
    autopilot's local-NED origin) so analyze.py can put results on a map.
  * Sends NO flight commands; the only thing it transmits is a request for
    position telemetry.
  * Writes data/STATUS.txt every 2 s: what it is doing right now.
  * Every row is flushed, so pulling the battery loses at most the last row.
  * If the Pixhawk goes silent or the Pluto errors it exits non-zero and the
    service manager restarts it, so a loose cable is recovered, not fatal.

Same CSV columns as logger.py, so estimator.py / analyze.py read it directly.

    python3 autolog.py                      # real Pixhawk + Pluto
    python3 autolog.py --conn udp:127.0.0.1:14551 --beacon 22 14 0 --fading 5   # SITL
"""

import argparse
import csv
import glob
import os
import signal
import sys
import time

from pymavlink import mavutil

from fake_sdr import FakeSdr, local_position_from_master
from logger import COLUMNS, try_write_origin
from search import (BEACON_TONE_OFFSET_HZ, connect_pluto, read_tone_power_dbfs,
                    request_message_rate)

HEARTBEAT_LOST_S = 15.0


def port_candidates():
    """Serial devices that might be the autopilot, most specific first."""
    found = []
    for pattern in ("/dev/serial/by-id/*ArduPilot*", "/dev/serial/by-id/*Pixhawk*",
                    "/dev/serial/by-id/*CubePilot*", "/dev/serial/by-id/*",
                    "/dev/ttyACM*"):
        for p in sorted(glob.glob(pattern)):
            if os.path.realpath(p) not in [os.path.realpath(x) for x in found]:
                found.append(p)
    return found


def open_autopilot(conn, baud, stop):
    """Block until some connection yields a MAVLink heartbeat; return it."""
    while not stop["flag"]:
        for c in ([conn] if conn != "auto" else port_candidates()):
            try:
                m = mavutil.mavlink_connection(c, baud=baud)
                if m.wait_heartbeat(timeout=4) is not None:
                    print(f"Autopilot found on {c} (system {m.target_system})", flush=True)
                    return m
                m.close()
            except Exception as exc:     # busy port, permissions, etc.
                print(f"  {c}: {exc}", flush=True)
        time.sleep(2)
    return None


class Status:
    def __init__(self, path):
        self.path, self._last = path, 0.0

    def update(self, state, **info):
        if time.time() - self._last < 2.0:
            return
        self._last = time.time()
        lines = [f"updated  {time.strftime('%Y-%m-%d %H:%M:%S')}", f"state    {state}"]
        lines += [f"{k:<8} {v}" for k, v in info.items()]
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, self.path)


def next_flight_path(data_dir):
    n = 1 + max([int(os.path.basename(p).split("_")[1])
                 for p in glob.glob(os.path.join(data_dir, "flight_*.csv"))
                 if os.path.basename(p).split("_")[1].isdigit()] or [0])
    return os.path.join(data_dir, f"flight_{n:03d}_{time.strftime('%Y%m%d-%H%M%S')}.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--conn", default="auto", help="'auto' (probe serial ports) or a connection string")
    p.add_argument("--baud", type=int, default=115200, help="ignored over USB")
    p.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    p.add_argument("--hz", type=float, default=5.0)
    p.add_argument("--beacon", nargs=3, type=float, metavar=("N", "E", "D"),
                   help="SIM only: hidden beacon, local NED m (omit to use the real Pluto)")
    p.add_argument("--fading", type=float, default=0.0, help="SIM only: fading std (dB)")
    p.add_argument("--noise-floor", type=float, default=-60.0, help="SIM only")
    p.add_argument("--seed", type=int, default=None, help="SIM only")
    args = p.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    status = Status(os.path.join(args.data_dir, "STATUS.txt"))
    stop = {"flag": False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.update(flag=True))

    sim = args.beacon is not None
    print("autolog starting; data dir:", args.data_dir, flush=True)
    status.update("starting")

    # SDR first: if it is missing we want to fail and be restarted, loudly.
    sdr = None if sim else connect_pluto()

    master = open_autopilot(args.conn, args.baud, stop)
    if master is None:
        return 0
    request_message_rate(master, mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 10)
    request_message_rate(master, mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 5)
    if sim:
        sdr = FakeSdr(beacon_ned=args.beacon, get_position=local_position_from_master(master),
                      tone_hz=BEACON_TONE_OFFSET_HZ, noise_floor_dbfs=args.noise_floor,
                      fading_std_db=args.fading, seed=args.seed)
        print(f"SIM: hidden beacon at N={args.beacon[0]} E={args.beacon[1]} D={args.beacon[2]}", flush=True)

    f = w = None
    rows, t0, path, origin_done, last_power = 0, 0.0, "", False, float("nan")
    period = 1.0 / args.hz
    next_tick = time.time()

    def close_file(reason):
        nonlocal f, w
        if f is not None:
            f.flush(); os.fsync(f.fileno()); f.close()
            print(f"Closed {os.path.basename(path)}: {rows} rows ({reason})", flush=True)
            f = w = None

    try:
        while not stop["flag"]:
            while master.recv_match(blocking=False) is not None:
                pass
            if master.time_since("HEARTBEAT") > HEARTBEAT_LOST_S:
                close_file("autopilot silent")
                print("Autopilot heartbeat lost; exiting so the service restarts.", flush=True)
                return 2

            pos = master.messages.get("LOCAL_POSITION_NED")
            armed = bool(master.motors_armed())

            if armed and pos is not None:
                if f is None:
                    path = next_flight_path(args.data_dir)
                    f = open(path, "w", newline="")
                    w = csv.writer(f)
                    w.writerow(COLUMNS)
                    rows, t0, origin_done = 0, time.time(), False
                    print(f"ARMED -> logging to {os.path.basename(path)}", flush=True)
                if not origin_done:
                    origin_done = try_write_origin(master, pos, os.path.splitext(path)[0] + ".origin.json")
                last_power = read_tone_power_dbfs(sdr)
                dist = sdr.last_distance_m if sim else None
                true = sdr.last_true_dbfs if sim else None
                w.writerow([f"{time.time() - t0:.2f}", f"{pos.x:.2f}", f"{pos.y:.2f}", f"{pos.z:.2f}",
                            f"{last_power:.2f}", master.flightmode, 1,
                            f"{dist:.2f}" if dist is not None else "",
                            f"{true:.2f}" if true is not None else ""])
                f.flush()
                rows += 1
                if rows % 50 == 0:
                    os.fsync(f.fileno())
                status.update("LOGGING", file=os.path.basename(path), rows=rows,
                              power=f"{last_power:.1f} dBFS", mode=master.flightmode,
                              alt=f"{-pos.z:.1f} m")
            else:
                if f is not None:
                    close_file("disarmed")
                status.update("READY (waiting for arm)", mode=master.flightmode,
                              gps_origin="yes" if pos is not None else "no position yet")

            next_tick += period
            time.sleep(max(0.0, next_tick - time.time()))
    finally:
        close_file("stopped")
        status._last = 0.0          # force the final status past the throttle
        status.update("STOPPED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
