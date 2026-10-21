"""
rssi_homing_controller.py

Companion-computer controller for RSSI-based beacon homing. Reads signal
power from an ADALM-PLUTO SDR and commands an ArduPilot flight controller
(over MAVLink/pymavlink) to fly toward increasing signal strength, then
hold above the beacon once locked on.

ASSUMPTIONS (change the constants below if any of these don't match):
  - Flight controller runs ArduCopter and accepts GUIDED-mode velocity
    setpoints (SET_POSITION_TARGET_LOCAL_NED).
  - Pi <-> FC link is a serial UART. Swap MAVLINK_CONN for a UDP string
    if you're going through mavlink-router / a GCS relay instead.
  - Vehicle is already armed and airborne before this script starts --
    it only takes over guidance, not arming or takeoff. It also waits
    for the pilot to switch to GUIDED before sending any setpoints,
    rather than forcing the mode change itself.
  - There's no directional/AoA antenna hardware, so bearing to the beacon
    has to be inferred from how power changes as the drone moves, not
    read directly from a single sample.

SEARCH STRATEGY -- run-and-tumble:
  A single RSSI reading is a magnitude with no bearing information, so
  instead of computing "which way is the beacon" the controller commits
  to a heading, flies it for RUN_DURATION_S, and checks whether power
  improved. Better -> keep the heading. Worse/flat -> pick a new one.
  This is the same principle bacteria use for chemotaxis with nothing
  but a scalar concentration sensor, and it needs no yaw control --
  every velocity command below is in the world (NED) frame.

INTEGRATING WITH OTHER SUBROUTINES:
  HomingController.compute_velocity(power, relative_alt_m) is a pure
  function -- it never touches MAVLink. If this needs to run alongside
  other movement subroutines (e.g. a lawnmower search pattern used when
  no signal is present), have ONE loop own `master` and `sdr`
  exclusively: call compute_velocity() only when signal_present(power)
  is True, call the other subroutine's equivalent otherwise, and send
  exactly one velocity command per tick from that one place. Never run
  two things that both call send_velocity_ned independently -- that's
  how you get the FC fighting itself. Keep anything heavy or slow (YOLO
  inference, etc.) in its own thread writing to a shared flag instead
  of inline in this tick loop, since it doesn't need MAVLink or the SDR
  at all.

TERMINAL STANDOFF -- READ BEFORE RELYING ON IT:
  The controller parks at STANDOFF_SLANT_M / STANDOFF_ELEVATION_DEG, but
  only the ALTITUDE half of that is trustworthy: it comes from the
  baro/EKF. The horizontal half is inferred from signal strength, and
  RSSI is a poor ruler:

    - It gives range only, never bearing. "Back off" can only mean
      reversing the heading that got us here, so the parked position is
      somewhere on a ring around the beacon, not a chosen point on it.
    - Ground-reflection multipath at 866 MHz (wavelength ~0.35 m) swings
      received power by 10 dB or more over small position changes. With
      a path-loss exponent near 2, a 10 dB error is a ~3x distance
      error -- so a nominal 4 m hold can sit anywhere from ~1.3 m to
      ~12 m.

  Treat the RSSI standoff as "stops somewhere close and stays there",
  not as a metre-accurate position hold. For a genuine 4 m standoff,
  close the last few metres with a sensor that measures geometry rather
  than power -- the onboard camera is the obvious candidate, since a
  known-size visual marker on the beacon gives both bearing (pixel
  offset) and range (apparent size) directly. Use RSSI for acquisition
  beyond visual range, vision for the terminal hold.

CALIBRATING STANDOFF_POWER_DBFS:
  Park the drone at exactly STANDOFF_SLANT_M from the beacon, at
  STANDOFF_ALT_M altitude, over ground similar to the real site. Log
  read_tone_power_dbfs for 30+ seconds. Use the MEDIAN, not a single
  reading, and note the spread -- that spread is your real position
  uncertainty, and if it exceeds a few dB, the RSSI-only standoff will
  not hold to 4 m no matter how the constants are tuned.

TEST IN SITL BEFORE ANY REAL FLIGHT. Tune RUN_DURATION_S,
STRONG_SIGNAL_DBFS and IMPROVEMENT_DB against real field data -- the
values below are starting guesses, not calibrated numbers. This script
also has no "beacon never found" failsafe (e.g. RTL after N minutes) --
add one before flying it for real.
"""

import math
import random
import time

import adi
import numpy as np
from pymavlink import mavutil

# ---------------- SDR CONFIG ----------------
PLUTO_URI = "ip:192.168.2.1"
RX_LO_HZ = 866_000_000
SAMPLE_RATE = 2_500_000
RX_BANDWIDTH = 1_000_000
RX_BUFFER_SIZE = 4096
BEACON_TONE_OFFSET_HZ = 100_000  # must match fc in beacon.py
TONE_BIN_WIDTH_HZ = 5_000        # detection bandwidth around the tone

# Gain mode matters more than anything else in this file.
#
# "slow_attack" AGC holds the DIGITAL level roughly constant by backing
# off receiver gain as received RF power rises -- which is exactly the
# quantity this controller is trying to measure. Left on, the ~23 dB of
# gradient between 60 m and 4 m collapses to near zero at the algorithm's
# input and run-and-tumble never converges. Use manual gain so dBFS
# tracks actual received power.
#
# Pick MANUAL_GAIN_DB so the tone stays inside the ADC's linear range
# across your whole working distance: high enough that the beacon is
# above the noise floor at max search range, low enough that it does not
# clip at closest approach. Check saturation warnings during the props-off
# walk test and adjust.
GAIN_MODE = "manual"             # "manual" strongly preferred; see above
MANUAL_GAIN_DB = 40              # 0-71 dB on the AD9361
SATURATION_WARN_FRACTION = 0.9   # warn if |sample| exceeds this fraction of full scale

# ---------------- MAVLINK CONFIG ----------------
MAVLINK_CONN = "/dev/ttyAMA0"    # or e.g. "udp:127.0.0.1:14550"
MAVLINK_BAUD = 921600

# ---------------- TERMINAL STANDOFF ----------------
# Where the drone should park once it has found the beacon, expressed as
# a slant range and an elevation angle so the geometry is unambiguous.
# 4 m at 45 deg puts it ~2.8 m up and ~2.8 m horizontally offset, rather
# than directly overhead.
#
# READ THE ACCURACY NOTE in the module docstring before trusting this --
# RSSI alone cannot hold this position to anything like metre precision.
STANDOFF_SLANT_M = 4.0
STANDOFF_ELEVATION_DEG = 45.0
STANDOFF_ALT_M = STANDOFF_SLANT_M * math.sin(math.radians(STANDOFF_ELEVATION_DEG))
STANDOFF_HORIZ_M = STANDOFF_SLANT_M * math.cos(math.radians(STANDOFF_ELEVATION_DEG))

# Gain-compensated tone power measured at STANDOFF_SLANT_M during
# calibration. MUST be measured in the real environment -- see docstring.
STANDOFF_POWER_DBFS = -15.0
STANDOFF_DEADBAND_DB = 2.0     # +/- band around it that counts as "parked"
BACKOFF_SPEED_MS = 0.5         # retreat speed when too close

# ---------------- GUIDANCE TUNING ----------------
SMOOTHING_ALPHA = 0.3          # EWMA smoothing on raw power
RUN_DURATION_S = 3.0           # commit to a heading this long before re-checking
SEARCH_SPEED_MS = 1.0          # m/s while hunting for a gradient
APPROACH_SPEED_MS = 2.0        # m/s once a run is improving
IMPROVEMENT_DB = 0.7           # gain over a run needed to count as "warmer"
SIGNAL_PRESENT_DBFS = -45.0    # below this, treat as no beacon in range at all
STRONG_SIGNAL_DBFS = -25.0     # power level considered "locked on"
SIGNAL_LOST_TIMEOUT_S = 4.0    # time below STRONG_SIGNAL before leaving HOLD
ALT_TOLERANCE_M = 0.5
ALT_CORRECT_SPEED_MS = 0.4
CONTROL_HZ = 5.0               # keep >= 2 Hz or GUIDED can time out
LOOP_DT = 1.0 / CONTROL_HZ


# ---------------- SDR ----------------

def connect_pluto():
    print(f"Connecting to Pluto SDR at {PLUTO_URI} ...")
    sdr = adi.Pluto(PLUTO_URI)
    sdr.rx_lo = int(RX_LO_HZ)
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.rx_rf_bandwidth = int(RX_BANDWIDTH)
    sdr.rx_buffer_size = RX_BUFFER_SIZE
    sdr.gain_control_mode_chan0 = GAIN_MODE
    if GAIN_MODE == "manual":
        sdr.rx_hardwaregain_chan0 = MANUAL_GAIN_DB
    print(f"Pluto ready, listening at {sdr.rx_lo / 1e6:.2f} MHz")
    print(f"Gain mode: {GAIN_MODE}"
          + (f" @ {MANUAL_GAIN_DB} dB" if GAIN_MODE == "manual" else
             "  <-- AGC flattens the RSSI gradient; prefer manual"))
    return sdr


def read_power_dbfs(sdr):
    """Broadband power across the whole RX bandwidth. Fine on the bench,
    but once the vehicle is flying, motor/ESC/VTX/GPS noise can push this
    up regardless of whether the beacon tone is actually present. Kept
    here for reference/bench testing; the flight loop uses
    read_tone_power_dbfs instead."""
    samples = sdr.rx()
    return 10 * np.log10(np.mean(np.abs(samples) ** 2) + 1e-12)


def read_tone_power_dbfs(sdr, tone_hz=BEACON_TONE_OFFSET_HZ, bin_width_hz=TONE_BIN_WIDTH_HZ):
    """
    Gain-compensated power in a narrow band around the beacon's known
    tone offset.

    Two things going on here:

    1. NARROWBAND. Integrating only around the tone rejects broadband EMI
       from the drone's own motors/ESCs/VTX/GPS that a whole-buffer RMS
       measurement would happily report as signal.

    2. GAIN COMPENSATION. The returned value has the receiver's current
       hardware gain subtracted out, so it tracks *received RF power*
       rather than the post-gain digital level. Without this, any AGC
       mode silently cancels the distance gradient the whole controller
       depends on. With it, readings stay comparable even if you switch
       gain modes or change MANUAL_GAIN_DB between flights -- though
       manual gain is still preferred, since AGC settling lags fast
       motion and its steps are quantised.

    Also flags ADC saturation: once samples clip, measured power stops
    rising with distance and the drone goes blind exactly when it is
    closest to the beacon. If you see this warning, lower MANUAL_GAIN_DB.
    """
    samples = sdr.rx()
    n = len(samples)

    full_scale = 2 ** 11  # AD9361 RX is 12-bit signed
    if np.max(np.abs(samples)) > SATURATION_WARN_FRACTION * full_scale:
        print("WARNING: RX near saturation -- lower MANUAL_GAIN_DB "
              "(power readings flatten out when clipped)")

    spectrum = np.fft.fft(samples)
    freqs = np.fft.fftfreq(n, d=1 / sdr.sample_rate)
    band = np.abs(freqs - tone_hz) <= (bin_width_hz / 2)
    band_power = np.sum(np.abs(spectrum[band]) ** 2) / n**2
    raw_dbfs = 10 * np.log10(band_power + 1e-12)

    return raw_dbfs - float(sdr.rx_hardwaregain_chan0)


class Ewma:
    """Exponentially-weighted moving average, used to smooth raw dBFS."""

    def __init__(self, alpha):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


# ---------------- MAVLINK ----------------

def connect_mavlink():
    print(f"Connecting to flight controller on {MAVLINK_CONN} ...")
    master = mavutil.mavlink_connection(MAVLINK_CONN, baud=MAVLINK_BAUD)
    master.wait_heartbeat()
    print(f"Heartbeat OK (system {master.target_system}, component {master.target_component})")
    return master


def request_message_rate(master, message_id, rate_hz):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0, message_id, int(1e6 / rate_hz), 0, 0, 0, 0, 0,
    )


def get_relative_alt_m(master, timeout=0.2):
    msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=timeout)
    if msg is None:
        return None
    return msg.relative_alt / 1000.0


def wait_for_guided(master, poll_hz=2):
    """
    Block until the pilot has armed, taken off, and switched to GUIDED
    (e.g. via an RC mode switch). The script deliberately does not force
    this itself -- launch and the decision to hand off control should
    stay a deliberate human action, not something that happens the
    moment this script starts.
    """
    print("Waiting for pilot to switch flight mode to GUIDED ...")
    while True:
        msg = master.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
        if msg is not None and mavutil.mode_string_v10(msg) == "GUIDED":
            print("GUIDED confirmed -- homing loop taking over.")
            return
        time.sleep(1.0 / poll_hz)


# Ignore position + acceleration + yaw, use velocity only. World frame
# (MAV_FRAME_LOCAL_NED) so we never need to know the vehicle's heading.
VELOCITY_ONLY_MASK = 0b0000111111000111


def send_velocity_ned(master, vn, ve, vd):
    """vn = North m/s, ve = East m/s, vd = Down m/s (NED: climbing is negative)."""
    master.mav.set_position_target_local_ned_send(
        0,
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        VELOCITY_ONLY_MASK,
        0, 0, 0,
        vn, ve, vd,
        0, 0, 0,
        0, 0,
    )


def hold(master):
    send_velocity_ned(master, 0, 0, 0)


# ---------------- RUN-AND-TUMBLE CONTROLLER ----------------

def signal_present(power_dbfs):
    """
    Whether the smoothed tone power indicates the beacon is actually in
    range, vs. just noise floor. An arbiter running RSSI homing
    alongside a search pattern (lawnmower, etc.) should call
    HomingController.compute_velocity() only when this is True, and run
    the search pattern otherwise.
    """
    return power_dbfs >= SIGNAL_PRESENT_DBFS


class HomingController:
    def __init__(self):
        self.state = "RUN"
        self.heading = random.uniform(0, 2 * math.pi)
        self.run_start_time = time.time()
        self.run_start_power = None
        self.last_strong_time = None

    def _velocity_for_heading(self, speed):
        vn = speed * math.cos(self.heading)
        ve = speed * math.sin(self.heading)
        return vn, ve

    def _tumble(self):
        self.heading = random.uniform(0, 2 * math.pi)
        self.run_start_time = time.time()
        self.run_start_power = None

    def compute_velocity(self, power_dbfs, relative_alt_m):
        """
        Pure function: latest smoothed power (+ current relative
        altitude) in, a proposed (vn, ve, vd) velocity out. Never
        touches MAVLink -- the caller decides whether to actually send
        it, which is what makes this safe to run alongside other
        subroutines that also drive the vehicle.
        """
        now = time.time()
        holding = self.state in ("HOLD", "PARKED")

        if power_dbfs >= STRONG_SIGNAL_DBFS:
            self.last_strong_time = now
            if not holding:
                self.state = "HOLD"
                holding = True
                print("[STATE] -> HOLD (signal locked, closing to standoff)")

        if holding:
            if self.last_strong_time and (now - self.last_strong_time) > SIGNAL_LOST_TIMEOUT_S:
                self.state = "RUN"
                self._tumble()
                print("[STATE] -> RUN (signal faded, resuming search)")
            else:
                return self._hold_velocity(relative_alt_m, power_dbfs)

        # RUN state (either continuing, or just dropped out of HOLD above)
        if self.run_start_power is None:
            self.run_start_power = power_dbfs

        elapsed = now - self.run_start_time
        improving = (power_dbfs - self.run_start_power) > 0
        speed = APPROACH_SPEED_MS if improving else SEARCH_SPEED_MS
        vn, ve = self._velocity_for_heading(speed)

        if elapsed >= RUN_DURATION_S:
            gain = power_dbfs - self.run_start_power
            if gain >= IMPROVEMENT_DB:
                print(f"[RUN] +{gain:.1f} dB -- keeping heading {math.degrees(self.heading):.0f} deg")
                self.run_start_time = now
                self.run_start_power = power_dbfs
            else:
                print(f"[RUN] {gain:+.1f} dB -- tumbling to a new heading")
                self._tumble()

        return vn, ve, 0.0

    def _hold_velocity(self, relative_alt_m, power_dbfs=None):
        """
        Park at the standoff. Two axes, handled separately:

        VERTICAL -- servo to STANDOFF_ALT_M. This part is reliable: it
        comes from the barometer/EKF, not from RSSI.

        HORIZONTAL -- range-hold on power rather than maximising it.
        Too strong means too close, so retreat along the reverse of the
        approach heading; too weak means back off too far, so resume
        closing. Inside STANDOFF_DEADBAND_DB, stop.

        The horizontal half is the weak one, and honestly so: RSSI gives
        range only, never bearing, so "retreat" can only mean reversing
        the heading that got us here. Expect metres of error, not
        centimetres -- see the accuracy note in the module docstring.
        """
        vd = 0.0
        if relative_alt_m is not None:
            error = STANDOFF_ALT_M - relative_alt_m
            if abs(error) > ALT_TOLERANCE_M:
                vd = -math.copysign(ALT_CORRECT_SPEED_MS, error)  # NED: negative = climb

        if power_dbfs is None:
            return 0.0, 0.0, vd

        power_error = power_dbfs - STANDOFF_POWER_DBFS
        if abs(power_error) <= STANDOFF_DEADBAND_DB:
            if self.state != "PARKED":
                self.state = "PARKED"
                print(f"[STATE] -> PARKED (~{STANDOFF_SLANT_M:.1f} m slant, "
                      f"{STANDOFF_ALT_M:.1f} m alt)")
            return 0.0, 0.0, vd

        if power_error > 0:   # too close -- back out along the reverse heading
            vn, ve = self._velocity_for_heading(BACKOFF_SPEED_MS)
            return -vn, -ve, vd

        vn, ve = self._velocity_for_heading(BACKOFF_SPEED_MS)  # too far -- close in
        return vn, ve, vd


def homing_step(sdr, master, power_filter, controller):
    """
    Run exactly one iteration -- read power, compute the desired
    velocity, and send it. No internal loop or sleep. Standalone runs
    (run_forever) call this directly since nothing else is driving the
    vehicle. If you're folding RSSI homing into an arbiter alongside
    other subroutines (a lawnmower search pattern, etc.), call
    controller.compute_velocity(power, relative_alt_m) yourself instead
    and only send when signal_present(power) says RSSI homing should be
    the active mode -- see INTEGRATING WITH OTHER SUBROUTINES above.
    """
    raw = read_tone_power_dbfs(sdr)
    power = power_filter.update(raw)
    alt = get_relative_alt_m(master)
    print(f"raw={raw:6.1f} dBFS  smoothed={power:6.1f} dBFS  state={controller.state}")
    vn, ve, vd = controller.compute_velocity(power, alt)
    send_velocity_ned(master, vn, ve, vd)


def run_forever(sdr, master):
    """Blocking loop -- what main() uses when this file runs standalone."""
    power_filter = Ewma(SMOOTHING_ALPHA)
    controller = HomingController()
    print("Homing loop running (Ctrl+C to stop and hold position) ...")
    try:
        while True:
            homing_step(sdr, master, power_filter, controller)
            time.sleep(LOOP_DT)
    except KeyboardInterrupt:
        print("\nStopping -- holding position.")
        hold(master)


def main():
    sdr = connect_pluto()
    master = connect_mavlink()
    request_message_rate(master, mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 5)
    wait_for_guided(master)
    run_forever(sdr, master)


if __name__ == "__main__":
    main()