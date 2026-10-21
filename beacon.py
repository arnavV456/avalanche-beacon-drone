"""
beacon.py

Stationary beacon transmitter. Runs on a laptop/ground machine with its
own ADALM-PLUTO and emits a continuous-wave (CW) tone that the drone's
receiver (rssi_homing_controller.py) homes in on.

WHAT IT EMITS:
  A pure complex exponential at TONE_OFFSET_HZ above the LO, transmitted
  cyclically. So with LO = 866.0 MHz and a 100 kHz offset, the actual
  emission sits at 866.1 MHz as a single narrow spike, not a spread of
  energy across the band.

WHY AN OFFSET TONE AND NOT JUST THE CARRIER:
  Transmitting exactly at the LO would put the beacon on top of the RX
  chain's own DC/LO leakage spike, which is always present and has
  nothing to do with the beacon. Offsetting by 100 kHz moves the beacon
  clear of that artifact so the receiver can measure it cleanly. This is
  also what lets the drone reject its own motor/ESC/VTX noise: the
  receiver only integrates power in a narrow band around this exact
  offset, so broadband interference elsewhere in the channel doesn't
  register as signal.

MUST STAY IN SYNC WITH THE RECEIVER:
  TONE_OFFSET_HZ here == BEACON_TONE_OFFSET_HZ in rssi_homing_controller.py
  LO_HZ here          == RX_LO_HZ in rssi_homing_controller.py
  If these drift apart, the drone's detector sits on an empty band and
  reads noise floor forever while the beacon transmits perfectly fine --
  an annoying failure because nothing errors out, it just never detects.

RF NOTE: 866 MHz sits inside India's 865-867 MHz de-licensed SRD band,
but de-licensed is not unconditional -- there are limits on radiated
power and carrier bandwidth. A CW tone is narrow, so bandwidth is not
the concern; transmit power is. Keep TX_GAIN_DB backed off unless you
have a specific reason, and verify against the current DoT notification
rather than assuming.
"""

import time

import adi
import numpy as np

# ---------------- CONFIG ----------------
PLUTO_URI = "ip:192.168.2.1"
LO_HZ = 866_000_000        # must match RX_LO_HZ on the drone
TONE_OFFSET_HZ = 100_000   # must match BEACON_TONE_OFFSET_HZ on the drone
SAMPLE_RATE = 2_500_000
TX_BANDWIDTH = 1_000_000

# Pluto TX gain is ATTENUATION: 0 = full power, -89.75 = minimum.
# Start well backed off for bench testing and only raise it for range tests.
TX_GAIN_DB = -20

TARGET_BUFFER_LEN = 2**14  # starting point; snapped up to a whole number of cycles
AMPLITUDE = 2**14          # Pluto DAC full-scale is roughly +/- 2**14


def snap_buffer_length(target, sample_rate, tone_hz, max_search=200_000):
    """
    Pick a buffer length holding a WHOLE number of tone cycles.

    tx_cyclic_buffer replays the buffer end-to-end forever, so if the
    buffer holds a fractional number of cycles the waveform jumps in
    phase at every wrap. That discontinuity smears energy across the
    spectrum -- and partly outside the receiver's narrow detection bin,
    costing range for no reason.

    The obvious 2**14 is exactly this case: at 2.5 MSPS a 100 kHz tone
    fits 655.36 times into 16384 samples. Nudging up to 16400 gives
    exactly 656 cycles and wraps seamlessly.

    Also keeps the length a multiple of 8 for DMA alignment.
    """
    n = int(target)
    limit = n + max_search
    while n <= limit:
        if n % 8 == 0 and (tone_hz * n) % sample_rate == 0:
            return n
        n += 1
    print(f"WARNING: no whole-cycle length found near {target}; using it as-is "
          f"(expect a phase glitch at each wrap).")
    return int(target)


def build_tone(sample_rate, tone_hz, n_samples, amplitude):
    """
    Build one period-aligned buffer of a complex CW tone.

    The tone is generated as I = cos, Q = sin, i.e. a complex exponential
    rotating in one direction only. That produces a single spike at
    +tone_hz rather than a real-valued cosine's mirrored pair at
    +/-tone_hz, which puts all the energy where the receiver is looking
    instead of splitting it.
    """
    t = np.arange(n_samples) / sample_rate
    phase = 2 * np.pi * tone_hz * t
    return (np.cos(phase) + 1j * np.sin(phase)) * amplitude


def main():
    print(f"Connecting to Pluto at {PLUTO_URI} ...")
    try:
        sdr = adi.Pluto(PLUTO_URI)
    except Exception as exc:
        print(f"Failed to connect to Pluto: {exc}")
        return

    sdr.tx_lo = int(LO_HZ)
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.tx_rf_bandwidth = int(TX_BANDWIDTH)
    sdr.tx_hardwaregain_chan0 = TX_GAIN_DB

    # Must be set BEFORE the first tx() call, otherwise the buffer plays
    # once and stops instead of repeating.
    sdr.tx_cyclic_buffer = True

    buffer_len = snap_buffer_length(TARGET_BUFFER_LEN, SAMPLE_RATE, TONE_OFFSET_HZ)
    iq = build_tone(SAMPLE_RATE, TONE_OFFSET_HZ, buffer_len, AMPLITUDE)

    emit_hz = LO_HZ + TONE_OFFSET_HZ
    print(f"Transmitting CW beacon at {emit_hz / 1e6:.3f} MHz "
          f"(LO {LO_HZ / 1e6:.3f} MHz + {TONE_OFFSET_HZ / 1e3:.0f} kHz offset)")
    print(f"Buffer: {buffer_len} samples = "
          f"{TONE_OFFSET_HZ * buffer_len / SAMPLE_RATE:.0f} whole cycles (seamless wrap)")
    print(f"TX attenuation: {TX_GAIN_DB} dB")
    sdr.tx(iq)

    print("Beacon is live. Ctrl+C to stop.")
    try:
        # The radio transmits from its own buffer in the background, so
        # this loop does nothing but keep the process (and therefore the
        # SDR context) alive.
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping beacon.")
    finally:
        sdr.tx_destroy_buffer()


if __name__ == "__main__":
    main()