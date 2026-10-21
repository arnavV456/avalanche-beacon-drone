"""
fake_sdr.py

Drop-in stand-in for adi.Pluto so search.py's real DSP path
(read_tone_power_dbfs: FFT, narrow tone bin, gain compensation,
saturation check) runs unchanged in SITL.

Each rx() call:
  1. asks `get_position()` where the drone is (local NED metres),
  2. computes the 3D distance to the hidden beacon,
  3. turns that into received power with a log-distance path-loss model,
  4. returns complex IQ: a tone at BEACON_TONE_OFFSET_HZ at that power,
     plus receiver noise, scaled by the current hardware gain and clipped
     to the 12-bit ADC range.

Because read_tone_power_dbfs subtracts the hardware gain back out, the
number the controller sees is the physical power from step 3 plus noise --
exactly what the real radio is supposed to give you.

Path-loss model (defaults match search.py's constants):
    P(d) = REF_POWER_DBFS - 10 * n * log10(d / REF_DIST_M) + fading
  -15 dBFS at 4 m, n = 2  ->  -25 dBFS at ~12.6 m, -45 dBFS at ~126 m.

Fading: real 866 MHz multipath swings power >10 dB over a few
decimetres. The controller samples every ~0.6 m at cruise, so successive
reads are effectively independent; `fading_std_db` therefore adds an
independent zero-mean Gaussian (in dB) to every rx() call.

Usage:
    master = connect_mavlink()
    sdr = FakeSdr(beacon_ned=(60, 40, 0),
                  get_position=local_position_from_master(master))
    power = read_tone_power_dbfs(sdr)
"""

import math

import numpy as np


class FakeSdr:
    def __init__(
        self,
        beacon_ned,
        get_position,
        tone_hz=100_000,
        sample_rate=2_500_000,
        buffer_size=4096,
        gain_db=40,
        tone_bin_width_hz=5_000,
        ref_power_dbfs=-15.0,
        ref_dist_m=4.0,
        path_loss_exp=2.0,
        noise_floor_dbfs=-60.0,
        fading_std_db=0.0,
        min_dist_m=0.5,
        seed=None,
    ):
        self.beacon_ned = tuple(beacon_ned)
        self.get_position = get_position
        self.tone_hz = tone_hz
        self.sample_rate = sample_rate
        self.rx_buffer_size = buffer_size
        self.rx_hardwaregain_chan0 = gain_db
        self.ref_power_dbfs = ref_power_dbfs
        self.ref_dist_m = ref_dist_m
        self.path_loss_exp = path_loss_exp
        self.noise_floor_dbfs = noise_floor_dbfs
        self.fading_std_db = fading_std_db
        self.min_dist_m = min_dist_m
        self.rng = np.random.default_rng(seed)

        # How many FFT bins read_tone_power_dbfs will integrate; needed to
        # turn the requested noise floor into a per-sample noise variance.
        freqs = np.fft.fftfreq(buffer_size, d=1 / sample_rate)
        self._band_bins = int(np.sum(np.abs(freqs - tone_hz) <= tone_bin_width_hz / 2))
        self._phase = 0.0
        self.last_distance_m = None   # exposed for logging / scoring
        self.last_true_dbfs = None    # noise-free power, for scoring

    # -- physics --------------------------------------------------------

    def distance_m(self):
        pos = self.get_position()
        if pos is None:
            return None
        return max(math.dist(pos, self.beacon_ned), self.min_dist_m)

    def received_power_dbfs(self, dist_m):
        """Gain-compensated tone power at this distance (no fading)."""
        return (self.ref_power_dbfs
                - 10 * self.path_loss_exp * math.log10(dist_m / self.ref_dist_m))

    # -- adi.Pluto interface -------------------------------------------

    def rx(self):
        n = self.rx_buffer_size
        gain = float(self.rx_hardwaregain_chan0)

        dist = self.distance_m()
        self.last_distance_m = dist

        # Noise: choose variance so the band-integrated floor, after the
        # gain is subtracted, equals noise_floor_dbfs.
        sigma2 = (n / self._band_bins) * 10 ** ((self.noise_floor_dbfs + gain) / 10)
        noise = (self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n)) \
            * math.sqrt(sigma2 / 2)

        if dist is None:   # no position yet: receiver sees noise only
            self.last_true_dbfs = None
            iq = noise
        else:
            power = self.received_power_dbfs(dist)
            if self.fading_std_db:
                power += self.rng.normal(0.0, self.fading_std_db)
            self.last_true_dbfs = power
            # read_tone_power_dbfs gives |FFT peak|^2 / n^2 = A^2 for a
            # tone of amplitude A, so A = 10^((power + gain) / 20).
            amp = 10 ** ((power + gain) / 20)
            t = np.arange(n) / self.sample_rate
            tone = amp * np.exp(1j * (self._phase + 2 * np.pi * self.tone_hz * t))
            self._phase = (self._phase + 2 * np.pi * self.tone_hz * n / self.sample_rate) \
                % (2 * np.pi)
            iq = tone + noise

        # 12-bit ADC clip -- this is what makes the saturation warning
        # (and the "goes blind when too close" failure) show up in SITL.
        full_scale = 2 ** 11
        return (np.clip(iq.real, -full_scale, full_scale - 1)
                + 1j * np.clip(iq.imag, -full_scale, full_scale - 1))


def local_position_from_master(master):
    """
    Position source for SITL: the most recent LOCAL_POSITION_NED seen on
    this pymavlink connection, as (north, east, down) metres.

    Reads master.messages (pymavlink's cache of the last message of each
    type) instead of calling recv_match, so it never steals messages from
    the controller's own reads. It only updates while something else is
    calling recv_match on `master` -- the homing loop already does. Ask
    the autopilot for the stream with
    request_message_rate(master, MAVLINK_MSG_ID_LOCAL_POSITION_NED, 10).

    Frame note: local NED origin is the EKF origin (home in SITL), so
    beacon_ned must be given in that same frame.
    """
    def get():
        msg = master.messages.get("LOCAL_POSITION_NED")
        if msg is None:
            return None
        return (msg.x, msg.y, msg.z)
    return get


if __name__ == "__main__":
    # Offline self-check against search.py's REAL read_tone_power_dbfs.
    from search import read_tone_power_dbfs

    pos = [0.0, 0.0, -15.0]
    sdr = FakeSdr(beacon_ned=(60, 40, 0), get_position=lambda: tuple(pos), seed=1)

    def measure(n_reads=20):
        return float(np.median([read_tone_power_dbfs(sdr) for _ in range(n_reads)]))

    # 1. Recovered power matches the model at several ranges, and the
    #    gradient the controller depends on is intact.
    print("  dist(m)   model(dBFS)   measured(dBFS)")
    for north in (0, 20, 40, 55, 59):
        pos[0] = north
        sdr.rx()
        d = sdr.last_distance_m
        want, got = sdr.received_power_dbfs(d), measure()
        print(f"  {d:7.1f}   {want:11.2f}   {got:14.2f}")
        assert abs(want - got) < 0.5, (d, want, got)

    # 2. Calibration points quoted in search.py's constants.
    assert abs(sdr.received_power_dbfs(4.0) - -15.0) < 1e-9
    assert abs(sdr.received_power_dbfs(12.6) - -25.0) < 0.1
    assert abs(sdr.received_power_dbfs(126.0) - -45.0) < 0.1

    # 3. Gain changes must not change the compensated reading.
    pos[0] = 20
    a = measure()
    sdr.rx_hardwaregain_chan0 = 30
    b = measure()
    sdr.rx_hardwaregain_chan0 = 40
    assert abs(a - b) < 0.5, (a, b)

    # 4. Out of range -> reads the noise floor, far below SIGNAL_PRESENT.
    far = FakeSdr(beacon_ned=(5000, 0, 0), get_position=lambda: (0, 0, -15), seed=2)
    floor = float(np.median([read_tone_power_dbfs(far) for _ in range(20)]))
    print(f"  noise-only reading: {floor:.1f} dBFS (set floor -60)")
    assert floor < -45

    # 5. Fading spreads readings by roughly the requested sigma.
    fad = FakeSdr(beacon_ned=(60, 40, 0), get_position=lambda: (0, 0, -15),
                  fading_std_db=6.0, seed=3)
    spread = float(np.std([read_tone_power_dbfs(fad) for _ in range(300)]))
    print(f"  fading std: {spread:.1f} dB (asked 6.0)")
    assert 5.0 < spread < 7.0

    # 6. Too much gain at close range clips and the controller warns. (At
    #    the default 40 dB the tone peaks ~70 counts even at 1 m, far from
    #    the 2048 full scale, so it takes max gain to see this.)
    import contextlib
    import io
    pos[:] = [60, 40, -0.5]
    sdr.rx_hardwaregain_chan0 = 71
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        clipped = read_tone_power_dbfs(sdr)
    assert "saturation" in buf.getvalue(), "expected saturation warning"
    assert clipped < sdr.received_power_dbfs(0.5) - 1, "clipped reading should read low"
    print(f"  saturation warning fired; clipped reads {clipped:.1f} dBFS "
          f"vs true {sdr.received_power_dbfs(0.5):.1f}")
    sdr.rx_hardwaregain_chan0 = 40

    # 7. No position yet -> noise only, no crash.
    none = FakeSdr(beacon_ned=(0, 0, 0), get_position=lambda: None, seed=4)
    assert read_tone_power_dbfs(none) < -45

    print("OK")
