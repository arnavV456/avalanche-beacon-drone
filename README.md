# Avalanche Victim Detection Drone: RSSI Beacon Search

A drone-mounted software-defined radio that finds an avalanche beacon. People
carry a beacon that transmits a continuous-wave (CW) tone; the drone carries an
**ADALM-Pluto SDR** driven by a **Raspberry Pi**, flies a **lawnmower sweep**,
turns signal strength at each position into a **probability map** of where the
beacon is, then takes a **closer look** over the most likely area. A camera for
human detection is planned but not integrated.

> **Status.** The full pipeline works in ArduCopter SITL. **Nothing has been
> run on real hardware yet.** The signal model (section 3) is an assumption
> until it is measured on the real radios. Sections 11 and 12 list exactly what
> is and is not verified.

**The Pi is passive.** It listens and logs; the pilot flies everything from
QGroundControl. Analysis runs on a PC after each flight.

## Contents

1. [How the field test works](#1-how-the-field-test-works)
2. [Repository layout](#2-repository-layout)
3. [How it works (the algorithm)](#3-how-it-works-the-algorithm)
4. [Hardware and links](#4-hardware-and-links)
5. [One-time setup](#5-one-time-setup)
6. [Field-day procedure](#6-field-day-procedure)
7. [Checks before the first flight](#7-checks-before-the-first-flight)
8. [Safety and autopilot settings](#8-safety-and-autopilot-settings)
9. [Troubleshooting](#9-troubleshooting)
10. [After the flight](#10-after-the-flight)
11. [Results so far](#11-results-so-far)
12. [Known limitations and open items](#12-known-limitations-and-open-items)
13. [Design history](#13-design-history)
14. [Running the simulation](#14-running-the-simulation)
15. [Demo filming and pack list](#15-demo-filming-and-pack-list)

---

## 1. How the field test works

1. **Place the beacon** on the ground. Read its position off a phone and note
   latitude and longitude (that is the ground truth).
2. **Fly a lawnmower sweep** planned in QGroundControl, in AUTO. The Pi starts
   logging by itself when the drone arms and stops when it disarms.
3. **Land. Analyse on the PC:** `analyze.py` produces a heat map, a satellite
   map page, and an estimate of the beacon position.
4. **Check the heat map.** If it looks right, fly a **tighter lawnmower** over
   the hot area (draw your own in QGC, or use the generated `refine.plan`).
5. **Land. Analyse both flights together** for the final estimate and its
   distance from the phone reading.

The Pi needs no SSH session, no tmux and no file names in the field: after
the one-time setup its logger runs as a boot service.

## 2. Repository layout

| File | Runs on | Purpose |
| --- | --- | --- |
| `autolog.py` | Pi (boot service) | Always-on passive logger. Idle while disarmed; on arm opens `data/flight_NNN_<time>.csv` plus `.origin.json`, on disarm closes it. Finds the Pixhawk by probing serial ports for a MAVLink heartbeat. Writes `data/STATUS.txt`. Flushes every row. Exits non-zero on Pixhawk silence or SDR error so the service restarts it. Sends **no** flight commands (only two telemetry-rate requests). |
| `install_service.sh` | Pi, once | Installs and starts the `capstone-logger` boot service (`--print`, `--remove` also available). |
| `analyze.py` | PC | The post-flight command: flights + phone GPS -> `heatmap.png`, `map.html` (satellite), `result.txt/json`, optional `refine.plan`. |
| `estimator.py` | PC / Pi | The estimator (grid posterior). CLI and `--selftest`. |
| `patterns.py` | library | Lawnmower and refine-box geometry, lat/lon helpers. |
| `fake_sdr.py` | library | Simulated Pluto for SITL. Imported by the loggers even on real hardware. |
| `search.py` | library | SDR configuration and `read_tone_power_dbfs`. Also contains the original run-and-tumble RSSI homing controller, no longer used. |
| `beacon.py` | beacon laptop | Transmits the CW tone from a second Pluto. |
| `logger.py` | Pi | Manual single-run logger (superseded by `autolog.py`). Defines the CSV `COLUMNS` and the origin helper. |
| `make_plan.py` | PC / Pi | Older one-shot sweep CSV -> refine plan (`analyze.py --make-plan` supersedes it). |
| `refine.py` | Pi | Fully autonomous version (GUIDED hand-off, Pi flies the refine pass). Tested in SITL only; **not used** in the passive test. |
| `evaluate.py` | PC | Monte Carlo of the whole pipeline without SITL. |
| `sample/` | any | Two simulated flights (`sample/data/`), their outputs (`results_stage1/`, `results_stage2/`) and older samples, for dry runs. |

Data formats:

- **Flight CSV** columns: `t_s, n_m, e_m, d_m, power_dbfs, mode, armed, true_dist_m, true_dbfs` (`n/e/d` are local NED metres from the autopilot's origin; the last two are simulation-only).
- **`<flight>.origin.json`**: `{"lat0": ..., "lon0": ...}`, the latitude/longitude of the autopilot's local-NED origin. The origin resets at every power-up, so keep each CSV next to its origin file; the estimator uses them to align flights after a battery swap.
- **`data/STATUS.txt`**: what the logger is doing right now (`READY (waiting for arm)`, `LOGGING`, and so on).

## 3. How it works (the algorithm)

**Signal chain.** The beacon emits a tone at LO + 100 kHz. The receiver takes
4096 samples, FFTs them, integrates only a 5 kHz bin around the tone (which
rejects motor and ESC noise elsewhere in the band), subtracts the hardware
gain so the reading tracks received RF power, and reports dBFS. Gain is
**manual** (40 dB by default): AGC would flatten the distance gradient the
whole method depends on.

**Signal model (an assumption until measured):**

    P(d) = ref_power - 10 * n * log10(d / ref_dist)   [+ unknown constant offset]
    ref_power = -15 dBFS at ref_dist = 4 m,  n = 2 (free space)
    fading    = independent Gaussian in dB per sample, about 5 dB std

**Estimator** (`estimator.py`). A grid posterior over the beacon's (north,
east) position. For each candidate cell it predicts the power at every sample
position (3D distance; beacon assumed at ground level), takes the residuals,
subtracts their mean (this fits the unknown constant offset), and scores them
with a Gaussian likelihood. The output is the most likely position, the
posterior mean, `r90` (the radius around the estimate holding 90% of the
probability) and a heat map. It exists because the loudest single reading is a
poor estimator: fading swings individual readings by 10 dB or more.

**Two-stage search.**

1. Lawnmower sweep, then estimate.
2. A refine box centred on the estimate, half-width = 2 x `r90` (limited to 8 to 30 m), lanes **perpendicular** to the sweep lanes (that crossing geometry is what shrinks the uncertainty), 5 m lane spacing.
3. Re-estimate from all samples.

**Filters.** Only rows where the drone is armed, at least 3 m up, and with
power above -50 dBFS are used for estimation.

## 4. Hardware and links

- **Flight controller:** Pixhawk running ArduCopter (previously flown in AUTO and manually without issue).
- **Companion:** Raspberry Pi with an ADALM-Pluto on USB (default address `ip:192.168.2.1`).
- **Beacon:** a second Pluto + laptop running `beacon.py`.
- **Pixhawk to Pi:** USB cable. The device name is not fixed, because the Pluto can also appear as a `/dev/ttyACM*` console, so `autolog.py --conn auto` probes ports for a MAVLink heartbeat.
- **QGroundControl:** on a laptop over the telemetry radio (one program per USB port, so QGC must not use the Pixhawk USB cable).
- **Power:** the Pi from its own supply or power bank, not from the Pixhawk's USB.
- **Camera:** planned human detection; not part of this pipeline.

Frequency: the code uses **866 MHz** (`RX_LO_HZ` in `search.py`, `LO_HZ` in
`beacon.py`; they must match each other and the antennas). The beacon was
described as 433 MHz; see [section 12](#12-known-limitations-and-open-items).

## 5. One-time setup

On your PC, copy the project to the Pi (`<user>` and `<pi-address>` are the Pi's login and address):

```bash
rsync -av --exclude __pycache__ --exclude .vscode --exclude .venv --exclude results ./ <user>@<pi-address>:~/capstone/
```

On the Pi, install the packages:

```bash
sudo apt update && sudo apt install -y python3-venv python3-numpy python3-matplotlib libiio0 libiio-utils rsync
```

```bash
python3 -m venv --system-site-packages ~/capstone/venv && ~/capstone/venv/bin/pip install pymavlink pyadi-iio
```

Check the imports and the Pluto:

```bash
~/capstone/venv/bin/python3 -c "import numpy, matplotlib, pymavlink, adi; print('ok')"
```

```bash
ping -c 3 192.168.2.1
```

If there is no reply, `ip a` should show a new USB network interface; it may need the static address 192.168.2.10. Plug in the Pixhawk and Pluto and run `ls /dev/serial/by-id/` to confirm the Pixhawk is listed.

Install the always-on logger (once):

```bash
cd ~/capstone && bash install_service.sh
```

Then `cat ~/capstone/data/STATUS.txt` should say `READY (waiting for arm)`. If not, read `journalctl -u capstone-logger -n 30`.

**Prove it works (props OFF):** arm the Pixhawk for about 15 seconds and disarm; `ls ~/capstone/data` should show a `flight_001_...csv` and `.origin.json` with rows in it. Delete the test files (`rm ~/capstone/data/flight_*`). Then `sudo reboot` and check `STATUS.txt` again to prove it starts at power-up.

**Join the field network.** Make the Pi join your phone's hotspot so it connects by itself in the field: `sudo nmtui`, add the hotspot, check it connects.

**Dry-run the analysis on the PC** with the included sample data:

```bash
python3 analyze.py sample/data/flight_001_*.csv --truth -35.3630419 149.1653696 --make-plan --out /tmp/try
```

Open `/tmp/try/map.html` in a browser (needs internet for map tiles; it shows Canberra because the data is simulated). Then open `sample/results_stage1/refine.plan` in QGC (Plan, File, Open): you should see 8 waypoints in a small east-west lawnmower. If QGC will not load it, use `refine.waypoints` from the same folder.

Also pick a phone GPS app that shows accuracy and decimal degrees, and try it once.

## 6. Field-day procedure

1. **Check the drone.** Props, battery, GPS lock, RC mode switch (AUTO and LOITER), geofence and failsafes ([section 8](#8-safety-and-autopilot-settings)).
2. **Place the beacon.** Antenna vertical, nobody standing next to it. Stand at it and read your phone GPS (decimal degrees, wait about 30 seconds); write down latitude and longitude. Start `python3 beacon.py` on the beacon laptop.
3. **Power the Pi with the drone, before arming.** Optional check: `cat ~/capstone/data/STATUS.txt` should say `READY (waiting for arm)`.
4. **Plan the sweep in QGC.** Lawnmower about 40 m x 40 m, 5 m altitude, lane spacing about 10 m, 3 m/s, **return-to-launch unticked** (otherwise the drone flies home when the sweep ends). Upload. Optionally save the plan (File, Save) for the planned-versus-flown overlay.
5. **Fly the sweep** in AUTO, land, disarm. The Pi has written `flight_001_<time>.csv`.
6. **Pull the data and analyse on the PC:**

```bash
rsync -av <user>@<pi-address>:~/capstone/data/ ./data/
```

```bash
python3 analyze.py data/flight_001_*.csv --truth LAT LON --make-plan
```

   Add `--plan sweep.plan` if you saved the plan. Leave out `--make-plan` to draw your own refine mission.

7. **Check the heat map.** Open `results/<flight>/heatmap.png` and `map.html`. Good signs: the signal is strongest near the blob, and the estimate is within a few metres of the phone position. If the map is flat or the estimate sits on the edge of the swept area, do not refine; see [troubleshooting](#9-troubleshooting).
8. **Fly the closer look.** Draw a tighter lawnmower over the hot area in QGC, or open `results/<flight>/refine.plan` (Plan, File, Open), check the map, upload. Take off, AUTO, land. The Pi writes `flight_002_...` by itself.
9. **Final answer:**

```bash
rsync -av <user>@<pi-address>:~/capstone/data/ ./data/
```

```bash
python3 analyze.py data/flight_001_*.csv data/flight_002_*.csv --truth LAT LON
```

`analyze.py` options: `--truth LAT LON` (beacon from phone), `--plan FILE` (QGC mission to overlay), `--make-plan` (+ `--alt 5 --spacing 5 --speed 3`), `--sigma 5` (expected fading in dB), `--exp 2` (path-loss exponent), `--min-alt 3`, `--min-power -50`, `--origin LAT LON` (override a missing origin file), `--out DIR`, `--no-html`.

Outputs, in `results/<first-flight>_<n>flights/`: `heatmap.png` (signal along the path on the left, probability map on the right), `map.html` (Leaflet on satellite imagery: probability overlay, signal dots with dBFS tooltips, flown path, planned mission, estimate circle, phone-GPS beacon ring), `result.txt` and `result.json` (the numbers), and `refine.plan` / `refine.waypoints` with `--make-plan`.

> **How to read the "error".** Phone GPS is good to roughly 3 to 5 m and the
> drone's GPS adds 1 to 2 m, so an error under about 5 m means the two agree,
> not that one method beat another. In a simulated test with a deliberately
> 3 m-off phone reading, the reported "error" was 4.2 m while the estimate was
> about 1 m from the true beacon.

## 7. Checks before the first flight

Three ground checks, each removing one unknown before you commit a battery.
The logger service also uses the Pluto, so stop it for checks A and B
(`sudo systemctl stop capstone-logger`) and start it again afterwards
(`sudo systemctl start capstone-logger`).

| Check | What you do | Go if | If not |
| --- | --- | --- | --- |
| A. Receiver hears the beacon | Beacon on, then off, receiver 2 m away; read power | On is tens of dB above off, no saturation warning | Frequency mismatch, then antennas |
| B. Motors do not drown it | Props OFF, arm, spin motors, beacon off; read power | Stays below -50 dBFS | Move the Pluto or antenna away from the ESCs, or shield it |
| C. Walk test | Carry the armed drone, props off, along the sweep path, then analyse | Estimate within about 5 m of the measured spot | Do not fly; check frequency, try `--sigma 8` |

Command for checks A and B (on the Pi, Pluto attached):

```bash
~/capstone/venv/bin/python3 -c "from search import *; import time; s=connect_pluto(); [print(round(read_tone_power_dbfs(s),1)) or time.sleep(0.2) for _ in range(40)]"
```

**Check C, the walk test.** Remove the props first. With the service running,
arming starts a new log file by itself, and the estimator discards rows where
the drone was not armed, so arm in STABILIZE (props off), carry the airframe
at about 1.5 m, body away from the antenna, along a three-lane path marked
with flags, then disarm. Pull the data and run:

```bash
python3 analyze.py data/flight_001_*.csv --min-alt 0.5 --truth LAT LON
```

The default `--min-alt 3` would discard a hand-carried log, hence the lower
value. Delete the test files afterwards (`rm ~/capstone/data/flight_*` on the
Pi). This is the one check that shows whether the real signal behaves like the
simulation, and it needs no propellers.

## 8. Safety and autopilot settings

The Pi does not control the drone in this test, so the autopilot's own safety
settings are what protect you. Set them in QGC before the first flight
(confirm exact parameter names and values against your firmware):

| Setting | Suggested | Why |
| --- | --- | --- |
| Geofence (`FENCE_ENABLE`, `FENCE_TYPE`) | on, circle plus max altitude | Backstop for any bad plan |
| Fence radius (`FENCE_RADIUS`) | about 30 m beyond the sweep box | Contains the flight |
| Max altitude (`FENCE_ALT_MAX`) | about 30 m | Sweep is at 5 m |
| Fence action (`FENCE_ACTION`) | RTL or land | A breach should bring it home |
| Battery failsafe | RTL at low, land at critical | Sweep plus refine is roughly 5 minutes |
| RC failsafe | RTL | Lost link means come home |
| `WP_YAW_BEHAVIOR` | 0 (never change yaw) | A fixed heading keeps the antenna pattern constant; otherwise antenna gain varies with direction of travel and biases the map |

**Site.** Open ground, at least 150 m x 150 m, no people, trees or wires within
the box plus 30 m. Check your local drone regulations and any permission you
need. **Beacon placement:** on the ground or a stake at a fixed height, antenna
vertical, nobody next to it (a body shifts the signal). **People:** one pilot
with the RC in hand for the whole flight, a separate person on the laptop and
another filming; nobody within 20 m of the box during flight; never place the
beacon on a person.

**Abort criteria.** Pilot flips to LOITER or RTL for: drone leaves the box,
unexpected altitude change, unusual sounds or vibration, GPS or EKF warnings,
battery below 30 percent, anyone entering the area. The Pi sends nothing to
the drone, so nothing on it can fight the pilot.

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `STATUS.txt` does not say READY, or does not exist | Service not running, Pixhawk not found, or Pluto not connected | `systemctl status capstone-logger`, then `journalctl -u capstone-logger -n 30`. Check both USB cables and `ls /dev/serial/by-id/`. QGC must not use the Pixhawk USB cable |
| No `flight_...csv` after a flight | The Pi never saw the drone arm, or there was no position yet | Check `STATUS.txt` during the flight; wait for a GPS lock before arming |
| No `.origin.json`, or analysis says no origin | No GPS fix when the drone armed | Pass `--origin LAT LON` (the takeoff pad, from QGC) to `analyze.py` |
| `analyze.py` says too few usable samples | Rows count only if armed, 3 m or more up, and power above -50 dBFS | Check you flew above 3 m; run check A; look at the `power_dbfs` column |
| Saturation warning in the logger log | Gain too high for the distance (40 dB is an untested default) | `journalctl -u capstone-logger \| grep saturation`. Lower `MANUAL_GAIN_DB` in `search.py` (for example to 30), copy it to the Pi, `sudo systemctl restart capstone-logger` |
| Power barely changes with distance | Beacon too weak or antenna wrong | Check the `Gain mode: manual @ 40 dB` line in the log; move closer; raise `TX_GAIN_DB` in `beacon.py` within the legal limit |
| Estimate far off, or radius huge | Noise counted as signal, beacon outside the swept box, or weak signal | Re-run `analyze.py` with `--min-power -45 --sigma 8` and look at the map; fly a bigger sweep |
| QGC will not load the `.plan` | File layout not accepted by your QGC version | Open `refine.waypoints`, or type in the waypoint latitudes and longitudes `analyze.py` printed |
| `rsync` from the Pi fails | Pi not on the same network | Phone hotspot on and joined (`sudo nmtui`); or take the SD card or a USB stick |
| Two flights do not line up | An `.origin.json` is missing or not beside its CSV | Keep each pair together (`rsync` of the whole data folder does); `analyze.py` warns |

**Fallback demos, best first.** (1) A smaller flight with a clean result. (2)
Sweep only: fly, land, analyse, show the heat map. (3) The walk test (check C),
clearly labelled as carried by hand. (4) The simulated run from `sample/`,
clearly labelled as simulation. Be explicit about which clips are simulated and
which are real.

## 10. After the flight

- [ ] Pull the whole data folder and keep the `results/` folders.
- [ ] Download the autopilot log and the QGC telemetry log.
- [ ] Back up any video before reformatting cards.
- [ ] Write down the beacon's phone GPS position and its accuracy reading.

Record these, because the numbers mean nothing without them: frequency and
beacon `TX_GAIN_DB`; receiver `MANUAL_GAIN_DB`; sweep box, altitude, lane
spacing and speed; beacon position and height; wind speed and direction;
estimate after the sweep and after both flights with their 90% radii; the
fitted offset; anything odd (warnings, noise, motor behaviour).

**Reading the result.** Within about 5 m of the phone position: the approach
and the model work well enough to build on (5 m is a judgement, not a measured
threshold). 10 m or worse: the cause is most likely the signal model (path-loss
exponent or fading), not the flying; run the walk-test log with a larger
`--sigma` and a different `--exp` to see what fits. A large fitted offset (many
dB) means the -15 dBFS at 4 m reference is far off, which is harmless for the
position but is what calibration is for.

## 11. Results so far

All simulated; none on real hardware. The simulated signal uses the same model
as the estimator, so these numbers show the pipeline and flight logic work, not
that the real signal will behave this well.

| Test | Result |
| --- | --- |
| Autonomous pipeline in ArduCopter SITL, 10 full runs (7 beacons inside the box, 3 outside), 5 dB fading | Median final error **1.45 m**, worst 2.2 m; truth inside the reported 90% radius in 10 of 10; refine improved 9 of 10 |
| Monte Carlo, 600 trials (`evaluate.py`), median final error | Inside the box: 0.7 / 0.9 / 1.2 m at 3 / 5 / 8 dB fading. Outside: 0.6 / 0.8 / 1.3 m |
| Passive flow in SITL (sweep, plan, second flight, combine) | Sweep alone about 2 m, combined about 0.5 m from the true beacon |
| Always-on logger in SITL, two AUTO flights | Two numbered files with origins, status file correct, clean shutdown |
| `analyze.py` on those flights | Estimate within about 1 m of the true beacon; satellite page renders without errors |
| `fake_sdr.py` self-test | Recovered power matches the model within about 0.1 dB; gain compensation and saturation warning correct |

Caveats: the Monte Carlo assumes exact waypoint following; SITL is one
flight-dynamics model; there were seven inside-the-box SITL runs. SITL sweep
errors (median about 4 m inside) were worse than the Monte Carlo (about 1.8 m),
and the cause was not isolated. The estimator is overconfident when the beacon
is outside the swept box and readings fall below the noise cutoff (the 90%
radius covered the truth only about 68% of the time at 8 dB fading outside).

## 12. Known limitations and open items

- **Frequency.** The code is at 866 MHz (`RX_LO_HZ` in `search.py`, `LO_HZ` in `beacon.py`); the beacon was described as 433 MHz. Both files must match each other and the antennas. Decision pending; neither file has been changed. 865 to 867 MHz is the delicensed band cited in `beacon.py`; whether 433 MHz is covered in India is unverified.
- **Signal model unmeasured.** Exponent 2, -15 dBFS at 4 m (fitted away as an offset), about 5 dB fading, and `MANUAL_GAIN_DB = 40` are placeholders until a calibration walk is done. The -50 dBFS noise cutoff is a constant in `estimator.load_samples`.
- **Nothing run on real hardware:** Pi package installs (`pyadi-iio`, libiio), the boot service (the unit file was checked with `systemd-analyze verify` only), Pixhawk detection with both USB devices attached, real RSSI behaviour, ESC/motor noise at the tone, antenna pattern on the airframe.
- **QGC loading of the generated `.plan`** is untested (the layout follows QGC's documented format; `.waypoints` and a printed lat/lon list are fallbacks). Parsing of QGC Survey plans for the `--plan` overlay is unverified; only simple-waypoint plans were tested.
- **No geofence or beacon-never-found timeout in code.** Rely on autopilot settings.
- **Beacon height.** The estimator assumes the beacon is at takeoff altitude (`beacon_down`); wrong on slopes.
- **Camera human detection** is not integrated, and a buried victim is not visible to a camera anyway; the beacon search does the finding.
- `map.html` needs internet for the map library and tiles.
- Possible next work: a calibration-fit script (fit exponent, reference power and fading from a walk log), camera integration, in-code geofence and timeouts, a larger SITL batch.

## 13. Design history

1. The original `search.py` used a **run-and-tumble RSSI homing controller** (GUIDED velocity setpoints). Review found: the frequency mismatch; its 0.7 dB improvement threshold cannot be met at long range (a 3 m run toward the beacon gains only 0.45 dB at 60 m), so it degrades to a random walk beyond about 40 m; altitude was only controlled in the hold state; heartbeats from any system were accepted; no failsafes.
2. A **lawnmower sweep with a heat map** is better: bounded time, deterministic, and the samples can be fitted to a model rather than climbing a gradient.
3. QGC's survey tool flies the sweep in AUTO; the companion only observes.
4. Built the SDR simulator, logger, estimator, an autonomous refine pass (`refine.py`) and a Monte Carlo.
5. Decision: the Pi must **not** control the drone in the first real test, and the test must be as simple as possible: passive logging, pilot-flown missions, USB link, 5 m altitude.
6. Streamlined to the current flow: always-on logger, automatic per-flight files, phone GPS as ground truth, one analysis command, satellite map view.

## 14. Running the simulation

ArduPilot SITL must be built (`sim_vehicle.py`), with MAVProxy available. Three terminals:

```bash
python3 ~/ardupilot/Tools/autotest/sim_vehicle.py -v ArduCopter --no-rebuild --no-mavproxy --speedup 3
```

```bash
mavproxy.py --master tcp:127.0.0.1:5760 --out udp:127.0.0.1:14550 --out udp:127.0.0.1:14551 --daemon
```

```bash
python3 autolog.py --conn udp:127.0.0.1:14551 --beacon 22 14 0 --fading 5
```

`--beacon N E D` places a hidden simulated beacon (local NED metres) and makes
the logger use the simulated Pluto. Fly missions from QGC on port 14550 (or a
script); each arm-to-disarm writes a flight file. Gotchas learned: in AUTO,
ArduCopter will not take off from the ground and auto-disarms unless
`AUTO_OPTIONS=3` is set; SITL's extra serial ports only open after a first
client connects, which is why MAVProxy sits in front.

Other self-tests: `python3 fake_sdr.py`, `python3 estimator.py --selftest`,
`python3 patterns.py`, and `python3 evaluate.py` (about 4 minutes for 600 trials).

## 15. Demo filming and pack list

**Shots:** the problem (talking head); the beacon being placed with the tape
measure or phone reading; hardware close-ups (drone, Pi, Pluto, antenna); the
QGC survey plan; takeoff and sweep (wide, tripod); the analysis printing the
estimate; the first heat map and loading the refine plan; the second flight;
the final heat map next to the first (the before-and-after is the key shot);
walking to the estimated spot and finding the beacon. Show real warnings; say
what each stage does out loud while filming. Keep spectators 20 m or more from
the drone. Do not claim the camera detects a human: it is not part of this
pipeline. Be explicit about which clips are simulated and which are real.

**Pack list.** Aircraft: drone, spare props and tools, at least three flight
batteries and charger, LiPo bag, Pi with its own power bank, receiver Pluto and
antenna, spare USB cables (including a short Pixhawk-to-Pi cable), RC
transmitter. Beacon: beacon Pluto, antenna, laptop, power bank, stake or box.
Ground: laptop with QGC and telemetry radio, phone with hotspot and a GPS app,
spare SD card. Measuring and safety: 30 m tape, compass, flags or cones,
hi-vis vests, first aid kit, fire extinguisher or sand bucket for LiPos.
Filming: two cameras or phones, tripod, spare batteries, a way to screen-record.
