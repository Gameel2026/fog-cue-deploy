# Amendment 4: device set for the remaining runs

Committed before E2 and C1 on the added device, and before the live, streaming and energy runs are repeated with the Amendment 3 version.

## Device labels

A third device is added to extend the screen-off evaluation to a second vendor build. From this amendment on, the devices are labelled as follows.

| Label from this amendment on | Label in the plan and Amendments 1–3 |
|---|---|
| Device A (Android 13, vendor build) | added by this amendment |
| Device B (Android 17, stock build) | Device B |
| Device C (Android 16, vendor build) | Device A |

The result folders of Device C were renamed to match (`out_C`, `out_C2`, and live runs ending in `_C`). Their contents are unchanged.

## Reason

Device C has no wake-up accelerometer. In a 15-min screen-off check with the Amendment 3 version, its vendor build again suspended the application after the screen was turned off. The run gave 1180 of the expected 1800 decisions. The timing correction of Amendment 3 worked as intended, and the run ended after 938 s.

Device C passed every test that keeps the screen on: E2 (both runs), C1 and the screen-on live run.

## Added device

Device A runs the same application version (model hash `5b53630c…`). Before this amendment it was checked as follows:

- sensor check with a 10 ms period: 100.09 Hz, interval SD 0.023 ms, 100% of intervals within 5%, no gaps; the accelerometer has no wake-up variant;
- 15-min screen-off live check, unplugged: 1779 decisions, decision agreement 100% with offline replay, max |Δp| 5.2e-8, no sample gaps, all decisions within 100 ms of the last sample of their window (median 21.7 ms, max 43.9 ms).

These checks are kept in the repository under `phone_results/A/`. They are not counted as L1 runs.

On Device A, battery use is set to unrestricted for the application and automatic restart is turned off.

## Runs per device

| Device | E2 | C1 | L1 screen off | L1 screen on | S1 | EN1 |
|---|---|---|---|---|---|---|
| A (Android 13, vendor build) | 1 run | 1 run | 2 runs | 1 run | 1 run | 1 run |
| B (Android 17, stock build) | done | done | 2 runs | 1 run | 1 run | 1 run |
| C (Android 16, vendor build) | done | done | – | 1 run | – | – |

E2 and C1 on Device A use `fd_phone.sh` with the Amendment 2 build. All L1, S1 and EN1 runs listed above use the Amendment 3 version. The Device C live runs are kept in the repository and reported as an additional screen-on check.

## Run conditions

| Condition | L1 and S1 | EN1 |
|---|---|---|
| Charger | unplugged before the run starts | unplugged before the run starts |
| Battery level at start | 60–90% | 80–90% |
| Airplane mode | off (S1 needs Wi-Fi) | on |
| Other applications | closed | closed |
| Power saving mode | off | off |
| Device position | on a desk, moved by hand at the planned minutes | on a desk, not moved |

For each run, the start time, battery level, screen state and airplane mode are noted. Each run is analysed before the next one starts. The application is not changed between runs.

## Current units

The battery service of Device A reports current in mA, while Devices B and C report µA. `fd_live.py` takes `--current-unit ua|ma`, with µA as the default, and writes the unit used to `energy.json`. The charge counter of Device A did not change during the 15-min check. The charge-counter estimate is therefore reported only for runs where the counter changes, and is left out otherwise.

## Raw sample files

Raw sample logs are stored compressed with gzip (`raw.csv.gz`) to stay within the upload size limit. The compression is lossless. `fd_live.py` reads `raw.csv` or `raw.csv.gz`.

The model, thresholds, experiments and criteria are unchanged.
