# Run log

Device labels follow Amendment 4. Times are local device time (UTC+3) on 29 October 2026. All application runs use the model with hash `5b53630c…`.

## E2 and C1 (Termux, `fd_phone.sh`)

| Folder | Device | Build | Result |
|---|---|---|---|
| `out_C` | C | before Amendment 2 | decision agreement 100%; max \|Δp\| 1.2e-3 in 42 DeFOG windows (see Amendment 2) |
| `out_B` | B | before Amendment 2 | as `out_C` |
| `out_C2` | C | Amendment 2 | decision agreement 100%; max \|Δp\| 1.64e-6 |
| `out_B2` | B | Amendment 2 | decision agreement 100%; max \|Δp\| 1.64e-6 |
| `out_A` | A | Amendment 2 | decision agreement 100%; max \|Δp\| 1.64e-6; results identical to `out_B2` byte for byte |

C1 median processing per decision: 0.42 ms (C), 0.38 ms (B), 1.79 ms (A).

## Checks before Amendment 4 (not counted as L1 runs)

| Folder | Device | Content |
|---|---|---|
| `A/sensorcheck_20261029_120007` | A | sensor check |
| `A/live_20261029_120751` | A | 15-min screen-off check |
| `live/live_20261029_110747_C` | C | 15-min screen-off check, Amendment 3 version |

## L1 live runs (60 min)

Runs made with the version before Amendment 3:

| Folder | Device | Screen | Agreement | Within 100 ms | Max delay |
|---|---|---|---|---|---|
| `live_20261029_073341_B` | B | off | 100% | 99.97% | 117 ms |
| `live_20261029_074812_C` | C | on | 100% | 100% | – |
| `live_20261029_092656_B` | B | off | 100% | 99.99% | 102 ms |
| `live_20261029_092656_C` | C | off | suspended by the vendor build (see Amendment 3) | | |

The Device C screen-on run listed in Amendment 4 is `live_20261029_074812_C`, made with the version before Amendment 3. The Amendment 3 changes concern screen-off execution, run timing and battery logging.

Runs made with the Amendment 3 version, Devices A and B started together:

| Run | Folder | Device | Screen | Battery at start | Agreement | Max \|Δp\| | Within 100 ms | Median / max delay |
|---|---|---|---|---|---|---|---|---|
| 1 | `live_20261029_143235_A` | A | off | 90% | 100% | 5.4e-7 | 100% | 21.6 / 82.2 ms |
| 1 | `live_20261029_143235_B` | B | off | 71% | 100% | 1.1e-6 | 100% | 29.7 / 96.5 ms |
| 2 | `live_20261029_153600_A` | A | off | 91% | 100% | 4.3e-7 | 100% | 21.6 / 59.6 ms |
| 2 | `live_20261029_153602_B` | B | off | 71% | 100% | 1.1e-6 | 100% | 30.0 / 82.4 ms |
| 3 | `live_20261029_170417_A` | A | on | 90% | 100% | 7.2e-7 | 100% | 21.2 / 47.7 ms |
| 3 | `live_20261029_170420_B` | B | on | 75% | 100% | 9.2e-7 | 100% | 30.0 / 65.8 ms |

Each run gave 7179 decisions, with no sample gaps on Device A. All runs were unplugged, with airplane mode off. Motion was given at minutes 5, 15, 25, 35, 45 and 55, as confirmed by the logged accelerometer signal. Device A started Run 2 at 91% battery.

## EN1 energy runs (2 h, screen off, unplugged, airplane mode on)

| Folder | Device | Battery | Capacity used | Live agreement | Live − idle (mA) | Loop − idle (mA) | Energy per decision |
|---|---|---|---|---|---|---|---|
| `energy_20261029_183133_A` | A | 86% → 85% | 7040 mAh (typical; rated 6840 mAh) | 100% | −1.5 [−29.5, 26.6] | 66.9 [34.1, 99.7] | 1.20 mJ |
| `energy_20261029_183124_B` | B | 88% → 80% | 4355 mAh | 100% | 5.4 [−20.9, 31.7] | 363.3 [326.9, 399.7] | 0.93 mJ |

Device A current is analysed with `--current-unit ma`. Projected battery use of the controller over 8 h is 0.07% (A) and 0.08% (B).

## S1 streaming (30 min, laptop at the same Wi-Fi network)

| Folder | Device | Start | Packets | Sequence errors | Decisions | Agreement | Round trip median / p95 / p99 |
|---|---|---|---|---|---|---|---|
| `out_A_stream` | A | 22:11 | 45,000 | 0 | 3600 | 100% | 24.8 / 47.6 / 138.1 ms |
| `out_B_stream` | B | 23:17 | 45,000 | 0 | 3600 | 100% | 18.5 / 53.1 / 118.4 ms |

Both devices received the same input samples. On-phone processing median was 18.3 ms (A) and 9.4 ms (B). CPU duty cycle was 4.9% (A) and 2.9% (B).
