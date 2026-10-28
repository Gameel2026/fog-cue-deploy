# Analysis plan: decision fidelity of a freezing-of-gait cue controller from training to a smartphone

This plan is committed together with the code and the data review (`data_review.txt`) before any model is trained. A short development run (`--quick`, written to `results_quick/`) checks that the code runs on the full data; its numbers are not reported.

## 1. Aim

To test whether the cue decisions of a trained freezing-of-gait (FoG) cue controller are preserved along the full deployment chain: reference training pipeline, independent streaming implementation, C implementation, and an Android application that reads the phone's own accelerometer in real time. The model is a test case for the chain. Its detection and prediction performance are reported on development and external data.

## 2. Data

- Development (107 patients): Kaggle tDCS-FoG (lab, 62 patients, 128 Hz, m/s²) and DeFOG (home, 45 patients including the `notype` recordings, 100 Hz, g), lower-back accelerometer (AccV, AccML, AccAP). DeFOG samples outside `Valid` and `Task` are excluded. FoG = any of StartHesitation, Turn, Walking or Event.
- External tests, never used for training, selection or thresholds:
  - Daphnet trunk (lower-back) accelerometer, 10 patients, 64 Hz, mg. Samples labelled 0 are excluded.
  - Multimodal FoG dataset, lower-back (L5) accelerometer, the patients who wore it (6 of 12), filtered data at 500 Hz from 100 Hz sensors, counts (8192 per g). Task files without a lower-back sensor are skipped.
  - FoG-STAR lower-back accelerometer, 22 patients, 60 Hz, g. Episodes separated by less than 2 s are merged, as the 10 fps video labels fragment episodes.
- Preparation (`fd_prepare.py`): all signals in g, axes ordered vertical, mediolateral, anteroposterior, at 100 Hz (DeFOG native, others by polyphase resampling). Non-finite samples split a recording into separate parts and are excluded; the number dropped is reported.
- Axis mapping from gravity and gait variance, fixed from inspection before training: FoG-STAR `back_acc_x/y/z` → V/ML/AP; Daphnet `-tr_y`, `tr_z`, `tr_x` → V/ML/AP; Multimodal `-x`, `y`, `z` of the waist sensor → V/ML/AP.
- Data review before training (`tools/inspect_episodes.py`, `tools/compare_datasets.py`, saved in `data_review.txt`): episode counts, durations, clean lead time before onset, gravity per axis, share of still windows in non-FoG, pre-FoG and FoG windows, and model-free separability of FoG and pre-FoG windows. This review describes the data and is not used to change the model after training.

## 3. Sampling rate and phones

The rate is 100 Hz. A sensor check on the two phones, before this plan, gave 99.35 Hz on Device A (Android 16, vendor build) and 121.56 Hz on Device B (Android 17, stock build) when a 10 ms period was requested, stable with the screen on and off. Other requested rates gave rates that changed with screen state, or batched delivery with about 10 s latency at 200 Hz with the screen off. Both phones request a 10 ms period, and on both the samples are linearly interpolated onto a 100 Hz grid using sensor timestamps.

## 4. Controller

- Causal 4th-order Butterworth band-pass 0.5–20 Hz (second-order sections, state initialised from the first sample).
- Non-overlapping 0.5 s windows (50 samples). A window is FoG when at least 30 of its samples are FoG.
- 72 features per window: 14 statistical and 10 spectral features per axis.
- Two-layer LSTM (64 units) over the last n windows, softmax output, then hysteresis (on at θon, off below θoff).
- Cue target: the 8 windows before FoG onset, FoG itself, and the 6 windows after FoG.

## 5. Development, thresholds and selection

- Six candidates: context 8, 16 or 32 windows × cue-class weight 1 or 2.
- Five-fold cross-validation grouped by patient over the 107 development patients. Within each fold, 20% of training patients are held out for early stopping (balanced accuracy, patience 3, at most 12 epochs). Standardisation uses the fitting patients only.
- One model, two operating points from the out-of-fold probabilities: θhome from DeFOG patients and θlab from tDCS-FoG patients. Each is the point with the highest episode coverage with false alarms ≤ 15 per hour and cue specificity ≥ 0.70, ties broken by timely activation and then fewer false alarms. If no point meets both limits, the point with fewest false alarms among those with specificity ≥ 0.70 is used.
- Candidate selection: the candidate whose lower coverage of the two contexts is highest, ties broken by mean timely activation and then mean false alarms.
- The final model is trained on all development patients for the median best epoch of its folds and frozen with both thresholds (`results/frozen.json` with file hashes). θhome is used for DeFOG and on the phones; θlab is used for tDCS-FoG and the three external tests. Results with the other threshold are also reported.

## 6. Metrics

False alarms per hour, episode coverage, timely activation (cue on at onset and started no more than 8 windows before), cue sensitivity and specificity, window AUC. Coverage and timely activation are compared with a circular-shift surrogate (2,000 shifts per recording); one-sided p < 0.05 is taken as above chance. Results are given per dataset and per patient, and for FoG-STAR also by FoG type (shuffling, trembling, akinesia).

## 7. Implementations

1. Reference: batch features with SciPy (`fd_data.py`) and the PyTorch model.
2. Streaming: independent NumPy implementation processing one window at a time (`fd_core.py`).
3. C: `edge/fd_edge.c`, used unchanged in Termux and inside the Android application through JNI.
4. Android application: foreground service with a partial wake lock, accelerometer at a 10 ms period, interpolation onto 100 Hz, logging of raw samples, decisions and battery readings, so each live run can be replayed offline.

## 8. Experiments and criteria

| ID | Test | Criterion |
|---|---|---|
| E1 | Streaming implementation vs reference, all five datasets | decision agreement 100% in every patient; max \|Δp\| ≤ 1e-5 |
| E2 | C on each phone vs streaming implementation, DeFOG and the three external datasets, with their thresholds | decision agreement 100% in every patient; max \|Δp\| ≤ 1e-5 |
| C1 | Compute time on each phone (bench, 5 passes) | median and 99th percentile processing below 50 ms per decision |
| L1 | Live sensor runs per device: two 60-min runs with the screen off and one with the screen on, phone on a desk, moved by hand for 1 min at minutes 5, 15, 25, 35, 45 and 55 | live vs offline replay of the logged samples: agreement 100%, max \|Δp\| ≤ 1e-5; ≥ 99% of decisions within 100 ms of the last sample of their window |
| S1 | Streaming over Wi-Fi from a laptop, 30 min per device | no sequence errors; decisions identical to offline replay of the received samples; round-trip time reported |
| P1 | Weight precision variants (fp16, int8, int6, int4, int8 weights with int8 activations) | decision agreement with fp32 and change in event metrics; ≥ 99% agreement is taken as acceptable for deployment |
| F1 | Fault injection: filter reset per window, zero initial filter state, zero-phase filter, input in m/s², swapped axes, Device B rate without resampling, Device B rate with resampling, float32 features, context reset per window, threshold without hysteresis, int4 weights | decision agreement and change in event metrics for each fault |
| EN1 | Energy per device: 2-h protocol of 10-min blocks (idle, live, idle, replay loop) × 3, screen off, unplugged, airplane mode | live minus idle current and energy per decision with 95% confidence intervals; projected battery use over 8 h |
| D1 | Detection: cross-validation (tDCS-FoG, DeFOG) and the three external tests | episode coverage above the surrogate reference and false alarms per hour at the frozen operating point |
| D2 | Prediction (cue on before onset): same datasets | timely activation against the surrogate reference, with the number of eligible episodes |

Delays above 500 ms, if any, are reported with their count and timing. Monotonic and boot-time clocks are compared to separate processing delays from system suspension.

## 9. Reporting and corrections

All runs listed above are reported. Devices are named by Android version and build type. Corrections needed for the code to run on a device or on the full data, and corrections of data errors found during the analysis (for example corrupted or missing samples), are documented and committed before the affected analyses are run again, and all affected results are re-run with the correction. Further amendments may be added; each is committed before its own analysis.
