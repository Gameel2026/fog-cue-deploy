# fog-cue-deploy

Decision fidelity of a freezing-of-gait cue controller from training to Android devices.

The analysis plan is in `PLAN.md`. Amendments are in `AMENDMENT_1.md` to `AMENDMENT_4.md`, each committed before the analysis it affects. Device labels follow Amendment 4. All device runs are listed in `phone_results/RUNS.md`.

## Requirements

Python 3.10+ with the packages in `requirements.txt`. A C compiler (clang in Termux) for `edge/`. Android Studio for the application.

## How to run

```
python fd_prepare.py
python tools/inspect_episodes.py > data_review.txt
python tools/compare_datasets.py >> data_review.txt

python fd_develop.py --quick
python fd_external.py --quick
python fd_precision.py --quick --datasets FoG-STAR --max-stretches 5
python fd_faults.py --quick --datasets FoG-STAR
python fd_export.py --quick

python fd_develop.py
python fd_external.py
python fd_precision.py
python fd_faults.py
python fd_export.py
```

Paths to the raw data are set in `fd_config.py`. `--quick` writes to `results_quick/` and is only a code check.

## Files

| File | Role |
|---|---|
| `fd_prepare.py` | raw tDCS-FoG, DeFOG, Daphnet, Multimodal and FoG-STAR files to 100 Hz recordings in g |
| `fd_data.py` | reference batch features (SciPy), labels, windows |
| `fd_core.py` | independent streaming implementation (NumPy) |
| `fd_model.py` | PyTorch model, training, export |
| `fd_metrics.py` | event metrics and circular-shift surrogate |
| `fd_develop.py` | cross-validation, operating point, selection, final model |
| `fd_external.py` | external tests on Daphnet, Multimodal and FoG-STAR |
| `fd_precision.py` | streaming fidelity and precision variants |
| `fd_faults.py` | fault injection |
| `fd_export.py` | model, replay and stream files for the phone |
| `edge/` | C controller and command-line runner |
| `fd_phone.sh` | Termux runs |
| `fd_stream_sender.py` | laptop side of the Wi-Fi streaming run |
| `fd_summarize.py` | summary of Termux outputs |
| `fd_live.py` | live sensor and energy runs from the Android application (`--capacity`, `--current-unit ua\|ma`) |
| `android/` | Android application source (Java and JNI) |
| `results/` | frozen model record, cross-validation, external, precision, fault and fidelity results |
| `results/edge/` | model, replay and hash files used on the devices |
| `phone_results/` | outputs of all device runs; raw sample logs are gzip-compressed |
| `figures/` | coverage and false-alarm trade-off of the selected candidate |

## Data

All datasets are public: tDCS-FoG and DeFOG (Zenodo, doi:10.5281/zenodo.12565292), Daphnet (UCI Machine Learning Repository), the multimodal FoG dataset of Zhang et al. (Scientific Data 2022) and FoG-STAR (Zenodo, doi:10.5281/zenodo.17838806). The large stream files `results/edge/streams_*.bin` and the cross-validation fold files are not stored here and are produced by the scripts above.

## Licence

MIT, see `LICENSE`.
