# fog-cue-deploy

Decision fidelity of a freezing-of-gait cue controller from training to an Android phone. The analysis plan is in `PLAN.md`.

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
| `fd_prepare.py` | raw Kaggle and FoG-STAR files to 100 Hz recordings in g |
| `fd_data.py` | reference batch features (SciPy), labels, windows |
| `fd_core.py` | independent streaming implementation (NumPy) |
| `fd_model.py` | PyTorch model, training, export |
| `fd_metrics.py` | event metrics and circular-shift surrogate |
| `fd_develop.py` | cross-validation, operating point, selection, final model |
| `fd_external.py` | FoG-STAR external test |
| `fd_precision.py` | streaming fidelity and precision variants |
| `fd_faults.py` | fault injection |
| `fd_export.py` | model, replay and stream files for the phone |
| `edge/` | C controller and command-line runner |
| `fd_phone.sh` | Termux runs |
| `fd_stream_sender.py` | laptop side of the Wi-Fi streaming run |
| `fd_summarize.py` | summary of Termux outputs |
| `fd_live.py` | live sensor and energy runs from the Android application |
