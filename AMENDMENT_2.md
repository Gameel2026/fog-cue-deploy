# Amendment 2: floating-point contraction on the phones

Committed before the live, streaming and energy runs, and before E2 is run again.

The first E2 run on both phones gave decision agreement of 100% in every dataset and patient. The probability difference was at most 7.2e-7 in Daphnet, Multimodal and FoG-STAR, but reached 1.2e-3 in 42 of 148,858 DeFOG windows, in two recordings. Both phones gave identical probabilities.

The cause is fused multiply-add contraction by the ARM compiler. It changes rounding in the skewness of the first window of a recording, where the filtered signal has almost no variance, and the difference then persists through the 32-window context. The same difference was reproduced on a PC by compiling with contraction enabled, and it disappears when contraction is disabled.

Correction: the C code is compiled with `-ffp-contract=off` in Termux (`fd_phone.sh`) and in the Android application (`android/app/src/main/cpp/CMakeLists.txt`). E2 is run again with this build, and both E2 runs are reported. The Android application source is added under `android/`. The model, thresholds, experiments and criteria are unchanged.
