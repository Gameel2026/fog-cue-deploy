# Amendment 3: battery logging for the energy runs

Committed before the energy runs.

In the first live runs, the current reported by the Android battery service had different units and signs on the two phones. To check the energy estimate independently of these conventions, the application now also logs the battery charge counter (µAh), the charger state and the battery status every second (`LiveService.java`). The energy analysis (`fd_live.py`) reports, next to the current-based estimate, the current derived from the slope of the charge counter in each block, and the fraction of each block with a charger connected. Live-run processing, the model, thresholds, experiments and criteria are unchanged.
