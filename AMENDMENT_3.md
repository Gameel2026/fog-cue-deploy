# Amendment 3: background execution and battery logging

Committed before the live and energy runs are repeated.

In a screen-off live run on Device A (60 min requested), the vendor power management suspended the application although it ran as a foreground service with a partial wake lock and unrestricted background activity. Battery logging stopped for up to 590 s at a time, about 60% of the expected accelerometer samples were never delivered, and samples that were delivered arrived up to 590 s late. Because the run length was timed with the system uptime clock, which stops during suspension, the run lasted longer than 60 min. The same version ran without suspension on Device B. The current reported by the Android battery service also had different units and signs on the two phones.

Corrections:

- the application requests the wake-up variant of the accelerometer when the phone has one, so that sensor events wake the processor; the sensor name and type are logged;
- run length, motion prompts and energy blocks are timed with the elapsed real-time clock;
- on Device A, the vendor setting "Sleep standby optimization" is turned off, in addition to unrestricted background activity;
- the battery charge counter, charger state and battery status are logged every second, and the energy analysis (`fd_live.py`) reports the current derived from the charge counter next to the current-based estimate.

All live runs listed in the plan are repeated on both devices with this version. The earlier runs are kept in the repository. The model, thresholds, experiments and criteria are unchanged.
