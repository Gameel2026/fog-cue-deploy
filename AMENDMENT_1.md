# Amendment 1: efficiency of the C implementation

Committed before any device run of the final model.

A two-minute live test of the application on both phones, using the development-check model, gave decision agreement of 100% with the offline replay. Processing took about 21 ms per decision on Device A and 47 ms on Device B with a 32-window context. To keep a margin for screen-off runs, the C implementation in `edge/fd_edge.c` was made faster without changing its arithmetic:

- weight matrices are stored transposed so that the gate loops can be vectorised, while each gate is still summed in the same order;
- the first-layer input projection of each window is computed once and reused while the window stays in the context.

On the replay file the new version gave probabilities identical to the previous version, and decisions identical to the Python streaming implementation on all prepared streams. The model, thresholds, experiments and criteria are unchanged.
