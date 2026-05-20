# `test_new.py` Solve Results

Note: pre-solve results on the unperturbed problem are excluded from warm-start/update case rows.

## Solver: `osqp`

| Case | Objective value | Total time | Timing details |
|---|---:|---:|---|
| Basic: Original CVXPY, No warm start | 471075.373 | 2.453 | Canonical: 0.146, setup: 0.374, update: 0.000, solve: 1.933, polish: 0.000, run: 2.307, iter: 1200 |
| New: Original CVXPY, No warm start | 470258.989 | 2.436 | Canonical: 0.133, setup: 0.372, update: 0.000, solve: 1.931, polish: 0.000, run: 2.303, iter: 1200 |
| New: Original CVXPY, Warm start | 470258.861 | 1.238 | Canonical: 0.018, setup: 0.374, update: 2e-05, solve: 1.220, polish: 0.000, run: 1.220, iter: 750 |
| New: New CVXPY, No warm start, No update | 470258.989 | 2.426 | Canonical: 0.131, setup: 0.374, update: 0.000, solve: 1.921, polish: 0.000, run: 2.295, iter: 1200 |
| New: New CVXPY, Warm start, No update | 470258.861 | 1.606 | Canonical: 0.018, setup: 0.377, update: 0.000, solve: 1.211, polish: 0.000, run: 1.588, iter: 750 |
| New: New CVXPY, No warm start, Update | 470258.989 | 1.947 | Canonical: 0.018, setup: 0.372, update: 2e-05, solve: 1.930, polish: 0.000, run: 1.930, iter: 1200 |
| New: New CVXPY, Warm start, Update | 470258.861 | 1.223 | Canonical: 0.018, setup: 0.373, update: 2e-05, solve: 1.205, polish: 0.000, run: 1.205, iter: 750 |

## Solver: `scs`

| Case | Objective value | Total time | Timing details |
|---|---:|---:|---|
| Basic: Original CVXPY, No warm start | 471084.710 | 2.398 | Canonical: 0.170, setup: 0.388, solve: 1.841, cone: 0.003, accel: 0.002, linear system: 1.458, iter: 900 |
| New: Original CVXPY, No warm start | 470267.562 | 2.013 | Canonical: 0.161, setup: 0.393, solve: 1.459, cone: 0.002, accel: 0.001, linear system: 1.090, iter: 675 |
| New: Original CVXPY, Warm start | 470267.079 | 1.492 | Canonical: 0.007, setup: 0.395, solve: 1.089, cone: 0.001, accel: 0.001, linear system: 0.730, iter: 450 |
| New: New CVXPY, No warm start, No update | 470267.562 | 2.022 | Canonical: 0.161, setup: 0.388, solve: 1.472, cone: 0.002, accel: 0.001, linear system: 1.100, iter: 675 |
| New: New CVXPY, Warm start, No update | 470267.079 | 1.485 | Canonical: 0.014, setup: 0.387, solve: 1.084, cone: 0.001, accel: 0.001, linear system: 0.729, iter: 450 |
| New: New CVXPY, No warm start, Update | 470267.029 | 0.814 | Canonical: 0.007, setup: 1e-05, solve: 0.806, cone: 0.001, accel: 0.001, linear system: 0.775, iter: 475 |
| New: New CVXPY, Warm start, Update | 470267.088 | 0.819 | Canonical: 0.013, setup: 1e-05, solve: 0.806, cone: 0.001, accel: 0.001, linear system: 0.775, iter: 475 |

## Solver: `qpalm`

| Case | Objective value | Total time | Timing details |
|---|---:|---:|---|
| Basic: Original CVXPY, No warm start | 471080.785 | 6.434 | Canonical: 0.145, setup: 0.070, solve: 6.219, run: 6.289, iter: 108 |
| New: Original CVXPY, No warm start | 470267.219 | 6.888 | Canonical: 0.135, setup: 0.071, solve: 6.682, run: 6.753, iter: 111 |
| New: Original CVXPY, Warm start | 470268.777 | 0.924 | Canonical: 0.029, setup: 0.069, solve: 0.825, run: 0.895, iter: 57 |
| New: New CVXPY, No warm start, No update | 470267.219 | 6.787 | Canonical: 0.123, setup: 0.071, solve: 6.593, run: 6.665, iter: 111 |
| New: New CVXPY, Warm start, No update | 470268.777 | 0.920 | Canonical: 0.032, setup: 0.068, solve: 0.820, run: 0.888, iter: 57 |
| New: New CVXPY, No warm start, Update | 470267.219 | 6.702 | Canonical: 0.030, setup: 0.067, solve: 6.606, run: 6.672, iter: 111 |
| New: New CVXPY, Warm start, Update | 470268.777 | 0.919 | Canonical: 0.031, setup: 0.066, solve: 0.822, run: 0.888, iter: 57 |

## Solver: `proxqp`

| Case | Objective value | Total time | Timing details |
|---|---:|---:|---|
| Basic: Original CVXPY, No warm start | 471084.645 | 1.428 | Canonical: 0.146, setup: 0.050, solve: 1.232, run: 1.282, iter: 62 |
| New: Original CVXPY, No warm start | 470267.268 | 1.399 | Canonical: 0.133, setup: 0.050, solve: 1.216, run: 1.266, iter: 61 |
| New: Original CVXPY, Warm start | 470267.268 | 0.475 | Canonical: 0.011, setup: 0.019, solve: 0.445, run: 0.464, iter: 21 |
| New: New CVXPY, No warm start, No update | 470267.268 | 1.456 | Canonical: 0.116, setup: 0.045, solve: 1.294, run: 1.339, iter: 64 |
| New: New CVXPY, Warm start, No update | 470267.268 | 0.533 | Canonical: 0.011, setup: 0.055, solve: 0.467, run: 0.522, iter: 21 |
| New: New CVXPY, No warm start, Update | 470267.268 | 1.281 | Canonical: 0.012, setup: 0.020, solve: 1.249, run: 1.269, iter: 64 |
| New: New CVXPY, Warm start, Update | 470267.268 | 0.500 | Canonical: 0.018, setup: 0.025, solve: 0.457, run: 0.482, iter: 21 |
