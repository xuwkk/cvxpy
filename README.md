CVXPY
=====================

## My Modification

This is my mofification of CVXPY to better control the warm start and update behaviour of the OSQP and scs solver. More solvers may be supported in the future.

In original CVXPY, the warm start and update functions are controlled by the single `warm_start` flag and the same problem instance must be used for repeated solves. This setting lacks flexibility:
1. When repeatedly solving large number of problems of similar structure, it is memeory-inefficient to store the problem instances for all samples; instead, we would expect to only instantiate smaller amount of problem instances and reuse them for repeated solves (like batch-by-batch im deep learning training). And,
2. The warm start and update cannot be used separately. And CVXPY does not support warm start from user-defined value but only from the previous solution.

### OSQP

The original CVXPY supports warm start from the previous solve and update, both from the internal cached data. I modify [osqp_qpif](cvxpy/reductions/solvers/qp_solvers/osqp_qpif.py). It supports lower-level control by passing extra data to the solver. An example is stored in `test_osqp.py`.

The table below summarizes the new options for controlling OSQP warm start and update behaviour in this modified CVXPY:

| Option Name | Where to Set (in `data` dict) | Example Value / Usage |
|-------------|---------------------------------|---------|
| update    | `data['update']`                | `True` or `False` |
| warm start | `data['warm_start']`            | `True` or `False` |
| warm start values | `data['warm_start_solution_dict']`  | `{'x': ndarray, 'y': ndarray}` |

### SCS

The original CVXPY only supports warm start from the previous solve, by the internal cached data and no update is supported. I modify [scs_conif](cvxpy/reductions/solvers/conic_solvers/scs_conif.py).

The table below summarizes the new options for controlling SCS warm start and update behaviour in this modified CVXPY:
| Option Name | Where to Set (in `data` dict) | Example Value / Usage |
|-------------|---------------------------------|---------|
| update    | `data['update']`                | `True` or `False` |
| warm start | `data['warm_start']`            | `True` or `False` |
| warm start values | `data['warm_start_solution_dict']`  | `{'x': ndarray, 'y': ndarray, 's': ndarray}` |


> NOTE: when using this feature, you need to simultaneously include `update` and `warm_start` in the `data` dict.

You may refer to the following minimal example usage (see `text_osqp.py` for more detail):

As the solver's warm start requires initialization for the canornical (standard) form, the warm start function is implemented at low-level of CVXPY inference through compiling `get_problem_data` and the low-level solve `solve_via_data` function.

```python
# prob is the predefined problem instance

# Compile
data, chain, inverse_data = prob.get_problem_data(solver = cp.OSQP)

# Set new options
data['update'] = True
data['warm_start'] = True
data['warm_start_solution_dict'] = {'x': previous_x, 'y': previous_y}

results = chain.solve_via_data(problem=prob, data=data, warm_start=False/True, verbose=False, solver_opts={'polish': False})
```

> NOTE: The original `warm_start` flag is ignored.

See `test_new.py` for concrete usage patterns.

## How to install

```bash
pip install git+https://github.com/xuwkk/cvxpy.git
```
or
```bash
pip install git+https://github.com/xuwkk/cvxpy.git@v1.9.0-lapso.1
```

For development, clone and isntall in development mode:
```bash
git clone https://github.com/xuwkk/cvxpy.git
pip install -e .
```

After installation, use CVXPY as usual:
```python
import cvxpy as cp
```

> NOTE: You should uninstall the original CVXPY installation before installing this modified version.

Below is the original README.md from the CVXPY repository.


------------------------------------------------------------------------------------------------


[![Build Status](https://github.com/cvxpy/cvxpy/actions/workflows/build.yml/badge.svg?event=push)](https://github.com/cvxpy/cvxpy/actions/workflows/build.yml)
![PyPI - downloads](https://img.shields.io/pypi/dm/cvxpy.svg?label=Pypi%20downloads)
![Conda - downloads](https://img.shields.io/conda/dn/conda-forge/cvxpy.svg?label=Conda%20downloads)
[![Discord](https://img.shields.io/badge/Chat-Discord-Blue?color=5865f2)](https://discord.gg/4urRQeGBCr)
[![Benchmarks](http://img.shields.io/badge/benchmarked%20by-asv-blue.svg?style=flat)](https://cvxpy.github.io/benchmarks/)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/cvxpy/cvxpy/badge)](https://api.securityscorecards.dev/projects/github.com/cvxpy/cvxpy)

**The CVXPY documentation is at [cvxpy.org](https://www.cvxpy.org/).**

*We are building a CVXPY community on [Discord](https://discord.gg/4urRQeGBCr). Join the conversation! For issues and long-form discussions, use [Github Issues](https://github.com/cvxpy/cvxpy/issues) and [Github Discussions](https://github.com/cvxpy/cvxpy/discussions).*

**Contents**
- [Installation](#installation)
- [Getting started](#getting-started)
- [Issues](#issues)
- [Community](#community)
- [Contributing](#contributing)
- [Team](#team)
- [Citing](#citing)


CVXPY is a Python-embedded modeling language for convex optimization problems. It allows you to express your problem in a natural way that follows the math, rather than in the restrictive standard form required by solvers.

For example, the following code solves a least-squares problem where the variable is constrained by lower and upper bounds:

```python3
import cvxpy as cp
import numpy

# Problem data.
m = 30
n = 20
numpy.random.seed(1)
A = numpy.random.randn(m, n)
b = numpy.random.randn(m)

# Construct the problem.
x = cp.Variable(n)
objective = cp.Minimize(cp.sum_squares(A @ x - b))
constraints = [0 <= x, x <= 1]
prob = cp.Problem(objective, constraints)

# The optimal objective is returned by prob.solve().
result = prob.solve()
# The optimal value for x is stored in x.value.
print(x.value)
# The optimal Lagrange multiplier for a constraint
# is stored in constraint.dual_value.
print(constraints[0].dual_value)
```

With CVXPY, you can model
* convex optimization problems,
* mixed-integer convex optimization problems,
* geometric programs, and
* quasiconvex programs.

CVXPY is not a solver. It relies upon the open source solvers 
[Clarabel](https://github.com/oxfordcontrol/Clarabel.rs), [SCS](https://github.com/bodono/scs-python),
[OSQP](https://github.com/oxfordcontrol/osqp) and [HiGHS](https://github.com/ERGO-Code/HiGHS).
Additional solvers are [available](https://www.cvxpy.org/tutorial/solvers/index.html#choosing-a-solver),
but must be installed separately.

CVXPY began as a Stanford University research project. It is now developed by
many people, across many institutions and countries.


## Installation
CVXPY is available on PyPI, and can be installed with
```
pip install cvxpy
```

CVXPY can also be installed with conda, using
```
conda install -c conda-forge cvxpy
```

CVXPY has the following dependencies:

- Python >= 3.11
- Clarabel >= 0.5.0
- OSQP >= 1.0.0
- SCS >= 3.2.4.post1
- NumPy >= 2.0.0
- SciPy >= 1.13.0
- highspy >= 1.11.0

For detailed instructions, see the [installation
guide](https://www.cvxpy.org/install/index.html).

## Getting started
To get started with CVXPY, check out the following:
* [official CVXPY tutorial](https://www.cvxpy.org/tutorial/index.html)
* [example library](https://www.cvxpy.org/examples/index.html)
* [API reference](https://www.cvxpy.org/api_reference/cvxpy.html)

## Issues
We encourage you to report issues using the [Github tracker](https://github.com/cvxpy/cvxpy/issues). We welcome all kinds of issues, especially those related to correctness, documentation, performance, and feature requests.

For basic usage questions (e.g., "Why isn't my problem DCP?"), please use [StackOverflow](https://stackoverflow.com/questions/tagged/cvxpy) instead.

## Community
The CVXPY community consists of researchers, data scientists, software engineers, and students from all over the world. We welcome you to join us!

* To chat with the CVXPY community in real-time, join us on [Discord](https://discord.gg/4urRQeGBCr).
* To have longer, in-depth discussions with the CVXPY community, use [Github Discussions](https://github.com/cvxpy/cvxpy/discussions).
* To share feature requests and bug reports, use [Github Issues](https://github.com/cvxpy/cvxpy/issues).

Please be respectful in your communications with the CVXPY community, and make sure to abide by our [code of conduct](https://github.com/cvxpy/cvxpy/blob/master/CODE_OF_CONDUCT.md).

## Contributing
We appreciate all contributions. You don't need to be an expert in convex
optimization to help out.

You should first
install [CVXPY from source](https://www.cvxpy.org/install/index.html#install-from-source).
Here are some simple ways to start contributing immediately:
* Read the CVXPY source code and improve the documentation, or address TODOs
* Enhance the [website documentation](https://github.com/cvxpy/cvxpy/tree/master/doc)
* Browse the [issue tracker](https://github.com/cvxpy/cvxpy/issues), and look for issues tagged as "help wanted"
* Polish the [example library](https://github.com/cvxpy/examples)
* Add a [benchmark](https://github.com/cvxpy/benchmarks)

If you'd like to add a new example to our library, or implement a new feature,
please get in touch with us first to make sure that your priorities align with
ours. 

Contributions should be submitted as [pull requests](https://github.com/cvxpy/cvxpy/pulls).
A member of the CVXPY development team will review the pull request and guide
you through the contributing process.

Before starting work on your contribution, please read the [contributing guide](https://github.com/cvxpy/cvxpy/blob/master/CONTRIBUTING.md).

## Team
CVXPY is a community project, built from the contributions of many
researchers and engineers.

CVXPY is developed and maintained by [Steven
Diamond](https://stevendiamond.me/), [Akshay
Agrawal](https://akshayagrawal.com), [Riley Murray](https://rileyjmurray.wordpress.com/), 
[Philipp Schiele](https://www.philippschiele.com/),
[Bartolomeo Stellato](https://stellato.io/),
and [Parth Nobel](https://ptnobel.github.io), with many others contributing
significantly.
A non-exhaustive list of people who have shaped CVXPY over the
years includes Stephen Boyd, Eric Chu, Robin Verschueren,
Jaehyun Park, Enzo Busseti, AJ Friend, Judson Wilson, Chris Dembia, and
William Zhang.

For more information about the team and our processes, see our [governance document](https://github.com/cvxpy/org/blob/main/governance.md).

## Citing
If you use CVXPY for academic work, we encourage you to [cite our papers](https://www.cvxpy.org/resources/citing/index.html). If you use CVXPY in industry, we'd love to hear from you as well, on Discord or over email.
