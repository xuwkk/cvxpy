"""
Copyright 2022, the CVXPY Authors

Licensed under the Apache License, Version 2.0 (the "License");

you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import scipy.sparse as sp

import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
from cvxpy.utilities.citations import CITATION_DICT

# for thread safety
import threading
_PROXQP_INIT_LOCK = threading.Lock()

def _dense_matrix_data_changed(new, old) -> bool:
    if sp.issparse(old):
        old = old.toarray()
    return not np.array_equal(new, old)


def _sparse_matrix_data_changed(new, old) -> bool:
    return (new.shape != old.shape
            or not np.array_equal(new.indptr, old.indptr)
            or not np.array_equal(new.indices, old.indices)
            or not np.array_equal(new.data, old.data))


class PROXQP(QpSolver):
    """QP interface for the PROXQP solver"""

    MIP_CAPABLE = False

    # Map of Proxqp status to CVXPY status.
    STATUS_MAP = {"PROXQP_SOLVED": s.OPTIMAL,
                  "PROXQP_MAX_ITER_REACHED": s.USER_LIMIT,
                  "PROXQP_PRIMAL_INFEASIBLE": s.INFEASIBLE,
                  "PROXQP_DUAL_INFEASIBLE": s.UNBOUNDED}

    VAR_MAP = {"P": "H",
               "q": "g",
               "A": "A",
               "b": "b",
               "F": "C",
               "l": "l",
               "lb": "l",
               "G": "u"}

    def name(self):
        return s.PROXQP

    def import_solver(self) -> None:
        import proxsuite
        proxsuite

    def invert(self, solution, inverse_data):
        attr = {s.SOLVE_TIME: solution.info.run_time}
        attr[s.EXTRA_STATS] = {"solution": solution}

        # Map PROXQP statuses back to CVXPY statuses
        status = self.STATUS_MAP.get(solution.info.status.name, s.SOLVER_ERROR)

        if status in s.SOLUTION_PRESENT:
            opt_val = solution.info.objValue + inverse_data[s.OFFSET]
            primal_vars = {
                PROXQP.VAR_ID:
                intf.DEFAULT_INTF.const_to_matrix(np.array(solution.x))
            }

            # Build dual vars dict keyed by constraint IDs
            # PROXQP returns solution.y (eq_duals) and solution.z (ineq_duals)
            eq_dual = utilities.get_dual_values(
                solution.y,
                utilities.extract_dual_value,
                inverse_data[self.EQ_CONSTR])
            ineq_dual = utilities.get_dual_values(
                solution.z,
                utilities.extract_dual_value,
                inverse_data[self.NEQ_CONSTR])
            dual_vars = {}
            dual_vars.update(eq_dual)
            dual_vars.update(ineq_dual)
            attr[s.NUM_ITERS] = solution.info.iter
            sol = Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            sol = failure_solution(status, attr)
        return sol

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts,
                       solver_cache=None):
        import proxsuite

        solver_opts = solver_opts.copy()
        custom_mode = False
        if "update" in data and "warm_start" in data:
            custom_mode = True
            update = data["update"]
            warm_start = data["warm_start"]
            if not isinstance(update, (bool, np.bool_)):
                raise TypeError("data['update'] must be a bool.")
            if not isinstance(warm_start, (bool, np.bool_)):
                raise TypeError("data['warm_start'] must be a bool.")
        elif "update" in data:
            raise ValueError(
                "warm_start is not found in data. Please set warm_start to True or False."
            )
        elif "warm_start" in data:
            raise ValueError(
                "update is not found in data. Please set update to True or False."
            )

        solver_opts['backend'] = solver_opts.get('backend', 'dense')
        backend = solver_opts['backend']

        if backend == "dense":
            # Convert sparse to dense matrices
            P = data[s.P].toarray()
            A = data[s.A].toarray()
            F = data[s.F].toarray()
        elif backend == "sparse":
            P = data[s.P]
            A = data[s.A]
            F = data[s.F]
        else:
            raise ValueError("Wrong input, backend most be either dense or sparse")

        q = data[s.Q]
        b = data[s.B]
        g = data[s.G]

        lb = -np.inf*np.ones(data[s.G].shape)
        data['lb'] = lb

        n_var = data['n_var']
        n_eq = data['n_eq']
        n_ineq = data['n_ineq']

        # Overwrite default values
        solver_opts['eps_abs'] = solver_opts.get('eps_abs', 1e-8)
        solver_opts['eps_rel'] = solver_opts.get('eps_rel', 0.)
        solver_opts['max_iter'] = solver_opts.get('max_iter', 10000)
        solver_opts['rho'] = solver_opts.get('rho', 1e-6)
        solver_opts['mu_eq'] = solver_opts.get('mu_eq', 1e-3)
        solver_opts['mu_in'] = solver_opts.get('mu_in', 1e-1)
        # ProxQP only fills timing fields when compute_timings is enabled.
        compute_timings = solver_opts.get('compute_timings', True)

        def apply_solver_settings(_solver, pre_init: bool = False) -> None:
            _solver.settings.compute_timings = compute_timings
            if pre_init:
                return
            _solver.settings.eps_abs = solver_opts['eps_abs']
            _solver.settings.eps_rel = solver_opts['eps_rel']
            _solver.settings.max_iter = solver_opts['max_iter']
            _solver.settings.verbose = verbose

        if custom_mode:
            # Self-implemented warm start and update controls.
            def matrix_data_changed(new, old) -> bool:
                if backend == "dense":
                    return _dense_matrix_data_changed(new, old)
                return _sparse_matrix_data_changed(new, old)

            if update:
                if solver_cache is None or self.name() not in solver_cache:
                    raise ValueError(
                        "Solver cache is not found. Solve once before using data['update']=True."
                    )
                solver, old_data, _ = solver_cache[self.name()]
                new_args = {}
                for key in ['q', 'b', 'G', 'lb']:
                    if not np.array_equal(data[key], old_data[key]):
                        new_args[self.VAR_MAP[key]] = data[key]
                if matrix_data_changed(P, old_data[s.P]):
                    new_args['H'] = P
                if matrix_data_changed(A, old_data[s.A]):
                    new_args['A'] = A
                if matrix_data_changed(F, old_data[s.F]):
                    new_args['C'] = F
                apply_solver_settings(solver)
                if new_args:
                    solver.update(**new_args)
            else:
                if backend == "dense":
                    solver = proxsuite.proxqp.dense.QP(n_var, n_eq, n_ineq)
                elif backend == "sparse":
                    solver = proxsuite.proxqp.sparse.QP(n_var, n_eq, n_ineq)
                apply_solver_settings(solver, pre_init=True)

                with _PROXQP_INIT_LOCK:
                    solver.init(H=P,
                                g=q,
                                A=A,
                                b=b,
                                C=F,
                                l=lb,
                                u=g,
                                rho=solver_opts['rho'],
                                mu_eq=solver_opts['mu_eq'],
                                mu_in=solver_opts['mu_in'])

                apply_solver_settings(solver)

            if warm_start:
                ws_dict = data.get("warm_start_solution_dict")
                if not isinstance(ws_dict, dict) or len(ws_dict) == 0:
                    raise ValueError(
                        "data['warm_start_solution_dict'] must be a non-empty dict when "
                        "data['warm_start']=True."
                    )
                missing = {"x", "y", "z"} - set(ws_dict.keys())
                if missing:
                    raise ValueError(
                        "data['warm_start_solution_dict'] is missing required keys: "
                        f"{sorted(missing)}."
                    )
                x_ws = np.asarray(ws_dict["x"]).reshape(-1)
                y_ws = np.asarray(ws_dict["y"]).reshape(-1)
                z_ws = np.asarray(ws_dict["z"]).reshape(-1)
                if x_ws.size != n_var:
                    raise ValueError(
                        "Invalid warm-start shape for 'x': expected length "
                        f"{n_var}, got {x_ws.size}."
                    )
                if y_ws.size != n_eq:
                    raise ValueError(
                        "Invalid warm-start shape for 'y': expected length "
                        f"{n_eq}, got {y_ws.size}."
                    )
                if z_ws.size != n_ineq:
                    raise ValueError(
                        "Invalid warm-start shape for 'z': expected length "
                        f"{n_ineq}, got {z_ws.size}."
                    )
                if (not np.all(np.isfinite(x_ws))
                        or not np.all(np.isfinite(y_ws))
                        or not np.all(np.isfinite(z_ws))):
                    raise ValueError("Warm-start values for 'x', 'y', and 'z' must be finite.")
                solver.solve(x_ws, y_ws, z_ws)
            else:
                solver.solve(np.zeros(n_var), np.zeros(n_eq), np.zeros(n_ineq))
        else:
            # Original CVXPY implementation.
            if warm_start and solver_cache is not None and self.name() in solver_cache:
                solver, old_data, results = solver_cache[self.name()]
                new_args = {}
                for key in ['q', 'b', 'G', 'lb']:
                    if any(data[key] != old_data[key]):
                        new_args[self.VAR_MAP[key]] = data[key]
                if P.data.shape != old_data[s.P].data.shape or any(
                        P.data != old_data[s.P].data):
                    new_args['H'] = P
                if A.data.shape != old_data[s.A].data.shape or any(
                        A.data != old_data[s.A].data):
                    new_args['A'] = A
                if F.data.shape != old_data[s.F].data.shape or any(
                        F.data != old_data[s.F].data):
                    new_args['C'] = F

                status = self.STATUS_MAP.get(results.info.status.name, s.SOLVER_ERROR)
                if status == s.OPTIMAL:
                    x_warm_start = results.x.copy()
                    y_warm_start = results.y.copy()
                    z_warm_start = results.z.copy()

                if new_args:
                    solver.update(**new_args)

                apply_solver_settings(solver)

                if status == s.OPTIMAL:
                    solver.solve(x_warm_start, y_warm_start, z_warm_start)
                else:
                    solver.solve()
            else:
                if backend == "dense":
                    solver = proxsuite.proxqp.dense.QP(n_var, n_eq, n_ineq)
                elif backend == "sparse":
                    solver = proxsuite.proxqp.sparse.QP(n_var, n_eq, n_ineq)

                apply_solver_settings(solver, pre_init=True)

                solver.init(H=P,
                            g=q,
                            A=A,
                            b=b,
                            C=F,
                            l=lb,
                            u=g,
                            rho=solver_opts['rho'],
                            mu_eq=solver_opts['mu_eq'],
                            mu_in=solver_opts['mu_in'])

                apply_solver_settings(solver)

                solver.solve()

        results = solver.results

        if solver_cache is not None:
            solver_cache[self.name()] = (solver, data, results)

        return results

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["PROXQP"]
