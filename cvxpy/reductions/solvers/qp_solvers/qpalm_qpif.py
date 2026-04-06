"""
Copyright, the CVXPY authors

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


def _sparse_matrix_data_changed(new, old) -> bool:
    return (new.shape != old.shape
            or not np.array_equal(new.indptr, old.indptr)
            or not np.array_equal(new.indices, old.indices)
            or not np.array_equal(new.data, old.data))


class QPALM(QpSolver):
    """QP interface for the QPALM solver"""

    MIP_CAPABLE = False

    def name(self):
        return s.QPALM

    def import_solver(self) -> None:
        import qpalm
        qpalm

    def invert(self, solution, inverse_data):
        import qpalm

        # Map of QPALM status to CVXPY status.
        STATUS_MAP = {
            qpalm.Info.SOLVED: s.OPTIMAL,
            qpalm.Info.PRIMAL_INFEASIBLE: s.INFEASIBLE,
            qpalm.Info.DUAL_INFEASIBLE: s.UNBOUNDED,
            qpalm.Info.MAX_ITER_REACHED: s.USER_LIMIT,
            qpalm.Info.TIME_LIMIT_REACHED: s.USER_LIMIT,
            qpalm.Info.UNSOLVED: s.SOLVER_ERROR,
            qpalm.Info.ERROR: s.SOLVER_ERROR,
        }

        # Map QPALM statuses back to CVXPY statuses
        status = STATUS_MAP.get(solution.info.status_val, s.SOLVER_ERROR)

        attr = {s.SOLVE_TIME: solution.info.run_time}
        attr[s.EXTRA_STATS] = {"info": solution.info, "solver": solution}

        if status in s.SOLUTION_PRESENT:
            opt_val = solution.info.objective + inverse_data[s.OFFSET]
            primal_vars = {
                QPALM.VAR_ID:
                intf.DEFAULT_INTF.const_to_matrix(solution.solution.x.copy())
            }
            # Build dual vars dict keyed by constraint IDs
            # QPALM returns duals for [eq_constrs; ineq_constrs]
            y = solution.solution.y.copy()
            n_eq = inverse_data[self.DIMS].zero
            eq_dual = utilities.get_dual_values(
                y[:n_eq],
                utilities.extract_dual_value,
                inverse_data[self.EQ_CONSTR])
            ineq_dual = utilities.get_dual_values(
                y[n_eq:],
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
        import qpalm

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
            raise ValueError("warm_start is not found in data. Please set warm_start to True or False.")
        elif "warm_start" in data:
            raise ValueError("update is not found in data. Please set update to True or False.")

        P = data[s.P]
        q = data[s.Q]
        A = sp.vstack([data[s.A], data[s.F]]).tocsc()
        b_max = np.concatenate((data[s.B], data[s.G]))
        b_min = np.concatenate([data[s.B], -np.inf * np.ones_like(data[s.G])])
        n_con, n_var = A.shape

        qp_data = qpalm.Data(n_var, n_con)
        qp_data.Q = sp.triu(P).tocsc()
        qp_data.q = q
        qp_data.A = A
        qp_data.bmin = b_min
        qp_data.bmax = b_max

        settings = qpalm.Settings()
        # Chosen to match PIQP's default tolerances:
        # https://github.com/PREDICT-EPFL/piqp/blob/5115f0c08b86de40aff90f7f717956f0a573c627/include/piqp/settings.hpp#L48-L49
        settings.eps_abs = 1e-8
        settings.eps_rel = 1e-9
        # By default, QPALM is a bit too eager in declaring infeasibility when
        # decreasing eps_{abs,rel}, so also decrease the feasibility tolerances
        settings.eps_dual_inf = 1e-8
        settings.eps_prim_inf = 1e-8
        settings.verbose = verbose
        for k, v in solver_opts.items():
            try:
                setattr(settings, k, v)
            except TypeError as e:
                raise TypeError(f"QPALM: Incorrect type for setting '{k}'.") from e
            except AttributeError as e:
                raise TypeError(f"QPALM: Unrecognized solver setting '{k}'.") from e

        def sp_neq(a, b):
            return a.data.shape != b.data.shape or any(a.data != b.data)

        if custom_mode:
            # Self-implemented warm-start and update controls.
            if update:
                if solver_cache is None or self.name() not in solver_cache:
                    raise ValueError(
                        "Solver cache is not found. Solve once before using data['update']=True."
                    )
                solver, old_data = solver_cache[self.name()]
                if (_sparse_matrix_data_changed(qp_data.Q, old_data.Q)
                        or _sparse_matrix_data_changed(qp_data.A, old_data.A)):
                    solver.update_Q_A(qp_data.Q.data, qp_data.A.data)
                if not np.array_equal(old_data.q, qp_data.q):
                    solver.update_q(qp_data.q)
                if (not np.array_equal(old_data.bmin, qp_data.bmin)
                        or not np.array_equal(old_data.bmax, qp_data.bmax)):
                    solver.update_bounds(bmin=qp_data.bmin, bmax=qp_data.bmax)
                solver.update_settings(settings)
            else:
                solver = qpalm.Solver(qp_data, settings)

            if warm_start:
                ws_dict = data.get("warm_start_solution_dict")
                if not isinstance(ws_dict, dict) or len(ws_dict) == 0:
                    raise ValueError(
                        "data['warm_start_solution_dict'] must be a non-empty dict when "
                        "data['warm_start']=True."
                    )
                missing = {"x", "y"} - set(ws_dict.keys())
                if missing:
                    raise ValueError(
                        "data['warm_start_solution_dict'] is missing required keys: "
                        f"{sorted(missing)}."
                    )
                x_ws = np.asarray(ws_dict["x"]).reshape(-1)
                y_ws = np.asarray(ws_dict["y"]).reshape(-1)
                if x_ws.size != n_var:
                    raise ValueError(
                        "Invalid warm-start shape for 'x': expected length "
                        f"{n_var}, got {x_ws.size}."
                    )
                if y_ws.size != n_con:
                    raise ValueError(
                        "Invalid warm-start shape for 'y': expected length "
                        f"{n_con}, got {y_ws.size}."
                    )
                if not np.all(np.isfinite(x_ws)) or not np.all(np.isfinite(y_ws)):
                    raise ValueError("Warm-start values for 'x' and 'y' must be finite.")
                solver.warm_start(x_ws, y_ws)
            else:
                solver.warm_start(np.zeros(n_var), np.zeros(n_con))
        else:
            # Original CVXPY implementation.
            if warm_start and self.name() in solver_cache:
                solver, old_data = solver_cache[self.name()]
                if sp_neq(old_data.Q, qp_data.Q) or sp_neq(old_data.A, qp_data.A):
                    solver.update_Q_A(qp_data.Q.data, qp_data.A.data)
                if (old_data.q != qp_data.q).any():
                    solver.update_q(qp_data.q)
                if (old_data.bmin != qp_data.bmin).any() or (old_data.bmax != qp_data.bmax).any():
                    solver.update_bounds(bmin=qp_data.bmin, bmax=qp_data.bmax)
                solver.update_settings(settings)
                solver.warm_start(solver.solution.x, solver.solution.y)
            else:
                solver = qpalm.Solver(qp_data, settings)
        solver.solve()

        if solver_cache is not None:
            solver_cache[self.name()] = solver, qp_data

        return solver

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["QPALM"]
