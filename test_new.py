"""
Demonstrate custom warm-start/update behavior for OSQP, SCS, QPALM, and PROXQP.
NOTE: In these case studies, we always reset and solve the baseline problem
before re-solving with warm-start/update options.
"""

import argparse
import time
import warnings

import cvxpy as cp
import numpy as np


def build_problem():
    """Build a parameterized QP and two nearby parameter settings."""
    m = 750
    n = 500
    p = 100
    np.random.seed(1)

    P = np.random.randn(n, n)
    P = cp.psd_wrap(P.T @ P)
    A = np.random.randn(p, n)
    G = np.random.randn(m, n)

    q_param = cp.Parameter(n, name="q")
    h_param = cp.Parameter(m, name="h")
    b_param = cp.Parameter(p, name="b")

    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(x, P) + q_param.T @ x),
        [G @ x <= h_param, A @ x == b_param],
    )

    q = np.random.randn(n)
    h = G @ np.random.randn(n)
    b = np.random.randn(p)

    q_modified = q + 0.2 * np.random.randn(n)
    b_modified = b + 0.2 * np.random.randn(p)
    h_modified = h + 0.2 * np.random.randn(m)
    return prob, q, h, b, q_modified, b_modified, h_modified


def assign_parameters(prob, q, h, b):
    prob.param_dict["q"].value = q
    prob.param_dict["h"].value = h
    prob.param_dict["b"].value = b


def report_time_osqp(soln, canonical_time, solve_call_time, end_to_end_time):
    setup_time = soln.info.setup_time
    solve_time = soln.info.solve_time
    update_time = soln.info.update_time
    polish_time = soln.info.polish_time
    run_time = soln.info.run_time
    iter = soln.info.iter
    print(f"Total time (canonical + run): {round(canonical_time + run_time, 5)}")
    print(f"Solve call wall time: {round(solve_call_time, 5)}")
    print(f"End-to-end wall time: {round(end_to_end_time, 5)}")
    print(
        "Canonical: "
        f"{round(canonical_time, 5)}, "
        f"setup: {round(setup_time, 5)}, "
        f"update: {round(update_time, 5)}, "
        f"solve: {round(solve_time, 5)}, "
        f"polish: {round(polish_time, 5)}, "
        f"run: {round(run_time, 5)}, "
        f"iter: {iter}"
    )


def report_time_scs(soln, canonical_time, solve_call_time, end_to_end_time):
    solve_time = round(soln["info"]["solve_time"] / 1000, 5)
    setup_time = round(soln["info"]["setup_time"] / 1000, 5)
    cone_time = round(soln["info"]["cone_time"] / 1000, 5)
    accel_time = round(soln["info"]["accel_time"] / 1000, 5)
    lin_sys_time = round(soln["info"]["lin_sys_time"] / 1000, 5)
    iter = soln["info"]["iter"]
    canonical_time = round(canonical_time, 5)
    print(
        f"Total time (canonical + setup + solve): "
        f"{round(canonical_time + setup_time + solve_time, 5)}"
    )
    print(f"Solve call wall time: {round(solve_call_time, 5)}")
    print(f"End-to-end wall time: {round(end_to_end_time, 5)}")
    print(
        "Canonical: "
        f"{canonical_time}, "
        f"setup: {setup_time}, "
        f"solve: {solve_time}, "
        f"cone: {cone_time}, "
        f"accel: {accel_time}, "
        f"linear system: {lin_sys_time}, "
        f"iter: {iter}"
    )


def report_time_qpalm(soln, canonical_time, solve_call_time, end_to_end_time):
    run_time = soln.info.run_time
    setup_time = soln.info.setup_time
    solve_time = soln.info.solve_time
    iter = soln.info.iter
    
    print(f"Total time (canonical + run): {round(canonical_time + run_time, 5)}")
    print(f"Solve call wall time: {round(solve_call_time, 5)}")
    print(f"End-to-end wall time: {round(end_to_end_time, 5)}")
    print(
        "Canonical: "
        f"{round(canonical_time, 5)}, "
        f"setup: {round(setup_time, 5)}, "
        f"solve: {round(solve_time, 5)}, "
        f"run: {round(run_time, 5)}, " # this is the in-solver time
        f"iter: {iter}"
    )


def report_time_proxqp(soln, canonical_time, solve_call_time, end_to_end_time):
    # PROXQP timing fields are reported in microseconds.
    run_time = soln.info.run_time / 1e6
    solve_time = soln.info.solve_time / 1e6
    setup_time = run_time - solve_time
    iter = soln.info.iter
    print(f"Total time (canonical + run): {round(canonical_time + run_time, 5)}")
    print(f"Solve call wall time: {round(solve_call_time, 5)}")
    print(f"End-to-end wall time: {round(end_to_end_time, 5)}")
    print(
        "Canonical: "
        f"{round(canonical_time, 5)}, "
        f"setup: {round(setup_time, 5)}, "
        f"solve: {round(solve_time, 5)}, "
        f"run: {round(run_time, 5)}, "
        f"iter: {iter}"
    )

def report_time(soln, solver, canonical_time, solve_call_time, end_to_end_time):
    if solver == cp.OSQP:
        report_time_osqp(soln, canonical_time, solve_call_time, end_to_end_time)
    elif solver == cp.SCS:
        report_time_scs(soln, canonical_time, solve_call_time, end_to_end_time)
    elif solver == cp.QPALM:
        report_time_qpalm(soln, canonical_time, solve_call_time, end_to_end_time)
    else:
        report_time_proxqp(soln, canonical_time, solve_call_time, end_to_end_time)


def extract_objective_value(soln, solver):
    if solver == cp.OSQP:
        return float(soln.info.obj_val)
    if solver == cp.SCS:
        return float(soln["info"]["pobj"])
    if solver == cp.QPALM:
        return float(soln.info.objective)
    return float(soln.info.objValue)


def print_objective_value(soln, solver):
    obj_val = extract_objective_value(soln, solver)
    print(f"Objective value: {obj_val:.8f}")


def solve_problem_original(prob, solver, solver_opts, warm_start, show_time, verbose=False, print_obj=True):
    end_to_end_start = time.time()
    canonical_start = time.time()
    data, chain, _ = prob.get_problem_data(solver=solver)
    canonical_time = time.time() - canonical_start
    solve_start = time.time()
    soln = chain.solve_via_data(
        problem=prob,
        data=data,
        warm_start=warm_start,
        verbose=verbose,
        solver_opts=solver_opts,
    )
    solve_call_time = time.time() - solve_start
    end_to_end_time = time.time() - end_to_end_start
    if print_obj:
        print_objective_value(soln, solver)
        
    # print(dir(soln))
    # print(dir(soln.info))
    # print(dir(soln.solution))
    
    # exit()

    if show_time:
        report_time(soln, solver, canonical_time, solve_call_time, end_to_end_time)
    return soln


def solve_problem_custom(
    prob,
    solver,
    solver_opts,
    warm_start,
    update,
    warm_start_solution_dict=None,
    show_time=False,
    verbose=False,
    print_obj: bool = True
):
    end_to_end_start = time.time()
    canonical_start = time.time()
    data, chain, _ = prob.get_problem_data(solver=solver)
    canonical_time = time.time() - canonical_start
    data["warm_start"] = warm_start
    data["update"] = update
    if warm_start_solution_dict is not None:
        data["warm_start_solution_dict"] = warm_start_solution_dict
    # In custom mode, data["warm_start"] controls behavior,
    # which overrides the warm_start argument passed to solve_via_data.
    solve_start = time.time()
    soln = chain.solve_via_data(
        problem=prob,
        data=data,
        warm_start=False, # This argument is ignored in custom mode.
        verbose=verbose,
        solver_opts=solver_opts,
    )
    solve_call_time = time.time() - solve_start
    end_to_end_time = time.time() - end_to_end_start
    if print_obj:
        print_objective_value(soln, solver)

    if show_time:
        report_time(soln, solver, canonical_time, solve_call_time, end_to_end_time)
    return soln


def build_warm_start_dict(solver_name, soln):
    if solver_name == "osqp":
        return {"x": soln.x, "y": soln.y}
    if solver_name == "scs":
        return {"x": soln["x"], "y": soln["y"], "s": soln["s"]}
    if solver_name == "qpalm":
        return {"x": soln.solution.x.copy(), "y": soln.solution.y.copy()}
    return {"x": soln.x.copy(), "y": soln.y.copy(), "z": soln.z.copy()}


def print_case(title):
    print("=" * 100)
    print(title)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver",
        type=str,
        default="osqp",
        choices=["osqp", "scs", "qpalm", "proxqp"],
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
    )
    args = parser.parse_args()
    solver_name = args.solver

    print(f"Testing {solver_name} solver")
    verbose = args.verbose
    pre_solve_verbose = False
    
    if solver_name == "osqp":
        solver = cp.OSQP
        solver_opts = {"polishing": False}
    elif solver_name == "scs":
        solver = cp.SCS
        solver_opts = {"adaptive_scale": True}
    elif solver_name == "qpalm":
        solver = cp.QPALM
        solver_opts = {"eps_abs": 1e-6, "eps_rel": 1e-6}
    else:
        solver = cp.PROXQP
        solver_opts = {"backend": "sparse", "compute_timings": True}

    print_case("Basic problem: Original CVXPY, No warm start")
    prob, q, h, b, q_mod, b_mod, h_mod = build_problem()
    assign_parameters(prob, q, h, b)
    soln = solve_problem_original(prob, solver, solver_opts, False, True, verbose=verbose)

    print_case("New problem: Original CVXPY, No warm start")
    prob, q, h, b, q_mod, b_mod, h_mod = build_problem()
    assign_parameters(prob, q_mod, h_mod, b_mod)
    _ = solve_problem_original(prob, solver, solver_opts, False, True, verbose=verbose)

    print_case("New problem: Original CVXPY, Warm start")
    prob, q, h, b, q_mod, b_mod, h_mod = build_problem()
    assign_parameters(prob, q, h, b)
    _ = solve_problem_original(
        prob, solver, solver_opts, False, False, verbose=pre_solve_verbose, print_obj=False
    )
    assign_parameters(prob, q_mod, h_mod, b_mod)
    _ = solve_problem_original(prob, solver, solver_opts, True, True, verbose=verbose)

    print("=" * 100)
    print()

    print_case("New problem: New CVXPY, No warm start, No update")
    prob, q, h, b, q_mod, b_mod, h_mod = build_problem()
    assign_parameters(prob, q_mod, h_mod, b_mod)
    _ = solve_problem_custom(
        prob,
        solver,
        solver_opts,
        warm_start=False,
        update=False,
        warm_start_solution_dict=None,
        show_time=True,
        verbose=verbose,
    )

    print_case("New problem: New CVXPY, Warm start, No update")
    prob, q, h, b, q_mod, b_mod, h_mod = build_problem()
    assign_parameters(prob, q, h, b)
    soln = solve_problem_custom(
        prob, solver, solver_opts, False, False, verbose=pre_solve_verbose, print_obj=False
    )
    assign_parameters(prob, q_mod, h_mod, b_mod)
    warm_start_solution_dict = build_warm_start_dict(solver_name, soln)
    _ = solve_problem_custom(
        prob,
        solver,
        solver_opts,
        warm_start=True,
        update=False,
        warm_start_solution_dict=warm_start_solution_dict,
        show_time=True,
        verbose=verbose,
    )

    print_case("New problem: New CVXPY, No warm start, Update")
    if solver_name == "osqp":
        warnings.warn(
            "When update=True, OSQP may still report previous setup time. "
            "This is not included in the total time."
        )
    prob, q, h, b, q_mod, b_mod, h_mod = build_problem()
    assign_parameters(prob, q, h, b)
    _ = solve_problem_custom(
        prob, solver, solver_opts, False, False, verbose=pre_solve_verbose, print_obj=False
    )
    assign_parameters(prob, q_mod, h_mod, b_mod)
    _ = solve_problem_custom(
        prob,
        solver,
        solver_opts,
        warm_start=False,
        update=True,
        warm_start_solution_dict=None,
        show_time=True,
        verbose=verbose,
    )

    print_case("New problem: New CVXPY, Warm start, Update")
    if solver_name == "osqp":
        warnings.warn(
            "When update=True, OSQP may still report previous setup time. "
            "This is not included in the total time."
        )
    prob, q, h, b, q_mod, b_mod, h_mod = build_problem()
    assign_parameters(prob, q, h, b)
    soln = solve_problem_custom(
        prob, solver, solver_opts, False, False, verbose=pre_solve_verbose, print_obj=False
    )
    assign_parameters(prob, q_mod, h_mod, b_mod)
    warm_start_solution_dict = build_warm_start_dict(solver_name, soln)
    _ = solve_problem_custom(
        prob,
        solver,
        solver_opts,
        warm_start=True,
        update=True,
        warm_start_solution_dict=warm_start_solution_dict,
        show_time=True,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
