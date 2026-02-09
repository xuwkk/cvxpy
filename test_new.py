"""
Test new implemenration of osqp and scs
"""

# Import packages.
import cvxpy as cp
import numpy as np
import time
import warnings

def return_problem():
    # Generate a random non-trivial quadratic program.
    m = 750
    n = 500
    p = 100
    np.random.seed(1)
    
    P = np.random.randn(n, n)
    P = cp.psd_wrap(P.T @ P)
    A = np.random.randn(p, n)
    G = np.random.randn(m, n)
    
    q_param = cp.Parameter(n, name = 'q')
    h_param = cp.Parameter(m, name = 'h')
    b_param = cp.Parameter(p, name = 'b')
    
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q_param.T @ x),
                    [G @ x <= h_param,
                    A @ x == b_param])
    
    # original problem data
    q = np.random.randn(n)
    h = G @ np.random.randn(n)
    b = np.random.randn(p)
    
    # Modified version
    q_modified = q + 0.2 * np.random.randn(n)
    b_modified = b + 0.2 * np.random.randn(p)
    h_modified = h + 0.2 * np.random.randn(m)
    return prob, q, h, b, q_modified, b_modified, h_modified

def assign_parameters(prob, q, h, b):
    prob.param_dict['q'].value = q
    prob.param_dict['h'].value = h
    prob.param_dict['b'].value = b
    
def report_time_osqp(soln, cannonical_time):
    setup_time = soln.info.setup_time
    solve_time = soln.info.solve_time
    update_time = soln.info.update_time
    polish_time = soln.info.polish_time
    run_time = soln.info.run_time
    print(f'Total time (cannonical + run): {round(cannonical_time + run_time, 5)}')
    print(f'Cannonical time: {round(cannonical_time, 5)}, setup: {round(setup_time, 5)}, update: {round(update_time, 5)}, solve: {round(solve_time, 5)}, polish: {round(polish_time, 5)}')

def report_time_scs(soln, cannonical_time):
    solve_time = round(soln["info"]["solve_time"]/1000, 5)
    setup_time = round(soln["info"]["setup_time"]/1000, 5)
    cone_time = round(soln["info"]["cone_time"]/1000, 5)
    accel_time = round(soln["info"]["accel_time"]/1000, 5)
    lin_sys_time = round(soln["info"]["lin_sys_time"]/1000, 5)
    cannonical_time = round(cannonical_time, 5)
    
    print(f'Total time (cannonical + setup + solve): {round(cannonical_time + setup_time + solve_time, 5)}')
    print(f'Cannonical: {cannonical_time}, setup: {setup_time}, solve: {solve_time}, cone: {cone_time}, accel: {accel_time}, linear system: {lin_sys_time}')

def solve_problem_ori(prob, solver, solver_opts, warm_start, show_time, verbose = False):
    cannonical_time = time.time()
    data, chain, inverse_data = prob.get_problem_data(solver = solver)
    cannonical_time = time.time() - cannonical_time
    soln = chain.solve_via_data(problem = prob, data = data, warm_start = warm_start, verbose = verbose, solver_opts = solver_opts)
    if show_time:
        report_time_osqp(soln, cannonical_time) if solver == cp.OSQP else report_time_scs(soln, cannonical_time)
    return soln

def solve_problem_new(prob, solver, solver_opts, warm_start, 
                      update, warm_start_solution_dict = None, show_time = False, verbose = False):
    cannonical_time = time.time()
    data, chain, inverse_data = prob.get_problem_data(solver = solver)
    cannonical_time = time.time() - cannonical_time
    data["warm_start"] = warm_start
    data["update"] = update
    if warm_start_solution_dict is not None:
        data["warm_start_solution_dict"] = warm_start_solution_dict
    # The original warm start will be ignored
    soln = chain.solve_via_data(problem = prob, data = data, warm_start = False, verbose = verbose, solver_opts = solver_opts)
    if show_time:
        report_time_osqp(soln, cannonical_time) if solver == cp.OSQP else report_time_scs(soln, cannonical_time)
    return soln

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', type=str, default='osqp', choices=['osqp', 'scs'])
    args = parser.parse_args()
    
    print(f' Testing {args.solver} solver')
    
    verbose = False
    if args.solver == 'osqp':
        solver = cp.OSQP
        solver_opts = {'polish': False}
    elif args.solver == 'scs':
        solver = cp.SCS
        solver_opts = {
            'adaptive_scale': True,
            # 'acceleration_lookback': 1,
            # 'acceleration_interval': 1
        }
    else:
        raise ValueError(f'Invalid solver: {args.solver}')
    
    print('='*100)
    print('Basic problem: Original CVXPY, No warm start')
    
    prob, q, h, b, q_modified, b_modified, h_modified = return_problem()
    assign_parameters(prob, q, h, b)
    soln = solve_problem_ori(prob, solver, solver_opts, False, True, verbose = verbose)
    
    print('='*100)
    print('New problem: Original CVXPY, No warm start')
    
    prob, q, h, b, q_modified, b_modified, h_modified = return_problem()
    assign_parameters(prob, q_modified, h_modified, b_modified)
    soln = solve_problem_ori(prob, solver, solver_opts, False, True)
    
    print('='*100)
    print('New problem: Original CVXPY, Warm start')
    
    # Resolve the basic problem again
    prob, q, h, b, q_modified, b_modified, h_modified = return_problem()
    assign_parameters(prob, q, h, b)
    soln = solve_problem_ori(prob, solver, solver_opts, False, False)
    
    assign_parameters(prob, q_modified, h_modified, b_modified)
    soln = solve_problem_ori(prob, solver, solver_opts, True, True)
    
    
    print('='*100)
    print('\n')
    
    print('='*100)
    print('New problem: New CVXPY, No warm start, No update')
    prob, q, h, b, q_modified, b_modified, h_modified = return_problem()
    assign_parameters(prob, q_modified, h_modified, b_modified)
    soln = solve_problem_new(prob, solver, solver_opts, 
                             warm_start = False, 
                             update = False, 
                             warm_start_solution_dict = None, 
                             show_time = True,
                             verbose = False)
    
    print('='*100)
    print('New problem: New CVXPY, Warm start, No update')
    
    # Start by solving the basic problem
    prob, q, h, b, q_modified, b_modified, h_modified = return_problem()
    assign_parameters(prob, q, h, b)
    soln = solve_problem_ori(prob, solver, solver_opts, False, False)
    
    # Solve the new problem
    assign_parameters(prob, q_modified, h_modified, b_modified)
    if args.solver == 'osqp':
        warm_start_solution_dict = {
            "x": soln.x,  # Primal
            "y": soln.y   # Dual
        }
    elif args.solver == 'scs':
        warm_start_solution_dict = {
            "x": soln["x"],  # Primal
            "y": soln["y"],   # Dual
            "s": soln["s"]   # Slack
        }
        
    soln = solve_problem_new(prob, solver, solver_opts, 
                             warm_start = True, 
                             update = False, 
                             warm_start_solution_dict = warm_start_solution_dict, 
                             show_time = True)
    
    print('='*100)
    print('New problem: New CVXPY, No warm start, Update')
    
    if args.solver == 'osqp':
        message = "When update = True, the OSQP still reports the previous setup time."
    
        warnings.warn(message)
        
    # Start by solving the basic problem
    prob, q, h, b, q_modified, b_modified, h_modified = return_problem()
    assign_parameters(prob, q, h, b)
    soln = solve_problem_new(prob, solver, solver_opts, False, False, verbose = False)
    
    # Solve the new problem
    assign_parameters(prob, q_modified, h_modified, b_modified)
    soln = solve_problem_new(prob, solver, solver_opts, 
                             warm_start = False, 
                             update = True, 
                             warm_start_solution_dict = None, 
                             show_time = True, verbose = False)
    
    print('='*100)
    print('New problem: New CVXPY, Warm start, Update')
    
    if args.solver == 'osqp':
        message = "When update = True, the OSQP still reports the previous setup time."
    
        warnings.warn(message)
    
    # Start by solving the basic problem
    prob, q, h, b, q_modified, b_modified, h_modified = return_problem()
    assign_parameters(prob, q, h, b)
    soln = solve_problem_new(prob, solver, solver_opts, False, False, verbose = False)
    
    # Solve the new problem
    assign_parameters(prob, q_modified, h_modified, b_modified)
    if args.solver == 'osqp':
        warm_start_solution_dict = {
            "x": soln.x,  # Primal
            "y": soln.y   # Dual
        }
    elif args.solver == 'scs':
        warm_start_solution_dict = {
            "x": soln["x"],  # Primal
            "y": soln["y"],   # Dual
            "s": soln["s"]   # Slack
        }
    soln = solve_problem_new(prob, solver, solver_opts, 
                             warm_start = True, 
                             update = True, 
                             warm_start_solution_dict = warm_start_solution_dict, 
                             show_time = True,
                             verbose = False)
    