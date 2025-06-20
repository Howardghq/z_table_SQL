#!/usr/bin/env python3
"""
Iterative approach to ensure binary variables are correctly handled
when solving the COUNT(*) optimization problem.
"""

import pandas as pd
import pulp
import numpy as np
import os, sys

# Import alternative solver support
try:
    from pulp import GUROBI_CMD
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False

try:
    from pulp import CPLEX_CMD
    HAS_CPLEX = True
except ImportError:
    HAS_CPLEX = False

try:
    from pulp import GLPK_CMD
    HAS_GLPK = True
except ImportError:
    HAS_GLPK = False

# Always available
from pulp import PULP_CBC_CMD


def solve_with_fixed_variables(df_x, df_y, df_z, fixed_vars=None, 
                              solver_name='cbc', epsilon=1e-6, M=1e3):
    """
    Solve the optimization problem with some variables fixed.
    
    Args:
        df_x, df_y, df_z: DataFrames
        fixed_vars: Dictionary with keys as variable indices and values as 0 or 1
        solver_name: Name of solver to use
        epsilon, M: Constant parameters
    """
    prob = pulp.LpProblem("Maximize_Lambda_Sum_Fixed", pulp.LpMaximize)
    
    # Create binary variables
    lambda_vars = {}
    for idx in range(len(df_x)):
        if fixed_vars and idx in fixed_vars:
            # For fixed variables, we don't need to create a variable
            continue
        else:
            lambda_vars[idx] = pulp.LpVariable(f"lambda_{idx}", cat="Binary")
    
    # Create error variables for zonotope representation
    e_vars = {}
    
    # Define objective: maximize sum of λ_r
    if fixed_vars:
        # For fixed variables, directly add their values to the objective
        fixed_sum = sum(fixed_vars.values())
        objective = fixed_sum + pulp.lpSum(lambda_vars.values())
    else:
        objective = pulp.lpSum(lambda_vars.values())
    
    prob += objective
    
    # Add constraints for each row
    for idx in range(len(df_x)):
        x_r = df_x.iloc[idx]["center"]
        z_r = df_z.iloc[idx]["center"]
        
        # Get the zonotope representation for y_r
        row = df_y.iloc[idx]
        y_center = row["center"]
        
        # Build the error expression for y_r
        error_expr = 0
        pairs = []
        i = 1
        while f"g{i}" in row and f"idx{i}" in row:
            if pd.isna(row[f"g{i}"]) or pd.isna(row[f"idx{i}"]):
                break
            coeff = row[f"g{i}"]
            err_idx = int(row[f"idx{i}"])
            pairs.append((coeff, err_idx))
            i += 1
            
        # Create error variables if they don't exist yet
        for (coeff, err_idx) in pairs:
            key = int(err_idx)
            if key not in e_vars:
                e_vars[key] = pulp.LpVariable(f"e_{key}",
                                            lowBound=-1, upBound=1,
                                            cat="Continuous")
            error_expr += coeff * e_vars[key]
        
        # The full expression for y_r
        y_expr = y_center + error_expr
        
        # If this variable is fixed, handle differently
        if fixed_vars and idx in fixed_vars:
            if fixed_vars[idx] == 1:
                # If lambda_r is fixed to 1, constraints must be satisfied strictly
                prob += y_expr - x_r <= -epsilon
                prob += x_r - y_expr <= 0  # Equivalent to x_r >= y_expr
                prob += z_r - y_expr <= -epsilon
                prob += y_expr - z_r <= 0  # Equivalent to y_expr >= z_r
        else:
            # Regular big-M constraints
            lambda_r = lambda_vars[idx]
            
            # Constraint 1: y_r - x_r ≤ -ε + M(1-λ_r)
            prob += y_expr - x_r <= -epsilon + M * (1 - lambda_r)
            
            # Constraint 2: x_r - y_r ≤ M λ_r
            prob += x_r - y_expr <= M * lambda_r
            
            # Constraint 3: z_r - y_r ≤ -ε + M(1-λ_r)
            prob += z_r - y_expr <= -epsilon + M * (1 - lambda_r)
            
            # Constraint 4: y_r - z_r ≤ M λ_r
            prob += y_expr - z_r <= M * lambda_r
    
    solver = PULP_CBC_CMD(msg=False, gapRel=0, gapAbs=0)
    
    # Solve the problem
    prob.solve(solver)
    
    status = pulp.LpStatus[prob.status]
    
    # Check problem status
    if prob.status == pulp.LpStatusOptimal:
        objective_value = pulp.value(prob.objective)
        
        # Extract lambda variable values
        lambda_values = {}
        
        # First add fixed variables
        if fixed_vars:
            lambda_values.update(fixed_vars)
            
        # Then add solved variables
        for idx, var in lambda_vars.items():
            val = var.value()
            if val is not None:
                lambda_values[idx] = round(val)  # Round to nearest integer since it's a binary variable
            else:
                lambda_values[idx] = 0  # Default to 0 if no value
    else:
        objective_value = None
        lambda_values = {}
        if fixed_vars:
            lambda_values.update(fixed_vars)
    
    return status, objective_value, lambda_values


def iterative_solve(df_x, df_y, df_z, solver_name='cbc', epsilon=1e-6, M=1e3):
    """
    Use an iterative approach to ensure all variables are binary.
    """
    # Initialize: no fixed variables
    fixed_vars = {}
    iteration = 0
    max_iterations = 10  # Maximum number of iterations
    
    while iteration < max_iterations:
        print(f"\nIteration {iteration+1}:")
        status, obj_value, lambda_values = solve_with_fixed_variables(
            df_x, df_y, df_z, fixed_vars, solver_name, epsilon, M)
        
        print(f"Status: {status}")
        if obj_value is not None:
            print(f"Objective value: {obj_value}")
            print(f"Variables with λ=1: {sum(val == 1 for val in lambda_values.values())}")
        
        # Check if all values are binary
        non_binary = {idx: val for idx, val in lambda_values.items() 
                     if val is not None and not (abs(val) < 1e-6 or abs(val-1) < 1e-6)}
        
        if not non_binary or status != "Optimal":
            # If all values are binary or problem is no longer solvable, end iteration
            print("All variables are binary or problem is not solvable, ending iterations")
            break
        
        # Fix some variables that are close to 1 or 0
        num_fixed = 0
        for idx, val in sorted(non_binary.items(), key=lambda x: abs(x[1]-0.5), reverse=True)[:10]:
            if val > 0.8:  # If value is close to 1, fix to 1
                fixed_vars[idx] = 1
                num_fixed += 1
                print(f"  Fixed lambda_{idx} = 1 (current value: {val})")
            elif val < 0.2:  # If value is close to 0, fix to 0
                fixed_vars[idx] = 0
                num_fixed += 1
                print(f"  Fixed lambda_{idx} = 0 (current value: {val})")
        
        if num_fixed == 0:
            # If we couldn't fix any variables, fix the most 'extreme' ones
            for idx, val in sorted(non_binary.items(), key=lambda x: abs(x[1]-0.5), reverse=True)[:5]:
                fixed_vars[idx] = 1 if val >= 0.5 else 0
                print(f"  Forced fixing lambda_{idx} = {1 if val >= 0.5 else 0} (current value: {val})")
        
        iteration += 1
    
    if iteration == max_iterations:
        print("Reached maximum number of iterations")
    
    return status, obj_value, lambda_values


def verify_solution(df_x, df_y, df_z, lambda_values):
    """
    Verify that the solution satisfies the constraints x > y and y > z
    for rows where lambda = 1
    """
    valid_count = 0
    violated_count = 0
    
    for idx, val in lambda_values.items():
        if val == 1:
            x_r = df_x.iloc[idx]["center"]
            y_r = df_y.iloc[idx]["center"]
            z_r = df_z.iloc[idx]["center"]
            
            # Check if x_r > y_r and y_r > z_r
            if x_r > y_r and y_r > z_r:
                valid_count += 1
            else:
                violated_count += 1
                if violated_count <= 5:  # Only show first 5 violations
                    print(f"  Row {idx} violates constraints: x={x_r}, y={y_r}, z={z_r}")
    
    return valid_count, violated_count


def main():
    """
    Main function to read data and solve the optimization problem using an iterative approach.
    """
    print("Available solvers:")
    if HAS_GUROBI:
        print("- Gurobi")
    if HAS_CPLEX:
        print("- CPLEX")
    if HAS_GLPK:
        print("- GLPK")
    print("- CBC (default)")
    
    # Determine which solver to use
    if HAS_GUROBI:
        default_solver = 'gurobi'
    elif HAS_CPLEX:
        default_solver = 'cplex'
    elif HAS_GLPK:
        default_solver = 'glpk'
    else:
        default_solver = 'cbc'
    
    # Let the user choose a solver
    solver_name = input(f"Enter solver to use [{default_solver}]: ") or default_solver
    
    # File paths (adjust as needed)
    num_groups = 1
    size = 1000
    file_x = f"./Data/COUNT/count_x_{num_groups}_groups_size_{size}.parquet"
    file_y = f"./Data/COUNT/count_y_{num_groups}_groups_size_{size}.parquet"
    file_z = f"./Data/COUNT/count_z_{num_groups}_groups_size_{size}.parquet"
    
    try:
        df_x = pd.read_parquet(file_x)
        df_y = pd.read_parquet(file_y)
        df_z = pd.read_parquet(file_z)
        print("Files loaded successfully.")
    except Exception as e:
        print(f"Error reading files: {e}")
        return
    
    # Check that all DataFrames have the same number of rows
    if not (len(df_x) == len(df_y) == len(df_z)):
        print("Error: The files for x, y, and z must have the same number of rows for row-wise comparison.")
        return
    
    print("\nSolving optimization problem using iterative approach...")
    status, obj_value, lambda_values = iterative_solve(
        df_x, df_y, df_z, solver_name=solver_name)
    
    print("\nFinal Results:")
    print(f"Status: {status}")
    if obj_value is not None:
        print(f"Objective Value (sum of λ_r): {obj_value}")
        binary_count = sum(val == 1 for val in lambda_values.values())
        print(f"Number of rows where λ_r = 1: {binary_count}")
        
        # Verify the solution
        print("\nVerifying solution...")
        valid_count, violated_count = verify_solution(df_x, df_y, df_z, lambda_values)
        
        print(f"  Rows satisfying constraints: {valid_count}/{binary_count}")
        print(f"  Rows violating constraints: {violated_count}/{binary_count}")
    else:
        print("Failed to obtain a valid objective value and variable values")


if __name__ == "__main__":
    main()