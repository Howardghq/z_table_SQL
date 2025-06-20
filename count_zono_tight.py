#!/usr/bin/env python3
"""
LP-Based Simulation for COUNT(*)
  SELECT COUNT(*) FROM T
  WHERE x > y AND y > z;

"""

import pandas as pd
import pulp
import os, sys
import numpy as np

from pulp import PULP_CBC_CMD

utilDir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(utilDir)


def bound(z):
    """
    Compute the lower and upper bounds of a zonotope z.
    z is a pandas Series: [center, g1, idx1, g2, idx2, ..., gn, idxn]
    """
    c = z['center']
    # Extract coefficients: g1, g2, ..., gn
    coeff_cols = [col for col in z.index if col.startswith('g')]
    coefficients = z[coeff_cols].values
    # Use np.nansum to ignore NaN values (treat NaN as 0)
    abs_sum = np.nansum(np.abs(coefficients))
    lower_bound = c - abs_sum
    upper_bound = c + abs_sum
    return lower_bound, upper_bound


def compute_count_with_constraint(df, df_x, df_z, obj, epsilon=1e-6, M=400, relax=False):
    if obj == 'min':
        prob = pulp.LpProblem("Count_Rows_Min", pulp.LpMinimize)
    else:
        prob = pulp.LpProblem("Count_Rows_Max", pulp.LpMaximize)
    
    total_count_expr = 0
    
    e_vars = {}
    
    binary_vars = {}
    #def_out = 0
    for idx, row in df.iterrows():
        x_threshold = df_x.iloc[idx]["center"]
        z_threshold = df_z.iloc[idx]["center"]

        y_lb, y_ub = bound(row)
        c = row["center"]
        # Gather generator pairs from the row.
        pairs = []
        i = 1
        while f"g{i}" in row and f"idx{i}" in row:
            if pd.isna(row[f"g{i}"]) or pd.isna(row[f"idx{i}"]):
                break
            pairs.append( (row[f"g{i}"], int(row[f"idx{i}"])) )
            i += 1
        # Case 1: Definitely In:
        if y_ub < x_threshold and y_lb > z_threshold:
            total_count_expr += 1  # count this row
            continue
        # Case 2: Definitely Out:
        if y_lb >= x_threshold or y_ub <= z_threshold:
            #def_out += 1
            continue
        # Case 3: Uncertain:
        if relax:
            i_var = pulp.LpVariable(f"i_{idx}",lowBound=0, upBound=1, cat='Continuous')
        else:
            i_var = pulp.LpVariable(f"i_{idx}", lowBound=0, upBound=1, cat="Binary")
        binary_vars[idx] = i_var
        total_count_expr += i_var
        
        error_expr = 0
        for (coeff, err_idx) in pairs:
            key = int(err_idx)
            if key not in e_vars:
                e_vars[key] = pulp.LpVariable(f"e_{key}",
                                            lowBound=-1, upBound=1,
                                            cat="Continuous")
            error_expr += coeff * e_vars[key]

        y_expr = c + error_expr
        # x > y AND y > z;
        # constraints to force that if i_var == 1, then both:
        #   (i) y_expr <= x_threshold - epsilon
        #   (ii) y_expr >= z_threshold + epsilon
        # We use big-M to “relax” the constraint when i_var == 0.
        prob += y_expr <= x_threshold - epsilon + M*(1 - i_var)
        prob += y_expr >= z_threshold + epsilon - M*(1 - i_var)
        prob += y_expr >= x_threshold - M*i_var
        prob += y_expr <= z_threshold + M*i_var
        """"
        # print uncertain row to debug.
        print(f"Row {idx}: x = {x_threshold}, z = {z_threshold}, center = {c}, "
              f"y_lb = {y_lb}, y_ub = {y_ub}, computed y_expr (symbolic) = {y_expr}")
        """
    prob += total_count_expr
    #print(f"Peng, definitely out are: {def_out}")
    # Solve the LP (forcing binary variables to be 0 or 1)
    solver = PULP_CBC_CMD(msg=0)
    prob.solve(solver)
    
    status = pulp.LpStatus[prob.status]
    count_val = pulp.value(prob.objective)
    lambda_values = {idx: pulp.value(var) for idx, var in binary_vars.items()}
    """"
    # For debugging, print each binary variable’s value.
    print("Binary variables (row indicators):")
    for var in prob.variables():
        if var.name.startswith("i_"):
            print(f"{var.name} = {var.varValue}")
    """
    return status, count_val, lambda_values

def compute_count_with_constraint_tight(df_y, df_x, df_z,
                                        obj='min',
                                        epsilon=1e-6,
                                        relax=False):
    """
    LP / MIP solver for
        SELECT COUNT(*) FROM T
        WHERE x > y AND y > z
    with row‑level Big‑M tightening.

    Parameters match the original function.
    """
    sense = pulp.LpMinimize if obj == 'min' else pulp.LpMaximize
    prob  = pulp.LpProblem(f"Count_Rows_{obj}", sense)

    total_count = 0
    i_vars, e_vars = {}, {}

    for idx, row in df_y.iterrows():
        x_thr = df_x.iloc[idx]['center']
        z_thr = df_z.iloc[idx]['center']
        y_lb, y_ub = bound(row)          # same helper you already have
        c = row['center']

        # --- Rows that are definitely IN or OUT -------------------------
        if y_ub < x_thr and y_lb > z_thr:   # always satisfies x>y>z
            total_count += 1
            continue
        if y_lb >= x_thr or y_ub <= z_thr:  # can never satisfy x>y>z
            continue

        # --- Uncertain rows: create indicator ---------------------------
        i_var = pulp.LpVariable(f"i_{idx}",
                                lowBound=0, upBound=1,
                                cat='Continuous' if relax else 'Binary')
        i_vars[idx] = i_var
        total_count += i_var

        # Build y_expr = center + Σ g·e
        err_expr = 0
        k = 1
        while f"g{k}" in row and f"idx{k}" in row and not pd.isna(row[f"g{k}"]):
            coeff = row[f"g{k}"]
            key   = int(row[f"idx{k}"])
            if key not in e_vars:
                e_vars[key] = pulp.LpVariable(f"e_{key}",
                                              lowBound=-1, upBound=1,
                                              cat='Continuous')
            err_expr += coeff * e_vars[key]
            k += 1
        y_expr = c + err_expr

        # Row‑specific Big‑M constants
        M1 = max(0.0, y_ub - (x_thr - epsilon))
        M2 = max(0.0, (z_thr + epsilon) - y_lb)
        M = max(M1, M2)

        # Big‑M constraints
        prob += y_expr <= x_thr - epsilon + M * (1 - i_var)
        prob += y_expr >= z_thr + epsilon - M * (1 - i_var)
        prob += y_expr >= x_thr - M * i_var
        prob += y_expr <= z_thr + M * i_var

    # Objective: min or max the count
    prob += total_count

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[prob.status]
    count_val = pulp.value(prob.objective)
    lambda_vals = {idx: pulp.value(v) for idx, v in i_vars.items()}

    return status, count_val, lambda_vals

def main():
    """
    data = [
        {"center": 3.5, "g1": 2.5, "idx1": 1}
    ]
    df_y = pd.DataFrame(data)
    print("Input zonotope for y:")
    print(df_y)

    x = [
        {"center": 2}
    ]
    df_x = pd.DataFrame(x)

    z = [
        {"center": 5}
    ]
    df_z = pd.DataFrame(z)
    """
    num_groups = 1
    size = 1000
    relation_degree = 0.7
    file_x = f"./Data/COUNT/count_x_{num_groups}_groups_size_{size}.parquet"
    file_y = f"./Data/COUNT/count_y_{num_groups}_groups_size_{size}_relation_degree_{int(relation_degree * 100)}.parquet"
    file_z = f"./Data/COUNT/count_z_{num_groups}_groups_size_{size}.parquet"
    
    try:
        df_x = pd.read_parquet(file_x)
        df_y = pd.read_parquet(file_y)
        df_z = pd.read_parquet(file_z)
        print("Files loaded successfully.")
    except Exception as e:
        print(f"Error reading files: {e}")
        return
    
    print(df_y.head())
    
    # Check that all DataFrames have the same number of rows.
    if not (len(df_x) == len(df_y) == len(df_z)):
        print("Error: The files for x, y, and z must have the same number of rows for row-wise comparison.")
        return
    
    # Also run the original count computation for comparison
    status_min, count_min, lambda_min = compute_count_with_constraint(df_y, df_x, df_z, 'min', relax=False)
    status_max, count_max, lambda_max = compute_count_with_constraint(df_y, df_x, df_z, 'max', relax=False)
    
    print("\nOriginal COUNT Computation (for rows satisfying x > y and y > z):")
    print(f"Lower and Upper Bound under Original MILP: [{int(count_min)}, {int(count_max)}]")

    status_min, count_min, lambda_min = compute_count_with_constraint(df_y, df_x, df_z, 'min', relax=True)
    status_max, count_max, lambda_max = compute_count_with_constraint(df_y, df_x, df_z, 'max', relax=True)
    
    print("\nOriginal COUNT Computation (for rows satisfying x > y and y > z):")
    print(f"Lower and Upper Bound under Fully Relaxed MILP: [{int(count_min)}, {int(count_max)}]")

    status_min, count_min, lambda_min = compute_count_with_constraint_tight(df_y, df_x, df_z, 'min', relax=True)
    status_max, count_max, lambda_max = compute_count_with_constraint_tight(df_y, df_x, df_z, 'max', relax=True)
    
    print("\nOriginal COUNT Computation (for rows satisfying x > y and y > z):")
    print(f"Lower and Upper Bound under Relaxed MILP with Tighter Constraints: [{int(count_min)}, {int(count_max)}]")
    
    print("\n")
    
if __name__ == "__main__":
    main()
