#!/usr/bin/env python3
"""
   SELECT SUM(salary) FROM Employee WHERE salary > AVG(salary);
"""

import pandas as pd
import numpy as np
import pulp, time
from sum_interval import sum_filtered_rows

from pulp import PULP_CBC_CMD

def bound(z):
    c = z['center']
    # Find all generator keys
    coeff_keys = [k for k in z.keys() if k.startswith('g')]
    coeffs = [z[k] for k in coeff_keys if pd.notna(z[k])]
    abs_sum = np.sum(np.abs(coeffs))
    return c - abs_sum, c + abs_sum

def sum_zonotopes(dataset):

    total_center = 0.0
    total_gens = {}
    for _, row in dataset.iterrows():
        total_center += row["center"]
        for key in row.index:
            if key.startswith("g") and pd.notna(row[key]):
                idx_key = "idx" + key[1:]
                if idx_key in row and pd.notna(row[idx_key]):
                    err_idx = int(row[idx_key])
                    total_gens[err_idx] = total_gens.get(err_idx, 0) + row[key]
    result = {"center": total_center}
    i = 1
    for err_idx, coeff in sorted(total_gens.items(), key=lambda x: x[0]):
        result[f"g{i}"] = coeff
        result[f"idx{i}"] = err_idx
        i += 1
    return result

def divide_zonotope(z, divisor):
    if divisor == 0:
        raise ValueError("Divisor cannot be zero.")
    new_z = {"center": z["center"] / divisor}
    gen_keys = [k for k in z.keys() if k.startswith("g") and pd.notna(z[k])]
    gen_keys = sorted(gen_keys, key=lambda k: int(k[1:]))
    i = 1
    for g in gen_keys:
        idx_key = "idx" + g[1:]
        if idx_key in z and pd.notna(z[idx_key]):
            new_z[f"g{i}"] = z[g] / divisor
            new_z[f"idx{i}"] = z[idx_key]
            i += 1
    return pd.Series(new_z)

def compute_average(dataset):
    total = sum_zonotopes(dataset)
    return divide_zonotope(total, len(dataset))

def compute_total_sum_with_threshold(dataset, obj, threshold=None, relax = True, M=1e3):
    """
      - If ub <= t_lb, then z is definitely below threshold
      - If lb >= t_ub, then z is definitely above threshold
      - Otherwise, z is uncertain
    """
    if obj == 'min':
        prob = pulp.LpProblem("Zonotope_Total_Sum_Min", pulp.LpMinimize)
    else:
        prob = pulp.LpProblem("Zonotope_Total_Sum_Max", pulp.LpMaximize)
    epsilon = 1e-6
    total_expr = 0
    gen_exprs = {}

    e_vars_global = {}
    
    if threshold is not None:
        if isinstance(threshold, (int, float)):
            t_lb = t_ub = float(threshold)
        else:
            T_center = threshold['center']
            coeff_cols = [c for c in threshold.index if c.startswith('g')]
            idx_cols   = [c for c in threshold.index if c.startswith('idx')]
            T_pairs = []
            for g_col, idx_col in zip(coeff_cols, idx_cols):
                if not pd.isna(threshold[idx_col]):
                    t_coeff = threshold[g_col]
                    t_idx   = int(threshold[idx_col])
                    T_pairs.append((t_coeff, t_idx))

            threshold_expr = T_center
            for (t_coeff, t_idx) in T_pairs:
                if t_idx not in e_vars_global:
                    e_vars_global[t_idx] = pulp.LpVariable(
                        f"e_threshold_{t_idx}", lowBound=-1, upBound=1, cat='Continuous'
                    )
                t_evar = e_vars_global[t_idx]
                threshold_expr += (t_coeff * t_evar)
            t_lb, t_ub = bound(threshold)
    else:
        t_lb = -np.inf
        t_ub = np.inf
    
    for idx, z in dataset.iterrows():
        lb, ub = bound(z)
        c = z['center']
        coeff_keys = [k for k in z.index if k.startswith('g')]
        idx_keys = [k for k in z.index if k.startswith('idx')]

        pairs = []
        for k, k_idx in zip(sorted(coeff_keys), sorted(idx_keys)):
            if pd.notna(z[k]) and pd.notna(z[k_idx]):
                pairs.append((z[k], int(z[k_idx])))
        if ub <= t_lb:
            continue
        elif lb >= t_ub:
            total_expr += c
            for coeff, err_idx in pairs:
                if err_idx in gen_exprs:
                    gen_exprs[err_idx] += coeff
                else:
                    gen_exprs[err_idx] = coeff
                if err_idx not in e_vars_global:
                    e_vars_global[err_idx] = pulp.LpVariable(f"e_{err_idx}", lowBound=-1, upBound=1, cat='Continuous')
        else:
            i_var = pulp.LpVariable(f"i_{idx}",lowBound=0, upBound=1, cat='Binary') if not relax else pulp.LpVariable(f"i_{idx}", lowBound=0, upBound=1, cat='Continuous')

            total_expr += c * i_var

            for coeff, err_idx in pairs:
                if err_idx not in e_vars_global:
                    e_vars_global[err_idx] = pulp.LpVariable(f"e_{err_idx}", lowBound=-1, upBound=1, cat='Continuous')
                e_var = e_vars_global[err_idx]
                
                v_var = pulp.LpVariable(f"v_{idx}_{err_idx}", lowBound=None, upBound=None, cat='Continuous')
                
                a = coeff
                
                if a >= 0:
                    prob += v_var >= a*(-1)*i_var
                    prob += v_var <= a*(+1)*i_var
                    prob += v_var >= a*e_var - a*(+1)*(1 - i_var)
                    prob += v_var <= a*e_var - a*(-1)*(1 - i_var)
                else:
                    prob += v_var >= a*(+1)*i_var
                    prob += v_var <= a*(-1)*i_var
                    prob += v_var >= a*e_var - a*(-1)*(1 - i_var)
                    prob += v_var <= a*e_var - a*(+1)*(1 - i_var)
                
                total_expr += v_var

            z_expr = c + pulp.lpSum(coeff * e_vars_global[err_idx] for coeff, err_idx in pairs)

            # Using big-M:
            M = max(0.0, (t_ub + epsilon) - lb)
            prob += (z_expr - threshold_expr) >= (epsilon - M * (1 - i_var))
            prob += (z_expr - threshold_expr) <= M * i_var
            
    # Build the overall objective: sum the centers plus generator contributions.
    # For each error variable, multiply its total coefficient.
    gen_term = pulp.lpSum([coef * e_vars_global[err_idx] for err_idx, coef in gen_exprs.items()])
    prob += total_expr + gen_term

    solver = PULP_CBC_CMD(msg=0)
    prob.solve(solver)
    """"
    # Print all variables for debugging:
    print("\n--- LP Variable Values ---")
    for var in prob.variables():
        print(f"{var.name} = {var.varValue}")
    """
    status = pulp.LpStatus[prob.status]
    obj_value = pulp.value(prob.objective)
    return status, obj_value

def main():
    """
    # Average: [5766.666666666667, 6566.666666666667]
    data = [
        {"center": 6000, "g1": 500, "idx1": 1}, #[5500, 6500] possibly in
        {"center": 7000, "g1": 400, "idx1": 2}, #[6600, 7400] definitely in
        {"center": 5500, "g1": 300, "idx1": 3}  #[5200, 5800] possibly in
    ]
    dataset_original = pd.DataFrame(data)
    print("Salary zonotopes:")
    print(dataset_original)
    """
    num_groups = 1
    size = 1000
    nvar = 1
    relational_degree = 0.5
    # input_file = f'./Data/AVG/z_{num_groups}_groups_size_{size}_error_num_{nvar}_.parquet'
    input_file = f'./Data/AVG/sum_z_{num_groups}_groups_size_{size}_relation_degree_{int(relational_degree * 100)}.parquet'
    print(f"Reading zonotope data from {input_file}...")
    try:
        dataset_original = pd.read_parquet(input_file)
        print("Zonotope data successfully loaded.")
    except Exception as e:
        raise ValueError(f"Error reading zonotope data: {e}")
    
    avg_z = compute_average(dataset_original)
    th_lb, th_ub = bound(avg_z)
    # print("\nAverage Salary Zonotope (Threshold):", avg_z)
    print("Average Salary Range: [{}, {}]".format(th_lb, th_ub))
    start_time = time.time()
    status_lower, sum_lower = compute_total_sum_with_threshold(dataset_original, 'min', threshold=avg_z, relax=True, M=1e3)
    status_upper, sum_upper = compute_total_sum_with_threshold(dataset_original, 'max', threshold=avg_z, relax=True, M=1e3)
    end_time = time.time()
    print("\nComputed Total Sum with zonotope (salary > AVG(salary)):")
    print(f"Lower and Upper Bound:[{sum_lower}, {sum_upper}]")

    print(f"Time : {end_time - start_time}")

    start_time = time.time()
    sum_interval = sum_filtered_rows(dataset_original)
    end_time = time.time()
    print("\nComputed Total Sum with interval (salary > AVG(salary)):")
    print(f"Lower and Upper Bound:[{sum_interval.lb}, {sum_interval.ub}]")
    print(f"Time : {end_time - start_time}")

if __name__ == "__main__":
    main()
