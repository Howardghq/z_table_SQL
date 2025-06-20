#!/usr/bin/env python3
"""
Simulate the query:
    SELECT SUM(salary) FROM Employee WHERE salary > AVG(salary);
Assuming:
    - salary (for each row) is a zonotope represented as a pandas Series.
      The zonotope's range is [center – sum(|g_i|), center + sum(|g_i|)].
"""

import pandas as pd
import numpy as np
import os, sys

utilDir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(utilDir)
from range_class import R, bound_from_row

def sum_zonotopes(df_subset):
    """
    Sum multiple zonotopes. Each row represents a zonotope.
    Returns a single-row DataFrame representing the sum.
    """
    if len(df_subset) == 0:
        return pd.DataFrame([{'center': 0.0}])
    
    # Sum all centers
    total_center = df_subset['center'].sum()
    
    # Collect generators by their error index
    generators = {}
    for i, row in df_subset.iterrows():
        # Find all generator columns (g1, g2, etc.)
        gen_cols = [col for col in row.index if col.startswith('g') and pd.notna(row[col])]
        
        for gen_col in gen_cols:
            idx_col = 'idx' + gen_col[1:]  # g1 -> idx1, g2 -> idx2, etc.
            if idx_col in row.index and pd.notna(row[idx_col]):
                error_idx = int(row[idx_col])
                coeff = row[gen_col]
                
                # Add coefficient to the same error index
                if error_idx in generators:
                    generators[error_idx] += abs(coeff)
                else:
                    generators[error_idx] = abs(coeff)
    
    # Build result as single-row DataFrame
    result_data = {'center': total_center}
    
    # Add generators in order
    gen_idx = 1
    for error_idx in sorted(generators.keys()):
        if abs(generators[error_idx]) > 1e-10:  # Skip near-zero coefficients
            result_data[f'g{gen_idx}'] = generators[error_idx]
            result_data[f'idx{gen_idx}'] = error_idx
            gen_idx += 1
            
    return pd.DataFrame([result_data])

def compute_average_zonotope(df_z):
    """
    Compute the average zonotope from a DataFrame of zonotopes.
    """
    n = len(df_z)
    
    # Sum all zonotopes first
    sum_df = sum_zonotopes(df_z)
    sum_row = sum_df.iloc[0]
    
    # Divide by n to get average
    avg_data = {'center': sum_row['center'] / n}
    
    # Divide all generators by n
    gen_cols = [col for col in sum_row.index if col.startswith('g') and pd.notna(sum_row[col])]
    
    gen_idx = 1
    for gen_col in gen_cols:
        idx_col = 'idx' + gen_col[1:]
        if idx_col in sum_row.index and pd.notna(sum_row[idx_col]):
            avg_data[f'g{gen_idx}'] = sum_row[gen_col] / n
            avg_data[f'idx{gen_idx}'] = sum_row[idx_col]
            gen_idx += 1
    
    return pd.DataFrame([avg_data])

def sum_filtered_rows(df_z):
    """
    Compute the interval for SUM(salary) WHERE salary > AVG(salary).
    """
    # Compute average salary zonotope
    avg_df = compute_average_zonotope(df_z)
    avg_row = avg_df.iloc[0]
    avg_interval = bound_from_row(avg_row)
    
    # print(f"Average salary range: [{avg_interval.lb:.2f}, {avg_interval.ub:.2f}]")
    
    # For each row, compute salary interval and check condition
    definitely_in_mask = pd.Series(False, index=df_z.index)
    possibly_in_mask = pd.Series(False, index=df_z.index)
    
    for idx in df_z.index:
        salary_row = df_z.iloc[idx]
        salary_interval = bound_from_row(salary_row)
        
        # Definitely greater than average
        if salary_interval.lb > avg_interval.ub:
            definitely_in_mask[idx] = True
            possibly_in_mask[idx] = True
        # Definitely not greater than average
        elif salary_interval.ub <= avg_interval.lb:
            definitely_in_mask[idx] = False
            possibly_in_mask[idx] = False
        # Uncertain
        else:
            definitely_in_mask[idx] = False
            possibly_in_mask[idx] = True
    
    # print(f"Definitely included rows: {definitely_in_mask.sum()}")
    # print(f"Possibly included rows: {possibly_in_mask.sum()}")
    
    # Compute lower bound (only definitely included)
    if definitely_in_mask.any():
        lower_sum_df = sum_zonotopes(df_z[definitely_in_mask])
        lower_sum_row = lower_sum_df.iloc[0]
        lower_bound = bound_from_row(lower_sum_row).lb
    else:
        lower_bound = 0.0
    
    # Compute upper bound (all possibly included)
    if possibly_in_mask.any():
        upper_sum_df = sum_zonotopes(df_z[possibly_in_mask])
        upper_sum_row = upper_sum_df.iloc[0]
        upper_bound = bound_from_row(upper_sum_row).ub
    else:
        upper_bound = 0.0
    
    return R(lower_bound, upper_bound)

def main():
    # Test with the example data you provided
    # test_data = [
    #     {"center": 5149.967147, "g1": 346.969327, "idx1": 168},
    #     {"center": 7037.945487, "g1": 369.520348, "idx1": 549},
    #     {"center": 9765.959238, "g1": 427.824266, "idx1": 64},
    #     {"center": 6633.310220, "g1": -364.147846, "idx1": 921},
    #     {"center": 6228.063529, "g1": 31.969130, "idx1": 582},
    # ]
    
    # df_z = pd.DataFrame(test_data)
    # print("Test data loaded:")
    # print(df_z)
    # print()
    
    # # Compute the interval for SUM(salary) WHERE salary > AVG(salary)
    # sum_interval = sum_filtered_rows(df_z)
    # print(f"\nResult interval for SUM(salary) WHERE salary > AVG(salary): {sum_interval}")
    
    # Uncomment below to use real data files
    
    num_groups = 1
    size = 1000
    relation_degree = 0.5
    input_file = f'./Data/AVG/sum_z_{num_groups}_groups_size_{size}_relation_degree_{int(relation_degree * 100)}.parquet'
    
    try:
        df_z = pd.read_parquet(input_file)
        print(f"Loaded {len(df_z)} salary records from {input_file}")
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    sum_interval = sum_filtered_rows(df_z)
    print(f"\nResult interval for SUM(salary) WHERE salary > AVG(salary): {sum_interval}")
    

if __name__ == "__main__":
    main()