#!/usr/bin/env python3
"""
Dantzig-Wolfe Decomposition for COUNT(*) Query Optimization
  SELECT COUNT(*) FROM T
  WHERE x > y AND y > z;

Experiment with real data from parquet files.
"""

import pandas as pd
import pulp
import numpy as np
from pulp import PULP_CBC_CMD
import time
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker

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

def solve_subproblem(row_idx, row_y, x_threshold, z_threshold, epsilon=1e-6, obj='max', relax=False):
    """
    Solve the subproblem for row r to determine if it can satisfy
    the condition x > y and y > z.
    
    Parameters:
        row_idx: Index of the row
        row_y: Zonotope representation of y for this row
        x_threshold: Value of x for this row
        z_threshold: Value of z for this row
        epsilon: Small positive constant
        obj: 'min' or 'max' for objective
        relax: Whether to relax the binary variable to continuous
        
    Returns:
        lambda_value: 0 or 1 indicating if row can satisfy condition
        y_value: Value of y that satisfies the condition (or None)
        feasible: Whether the subproblem is feasible
    """
    y_lb, y_ub = bound(row_y)
    center = row_y["center"]
    
    # Case 1: Definitely In (always satisfies condition)
    if y_ub < x_threshold - epsilon and y_lb > z_threshold + epsilon:
        return 1, center, True
    
    # Case 2: Definitely Out (never satisfies condition)
    if y_lb >= x_threshold - epsilon or y_ub <= z_threshold + epsilon:
        return 0, None, False
    
    # Case 3: Uncertain - need to solve LP
    # Create model for this row's subproblem
    subprob = pulp.LpProblem(f"Subproblem_Row_{row_idx}", 
                            pulp.LpMaximize if obj == 'max' else pulp.LpMinimize)
    
    # Create variables
    if relax:
        lambda_var = pulp.LpVariable(f"lambda_{row_idx}", lowBound=0, upBound=1, cat="Continuous")
    else:
        lambda_var = pulp.LpVariable(f"lambda_{row_idx}", cat="Binary")
    
    y_var = pulp.LpVariable(f"y_{row_idx}", lowBound=y_lb, upBound=y_ub)
    
    # Objective: maximize/minimize lambda
    subprob += lambda_var
    
    # Big-M constraints for x > y
    M1 = max(0.0, y_ub - (x_threshold - epsilon))
    subprob += y_var - (x_threshold - epsilon) <= M1 * (1 - lambda_var)
    
    # Big-M constraints for y > z
    M2 = max(0.0, (z_threshold + epsilon) - y_lb)
    subprob += (z_threshold + epsilon) - y_var <= M2 * (1 - lambda_var)
    
    # Solve subproblem
    subprob.solve(PULP_CBC_CMD(msg=0))
    
    # Check if solution found
    if subprob.status == pulp.LpStatusOptimal:
        return pulp.value(lambda_var), pulp.value(y_var), True
    else:
        return 0, None, False

def dantzig_wolfe_count(df_y, df_x, df_z, obj='max', epsilon=1e-6, relax=False):
    """
    Implement Dantzig-Wolfe decomposition for COUNT(*) query.
    
    Parameters:
        df_y: DataFrame with zonotope representation of y column
        df_x: DataFrame with center values for x column
        df_z: DataFrame with center values for z column
        obj: 'min' or 'max' for the objective
        epsilon: Small positive constant
        relax: Whether to relax binary variables to continuous
        
    Returns:
        count_value: Optimal count value
        processing_time: Time taken for processing
    """
    start_time = time.time()
    
    # Initialize count and tracking variables
    count_value = 0
    definite_in = 0
    definite_out = 0
    uncertain = 0
    
    # Process each row
    for idx, row in df_y.iterrows():
        x_threshold = df_x.iloc[idx]["center"]
        z_threshold = df_z.iloc[idx]["center"]
        
        # Get bounds for current row
        y_lb, y_ub = bound(row)
        
        # Case 1: Definitely In
        if y_ub < x_threshold - epsilon and y_lb > z_threshold + epsilon:
            count_value += 1
            definite_in += 1
            continue
            
        # Case 2: Definitely Out
        if y_lb >= x_threshold - epsilon or y_ub <= z_threshold + epsilon:
            definite_out += 1
            continue
            
        # Case 3: Uncertain - need to solve subproblem
        uncertain += 1
        lambda_val, _, _ = solve_subproblem(
            idx, row, x_threshold, z_threshold, epsilon, obj, relax
        )
        
        # Add to count if row satisfies condition (or partially for relaxed case)
        count_value += lambda_val
    
    processing_time = time.time() - start_time
    
    # Only print detailed stats when not in experiment mode
    if len(df_y) < 5000:
        print(f"Rows processed: {len(df_y)}")
        print(f"  Definitely satisfying: {definite_in}")
        print(f"  Definitely not satisfying: {definite_out}")
        print(f"  Uncertain (requiring subproblem): {uncertain}")
    
    return count_value, processing_time

def load_data_slice(num_groups, full_size, slice_size):
    """
    Load a slice of data from parquet files.
    
    Parameters:
        num_groups: Number of error groups
        full_size: Size of the full dataset
        slice_size: Number of rows to include in the slice
        
    Returns:
        df_x, df_y, df_z: DataFrames for x, y, z columns (sliced)
    """
    # File paths
    file_x = f"./Data/COUNT/count_x_{num_groups}_groups_size_{full_size}.parquet"
    file_y = f"./Data/COUNT/count_y_{num_groups}_groups_size_{full_size}.parquet"
    file_z = f"./Data/COUNT/count_z_{num_groups}_groups_size_{full_size}.parquet"
    
    try:
        df_x = pd.read_parquet(file_x)
        df_y = pd.read_parquet(file_y)
        df_z = pd.read_parquet(file_z)
        
        # Take only the first slice_size rows
        df_x = df_x.head(slice_size)
        df_y = df_y.head(slice_size)
        df_z = df_z.head(slice_size)
        
        return df_x, df_y, df_z
    except Exception as e:
        print(f"Error reading or slicing files: {e}")
        return None, None, None

def run_experiments(sizes, num_groups=1, full_size=10000, epsilon=1e-6):
    """
    Run experiments for different data sizes and methods using slices of real data.
    
    Parameters:
        sizes: List of data sizes to test
        num_groups: Number of error groups
        full_size: Size of the full dataset
        epsilon: Small positive constant
        
    Returns:
        results: Dictionary containing experiment results
    """
    # Import necessary functions
    from count_zono import compute_count_with_constraint, compute_count_with_constraint_tight
    
    results = {
        'sizes': sizes,
        'times': {
            'original_true': [],  # Original method with relax=False
            'original_false': [], # Original method with relax=True
            'tight_true': [],     # Tight method with relax=False
            'tight_false': [],    # Tight method with relax=True
            'dw_true': [],        # Dantzig-Wolfe with relax=False
            'dw_false': []        # Dantzig-Wolfe with relax=True
        },
        'intervals': {
            'original_true': [],
            'original_false': [],
            'tight_true': [],
            'tight_false': [],
            'dw_true': [],
            'dw_false': []
        }
    }
    
    for size in sizes:
        print(f"\nRunning experiments for size {size}...")
        
        # Load slice of real data
        df_x, df_y, df_z = load_data_slice(num_groups, full_size, size)
        
        if df_x is None or df_y is None or df_z is None:
            print(f"Skipping size {size} due to data loading error.")
            continue
        
        print(f"Data loaded successfully with {len(df_x)} rows.")
        
        # 1. Original method with relax=False
        print("  Testing original method (relax=False)...")
        try:
            start = time.time()
            status_min, count_min, _ = compute_count_with_constraint(df_y, df_x, df_z, 'min', epsilon, relax=False)
            status_max, count_max, _ = compute_count_with_constraint(df_y, df_x, df_z, 'max', epsilon, relax=False)
            elapsed = time.time() - start
            results['times']['original_true'].append(elapsed)
            results['intervals']['original_true'].append(count_max - count_min)
            print(f"    Completed in {elapsed:.2f} seconds. Interval: {count_max - count_min}")
        except Exception as e:
            print(f"    Error: {e}")
            results['times']['original_true'].append(None)
            results['intervals']['original_true'].append(None)
        
        # 2. Original method with relax=True
        print("  Testing original method (relax=True)...")
        try:
            start = time.time()
            status_min, count_min, _ = compute_count_with_constraint(df_y, df_x, df_z, 'min', epsilon, relax=True)
            status_max, count_max, _ = compute_count_with_constraint(df_y, df_x, df_z, 'max', epsilon, relax=True)
            elapsed = time.time() - start
            results['times']['original_false'].append(elapsed)
            results['intervals']['original_false'].append(count_max - count_min)
            print(f"    Completed in {elapsed:.2f} seconds. Interval: {count_max - count_min}")
        except Exception as e:
            print(f"    Error: {e}")
            results['times']['original_false'].append(None)
            results['intervals']['original_false'].append(None)
        
        # 3. Tight method with relax=False
        print("  Testing tight method (relax=False)...")
        try:
            start = time.time()
            status_min, count_min, _ = compute_count_with_constraint_tight(df_y, df_x, df_z, 'min', epsilon, relax=False)
            status_max, count_max, _ = compute_count_with_constraint_tight(df_y, df_x, df_z, 'max', epsilon, relax=False)
            elapsed = time.time() - start
            results['times']['tight_true'].append(elapsed)
            results['intervals']['tight_true'].append(count_max - count_min)
            print(f"    Completed in {elapsed:.2f} seconds. Interval: {count_max - count_min}")
        except Exception as e:
            print(f"    Error: {e}")
            results['times']['tight_true'].append(None)
            results['intervals']['tight_true'].append(None)
        
        # 4. Tight method with relax=True
        print("  Testing tight method (relax=True)...")
        try:
            start = time.time()
            status_min, count_min, _ = compute_count_with_constraint_tight(df_y, df_x, df_z, 'min', epsilon, relax=True)
            status_max, count_max, _ = compute_count_with_constraint_tight(df_y, df_x, df_z, 'max', epsilon, relax=True)
            elapsed = time.time() - start
            results['times']['tight_false'].append(elapsed)
            results['intervals']['tight_false'].append(count_max - count_min)
            print(f"    Completed in {elapsed:.2f} seconds. Interval: {count_max - count_min}")
        except Exception as e:
            print(f"    Error: {e}")
            results['times']['tight_false'].append(None)
            results['intervals']['tight_false'].append(None)
        
        # 5. Dantzig-Wolfe with relax=False
        print("  Testing Dantzig-Wolfe (relax=False)...")
        try:
            start = time.time()
            count_min, _ = dantzig_wolfe_count(df_y, df_x, df_z, 'min', epsilon, relax=False)
            count_max, _ = dantzig_wolfe_count(df_y, df_x, df_z, 'max', epsilon, relax=False)
            elapsed = time.time() - start
            results['times']['dw_true'].append(elapsed)
            results['intervals']['dw_true'].append(count_max - count_min)
            print(f"    Completed in {elapsed:.2f} seconds. Interval: {count_max - count_min}")
        except Exception as e:
            print(f"    Error: {e}")
            results['times']['dw_true'].append(None)
            results['intervals']['dw_true'].append(None)
        
        # 6. Dantzig-Wolfe with relax=True
        print("  Testing Dantzig-Wolfe (relax=True)...")
        try:
            start = time.time()
            count_min, _ = dantzig_wolfe_count(df_y, df_x, df_z, 'min', epsilon, relax=True)
            count_max, _ = dantzig_wolfe_count(df_y, df_x, df_z, 'max', epsilon, relax=True)
            elapsed = time.time() - start
            results['times']['dw_false'].append(elapsed)
            results['intervals']['dw_false'].append(count_max - count_min)
            print(f"    Completed in {elapsed:.2f} seconds. Interval: {count_max - count_min}")
        except Exception as e:
            print(f"    Error: {e}")
            results['times']['dw_false'].append(None)
            results['intervals']['dw_false'].append(None)
    
    return results

def plot_results(results, output_dir="./plots"):
    """
    Plot experiment results.
    
    Parameters:
        results: Dictionary containing experiment results
        output_dir: Directory to save plots
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('ggplot')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'x', '+']
    
    # Method names for legend
    method_names = {
        'original_true': 'Original MILP (Binary)',
        'original_false': 'Original MILP (Relaxed)',
        'tight_true': 'Tight MILP (Binary)',
        'tight_false': 'Tight MILP (Relaxed)',
        'dw_true': 'Dantzig-Wolfe (Binary)',
        'dw_false': 'Dantzig-Wolfe (Relaxed)'
    }
    
    # Filter out None values from results
    valid_sizes = results['sizes']
    valid_methods = {}
    
    for method in results['times']:
        valid_indices = [i for i, t in enumerate(results['times'][method]) if t is not None]
        if valid_indices:
            valid_methods[method] = {
                'sizes': [valid_sizes[i] for i in valid_indices],
                'times': [results['times'][method][i] for i in valid_indices],
                'intervals': [results['intervals'][method][i] for i in valid_indices]
            }
    
    # 1. Plot processing times
    plt.figure(figsize=(12, 8))
    for i, method in enumerate(valid_methods):
        plt.plot(valid_methods[method]['sizes'], valid_methods[method]['times'], 
                 marker=markers[i], color=colors[i], 
                 label=method_names[method], linewidth=2, markersize=8)
    
    plt.yscale('log')
    plt.xlabel('Data Size (Number of Rows)', fontsize=14)
    plt.ylabel('Processing Time (seconds)', fontsize=14)
    plt.title('Processing Time Comparison for Different Methods', fontsize=16)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/processing_time_comparison.png", dpi=300, bbox_inches='tight')
    
    # 2. Plot interval widths (count_max - count_min)
    plt.figure(figsize=(12, 8))
    for i, method in enumerate(valid_methods):
        plt.plot(valid_methods[method]['sizes'], valid_methods[method]['intervals'], 
                 marker=markers[i], color=colors[i], 
                 label=method_names[method], linewidth=2, markersize=8)
    
    plt.xlabel('Data Size (Number of Rows)', fontsize=14)
    plt.ylabel('Interval Width (count_max - count_min)', fontsize=14)
    plt.title('Interval Width Comparison for Different Methods', fontsize=16)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/interval_width_comparison.png", dpi=300, bbox_inches='tight')
    
    # 3. Plot speedup relative to original method
    plt.figure(figsize=(12, 8))
    reference_method = None
    
    # Find a valid reference method (prioritize original_true)
    if 'original_true' in valid_methods:
        reference_method = 'original_true'
    elif valid_methods:
        reference_method = list(valid_methods.keys())[0]
    
    if reference_method:
        ref_sizes = valid_methods[reference_method]['sizes']
        ref_times = valid_methods[reference_method]['times']
        
        for i, method in enumerate(valid_methods):
            if method != reference_method:
                # Find common sizes
                common_sizes = []
                speedups = []
                
                for j, size in enumerate(valid_methods[method]['sizes']):
                    if size in ref_sizes:
                        ref_idx = ref_sizes.index(size)
                        common_sizes.append(size)
                        speedups.append(ref_times[ref_idx] / valid_methods[method]['times'][j])
                
                if common_sizes:
                    plt.plot(common_sizes, speedups, marker=markers[i], color=colors[i], 
                             label=method_names[method], linewidth=2, markersize=8)
        
        plt.xlabel('Data Size (Number of Rows)', fontsize=14)
        plt.ylabel(f'Speedup (relative to {method_names[reference_method]})', fontsize=14)
        plt.title('Performance Speedup Comparison', fontsize=16)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.axhline(y=1, color='k', linestyle='--', alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/speedup_comparison.png", dpi=300, bbox_inches='tight')
    
    print(f"Plots saved to {output_dir}/")

def main():
    """Main function to run experiments with real data"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run COUNT(*) query experiments with Dantzig-Wolfe decomposition')
    parser.add_argument('--sizes', type=str, default='2000,4000,6000,8000,10000', 
                        help='Comma-separated list of data sizes for experiments')
    parser.add_argument('--full-size', type=int, default=10000, 
                        help='Size of the full dataset to load')
    parser.add_argument('--groups', type=int, default=1, 
                        help='Number of error groups in the dataset')
    parser.add_argument('--output-dir', type=str, default='./plots', 
                        help='Directory to save plots')
    args = parser.parse_args()
    
    # Parse sizes
    sizes = [int(s) for s in args.sizes.split(',')]
    print(f"Running experiments for sizes: {sizes}")
    
    # Run experiments with real data
    results = run_experiments(sizes, args.groups, args.full_size)
    
    # Generate plots
    plot_results(results, args.output_dir)

if __name__ == "__main__":
    main()