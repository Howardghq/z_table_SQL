import pandas as pd
import pulp
import numpy as np
import time
from matplotlib import pyplot as plt
from count_zono import bound, compute_count_with_constraint, compute_count_with_constraint_tight
from count_interval import count_filtered_rows

def compute_count_with_iterative_independence(df_y, df_x, df_z, obj='min', epsilon=1e-6, batch_size=100, max_iterations=10, relax=False):
    # Find all unique position indices
    position_indices = set()
    for idx, row in df_y.iterrows():
        i = 1
        while f"g{i}" in row and f"idx{i}" in row and not pd.isna(row[f"g{i}"]):
            position_indices.add(int(row[f"idx{i}"]))
            i += 1
    
    # Sort position indices
    position_indices = sorted(list(position_indices))
    
    # Limit iterations to available positions
    max_possible_iterations = (len(position_indices) + batch_size - 1) // batch_size  # Ceiling division
    max_iterations = min(max_iterations, max_possible_iterations)
    results = []
    
    # Record overall start time
    total_start_time = time.time()
    
    for iteration in range(max_iterations + 1):  # +1 for the initial independence assumption
        # Record iteration start time
        iter_start_time = time.time()
        
        # In iteration i, positions 1 to i are constrained to be the same across rows
        num_constrained = min(iteration * batch_size, len(position_indices))
        constrained_positions = position_indices[:num_constrained]
        
        # Set up the optimization problem
        sense = pulp.LpMinimize if obj == 'min' else pulp.LpMaximize
        prob = pulp.LpProblem(f"Count_Rows_{obj}_Iter_{iteration}", sense)
        
        total_count = 0
        i_vars = {}  # Binary indicator variables for each row
        
        # Create shared error variables for constrained positions
        shared_error_vars = {}
        for pos in constrained_positions:
            shared_error_vars[pos] = pulp.LpVariable(f"e_{pos}", 
                                                    lowBound=-1, upBound=1, 
                                                    cat='Continuous')
        
        # Row-specific error variables dictionary (for unconstrained positions)
        row_error_vars = {}
        
        # Record problem setup time
        setup_time = time.time() - iter_start_time
        
        # Process each row
        processing_start = time.time()
        for idx, row in df_y.iterrows():
            x_thr = df_x.iloc[idx]['center']
            z_thr = df_z.iloc[idx]['center']
            c = row['center']
            y_lb, y_ub = bound(row)
            
            # Case 1: Definitely In (y_ub < x and y_lb > z)
            if y_ub < x_thr and y_lb > z_thr:
                total_count += 1
                continue
            
            # Case 2: Definitely Out (y_lb >= x or y_ub <= z)
            if y_lb >= x_thr or y_ub <= z_thr:
                continue
            
            # Case 3: Uncertain - create indicator variable
            i_var = pulp.LpVariable(f"i_{idx}", 
                                   lowBound=0, upBound=1, 
                                   cat='Continuous' if relax else 'Binary')
            i_vars[idx] = i_var
            total_count += i_var
            
            # Gather all error terms from this row
            error_terms = []
            k = 1
            while f"g{k}" in row and f"idx{k}" in row and not pd.isna(row[f"g{k}"]):
                coeff = row[f"g{k}"]
                key = int(row[f"idx{k}"])
                pos = k  # Position index (1, 2, 3, ...)
                error_terms.append((key, coeff))
                k += 1
            
            # Separate constrained and unconstrained error terms
            constrained_terms = [(key, coeff) for key, coeff in error_terms if key in constrained_positions]
            unconstrained_terms = [(key, coeff) for key, coeff in error_terms if key not in constrained_positions]
            
            # Create row-specific error variables for unconstrained positions
            for key, _ in unconstrained_terms:
                if (idx, key) not in row_error_vars:
                    var_name = f"e_{idx}_{key}"
                    row_error_vars[(idx, key)] = pulp.LpVariable(var_name, 
                                                             lowBound=-1, upBound=1, 
                                                             cat='Continuous')
            
            # Build y_expr with zonotope representation
            y_expr = c
            
            # Add constrained error terms (shared across rows by position)
            for key, coeff in constrained_terms:
                y_expr += coeff * shared_error_vars[key]
            
            # Add unconstrained error terms (specific to this row)
            for key, coeff in unconstrained_terms:
                y_expr += coeff * row_error_vars[(idx, key)]
            
            # Calculate M1 and M2 for Big-M constraints
            
            # M1 for constraint: y <= x - epsilon + M1 * (1 - lambda)
            M1 = max(0.0, y_ub - (x_thr - epsilon))
            
            # M2 for constraint: y >= z + epsilon - M2 * (1 - lambda)
            M2 = max(0.0, (z_thr + epsilon) - y_lb)
            
            # Big-M constraints
            prob += y_expr <= x_thr - epsilon + M1 * (1 - i_var)
            prob += y_expr >= z_thr + epsilon - M2 * (1 - i_var)
        
        processing_time = time.time() - processing_start
        
        # Objective: min or max the count
        prob += total_count
        
        # Solve the problem
        solve_start = time.time()
        solver = pulp.PULP_CBC_CMD(msg=0)
        prob.solve(solver)
        solve_time = time.time() - solve_start
        
        # Record total iteration time
        iter_total_time = time.time() - iter_start_time
        
        # Record results
        status = pulp.LpStatus[prob.status]
        count_val = pulp.value(prob.objective)
        lambda_vals = {idx: pulp.value(v) for idx, v in i_vars.items()}
        
        # Track constrained positions and number of variables
        results.append({
            'iteration': iteration,
            'status': status,
            'count': count_val,
            'num_constrained': num_constrained,
            'num_variables': len(prob.variables()),
            'shared_vars': len(shared_error_vars),
            'row_specific_vars': len(row_error_vars),
            # Time breakdowns
            'setup_time': setup_time,
            'processing_time': processing_time,
            'solve_time': solve_time,
            'iter_total_time': iter_total_time,
            'cumulative_time': time.time() - total_start_time
        })
        
        # If all positions are constrained or max iterations reached, stop
        if len(constrained_positions) >= len(position_indices) or iteration >= max_iterations:
            break
    
    # Return the final result
    final_result = results[-1]
    return final_result['status'], final_result['count'], lambda_vals, results

def compute_count_with_iterative_reduction(df_y, df_x, df_z, obj='min', epsilon=1e-6, batch_size=100, max_iterations=10, relax=False):
    # Find all unique position indices
    position_indices = set()
    for idx, row in df_y.iterrows():
        i = 1
        while f"g{i}" in row and f"idx{i}" in row and not pd.isna(row[f"g{i}"]):
            position_indices.add(int(row[f"idx{i}"]))
            i += 1
    
    # Sort position indices
    position_indices = sorted(list(position_indices))
    
    # Limit iterations to available positions
    max_possible_iterations = (len(position_indices) + batch_size - 1) // batch_size  # Ceiling division
    max_iterations = min(max_iterations, max_possible_iterations)
    results = []

    for iteration in range(max_iterations + 1):  # +1 for the initial order reduction
        # In iteration i, indices 0 to i-1 are extracted
        extracted_indices = position_indices[:min(iteration * batch_size, len(position_indices))]
        
        # Set up the optimization problem
        sense = pulp.LpMinimize if obj == 'min' else pulp.LpMaximize
        prob = pulp.LpProblem(f"Count_Rows_{obj}_Iter_{iteration}", sense)
        
        total_count = 0
        i_vars = {}  # Binary indicator variables
        e_vars = {}  # Individual error variables
        
        # Create variables for extracted error terms
        for idx in extracted_indices:
            e_vars[idx] = pulp.LpVariable(f"e_{idx}", lowBound=-1, upBound=1, cat='Continuous')
        
        # Create common error variable if we have unextracted indices
        e_common = None
        if len(extracted_indices) < len(position_indices):
            e_common = pulp.LpVariable("e_common", lowBound=-1, upBound=1, cat='Continuous')
        
        # Process each row
        for idx, row in df_y.iterrows():
            x_thr = df_x.iloc[idx]['center']
            z_thr = df_z.iloc[idx]['center']
            c = row['center']
            y_lb, y_ub = bound(row)
            
            # Case 1: Definitely In (y_ub < x and y_lb > z)
            if y_ub < x_thr and y_lb > z_thr:
                total_count += 1
                continue
            
            # Case 2: Definitely Out (y_lb >= x or y_ub <= z)
            if y_lb >= x_thr or y_ub <= z_thr:
                continue
            
            # Case 3: Uncertain - create indicator variable
            i_var = pulp.LpVariable(f"i_{idx}", 
                                   lowBound=0, upBound=1, 
                                   cat='Continuous' if relax else 'Binary')
            i_vars[idx] = i_var
            total_count += i_var
            
            # Collect coefficients for extracted and remaining error terms
            extracted_coefs = {}
            remaining_coefs = []

            k = 1
            while f"g{k}" in row and f"idx{k}" in row and not pd.isna(row[f"g{k}"]):
                coeff = row[f"g{k}"]
                error_idx = int(row[f"idx{k}"])
                
                if error_idx in extracted_indices:
                    extracted_coefs[error_idx] = coeff
                else:
                    remaining_coefs.append(abs(coeff))
                
                k += 1

            # Build y_expr with zonope representation
            y_expr = c
            
            # Add extracted terms with their original coefficients
            for error_idx, coeff in extracted_coefs.items():
                y_expr += coeff * e_vars[error_idx]
            
            # Add common term for remaining error variables (order reduction)
            if e_common is not None and remaining_coefs:
                common_coeff = sum(remaining_coefs)
                y_expr += common_coeff * e_common
            
            # Calculate M1 and M2 for Big-M constraints
            
            # M1 for constraint: y_expr <= x_thr - epsilon + M1 * (1 - i_var)
            M1 = max(0.0, y_ub - (x_thr - epsilon))
            
            # M2 for constraint: y_expr >= z_thr + epsilon - M2 * (1 - i_var)
            M2 = max(0.0, (z_thr + epsilon) - y_lb)
            
            # Big-M constraints
            prob += y_expr <= x_thr - epsilon + M1 * (1 - i_var)
            prob += y_expr >= z_thr + epsilon - M2 * (1 - i_var)
        
        # Objective: min or max the count
        prob += total_count
        
        # Solve the problem
        solver = pulp.PULP_CBC_CMD(msg=0)
        prob.solve(solver)
        
        # Record results
        status = pulp.LpStatus[prob.status]
        count_val = pulp.value(prob.objective)
        lambda_vals = {idx: pulp.value(v) for idx, v in i_vars.items()}
        
        results.append({
            'iteration': iteration,
            'status': status,
            'count': count_val,
            'extracted_indices': list(extracted_indices)
        })
        
        # If all indices are extracted or max iterations reached, stop
        if len(extracted_indices) >= len(position_indices) or iteration >= max_iterations:
            break
    
    # Return the final result
    final_result = results[-1]
    return final_result['status'], final_result['count'], lambda_vals, results

def main():
    # File paths (adjust these paths as needed)
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
    
    # Set parameters
    max_iterations = 10
    batch_size = 100
    relax = True  # Use relaxed variables for faster computation
    
    # Data structures to store results from all methods
    all_methods_data = {}
    
    # ==========================================================================
    # Method 1: Partition-based frequency approach
    # ==========================================================================
    print(f"\nRunning Partition-based method (batch size = {batch_size}, max iterations = {max_iterations})...")
    
    start_time = time.time()
    
    # Min (lower bound)
    status_min, count_min, lambda_min, results_min = compute_count_with_iterative_independence(
        df_y, df_x, df_z, 'min', batch_size=batch_size, max_iterations=max_iterations, relax=relax
    )
    
    # Max (upper bound)
    status_max, count_max, lambda_max, results_max = compute_count_with_iterative_independence(
        df_y, df_x, df_z, 'max', batch_size=batch_size, max_iterations=max_iterations, relax=relax
    )
    
    batch_time = time.time() - start_time
    print(f"Partition method execution time: {batch_time:.2f} seconds")
    
    # Prepare data for the Partition method
    iterations = []
    lower_bounds = []
    upper_bounds = []
    interval_lengths = []
    num_constrained = []
    
    for i in range(len(results_min)):
        if i < len(results_max):  # Ensure we have both min and max results
            iterations.append(i)
            lower_bounds.append(results_min[i]['count'])
            upper_bounds.append(results_max[i]['count'])
            interval_lengths.append(results_max[i]['count'] - results_min[i]['count'])
            num_constrained.append(results_min[i]['num_constrained'])
    
    # Store batch method results
    all_methods_data['Partition Method'] = {
        'iterations': iterations,
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'interval_lengths': interval_lengths,
        'num_constrained': num_constrained,
        'time': batch_time,
        'color': 'blue'
    }
    
    # ==========================================================================
    # Method 2: Order reduction approach
    # ==========================================================================
    print(f"\nRunning order reduction method (max iterations = {max_iterations})...")
    
    start_time = time.time()
    
    # If original function uses different parameters, adjust as needed
    redux_max_iterations = max_iterations
    
    # Min (lower bound)
    status_min, count_min, lambda_min, results_min = compute_count_with_iterative_reduction(
        df_y, df_x, df_z, 'min', batch_size=batch_size, max_iterations=redux_max_iterations, relax=relax
    )
    
    # Max (upper bound)
    status_max, count_max, lambda_max, results_max = compute_count_with_iterative_reduction(
        df_y, df_x, df_z, 'max', batch_size=batch_size, max_iterations=redux_max_iterations, relax=relax
    )
    
    redux_time = time.time() - start_time
    print(f"Order reduction method execution time: {redux_time:.2f} seconds")
    
    # Prepare data for the order reduction method
    redux_iterations = []
    redux_lower_bounds = []
    redux_upper_bounds = []
    redux_interval_lengths = []
    redux_num_constrained = []
    
    for i in range(len(results_min)):
        if i < len(results_max):  # Ensure we have both min and max results
            redux_iterations.append(i)
            redux_lower_bounds.append(results_min[i]['count'])
            redux_upper_bounds.append(results_max[i]['count'])
            redux_interval_lengths.append(results_max[i]['count'] - results_min[i]['count'])
            # Number of extracted indices represents constrained variables
            redux_num_constrained.append(len(results_min[i]['extracted_indices']))
    
    # Store order reduction method results
    all_methods_data['Order Reduction'] = {
        'iterations': redux_iterations,
        'lower_bounds': redux_lower_bounds,
        'upper_bounds': redux_upper_bounds,
        'interval_lengths': redux_interval_lengths,
        'num_constrained': redux_num_constrained,
        'time': redux_time,
        'color': 'green'
    }
    
    
    # ==========================================================================
    # Method 3: Original methods
    # ==========================================================================
    print("\nRunning original methods for comparison...")
    
    # Original MILP
    start_time = time.time()
    status_min, count_min, _ = compute_count_with_constraint(df_y, df_x, df_z, 'min', relax=relax)
    status_max, count_max, _ = compute_count_with_constraint(df_y, df_x, df_z, 'max', relax=relax)
    original_time = time.time() - start_time
    
    print(f"Original MILP: [{int(count_min)}, {int(count_max)}], length = {int(count_max - count_min)}")
    print(f"Original MILP execution time: {original_time:.2f} seconds")
    
    # Add as reference lines in plots
    all_methods_data['Original MILP'] = {
        'lower_bound': count_min,
        'upper_bound': count_max,
        'interval_length': count_max - count_min,
        'time': original_time,
        'color': 'red'
    }
    
    # Tight constraints
    start_time = time.time()
    status_min, count_min, _ = compute_count_with_constraint_tight(df_y, df_x, df_z, 'min', relax=relax)
    status_max, count_max, _ = compute_count_with_constraint_tight(df_y, df_x, df_z, 'max', relax=relax)
    tight_time = time.time() - start_time
    
    print(f"Tight constraints: [{int(count_min)}, {int(count_max)}], length = {int(count_max - count_min)}")
    print(f"Tight constraints execution time: {tight_time:.2f} seconds")
    
    # Add as reference lines in plots
    all_methods_data['Tight Constraints'] = {
        'lower_bound': count_min,
        'upper_bound': count_max,
        'interval_length': count_max - count_min,
        'time': tight_time,
        'color': 'purple'
    }
    
    # ==========================================================================
    # Visualizations: Compare all methods
    # ==========================================================================
    print("\nGenerating comparative visualizations...")
    
    # Plot 1: Interval Length vs Iteration for all iterative methods
    plt.figure(figsize=(12, 6))
    
    # Plot iterative methods
    for method_name, data in all_methods_data.items():
        if 'interval_lengths' in data:  # Only plot iterative methods here
            plt.plot(data['iterations'], data['interval_lengths'], 'o-', 
                     color=data['color'], linewidth=2, markersize=6,
                     label=f"{method_name} (time: {data['time']:.2f}s)")
    
    # Add reference lines for non-iterative methods
    for method_name, data in all_methods_data.items():
        if 'interval_length' in data:  # Non-iterative methods
            plt.axhline(y=data['interval_length'], linestyle='--', 
                        color=data['color'], linewidth=2,
                        label=f"{method_name} (time: {data['time']:.2f}s)")
    
    plt.xlabel('Iteration')
    plt.ylabel('Interval Length (Upper - Lower)')
    plt.title('Comparison of Interval Length Convergence Across Methods')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('interval_comparison.png', dpi=300)
    print(f"Plot saved as 'interval_comparison.png'")
    
    # Plot 2: Upper and Lower Bounds vs Iteration for all methods
    plt.figure(figsize=(12, 6))
    
    # Plot bounds for iterative methods
    for method_name, data in all_methods_data.items():
        if 'upper_bounds' in data:  # Only plot iterative methods
            plt.plot(data['iterations'], data['upper_bounds'], '^-', 
                     color=data['color'], linewidth=2, markersize=6,
                     label=f"{method_name} Upper Bound")
            plt.plot(data['iterations'], data['lower_bounds'], 'v-', 
                     color=data['color'], linewidth=2, markersize=6,
                     label=f"{method_name} Lower Bound")
            plt.fill_between(data['iterations'], data['lower_bounds'], data['upper_bounds'], 
                            color=data['color'], alpha=0.1)
    
    # Add reference lines for non-iterative methods
    for method_name, data in all_methods_data.items():
        if 'upper_bound' in data:  # Non-iterative methods
            plt.axhline(y=data['upper_bound'], linestyle='--', 
                        color=data['color'], linewidth=2,
                        label=f"{method_name} Upper Bound")
            plt.axhline(y=data['lower_bound'], linestyle=':', 
                        color=data['color'], linewidth=2,
                        label=f"{method_name} Lower Bound")
    
    plt.xlabel('Iteration')
    plt.ylabel('COUNT(*)')
    plt.title('Upper and Lower Bounds Across Different Methods')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('bounds_comparison.png', dpi=300)
    print(f"Plot saved as 'bounds_comparison.png'")
    
    # Plot 3: Interval Length vs Constrained Variables for iterative methods
    plt.figure(figsize=(12, 6))
    
    for method_name, data in all_methods_data.items():
        if 'num_constrained' in data:  # Only plot iterative methods
            plt.plot(data['num_constrained'], data['interval_lengths'], 'o-', 
                     color=data['color'], linewidth=2, markersize=6,
                     label=f"{method_name}")
    
    # Add reference lines for non-iterative methods
    for method_name, data in all_methods_data.items():
        if 'interval_length' in data:  # Non-iterative methods
            plt.axhline(y=data['interval_length'], linestyle='--', 
                        color=data['color'], linewidth=2,
                        label=f"{method_name}")
    
    plt.xlabel('Number of Constrained Variables')
    plt.ylabel('Interval Length (Upper - Lower)')
    plt.title('Effect of Constrained Variables on Bound Tightness')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('variables_effect_comparison.png', dpi=300)
    print(f"Plot saved as 'variables_effect_comparison.png'")
    
    # Print final summary
    print("\nFinal Results Summary:")
    print("-" * 80)
    print(f"{'Method':<20} {'Lower Bound':<15} {'Upper Bound':<15} {'Interval':<15} {'Time (s)':<10}")
    print("-" * 80)
    
    # Print iterative methods' final results
    for method_name, data in all_methods_data.items():
        if 'interval_lengths' in data:  # Iterative methods
            last_idx = len(data['interval_lengths']) - 1
            print(f"{method_name:<20} {data['lower_bounds'][last_idx]:<15.2f} "
                  f"{data['upper_bounds'][last_idx]:<15.2f} "
                  f"{data['interval_lengths'][last_idx]:<15.2f} "
                  f"{data['time']:<10.2f}")
    
    # Print non-iterative methods' results
    for method_name, data in all_methods_data.items():
        if 'interval_length' in data:  # Non-iterative methods
            print(f"{method_name:<20} {data['lower_bound']:<15.2f} "
                  f"{data['upper_bound']:<15.2f} "
                  f"{data['interval_length']:<15.2f} "
                  f"{data['time']:<10.2f}")
    
    print("-" * 80)
    
if __name__ == "__main__":
    main()