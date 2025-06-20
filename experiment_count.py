import pandas as pd
import pulp
import numpy as np
import time
from matplotlib import pyplot as plt
from count_zono import bound, compute_count_with_constraint, compute_count_with_constraint_tight
from count_zono_iter import compute_count_with_iterative_independence, compute_count_with_iterative_reduction
from count_interval import count_filtered_rows

def main():
    # File paths (adjust these paths as needed)
    num_groups = 1
    size = 1000
    relation_degree = 1
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
    # Method 0: Count filtered rows (baseline)
    # ==========================================================================
    print("\nRunning baseline count filtered rows...")
    start_time = time.time()

    interval = count_filtered_rows(df_x, df_y, df_z)
    count_min = interval.lb
    count_max = interval.ub
    count_length = count_max - count_min
    interval_time = time.time() - start_time
    print(f"Count filtered rows: [{int(count_min)}, {int(count_max)}], length = {int(count_length)}")
    print(f"Interval method execution time: {interval_time:.2f} seconds")
    # Add as reference lines in plots
    all_methods_data['Interval Based Method'] = {
        'lower_bound': count_min,
        'upper_bound': count_max,
        'interval_length': count_max - count_min,
        'time': interval_time,
        'color': 'black'
    }

    
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
    
    # Time analysis data
    cumulative_times_min = []
    cumulative_times_max = []
    iter_total_times_min = []
    iter_total_times_max = []
    setup_times_min = []
    setup_times_max = []
    processing_times_min = []
    processing_times_max = []
    solve_times_min = []
    solve_times_max = []
    
    for i in range(len(results_min)):
        if i < len(results_max):  # Ensure we have both min and max results
            iterations.append(i)
            lower_bounds.append(results_min[i]['count'])
            upper_bounds.append(results_max[i]['count'])
            interval_lengths.append(results_max[i]['count'] - results_min[i]['count'])
            num_constrained.append(results_min[i]['num_constrained'])
            
            # Collect time data
            cumulative_times_min.append(results_min[i]['cumulative_time'])
            cumulative_times_max.append(results_max[i]['cumulative_time'])
            iter_total_times_min.append(results_min[i]['iter_total_time'])
            iter_total_times_max.append(results_max[i]['iter_total_time'])
            setup_times_min.append(results_min[i]['setup_time'])
            setup_times_max.append(results_max[i]['setup_time'])
            processing_times_min.append(results_min[i]['processing_time'])
            processing_times_max.append(results_max[i]['processing_time'])
            solve_times_min.append(results_min[i]['solve_time'])
            solve_times_max.append(results_max[i]['solve_time'])
    
    # Store batch method results
    all_methods_data['Partition Method'] = {
        'iterations': iterations,
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'interval_lengths': interval_lengths,
        'num_constrained': num_constrained,
        'time': batch_time,
        'color': 'blue',
        # Add time breakdown data
        'cumulative_times_min': cumulative_times_min,
        'cumulative_times_max': cumulative_times_max,
        'iter_total_times_min': iter_total_times_min,
        'iter_total_times_max': iter_total_times_max,
        'setup_times_min': setup_times_min,
        'setup_times_max': setup_times_max,
        'processing_times_min': processing_times_min,
        'processing_times_max': processing_times_max,
        'solve_times_min': solve_times_min,
        'solve_times_max': solve_times_max
    }
    
    # ==========================================================================
    # Method 2: Order reduction approach
    # ==========================================================================
    # print(f"\nRunning order reduction method (max iterations = {max_iterations})...")
    
    # start_time = time.time()
    
    # # If original function uses different parameters, adjust as needed
    # redux_max_iterations = max_iterations
    
    # # Min (lower bound)
    # status_min, count_min, lambda_min, results_min = compute_count_with_iterative_reduction(
    #     df_y, df_x, df_z, 'min', batch_size=batch_size, max_iterations=redux_max_iterations, relax=relax
    # )
    
    # # Max (upper bound)
    # status_max, count_max, lambda_max, results_max = compute_count_with_iterative_reduction(
    #     df_y, df_x, df_z, 'max', batch_size=batch_size, max_iterations=redux_max_iterations, relax=relax
    # )
    
    # redux_time = time.time() - start_time
    # print(f"Order reduction method execution time: {redux_time:.2f} seconds")
    
    # # Prepare data for the order reduction method
    # redux_iterations = []
    # redux_lower_bounds = []
    # redux_upper_bounds = []
    # redux_interval_lengths = []
    # redux_num_constrained = []
    
    # for i in range(len(results_min)):
    #     if i < len(results_max):  # Ensure we have both min and max results
    #         redux_iterations.append(i)
    #         redux_lower_bounds.append(results_min[i]['count'])
    #         redux_upper_bounds.append(results_max[i]['count'])
    #         redux_interval_lengths.append(results_max[i]['count'] - results_min[i]['count'])
    #         # Number of extracted indices represents constrained variables
    #         redux_num_constrained.append(len(results_min[i]['extracted_indices']))
    
    # # Store order reduction method results
    # all_methods_data['Order Reduction'] = {
    #     'iterations': redux_iterations,
    #     'lower_bounds': redux_lower_bounds,
    #     'upper_bounds': redux_upper_bounds,
    #     'interval_lengths': redux_interval_lengths,
    #     'num_constrained': redux_num_constrained,
    #     'time': redux_time,
    #     'color': 'green'
    # }
    
    
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
    plt.savefig(f'./Plots/COUNT/interval_comparison_groups_size_{size}_relation_degree_{int(relation_degree * 100)}.png', dpi=300)
    print(f"Plot saved as 'interval_comparison_groups_size_{size}_relation_degree_{int(relation_degree * 100)}.png'")
    
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
    plt.savefig(f'./Plots/COUNT/bounds_comparison_groups_size_{size}_relation_degree_{int(relation_degree * 100)}.png', dpi=300)
    print(f"Plot saved as 'bounds_comparison_groups_size_{size}_relation_degree_{int(relation_degree * 100)}.png'")
    
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
    plt.savefig(f'./Plots/COUNT/variables_effect_comparison_groups_size_{size}_relation_degree_{int(relation_degree * 100)}.png', dpi=300)
    print(f"Plot saved as 'variables_effect_comparison_groups_size_{size}_relation_degree_{int(relation_degree * 100)}.png'")
    
    # ==========================================================================
    # NEW: Time Analysis Visualizations for Partition Method
    # ==========================================================================
    
    # Plot 4: Cumulative Time vs Iteration
    plt.figure(figsize=(14, 8))
    
    # Create subplots for detailed time analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: Cumulative Time vs Iteration
    partition_data = all_methods_data['Partition Method']
    ax1.plot(partition_data['iterations'], partition_data['cumulative_times_min'], 
             'o-', color='blue', linewidth=2, markersize=6, label='Min Problem')
    ax1.plot(partition_data['iterations'], partition_data['cumulative_times_max'], 
             's-', color='red', linewidth=2, markersize=6, label='Max Problem')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cumulative Time (seconds)')
    ax1.set_title('Cumulative Execution Time vs Iteration')
    ax1.grid(True)
    ax1.legend()
    
    # Subplot 2: Per-Iteration Time vs Iteration
    ax2.plot(partition_data['iterations'], partition_data['iter_total_times_min'], 
             'o-', color='blue', linewidth=2, markersize=6, label='Min Problem')
    ax2.plot(partition_data['iterations'], partition_data['iter_total_times_max'], 
             's-', color='red', linewidth=2, markersize=6, label='Max Problem')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Per-Iteration Time (seconds)')
    ax2.set_title('Per-Iteration Execution Time')
    ax2.grid(True)
    ax2.legend()
    
    # Subplot 3: Time Breakdown for Min Problem
    iterations_array = np.array(partition_data['iterations'])
    setup_times = np.array(partition_data['setup_times_min'])
    processing_times = np.array(partition_data['processing_times_min'])
    solve_times = np.array(partition_data['solve_times_min'])
    
    ax3.bar(iterations_array - 0.15, setup_times, 0.3, label='Setup Time', alpha=0.8)
    ax3.bar(iterations_array + 0.15, processing_times, 0.3, label='Processing Time', alpha=0.8)
    ax3.bar(iterations_array, solve_times, 0.3, label='Solve Time', alpha=0.8, bottom=setup_times+processing_times)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Time Breakdown per Iteration (Min Problem)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Time Breakdown for Max Problem
    setup_times_max = np.array(partition_data['setup_times_max'])
    processing_times_max = np.array(partition_data['processing_times_max'])
    solve_times_max = np.array(partition_data['solve_times_max'])
    
    ax4.bar(iterations_array - 0.15, setup_times_max, 0.3, label='Setup Time', alpha=0.8)
    ax4.bar(iterations_array + 0.15, processing_times_max, 0.3, label='Processing Time', alpha=0.8)
    ax4.bar(iterations_array, solve_times_max, 0.3, label='Solve Time', alpha=0.8, bottom=setup_times_max+processing_times_max)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Time (seconds)')
    ax4.set_title('Time Breakdown per Iteration (Max Problem)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'./Plots/COUNT/time_analysis_partition_method_size_{size}_relation_degree_{int(relation_degree * 100)}.png', dpi=300)
    print(f"Plot saved as 'time_analysis_partition_method_size_{size}_relation_degree_{int(relation_degree * 100)}.png'")
    
    # Plot 5: Time vs Number of Constrained Variables
    plt.figure(figsize=(12, 6))
    
    plt.plot(partition_data['num_constrained'], partition_data['cumulative_times_min'], 
             'o-', color='blue', linewidth=2, markersize=6, label='Min Problem Cumulative Time')
    plt.plot(partition_data['num_constrained'], partition_data['cumulative_times_max'], 
             's-', color='red', linewidth=2, markersize=6, label='Max Problem Cumulative Time')
    plt.plot(partition_data['num_constrained'], partition_data['iter_total_times_min'], 
             '^-', color='green', linewidth=2, markersize=6, label='Min Problem Per-Iteration Time')
    plt.plot(partition_data['num_constrained'], partition_data['iter_total_times_max'], 
             'v-', color='orange', linewidth=2, markersize=6, label='Max Problem Per-Iteration Time')
    
    plt.xlabel('Number of Constrained Variables')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time vs Number of Constrained Variables')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./Plots/COUNT/time_vs_constrained_vars_size_{size}_relation_degree_{int(relation_degree * 100)}.png', dpi=300)
    print(f"Plot saved as 'time_vs_constrained_vars_size_{size}_relation_degree_{int(relation_degree * 100)}.png'")
    
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
    
    # Print detailed time analysis for Partition Method
    print("\nDetailed Time Analysis for Partition Method:")
    print("-" * 100)
    print(f"{'Iter':<6} {'Constrained':<12} {'Setup(Min)':<12} {'Proc(Min)':<12} {'Solve(Min)':<12} {'Total(Min)':<12} {'Total(Max)':<12}")
    print("-" * 100)
    
    partition_data = all_methods_data['Partition Method']
    for i in range(len(partition_data['iterations'])):
        print(f"{partition_data['iterations'][i]:<6} "
              f"{partition_data['num_constrained'][i]:<12} "
              f"{partition_data['setup_times_min'][i]:<12.3f} "
              f"{partition_data['processing_times_min'][i]:<12.3f} "
              f"{partition_data['solve_times_min'][i]:<12.3f} "
              f"{partition_data['iter_total_times_min'][i]:<12.3f} "
              f"{partition_data['iter_total_times_max'][i]:<12.3f}")
    
    print("-" * 100)
    
if __name__ == "__main__":
    main()