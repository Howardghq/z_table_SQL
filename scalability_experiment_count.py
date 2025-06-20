import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
from count_zono import compute_count_with_constraint, compute_count_with_constraint_tight
from count_zono_iter import compute_count_with_iterative_independence
from count_interval import count_filtered_rows

def run_experiments_for_size(df_x, df_y, df_z, size, max_iterations=10, batch_size=100, relax=True):
    """
    Run all algorithms for a specific dataset size and return timing results
    """
    # Take only the first 'size' rows
    df_x_subset = df_x.iloc[:size].copy()
    df_y_subset = df_y.iloc[:size].copy()
    df_z_subset = df_z.iloc[:size].copy()
    
    results = {'size': size}
    
    print(f"\nRunning experiments for dataset size: {size}")
    
    # ==========================================================================
    # Method 0: Interval Based Method
    # ==========================================================================
    print("  Running Interval Based Method...")
    start_time = time.time()
    try:
        interval = count_filtered_rows(df_x_subset, df_y_subset, df_z_subset)
        interval_time = time.time() - start_time
        results['interval_time'] = interval_time
        results['interval_lower'] = interval.lb
        results['interval_upper'] = interval.ub
        results['interval_length'] = interval.ub - interval.lb
        print(f"    Completed in {interval_time:.3f}s")
    except Exception as e:
        print(f"    Error: {e}")
        results['interval_time'] = None
    
    # ==========================================================================
    # Method 1: Partition Method
    # ==========================================================================
    print("  Running Partition Method...")
    start_time = time.time()
    try:
        # Min bound
        status_min, count_min, _, results_min = compute_count_with_iterative_independence(
            df_y_subset, df_x_subset, df_z_subset, 'min', 
            batch_size=batch_size, max_iterations=max_iterations, relax=relax
        )
        
        # Max bound
        status_max, count_max, _, results_max = compute_count_with_iterative_independence(
            df_y_subset, df_x_subset, df_z_subset, 'max', 
            batch_size=batch_size, max_iterations=max_iterations, relax=relax
        )
        
        partition_time = time.time() - start_time
        results['partition_time'] = partition_time
        results['partition_lower'] = count_min
        results['partition_upper'] = count_max
        results['partition_length'] = count_max - count_min
        results['partition_iterations'] = len(results_min)
        
        # Store detailed iteration times
        results['partition_cumulative_times_min'] = [r['cumulative_time'] for r in results_min]
        results['partition_cumulative_times_max'] = [r['cumulative_time'] for r in results_max]
        results['partition_final_constrained'] = results_min[-1]['num_constrained'] if results_min else 0
        
        print(f"    Completed in {partition_time:.3f}s ({len(results_min)} iterations)")
    except Exception as e:
        print(f"    Error: {e}")
        results['partition_time'] = None
    
    # ==========================================================================
    # Method 3a: Original MILP
    # ==========================================================================
    print("  Running Original MILP...")
    start_time = time.time()
    try:
        # Min bound
        status_min, count_min, _ = compute_count_with_constraint(
            df_y_subset, df_x_subset, df_z_subset, 'min', relax=relax
        )
        
        # Max bound
        status_max, count_max, _ = compute_count_with_constraint(
            df_y_subset, df_x_subset, df_z_subset, 'max', relax=relax
        )
        
        original_time = time.time() - start_time
        results['original_time'] = original_time
        results['original_lower'] = count_min
        results['original_upper'] = count_max
        results['original_length'] = count_max - count_min
        print(f"    Completed in {original_time:.3f}s")
    except Exception as e:
        print(f"    Error: {e}")
        results['original_time'] = None
    
    # ==========================================================================
    # Method 3b: Tight Constraints
    # ==========================================================================
    print("  Running Tight Constraints...")
    start_time = time.time()
    try:
        # Min bound
        status_min, count_min, _ = compute_count_with_constraint_tight(
            df_y_subset, df_x_subset, df_z_subset, 'min', relax=relax
        )
        
        # Max bound
        status_max, count_max, _ = compute_count_with_constraint_tight(
            df_y_subset, df_x_subset, df_z_subset, 'max', relax=relax
        )
        
        tight_time = time.time() - start_time
        results['tight_time'] = tight_time
        results['tight_lower'] = count_min
        results['tight_upper'] = count_max
        results['tight_length'] = count_max - count_min
        print(f"    Completed in {tight_time:.3f}s")
    except Exception as e:
        print(f"    Error: {e}")
        results['tight_time'] = None
    
    return results

def main():
    """
    Main function to run scalability experiments across different dataset sizes
    """
    # Configuration
    num_groups = 1
    full_size = 1000
    relation_degree = 1
    
    # Dataset sizes to test
    sizes_to_test = [100, 200, 500, 1000]
    
    # Algorithm parameters
    max_iterations = 10
    batch_size = 100
    relax = True
    
    # File paths
    file_x = f"./Data/COUNT/count_x_{num_groups}_groups_size_{full_size}.parquet"
    file_y = f"./Data/COUNT/count_y_{num_groups}_groups_size_{full_size}_relation_degree_{int(relation_degree * 100)}.parquet"
    file_z = f"./Data/COUNT/count_z_{num_groups}_groups_size_{full_size}.parquet"
    
    # Load full datasets
    print("Loading datasets...")
    try:
        df_x = pd.read_parquet(file_x)
        df_y = pd.read_parquet(file_y)
        df_z = pd.read_parquet(file_z)
        print(f"Successfully loaded datasets with {len(df_x)} rows")
    except Exception as e:
        print(f"Error loading files: {e}")
        return
    
    # Verify data consistency
    if not (len(df_x) == len(df_y) == len(df_z)):
        print("Error: All datasets must have the same number of rows")
        return
    
    # Run experiments for each size
    all_results = []
    
    print(f"\nRunning scalability experiments for sizes: {sizes_to_test}")
    print("=" * 60)
    
    for size in sizes_to_test:
        if size > len(df_x):
            print(f"Skipping size {size} (larger than available data: {len(df_x)})")
            continue
        
        results = run_experiments_for_size(
            df_x, df_y, df_z, size, 
            max_iterations=max_iterations, 
            batch_size=batch_size, 
            relax=relax
        )
        all_results.append(results)
    
    # ==========================================================================
    # Visualization: Time Scaling Analysis
    # ==========================================================================
    print("\nGenerating scalability visualizations...")
    
    # Extract data for plotting
    sizes = [r['size'] for r in all_results]
    interval_times = [r.get('interval_time', None) for r in all_results]
    partition_times = [r.get('partition_time', None) for r in all_results]
    original_times = [r.get('original_time', None) for r in all_results]
    tight_times = [r.get('tight_time', None) for r in all_results]
    
    # Filter out None values for plotting
    def filter_valid_data(sizes, times):
        valid_indices = [i for i, t in enumerate(times) if t is not None]
        return [sizes[i] for i in valid_indices], [times[i] for i in valid_indices]
    
    sizes_interval, times_interval = filter_valid_data(sizes, interval_times)
    sizes_partition, times_partition = filter_valid_data(sizes, partition_times)
    sizes_original, times_original = filter_valid_data(sizes, original_times)
    sizes_tight, times_tight = filter_valid_data(sizes, tight_times)
    
    # Plot 1: Linear scale time comparison
    plt.figure(figsize=(12, 8))
    
    if times_interval:
        plt.plot(sizes_interval, times_interval, 'o-', color='black', linewidth=2, 
                markersize=8, label='Interval Based Method')
    if times_partition:
        plt.plot(sizes_partition, times_partition, 's-', color='blue', linewidth=2, 
                markersize=8, label='Partition Method')
    if times_original:
        plt.plot(sizes_original, times_original, '^-', color='red', linewidth=2, 
                markersize=8, label='Original MILP')
    if times_tight:
        plt.plot(sizes_tight, times_tight, 'v-', color='purple', linewidth=2, 
                markersize=8, label='Tight Constraints')
    
    plt.xlabel('Dataset Size (number of rows)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Scalability Analysis: Execution Time vs Dataset Size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./Plots/COUNT/scalability_linear_relation_degree_{int(relation_degree * 100)}.png', dpi=300)
    print("Plot saved as 'scalability_linear_relation_degree_*.png'")
    
    # Plot 2: Log-scale time comparison
    plt.figure(figsize=(12, 8))
    
    if times_interval:
        plt.semilogy(sizes_interval, times_interval, 'o-', color='black', linewidth=2, 
                    markersize=8, label='Interval Based Method')
    if times_partition:
        plt.semilogy(sizes_partition, times_partition, 's-', color='blue', linewidth=2, 
                    markersize=8, label='Partition Method')
    if times_original:
        plt.semilogy(sizes_original, times_original, '^-', color='red', linewidth=2, 
                    markersize=8, label='Original MILP')
    if times_tight:
        plt.semilogy(sizes_tight, times_tight, 'v-', color='purple', linewidth=2, 
                    markersize=8, label='Tight Constraints')
    
    plt.xlabel('Dataset Size (number of rows)')
    plt.ylabel('Execution Time (seconds, log scale)')
    plt.title('Scalability Analysis: Execution Time vs Dataset Size (Log Scale)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./Plots/COUNT/scalability_log_relation_degree_{int(relation_degree * 100)}.png', dpi=300)
    print("Plot saved as 'scalability_log_relation_degree_*.png'")
    
    # Plot 3: Interval length analysis
    plt.figure(figsize=(12, 8))
    
    interval_lengths = [r.get('interval_length', None) for r in all_results]
    partition_lengths = [r.get('partition_length', None) for r in all_results]
    original_lengths = [r.get('original_length', None) for r in all_results]
    tight_lengths = [r.get('tight_length', None) for r in all_results]
    
    sizes_int_len, lengths_interval = filter_valid_data(sizes, interval_lengths)
    sizes_part_len, lengths_partition = filter_valid_data(sizes, partition_lengths)
    sizes_orig_len, lengths_original = filter_valid_data(sizes, original_lengths)
    sizes_tight_len, lengths_tight = filter_valid_data(sizes, tight_lengths)
    
    if lengths_interval:
        plt.plot(sizes_int_len, lengths_interval, 'o-', color='black', linewidth=2, 
                markersize=8, label='Interval Based Method')
    if lengths_partition:
        plt.plot(sizes_part_len, lengths_partition, 's-', color='blue', linewidth=2, 
                markersize=8, label='Partition Method')
    if lengths_original:
        plt.plot(sizes_orig_len, lengths_original, '^-', color='red', linewidth=2, 
                markersize=8, label='Original MILP')
    if lengths_tight:
        plt.plot(sizes_tight_len, lengths_tight, 'v-', color='purple', linewidth=2, 
                markersize=8, label='Tight Constraints')
    
    plt.xlabel('Dataset Size (number of rows)')
    plt.ylabel('Interval Length (Upper - Lower Bound)')
    plt.title('Scalability Analysis: Solution Quality vs Dataset Size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./Plots/COUNT/scalability_quality_relation_degree_{int(relation_degree * 100)}.png', dpi=300)
    print("Plot saved as 'scalability_quality_relation_degree_*.png'")
    
    # Plot 4: Combined efficiency analysis (time vs quality trade-off)
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot of execution time vs interval length
    if times_interval and lengths_interval:
        plt.scatter([times_interval[i] for i in range(len(times_interval))], 
                   [lengths_interval[i] for i in range(len(lengths_interval))], 
                   s=100, c='black', marker='o', label='Interval Based')
    if times_partition and lengths_partition:
        plt.scatter([times_partition[i] for i in range(len(times_partition))], 
                   [lengths_partition[i] for i in range(len(lengths_partition))], 
                   s=100, c='blue', marker='s', label='Partition Method')
    if times_original and lengths_original:
        plt.scatter([times_original[i] for i in range(len(times_original))], 
                   [lengths_original[i] for i in range(len(lengths_original))], 
                   s=100, c='red', marker='^', label='Original MILP')
    if times_tight and lengths_tight:
        plt.scatter([times_tight[i] for i in range(len(times_tight))], 
                   [lengths_tight[i] for i in range(len(lengths_tight))], 
                   s=100, c='purple', marker='v', label='Tight Constraints')
    
    # Add arrows to show dataset size progression
    for i, (times, lengths, sizes_list, color) in enumerate([
        (times_interval, lengths_interval, sizes_interval, 'black'),
        (times_partition, lengths_partition, sizes_partition, 'blue'),
        (times_original, lengths_original, sizes_original, 'red'),
        (times_tight, lengths_tight, sizes_tight, 'purple')
    ]):
        if times and lengths and len(times) > 1:
            for j in range(len(times) - 1):
                plt.annotate('', xy=(times[j+1], lengths[j+1]), xytext=(times[j], lengths[j]),
                            arrowprops=dict(arrowstyle='->', color=color, alpha=0.6))
    
    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('Interval Length (Upper - Lower Bound)')
    plt.title('Efficiency Analysis: Time vs Quality Trade-off\n(Arrow direction indicates increasing dataset size)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./Plots/COUNT/scalability_efficiency_relation_degree_{int(relation_degree * 100)}.png', dpi=300)
    print("Plot saved as 'scalability_efficiency_relation_degree_*.png'")
    
    # ==========================================================================
    # Print Summary Results
    # ==========================================================================
    print("\nScalability Experiment Results Summary")
    print("=" * 100)
    print(f"{'Size':<8} {'Interval':<12} {'Partition':<12} {'Original':<12} {'Tight':<12}")
    print(f"{'':^8} {'Time':<12} {'Time':<12} {'Time':<12} {'Time':<12}")
    print("-" * 100)
    
    for result in all_results:
        size = result['size']
        interval_t = f"{result.get('interval_time', 'ERROR'):.3f}" if result.get('interval_time') else "ERROR"
        partition_t = f"{result.get('partition_time', 'ERROR'):.3f}" if result.get('partition_time') else "ERROR"
        original_t = f"{result.get('original_time', 'ERROR'):.3f}" if result.get('original_time') else "ERROR"
        tight_t = f"{result.get('tight_time', 'ERROR'):.3f}" if result.get('tight_time') else "ERROR"
        
        print(f"{size:<8} {interval_t:<12} {partition_t:<12} {original_t:<12} {tight_t:<12}")
    
    print("-" * 100)
    
    # Quality summary
    print("\nSolution Quality Summary (Interval Lengths)")
    print("=" * 100)
    print(f"{'Size':<8} {'Interval':<12} {'Partition':<12} {'Original':<12} {'Tight':<12}")
    print(f"{'':^8} {'Length':<12} {'Length':<12} {'Length':<12} {'Length':<12}")
    print("-" * 100)
    
    for result in all_results:
        size = result['size']
        interval_l = f"{result.get('interval_length', 'ERROR'):.1f}" if result.get('interval_length') is not None else "ERROR"
        partition_l = f"{result.get('partition_length', 'ERROR'):.1f}" if result.get('partition_length') is not None else "ERROR"
        original_l = f"{result.get('original_length', 'ERROR'):.1f}" if result.get('original_length') is not None else "ERROR"
        tight_l = f"{result.get('tight_length', 'ERROR'):.1f}" if result.get('tight_length') is not None else "ERROR"
        
        print(f"{size:<8} {interval_l:<12} {partition_l:<12} {original_l:<12} {tight_l:<12}")
    
    print("-" * 100)
    
    # Calculate and print scaling factors
    print("\nTime Scaling Analysis:")
    print("=" * 60)
    
    def calculate_scaling_factor(sizes, times):
        if len(sizes) < 2 or len(times) < 2:
            return "N/A"
        # Calculate average time increase per size increase
        time_ratios = []
        size_ratios = []
        for i in range(1, len(sizes)):
            if times[i-1] > 0:
                time_ratios.append(times[i] / times[i-1])
                size_ratios.append(sizes[i] / sizes[i-1])
        if time_ratios:
            avg_time_factor = np.mean(time_ratios)
            avg_size_factor = np.mean(size_ratios)
            # Approximate complexity: O(n^k) where k = log(time_factor) / log(size_factor)
            if avg_size_factor > 1:
                complexity_exp = np.log(avg_time_factor) / np.log(avg_size_factor)
                return f"~O(n^{complexity_exp:.2f})"
        return "N/A"
    
    methods = [
        ("Interval Based", sizes_interval, times_interval),
        ("Partition Method", sizes_partition, times_partition),
        ("Original MILP", sizes_original, times_original),
        ("Tight Constraints", sizes_tight, times_tight)
    ]
    
    for method_name, method_sizes, method_times in methods:
        scaling = calculate_scaling_factor(method_sizes, method_times)
        print(f"{method_name:<20}: {scaling}")
    
    print("=" * 60)
    print("Scalability experiment completed successfully!")

if __name__ == "__main__":
    main()