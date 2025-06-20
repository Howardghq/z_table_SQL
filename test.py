from count_zono_iter import *

# File paths (adjust these paths as needed)
num_groups = 1
size = 1000
file_x = f"./Data/COUNT/count_x_{num_groups}_groups_size_{size}.parquet"
file_y = f"./Data/COUNT/count_y_{num_groups}_groups_size_{size}.parquet"
file_z = f"./Data/COUNT/count_z_{num_groups}_groups_size_{size}.parquet"

df_x = pd.read_parquet(file_x)
df_y = pd.read_parquet(file_y)
df_z = pd.read_parquet(file_z)
print("Files loaded successfully.")

# Set parameters
max_iterations = 1
batch_size = 2000
relax = False  # Use relaxed variables for faster computation

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

print(f"\nResults with binary variables:")
print(f"Lower and Upper Bound: [{int(count_min)}, {int(count_max)}]")

# Print iteration results
print("\nIteration results (MIN):")
for i, result in enumerate(results_min):
    iteration = result['iteration']
    count = result['count']
    extracted = result['extracted_indices']
    print(f"Iteration {iteration}: Count = {count}")

print("\nIteration results (MAX):")
for i, result in enumerate(results_max):
    iteration = result['iteration']
    count = result['count']
    extracted = result['extracted_indices']
    print(f"Iteration {iteration}: Count = {count}")