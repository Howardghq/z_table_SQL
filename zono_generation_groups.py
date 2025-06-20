import numpy as np
import pandas as pd
import warnings
from multiprocessing import Pool, cpu_count
import argparse
import math

def generate_shared_tuples(n_shared, shared_pool, num_variables, var_distribution):
    shared_tuples = []
    for _ in range(n_shared):
        n_vars = np.random.choice(num_variables, p=var_distribution)
        if len(shared_pool) >= n_vars:
            indices = np.random.choice(shared_pool, size=n_vars, replace=False)
        else:
            indices = np.random.choice(shared_pool, size=n_vars, replace=True)
            warnings.warn(
                f"Shared Pool: Not enough variables to select {n_vars} unique variables. Allowing replacement."
            )
        shared_tuples.append(indices)
    return shared_tuples

def generate_unique_tuples(n_unique, start_idx, num_variables, var_distribution):
    unique_tuples = []
    current_idx = start_idx
    for _ in range(n_unique):
        n_vars = np.random.choice(num_variables, p=var_distribution)
        indices = np.arange(current_idx, current_idx + n_vars)
        unique_tuples.append(indices)
        current_idx += n_vars
    return unique_tuples

def generate_zonotope_representation(args):
    indices, center_range, coeff_range, max_variables = args
    center = np.random.uniform(center_range[0], center_range[1])
    coefficients = np.random.uniform(coeff_range[0], coeff_range[1], size=len(indices))
    zonotope_representation = [center]
    #A number has only one index and it's 0 representing a center in zonotopes
    if (indices[0] != 0):
        for coeff, idx in zip(coefficients, indices):
            zonotope_representation.extend([coeff, idx])
    # Pad with NaNs if necessary
    expected_length = 1 + 2 * max_variables#[center, g1, idx1, g2, idx 2, ...]
    actual_length = len(zonotope_representation)
    if actual_length < expected_length:
        zonotope_representation.extend([np.nan] * (expected_length - actual_length))
    return zonotope_representation

def generate_zonotope_group_data(args):
    (
        row_id,#group_id changed to row_id, since we're using a hacking way in bound() to find idx starting with 'g'. Should refactor that part.
        group_size,
        zono_size,
        shared_pool_size,
        starting_error_term_idx,
        num_variables,
        var_distribution,
        center_range,
        coeff_range,
        relation_degree
    ) = args

    n_shared_tuples = int(math.ceil(relation_degree * zono_size))
    n_unique_tuples = zono_size - n_shared_tuples

    shared_pool = np.arange(starting_error_term_idx, starting_error_term_idx + shared_pool_size)
    
    unique_start_idx = starting_error_term_idx + shared_pool_size
    unique_tuples = generate_unique_tuples(n_unique_tuples, unique_start_idx, num_variables, var_distribution)
    
    shared_tuples = generate_shared_tuples(n_shared_tuples, shared_pool, num_variables, var_distribution)
    
    all_tuples_indices = shared_tuples + unique_tuples + [np.zeros(1) for _ in range(group_size - zono_size)]
    np.random.shuffle(all_tuples_indices)
    
    # Determine maximum number of variables per tuple
    max_variables = max(num_variables)
    args_list = [(indices, center_range, coeff_range, max_variables) for indices in all_tuples_indices]
    
    zonotope_data = [generate_zonotope_representation(arg) for arg in args_list]
    
    column_names = ['center']
    for i in range(max_variables):
        column_names.extend([f'g{i+1}', f'idx{i+1}'])
    
    zonotope_df = pd.DataFrame(zonotope_data, columns=column_names)
    
    zonotope_df['row_id'] = row_id
    
    return zonotope_df

def generate_zonotope_data(
    num_groups,
    group_size,
    shared_pool_size,
    zono_distribution=(0.05, 0.95),#5% numbers, 95% zonotopes
    num_variables=(1, 2, 3),
    var_distribution=(0.8, 0.15, 0.05),
    center_range=(-5, 5),
    coeff_range=(-2, 2),
    relation_degree=0.8,
    max_total_unknowns=None,
    output_file='zonotope_data.parquet',
    seed=34
):
    # Set the seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        print(f"Random seed set to: {seed}")

    # Validate inputs
    if not isinstance(num_variables, (list, tuple)):
        raise ValueError("num_variables must be a list or tuple.")
    if not np.isclose(sum(zono_distribution), 1.0):
        raise ValueError("The probabilities in zono_distribution must sum to 1.")
    if len(var_distribution) != len(num_variables):
        raise ValueError("var_distribution must have the same length as num_variables.")
    if not np.isclose(sum(var_distribution), 1.0):
        raise ValueError("The probabilities in var_distribution must sum to 1.")
    if not (0 <= relation_degree <= 1):
        raise ValueError("relation_degree must be between 0 and 1.")
    # It's possible that none zonotopes are sharing any error terms. No correlation across tuples
    #if shared_pool_size < max(num_variables):
        #raise ValueError("shared_pool_size must be at least as large as max(num_variables).")

    # Initialize variables
    current_error_term_idx = 1  # Start indexing error terms from 1

    zono_size = math.ceil(group_size * zono_distribution[1])
    # Calculate total required error terms
    total_shared_error_terms = shared_pool_size * num_groups
    estimated_unique_vars_per_group = math.ceil((1 - relation_degree) * zono_size) * max(num_variables)
    total_unique_error_terms = estimated_unique_vars_per_group * num_groups

    # Check if total required variables exceed max_total_unknowns
    if max_total_unknowns is not None:
        total_required_vars = total_shared_error_terms + total_unique_error_terms
        if total_required_vars > max_total_unknowns:
            raise ValueError(
                f"max_total_unknowns ({max_total_unknowns}) is less than the required total unique variables ({total_required_vars})."
            )

    # Function arguments for multiprocessing
    group_args = []
    for row_id in range(1, num_groups + 1):
        args = (
            row_id,
            group_size,
            zono_size,
            shared_pool_size,
            current_error_term_idx,
            num_variables,
            var_distribution,
            center_range,
            coeff_range,
            relation_degree
        )
        group_args.append(args)
        # Update starting_error_term_idx for next group
        n_shared_tuples = int(math.ceil(relation_degree * group_size))
        n_unique_tuples = group_size - n_shared_tuples
        unique_error_terms = n_unique_tuples * max(num_variables)
        current_error_term_idx += shared_pool_size + unique_error_terms

    print(f"Generating zonotope data for {num_groups} groups, each with {group_size} zonotopes...")
    with Pool(processes=cpu_count()) as pool:
        zonotope_data_list = pool.map(generate_zonotope_group_data, group_args)

    print("Concatenating all groups' data...")
    all_zonotopes_df = pd.concat(zonotope_data_list, ignore_index=True)

    total_unique_variables = all_zonotopes_df[[col for col in all_zonotopes_df.columns if 'idx' in col]].stack().nunique()
    if max_total_unknowns is not None and total_unique_variables > max_total_unknowns:
        warnings.warn(
            f"Total unique variables used ({total_unique_variables}) exceed 'max_total_unknowns' ({max_total_unknowns})."
        )

    all_zonotopes_df.to_parquet(output_file, index=False)
    print(f"Zonotope data generated and saved to {output_file}.")
    print(f"Total unique variables used: {total_unique_variables}")
    print(f"Total zonotopes generated: {len(all_zonotopes_df)}")

    return all_zonotopes_df

def main():
    num_groups = 10000
    group_size = 100

    parser = argparse.ArgumentParser(description='Optimized Zonotope Data Generator for Multiple Groups')
    parser.add_argument('--num_groups', type=int, default=num_groups, help='Number of groups to generate.')
    parser.add_argument('--group_size', type=int, default=group_size, help='Number of zonotopes per group.')
    parser.add_argument('--shared_pool_size', type=int, default=10, help='Number of variables in the shared pool per group.')
    parser.add_argument('--zono_distribution', nargs='+', type=float, default=[0.2, 0.8], help='Distribution of numbers and zonotopes in the dataset.')
    parser.add_argument('--num_variables', nargs='+', type=int, default=[1, 2, 3], help='Possible number of variables per tuple.')
    parser.add_argument('--var_distribution', nargs='+', type=float, default=[0.8, 0.15, 0.05], help='Distribution of variables per tuple.')
    parser.add_argument('--center_range', nargs=2, type=float, default=[-5, 5], help='Range for center values.')
    parser.add_argument('--coeff_range', nargs=2, type=float, default=[-2, 2], help='Range for generator coefficients.')
    parser.add_argument('--relation_degree', type=float, default=0.8, help='Proportion of tuples using shared variables.')
    parser.add_argument('--max_total_unknowns', type=int, default=None, help='Maximum total unknown variables allowed.')
    parser.add_argument('--output_file', type=str, default=f'zonotope_data_{num_groups}_{group_size}.parquet', help='Output file path.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')

    args = parser.parse_args()

    # Convert lists to tuples
    num_variables = tuple(args.num_variables)
    var_distribution = tuple(args.var_distribution)

    # Generate the zonotope data
    zonotope_df = generate_zonotope_data(
        num_groups=args.num_groups,
        group_size=args.group_size,
        shared_pool_size=args.shared_pool_size,
        zono_distribution=args.zono_distribution,
        num_variables=num_variables,
        var_distribution=var_distribution,
        center_range=tuple(args.center_range),
        coeff_range=tuple(args.coeff_range),
        relation_degree=args.relation_degree,
        max_total_unknowns=args.max_total_unknowns,
        output_file=args.output_file,
        seed=args.seed
    )

    # Show a sample of the generated data
    print(zonotope_df.head(10))

if __name__ == "__main__":
    main()
