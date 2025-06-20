import sys
import os
direc = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '20241004'))
sys.path.append(direc)
from zono_generation_groups import generate_zonotope_data

# Generate data for SELECT Salary
def main():
    num_groups = 1
    zono_distribution = [0, 1] # 100% numbers, 0% zonotopes
    coeff_range = [-500, 500]
    center_range = [5000, 10000]
    relation_degree = 0.5 # 0% zonotopes in a group are related (shared error terms)
    max_total_unknowns = None
    size = 1000
    num_variables = [1] # number of variables: 1
    var_distribution = [1] # corresponding to num_variables.
    shared_errors = 1000 # number of possible shared errors
    seed = 13

    common_params = {
        "group_size": size,
        "num_groups": num_groups,
        "shared_pool_size": shared_errors,
        "zono_distribution": zono_distribution,
        "num_variables": num_variables,
        "var_distribution": var_distribution,
        "coeff_range": coeff_range,
        "center_range": center_range,
        "relation_degree": relation_degree,
        "output_file": None,
        "max_total_unknowns": max_total_unknowns,
        "seed": seed
    }

    output_dir = f'./Data/AVG'
    os.makedirs(output_dir, exist_ok=True)

    common_params["output_file"] = f'sum_z_{num_groups}_groups_size_{size}_relation_degree_{int(relation_degree * 100)}.parquet'

    # Save the file in the correct directory
    output_path = os.path.join(output_dir, common_params["output_file"])
    common_params["output_file"] = output_path
    zonotope_df = generate_zonotope_data(**common_params)

    #print(f"Generated data for group_size={size}:")
    #print(zonotope_df.head())
    #print(f"Data saved to ./{num_groups}_groups_{size}_shared_{nvar}/{common_params['output_file']}")

if __name__ == "__main__":
    main()
