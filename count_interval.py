#!/usr/bin/env python3
"""
Simulate the query:
    SELECT COUNT(*) FROM T WHERE x > y AND y > z;
Assuming:
    - x and z are constant numbers.
    - y (for each row) is a zonotope represented as a pandas Series.
      The zonotope’s range is [center – sum(|g_i|), center + sum(|g_i|)].
For each row we decide:
    - If x > (y.upper) AND (y.lower) > z, then the condition is definitely True.
    - If x <= (y.lower) OR (y.upper) <= z, then the condition is definitely False.
    - Otherwise, the row is uncertain (its contribution is between 0 and 1).
We then sum the definite and uncertain indicators to get an interval for the count.
"""

import pandas as pd
import os, sys

utilDir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(utilDir)
from range_class import R, bound_from_row

def count_condition(row_interval, x, z):
    """
      - Definitely true if (x > y.ub AND y.lb > z).
      - Definitely false if (x <= y.lb OR y.ub <= z).
      - Otherwise uncertain, meaning the row might contribute either 0 or 1.
    """
    if x > row_interval.ub and row_interval.lb > z:
        return (1, 1)
    if x <= row_interval.lb or row_interval.ub <= z:
        return (0, 0)
    return (0, 1)

def count_filtered_rows(df_x, df_y, df_z):
    total_lower = 0
    total_upper = 0
    n = len(df_y)
    for i in range(n):
        x_val = df_x.iloc[i]["center"]
        z_val = df_z.iloc[i]["center"]
        row_y = df_y.iloc[i]
        y_interval = bound_from_row(row_y)
        lb, ub = count_condition(y_interval, x_val, z_val)
        total_lower += lb
        total_upper += ub
    return R(total_lower, total_upper)

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
    
    if not (len(df_x) == len(df_y) == len(df_z)):
        print("Error: The files for x, y, and z must have the same number of rows for row-wise comparison.")
        return
    
    count_interval = count_filtered_rows(df_x, df_y, df_z)
    print("Simulated interval COUNT(*) for condition (x > y AND y > z):", count_interval)

if __name__ == "__main__":
    main()
