#!/usr/bin/env python3
"""
  SELECT COUNT(*) FROM T
  WHERE (x > y) AND (y > z);

 - If entire zonotope is definitely inside => row contributes 1 (def_in).
 - If entire zonotope is definitely outside => row contributes 0 (def_out).
 - Otherwise uncertain => we add continuous i_r ∈ [0,1].
"""

import pandas as pd
import numpy as np
import pulp
from pulp import PULP_CBC_CMD
import math
import os, sys

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

class PsplitCountLP:
    def __init__(self, df_y, df_x, df_z, P, epsilon=1e-6):
        self.df_y = df_y
        self.df_x = df_x
        self.df_z = df_z
        self.P = P
        self.eps = epsilon
        self.n = len(df_y)

        self.def_in = set()
        self.def_out = set()
        self.uncertain = []
        for r in range(self.n):
            rowy = df_y.iloc[r]
            x_val = df_x.iloc[r]["center"]
            z_val = df_z.iloc[r]["center"]
            y_lb, y_ub = bound(rowy)

            if y_ub < x_val and y_lb > z_val:
                self.def_in.add(r)
            elif y_lb >= x_val or y_ub <= z_val:
                self.def_out.add(r)
            else:
                self.uncertain.append(r)

        self.i_vars = {}
        self.lam_vars = {}
        self.y_aux_vars = {}
        self.e_vars = {}

    def build_lp(self, objective="max"):
        if objective == "min":
            prob = pulp.LpProblem("Psplit_Count_Min", pulp.LpMinimize)
        else:
            prob = pulp.LpProblem("Psplit_Count_Max", pulp.LpMaximize)

        obj_expr = 0.0

        def_in_count = len(self.def_in)

        for r in self.uncertain:
            i_var = pulp.LpVariable(f"i_{r}", lowBound=0, upBound=1, cat="Continuous")
            self.i_vars[r] = i_var
            obj_expr += i_var

        for r in self.uncertain:
            rowy = self.df_y.iloc[r]
            x_val = self.df_x.iloc[r]["center"]
            z_val = self.df_z.iloc[r]["center"]
            i_var = self.i_vars[r]

            step = (x_val - z_val) / self.P
            b = [z_val + p*step for p in range(self.P+1)]

            lam_list = []
            for p in range(1,self.P+1):
                lam = pulp.LpVariable(f"lam_{r}_{p}", lowBound=0, upBound=1, cat="Continuous")
                self.lam_vars[(r,p)] = lam
                lam_list.append(lam)
            prob += pulp.lpSum(lam_list) == i_var, f"lam_sum_{r}"

            y_aux_list = []
            for p in range(1,self.P+1):
                lam = self.lam_vars[(r,p)]
                y_aux = pulp.LpVariable(f"y_{r}_{p}", lowBound=(b[p-1]+self.eps)*0, upBound=(b[p]-self.eps)*1, cat="Continuous")
                self.y_aux_vars[(r,p)] = y_aux
                y_aux_list.append(y_aux)

                prob += y_aux >= (b[p-1] + self.eps)*lam, f"psplit_lb_{r}_{p}"
                prob += y_aux <= (b[p]   - self.eps)*lam, f"psplit_ub_{r}_{p}"

            cval = rowy["center"]
            y_expr = cval

            i_gen = 1
            while f"g{i_gen}" in rowy and f"idx{i_gen}" in rowy:
                gval = rowy[f"g{i_gen}"]
                idx = rowy[f"idx{i_gen}"]
                if pd.isna(gval) or pd.isna(idx):
                    break
                idx = int(idx)
                if idx not in self.e_vars:
                    self.e_vars[idx] = pulp.LpVariable(f"e_{idx}", lowBound=-1, upBound=1, cat="Continuous")
                e_var = self.e_vars[idx]
                y_expr += gval * e_var
                i_gen += 1

            prob += pulp.lpSum(y_aux_list) == y_expr, f"y_expr_{r}"

        final_obj = obj_expr + def_in_count
        prob += final_obj, "TotalCount"

        return prob, def_in_count

    def solve_lp(self, objective="max"):
        prob, def_in_count = self.build_lp(objective=objective)
        #print(f"Definitelly in: {def_in_count}")
        solver = PULP_CBC_CMD(msg=0)
        prob.solve(solver)
        status = pulp.LpStatus[prob.status]
        obj_val = pulp.value(prob.objective)

        return status, obj_val

    def run(self):
        st_min, val_min = self.solve_lp(objective="min")
        st_max, val_max = self.solve_lp(objective="max")

        return st_min, val_min, st_max, val_max

def main():
    
    # data = [
    #     {"center": 3.5, "g1": 2.5, "idx1": 1}
    # ]
    # df_y = pd.DataFrame(data)
    # print("Input zonotope for y:")
    # print(df_y)

    # x = [
    #     {"center": 2}
    # ]
    # df_x = pd.DataFrame(x)

    # z = [
    #     {"center": 5}
    # ]
    # df_z = pd.DataFrame(z)
    
    num_groups = 1
    size = 10000
    file_x = f"../Data/COUNT/count_x_{num_groups}_groups_size_{size}.parquet"
    file_y = f"../Data/COUNT/count_y_{num_groups}_groups_size_{size}.parquet"
    file_z = f"../Data/COUNT/count_z_{num_groups}_groups_size_{size}.parquet"
    
    try:
        df_x = pd.read_parquet(file_x)
        df_y = pd.read_parquet(file_y)
        df_z = pd.read_parquet(file_z)
        print("Files loaded successfully.")
    except Exception as e:
        print(f"Error reading files: {e}")
        return
    
    timestamp = pd.Timestamp.now()
    psolver = PsplitCountLP(df_y, df_x, df_z, P=30, epsilon=1e-6)

    _, val_min, _, val_max = psolver.run()
    running_time = (pd.Timestamp.now() - timestamp)

    #print(f"\nP-Split LP, min_count={val_min}, max_count={val_max}")
    if val_min is not None and val_max is not None:
        print(f"COUNT range: [{int(val_min)}, {int(val_max)}]")
        print(f"Running time: {running_time}")

if __name__=="__main__":
    main()
