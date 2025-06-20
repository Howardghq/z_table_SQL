#!/usr/bin/env python3
"""
LP‑Based COUNT query solver with **zonotope order‑reduction**
------------------------------------------------------------
Handles the SQL query
    SELECT COUNT(*) FROM T WHERE x > y AND y > z;
Assumptions
    • x  (per‑row) is a known constant.
    • z  (per‑row) is a known constant.
    • y  (per‑row) is uncertain, represented as a *scalar* zonotope:
          y  =  c   +  Σ  g_j · e_j ,        -1 ≤ e_j ≤ 1.
      The generator count *p* can be very large.  We first shrink it to a
      user‑chosen order  *k*  (default k = 1) **per row**, then build the
      tightened MILP.

Main entry‑point  : ``main()`` (see bottom of file).

Major additions
================
1. **order_reduce_row() / order_reduce_df()**
   Implements a *Box*-style reduction for the scalar case (n = 1):
       –  Keep the |g| largest *k* generators.
       –  Fold the remaining (p − k) generators into **one** aggregated
          generator  g_fold = Σ |g_rest|  with its *own* uncertainty
          variable.  This keeps the MILP sound while shrinking variable
          count from p  →  k + 1  (usually 2).
2. **compute_count_with_constraint_tight()** now receives the already
   reduced ``df_y`` – no further changes were needed except a comment.

External deps:  pandas, numpy, pulp  (CBC packaged with pulp).
"""

import os, sys, math, itertools
import pandas as pd
import numpy  as np
import pulp
from   pulp import PULP_CBC_CMD
import time

# ---------------------------------------------------------------------------
#  Utilities:  zonotope helpers + order‑reduction
# ---------------------------------------------------------------------------

def bound(row: pd.Series):
    """Return (lower, upper) of a **scalar** zonotope stored in *row*."""
    c = row['center']
    coeff_cols = [col for col in row.index if col.startswith('g')]
    coeffs     = row[coeff_cols].fillna(0.0).values   # NaN → 0
    abs_sum    = np.sum(np.abs(coeffs))
    return c - abs_sum, c + abs_sum


def _extract_generators(row: pd.Series):
    """Yield pairs  (coeff, idx_name)  until g{j}/idx{j} stops."""
    j = 1
    while f"g{j}" in row and f"idx{j}" in row and not (pd.isna(row[f"g{j}"]) or pd.isna(row[f"idx{j}"])):
        yield float(row[f"g{j}"]), int(row[f"idx{j}"])
        j += 1


def order_reduce_row(row: pd.Series, k: int = 1, idx_seed: int = 10_000_000):
    """Return *new* row Series with at most  k  + 1  generators.

    Strategy (scalar Box‑method):
        1. sort generators by |g|  descending.
        2. keep largest k as‑is.
        3. fold the rest into  g_fold = Σ |g_rest|, assign it a *fresh*
           error‑variable index so that its uncertainty is still modelled.
    """
    gens = [(abs(g), g, idx) for g, idx in _extract_generators(row)]
    if len(gens) <= k:
        return row                # nothing to reduce

    gens.sort(reverse=True)       # by |g|
    keep   = gens[:k]
    folded = gens[k:]
    g_fold = sum(abs_g for abs_g, _, _ in folded)

    # build new row ---------------------------------------------------------
    new_row           = row.copy()
    # first drop all old g*/idx*
    for j in itertools.count(1):
        g_key, idx_key = f"g{j}", f"idx{j}"
        if g_key not in new_row:  break
        new_row.pop(g_key)
        new_row.pop(idx_key)

    # re‑insert kept generators
    for j, (_, g_val, idx_val) in enumerate(keep, start=1):
        new_row[f"g{j}"]   = g_val
        new_row[f"idx{j}"] = idx_val

    # append folded generator (if any) as the last one ----------------------
    j_last                  = len(keep) + 1
    new_row[f"g{j_last}"]   = g_fold
    new_row[f"idx{j_last}"] = idx_seed + int(row.name)   # unique per row

    return new_row


def order_reduce_df(df_y: pd.DataFrame, k: int = 1) -> pd.DataFrame:
    """Apply order_reduce_row to the entire y‑DataFrame and return a copy."""
    reduced_rows = [order_reduce_row(row, k) for _, row in df_y.iterrows()]
    return pd.DataFrame(reduced_rows).reset_index(drop=True)

# ---------------------------------------------------------------------------
#  MILP core (largely kept from user code, with minor refactor)
# ---------------------------------------------------------------------------

def compute_count_with_constraint_tight(df_y: pd.DataFrame,
                                        df_x: pd.DataFrame,
                                        df_z: pd.DataFrame,
                                        obj: str = 'min',
                                        epsilon: float = 1e-6,
                                        relax: bool = False):
    """Solve the tightened MILP after order‑reduction (x, z are constants)."""
    sense = pulp.LpMinimize if obj == 'min' else pulp.LpMaximize
    prob  = pulp.LpProblem(f"Count_Rows_{obj}", sense)

    total_cnt, i_vars, e_vars = 0, {}, {}

    for idx, row in df_y.iterrows():
        x_thr = df_x.iloc[idx]['center']
        z_thr = df_z.iloc[idx]['center']

        y_lb, y_ub = bound(row)
        c          = row['center']

        if y_ub < x_thr and y_lb > z_thr:
            total_cnt += 1
            continue
        if y_lb >= x_thr or y_ub <= z_thr:
            continue

        i_var = pulp.LpVariable(f"i_{idx}", lowBound=0, upBound=1,
                                cat='Continuous' if relax else 'Binary')
        i_vars[idx] = i_var
        total_cnt  += i_var

        err_expr, j = 0, 1
        while f"g{j}" in row and f"idx{j}" in row and not pd.isna(row[f"g{j}"]):
            coeff = float(row[f"g{j}"])
            key   = int(row[f"idx{j}"])
            if key not in e_vars:
                e_vars[key] = pulp.LpVariable(f"e_{key}", lowBound=-1, upBound=1, cat='Continuous')
            err_expr += coeff * e_vars[key]
            j += 1
        y_expr = c + err_expr

        M1 = max(0.0, y_ub - (x_thr - epsilon))
        M2 = max(0.0, (z_thr + epsilon) - y_lb)

        prob += y_expr <= x_thr - epsilon + M1 * (1 - i_var)
        prob += y_expr >= z_thr + epsilon - M2 * (1 - i_var)

    prob += total_cnt
    prob.solve(PULP_CBC_CMD(msg=0))

    return (pulp.LpStatus[prob.status],
            int(round(pulp.value(prob.objective))),
            {idx: pulp.value(v) for idx, v in i_vars.items()})

# ---------------------------------------------------------------------------
#  Demo / CLI
# ---------------------------------------------------------------------------

def main():
    """Example run that loads parquet files, reduces order, solves MILP."""
    num_groups, size = 1, 1000
    base_path        = "./Data/COUNT"
    f_x = f"{base_path}/count_x_{num_groups}_groups_size_{size}.parquet"
    f_y = f"{base_path}/count_y_{num_groups}_groups_size_{size}.parquet"
    f_z = f"{base_path}/count_z_{num_groups}_groups_size_{size}.parquet"

    try:
        df_x, df_y, df_z = (pd.read_parquet(f) for f in (f_x, f_y, f_z))
        print("Parquet files loaded.")
    except Exception as exc:
        print("[Error] Could not load parquet files:", exc)
        return

    # 1) *Baseline* without reduction --------------------------------------
    start_time = time.time()
    s_min, c_min, _ = compute_count_with_constraint_tight(df_y, df_x, df_z, 'min', relax=True)
    s_max, c_max, _ = compute_count_with_constraint_tight(df_y, df_x, df_z, 'max', relax=True)
    print(f"Baseline bounds  : [{c_min}, {c_max}] (status {s_min}/{s_max})")
    print(f"Time elapsed     : {time.time() - start_time:.2f} seconds")

    # 2) Order‑reduce y (keep k = 1 generator) -----------------------------
    start_time = time.time()
    df_y_red = order_reduce_df(df_y, k=1)

    s_min, c_min, _ = compute_count_with_constraint_tight(df_y_red, df_x, df_z, 'min', relax=True)
    s_max, c_max, _ = compute_count_with_constraint_tight(df_y_red, df_x, df_z, 'max', relax=True)
    print(f"After reduction  : [{c_min}, {c_max}] (status {s_min}/{s_max})")
    print(f"Time elapsed     : {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
