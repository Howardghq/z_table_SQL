import pandas as pd
import numpy as np

class R:
    def __init__(self, lb, ub):
        assert(lb<=ub)
        self.lb = lb
        self.ub = ub
        
    def __add__(self, o):
        if isinstance(o, self.__class__):
            return R(self.lb+o.lb, self.ub+o.ub)
        elif isinstance(o, (int, float)):
            return R(self.lb+o, self.ub+o)
        else:
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'").format(self.__class__, type(o))
        
    
    def __and__(self, o):
        if not (isinstance(self.lb, bool) and isinstance(self.ub, bool) and
                isinstance(o.lb, bool) and isinstance(o.ub, bool)):
            raise ValueError("Both operands must be R objects with boolean bounds.")
        
        # Calculate the new bounds
        new_lb = self.lb and o.lb
        new_ub = self.ub and o.ub
        
        return R(new_lb, new_ub)
    
    def __or__(self, o):
        if not (isinstance(self.lb, bool) and isinstance(self.ub, bool) and
                isinstance(o.lb, bool) and isinstance(o.ub, bool)):
            raise ValueError("Both operands must be R objects with boolean bounds.")
        
        # Calculate the new bounds
        new_lb = self.lb or o.lb
        new_ub = self.ub or o.ub
        
        return R(new_lb, new_ub)
    
    def __mul__(self, o):
        if isinstance(o, self.__class__):
            lb = min(self.lb*o.lb, self.lb*o.ub, self.ub*o.lb, self.ub*o.ub)
            ub = max(self.lb*o.lb, self.lb*o.ub, self.ub*o.lb, self.ub*o.ub)
            return R(lb, ub)
        elif isinstance(o, (int, float)):
            lb = min(self.lb*o, self.ub*o)
            ub = max(self.lb*o, self.ub*o)
            return R(lb, ub)
        else:
            raise TypeError("unsupported operand type(s) for *: '{}' and '{}'").format(self.__class__, type(o))
    
    def __truediv__(self, o):
        if isinstance(o, (int, float)):
            if o == 0:
                raise ValueError("Division by zero.")
            candidates = [self.lb / o, self.ub / o]
            return R(min(candidates), max(candidates))
        else:
            raise TypeError("Can only divide by a number.")

    def __hash__(self):
        # Combine the lower and upper bounds into a hashable representation
        return hash(self.lb, self.ub)
    
    def __eq__(self, o):
        return R(self.lb==self.ub and self.lb==o.lb and o.lb==o.ub, self.i(o) is not None)

    def __gt__(self, o):
        return R(self.lb > o.ub, self.ub > o.lb)

    def __ge__(self, o):
        return R(self.lb >= o.ub, self.ub >= o.lb)
    
    def __lt__(self, o):
        return R(self.lb < o.ub, self.ub < o.lb)

    def __le__(self, o):
        return R(self.lb <= o.ub, self.ub <= o.lb)

    def u(self, o):
        return R(min(self.lb,o.lb), max(self.ub,o.ub))
    
    def i(self, o):
        lb = max(self.lb,o.lb)
        ub = min(self.ub,o.ub)
        if lb <= ub:
            return R(max(self.lb,o.lb), min(self.ub,o.ub))
        return None
    
    def __eq__(self, other):
        if self.ub >= other.lb and other.ub >= self.lb:
            return True
        return False
    
    def __repr__(self):
        return f"[{self.lb}, {self.ub}]"
    
    def __str__(self):
        return f"[{self.lb}, {self.ub}]"

def bound_from_row(row):
    c = row['center']
    gen_keys = [k for k in row.index if k.startswith('g') and pd.notna(row[k])]
    coeffs = [abs(row[k]) for k in gen_keys]
    return R(c - sum(coeffs), c + sum(coeffs))

def interval_sum_with_filter(df):
    # Convert each row to an interval.
    intervals = [bound_from_row(row) for _, row in df.iterrows()]
    
    n = len(intervals)
    if not intervals:
        return R(0,0)
    
    total = R(0,0)
    for r in intervals:
        total = total + r
    avg = total / n
    
    print("Computed average interval:", avg)
    
    selected = []
    for r in intervals:
        comp = r > avg
        if comp.ub != False:
            adjust_r = R(0, r.ub) if comp.lb == False else r
            selected.append(adjust_r)
    
    result = R(0,0)
    for r in selected:
        result = result + r
    return result

def main():
    """
    # Average: [5766.666666666667, 6566.666666666667]
    data = [
        {"center": 6000, "g1": 500, "idx1": 1}, #[5500, 6500] possibly in
        {"center": 7000, "g1": 400, "idx1": 2}, #[6600, 7400] definitely in
        {"center": 5500, "g1": 300, "idx1": 3}  #[5200, 5800] possibly in
    ]
    dataset_original = pd.DataFrame(data)
    print("Salary zonotopes:")
    print(dataset_original)
    """
    num_groups = 1
    size = 1000
    nvar = 1
    input_file = f'./Data/AVG/z_{num_groups}_groups_size_{size}_error_num_{nvar}_.parquet'
    print(f"Reading zonotope data from {input_file}...")
    try:
        dataset_original = pd.read_parquet(input_file)
        print("Zonotope data successfully loaded.")
    except Exception as e:
        raise ValueError(f"Error reading zonotope data: {e}")
    
    total = R(0,0)
    for _, row in dataset_original.iterrows():
        total = total + bound_from_row(row)
    avg = total / len(dataset_original)
    print("\nAverage Salary Interval:", avg)
    
    result = interval_sum_with_filter(dataset_original)
    print("\nResulting Sum Interval (salary > average):", result)

if __name__ == "__main__":
    main()