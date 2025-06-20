#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pulp
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional, Callable
from pulp import PULP_CBC_CMD


class Zonotope:
    """Zonotope class representing uncertain numerical values"""
    
    def __init__(self, center: float, generators: Dict[int, float] = None):
        self.center = center
        self.generators = generators or {}
    
    @classmethod
    def from_series(cls, series: pd.Series):
        """Create Zonotope from pandas Series"""
        center = series.get('center', 0.0)
        generators = {}
        
        i = 1
        while f'g{i}' in series and f'idx{i}' in series:
            if pd.notna(series[f'g{i}']) and pd.notna(series[f'idx{i}']):
                err_idx = int(series[f'idx{i}'])
                generators[err_idx] = series[f'g{i}']
            i += 1
        
        return cls(center, generators)
    
    def to_series(self) -> pd.Series:
        """Convert to pandas Series"""
        result = {'center': self.center}
        for i, (err_idx, coeff) in enumerate(sorted(self.generators.items()), 1):
            result[f'g{i}'] = coeff
            result[f'idx{i}'] = err_idx
        return pd.Series(result)
    
    def bound(self) -> Tuple[float, float]:
        """Compute lower and upper bounds of the zonotope"""
        abs_sum = sum(abs(coeff) for coeff in self.generators.values())
        return self.center - abs_sum, self.center + abs_sum
    
    def __add__(self, other):
        if isinstance(other, Zonotope):
            new_generators = self.generators.copy()
            for err_idx, coeff in other.generators.items():
                new_generators[err_idx] = new_generators.get(err_idx, 0) + coeff
            return Zonotope(self.center + other.center, new_generators)
        else:
            return Zonotope(self.center + other, self.generators.copy())
    
    def __mul__(self, scalar):
        new_generators = {err_idx: coeff * scalar for err_idx, coeff in self.generators.items()}
        return Zonotope(self.center * scalar, new_generators)
    
    def __truediv__(self, scalar):
        if scalar == 0:
            raise ValueError("Division by zero")
        return self * (1.0 / scalar)


class Relation:
    """Relation class representing a table with uncertain data"""
    
    def __init__(self, data: pd.DataFrame, schema: Dict[str, str] = None):
        self.data = data
        self.schema = schema or {}  # Column name to type mapping
    
    def get_zonotope_column(self, col_name: str) -> List[Zonotope]:
        """Get zonotope list for the specified column"""
        zonotopes = []
        for _, row in self.data.iterrows():
            if col_name in row:
                if isinstance(row[col_name], (int, float)):
                    zonotopes.append(Zonotope(float(row[col_name])))
                else:
                    zonotopes.append(Zonotope.from_series(row))
            else:
                # Assume the entire row is one zonotope
                zonotopes.append(Zonotope.from_series(row))
        return zonotopes
    
    def __len__(self):
        return len(self.data)


class RelationalOperator(ABC):
    """Abstract base class for relational algebra operators"""
    
    @abstractmethod
    def execute(self, *relations: Relation) -> Union[Relation, Tuple[float, float]]:
        """Execute the operation"""
        pass


class Selection(RelationalOperator):
    """Selection operation σ"""
    
    def __init__(self, predicate: Callable, optimize_objective: str = 'count', relax: bool = True):
        self.predicate = predicate
        self.optimize_objective = optimize_objective  # 'count', 'sum', etc.
        self.relax = relax
    
    def execute(self, relation: Relation) -> Tuple[float, float]:
        """Execute selection operation, return lower and upper bounds of record count"""
        return self._optimize_selection(relation, 'min'), self._optimize_selection(relation, 'max')
    
    def _optimize_selection(self, relation: Relation, obj_type: str) -> float:
        """Use mixed integer optimization for selection"""
        if obj_type == 'min':
            prob = pulp.LpProblem("Selection_Min", pulp.LpMinimize)
        else:
            prob = pulp.LpProblem("Selection_Max", pulp.LpMaximize)
        
        total_expr = 0
        e_vars = {}
        
        for idx, row in relation.data.iterrows():
            z = Zonotope.from_series(row)
            lb, ub = z.bound()
            
            # Check predicate result
            pred_result = self.predicate(z, lb, ub)
            
            if pred_result == 'definitely_in':
                total_expr += 1
            elif pred_result == 'definitely_out':
                continue
            else:  # uncertain
                if self.relax:
                    i_var = pulp.LpVariable(f"i_{idx}", lowBound=0, upBound=1, cat='Continuous')
                else:
                    i_var = pulp.LpVariable(f"i_{idx}", lowBound=0, upBound=1, cat='Binary')
                
                total_expr += i_var
                
                # Add constraints
                self._add_predicate_constraints(prob, z, i_var, e_vars, idx)
        
        prob += total_expr
        
        solver = PULP_CBC_CMD(msg=0)
        prob.solve(solver)
        
        return pulp.value(prob.objective) if prob.status == 1 else 0


    def _add_predicate_constraints(self, prob, z: Zonotope, i_var, e_vars: Dict, idx: int):
        """Add predicate constraints using Big-M encoding"""
        # Build zonotope expression
        z_expr = z.center
        for err_idx, coeff in z.generators.items():
            if err_idx not in e_vars:
                e_vars[err_idx] = pulp.LpVariable(f"e_{err_idx}", lowBound=-1, upBound=1, cat='Continuous')
            z_expr += coeff * e_vars[err_idx]
        
        # This is a placeholder - actual implementation would use the BigMEncoder
        # from the Big-M encoding module to properly encode complex predicates
        # For now, we'll implement a simple threshold comparison
        
        # Example implementation for z > threshold case:
        # The predicate function should provide the necessary constraint information
        # This would be replaced by proper Big-M encoding in real usage
        
        M = 1e6  # Big-M constant
        epsilon = 1e-6
        
        # If predicate involves a threshold comparison, implement basic Big-M constraint
        # This is simplified - full implementation would parse the predicate structure
        # and generate appropriate constraints using BigMEncoder
        
        # For uncertain cases where we need to enforce the predicate conditionally:
        # If i_var = 1, then predicate must be satisfied
        # If i_var = 0, then predicate constraint is relaxed by Big-M
        
        # Note: Actual constraint addition would depend on the specific predicate
        # This is where BigMEncoder from the separate module would be utilized
        pass


class Projection(RelationalOperator):
    """Projection operation π"""
    
    def __init__(self, columns: List[str], rename_map: Dict[str, str] = None):
        self.columns = columns
        self.rename_map = rename_map or {}
    
    def execute(self, relation: Relation) -> Relation:
        """Execute projection operation"""
        projected_data = relation.data[self.columns].copy()
        
        # Rename columns
        if self.rename_map:
            projected_data = projected_data.rename(columns=self.rename_map)
        
        new_schema = {self.rename_map.get(col, col): relation.schema.get(col, 'unknown') 
                     for col in self.columns}
        
        return Relation(projected_data, new_schema)


class Aggregation(RelationalOperator):
    """Aggregation operation Γ"""
    
    def __init__(self, agg_func: str, column: str, group_by: List[str] = None, 
                 threshold: Union[float, Zonotope] = None, relax: bool = True):
        self.agg_func = agg_func  # 'sum', 'count', 'avg', etc.
        self.column = column
        self.group_by = group_by or []
        self.threshold = threshold
        self.relax = relax
    
    def execute(self, relation: Relation) -> Union[Relation, Tuple[float, float]]:
        """Execute aggregation operation"""
        if self.agg_func == 'sum':
            return self._compute_sum_with_threshold(relation)
        elif self.agg_func == 'count':
            return self._compute_count(relation)
        elif self.agg_func == 'avg':
            return self._compute_average(relation)
        else:
            raise ValueError(f"Unsupported aggregation function: {self.agg_func}")
    
    def _compute_sum_with_threshold(self, relation: Relation) -> Tuple[float, float]:
        """Compute SUM with threshold"""
        min_val = self._optimize_sum(relation, 'min')
        max_val = self._optimize_sum(relation, 'max')
        return min_val, max_val
    
    def _optimize_sum(self, relation: Relation, obj_type: str) -> float:
        """Optimize SUM computation"""
        if obj_type == 'min':
            prob = pulp.LpProblem("Sum_Min", pulp.LpMinimize)
        else:
            prob = pulp.LpProblem("Sum_Max", pulp.LpMaximize)
        
        total_expr = 0
        gen_exprs = {}
        e_vars_global = {}
        
        # Handle threshold
        if self.threshold is not None:
            if isinstance(self.threshold, (int, float)):
                t_lb = t_ub = float(self.threshold)
            else:
                t_lb, t_ub = self.threshold.bound()
                # Add threshold constraint variables
                threshold_expr = self.threshold.center
                for err_idx, coeff in self.threshold.generators.items():
                    if err_idx not in e_vars_global:
                        e_vars_global[err_idx] = pulp.LpVariable(
                            f"e_threshold_{err_idx}", lowBound=-1, upBound=1, cat='Continuous'
                        )
                    threshold_expr += coeff * e_vars_global[err_idx]
        else:
            t_lb = -np.inf
            t_ub = np.inf
        
        for idx, row in relation.data.iterrows():
            z = Zonotope.from_series(row)
            lb, ub = z.bound()
            
            if self.threshold is not None:
                if ub <= t_lb:
                    continue  # Definitely does not satisfy condition
                elif lb >= t_ub:
                    # Definitely satisfies condition
                    total_expr += z.center
                    for err_idx, coeff in z.generators.items():
                        if err_idx in gen_exprs:
                            gen_exprs[err_idx] += coeff
                        else:
                            gen_exprs[err_idx] = coeff
                        if err_idx not in e_vars_global:
                            e_vars_global[err_idx] = pulp.LpVariable(
                                f"e_{err_idx}", lowBound=-1, upBound=1, cat='Continuous'
                            )
                else:
                    # Uncertain case
                    if self.relax:
                        i_var = pulp.LpVariable(f"i_{idx}", lowBound=0, upBound=1, cat='Continuous')
                    else:
                        i_var = pulp.LpVariable(f"i_{idx}", lowBound=0, upBound=1, cat='Binary')
                    
                    total_expr += z.center * i_var
                    
                    # Handle generators
                    for err_idx, coeff in z.generators.items():
                        if err_idx not in e_vars_global:
                            e_vars_global[err_idx] = pulp.LpVariable(
                                f"e_{err_idx}", lowBound=-1, upBound=1, cat='Continuous'
                            )
                        
                        v_var = pulp.LpVariable(f"v_{idx}_{err_idx}", lowBound=None, upBound=None, cat='Continuous')
                        
                        # Big-M constraints
                        if coeff >= 0:
                            prob += v_var >= coeff * (-1) * i_var
                            prob += v_var <= coeff * (+1) * i_var
                            prob += v_var >= coeff * e_vars_global[err_idx] - coeff * (+1) * (1 - i_var)
                            prob += v_var <= coeff * e_vars_global[err_idx] - coeff * (-1) * (1 - i_var)
                        else:
                            prob += v_var >= coeff * (+1) * i_var
                            prob += v_var <= coeff * (-1) * i_var
                            prob += v_var >= coeff * e_vars_global[err_idx] - coeff * (-1) * (1 - i_var)
                            prob += v_var <= coeff * e_vars_global[err_idx] - coeff * (+1) * (1 - i_var)
                        
                        total_expr += v_var
                    
                    # Add threshold constraints
                    z_expr = z.center + pulp.lpSum(coeff * e_vars_global[err_idx] for err_idx, coeff in z.generators.items())
                    M = max(0.0, (t_ub + 1e-6) - lb)
                    prob += (z_expr - threshold_expr) >= (1e-6 - M * (1 - i_var))
                    prob += (z_expr - threshold_expr) <= M * i_var
            else:
                # No threshold, add directly
                total_expr += z.center
                for err_idx, coeff in z.generators.items():
                    if err_idx in gen_exprs:
                        gen_exprs[err_idx] += coeff
                    else:
                        gen_exprs[err_idx] = coeff
                    if err_idx not in e_vars_global:
                        e_vars_global[err_idx] = pulp.LpVariable(
                            f"e_{err_idx}", lowBound=-1, upBound=1, cat='Continuous'
                        )
        
        # Add generator terms
        gen_term = pulp.lpSum([coef * e_vars_global[err_idx] for err_idx, coef in gen_exprs.items()])
        prob += total_expr + gen_term
        
        solver = PULP_CBC_CMD(msg=0)
        prob.solve(solver)
        
        return pulp.value(prob.objective) if prob.status == 1 else 0
    
    def _compute_count(self, relation: Relation) -> Tuple[float, float]:
        """Compute COUNT"""
        # Reuse Selection logic
        selection = Selection(self._default_predicate, relax=self.relax)
        return selection.execute(relation)
    
    def _compute_average(self, relation: Relation) -> Zonotope:
        """Compute average value"""
        total_center = 0.0
        total_gens = {}
        
        for _, row in relation.data.iterrows():
            z = Zonotope.from_series(row)
            total_center += z.center
            for err_idx, coeff in z.generators.items():
                total_gens[err_idx] = total_gens.get(err_idx, 0) + coeff
        
        # Divide by number of records
        avg_center = total_center / len(relation)
        avg_gens = {err_idx: coeff / len(relation) for err_idx, coeff in total_gens.items()}
        
        return Zonotope(avg_center, avg_gens)
    
    def _default_predicate(self, z: Zonotope, lb: float, ub: float) -> str:
        """Default predicate, always returns uncertain"""
        return 'uncertain'


class CartesianProduct(RelationalOperator):
    """Cartesian product operation ×"""
    
    def execute(self, relation1: Relation, relation2: Relation) -> Relation:
        """Execute Cartesian product"""
        result_data = []
        
        for _, row1 in relation1.data.iterrows():
            for _, row2 in relation2.data.iterrows():
                combined_row = pd.concat([row1, row2])
                result_data.append(combined_row)
        
        result_df = pd.DataFrame(result_data)
        
        # Merge schemas
        combined_schema = relation1.schema.copy()
        combined_schema.update(relation2.schema)
        
        return Relation(result_df, combined_schema)


class RelationalAlgebraEngine:
    """Relational algebra execution engine"""
    
    def __init__(self):
        self.epsilon = 1e-6
        self.big_m = 1e3
    
    def execute_query(self, operations: List[RelationalOperator], 
                     relations: List[Relation]) -> Union[Relation, Tuple[float, float]]:
        """Execute a series of relational algebra operations"""
        result = relations[0] if relations else None
        
        for op in operations:
            if isinstance(op, (Selection, Aggregation)):
                result = op.execute(result)
            elif isinstance(op, Projection):
                result = op.execute(result)
            elif isinstance(op, CartesianProduct) and len(relations) > 1:
                result = op.execute(result, relations[1])
            else:
                raise ValueError(f"Unsupported operation: {type(op)}")
        
        return result
    
    def create_threshold_predicate(self, threshold_value: Union[float, Zonotope], 
                                 comparison: str = '>') -> Callable:
        """Create threshold comparison predicate"""
        def predicate(z: Zonotope, lb: float, ub: float) -> str:
            if isinstance(threshold_value, (int, float)):
                t_lb = t_ub = float(threshold_value)
            else:
                t_lb, t_ub = threshold_value.bound()
            
            if comparison == '>':
                if lb > t_ub:
                    return 'definitely_in'
                elif ub <= t_lb:
                    return 'definitely_out'
                else:
                    return 'uncertain'
            elif comparison == '<':
                if ub < t_lb:
                    return 'definitely_in'
                elif lb >= t_ub:
                    return 'definitely_out'
                else:
                    return 'uncertain'
            else:
                return 'uncertain'
        
        return predicate


# Usage example
def example_usage():
    """Usage example"""
    # Create test data
    data = [
        {"center": 6000, "g1": 500, "idx1": 1},
        {"center": 7000, "g1": 400, "idx1": 2},
        {"center": 5500, "g1": 300, "idx1": 3}
    ]
    df = pd.DataFrame(data)
    relation = Relation(df, {"salary": "zonotope"})
    
    # Create execution engine
    engine = RelationalAlgebraEngine()
    
    # Compute average
    avg_agg = Aggregation('avg', 'salary')
    avg_result = avg_agg.execute(relation)
    print(f"Average salary zonotope: center={avg_result.center}, generators={avg_result.generators}")
    
    # Compute SUM(salary) WHERE salary > AVG(salary)
    sum_with_threshold = Aggregation('sum', 'salary', threshold=avg_result)
    min_sum, max_sum = sum_with_threshold.execute(relation)
    print(f"SUM(salary) WHERE salary > AVG(salary): [{min_sum}, {max_sum}]")
    
    # Projection operation
    projection = Projection(['center'])
    projected_relation = projection.execute(relation)
    print(f"Projected relation shape: {projected_relation.data.shape}")


if __name__ == "__main__":
    example_usage()