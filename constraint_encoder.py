#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pulp
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional, Callable, Any
from pulp import PULP_CBC_CMD
from enum import Enum
from relational_algebra import Zonotope

class ExpressionType(Enum):
    ZONOTOPE = "zonotope"
    CONSTANT = "constant"
    COMPARISON = "comparison"
    LOGICAL_AND = "logical_and"
    LOGICAL_OR = "logical_or"
    LOGICAL_NOT = "logical_not"
    ADDITION = "addition"

class Expression:
    
    def __init__(self, expr_type: ExpressionType, value: Any = None):
        self.expr_type = expr_type
        self.value = value
        self.numeric_var = None  # v - continuous variable for numeric value
        self.boolean_var = None  # b - binary variable for boolean truth value
        self.id = None

class ZonotopeExpression(Expression):
    
    def __init__(self, zonotope, row_idx: int):
        super().__init__(ExpressionType.ZONOTOPE, zonotope)
        self.zonotope = zonotope
        self.row_idx = row_idx
        self.id = f"zono_{row_idx}"

class ComparisonExpression(Expression):
    
    def __init__(self, left_expr: Expression, right_expr: Expression, operator: str):
        super().__init__(ExpressionType.COMPARISON)
        self.left = left_expr
        self.right = right_expr
        self.operator = operator  # '<', '<=', '>', '>=', '==', '!='
        self.id = f"cmp_{left_expr.id}_{operator}_{right_expr.id}"

class LogicalExpression(Expression):
    
    def __init__(self, expr_type: ExpressionType, operands: List[Expression]):
        super().__init__(expr_type)
        self.operands = operands
        if expr_type == ExpressionType.LOGICAL_AND:
            self.id = f"and_{'_'.join([op.id for op in operands])}"
        elif expr_type == ExpressionType.LOGICAL_OR:
            self.id = f"or_{'_'.join([op.id for op in operands])}"
        elif expr_type == ExpressionType.LOGICAL_NOT:
            self.id = f"not_{operands[0].id}"

class ArithmeticExpression(Expression):
    
    def __init__(self, expr_type: ExpressionType, operands: List[Expression]):
        super().__init__(expr_type)
        self.operands = operands
        if expr_type == ExpressionType.ADDITION:
            self.id = f"add_{'_'.join([op.id for op in operands])}"

class BigMEncoder:
    
    def __init__(self, big_m: float = 1e6):
        self.big_m = big_m
        self.expressions = {}  # id -> Expression
        self.e_vars = {}  # error_idx -> PuLP variable
        
    def create_variables(self, prob: pulp.LpProblem, expr: Expression):
        if expr.id not in self.expressions:
            self.expressions[expr.id] = expr
            
        # Create numeric variable (always needed)
        expr.numeric_var = pulp.LpVariable(
            f"v_{expr.id}", lowBound=None, upBound=None, cat='Continuous'
        )
        
        # Create boolean variable for non-zonotope expressions
        if expr.expr_type != ExpressionType.ZONOTOPE:
            expr.boolean_var = pulp.LpVariable(
                f"b_{expr.id}", lowBound=0, upBound=1, cat='Binary'
            )
    
    def encode_zonotope_expression(self, prob: pulp.LpProblem, expr: ZonotopeExpression):
        z = expr.zonotope
        
        # Build the zonotope linear expression: v = center + sum(coeff_i * e_i)
        zonotope_expr = z.center
        
        for err_idx, coeff in z.generators.items():
            if err_idx not in self.e_vars:
                self.e_vars[err_idx] = pulp.LpVariable(
                    f"e_{err_idx}", lowBound=-1, upBound=1, cat='Continuous'
                )
            zonotope_expr += coeff * self.e_vars[err_idx]
        
        # Constraint: v_expr = zonotope_expr
        prob += expr.numeric_var == zonotope_expr
        
        return expr.numeric_var
    
    def encode_comparison_expression(self, prob: pulp.LpProblem, expr: ComparisonExpression):

        self.create_variables(prob, expr)
        
        # Ensure child expressions are encoded
        if expr.left.numeric_var is None:
            self.encode_expression(prob, expr.left)
        if expr.right.numeric_var is None:
            self.encode_expression(prob, expr.right)
            
        v1 = expr.left.numeric_var
        v2 = expr.right.numeric_var
        b = expr.boolean_var
        M = self.big_m
        
        if expr.operator == '<':
            # c := c1 < c2
            prob += v1 - v2 + b * M >= 0
            prob += v2 - v1 + (1 - b) * M >= 1e-6
            
        elif expr.operator == '<=':
            # c := c1 <= c2
            prob += v1 - v2 + b * M >= 1e-6 
            prob += v2 - v1 + (1 - b) * M >= 0
            
        elif expr.operator == '>':
            # c := c1 > c2 (equivalent to c2 < c1)
            prob += v2 - v1 + b * M >= 0
            prob += v1 - v2 + (1 - b) * M >= 1e-6
            
        elif expr.operator == '>=':
            # c := c1 >= c2 (equivalent to c2 <= c1)
            prob += v2 - v1 + b * M >= 1e-6
            prob += v1 - v2 + (1 - b) * M >= 0
            
        elif expr.operator == '==':
            # c := c1 == c2 (both directions must be <=)
            b_leq = pulp.LpVariable(f"b_leq_{expr.id}", lowBound=0, upBound=1, cat='Binary')
            b_geq = pulp.LpVariable(f"b_geq_{expr.id}", lowBound=0, upBound=1, cat='Binary')
            
            # c1 <= c2
            prob += v1 - v2 + b_leq * M >= 1e-6
            prob += v2 - v1 + (1 - b_leq) * M >= 0
            
            # c1 >= c2  
            prob += v2 - v1 + b_geq * M >= 1e-6
            prob += v1 - v2 + (1 - b_geq) * M >= 0
            
            prob += b == b_leq
            prob += b == b_geq
            
        # The numeric value is not directly meaningful for comparisons
        # Set it to be equal to the boolean value for consistency
        prob += expr.numeric_var == expr.boolean_var
        
        return expr.boolean_var
    
    def encode_logical_and(self, prob: pulp.LpProblem, expr: LogicalExpression):
        """
        For c := c1 ∧ c2:
        - b1 + b2 - 2b - 1 <= 0
        - b1 + b2 - 2b >= 0
        """
        self.create_variables(prob, expr)
        
        # Ensure all operands are encoded
        operand_bools = []
        for operand in expr.operands:
            if operand.boolean_var is None:
                self.encode_expression(prob, operand)
            operand_bools.append(operand.boolean_var)
        
        b = expr.boolean_var
        
        if len(operand_bools) == 2:
            b1, b2 = operand_bools
            prob += b1 + b2 - 2*b - 1 <= 0
            prob += b1 + b2 - 2*b >= 0
        else:
            # For more than 2 operands, use generalized form
            # b = 1 iff all operands are 1
            n = len(operand_bools)
            prob += pulp.lpSum(operand_bools) - n*b >= 0
            prob += pulp.lpSum(operand_bools) - n*b <= n - 1
        
        # Numeric value equals boolean value
        prob += expr.numeric_var == expr.boolean_var
        
        return expr.boolean_var
    
    def encode_logical_or(self, prob: pulp.LpProblem, expr: LogicalExpression):
        """
        For c := c1 ∨ c2:
        - b1 + b2 - 2b <= 0  
        - b1 + b2 - b >= 0
        """
        self.create_variables(prob, expr)
        
        # Ensure all operands are encoded
        operand_bools = []
        for operand in expr.operands:
            if operand.boolean_var is None:
                self.encode_expression(prob, operand)
            operand_bools.append(operand.boolean_var)
        
        b = expr.boolean_var
        
        if len(operand_bools) == 2:
            b1, b2 = operand_bools
            prob += b1 + b2 - 2*b <= 0
            prob += b1 + b2 - b >= 0
        else:
            # For more than 2 operands
            # b = 1 iff at least one operand is 1
            prob += pulp.lpSum(operand_bools) - b >= 0
            prob += pulp.lpSum(operand_bools) - len(operand_bools)*b <= 0
        
        # Numeric value equals boolean value
        prob += expr.numeric_var == expr.boolean_var
        
        return expr.boolean_var
    
    def encode_logical_not(self, prob: pulp.LpProblem, expr: LogicalExpression):
        """
        For c := ¬c1:
        - b + b1 = 1
        """
        self.create_variables(prob, expr)
        
        # Ensure operand is encoded
        operand = expr.operands[0]
        if operand.boolean_var is None:
            self.encode_expression(prob, operand)
        
        b = expr.boolean_var
        b1 = operand.boolean_var
        
        prob += b + b1 == 1
        
        # Numeric value equals boolean value
        prob += expr.numeric_var == expr.boolean_var
        
        return expr.boolean_var
    
    def encode_addition(self, prob: pulp.LpProblem, expr: ArithmeticExpression):
        """
        For c := c1 + c2:
        - v1 + v2 - v = 0
        """
        self.create_variables(prob, expr)
        
        # Ensure all operands are encoded
        operand_numerics = []
        for operand in expr.operands:
            if operand.numeric_var is None:
                self.encode_expression(prob, operand)
            operand_numerics.append(operand.numeric_var)
        
        v = expr.numeric_var
        
        # v = sum of all operands
        prob += pulp.lpSum(operand_numerics) - v == 0
        
        return expr.numeric_var
    
    def encode_expression(self, prob: pulp.LpProblem, expr: Expression):
        if expr.expr_type == ExpressionType.ZONOTOPE:
            return self.encode_zonotope_expression(prob, expr)
        elif expr.expr_type == ExpressionType.COMPARISON:
            return self.encode_comparison_expression(prob, expr)
        elif expr.expr_type == ExpressionType.LOGICAL_AND:
            return self.encode_logical_and(prob, expr)
        elif expr.expr_type == ExpressionType.LOGICAL_OR:
            return self.encode_logical_or(prob, expr)
        elif expr.expr_type == ExpressionType.LOGICAL_NOT:
            return self.encode_logical_not(prob, expr)
        elif expr.expr_type == ExpressionType.ADDITION:
            return self.encode_addition(prob, expr)
        else:
            raise ValueError(f"Unsupported expression type: {expr.expr_type}")

class SmartPredicateFactory:
    
    def __init__(self, big_m: float = 1e6):
        self.big_m = big_m
    
    def create_threshold_predicate(self, threshold_value: Union[float, "Zonotope"], comparison: str = '>'):
        """Create a smart threshold predicate"""
        def predicate_evaluator(relation_data, optimize_type='count'):
            encoder = BigMEncoder(self.big_m)
            
            if optimize_type == 'min':
                prob = pulp.LpProblem("Smart_Predicate_Min", pulp.LpMinimize)
            else:
                prob = pulp.LpProblem("Smart_Predicate_Max", pulp.LpMaximize)
            
            total_expr = 0
            
            for idx, row in relation_data.iterrows():
                # Create zonotope expression for this row
                from relational_algebra import Zonotope  # Import here to avoid circular imports
                z = Zonotope.from_series(row)
                zono_expr = ZonotopeExpression(z, idx)
                
                # Create threshold expression (constant or zonotope)
                if isinstance(threshold_value, (int, float)):
                    # For constants, we can do simpler bound checking
                    lb, ub = z.bound()
                    if comparison == '>' and lb > threshold_value:
                        total_expr += 1  # Definitely in
                        continue
                    elif comparison == '>' and ub <= threshold_value:
                        continue  # Definitely out
                    # Otherwise, uncertain - need full encoding
                
                # Create comparison expression
                if isinstance(threshold_value, (int, float)):
                    # Create a constant expression for threshold
                    threshold_expr = Expression(ExpressionType.CONSTANT, threshold_value)
                    threshold_expr.numeric_var = pulp.LpVariable(
                        f"const_thresh_{idx}", lowBound=threshold_value, upBound=threshold_value, cat='Continuous'
                    )
                else:
                    threshold_expr = ZonotopeExpression(threshold_value, f"thresh_{idx}")
                
                comp_expr = ComparisonExpression(zono_expr, threshold_expr, comparison)
                
                # Encode the comparison
                boolean_result = encoder.encode_expression(prob, comp_expr)
                total_expr += boolean_result
            
            prob += total_expr
            
            solver = PULP_CBC_CMD(msg=0)
            prob.solve(solver)
            
            return pulp.value(prob.objective) if prob.status == 1 else 0
        
        return predicate_evaluator
    
    def create_compound_predicate(self, predicates: List, logic_op: str = 'AND'):
        """Create compound predicates with proper Big-M encoding"""
        def compound_evaluator(relation_data, optimize_type='count'):
            encoder = BigMEncoder(self.big_m)
            
            if optimize_type == 'min':
                prob = pulp.LpProblem("Compound_Predicate_Min", pulp.LpMinimize)
            else:
                prob = pulp.LpProblem("Compound_Predicate_Max", pulp.LpMaximize)
            
            total_expr = 0
            
            for idx, row in relation_data.iterrows():
                # Evaluate all sub-predicates for this row
                sub_expressions = []
                
                for pred_func in predicates:
                    # This is a simplified approach - in practice, you'd want to 
                    # convert each predicate function to an Expression object
                    pass
                
                # Create logical expression based on operation
                if logic_op == 'AND':
                    logical_expr = LogicalExpression(ExpressionType.LOGICAL_AND, sub_expressions)
                elif logic_op == 'OR':
                    logical_expr = LogicalExpression(ExpressionType.LOGICAL_OR, sub_expressions)
                
                # Encode and add to total
                boolean_result = encoder.encode_expression(prob, logical_expr)
                total_expr += boolean_result
            
            prob += total_expr
            
            solver = PULP_CBC_CMD(msg=0)
            prob.solve(solver)
            
            return pulp.value(prob.objective) if prob.status == 1 else 0
        
        return compound_evaluator

# Usage example
def demonstrate_bigm_encoding():
    """Demonstrate the Big-M encoding implementation"""
    
    # Example: Create expressions for x > 5 AND y <= 10
    encoder = BigMEncoder()
    prob = pulp.LpProblem("Demo", pulp.LpMaximize)
    
    # Create zonotope expressions
    z1 = Zonotope(6, {1: 1})  # x = 6 ± 1
    z2 = Zonotope(8, {2: 3})  # y = 8 ± 3
    
    x_expr = ZonotopeExpression(z1, 1)
    y_expr = ZonotopeExpression(z2, 2)
    
    # Create constant for threshold 5
    const_5 = Expression(ExpressionType.CONSTANT, 5)
    const_5.numeric_var = pulp.LpVariable("const_5", lowBound=5, upBound=5, cat='Continuous')
    
    # Create constant for threshold 10
    const_10 = Expression(ExpressionType.CONSTANT, 10)
    const_10.numeric_var = pulp.LpVariable("const_10", lowBound=10, upBound=10, cat='Continuous')
    
    # Create comparison expressions
    x_gt_5 = ComparisonExpression(x_expr, const_5, '>')
    y_leq_10 = ComparisonExpression(y_expr, const_10, '<=')
    
    # Create logical AND
    and_expr = LogicalExpression(ExpressionType.LOGICAL_AND, [x_gt_5, y_leq_10])
    
    # Encode everything
    result = encoder.encode_expression(prob, and_expr)
    prob += result
    
    print("Big-M encoding demonstration:")
    print(f"x = {z1.center} + {z1.generators}")
    print(f"y = {z2.center} + {z2.generators}")
    print("Condition: x > 5 AND y <= 10")
    print("Encoded as mixed-integer linear program with Big-M constraints")
    
    return prob, encoder

if __name__ == "__main__":
    demonstrate_bigm_encoding()