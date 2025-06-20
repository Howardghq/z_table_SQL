#!/usr/bin/env python3
"""
Usage Examples for Relational Algebra Framework
Demonstrates how to use the framework for processing complex SQL queries
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import sys
import os

# Assume framework code is in the same directory
from relational_algebra import *

class ComplexQueryProcessor:
    """Complex query processor demonstrating composition of multiple relational algebra operations"""
    
    def __init__(self):
        self.engine = RelationalAlgebraEngine()
    
    def process_sum_with_avg_threshold(self, salary_data: pd.DataFrame) -> Tuple[float, float]:
        """
        Process query: SELECT SUM(salary) FROM Employee WHERE salary > AVG(salary)
        Corresponding relational algebra: Γ_SUM,salary(σ_salary>avg(σ_y=c(T × π_s←0.3×sum_salary(Γ_SUM,salary→sum_salary(σ_y=c(T))))))
        """
        # 1. Create relation
        relation = Relation(salary_data, {"salary": "zonotope"})
        
        # 2. Compute average as threshold
        avg_agg = Aggregation('avg', 'salary')
        avg_threshold = avg_agg.execute(relation)
        
        # 3. Compute sum of salaries meeting condition
        sum_with_threshold = Aggregation('sum', 'salary', threshold=avg_threshold)
        min_sum, max_sum = sum_with_threshold.execute(relation)
        
        return min_sum, max_sum
    
    def process_count_with_conditions(self, x_data: pd.DataFrame, y_data: pd.DataFrame, 
                                    z_data: pd.DataFrame) -> Tuple[float, float]:
        """
        Process query: SELECT COUNT(*) FROM T WHERE x > y AND y > z
        """
        # Create combined relation
        combined_data = pd.concat([
            x_data.add_suffix('_x'),
            y_data.add_suffix('_y'), 
            z_data.add_suffix('_z')
        ], axis=1)
        
        relation = Relation(combined_data)
        
        # Create compound condition predicate
        def compound_predicate(z_y, lb_y, ub_y):
            # Here we need to access x and z values from the same row
            # Simplified handling, actual implementation requires more complex logic
            return 'uncertain'
        
        # Use selection operation
        selection = Selection(compound_predicate)
        min_count, max_count = selection.execute(relation)
        
        return min_count, max_count
    
    def process_projection_and_aggregation(self, employee_data: pd.DataFrame, 
                                         columns: List[str]) -> Relation:
        """
        Process projection and aggregation combination queries
        E.g.: SELECT department, AVG(salary) FROM Employee GROUP BY department
        """
        relation = Relation(employee_data)
        
        # 1. Projection operation
        projection = Projection(columns)
        projected_relation = projection.execute(relation)
        
        # 2. Group aggregation (simplified version)
        # Actual implementation requires more complex grouping logic
        
        return projected_relation
    
    def process_cartesian_product_query(self, table1: pd.DataFrame, 
                                      table2: pd.DataFrame) -> Relation:
        """
        Process queries involving Cartesian product
        E.g.: SELECT * FROM Table1, Table2 WHERE condition
        """
        relation1 = Relation(table1)
        relation2 = Relation(table2)
        
        # Cartesian product
        cartesian = CartesianProduct()
        result_relation = cartesian.execute(relation1, relation2)
        
        return result_relation


class AdvancedPredicates:
    """Advanced predicate definitions for complex WHERE conditions"""
    
    @staticmethod
    def create_range_predicate(min_val: float, max_val: float):
        """Create range predicate: min_val <= x <= max_val"""
        def predicate(z, lb, ub):
            if lb >= min_val and ub <= max_val:
                return 'definitely_in'
            elif ub < min_val or lb > max_val:
                return 'definitely_out'
            else:
                return 'uncertain'
        return predicate
    
    @staticmethod
    def create_comparison_predicate(other_column_data: List[float], 
                                  comparison: str = '>'):
        """Create inter-column comparison predicate"""
        def predicate(z, lb, ub):
            # Simplified implementation, actual requires more complex logic
            return 'uncertain'
        return predicate
    
    @staticmethod
    def create_compound_predicate(predicates: List, logic_op: str = 'AND'):
        """Create compound predicate (AND/OR combination)"""
        def predicate(z, lb, ub):
            results = [pred(z, lb, ub) for pred in predicates]
            
            if logic_op == 'AND':
                if all(r == 'definitely_in' for r in results):
                    return 'definitely_in'
                elif any(r == 'definitely_out' for r in results):
                    return 'definitely_out'
                else:
                    return 'uncertain'
            elif logic_op == 'OR':
                if any(r == 'definitely_in' for r in results):
                    return 'definitely_in'
                elif all(r == 'definitely_out' for r in results):
                    return 'definitely_out'
                else:
                    return 'uncertain'
            
            return 'uncertain'
        return predicate


class QueryOptimizer:
    """Query optimizer for optimizing execution order of relational algebra expressions"""
    
    def __init__(self):
        self.cost_model = {}
    
    def optimize_selection_order(self, selections: List[Selection]) -> List[Selection]:
        """Optimize execution order of selection operations, prioritizing high selectivity operations"""
        # Simplified implementation: sort by estimated selectivity
        return sorted(selections, key=lambda s: self._estimate_selectivity(s))
    
    def _estimate_selectivity(self, selection: Selection) -> float:
        """Estimate selectivity of selection operation"""
        # Simplified implementation, returns fixed value
        return 0.5
    
    def push_down_selections(self, operations: List) -> List:
        """Selection pushdown optimization"""
        # Execute selection operations as early as possible
        selections = [op for op in operations if isinstance(op, Selection)]
        others = [op for op in operations if not isinstance(op, Selection)]
        
        return selections + others


def demo_employee_salary_analysis():
    """Demonstrate employee salary analysis queries"""
    print("=== Employee Salary Analysis Demo ===")
    
    # Create test data (zonotope format)
    salary_data = pd.DataFrame([
        {"center": 6000, "g1": 500, "idx1": 1},   # [5500, 6500]
        {"center": 7000, "g1": 400, "idx1": 2},   # [6600, 7400] 
        {"center": 5500, "g1": 300, "idx1": 3},   # [5200, 5800]
        {"center": 8000, "g1": 600, "idx1": 4},   # [7400, 8600]
        {"center": 5000, "g1": 200, "idx1": 5},   # [4800, 5200]
    ])
    
    processor = ComplexQueryProcessor()
    
    # Execute query: SELECT SUM(salary) FROM Employee WHERE salary > AVG(salary)
    min_sum, max_sum = processor.process_sum_with_avg_threshold(salary_data)
    print(f"Sum of salaries above average: [{min_sum:.2f}, {max_sum:.2f}]")
    
    # Calculate average salary for comparison
    avg_agg = Aggregation('avg', 'salary')
    relation = Relation(salary_data)
    avg_salary = avg_agg.execute(relation)
    avg_lb, avg_ub = avg_salary.bound()
    print(f"Average salary range: [{avg_lb:.2f}, {avg_ub:.2f}]")


def demo_count_with_conditions():
    """Demonstrate COUNT queries with conditions"""
    print("\n=== Conditional COUNT Query Demo ===")
    
    # Create three column data
    x_data = pd.DataFrame([{"center": 2}, {"center": 4}, {"center": 1}])
    y_data = pd.DataFrame([
        {"center": 3.5, "g1": 2.5, "idx1": 1},  # [1, 6]
        {"center": 2.0, "g1": 1.0, "idx1": 2},  # [1, 3] 
        {"center": 3.0, "g1": 0.5, "idx1": 3}   # [2.5, 3.5]
    ])
    z_data = pd.DataFrame([{"center": 5}, {"center": 0}, {"center": 4}])
    
    processor = ComplexQueryProcessor()
    min_count, max_count = processor.process_count_with_conditions(x_data, y_data, z_data)
    print(f"Records satisfying x > y AND y > z: [{int(min_count)}, {int(max_count)}]")


def demo_advanced_predicates():
    """Demonstrate usage of advanced predicates"""
    print("\n=== Advanced Predicates Demo ===")
    
    # Create test data
    data = pd.DataFrame([
        {"center": 1000, "g1": 100, "idx1": 1},
        {"center": 2000, "g1": 200, "idx1": 2},
        {"center": 3000, "g1": 150, "idx1": 3}
    ])
    relation = Relation(data)
    
    # Range predicate: 1500 <= salary <= 2500
    range_pred = AdvancedPredicates.create_range_predicate(1500, 2500)
    range_selection = Selection(range_pred)
    min_count, max_count = range_selection.execute(relation)
    print(f"Employees with salary in range 1500-2500: [{int(min_count)}, {int(max_count)}]")


def demo_query_optimization():
    """Demonstrate query optimization"""
    print("\n=== Query Optimization Demo ===")
    
    # Create multiple selection operations
    pred1 = AdvancedPredicates.create_range_predicate(1000, 5000)
    pred2 = AdvancedPredicates.create_range_predicate(2000, 3000)
    
    selections = [
        Selection(pred1),
        Selection(pred2)
    ]
    
    optimizer = QueryOptimizer()
    optimized_selections = optimizer.optimize_selection_order(selections)
    
    print(f"Original number of selection operations: {len(selections)}")
    print(f"Optimized number of selection operations: {len(optimized_selections)}")


def main():
    """Main function running all demos"""
    try:
        demo_employee_salary_analysis()
        demo_count_with_conditions() 
        demo_advanced_predicates()
        demo_query_optimization()
        
        print("\n=== Demo Complete ===")
        print("This framework supports the following features:")
        print("1. Representation and processing of uncertain data (Zonotope)")
        print("2. Basic relational algebra operations (Selection, Projection, Aggregation, Cartesian Product)")
        print("3. Mixed integer optimization solving")
        print("4. Composite processing of complex queries")
        print("5. Query optimization techniques")
        print("6. Flexible predicate definition system")
        
    except Exception as e:
        print(f"Error occurred during demo: {e}")
        print("Please ensure the relational algebra framework module is properly imported")


if __name__ == "__main__":
    main()