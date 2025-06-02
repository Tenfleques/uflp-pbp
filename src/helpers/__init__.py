import numpy as np
from typing import Optional, List, Tuple, Dict
# from time_logger import time_logger, print_timing_summary
import pandas as pd


def extract_p_from_filename(filename: str) -> Optional[int]:
    """
    Extract the p value from the filename.
    """
    return int(filename.split('-')[-1].split('.')[0])


def calculate_solution_cost(c: np.ndarray, solution: List[int], f: np.ndarray = None) -> float:
    """
    Calculate the total cost of a p-median solution.
    
    Args:
        c: Cost matrix
        solution: List of selected facility indices
        f: Optional vector of fixed costs for each facility
        
    Returns:
        float: Total cost of the solution (including fixed costs if provided)
    """
    solution = np.array(solution)
    
    # Calculate variable costs (assignment costs)
    costs = c[:, solution]
    min_costs = np.min(costs, axis=1)
    variable_cost = np.sum(min_costs)
    
    # Add fixed costs if provided
    if f is not None:
        fixed_cost = np.sum(f[solution])
        return variable_cost + fixed_cost
    
    return variable_cost


# @time_logger
def get_dep_list(num_vars, pbp: pd.DataFrame, with_cost=False):
    available_variables = 2**np.arange(num_vars)
    if with_cost:
        dep_list = np.zeros((num_vars, pbp.shape[0] + 1), dtype=int)
    else:
        dep_list = np.zeros((num_vars, pbp.shape[0]), dtype=int)
    
    for i, y in enumerate(available_variables):
        dep_indices = np.where(y & pbp['y'])[0]
        dep_list[i][:dep_indices.shape[0]] = dep_indices
        if with_cost:
            dep_list[i][-1] = pbp.iloc[dep_indices]["coeffs"].sum()

    return dep_list

def validate_uwlp_solution(c: np.ndarray, solution: List[int], f: np.ndarray, 
                         optimal_cost: Optional[float] = None) -> Tuple[bool, str, float]:
    """
    Validate a solution for the Uncapacitated Warehouse Location Problem.
    
    Args:
        c: Cost matrix
        solution: List of selected facility indices
        f: Vector of fixed costs for each facility
        optimal_cost: Optional known optimal cost for comparison
        
    Returns:
        Tuple of (is_valid, message, calculated_cost)
    """
    if f is None:
        return False, "Fixed costs must be provided for UWLP", 0.0
        
    # Convert solution to numpy array
    solution = np.array(solution)
    
    # Check feasibility
    if len(set(solution)) != len(solution):
        return False, "Duplicate facilities in solution", 0.0
    
    if solution.min() < 0 or solution.max() >= len(c):
        return False, f"Invalid facility indices: must be between 0 and {len(c)-1}", 0.0
    
    # Calculate total cost
    total_cost = calculate_solution_cost(c, solution, f)
    
    # Compare with optimal if provided
    if optimal_cost is not None:
        gap = abs(total_cost - optimal_cost)
        if gap < 1e-10:  # Effectively zero gap
            return True, "Solution is optimal", total_cost
        else:
            return True, f"Solution is feasible but not optimal (gap: {gap:.2f})", total_cost
    
    return True, "Solution is feasible", total_cost

def verify_solution_quality(c: np.ndarray, solution: List[int], f: np.ndarray) -> Dict[str, float]:
    """
    Analyze the quality of a UWLP solution.
    
    Args:
        c: Cost matrix
        solution: List of selected facility indices
        f: Vector of fixed costs for each facility
        
    Returns:
        Dictionary containing quality metrics
    """
    if f is None:
        raise ValueError("Fixed costs must be provided for UWLP")
        
    solution = np.array(solution)
    
    # Calculate various metrics
    assignment_costs = c[:, solution]
    min_costs = np.min(assignment_costs, axis=1)
    fixed_costs = f[solution]
    
    metrics = {
        'total_cost': calculate_solution_cost(c, solution, f),
        'num_facilities': len(solution),
        'avg_assignment_cost': np.mean(min_costs),
        'max_assignment_cost': np.max(min_costs),
        'min_assignment_cost': np.min(min_costs),
        'std_assignment_cost': np.std(min_costs),
        'total_fixed_cost': np.sum(fixed_costs),
        'avg_fixed_cost': np.mean(fixed_costs),
        'max_fixed_cost': np.max(fixed_costs),
        'min_fixed_cost': np.min(fixed_costs),
        'fixed_cost_ratio': np.sum(fixed_costs) / calculate_solution_cost(c, solution, f)
    }
    
    return metrics