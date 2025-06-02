"""
Khumawala Rules Implementation for Uncapacitated Facility Location Problems.

This module implements the classical Khumawala dominance rules and branching strategies
for solving uncapacitated facility location problems (UFLP) using pseudo-Boolean
polynomial representations.

Based on Khumawala (1972) "An efficient branch and bound algorithm for the warehouse location problem"
and subsequent research on dominance rules for facility location problems.
"""

import numpy as np
import pandas as pd
from bitarray import bitarray
from bitarray.util import int2ba, ba2int
from typing import Dict, List, Tuple, Set, Optional
import logging

logger = logging.getLogger(__name__)

class KhumawalaRules:
    """
    Implementation of Khumawala rules for uncapacitated facility location problems.
    
    The Khumawala rules are dominance rules that can eliminate variables (fix them to 0 or 1)
    without losing optimality in branch-and-bound algorithms for facility location problems.
    """
    
    def __init__(self, c: np.ndarray, f: np.ndarray):
        """
        Initialize Khumawala rules with problem data.
        
        Args:
            c (np.ndarray): Transport cost matrix (customers × facilities)
            f (np.ndarray): Fixed facility opening costs
        """
        self.c = c.copy()
        self.f = f.copy()
        self.num_customers = c.shape[0]
        self.num_facilities = c.shape[1]
        
        # Track which facilities are fixed open (1) or closed (0)
        self.facility_status = {}  # facility_id -> 0 (closed), 1 (open), None (undecided)
        
        # Statistics for reporting
        self.rule1_applications = 0
        self.rule2_applications = 0
        self.variables_fixed_rule1 = 0
        self.variables_fixed_rule2 = 0
        
    def calculate_customer_assignment_costs(self, customer: int, 
                                          fixed_open: Set[int] = None) -> np.ndarray:
        """
        Calculate assignment costs for a customer to all facilities.
        
        Args:
            customer (int): Customer index
            fixed_open (Set[int], optional): Set of facilities that are fixed open
            
        Returns:
            np.ndarray: Assignment costs to each facility
        """
        costs = self.c[customer, :].copy()
        
        if fixed_open:
            # If some facilities are fixed open, customer can be assigned to cheapest open facility
            min_open_cost = min(costs[j] for j in fixed_open)
            for j in range(self.num_facilities):
                if j not in fixed_open:
                    costs[j] = max(costs[j], min_open_cost)
                    
        return costs
    
    def apply_first_khumawala_rule(self) -> int:
        """
        Apply the first Khumawala rule (cost-based dominance).
        
        The first rule states that if for facility j, there exists another facility k such that:
        f_j ≥ f_k + max_i(c_ik - c_ij), then facility j can be eliminated (fixed to 0).
        
        Returns:
            int: Number of facilities fixed by this rule
        """
        facilities_to_close = set()
        
        for j in range(self.num_facilities):
            if self.facility_status.get(j) is not None:
                continue  # Already fixed
                
            for k in range(self.num_facilities):
                if k == j or self.facility_status.get(k) == 0:
                    continue  # Skip same facility or already closed facilities
                
                # Calculate the maximum difference in assignment costs
                max_diff = np.max(self.c[:, k] - self.c[:, j])
                
                # Check if facility j is dominated by facility k
                if self.f[j] >= self.f[k] + max_diff:
                    facilities_to_close.add(j)
                    logger.debug(f"First Khumawala rule: Facility {j} dominated by {k}")
                    break
        
        # Fix dominated facilities to closed
        for j in facilities_to_close:
            self.facility_status[j] = 0
            self.variables_fixed_rule1 += 1
            
        if facilities_to_close:
            self.rule1_applications += 1
            logger.info(f"First Khumawala rule closed {len(facilities_to_close)} facilities: {facilities_to_close}")
            
        return len(facilities_to_close)
    
    def apply_second_khumawala_rule(self) -> int:
        """
        Apply the second Khumawala rule (customer-based dominance).
        
        The second rule examines customer assignments and can force facility openings
        when all cheaper alternatives for a customer are dominated.
        
        Returns:
            int: Number of facilities fixed by this rule
        """
        facilities_to_open = set()
        
        for i in range(self.num_customers):
            # Find the cheapest facility for customer i that is not fixed closed
            available_facilities = [j for j in range(self.num_facilities) 
                                  if self.facility_status.get(j) != 0]
            
            if not available_facilities:
                continue
                
            # Sort facilities by assignment cost for this customer
            sorted_facilities = sorted(available_facilities, key=lambda j: self.c[i, j])
            cheapest_facility = sorted_facilities[0]
            
            # Check if this customer would force opening of the cheapest facility
            must_open = True
            for j in sorted_facilities[1:]:  # Check other facilities
                # If customer i can be served more cheaply by j + opening cost difference
                cost_diff = self.c[i, j] - self.c[i, cheapest_facility]
                opening_cost_diff = self.f[cheapest_facility] - self.f[j]
                
                if cost_diff <= opening_cost_diff:
                    must_open = False
                    break
            
            if must_open and self.facility_status.get(cheapest_facility) is None:
                facilities_to_open.add(cheapest_facility)
                logger.debug(f"Second Khumawala rule: Customer {i} forces opening facility {cheapest_facility}")
        
        # Fix forced facilities to open
        for j in facilities_to_open:
            self.facility_status[j] = 1
            self.variables_fixed_rule2 += 1
            
        if facilities_to_open:
            self.rule2_applications += 1
            logger.info(f"Second Khumawala rule opened {len(facilities_to_open)} facilities: {facilities_to_open}")
            
        return len(facilities_to_open)
    
    def apply_iterative_khumawala_rules(self, max_iterations: int = 10) -> Dict[str, int]:
        """
        Apply Khumawala rules iteratively until no more variables can be fixed.
        
        Args:
            max_iterations (int): Maximum number of iterations
            
        Returns:
            Dict[str, int]: Statistics about rule applications
        """
        total_fixed = 0
        iteration = 0
        
        logger.info("Starting iterative application of Khumawala rules...")
        
        while iteration < max_iterations:
            iteration += 1
            fixed_this_iteration = 0
            
            # Apply first rule
            fixed_rule1 = self.apply_first_khumawala_rule()
            fixed_this_iteration += fixed_rule1
            
            # Apply second rule
            fixed_rule2 = self.apply_second_khumawala_rule()
            fixed_this_iteration += fixed_rule2
            
            total_fixed += fixed_this_iteration
            
            logger.info(f"Iteration {iteration}: Fixed {fixed_this_iteration} variables "
                       f"(Rule 1: {fixed_rule1}, Rule 2: {fixed_rule2})")
            
            if fixed_this_iteration == 0:
                logger.info(f"Convergence achieved after {iteration} iterations")
                break
        
        return self.get_statistics()
    
    def apply_rules(self, instance: Dict, max_iterations: int = 10) -> Dict:
        """
        Apply Khumawala rules to an instance and return comprehensive results.
        
        This is the main interface method expected by the MQL branching system.
        
        Args:
            instance: UFLP instance dictionary (not used as we use the matrices from constructor)
            max_iterations: Maximum iterations for iterative rule application
            
        Returns:
            Dict with solution status, statistics, and remaining variables
        """
        # Reset state for fresh application
        self.facility_status = {}
        self.rule1_applications = 0
        self.rule2_applications = 0
        self.variables_fixed_rule1 = 0
        self.variables_fixed_rule2 = 0
        
        # Apply iterative Khumawala rules
        stats = self.apply_iterative_khumawala_rules(max_iterations)
        
        # Determine remaining variables for branching
        undecided_facilities = self.get_undecided_facilities()
        remaining_variables = set()
        
        # For MQL branching, create tuples representing facility-customer assignments
        for j in undecided_facilities:
            for i in range(self.num_customers):
                remaining_variables.add((j, i))  # (facility, customer) pairs
        
        # Calculate variable values (fractional for undecided, 0/1 for decided)
        variable_values = {}
        for j in range(self.num_facilities):
            for i in range(self.num_customers):
                if self.facility_status.get(j) == 1:  # Facility open
                    variable_values[(j, i)] = 1.0
                elif self.facility_status.get(j) == 0:  # Facility closed
                    variable_values[(j, i)] = 0.0
                else:  # Undecided - use fractional value
                    variable_values[(j, i)] = 0.5
        
        # Calculate objective value if solved
        objective_value = float('inf')
        solution = {}
        
        if stats['solved_to_optimality']:
            objective_value = self.calculate_objective_lower_bound()
            solution = {
                'open_facilities': list(self.get_open_facilities()),
                'closed_facilities': list(self.get_closed_facilities()),
                'assignments': self._get_optimal_assignments()
            }
        
        return {
            'solved_to_optimality': stats['solved_to_optimality'],
            'objective_value': objective_value,
            'solution': solution,
            'remaining_variables': remaining_variables,
            'variable_values': variable_values,
            'rule1_applications': stats['rule1_applications'],
            'rule2_applications': stats['rule2_applications'],
            'statistics': stats
        }
    
    def _get_optimal_assignments(self) -> Dict[int, int]:
        """
        Get optimal customer-facility assignments for solved instances.
        
        Returns:
            Dict[int, int]: Customer to facility assignment mapping
        """
        assignments = {}
        open_facilities = self.get_open_facilities()
        
        if not open_facilities:
            return assignments
            
        for i in range(self.num_customers):
            # Assign each customer to cheapest open facility
            best_facility = min(open_facilities, key=lambda j: self.c[i, j])
            assignments[i] = best_facility
            
        return assignments
    
    def get_open_facilities(self) -> Set[int]:
        """Get facilities that are fixed to open."""
        return {j for j, status in self.facility_status.items() if status == 1}
    
    def get_closed_facilities(self) -> Set[int]:
        """Get facilities that are fixed to closed."""
        return {j for j, status in self.facility_status.items() if status == 0}
    
    def get_undecided_facilities(self) -> Set[int]:
        """Get facilities that are not yet fixed."""
        decided = set(self.facility_status.keys())
        all_facilities = set(range(self.num_facilities))
        return all_facilities - decided
    
    def is_solved_to_optimality(self) -> bool:
        """
        Check if the problem is solved to optimality by Khumawala rules alone.
        
        Returns:
            bool: True if all facilities are fixed, False otherwise
        """
        return len(self.facility_status) == self.num_facilities
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about Khumawala rule applications.
        
        Returns:
            Dict[str, int]: Statistics dictionary
        """
        open_facilities = self.get_open_facilities()
        closed_facilities = self.get_closed_facilities()
        undecided_facilities = self.get_undecided_facilities()
        
        return {
            'rule1_applications': self.rule1_applications,
            'rule2_applications': self.rule2_applications,
            'variables_fixed_rule1': self.variables_fixed_rule1,
            'variables_fixed_rule2': self.variables_fixed_rule2,
            'total_variables_fixed': self.variables_fixed_rule1 + self.variables_fixed_rule2,
            'facilities_opened': len(open_facilities),
            'facilities_closed': len(closed_facilities),
            'facilities_undecided': len(undecided_facilities),
            'solved_to_optimality': self.is_solved_to_optimality()
        }
    
    def get_reduced_problem(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Get the reduced problem after applying Khumawala rules.
        
        Returns:
            Tuple containing:
            - Reduced cost matrix (only undecided facilities)
            - Reduced fixed costs (only undecided facilities)
            - Mapping from reduced indices to original indices
        """
        undecided = list(self.get_undecided_facilities())
        undecided.sort()
        
        if not undecided:
            return np.array([]), np.array([]), {}
        
        # Create reduced problem
        reduced_c = self.c[:, undecided]
        reduced_f = self.f[undecided]
        
        # Create mapping
        index_mapping = {i: undecided[i] for i in range(len(undecided))}
        
        return reduced_c, reduced_f, index_mapping
    
    def calculate_objective_lower_bound(self) -> float:
        """
        Calculate a lower bound on the objective function value using fixed facilities.
        
        Returns:
            float: Lower bound on objective value
        """
        open_facilities = self.get_open_facilities()
        
        if not open_facilities:
            return 0.0
        
        # Fixed opening costs
        opening_cost = sum(self.f[j] for j in open_facilities)
        
        # Assignment costs (each customer assigned to cheapest open facility)
        assignment_cost = 0.0
        for i in range(self.num_customers):
            min_cost = min(self.c[i, j] for j in open_facilities)
            assignment_cost += min_cost
        
        return opening_cost + assignment_cost
    
    def print_status_report(self) -> None:
        """Print a detailed status report of Khumawala rule applications."""
        stats = self.get_statistics()
        
        print("=" * 60)
        print("KHUMAWALA RULES STATUS REPORT")
        print("=" * 60)
        
        print(f"\nPROBLEM SIZE:")
        print(f"  Customers: {self.num_customers}")
        print(f"  Facilities: {self.num_facilities}")
        print(f"  Total variables: {self.num_facilities}")
        
        print(f"\nRULE APPLICATIONS:")
        print(f"  1st Khumawala rule applications: {stats['rule1_applications']}")
        print(f"  2nd Khumawala rule applications: {stats['rule2_applications']}")
        
        print(f"\nVARIABLES FIXED:")
        print(f"  By 1st rule: {stats['variables_fixed_rule1']}")
        print(f"  By 2nd rule: {stats['variables_fixed_rule2']}")
        print(f"  Total fixed: {stats['total_variables_fixed']}")
        
        print(f"\nFACILITY STATUS:")
        print(f"  Opened: {stats['facilities_opened']}")
        print(f"  Closed: {stats['facilities_closed']}")
        print(f"  Undecided: {stats['facilities_undecided']}")
        
        print(f"\nSOLUTION STATUS:")
        if stats['solved_to_optimality']:
            print("  ✓ SOLVED TO OPTIMALITY by Khumawala rules alone!")
        else:
            print(f"  • {stats['facilities_undecided']} variables remaining for branch-and-bound")
        
        if self.get_open_facilities():
            print(f"\nOPEN FACILITIES: {sorted(self.get_open_facilities())}")
        if self.get_closed_facilities():
            print(f"CLOSED FACILITIES: {sorted(self.get_closed_facilities())}")
        if self.get_undecided_facilities():
            print(f"UNDECIDED FACILITIES: {sorted(self.get_undecided_facilities())}")
        
        print("=" * 60) 