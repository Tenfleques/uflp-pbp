#!/usr/bin/env python3
"""
Advanced Branching Rules for Uncapacitated Facility Location Problems

This module implements MQL (MakeQuadraticLinear) branching and antiKhumawala 
branching rules that work recursively with the 1st and 2nd Khumawala rules.

Implementation is based on research from binary quadratic programming and 
reformulation-linearization techniques applied to facility location problems.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import logging
from dataclasses import dataclass
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class BranchingNode:
    """Represents a node in the branching tree"""
    level: int
    fixed_facilities: Dict[int, int]  # facility_id -> {0: closed, 1: opened}
    lower_bound: float
    upper_bound: float
    is_integer: bool
    is_feasible: bool
    branching_variable: Optional[Tuple[int, int]] = None  # (i, k) for z_ik
    parent_node: Optional['BranchingNode'] = None


@dataclass
class BranchingStats:
    """Statistics for branching operations"""
    total_nodes: int = 0
    nodes_pruned: int = 0
    nodes_solved_by_khumawala: int = 0
    mql_branchings: int = 0
    khumawala_rule1_applications: int = 0
    khumawala_rule2_applications: int = 0
    max_depth: int = 0


class MQLBranching:
    """
    MakeQuadraticLinear (MQL) Branching Strategy
    
    This implements a linearization approach for quadratic terms by introducing
    auxiliary variables and constraints, then applying Khumawala rules recursively.
    
    The strategy identifies quadratic interactions in the pseudo-Boolean polynomial
    and creates linear representations through variable substitution.
    """
    
    def __init__(self, tolerance=1e-6):
        self.tolerance = tolerance
        self.stats = BranchingStats()
        self.active_nodes = []
        self.best_solution = None
        self.best_objective = float('inf')
        
    def apply_mql_branching(self, instance: Dict, khumawala_rules, 
                           max_iterations: int = 1000) -> Dict:
        """
        Apply MQL branching with recursive Khumawala rules
        
        Args:
            instance: UFLP instance dictionary
            khumawala_rules: KhumawalaRules instance
            max_iterations: Maximum branching iterations
            
        Returns:
            Dictionary with solution and statistics
        """
        logger.info("Starting MQL branching with recursive Khumawala rules")
        
        # Initialize root node
        root_node = BranchingNode(
            level=0,
            fixed_facilities={},
            lower_bound=0.0,
            upper_bound=float('inf'),
            is_integer=False,
            is_feasible=True
        )
        
        self.active_nodes = [root_node]
        self.stats = BranchingStats()
        
        iteration = 0
        while self.active_nodes and iteration < max_iterations:
            iteration += 1
            
            # Select node to branch on (best-first strategy)
            current_node = self._select_branching_node()
            if current_node is None:
                break
                
            logger.debug(f"Processing node at level {current_node.level}")
            
            # Apply current fixings to create subproblem
            subproblem = self._create_subproblem(instance, current_node)
            
            # Try to solve with Khumawala rules first
            khumawala_result = khumawala_rules.apply_rules(subproblem)
            
            if khumawala_result['solved_to_optimality']:
                # Problem solved by Khumawala rules alone
                self.stats.nodes_solved_by_khumawala += 1
                self._update_best_solution(khumawala_result, current_node)
                continue
            
            # Apply MQL linearization to remaining quadratic terms
            linearized_problem = self._apply_mql_linearization(
                subproblem, khumawala_result
            )
            
            # Create branching decisions
            branching_candidates = self._identify_branching_candidates(
                linearized_problem, khumawala_result
            )
            
            if not branching_candidates:
                # No more branching possible, node is solved
                self._update_best_solution(khumawala_result, current_node)
                continue
            
            # Branch on most promising variable
            branching_var = self._select_branching_variable(branching_candidates)
            self._create_child_nodes(current_node, branching_var, linearized_problem)
            
            self.stats.mql_branchings += 1
            self.stats.max_depth = max(self.stats.max_depth, current_node.level + 1)
            
            # Update statistics
            self.stats.khumawala_rule1_applications += khumawala_result.get('rule1_applications', 0)
            self.stats.khumawala_rule2_applications += khumawala_result.get('rule2_applications', 0)
        
        return self._generate_final_result()
    
    def _select_branching_node(self) -> Optional[BranchingNode]:
        """Select next node for branching using best-first strategy"""
        if not self.active_nodes:
            return None
            
        # Sort by lower bound (best-first)
        self.active_nodes.sort(key=lambda n: n.lower_bound)
        
        # Prune nodes with bound worse than current best
        pruned = []
        for node in self.active_nodes:
            if node.lower_bound >= self.best_objective - self.tolerance:
                pruned.append(node)
        
        for node in pruned:
            self.active_nodes.remove(node)
            self.stats.nodes_pruned += 1
            
        if not self.active_nodes:
            return None
            
        return self.active_nodes.pop(0)
    
    def _create_subproblem(self, instance: Dict, node: BranchingNode) -> Dict:
        """Create subproblem by applying node fixings"""
        subproblem = deepcopy(instance)
        
        # Apply facility fixings
        for facility_id, status in node.fixed_facilities.items():
            if status == 0:  # Facility closed
                # Remove facility from consideration
                if facility_id in subproblem.get('facilities', []):
                    subproblem['facilities'].remove(facility_id)
            elif status == 1:  # Facility opened
                # Force facility to be opened
                subproblem.setdefault('forced_open', []).append(facility_id)
        
        return subproblem
    
    def _apply_mql_linearization(self, subproblem: Dict, 
                                khumawala_result: Dict) -> Dict:
        """
        Apply MQL linearization to quadratic terms
        
        This converts quadratic interactions z_i * z_j into linear constraints
        using auxiliary variables and reformulation-linearization techniques.
        """
        linearized = deepcopy(subproblem)
        
        # Identify remaining quadratic terms after Khumawala reductions
        remaining_variables = khumawala_result.get('remaining_variables', set())
        
        if len(remaining_variables) <= 1:
            return linearized
            
        # Create auxiliary variables for quadratic interactions
        auxiliary_vars = {}
        linear_constraints = []
        
        variables = list(remaining_variables)
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                var_i, var_j = variables[i], variables[j]
                
                # Create auxiliary variable x_ij for z_i * z_j
                aux_var = f"x_{var_i}_{var_j}"
                auxiliary_vars[aux_var] = (var_i, var_j)
                
                # Add linearization constraints:
                # x_ij <= z_i
                # x_ij <= z_j  
                # x_ij >= z_i + z_j - 1
                linear_constraints.extend([
                    (aux_var, '<=', var_i),
                    (aux_var, '<=', var_j),
                    (aux_var, '>=', f"{var_i} + {var_j} - 1")
                ])
        
        linearized['auxiliary_variables'] = auxiliary_vars
        linearized['linearization_constraints'] = linear_constraints
        
        logger.debug(f"Created {len(auxiliary_vars)} auxiliary variables for MQL linearization")
        
        return linearized
    
    def _identify_branching_candidates(self, linearized_problem: Dict,
                                     khumawala_result: Dict) -> List[Tuple[int, int]]:
        """Identify variables that are candidates for branching"""
        candidates = []
        
        # Get fractional variables from the linearized problem
        remaining_vars = khumawala_result.get('remaining_variables', set())
        variable_values = khumawala_result.get('variable_values', {})
        
        for var in remaining_vars:
            if isinstance(var, tuple) and len(var) == 2:
                i, k = var
                value = variable_values.get(var, 0.5)
                
                # Consider variables that are not integer
                if abs(value - 0.0) > self.tolerance and abs(value - 1.0) > self.tolerance:
                    candidates.append((i, k))
        
        return candidates
    
    def _select_branching_variable(self, candidates: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Select the best variable for branching (most fractional)"""
        if not candidates:
            return None
            
        # For now, select the first candidate
        # Could be enhanced with better selection criteria
        return candidates[0]
    
    def _create_child_nodes(self, parent_node: BranchingNode, 
                           branching_var: Tuple[int, int],
                           problem: Dict):
        """Create child nodes by branching on the selected variable"""
        i, k = branching_var
        
        # Create left child: z_ik = 0
        left_fixings = deepcopy(parent_node.fixed_facilities)
        left_fixings[(i, k)] = 0
        
        left_child = BranchingNode(
            level=parent_node.level + 1,
            fixed_facilities=left_fixings,
            lower_bound=parent_node.lower_bound,
            upper_bound=parent_node.upper_bound,
            is_integer=False,
            is_feasible=True,
            branching_variable=branching_var,
            parent_node=parent_node
        )
        
        # Create right child: z_ik = 1
        right_fixings = deepcopy(parent_node.fixed_facilities)
        right_fixings[(i, k)] = 1
        
        right_child = BranchingNode(
            level=parent_node.level + 1,
            fixed_facilities=right_fixings,
            lower_bound=parent_node.lower_bound,
            upper_bound=parent_node.upper_bound,
            is_integer=False,
            is_feasible=True,
            branching_variable=branching_var,
            parent_node=parent_node
        )
        
        self.active_nodes.extend([left_child, right_child])
        self.stats.total_nodes += 2
    
    def _update_best_solution(self, result: Dict, node: BranchingNode):
        """Update best known solution"""
        if result.get('objective_value', float('inf')) < self.best_objective:
            self.best_objective = result['objective_value']
            self.best_solution = {
                'solution': result.get('solution', {}),
                'objective': self.best_objective,
                'node_level': node.level,
                'fixed_facilities': deepcopy(node.fixed_facilities)
            }
            logger.info(f"New best solution found: {self.best_objective}")
    
    def _generate_final_result(self) -> Dict:
        """Generate final result with statistics"""
        return {
            'solved_to_optimality': len(self.active_nodes) == 0,
            'best_solution': self.best_solution,
            'best_objective': self.best_objective,
            'statistics': {
                'total_nodes': self.stats.total_nodes,
                'nodes_pruned': self.stats.nodes_pruned,
                'nodes_solved_by_khumawala': self.stats.nodes_solved_by_khumawala,
                'mql_branchings': self.stats.mql_branchings,
                'khumawala_rule1_applications': self.stats.khumawala_rule1_applications,
                'khumawala_rule2_applications': self.stats.khumawala_rule2_applications,
                'max_depth': self.stats.max_depth
            }
        }


class AntiKhumawalaRules:
    """
    AntiKhumawala Branching Rules (B7)
    
    Implements various branching strategies that complement Khumawala rules:
    - LBA: Linear Branching A
    - SBA: Systematic Branching A  
    - FLBA: Fast Linear Branching A
    - FSBA: Fast Systematic Branching A
    - SLBA: Slow Linear Branching A
    - SSBA: Slow Systematic Branching A
    """
    
    def __init__(self):
        self.branching_counts = {
            'LBA': 0, 'SBA': 0, 'FLBA': 0, 
            'FSBA': 0, 'SLBA': 0, 'SSBA': 0
        }
    
    def apply_lba_rule(self, instance: Dict, khumawala_rules) -> Dict:
        """
        Linear Branching A (LBA) rule
        
        This implements a linear branching strategy that focuses on facility variables
        with highest impact on the objective function. The rule examines the linear
        terms in the pseudo-Boolean polynomial and branches on facilities that appear
        most frequently or have highest coefficients.
        
        Args:
            instance: UFLP instance dictionary  
            khumawala_rules: KhumawalaRules instance
            
        Returns:
            Dict with branching results and statistics
        """
        logger.info("Applying Linear Branching A (LBA) rule")
        
        # Apply Khumawala rules first
        khumawala_result = khumawala_rules.apply_rules(instance)
        
        if khumawala_result['solved_to_optimality']:
            logger.info("LBA: Problem solved by Khumawala rules alone")
            self.branching_counts['LBA'] += 1
            return {
                'rule': 'LBA',
                'applied': True,
                'solved_by_khumawala': True,
                'result': khumawala_result
            }
        
        # Get undecided facilities
        undecided_facilities = list(khumawala_rules.get_undecided_facilities())
        
        if not undecided_facilities:
            return {
                'rule': 'LBA', 
                'applied': False,
                'reason': 'No undecided facilities remaining'
            }
        
        # Calculate linear branching scores for facilities
        facility_scores = self._calculate_linear_scores(instance, undecided_facilities)
        
        # Sort facilities by linear impact (highest first)
        sorted_facilities = sorted(facility_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Branch on the facility with highest linear impact
        branching_facility = sorted_facilities[0][0]
        
        logger.info(f"LBA: Branching on facility {branching_facility} (score: {facility_scores[branching_facility]:.3f})")
        
        # Create subproblems for branching
        subproblems = self._create_linear_branch_subproblems(
            instance, branching_facility, khumawala_rules
        )
        
        self.branching_counts['LBA'] += 1
        
        return {
            'rule': 'LBA',
            'applied': True,
            'branching_facility': branching_facility,
            'facility_score': facility_scores[branching_facility],
            'subproblems': subproblems,
            'undecided_count': len(undecided_facilities)
        }
    
    def apply_sba_rule(self, instance: Dict, khumawala_rules) -> Dict:
        """
        Systematic Branching A (SBA) rule
        
        This implements a systematic branching strategy that considers both facility
        opening decisions and customer assignment decisions in a coordinated manner.
        The rule systematically explores the solution space by considering facility-customer
        pairs with highest impact on the objective function.
        
        Args:
            instance: UFLP instance dictionary
            khumawala_rules: KhumawalaRules instance
            
        Returns:
            Dict with branching results and statistics
        """
        logger.info("Applying Systematic Branching A (SBA) rule")
        
        # Apply Khumawala rules first
        khumawala_result = khumawala_rules.apply_rules(instance)
        
        if khumawala_result['solved_to_optimality']:
            logger.info("SBA: Problem solved by Khumawala rules alone")
            self.branching_counts['SBA'] += 1
            return {
                'rule': 'SBA',
                'applied': True,
                'solved_by_khumawala': True,
                'result': khumawala_result
            }
        
        # Get undecided facilities and analyze facility-customer pairs
        undecided_facilities = list(khumawala_rules.get_undecided_facilities())
        
        if not undecided_facilities:
            return {
                'rule': 'SBA',
                'applied': False,
                'reason': 'No undecided facilities remaining'
            }
        
        # Calculate systematic branching scores for facility-customer pairs
        pair_scores = self._calculate_systematic_scores(instance, undecided_facilities)
        
        if not pair_scores:
            return {
                'rule': 'SBA',
                'applied': False,
                'reason': 'No valid facility-customer pairs found'
            }
        
        # Sort pairs by systematic impact (highest first)
        sorted_pairs = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)
        best_pair = sorted_pairs[0][0]  # (facility, customer)
        best_score = sorted_pairs[0][1]
        
        logger.info(f"SBA: Branching on facility-customer pair {best_pair} (score: {best_score:.3f})")
        
        # Create subproblems for systematic branching
        subproblems = self._create_systematic_branch_subproblems(
            instance, best_pair, khumawala_rules
        )
        
        self.branching_counts['SBA'] += 1
        
        return {
            'rule': 'SBA',
            'applied': True,
            'branching_pair': best_pair,
            'pair_score': best_score,
            'subproblems': subproblems,
            'undecided_count': len(undecided_facilities),
            'total_pairs_analyzed': len(pair_scores)
        }
    
    def apply_flba_rule(self, instance: Dict, khumawala_rules) -> Dict:
        """
        Fast Linear Branching A rule
        
        This is a simplified version of LBA that uses heuristic facility selection
        instead of exhaustive evaluation. It quickly identifies facilities with
        highest cost ratios for branching.
        """
        logger.info("Applying Fast Linear Branching A (FLBA) rule")
        self.branching_counts['FLBA'] += 1
        
        # Apply Khumawala rules first
        initial_result = khumawala_rules.apply_rules(instance)
        if initial_result['solved_to_optimality']:
            logger.info("FLBA: Problem solved by initial Khumawala rules")
            return {
                'rule': 'FLBA',
                'solved_to_optimality': True,
                'solution': initial_result,
                'flba_branchings': 0,
                'khumawala_applications': 1
            }
        
        # Quick heuristic selection of branching facility
        facilities = instance.get('facilities', [])
        facility_costs = instance.get('facility_costs', [])
        assignment_costs = instance.get('assignment_costs', [])
        
        if not facilities or not facility_costs:
            return {'rule': 'FLBA', 'solved_to_optimality': False, 'error': 'No facilities to branch on'}
        
        # Fast scoring: highest facility cost / average assignment cost ratio
        best_facility = None
        best_score = -1
        
        for facility in facilities:
            if facility >= len(facility_costs):
                continue
                
            facility_cost = facility_costs[facility]
            
            if assignment_costs and len(assignment_costs) > 0:
                avg_assignment = np.mean([
                    assignment_costs[customer][facility] 
                    for customer in range(len(assignment_costs))
                    if facility < len(assignment_costs[customer])
                ])
                score = facility_cost / (avg_assignment + 1e-6)
            else:
                score = facility_cost
            
            if score > best_score:
                best_score = score
                best_facility = facility
        
        if best_facility is None:
            return {'rule': 'FLBA', 'solved_to_optimality': False, 'error': 'No suitable facility found'}
        
        # Create subproblems quickly
        subproblems = self._create_linear_branch_subproblems(instance, best_facility, khumawala_rules)
        
        best_result = None
        total_branchings = 1
        
        for branch_type, subproblem in subproblems.items():
            result = khumawala_rules.apply_rules(subproblem)
            if result['solved_to_optimality']:
                if best_result is None or result.get('objective', float('inf')) < best_result.get('objective', float('inf')):
                    best_result = result
        
        return {
            'rule': 'FLBA',
            'solved_to_optimality': best_result is not None,
            'solution': best_result,
            'flba_branchings': total_branchings,
            'khumawala_applications': len(subproblems) + 1,
            'branching_facility': best_facility
        }
    
    def apply_fsba_rule(self, instance: Dict, khumawala_rules) -> Dict:
        """
        Fast Systematic Branching A rule
        
        This is a simplified version of SBA that uses heuristic facility-customer
        pair selection instead of exhaustive evaluation. It quickly identifies
        pairs with optimal cost characteristics.
        """
        logger.info("Applying Fast Systematic Branching A (FSBA) rule")
        self.branching_counts['FSBA'] += 1
        
        # Apply Khumawala rules first
        initial_result = khumawala_rules.apply_rules(instance)
        if initial_result['solved_to_optimality']:
            logger.info("FSBA: Problem solved by initial Khumawala rules")
            return {
                'rule': 'FSBA',
                'solved_to_optimality': True,
                'solution': initial_result,
                'fsba_branchings': 0,
                'khumawala_applications': 1
            }
        
        # Quick heuristic selection of facility-customer pair
        facilities = instance.get('facilities', [])
        customers = instance.get('customers', [])
        facility_costs = instance.get('facility_costs', [])
        assignment_costs = instance.get('assignment_costs', [])
        
        if not facilities or not customers or not assignment_costs:
            return {'rule': 'FSBA', 'solved_to_optimality': False, 'error': 'Insufficient data for systematic branching'}
        
        # Fast scoring: minimize assignment cost / facility cost ratio
        best_pair = None
        best_score = float('inf')
        
        for facility in facilities:
            if facility >= len(facility_costs):
                continue
                
            facility_cost = facility_costs[facility]
            
            for customer in customers:
                if (customer >= len(assignment_costs) or 
                    facility >= len(assignment_costs[customer])):
                    continue
                
                assignment_cost = assignment_costs[customer][facility]
                score = assignment_cost / (facility_cost + 1e-6)
                
                if score < best_score:
                    best_score = score
                    best_pair = (facility, customer)
        
        if best_pair is None:
            return {'rule': 'FSBA', 'solved_to_optimality': False, 'error': 'No suitable facility-customer pair found'}
        
        # Create subproblems quickly
        subproblems = self._create_systematic_branch_subproblems(instance, best_pair, khumawala_rules)
        
        best_result = None
        total_branchings = 1
        
        for branch_type, subproblem in subproblems.items():
            result = khumawala_rules.apply_rules(subproblem)
            if result['solved_to_optimality']:
                if best_result is None or result.get('objective', float('inf')) < best_result.get('objective', float('inf')):
                    best_result = result
        
        return {
            'rule': 'FSBA',
            'solved_to_optimality': best_result is not None,
            'solution': best_result,
            'fsba_branchings': total_branchings,
            'khumawala_applications': len(subproblems) + 1,
            'branching_pair': best_pair
        }
    
    def apply_slba_rule(self, instance: Dict, khumawala_rules) -> Dict:
        """
        Slow Linear Branching A rule
        
        This is an exhaustive version of LBA that evaluates all possible facility
        branchings and uses iterative deepening to find optimal solutions.
        More thorough but computationally expensive.
        """
        logger.info("Applying Slow Linear Branching A (SLBA) rule")
        self.branching_counts['SLBA'] += 1
        
        # Apply Khumawala rules first
        initial_result = khumawala_rules.apply_rules(instance)
        if initial_result['solved_to_optimality']:
            logger.info("SLBA: Problem solved by initial Khumawala rules")
            return {
                'rule': 'SLBA',
                'solved_to_optimality': True,
                'solution': initial_result,
                'slba_branchings': 0,
                'khumawala_applications': 1
            }
        
        facilities = instance.get('facilities', [])
        if not facilities:
            return {'rule': 'SLBA', 'solved_to_optimality': False, 'error': 'No facilities to branch on'}
        
        # Exhaustive evaluation of all facilities
        best_result = None
        total_branchings = 0
        total_khumawala_applications = 1  # Initial application
        
        # Sort facilities by detailed scoring for systematic exploration
        facility_scores = self._calculate_linear_scores(instance, facilities)
        sorted_facilities = sorted(facilities, key=lambda f: facility_scores.get(f, 0), reverse=True)
        
        for facility in sorted_facilities:
            logger.debug(f"SLBA: Exploring facility {facility}")
            
            # Create subproblems
            subproblems = self._create_linear_branch_subproblems(instance, facility, khumawala_rules)
            total_branchings += 1
            
            # Evaluate all branches exhaustively
            for branch_type, subproblem in subproblems.items():
                result = khumawala_rules.apply_rules(subproblem)
                total_khumawala_applications += 1
                
                if result['solved_to_optimality']:
                    if best_result is None or result.get('objective', float('inf')) < best_result.get('objective', float('inf')):
                        best_result = result
                        best_result['branching_facility'] = facility
                        best_result['branch_type'] = branch_type
                
                # If we found a solution, continue checking for better ones
                # (this is the "slow" part - exhaustive search)
        
        return {
            'rule': 'SLBA',
            'solved_to_optimality': best_result is not None,
            'solution': best_result,
            'slba_branchings': total_branchings,
            'khumawala_applications': total_khumawala_applications,
            'facilities_explored': len(sorted_facilities)
        }
    
    def apply_ssba_rule(self, instance: Dict, khumawala_rules) -> Dict:
        """
        Slow Systematic Branching A rule
        
        This is an exhaustive version of SBA that evaluates all possible facility-customer
        pair branchings with iterative deepening. Most thorough but most expensive.
        """
        logger.info("Applying Slow Systematic Branching A (SSBA) rule")
        self.branching_counts['SSBA'] += 1
        
        # Apply Khumawala rules first
        initial_result = khumawala_rules.apply_rules(instance)
        if initial_result['solved_to_optimality']:
            logger.info("SSBA: Problem solved by initial Khumawala rules")
            return {
                'rule': 'SSBA',
                'solved_to_optimality': True,
                'solution': initial_result,
                'ssba_branchings': 0,
                'khumawala_applications': 1
            }
        
        facilities = instance.get('facilities', [])
        customers = instance.get('customers', [])
        
        if not facilities or not customers:
            return {'rule': 'SSBA', 'solved_to_optimality': False, 'error': 'Insufficient data for systematic branching'}
        
        # Exhaustive evaluation of all facility-customer pairs
        best_result = None
        total_branchings = 0
        total_khumawala_applications = 1  # Initial application
        
        # Sort pairs by detailed scoring for systematic exploration
        pair_scores = self._calculate_systematic_scores(instance, facilities)
        sorted_pairs = sorted(pair_scores.keys(), key=lambda p: pair_scores[p], reverse=True)
        
        for pair in sorted_pairs:
            facility, customer = pair
            logger.debug(f"SSBA: Exploring facility-customer pair ({facility}, {customer})")
            
            # Create subproblems
            subproblems = self._create_systematic_branch_subproblems(instance, pair, khumawala_rules)
            total_branchings += 1
            
            # Evaluate all branches exhaustively
            for branch_type, subproblem in subproblems.items():
                result = khumawala_rules.apply_rules(subproblem)
                total_khumawala_applications += 1
                
                if result['solved_to_optimality']:
                    if best_result is None or result.get('objective', float('inf')) < best_result.get('objective', float('inf')):
                        best_result = result
                        best_result['branching_pair'] = pair
                        best_result['branch_type'] = branch_type
                
                # Continue exhaustive search even after finding solutions
                # (this is the "slow" part - complete exploration)
        
        return {
            'rule': 'SSBA',
            'solved_to_optimality': best_result is not None,
            'solution': best_result,
            'ssba_branchings': total_branchings,
            'khumawala_applications': total_khumawala_applications,
            'pairs_explored': len(sorted_pairs)
        }
    
    def _calculate_linear_scores(self, instance: Dict, undecided_facilities: List[int]) -> Dict[int, float]:
        """
        Calculate linear branching scores for undecided facilities.
        
        Score is based on facility cost and average assignment costs.
        Higher scores indicate more impactful facilities for branching.
        """
        scores = {}
        facility_costs = instance.get('facility_costs', [])
        assignment_costs = instance.get('assignment_costs', [])
        
        for facility in undecided_facilities:
            if facility >= len(facility_costs):
                scores[facility] = 0.0
                continue
                
            # Base score from facility opening cost
            facility_cost = facility_costs[facility]
            
            # Add average assignment cost impact
            if assignment_costs and len(assignment_costs) > 0:
                avg_assignment_cost = np.mean([
                    assignment_costs[customer][facility] 
                    for customer in range(len(assignment_costs))
                    if facility < len(assignment_costs[customer])
                ])
            else:
                avg_assignment_cost = 0.0
            
            # Combine costs with weighting (facility cost has higher weight)
            scores[facility] = 0.7 * facility_cost + 0.3 * avg_assignment_cost
        
        return scores
    
    def _create_linear_branch_subproblems(self, instance: Dict, facility: int, 
                                        khumawala_rules) -> Dict[str, Dict]:
        """
        Create subproblems for linear branching on a facility.
        
        Returns two subproblems: one with facility closed, one with facility open.
        """
        base_instance = deepcopy(instance)
        
        # Subproblem 1: Force facility closed
        closed_instance = deepcopy(base_instance)
        closed_facilities = closed_instance.setdefault('forced_closed', [])
        closed_facilities.append(facility)
        
        # Subproblem 2: Force facility open  
        open_instance = deepcopy(base_instance)
        open_facilities = open_instance.setdefault('forced_open', [])
        open_facilities.append(facility)
        
        return {
            'closed': closed_instance,
            'open': open_instance
        }
    
    def _calculate_systematic_scores(self, instance: Dict, undecided_facilities: List[int]) -> Dict[Tuple[int, int], float]:
        """
        Calculate systematic branching scores for facility-customer pairs.
        
        Score considers both facility cost and specific assignment cost for each pair.
        Higher scores indicate more impactful facility-customer pairs for branching.
        """
        scores = {}
        facility_costs = instance.get('facility_costs', [])
        assignment_costs = instance.get('assignment_costs', [])
        customers = instance.get('customers', [])
        
        if not assignment_costs or not customers:
            return scores
        
        for facility in undecided_facilities:
            if facility >= len(facility_costs):
                continue
                
            facility_cost = facility_costs[facility]
            
            for customer in customers:
                if (customer >= len(assignment_costs) or 
                    facility >= len(assignment_costs[customer])):
                    continue
                
                assignment_cost = assignment_costs[customer][facility]
                
                # Systematic score combines facility opening cost and assignment cost
                # with emphasis on the specific facility-customer relationship
                systematic_score = (0.6 * facility_cost + 0.4 * assignment_cost) / (facility + customer + 1)
                
                scores[(facility, customer)] = systematic_score
        
        return scores
    
    def _create_systematic_branch_subproblems(self, instance: Dict, pair: Tuple[int, int],
                                            khumawala_rules) -> Dict[str, Dict]:
        """
        Create subproblems for systematic branching on a facility-customer pair.
        
        Returns subproblems with different assignment constraints.
        """
        facility, customer = pair
        base_instance = deepcopy(instance)
        
        # Subproblem 1: Force this facility-customer assignment
        assign_instance = deepcopy(base_instance)
        forced_assignments = assign_instance.setdefault('forced_assignments', [])
        forced_assignments.append((customer, facility))
        # Must also force facility open
        forced_open = assign_instance.setdefault('forced_open', [])
        if facility not in forced_open:
            forced_open.append(facility)
        
        # Subproblem 2: Prohibit this facility-customer assignment
        prohibit_instance = deepcopy(base_instance)
        prohibited_assignments = prohibit_instance.setdefault('prohibited_assignments', [])
        prohibited_assignments.append((customer, facility))
        
        return {
            'assign': assign_instance,
            'prohibit': prohibit_instance
        }
    
    def get_statistics(self) -> Dict:
        """Get branching rule statistics"""
        return {
            'branching_rule_counts': self.branching_counts.copy(),
            'total_branchings': sum(self.branching_counts.values())
        }


class AdvancedBranchingSolver:
    """
    Main solver combining MQL branching and antiKhumawala rules
    
    This orchestrates the application of advanced branching strategies
    with recursive Khumawala rule application.
    """
    
    def __init__(self, mql_enabled=True, anti_khumawala_enabled=False):
        self.mql_branching = MQLBranching() if mql_enabled else None
        self.anti_khumawala = AntiKhumawalaRules() if anti_khumawala_enabled else None
        
    def solve(self, instance: Dict, khumawala_rules, strategy='mql_first') -> Dict:
        """
        Solve UFLP instance using advanced branching strategies
        
        Args:
            instance: UFLP instance dictionary
            khumawala_rules: KhumawalaRules instance  
            strategy: Branching strategy ('mql_first', 'anti_khumawala_first', 'combined')
            
        Returns:
            Complete solution with statistics
        """
        logger.info(f"Starting advanced branching solver with strategy: {strategy}")
        
        if strategy == 'mql_first' and self.mql_branching:
            result = self.mql_branching.apply_mql_branching(instance, khumawala_rules)
            
            # Add antiKhumawala statistics if available
            if self.anti_khumawala:
                anti_stats = self.anti_khumawala.get_statistics()
                result['statistics'].update(anti_stats)
                
            return result
        
        elif strategy == 'anti_khumawala_first' and self.anti_khumawala:
            # Placeholder for antiKhumawala-first strategy
            return {
                'solved_to_optimality': False,
                'message': 'AntiKhumawala branching not fully implemented yet',
                'statistics': self.anti_khumawala.get_statistics()
            }
        
        else:
            return {
                'solved_to_optimality': False,
                'message': f'Strategy {strategy} not available or not enabled',
                'statistics': {}
            } 