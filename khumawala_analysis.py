#!/usr/bin/env python3
"""
Comprehensive Khumawala Analysis Script for Uncapacitated Facility Location Problems.

This script implements all requirements B1-B7:
B1. Number of non-zero linear terms
B2. Number of non-zero quadratic terms  
B3. Number of non-zero cubic terms
B4. Total number of all non-linear terms
B5. Application of 1st and 2nd Khumawala rules and optimality analysis
B6. MQL (MakeQuadraticLinear) branching with recursive Khumawala rules
B7. AntiKhumawala branching rules (LBA, SBA, FLBA, FSBA, SLBA, SSBA)

Usage:
    python khumawala_analysis.py [--instances <file>] [--output <file>] [--verbose]
"""

import sys
import os
import numpy as np
import pandas as pd
import argparse
import json
from typing import List, Tuple, Dict, Any
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pbp.generate import cap_to_pbp_df
from khumawala.analysis import TermAnalyzer, analyze_instance, batch_analyze_instances
from khumawala.rules import KhumawalaRules
from khumawala.branching import MQLBranching, AntiKhumawalaRules, AdvancedBranchingSolver

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_instances() -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Create a set of test instances for analysis.
    
    Returns:
        List of (c, f, name) tuples representing different UFLP instances
    """
    instances = []
    
    # Instance 1: Small test case
    f1 = np.array([100, 120, 80])
    c1 = np.array([
        [10, 12, 8],
        [11, 9, 15]
    ])
    instances.append((c1, f1, "Small_Instance"))
    
    # Instance 2: Medium test case
    f2 = np.array([200, 250, 180, 300, 220])
    c2 = np.array([
        [20, 25, 18, 30, 22],
        [22, 20, 25, 28, 24],
        [18, 22, 16, 26, 20],
        [25, 28, 22, 20, 26]
    ])
    instances.append((c2, f2, "Medium_Instance"))
    
    # Instance 3: Original example from generate.py
    f3 = np.array([1000, 1200, 800, 1500, 900, 1100, 1300])
    c3 = np.array([
        [50, 60, 40, 70, 45, 55, 65],
        [45, 55, 35, 65, 40, 50, 60],
        [60, 70, 50, 80, 55, 65, 75],
        [40, 50, 30, 60, 35, 45, 55],
        [55, 65, 45, 75, 50, 60, 70]
    ])
    instances.append((c3, f3, "Large_Instance"))
    
    # Instance 4: Designed to trigger Khumawala rules
    f4 = np.array([1000, 100, 200, 150])
    c4 = np.array([
        [10, 50, 20, 30],
        [15, 5, 25, 35],
        [12, 40, 15, 25],
        [8, 45, 18, 20]
    ])
    instances.append((c4, f4, "Khumawala_Test"))
    
    # Instance 5: Larger instance
    np.random.seed(42)  # For reproducibility
    f5 = np.random.randint(100, 1000, size=8)
    c5 = np.random.randint(10, 100, size=(6, 8))
    instances.append((c5, f5, "Random_Large"))
    
    return instances

def analyze_single_instance(c: np.ndarray, f: np.ndarray, instance_name: str, 
                          verbose: bool = False) -> Dict[str, Any]:
    """
    Perform complete analysis of a single UFLP instance.
    
    Args:
        c: Transport cost matrix
        f: Fixed facility costs
        instance_name: Name of the instance
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with complete analysis results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"ANALYZING INSTANCE: {instance_name}")
        print(f"{'='*60}")
        print(f"Problem size: {c.shape[0]} customers, {len(f)} facilities")
        print(f"Fixed costs: {f}")
        print(f"Transport costs shape: {c.shape}")
    
    results = {
        'instance_name': instance_name,
        'num_customers': c.shape[0],
        'num_facilities': len(f),
        'fixed_costs': f.tolist(),
        'transport_costs_shape': c.shape
    }
    
    # B1-B4: Term Analysis
    if verbose:
        print(f"\n{'-'*30}")
        print("PSEUDO-BOOLEAN POLYNOMIAL ANALYSIS (B1-B4)")
        print(f"{'-'*30}")
    
    try:
        # Generate PBP and analyze terms
        pbp_df = cap_to_pbp_df(c, f, verbose=False)
        analyzer = TermAnalyzer(pbp_df)
        
        if verbose:
            analyzer.print_analysis_report(num_vars=len(f))
        
        term_analysis = analyzer.get_basic_analysis()
        results.update({
            'B1_linear_terms': term_analysis['linear'],
            'B2_quadratic_terms': term_analysis['quadratic'],
            'B3_cubic_terms': term_analysis['cubic'],
            'B4_nonlinear_terms': term_analysis['nonlinear'],
            'total_terms': len(pbp_df),
            'degree_distribution': analyzer.get_degree_distribution().to_dict()
        })
        
    except Exception as e:
        logger.error(f"Error in term analysis for {instance_name}: {str(e)}")
        results.update({
            'B1_linear_terms': -1,
            'B2_quadratic_terms': -1,
            'B3_cubic_terms': -1,
            'B4_nonlinear_terms': -1,
            'error': str(e)
        })
    
    # B5: Khumawala Rules Analysis
    if verbose:
        print(f"\n{'-'*30}")
        print("KHUMAWALA RULES ANALYSIS (B5)")
        print(f"{'-'*30}")
    
    try:
        khumawala = KhumawalaRules(c, f)
        khumawala_stats = khumawala.apply_iterative_khumawala_rules()
        
        if verbose:
            khumawala.print_status_report()
        
        results.update({
            'B5_rule1_applications': khumawala_stats['rule1_applications'],
            'B5_rule2_applications': khumawala_stats['rule2_applications'], 
            'B5_variables_fixed_rule1': khumawala_stats['variables_fixed_rule1'],
            'B5_variables_fixed_rule2': khumawala_stats['variables_fixed_rule2'],
            'B5_total_variables_fixed': khumawala_stats['total_variables_fixed'],
            'B5_facilities_opened': khumawala_stats['facilities_opened'],
            'B5_facilities_closed': khumawala_stats['facilities_closed'],
            'B5_facilities_undecided': khumawala_stats['facilities_undecided'],
            'B5_solved_to_optimality': khumawala_stats['solved_to_optimality'],
            'B5_open_facilities': sorted(list(khumawala.get_open_facilities())),
            'B5_closed_facilities': sorted(list(khumawala.get_closed_facilities())),
            'B5_undecided_facilities': sorted(list(khumawala.get_undecided_facilities()))
        })
        
        # Calculate problem reduction
        reduction_percentage = (khumawala_stats['total_variables_fixed'] / len(f)) * 100
        results['B5_problem_reduction_percentage'] = reduction_percentage
        
    except Exception as e:
        logger.error(f"Error in Khumawala analysis for {instance_name}: {str(e)}")
        results.update({
            'B5_error': str(e),
            'B5_solved_to_optimality': False
        })
    
    # B6: MQL Branching Analysis
    if verbose:
        print(f"\n{'-'*30}")
        print("MQL BRANCHING ANALYSIS (B6)")
        print(f"{'-'*30}")
    
    try:
        # Create instance dictionary for MQL branching
        instance_dict = {
            'name': instance_name,
            'facilities': list(range(len(f))),
            'customers': list(range(c.shape[0])),
            'facility_costs': f.tolist(),
            'assignment_costs': c.tolist()
        }
        
        # Apply MQL branching with recursive Khumawala rules
        mql_solver = MQLBranching()
        khumawala_for_mql = KhumawalaRules(c, f)
        mql_result = mql_solver.apply_mql_branching(instance_dict, khumawala_for_mql, max_iterations=10)
        
        if verbose:
            print(f"MQL Branching Results:")
            print(f"  Solved to optimality: {mql_result['solved_to_optimality']}")
            print(f"  Best objective: {mql_result.get('best_objective', 'N/A')}")
            print(f"  Total nodes: {mql_result['statistics']['total_nodes']}")
            print(f"  MQL branchings: {mql_result['statistics']['mql_branchings']}")
            print(f"  Nodes solved by Khumawala: {mql_result['statistics']['nodes_solved_by_khumawala']}")
            print(f"  Max depth: {mql_result['statistics']['max_depth']}")
        
        results.update({
            'B6_mql_solved_to_optimality': mql_result['solved_to_optimality'],
            'B6_mql_best_objective': mql_result.get('best_objective', float('inf')),
            'B6_mql_total_nodes': mql_result['statistics']['total_nodes'],
            'B6_mql_branchings': mql_result['statistics']['mql_branchings'],
            'B6_mql_nodes_solved_by_khumawala': mql_result['statistics']['nodes_solved_by_khumawala'],
            'B6_mql_max_depth': mql_result['statistics']['max_depth'],
            'B6_mql_khumawala_rule1_applications': mql_result['statistics']['khumawala_rule1_applications'],
            'B6_mql_khumawala_rule2_applications': mql_result['statistics']['khumawala_rule2_applications']
        })
        
    except Exception as e:
        logger.error(f"Error in MQL branching analysis for {instance_name}: {str(e)}")
        results.update({
            'B6_mql_error': str(e),
            'B6_mql_solved_to_optimality': False
        })
    
    # B7: AntiKhumawala Branching Rules Analysis
    if verbose:
        print(f"\n{'-'*30}")
        print("ANTIKHUMAWALA BRANCHING ANALYSIS (B7)")
        print(f"{'-'*30}")
    
    try:
        # Apply all AntiKhumawala rules
        anti_khumawala = AntiKhumawalaRules()
        
        # Test LBA rule
        khumawala_for_lba = KhumawalaRules(c, f)
        lba_result = anti_khumawala.apply_lba_rule(instance_dict, khumawala_for_lba)
        
        # Test SBA rule  
        khumawala_for_sba = KhumawalaRules(c, f)
        sba_result = anti_khumawala.apply_sba_rule(instance_dict, khumawala_for_sba)
        
        # Test FLBA rule
        khumawala_for_flba = KhumawalaRules(c, f)
        flba_result = anti_khumawala.apply_flba_rule(instance_dict, khumawala_for_flba)
        
        # Test FSBA rule
        khumawala_for_fsba = KhumawalaRules(c, f)
        fsba_result = anti_khumawala.apply_fsba_rule(instance_dict, khumawala_for_fsba)
        
        # Test SLBA rule
        khumawala_for_slba = KhumawalaRules(c, f)
        slba_result = anti_khumawala.apply_slba_rule(instance_dict, khumawala_for_slba)
        
        # Test SSBA rule
        khumawala_for_ssba = KhumawalaRules(c, f)
        ssba_result = anti_khumawala.apply_ssba_rule(instance_dict, khumawala_for_ssba)
        
        # Get final statistics
        anti_stats = anti_khumawala.get_statistics()
        
        if verbose:
            print(f"AntiKhumawala Branching Results:")
            print(f"  LBA solved to optimality: {lba_result.get('solved_to_optimality', False)}")
            print(f"  SBA solved to optimality: {sba_result.get('solved_to_optimality', False)}")
            print(f"  FLBA solved to optimality: {flba_result.get('solved_to_optimality', False)}")
            print(f"  FSBA solved to optimality: {fsba_result.get('solved_to_optimality', False)}")
            print(f"  SLBA solved to optimality: {slba_result.get('solved_to_optimality', False)}")
            print(f"  SSBA solved to optimality: {ssba_result.get('solved_to_optimality', False)}")
            print(f"  Total branching rule applications: {anti_stats['total_branchings']}")
            for rule, count in anti_stats['branching_rule_counts'].items():
                print(f"  {rule} count: {count}")
        
        results.update({
            'B7_lba_solved_to_optimality': lba_result.get('solved_to_optimality', False),
            'B7_lba_branchings': lba_result.get('lba_branchings', 0),
            'B7_lba_khumawala_applications': lba_result.get('khumawala_applications', 0),
            
            'B7_sba_solved_to_optimality': sba_result.get('solved_to_optimality', False),
            'B7_sba_branchings': sba_result.get('sba_branchings', 0),
            'B7_sba_khumawala_applications': sba_result.get('khumawala_applications', 0),
            
            'B7_flba_solved_to_optimality': flba_result.get('solved_to_optimality', False),
            'B7_flba_branchings': flba_result.get('flba_branchings', 0),
            'B7_flba_khumawala_applications': flba_result.get('khumawala_applications', 0),
            
            'B7_fsba_solved_to_optimality': fsba_result.get('solved_to_optimality', False),
            'B7_fsba_branchings': fsba_result.get('fsba_branchings', 0),
            'B7_fsba_khumawala_applications': fsba_result.get('khumawala_applications', 0),
            
            'B7_slba_solved_to_optimality': slba_result.get('solved_to_optimality', False),
            'B7_slba_branchings': slba_result.get('slba_branchings', 0),
            'B7_slba_khumawala_applications': slba_result.get('khumawala_applications', 0),
            
            'B7_ssba_solved_to_optimality': ssba_result.get('solved_to_optimality', False),
            'B7_ssba_branchings': ssba_result.get('ssba_branchings', 0),
            'B7_ssba_khumawala_applications': ssba_result.get('khumawala_applications', 0),
            
            'B7_total_branching_applications': anti_stats['total_branchings'],
            'B7_lba_count': anti_stats['branching_rule_counts']['LBA'],
            'B7_sba_count': anti_stats['branching_rule_counts']['SBA'],
            'B7_flba_count': anti_stats['branching_rule_counts']['FLBA'],
            'B7_fsba_count': anti_stats['branching_rule_counts']['FSBA'],
            'B7_slba_count': anti_stats['branching_rule_counts']['SLBA'],
            'B7_ssba_count': anti_stats['branching_rule_counts']['SSBA']
        })
        
    except Exception as e:
        logger.error(f"Error in AntiKhumawala analysis for {instance_name}: {str(e)}")
        results.update({
            'B7_error': str(e),
            'B7_lba_applied': False,
            'B7_sba_applied': False
        })
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"COMPLETED ANALYSIS: {instance_name}")
        print(f"{'='*60}")
    
    return results

def analyze_all_instances(instances: List[Tuple[np.ndarray, np.ndarray, str]], 
                         verbose: bool = False) -> pd.DataFrame:
    """
    Analyze all instances and return results as DataFrame.
    
    Args:
        instances: List of (c, f, name) tuples
        verbose: Whether to print detailed output
        
    Returns:
        DataFrame with analysis results for all instances
    """
    results = []
    
    for i, (c, f, name) in enumerate(instances):
        logger.info(f"Analyzing instance {i+1}/{len(instances)}: {name}")
        
        try:
            result = analyze_single_instance(c, f, name, verbose=verbose)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to analyze {name}: {str(e)}")
            error_result = {
                'instance_name': name,
                'error': str(e),
                'B5_solved_to_optimality': False
            }
            results.append(error_result)
    
    return pd.DataFrame(results)

def create_summary_report(df: pd.DataFrame) -> str:
    """
    Create a comprehensive summary report.
    
    Args:
        df: DataFrame with analysis results
        
    Returns:
        Formatted summary report string
    """
    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE KHUMAWALA ANALYSIS SUMMARY REPORT")
    report.append("=" * 80)
    
    # Overall statistics
    total_instances = len(df)
    solved_optimally = df['B5_solved_to_optimality'].sum() if 'B5_solved_to_optimality' in df.columns else 0
    
    report.append(f"\nOVERALL STATISTICS:")
    report.append(f"  Total instances analyzed: {total_instances}")
    report.append(f"  Solved to optimality by Khumawala rules: {solved_optimally}")
    report.append(f"  Success rate: {(solved_optimally/total_instances)*100:.1f}%")
    
    # B1-B4 Summary
    if all(col in df.columns for col in ['B1_linear_terms', 'B2_quadratic_terms', 'B3_cubic_terms', 'B4_nonlinear_terms']):
        report.append(f"\nTERM ANALYSIS SUMMARY (B1-B4):")
        valid_rows = df[df['B1_linear_terms'] >= 0]  # Exclude error cases
        if not valid_rows.empty:
            report.append(f"  Average linear terms (B1): {valid_rows['B1_linear_terms'].mean():.1f}")
            report.append(f"  Average quadratic terms (B2): {valid_rows['B2_quadratic_terms'].mean():.1f}")
            report.append(f"  Average cubic terms (B3): {valid_rows['B3_cubic_terms'].mean():.1f}")
            report.append(f"  Average non-linear terms (B4): {valid_rows['B4_nonlinear_terms'].mean():.1f}")
    
    # B5 Summary  
    if all(col in df.columns for col in ['B5_variables_fixed_rule1', 'B5_variables_fixed_rule2']):
        report.append(f"\nKHUMAWALA RULES SUMMARY (B5):")
        valid_rows = df[df['B5_solved_to_optimality'].notna()]
        if not valid_rows.empty:
            report.append(f"  Average variables fixed by Rule 1: {valid_rows['B5_variables_fixed_rule1'].mean():.1f}")
            report.append(f"  Average variables fixed by Rule 2: {valid_rows['B5_variables_fixed_rule2'].mean():.1f}")
            report.append(f"  Average total variables fixed: {valid_rows['B5_total_variables_fixed'].mean():.1f}")
            
            if 'B5_problem_reduction_percentage' in valid_rows.columns:
                report.append(f"  Average problem reduction: {valid_rows['B5_problem_reduction_percentage'].mean():.1f}%")
    
    # B6 Summary (MQL Branching)
    if 'B6_mql_solved_to_optimality' in df.columns:
        report.append(f"\nMQL BRANCHING SUMMARY (B6):")
        valid_rows = df[df['B6_mql_solved_to_optimality'].notna()]
        if not valid_rows.empty:
            mql_solved = valid_rows['B6_mql_solved_to_optimality'].sum()
            report.append(f"  Instances solved by MQL branching: {mql_solved}/{len(valid_rows)}")
            report.append(f"  MQL success rate: {(mql_solved/len(valid_rows))*100:.1f}%")
            
            if 'B6_mql_branchings' in valid_rows.columns:
                avg_branchings = valid_rows['B6_mql_branchings'].mean()
                report.append(f"  Average MQL branchings per instance: {avg_branchings:.1f}")
            
            if 'B6_mql_nodes_solved_by_khumawala' in valid_rows.columns:
                avg_khumawala_nodes = valid_rows['B6_mql_nodes_solved_by_khumawala'].mean()
                report.append(f"  Average nodes solved by Khumawala in MQL: {avg_khumawala_nodes:.1f}")
    
    # B7 Summary (AntiKhumawala Rules)
    if 'B7_lba_solved_to_optimality' in df.columns:
        report.append(f"\nANTIKHUMAWALA RULES SUMMARY (B7):")
        valid_rows = df[df['B7_lba_solved_to_optimality'].notna()]
        if not valid_rows.empty:
            # Count solved instances for each rule
            lba_solved = valid_rows['B7_lba_solved_to_optimality'].sum()
            sba_solved = valid_rows['B7_sba_solved_to_optimality'].sum() if 'B7_sba_solved_to_optimality' in valid_rows.columns else 0
            flba_solved = valid_rows['B7_flba_solved_to_optimality'].sum() if 'B7_flba_solved_to_optimality' in valid_rows.columns else 0
            fsba_solved = valid_rows['B7_fsba_solved_to_optimality'].sum() if 'B7_fsba_solved_to_optimality' in valid_rows.columns else 0
            slba_solved = valid_rows['B7_slba_solved_to_optimality'].sum() if 'B7_slba_solved_to_optimality' in valid_rows.columns else 0
            ssba_solved = valid_rows['B7_ssba_solved_to_optimality'].sum() if 'B7_ssba_solved_to_optimality' in valid_rows.columns else 0
            
            total_instances = len(valid_rows)
            
            report.append(f"  LBA instances solved to optimality: {lba_solved}/{total_instances} ({(lba_solved/total_instances)*100:.1f}%)")
            report.append(f"  SBA instances solved to optimality: {sba_solved}/{total_instances} ({(sba_solved/total_instances)*100:.1f}%)")
            report.append(f"  FLBA instances solved to optimality: {flba_solved}/{total_instances} ({(flba_solved/total_instances)*100:.1f}%)")
            report.append(f"  FSBA instances solved to optimality: {fsba_solved}/{total_instances} ({(fsba_solved/total_instances)*100:.1f}%)")
            report.append(f"  SLBA instances solved to optimality: {slba_solved}/{total_instances} ({(slba_solved/total_instances)*100:.1f}%)")
            report.append(f"  SSBA instances solved to optimality: {ssba_solved}/{total_instances} ({(ssba_solved/total_instances)*100:.1f}%)")
            
            # Average branching counts
            avg_lba_branchings = valid_rows['B7_lba_branchings'].mean() if 'B7_lba_branchings' in valid_rows.columns else 0
            avg_sba_branchings = valid_rows['B7_sba_branchings'].mean() if 'B7_sba_branchings' in valid_rows.columns else 0
            avg_flba_branchings = valid_rows['B7_flba_branchings'].mean() if 'B7_flba_branchings' in valid_rows.columns else 0
            avg_fsba_branchings = valid_rows['B7_fsba_branchings'].mean() if 'B7_fsba_branchings' in valid_rows.columns else 0
            avg_slba_branchings = valid_rows['B7_slba_branchings'].mean() if 'B7_slba_branchings' in valid_rows.columns else 0
            avg_ssba_branchings = valid_rows['B7_ssba_branchings'].mean() if 'B7_ssba_branchings' in valid_rows.columns else 0
            
            report.append(f"  Average LBA branchings per instance: {avg_lba_branchings:.1f}")
            report.append(f"  Average SBA branchings per instance: {avg_sba_branchings:.1f}")
            report.append(f"  Average FLBA branchings per instance: {avg_flba_branchings:.1f}")
            report.append(f"  Average FSBA branchings per instance: {avg_fsba_branchings:.1f}")
            report.append(f"  Average SLBA branchings per instance: {avg_slba_branchings:.1f}")
            report.append(f"  Average SSBA branchings per instance: {avg_ssba_branchings:.1f}")
            
            if 'B7_total_branching_applications' in valid_rows.columns:
                total_branching = valid_rows['B7_total_branching_applications'].sum()
                report.append(f"  Total branching rule applications: {total_branching}")
    
    # Instance-by-instance details
    report.append(f"\nINSTANCE-BY-INSTANCE RESULTS:")
    report.append(f"{'Instance':<20} {'Size':<10} {'B1':<5} {'B2':<5} {'B3':<5} {'B4':<5} {'R1':<5} {'R2':<5} {'Opt':<5}")
    report.append("-" * 80)
    
    for _, row in df.iterrows():
        name = row.get('instance_name', 'Unknown')[:19]
        size = f"{row.get('num_customers', '?')}x{row.get('num_facilities', '?')}"
        b1 = row.get('B1_linear_terms', '?')
        b2 = row.get('B2_quadratic_terms', '?')
        b3 = row.get('B3_cubic_terms', '?')
        b4 = row.get('B4_nonlinear_terms', '?')
        r1 = row.get('B5_variables_fixed_rule1', '?')
        r2 = row.get('B5_variables_fixed_rule2', '?')
        opt = 'Yes' if row.get('B5_solved_to_optimality', False) else 'No'
        
        report.append(f"{name:<20} {size:<10} {b1:<5} {b2:<5} {b3:<5} {b4:<5} {r1:<5} {r2:<5} {opt:<5}")
    
    report.append("=" * 80)
    report.append("\nLegend:")
    report.append("  B1-B4: Number of linear, quadratic, cubic, and non-linear terms")
    report.append("  R1-R2: Variables fixed by 1st and 2nd Khumawala rules")
    report.append("  Opt: Solved to optimality by Khumawala rules alone")
    
    return "\n".join(report)

def main():
    """Main function to run the comprehensive analysis."""
    parser = argparse.ArgumentParser(description='Comprehensive Khumawala Analysis for UFLP')
    parser.add_argument('--output', '-o', type=str, help='Output file for results (CSV format)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--instances', '-i', type=str, help='JSON file with custom instances (not implemented)')
    
    args = parser.parse_args()
    
    # Create test instances
    logger.info("Creating test instances...")
    instances = create_test_instances()
    
    # Analyze all instances
    logger.info(f"Starting analysis of {len(instances)} instances...")
    results_df = analyze_all_instances(instances, verbose=args.verbose)
    
    # Create summary report
    summary = create_summary_report(results_df)
    print(summary)
    
    # Save results if requested
    if args.output:
        try:
            # Save detailed results
            results_df.to_csv(args.output, index=False)
            logger.info(f"Detailed results saved to {args.output}")
            
            # Save summary report
            summary_file = args.output.replace('.csv', '_summary.txt')
            with open(summary_file, 'w') as f:
                f.write(summary)
            logger.info(f"Summary report saved to {summary_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    # Return success based on whether any instances were solved optimally
    solved_count = results_df['B5_solved_to_optimality'].sum() if 'B5_solved_to_optimality' in results_df.columns else 0
    if solved_count > 0:
        logger.info(f"Analysis completed successfully. {solved_count} instances solved to optimality.")
        return 0
    else:
        logger.warning("Analysis completed but no instances were solved to optimality.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 