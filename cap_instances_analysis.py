#!/usr/bin/env python3
"""
Memory-Safe Cap Instances Analysis Script for B6-B7 Functionality

This script tests our B6-B7 implementation on real cap instances with memory monitoring
and safeguards to prevent system overload.

Usage:
    python cap_instances_analysis.py [--max-size 20] [--output cap_results.csv] [--verbose]
"""

import sys
import os
import numpy as np
import pandas as pd
import argparse
import psutil
import gc
from typing import List, Tuple, Dict, Any
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pbp.generate import cap_to_pbp_df
from khumawala.analysis import TermAnalyzer
from khumawala.rules import KhumawalaRules
from khumawala.branching import MQLBranching, AntiKhumawalaRules, AdvancedBranchingSolver
from cap.cap_matrix_reader import read_cap_matrix

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Monitor memory usage and provide safeguards"""
    
    def __init__(self, max_memory_gb=8.0):
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.process = psutil.Process()
        
    def get_memory_usage(self):
        """Get current memory usage in GB"""
        return self.process.memory_info().rss / (1024 * 1024 * 1024)
    
    def check_memory_limit(self):
        """Check if memory usage is approaching limit"""
        current_memory = self.process.memory_info().rss
        return current_memory > (self.max_memory_bytes * 0.8)  # 80% threshold
    
    def log_memory_status(self, operation=""):
        """Log current memory status"""
        memory_gb = self.get_memory_usage()
        logger.info(f"Memory usage {operation}: {memory_gb:.2f} GB")
        return memory_gb

def parse_cap_instance(filepath: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Parse a cap instance file and return cost matrices and instance name.
    
    Args:
        filepath: Path to cap instance file
        
    Returns:
        Tuple of (transport_costs, facility_costs, instance_name)
    """
    instance_name = Path(filepath).stem
    
    # Use the existing cap matrix reader
    m, n, facility_costs, transport_costs = read_cap_matrix(filepath)
    
    logger.info(f"Parsing {instance_name}: {n} customers, {m} facilities")
    
    return transport_costs, facility_costs, instance_name

def get_cap_instances(max_size=None) -> List[str]:
    """
    Get list of cap instance files, sorted by size.
    
    Args:
        max_size: Maximum number of facilities to consider
        
    Returns:
        List of file paths sorted by problem size
    """
    cap_dir = Path("data/cap")
    if not cap_dir.exists():
        logger.error(f"Cap directory not found: {cap_dir}")
        return []
    
    instance_files = []
    
    for cap_file in cap_dir.glob("cap*.txt"):
        if cap_file.name == "uncapopt.txt":
            continue
            
        try:
            # Quick check of problem size
            with open(cap_file, 'r') as f:
                first_line = f.readline().strip().split()
                num_customers = int(first_line[0])
                num_facilities = int(first_line[1])
                
                if max_size and num_facilities > max_size:
                    logger.debug(f"Skipping {cap_file.name}: {num_facilities} facilities > {max_size}")
                    continue
                    
                instance_files.append((cap_file, num_customers, num_facilities))
                
        except Exception as e:
            logger.warning(f"Could not parse {cap_file}: {e}")
            continue
    
    # Sort by total problem size (customers * facilities)
    instance_files.sort(key=lambda x: x[1] * x[2])
    
    logger.info(f"Found {len(instance_files)} cap instances")
    for filepath, customers, facilities in instance_files[:5]:
        logger.info(f"  {filepath.name}: {customers}x{facilities}")
    if len(instance_files) > 5:
        logger.info(f"  ... and {len(instance_files) - 5} more")
    
    return [str(filepath) for filepath, _, _ in instance_files]

def analyze_cap_instance_safe(filepath: str, memory_monitor: MemoryMonitor, 
                             verbose: bool = False) -> Dict[str, Any]:
    """
    Safely analyze a single cap instance with memory monitoring.
    
    Args:
        filepath: Path to cap instance file
        memory_monitor: MemoryMonitor instance
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with analysis results
    """
    memory_monitor.log_memory_status("before parsing")
    
    try:
        # Parse instance
        c, f, instance_name = parse_cap_instance(filepath)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"ANALYZING CAP INSTANCE: {instance_name}")
            print(f"{'='*60}")
            print(f"Problem size: {c.shape[0]} customers, {len(f)} facilities")
            print(f"Total variables: {len(f)}")
        
        results = {
            'instance_name': instance_name,
            'num_customers': c.shape[0],
            'num_facilities': len(f),
            'file_path': filepath
        }
        
        memory_monitor.log_memory_status("after parsing")
        
        # Check memory before PBP generation
        if memory_monitor.check_memory_limit():
            logger.warning(f"Memory limit approached, skipping PBP analysis for {instance_name}")
            results.update({
                'B1_linear_terms': -1,
                'B2_quadratic_terms': -1, 
                'B3_cubic_terms': -1,
                'B4_nonlinear_terms': -1,
                'memory_limit_reached': True
            })
        else:
            # B1-B4: Term Analysis (most memory intensive)
            if verbose:
                print(f"\n{'-'*30}")
                print("PSEUDO-BOOLEAN POLYNOMIAL ANALYSIS (B1-B4)")
                print(f"{'-'*30}")
            
            try:
                pbp_df = cap_to_pbp_df(c, f, verbose=False)
                memory_monitor.log_memory_status("after PBP generation")
                
                analyzer = TermAnalyzer(pbp_df)
                
                if verbose:
                    analyzer.print_analysis_report(num_vars=len(f))
                
                term_analysis = analyzer.get_basic_analysis()
                results.update({
                    'B1_linear_terms': term_analysis['linear'],
                    'B2_quadratic_terms': term_analysis['quadratic'],
                    'B3_cubic_terms': term_analysis['cubic'],
                    'B4_nonlinear_terms': term_analysis['nonlinear'],
                    'total_terms': len(pbp_df)
                })
                
                # Clean up PBP data immediately
                del pbp_df, analyzer
                gc.collect()
                memory_monitor.log_memory_status("after PBP cleanup")
                
            except Exception as e:
                logger.error(f"Error in PBP analysis for {instance_name}: {str(e)}")
                results.update({
                    'B1_linear_terms': -1,
                    'B2_quadratic_terms': -1,
                    'B3_cubic_terms': -1,
                    'B4_nonlinear_terms': -1,
                    'pbp_error': str(e)
                })
        
        # B5: Khumawala Rules Analysis (lighter on memory)
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
                'B5_solved_to_optimality': khumawala_stats['solved_to_optimality'],
                'B5_problem_reduction_percentage': (khumawala_stats['total_variables_fixed'] / len(f)) * 100
            })
            
            memory_monitor.log_memory_status("after Khumawala analysis")
            
        except Exception as e:
            logger.error(f"Error in Khumawala analysis for {instance_name}: {str(e)}")
            results.update({
                'B5_error': str(e),
                'B5_solved_to_optimality': False
            })
        
        # B6-B7: Advanced Branching (only if problem is small enough or Khumawala solved it)
        solved_by_khumawala = results.get('B5_solved_to_optimality', False)
        problem_size = len(f)
        
        if solved_by_khumawala or problem_size <= 30:  # Only test on small problems or solved ones
            if verbose:
                print(f"\n{'-'*30}")
                print("ADVANCED BRANCHING ANALYSIS (B6-B7)")
                print(f"{'-'*30}")
            
            try:
                # Create instance dictionary
                instance_dict = {
                    'name': instance_name,
                    'facilities': list(range(len(f))),
                    'customers': list(range(c.shape[0])),
                    'facility_costs': f.tolist(),
                    'assignment_costs': c.tolist()
                }
                
                # B6: MQL Branching (quick test)
                mql_solver = MQLBranching()
                khumawala_for_mql = KhumawalaRules(c, f)
                mql_result = mql_solver.apply_mql_branching(instance_dict, khumawala_for_mql, max_iterations=5)
                
                results.update({
                    'B6_mql_solved_to_optimality': mql_result['solved_to_optimality'],
                    'B6_mql_branchings': mql_result['statistics']['mql_branchings'],
                    'B6_mql_nodes_solved_by_khumawala': mql_result['statistics']['nodes_solved_by_khumawala']
                })
                
                # B7: Test only fast AntiKhumawala rules to save time
                anti_khumawala = AntiKhumawalaRules()
                
                # Test FLBA (fast rule)
                khumawala_for_flba = KhumawalaRules(c, f)
                flba_result = anti_khumawala.apply_flba_rule(instance_dict, khumawala_for_flba)
                
                results.update({
                    'B7_flba_solved_to_optimality': flba_result.get('solved_to_optimality', False),
                    'B7_flba_branchings': flba_result.get('flba_branchings', 0)
                })
                
                memory_monitor.log_memory_status("after B6-B7 analysis")
                
            except Exception as e:
                logger.error(f"Error in B6-B7 analysis for {instance_name}: {str(e)}")
                results.update({
                    'B6_B7_error': str(e),
                    'B6_mql_solved_to_optimality': False,
                    'B7_flba_solved_to_optimality': False
                })
        else:
            logger.info(f"Skipping B6-B7 for large instance {instance_name} (size: {problem_size})")
            results.update({
                'B6_mql_solved_to_optimality': 'skipped_large',
                'B7_flba_solved_to_optimality': 'skipped_large'
            })
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"COMPLETED ANALYSIS: {instance_name}")
            print(f"{'='*60}")
        
        # Clean up
        del c, f
        gc.collect()
        
        return results
        
    except Exception as e:
        import traceback
        logger.error(f"Failed to analyze {filepath}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            'instance_name': Path(filepath).stem,
            'error': str(e),
            'analysis_failed': True
        }

def main():
    """Main function to run cap instances analysis."""
    parser = argparse.ArgumentParser(description='Memory-Safe Cap Instances Analysis for B6-B7')
    parser.add_argument('--max-size', type=int, default=50, help='Maximum number of facilities to analyze')
    parser.add_argument('--max-memory', type=float, default=8.0, help='Maximum memory usage in GB')
    parser.add_argument('--output', '-o', type=str, help='Output file for results (CSV format)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--test-only', action='store_true', help='Test only a few small instances')
    
    args = parser.parse_args()
    
    # Initialize memory monitor
    memory_monitor = MemoryMonitor(max_memory_gb=args.max_memory)
    memory_monitor.log_memory_status("at startup")
    
    # Get cap instances
    if args.test_only:
        logger.info("Test mode: analyzing only small instances")
        max_size = min(args.max_size, 50)  # Allow up to 50 facilities for testing
    else:
        max_size = args.max_size
        
    instance_files = get_cap_instances(max_size=max_size)
    
    if not instance_files:
        logger.error("No cap instances found!")
        return 1
    
    # Limit number of instances in test mode
    if args.test_only:
        instance_files = instance_files[:3]
        logger.info(f"Test mode: analyzing {len(instance_files)} instances")
    
    # Analyze instances
    results = []
    successful_analyses = 0
    
    for i, filepath in enumerate(instance_files):
        logger.info(f"Analyzing instance {i+1}/{len(instance_files)}: {Path(filepath).name}")
        
        # Check memory before each instance
        if memory_monitor.check_memory_limit():
            logger.warning("Memory limit approached, stopping analysis")
            break
        
        try:
            result = analyze_cap_instance_safe(filepath, memory_monitor, verbose=args.verbose)
            results.append(result)
            
            if not result.get('analysis_failed', False):
                successful_analyses += 1
                
        except Exception as e:
            logger.error(f"Fatal error analyzing {filepath}: {str(e)}")
            results.append({
                'instance_name': Path(filepath).stem,
                'fatal_error': str(e)
            })
        
        # Force garbage collection between instances
        gc.collect()
        memory_monitor.log_memory_status(f"after instance {i+1}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "="*80)
    print("CAP INSTANCES ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total instances attempted: {len(instance_files)}")
    print(f"Successful analyses: {successful_analyses}")
    print(f"Final memory usage: {memory_monitor.get_memory_usage():.2f} GB")
    
    if successful_analyses > 0:
        # B5 Summary
        solved_by_khumawala = results_df['B5_solved_to_optimality'].sum() if 'B5_solved_to_optimality' in results_df.columns else 0
        print(f"\nKhumawala Rules Success Rate: {solved_by_khumawala}/{successful_analyses} ({(solved_by_khumawala/successful_analyses)*100:.1f}%)")
        
        # Problem sizes
        if 'num_facilities' in results_df.columns:
            valid_sizes = results_df[results_df['num_facilities'].notna()]
            if not valid_sizes.empty:
                print(f"Problem sizes: {valid_sizes['num_facilities'].min()}-{valid_sizes['num_facilities'].max()} facilities")
                print(f"Average problem size: {valid_sizes['num_facilities'].mean():.1f} facilities")
        
        # B6-B7 results if available
        if 'B6_mql_solved_to_optimality' in results_df.columns:
            b6_solved = results_df['B6_mql_solved_to_optimality'].apply(lambda x: x is True).sum()
            b6_tested = results_df['B6_mql_solved_to_optimality'].apply(lambda x: x is not None and x != 'skipped_large').sum()
            if b6_tested > 0:
                print(f"B6 MQL Branching Success Rate: {b6_solved}/{b6_tested} ({(b6_solved/b6_tested)*100:.1f}%)")
        
        if 'B7_flba_solved_to_optimality' in results_df.columns:
            b7_solved = results_df['B7_flba_solved_to_optimality'].apply(lambda x: x is True).sum()
            b7_tested = results_df['B7_flba_solved_to_optimality'].apply(lambda x: x is not None and x != 'skipped_large').sum()
            if b7_tested > 0:
                print(f"B7 FLBA Rule Success Rate: {b7_solved}/{b7_tested} ({(b7_solved/b7_tested)*100:.1f}%)")
    
    # Save results
    if args.output:
        try:
            results_df.to_csv(args.output, index=False)
            logger.info(f"Results saved to {args.output}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    print("="*80)
    
    return 0 if successful_analyses > 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 