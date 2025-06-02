"""
Analysis module for pseudo-Boolean polynomial term analysis.

This module provides functions to analyze pseudo-Boolean polynomials generated
from uncapacitated facility location problems, particularly for counting
different types of terms as required by the Khumawala rules implementation.
"""

import numpy as np
import pandas as pd
from bitarray import bitarray
from bitarray.util import int2ba
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

class TermAnalyzer:
    """
    Analyzer for pseudo-Boolean polynomial terms.
    
    This class provides methods to analyze the structure of pseudo-Boolean
    polynomials generated from uncapacitated facility location problems.
    """
    
    def __init__(self, pbp_df: pd.DataFrame):
        """
        Initialize the term analyzer with a pseudo-Boolean polynomial DataFrame.
        
        Args:
            pbp_df (pd.DataFrame): DataFrame with columns ['y', 'coeffs', 'degree']
                where 'y' contains bitarrays representing variable sets
        """
        self.pbp_df = pbp_df.copy()
        self.pbp_df = self.pbp_df[self.pbp_df['coeffs'] != 0]  # Remove zero coefficients
        
    def count_linear_terms(self) -> int:
        """
        Count the number of non-zero linear terms (degree 1).
        
        Returns:
            int: Number of linear terms (B1)
        """
        linear_terms = self.pbp_df[self.pbp_df['degree'] == 1]
        return len(linear_terms)
    
    def count_quadratic_terms(self) -> int:
        """
        Count the number of non-zero quadratic terms (degree 2).
        
        Returns:
            int: Number of quadratic terms (B2)
        """
        quadratic_terms = self.pbp_df[self.pbp_df['degree'] == 2]
        return len(quadratic_terms)
    
    def count_cubic_terms(self) -> int:
        """
        Count the number of non-zero cubic terms (degree 3).
        
        Returns:
            int: Number of cubic terms (B3)
        """
        cubic_terms = self.pbp_df[self.pbp_df['degree'] == 3]
        return len(cubic_terms)
    
    def count_nonlinear_terms(self) -> int:
        """
        Count the total number of all non-linear terms (degree > 1).
        
        Returns:
            int: Total number of non-linear terms (B4)
        """
        nonlinear_terms = self.pbp_df[self.pbp_df['degree'] > 1]
        return len(nonlinear_terms)
    
    def get_basic_analysis(self) -> Dict[str, int]:
        """
        Get all basic term counts (B1-B4) in a single call.
        
        Returns:
            Dict[str, int]: Dictionary with keys 'linear', 'quadratic', 'cubic', 'nonlinear'
        """
        return {
            'linear': self.count_linear_terms(),      # B1
            'quadratic': self.count_quadratic_terms(), # B2
            'cubic': self.count_cubic_terms(),        # B3
            'nonlinear': self.count_nonlinear_terms() # B4
        }
    
    def get_degree_distribution(self) -> pd.Series:
        """
        Get the distribution of terms by degree.
        
        Returns:
            pd.Series: Series with degree as index and count as values
        """
        return self.pbp_df['degree'].value_counts().sort_index()
    
    def get_coefficient_statistics(self) -> Dict[str, float]:
        """
        Get basic statistics about coefficients.
        
        Returns:
            Dict[str, float]: Statistics including mean, std, min, max of coefficients
        """
        coeffs = self.pbp_df['coeffs']
        return {
            'mean': float(coeffs.mean()),
            'std': float(coeffs.std()),
            'min': float(coeffs.min()),
            'max': float(coeffs.max()),
            'sum': float(coeffs.sum())
        }
    
    def get_variable_usage(self, num_vars: int) -> Dict[str, int]:
        """
        Analyze which variables appear most frequently in terms.
        
        Args:
            num_vars (int): Total number of variables in the problem
            
        Returns:
            Dict[str, int]: Dictionary with variable indices as keys and usage counts as values
        """
        var_usage = {f'var_{i+1}': 0 for i in range(num_vars)}
        
        for _, row in self.pbp_df.iterrows():
            y_bits = row['y']
            for i, bit in enumerate(y_bits):
                if bit and i < num_vars:
                    var_usage[f'var_{i+1}'] += 1
                    
        return var_usage
    
    def print_analysis_report(self, num_vars: int = None) -> None:
        """
        Print a comprehensive analysis report.
        
        Args:
            num_vars (int, optional): Number of variables for usage analysis
        """
        basic_stats = self.get_basic_analysis()
        coeff_stats = self.get_coefficient_statistics()
        degree_dist = self.get_degree_distribution()
        
        print("=" * 60)
        print("PSEUDO-BOOLEAN POLYNOMIAL ANALYSIS REPORT")
        print("=" * 60)
        
        print("\nBASIC TERM COUNTS:")
        print(f"  B1 - Linear terms (degree 1):     {basic_stats['linear']:>6}")
        print(f"  B2 - Quadratic terms (degree 2):  {basic_stats['quadratic']:>6}")
        print(f"  B3 - Cubic terms (degree 3):      {basic_stats['cubic']:>6}")
        print(f"  B4 - Non-linear terms (degree>1): {basic_stats['nonlinear']:>6}")
        
        print(f"\nTOTAL TERMS: {len(self.pbp_df)}")
        
        print("\nDEGREE DISTRIBUTION:")
        for degree, count in degree_dist.items():
            print(f"  Degree {degree}: {count} terms")
        
        print(f"\nCOEFFICIENT STATISTICS:")
        print(f"  Mean: {coeff_stats['mean']:>10.2f}")
        print(f"  Std:  {coeff_stats['std']:>10.2f}")
        print(f"  Min:  {coeff_stats['min']:>10.2f}")
        print(f"  Max:  {coeff_stats['max']:>10.2f}")
        print(f"  Sum:  {coeff_stats['sum']:>10.2f}")
        
        if num_vars:
            var_usage = self.get_variable_usage(num_vars)
            print(f"\nVARIABLE USAGE (Top 10):")
            sorted_usage = sorted(var_usage.items(), key=lambda x: x[1], reverse=True)
            for var, count in sorted_usage[:10]:
                print(f"  {var}: {count} times")
        
        print("=" * 60)


def analyze_instance(c: np.array, f: np.array, verbose: bool = True) -> Dict[str, int]:
    """
    Analyze a single UFLP instance and return basic term counts.
    
    Args:
        c (np.array): Transport cost matrix
        f (np.array): Fixed costs array
        verbose (bool): Whether to print detailed analysis
        
    Returns:
        Dict[str, int]: Dictionary with basic analysis results
    """
    # Import here to avoid circular imports
    import sys
    import os
    
    # Add the src directory to the path if not already there
    src_path = os.path.join(os.path.dirname(__file__), '..', '..')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    from src.pbp.generate import cap_to_pbp_df
    
    # Generate pseudo-Boolean polynomial
    pbp_df = cap_to_pbp_df(c, f, verbose=False)
    
    # Create analyzer and perform analysis
    analyzer = TermAnalyzer(pbp_df)
    
    if verbose:
        analyzer.print_analysis_report(num_vars=len(f))
    
    return analyzer.get_basic_analysis()


def batch_analyze_instances(instances: List[Tuple[np.array, np.array]], 
                          instance_names: List[str] = None) -> pd.DataFrame:
    """
    Analyze multiple UFLP instances and return results in a DataFrame.
    
    Args:
        instances (List[Tuple[np.array, np.array]]): List of (c, f) tuples
        instance_names (List[str], optional): Names for the instances
        
    Returns:
        pd.DataFrame: DataFrame with analysis results for all instances
    """
    results = []
    
    if instance_names is None:
        instance_names = [f"Instance_{i+1}" for i in range(len(instances))]
    
    for i, (c, f) in enumerate(instances):
        logger.info(f"Analyzing {instance_names[i]}...")
        
        try:
            analysis = analyze_instance(c, f, verbose=False)
            analysis['instance'] = instance_names[i]
            analysis['num_facilities'] = len(f)
            analysis['num_customers'] = c.shape[0]
            results.append(analysis)
            
        except Exception as e:
            logger.error(f"Error analyzing {instance_names[i]}: {str(e)}")
            continue
    
    df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    column_order = ['instance', 'num_facilities', 'num_customers', 
                   'linear', 'quadratic', 'cubic', 'nonlinear']
    df = df[column_order]
    
    return df 