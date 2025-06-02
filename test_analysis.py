#!/usr/bin/env python3
"""
Test script for the Khumawala analysis module.

This script tests the basic term analysis functionality (B1-B4) 
using the existing example data from generate.py.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pbp.generate import cap_to_pbp_df
from khumawala.analysis import TermAnalyzer, analyze_instance, batch_analyze_instances

def test_basic_analysis():
    """Test basic analysis with the example data from generate.py"""
    
    print("Testing Khumawala Analysis Module")
    print("=" * 50)
    
    # Use the same example data from generate.py
    f = np.array([1000, 1200, 800, 1500, 900, 1100, 1300])
    c = np.array([
        [50, 60, 40, 70, 45, 55, 65],
        [45, 55, 35, 65, 40, 50, 60],
        [60, 70, 50, 80, 55, 65, 75],
        [40, 50, 30, 60, 35, 45, 55],
        [55, 65, 45, 75, 50, 60, 70]
    ])
    
    print("Input Data:")
    print(f"Fixed costs (f): {f}")
    print(f"Transport costs (c) shape: {c.shape}")
    print()
    
    # Generate PBP
    print("Generating pseudo-Boolean polynomial...")
    pbp_df = cap_to_pbp_df(c, f, verbose=False)
    print(f"Generated {len(pbp_df)} terms")
    print()
    
    # Test TermAnalyzer
    print("Testing TermAnalyzer class:")
    analyzer = TermAnalyzer(pbp_df)
    
    # Get basic analysis
    basic_stats = analyzer.get_basic_analysis()
    print("Basic Analysis Results:")
    print(f"  B1 - Linear terms: {basic_stats['linear']}")
    print(f"  B2 - Quadratic terms: {basic_stats['quadratic']}")
    print(f"  B3 - Cubic terms: {basic_stats['cubic']}")
    print(f"  B4 - Non-linear terms: {basic_stats['nonlinear']}")
    print()
    
    # Test analyze_instance function
    print("Testing analyze_instance function:")
    instance_results = analyze_instance(c, f, verbose=True)
    print()
    
    # Test batch analysis
    print("Testing batch analysis:")
    instances = [(c, f)]
    instance_names = ["Example_Instance"]
    batch_results = batch_analyze_instances(instances, instance_names)
    print("Batch Analysis Results:")
    print(batch_results.to_string(index=False))
    print()
    
    return basic_stats, batch_results

def test_multiple_instances():
    """Test with multiple different instances"""
    
    print("Testing Multiple Instances")
    print("=" * 50)
    
    # Create a few different test instances
    instances = []
    names = []
    
    # Small instance
    f1 = np.array([100, 120, 80])
    c1 = np.array([
        [10, 12, 8],
        [11, 9, 15]
    ])
    instances.append((c1, f1))
    names.append("Small_Instance")
    
    # Medium instance
    f2 = np.array([200, 250, 180, 300, 220])
    c2 = np.array([
        [20, 25, 18, 30, 22],
        [22, 20, 25, 28, 24],
        [18, 22, 16, 26, 20],
        [25, 28, 22, 20, 26]
    ])
    instances.append((c2, f2))
    names.append("Medium_Instance")
    
    # Original example
    f3 = np.array([1000, 1200, 800, 1500, 900, 1100, 1300])
    c3 = np.array([
        [50, 60, 40, 70, 45, 55, 65],
        [45, 55, 35, 65, 40, 50, 60],
        [60, 70, 50, 80, 55, 65, 75],
        [40, 50, 30, 60, 35, 45, 55],
        [55, 65, 45, 75, 50, 60, 70]
    ])
    instances.append((c3, f3))
    names.append("Large_Instance")
    
    # Batch analyze all instances
    results = batch_analyze_instances(instances, names)
    
    print("Multi-Instance Analysis Results:")
    print(results.to_string(index=False))
    print()
    
    # Print summary statistics
    print("Summary Statistics:")
    numeric_cols = ['linear', 'quadratic', 'cubic', 'nonlinear']
    summary = results[numeric_cols].describe()
    print(summary.round(2))
    
    return results

if __name__ == "__main__":
    try:
        # Test basic functionality
        basic_stats, batch_results = test_basic_analysis()
        
        # Test multiple instances
        multi_results = test_multiple_instances()
        
        print("\n" + "=" * 50)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 