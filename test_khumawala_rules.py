#!/usr/bin/env python3
"""
Test script for the Khumawala rules implementation.

This script tests the Khumawala rules (B5) functionality using example data.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pbp.generate import cap_to_pbp_df
from khumawala.analysis import TermAnalyzer, analyze_instance
from khumawala.rules import KhumawalaRules

def test_khumawala_rules_basic():
    """Test basic Khumawala rules functionality"""
    
    print("Testing Khumawala Rules Implementation")
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
    
    # Create Khumawala rules instance
    khumawala = KhumawalaRules(c, f)
    
    # Apply rules iteratively
    print("Applying Khumawala rules iteratively...")
    stats = khumawala.apply_iterative_khumawala_rules()
    
    # Print detailed report
    khumawala.print_status_report()
    
    return khumawala, stats

def test_khumawala_rules_with_analysis():
    """Test Khumawala rules combined with term analysis"""
    
    print("\nTesting Combined Analysis and Khumawala Rules")
    print("=" * 50)
    
    # Create test instances
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
    
    results = []
    
    for i, (c, f) in enumerate(instances):
        instance_name = names[i]
        print(f"\nAnalyzing {instance_name}...")
        
        # Basic term analysis
        basic_analysis = analyze_instance(c, f, verbose=False)
        
        # Apply Khumawala rules
        khumawala = KhumawalaRules(c, f)
        khumawala_stats = khumawala.apply_iterative_khumawala_rules()
        
        # Combine results
        result = {
            'instance': instance_name,
            'num_facilities': len(f),
            'num_customers': c.shape[0],
            **basic_analysis,
            **khumawala_stats
        }
        results.append(result)
        
        print(f"  Term Analysis - Linear: {basic_analysis['linear']}, "
              f"Quadratic: {basic_analysis['quadratic']}, "
              f"Cubic: {basic_analysis['cubic']}")
        print(f"  Khumawala - Fixed by Rule 1: {khumawala_stats['variables_fixed_rule1']}, "
              f"Fixed by Rule 2: {khumawala_stats['variables_fixed_rule2']}")
        print(f"  Solution Status: {'OPTIMAL' if khumawala_stats['solved_to_optimality'] else 'PARTIAL'}")
    
    return results

def test_comprehensive_example():
    """Test with a more comprehensive example designed to trigger Khumawala rules"""
    
    print("\nTesting Comprehensive Example")
    print("=" * 50)
    
    # Create an instance where Khumawala rules should be effective
    # Facility 0 is expensive and dominated
    # Facility 1 is cheap but far from some customers
    # Facility 2 is moderate cost and well-positioned
    f = np.array([1000, 100, 200])  # Facility 0 very expensive
    c = np.array([
        [10, 50, 20],  # Customer 0: prefers facility 0, but it's expensive
        [15, 5, 25],   # Customer 1: strongly prefers facility 1
        [12, 40, 15]   # Customer 2: moderate preference for facilities 0 and 2
    ])
    
    print("Comprehensive Example Data:")
    print(f"Fixed costs (f): {f}")
    print(f"Transport costs (c):")
    print(c)
    print()
    
    # Apply basic analysis
    basic_analysis = analyze_instance(c, f, verbose=True)
    
    print("\nApplying Khumawala Rules:")
    khumawala = KhumawalaRules(c, f)
    stats = khumawala.apply_iterative_khumawala_rules()
    
    # Print detailed report
    khumawala.print_status_report()
    
    # Check if we can get reduced problem
    reduced_c, reduced_f, mapping = khumawala.get_reduced_problem()
    print(f"\nReduced Problem:")
    print(f"  Original facilities: {len(f)}")
    print(f"  Remaining facilities: {len(reduced_f)}")
    if len(reduced_f) > 0:
        print(f"  Reduced fixed costs: {reduced_f}")
        print(f"  Index mapping: {mapping}")
    
    return khumawala, basic_analysis, stats

if __name__ == "__main__":
    try:
        # Test basic functionality
        khumawala1, stats1 = test_khumawala_rules_basic()
        
        # Test combined analysis
        combined_results = test_khumawala_rules_with_analysis()
        
        # Test comprehensive example
        khumawala3, analysis3, stats3 = test_comprehensive_example()
        
        print("\n" + "=" * 50)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        # Summary statistics
        print("\nSUMMARY OF KHUMAWALA RULE EFFECTIVENESS:")
        for result in combined_results:
            instance = result['instance']
            total_fixed = result['total_variables_fixed']
            facilities = result['num_facilities']
            optimal = result['solved_to_optimality']
            print(f"  {instance}: {total_fixed}/{facilities} variables fixed, "
                  f"Optimal: {'Yes' if optimal else 'No'}")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 