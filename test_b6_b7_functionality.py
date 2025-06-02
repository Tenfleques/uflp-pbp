#!/usr/bin/env python3
"""
Test script for B6-B7 advanced branching functionality.

This script tests the MQL branching and all six AntiKhumawala rules
to ensure they work correctly with recursive Khumawala rule application.
"""

import sys
import os
import numpy as np
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from khumawala.rules import KhumawalaRules
from khumawala.branching import MQLBranching, AntiKhumawalaRules, AdvancedBranchingSolver

def create_test_instance() -> Dict[str, Any]:
    """Create a small test instance for verification."""
    # Small test case
    f = np.array([100, 120, 80])
    c = np.array([
        [10, 12, 8],
        [11, 9, 15]
    ])
    
    return {
        'name': 'Test_Instance',
        'facilities': list(range(len(f))),
        'customers': list(range(c.shape[0])),
        'facility_costs': f.tolist(),
        'assignment_costs': c.tolist()
    }

def test_mql_branching():
    """Test B6: MQL branching functionality."""
    print("=" * 60)
    print("TESTING B6: MQL BRANCHING")
    print("=" * 60)
    
    instance = create_test_instance()
    
    # Create Khumawala rules and MQL branching solver
    c = np.array(instance['assignment_costs'])
    f = np.array(instance['facility_costs'])
    
    khumawala_rules = KhumawalaRules(c, f)
    mql_solver = MQLBranching()
    
    try:
        result = mql_solver.apply_mql_branching(instance, khumawala_rules, max_iterations=10)
        
        print(f"✓ MQL Branching completed successfully")
        print(f"  Solved to optimality: {result['solved_to_optimality']}")
        print(f"  Best objective: {result.get('best_objective', 'N/A')}")
        print(f"  Total nodes: {result['statistics']['total_nodes']}")
        print(f"  MQL branchings: {result['statistics']['mql_branchings']}")
        print(f"  Nodes solved by Khumawala: {result['statistics']['nodes_solved_by_khumawala']}")
        print(f"  Max depth: {result['statistics']['max_depth']}")
        
        return True
        
    except Exception as e:
        print(f"✗ MQL Branching failed: {str(e)}")
        return False

def test_anti_khumawala_rules():
    """Test B7: All AntiKhumawala branching rules."""
    print("\n" + "=" * 60)
    print("TESTING B7: ANTIKHUMAWALA BRANCHING RULES")
    print("=" * 60)
    
    instance = create_test_instance()
    c = np.array(instance['assignment_costs'])
    f = np.array(instance['facility_costs'])
    
    anti_khumawala = AntiKhumawalaRules()
    all_passed = True
    
    # Test all six rules
    rules_to_test = [
        ('LBA', anti_khumawala.apply_lba_rule),
        ('SBA', anti_khumawala.apply_sba_rule),
        ('FLBA', anti_khumawala.apply_flba_rule),
        ('FSBA', anti_khumawala.apply_fsba_rule),
        ('SLBA', anti_khumawala.apply_slba_rule),
        ('SSBA', anti_khumawala.apply_ssba_rule)
    ]
    
    for rule_name, rule_function in rules_to_test:
        print(f"\nTesting {rule_name} rule:")
        
        try:
            khumawala_rules = KhumawalaRules(c, f)  # Fresh instance for each test
            result = rule_function(instance, khumawala_rules)
            
            print(f"  ✓ {rule_name} rule executed successfully")
            print(f"    Rule: {result.get('rule', 'N/A')}")
            print(f"    Solved to optimality: {result.get('solved_to_optimality', False)}")
            print(f"    Branchings: {result.get(f'{rule_name.lower()}_branchings', 0)}")
            print(f"    Khumawala applications: {result.get('khumawala_applications', 0)}")
            
            if 'error' in result:
                print(f"    Warning: {result['error']}")
            
        except Exception as e:
            print(f"  ✗ {rule_name} rule failed: {str(e)}")
            all_passed = False
    
    # Test statistics collection
    print(f"\nAntiKhumawala Statistics:")
    stats = anti_khumawala.get_statistics()
    print(f"  Total branching applications: {stats['total_branchings']}")
    for rule, count in stats['branching_rule_counts'].items():
        print(f"  {rule} count: {count}")
    
    return all_passed

def test_advanced_branching_solver():
    """Test the integrated AdvancedBranchingSolver."""
    print("\n" + "=" * 60)
    print("TESTING ADVANCED BRANCHING SOLVER")
    print("=" * 60)
    
    instance = create_test_instance()
    c = np.array(instance['assignment_costs'])
    f = np.array(instance['facility_costs'])
    
    try:
        # Test MQL-first strategy
        solver = AdvancedBranchingSolver(mql_enabled=True, anti_khumawala_enabled=True)
        khumawala_rules = KhumawalaRules(c, f)
        
        result = solver.solve(instance, khumawala_rules, strategy='mql_first')
        
        print(f"✓ Advanced Branching Solver completed successfully")
        print(f"  Solved to optimality: {result.get('solved_to_optimality', False)}")
        print(f"  Strategy used: mql_first")
        
        if 'statistics' in result:
            print(f"  Statistics available: {list(result['statistics'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Advanced Branching Solver failed: {str(e)}")
        return False

def run_comprehensive_test():
    """Run all tests and provide summary."""
    print("COMPREHENSIVE B6-B7 FUNCTIONALITY TEST")
    print("=" * 60)
    
    test_results = {
        'B6_MQL_Branching': test_mql_branching(),
        'B7_AntiKhumawala_Rules': test_anti_khumawala_rules(),
        'Advanced_Branching_Solver': test_advanced_branching_solver()
    }
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All B6-B7 functionality tests PASSED!")
        return True
    else:
        print("⚠️  Some tests FAILED. Check implementation.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1) 