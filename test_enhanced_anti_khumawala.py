#!/usr/bin/env python3
"""
Test Enhanced AntiKhumawala Rules

Tests the fully implemented LBA and SBA rules with recursive Khumawala application.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import unittest
import logging
import numpy as np
from src.khumawala.branching import AntiKhumawalaRules
from src.khumawala.rules import KhumawalaRules

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestEnhancedAntiKhumawalaRules(unittest.TestCase):
    """Test the enhanced AntiKhumawala rule implementations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.anti_khumawala = AntiKhumawalaRules()
        
        # Create test instances
        self.simple_instance = {
            'name': 'Simple_LBA_Test',
            'facilities': [0, 1, 2],
            'customers': [0, 1, 2],
            'facility_costs': [10, 15, 8],
            'assignment_costs': [
                [3, 7, 4],
                [6, 2, 5], 
                [4, 6, 3]
            ]
        }
        
        self.complex_instance = {
            'name': 'Complex_SBA_Test',
            'facilities': [0, 1, 2, 3],
            'customers': [0, 1, 2, 3, 4],
            'facility_costs': [12, 18, 10, 20],
            'assignment_costs': [
                [2, 8, 5, 9],
                [7, 3, 6, 4],
                [4, 9, 2, 7],
                [6, 5, 8, 3], 
                [3, 7, 4, 6]
            ]
        }
        
        # Create KhumawalaRules instances for testing
        c_simple = np.array(self.simple_instance['assignment_costs'])
        f_simple = np.array(self.simple_instance['facility_costs'])
        self.khumawala_simple = KhumawalaRules(c_simple, f_simple)
        
        c_complex = np.array(self.complex_instance['assignment_costs'])
        f_complex = np.array(self.complex_instance['facility_costs'])
        self.khumawala_complex = KhumawalaRules(c_complex, f_complex)
        
    def test_lba_rule_comprehensive(self):
        """Test LBA rule with comprehensive scenario"""
        logger.info("\n" + "="*60)
        logger.info("Testing Linear Branching A (LBA) Rule")
        logger.info("="*60)
        
        result = self.anti_khumawala.apply_lba_rule(
            self.simple_instance, 
            self.khumawala_simple
        )
        
        logger.info(f"LBA Result: {result}")
        
        # Verify rule was applied
        self.assertEqual(result['rule'], 'LBA')
        self.assertTrue(result['applied'])
        
        # Check if solved by Khumawala alone or requires branching
        if result.get('solved_by_khumawala', False):
            logger.info("✓ Simple instance solved by Khumawala rules alone!")
            self.assertIn('result', result)
        else:
            # Should have branching information
            self.assertIn('branching_facility', result)
            self.assertIn('facility_score', result)
            self.assertIn('subproblems', result)
            
            # Verify subproblems structure
            subproblems = result['subproblems']
            self.assertIn('closed', subproblems)
            self.assertIn('open', subproblems)
            
            logger.info(f"✓ LBA branching on facility {result['branching_facility']} "
                       f"(score: {result['facility_score']:.3f})")
            
        # Verify statistics were updated
        self.assertEqual(self.anti_khumawala.branching_counts['LBA'], 1)
        
    def test_sba_rule_comprehensive(self):
        """Test SBA rule with comprehensive scenario"""
        logger.info("\n" + "="*60)
        logger.info("Testing Systematic Branching A (SBA) Rule")
        logger.info("="*60)
        
        result = self.anti_khumawala.apply_sba_rule(
            self.complex_instance,
            self.khumawala_complex
        )
        
        logger.info(f"SBA Result: {result}")
        
        # Verify rule was applied
        self.assertEqual(result['rule'], 'SBA')
        self.assertTrue(result['applied'])
        
        # Check if solved by Khumawala alone or requires branching
        if result.get('solved_by_khumawala', False):
            logger.info("✓ Complex instance solved by Khumawala rules alone!")
            self.assertIn('result', result)
        else:
            # Should have branching information
            self.assertIn('branching_pair', result)
            self.assertIn('pair_score', result)
            self.assertIn('subproblems', result)
            self.assertIn('total_pairs_analyzed', result)
            
            # Verify subproblems structure
            subproblems = result['subproblems']
            self.assertIn('assign', subproblems)
            self.assertIn('prohibit', subproblems)
            
            branching_pair = result['branching_pair']
            logger.info(f"✓ SBA branching on facility-customer pair {branching_pair} "
                       f"(score: {result['pair_score']:.3f})")
            logger.info(f"✓ Analyzed {result['total_pairs_analyzed']} facility-customer pairs")
            
        # Verify statistics were updated
        self.assertEqual(self.anti_khumawala.branching_counts['SBA'], 1)
        
    def test_lba_with_no_undecided_facilities(self):
        """Test LBA behavior when all facilities are decided by Khumawala rules"""
        # Create instance where Khumawala rules solve everything
        trivial_instance = {
            'name': 'Trivial_Test',
            'facilities': [0, 1],
            'customers': [0, 1],
            'facility_costs': [1, 100],  # Facility 1 clearly dominated
            'assignment_costs': [
                [1, 10],
                [1, 10]
            ]
        }
        
        c = np.array(trivial_instance['assignment_costs'])
        f = np.array(trivial_instance['facility_costs'])
        khumawala = KhumawalaRules(c, f)
        
        result = self.anti_khumawala.apply_lba_rule(trivial_instance, khumawala)
        
        # Should be solved by Khumawala alone
        self.assertTrue(result.get('solved_by_khumawala', False))
        logger.info("✓ LBA correctly detected Khumawala-solved instance")
        
    def test_sba_with_no_undecided_facilities(self):
        """Test SBA behavior when all facilities are decided by Khumawala rules"""
        # Same trivial instance
        trivial_instance = {
            'name': 'Trivial_SBA_Test',
            'facilities': [0, 1],
            'customers': [0, 1],
            'facility_costs': [1, 100],
            'assignment_costs': [
                [1, 10],
                [1, 10]
            ]
        }
        
        c = np.array(trivial_instance['assignment_costs'])
        f = np.array(trivial_instance['facility_costs'])
        khumawala = KhumawalaRules(c, f)
        
        result = self.anti_khumawala.apply_sba_rule(trivial_instance, khumawala)
        
        # Should be solved by Khumawala alone
        self.assertTrue(result.get('solved_by_khumawala', False))
        logger.info("✓ SBA correctly detected Khumawala-solved instance")
        
    def test_linear_score_calculation(self):
        """Test the linear score calculation for LBA"""
        undecided = [0, 1, 2]
        scores = self.anti_khumawala._calculate_linear_scores(
            self.simple_instance, undecided
        )
        
        # Should have scores for all undecided facilities
        self.assertEqual(len(scores), 3)
        
        # Scores should be positive
        for facility, score in scores.items():
            self.assertGreater(score, 0)
            
        logger.info(f"✓ Linear scores calculated: {scores}")
        
    def test_systematic_score_calculation(self):
        """Test the systematic score calculation for SBA"""
        undecided = [0, 1, 2, 3]
        scores = self.anti_khumawala._calculate_systematic_scores(
            self.complex_instance, undecided
        )
        
        # Should have scores for facility-customer pairs
        self.assertGreater(len(scores), 0)
        
        # Check score structure
        for (facility, customer), score in scores.items():
            self.assertIn(facility, undecided)
            self.assertIn(customer, self.complex_instance['customers'])
            self.assertGreater(score, 0)
            
        logger.info(f"✓ Systematic scores calculated: {len(scores)} pairs")
        
    def test_combined_lba_sba_workflow(self):
        """Test combining LBA and SBA rules in sequence"""
        logger.info("\n" + "="*60)
        logger.info("Testing Combined LBA-SBA Workflow")
        logger.info("="*60)
        
        # Apply LBA first
        lba_result = self.anti_khumawala.apply_lba_rule(
            self.complex_instance,
            self.khumawala_complex
        )
        
        # Apply SBA second
        sba_result = self.anti_khumawala.apply_sba_rule(
            self.complex_instance,
            self.khumawala_complex  # Fresh instance for independent test
        )
        
        # Both should have been applied
        self.assertEqual(self.anti_khumawala.branching_counts['LBA'], 1)
        self.assertEqual(self.anti_khumawala.branching_counts['SBA'], 1)
        
        # Get final statistics
        stats = self.anti_khumawala.get_statistics()
        self.assertEqual(stats['total_branchings'], 2)
        
        logger.info(f"✓ Combined workflow completed:")
        logger.info(f"  LBA applied: {lba_result['applied']}")
        logger.info(f"  SBA applied: {sba_result['applied']}")
        logger.info(f"  Total branchings: {stats['total_branchings']}")


def run_enhanced_tests():
    """Run the enhanced AntiKhumawala rule tests"""
    print("Testing Enhanced AntiKhumawala Rules (LBA & SBA)")
    print("="*60)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEnhancedAntiKhumawalaRules)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\n{'='*60}")
    print(f"ENHANCED ANTI-KHUMAWALA TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
            
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    success = len(result.failures) + len(result.errors) == 0
    
    if success:
        print("🎉 All enhanced AntiKhumawala tests passed!")
    else:
        print(f"⚠️  {len(result.failures) + len(result.errors)} test(s) failed")
    
    return success


if __name__ == "__main__":
    success = run_enhanced_tests()
    sys.exit(0 if success else 1) 