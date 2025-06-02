#!/usr/bin/env python3

import pandas as pd
import numpy as np

def analyze_cap_results():
    """Analyze cap results to understand Khumawala performance"""
    
    df = pd.read_csv('cap_results_full.csv')
    
    print('ANALYSIS OF CAP INSTANCES RESULTS:')
    print('='*50)
    
    print('\nProblem sizes:')
    size_counts = df.groupby('num_facilities').size()
    print(size_counts)
    
    print('\nKhumawala reduction rates:')
    print(f'Mean reduction: {df["B5_problem_reduction_percentage"].mean():.1f}%')
    print(f'Max reduction: {df["B5_problem_reduction_percentage"].max():.1f}%')
    print(f'Min reduction: {df["B5_problem_reduction_percentage"].min():.1f}%')
    
    print('\nVariables fixed vs remaining:')
    print(f'Mean variables fixed: {df["B5_total_variables_fixed"].mean():.1f}')
    print(f'Mean variables remaining: {df["num_facilities"].mean() - df["B5_total_variables_fixed"].mean():.1f}')
    
    print('\nInstances with highest reduction:')
    top_reduced = df.nlargest(5, 'B5_problem_reduction_percentage')[['instance_name', 'num_facilities', 'B5_total_variables_fixed', 'B5_problem_reduction_percentage']]
    print(top_reduced.to_string(index=False))
    
    print('\nInstances closest to being solved:')
    df['variables_remaining'] = df['num_facilities'] - df['B5_total_variables_fixed']
    closest = df.nsmallest(10, 'variables_remaining')[['instance_name', 'num_facilities', 'B5_total_variables_fixed', 'variables_remaining', 'B5_problem_reduction_percentage']]
    print(closest.to_string(index=False))
    
    print('\nRule effectiveness:')
    print(f'Mean Rule 1 variables fixed: {df["B5_variables_fixed_rule1"].mean():.1f}')
    print(f'Mean Rule 2 variables fixed: {df["B5_variables_fixed_rule2"].mean():.1f}')
    
    # Check if any instance was actually solved
    fully_solved = df[df['variables_remaining'] == 0]
    print(f'\nInstances solved to optimality: {len(fully_solved)}')
    
    if len(fully_solved) > 0:
        print('Solved instances:')
        print(fully_solved[['instance_name', 'num_facilities', 'B5_total_variables_fixed']].to_string(index=False))
    else:
        print('No instances were solved to optimality by Khumawala rules alone.')
        print('This is expected for real-world instances - they require branch-and-bound.')

if __name__ == "__main__":
    analyze_cap_results() 