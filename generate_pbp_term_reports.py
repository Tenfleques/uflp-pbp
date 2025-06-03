#!/usr/bin/env python3
"""
Generate PBP term reports for different degrees.

This script generates CSV reports for:
- B1.linear-terms.csv: Linear terms (degree 1)
- B2.quadratic-terms.csv: Quadratic terms (degree 2)
- B3.cubic-terms.csv: Cubic terms (degree 3)
- B4.non-linear.csv: Non-linear terms grouped by degree (2 to m//2)
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
ROOT_DIR = os.path.dirname(__file__)

from pbp.generate import cap_to_pbp_df, count_linear_terms, count_quadratic_terms, count_cubic_terms, count_non_linear_terms
from cap.cap_matrix_reader import read_cap_matrix

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_instances() -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """Load all UFLP instances from the data directory."""
    instances = []
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    # List of directories to search for instances
    instance_dirs = [
        'cap',
    ]
    
    for instance_dir in instance_dirs:
        dir_path = os.path.join(data_dir, instance_dir)
        if not os.path.exists(dir_path):
            continue
            
        logger.info(f"Searching for instances in {instance_dir}")
        for filename in os.listdir(dir_path):
            if "uncapopt" in filename:
                continue

            if filename.endswith('.txt'):
                filepath = os.path.join(dir_path, filename)
                try:
                    instance_name = f"{filename.replace('.txt', '')}"
                    instances.append((instance_name))
                    logger.info(f"Loaded instance: {instance_name}")
                except Exception as e:
                    logger.error(f"Error loading {filepath}: {str(e)}")
    
    return instances

def analyze_terms(pbp_df: pd.DataFrame, max_degree: int) -> Dict:
    """Analyze terms in a PBP DataFrame."""
    # Get basic term counts
    linear_count = count_linear_terms(pbp_df)
    quadratic_count = count_quadratic_terms(pbp_df)
    cubic_count = count_cubic_terms(pbp_df)
    
    # Get non-linear term counts for each degree
    nonlinear_counts = {}
    for degree in range(2, max_degree + 1):
        nonlinear_counts[degree] = count_non_linear_terms(pbp_df, degree)
    
    return {
        'linear_terms': linear_count,
        'quadratic_terms': quadratic_count,
        'cubic_terms': cubic_count,
        'nonlinear_counts': nonlinear_counts
    }

def generate_reports():
    """Generate all PBP term reports."""
    instances = load_instances()
    logger.info(f"Loaded {len(instances)} instances")
    
    # Initialize results DataFrames
    linear_results = []
    quadratic_results = []
    cubic_results = []
    nonlinear_results = []

    for name in instances:
        f = os.path.join(ROOT_DIR, "data/cap/{}.txt".format(name))
        logger.info(f"Processing instance: {name}")
        m, n, f, c = read_cap_matrix(f)
    
        # break
        try:
            # Generate PBP
            pbp_df = cap_to_pbp_df(c, f, verbose=False)
            # print(pbp_df)
            # Get maximum degree (m//2)
            max_degree = c.shape[0] // 2

            # Analyze terms
            analysis = analyze_terms(pbp_df, max_degree)
            
            # Add to results
            linear_results.append({
                'instance': name,
                'linear_terms': analysis['linear_terms']
            })
            
            quadratic_results.append({
                'instance': name,
                'quadratic_terms': analysis['quadratic_terms']
            })
            
            cubic_results.append({
                'instance': name,
                'cubic_terms': analysis['cubic_terms']
            })
            
            # Add non-linear term counts by degree
            for degree, count in analysis['nonlinear_counts'].items():
                nonlinear_results.append({
                    'instance': name,
                    'degree': degree,
                    'term_count': count
                })
            
        except Exception as e:
            logger.error(f"Error processing instance {name}: {str(e)}")
            traceback.print_exc()
    
    
    # Convert to DataFrames and save
    pd.DataFrame(linear_results).to_csv('outputs/B1.linear-terms.csv', index=False)
    pd.DataFrame(quadratic_results).to_csv('outputs/B2.quadratic-terms.csv', index=False)
    pd.DataFrame(cubic_results).to_csv('outputs/B3.cubic-terms.csv', index=False)
    pd.DataFrame(nonlinear_results).to_csv('outputs/B4.non-linear.csv', index=False)
    
    logger.info("Reports generated successfully")

if __name__ == "__main__":
    generate_reports() 