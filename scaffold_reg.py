#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw
import argparse
from pathlib import Path
import sys
import re
import warnings
from typing import Dict, List, Tuple, Optional
import textwrap

warnings.filterwarnings('ignore')

def analyze_scaffolds(
    df: pd.DataFrame, 
    decomp_type: str, 
    confidence_weights: Optional[Dict[str, float]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyze scaffolds' contribution to activity and prediction reliability.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing molecular data
    decomp_type : str
        Type of decomposition ('brics' or 'murcko')
    confidence_weights : Dict[str, float], optional
        Custom weights for confidence levels
        
    Returns:
    --------
    pd.DataFrame, Dict
        DataFrame with scaffold statistics and dictionary with additional analysis
    """
    prefix = decomp_type.lower()
    
    # Collect substructure numbers and pair columns
    substructure_numbers = set()
    for col in df.columns:
        match = re.match(rf'{prefix}_substructure_(\d+)_', col, re.IGNORECASE)
        if match:
            substructure_numbers.add(int(match.group(1)))
    
    # Pair smiles and attribution columns by substructure number
    smiles_cols, attribution_cols = [], []
    for num in sorted(substructure_numbers):
        smiles_col = f'{prefix}_substructure_{num}_smiles'
        attr_col = f'{prefix}_substructure_{num}_attribution'
        if smiles_col in df.columns and attr_col in df.columns:
            smiles_cols.append(smiles_col)
            attribution_cols.append(attr_col)
        else:
            warnings.warn(f"Missing columns for substructure {num}", UserWarning)

    scaffold_stats = []
    for smiles_col, attr_col in zip(smiles_cols, attribution_cols):
        position = re.search(r'_(\d+)_', smiles_col).group(1)
        mask = df[smiles_col].notna()
        scaffolds = df[mask]
        
        for scaffold in scaffolds[smiles_col].unique():
            scaffold_data = scaffolds[scaffolds[smiles_col] == scaffold]
            
            stats = {
                'scaffold_smiles': scaffold,
                'position': position,
                'frequency': len(scaffold_data),
                'mean_attribution': scaffold_data[attr_col].mean(),
                'std_attribution': scaffold_data[attr_col].std(),
                'mean_fold_diff': scaffold_data['fold_diff'].mean(),
                'pct_A': (scaffold_data['confidence_level'] == 'A').mean(),
                'pct_B': (scaffold_data['confidence_level'] == 'B').mean(),
                'pct_C': (scaffold_data['confidence_level'] == 'C').mean(),
                'pct_D': (scaffold_data['confidence_level'] == 'D').mean(),
                'avg_abs_error': scaffold_data['abs_error'].mean()
            }
            scaffold_stats.append(stats)
    
    stats_df = pd.DataFrame(scaffold_stats)
    
    # Handle confidence weights
    default_weights = {'A': 1.0, 'B': 0.7, 'C': 0.3, 'D': 0.1}
    weights = confidence_weights or default_weights
    
    # Calculate importance score
    weighted_pcts = sum(
        stats_df[f'pct_{level}'] * weights.get(level, 0.0) 
        for level in ['A', 'B', 'C', 'D']
    )
    stats_df['importance_score'] = stats_df['mean_attribution'] * weighted_pcts
    
    # Identify significant contributors
    high_conf_mask = (stats_df['pct_A'] + stats_df['pct_B']) > 0.5
    pos_contrib_mask = stats_df['mean_attribution'] > 0
    significant_scaffolds = stats_df[high_conf_mask & pos_contrib_mask]
    
    analysis_results = {
        'total_scaffolds': len(stats_df),
        'significant_scaffolds': len(significant_scaffolds),
        'avg_fold_diff': df['fold_diff'].mean(),
        'confidence_distribution': df['confidence_level'].value_counts().to_dict(),
        'confidence_weights': weights
    }
    
    return stats_df, analysis_results

def plot_results(stats_df: pd.DataFrame, output_dir: Path):
    """Create visualizations of the analysis results."""
    # Attribution vs Frequency
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=stats_df,
        x='frequency',
        y='mean_attribution',
        hue='importance_score',
        palette='viridis',
        size='avg_abs_error',
        sizes=(20, 200)
    )
    plt.title('Scaffold Frequency vs Attribution')
    plt.savefig(output_dir / 'attribution_vs_frequency.png')
    plt.close()
    
    # Confidence Level Distribution
    plt.figure(figsize=(10, 6))
    conf_data = stats_df[['pct_A', 'pct_B', 'pct_C', 'pct_D']].mean()
    conf_data.plot(kind='bar', color=['#2ecc71', '#f1c40f', '#e74c3c', '#95a5a6'])
    plt.title('Average Confidence Level Distribution')
    plt.xticks(rotation=0)
    plt.ylabel('Percentage')
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_distribution.png')
    plt.close()

def save_results(stats_df: pd.DataFrame, analysis_results: Dict, output_dir: Path):
    """Save analysis results to files."""
    stats_df.sort_values('importance_score', ascending=False).to_csv(
        output_dir / 'scaffold_statistics.csv', index=False
    )
    
    with open(output_dir / 'analysis_summary.txt', 'w') as f:
        f.write(textwrap.dedent(f"""
            Scaffold Analysis Summary
            ========================
            
            Total scaffolds analyzed: {analysis_results['total_scaffolds']}
            Significant scaffolds: {analysis_results['significant_scaffolds']}
            Average fold difference: {analysis_results['avg_fold_diff']:.2f}
            
            Confidence Weights:
                A: {analysis_results['confidence_weights'].get('A', 0.0):.1f}
                B: {analysis_results['confidence_weights'].get('B', 0.0):.1f}
                C: {analysis_results['confidence_weights'].get('C', 0.0):.1f}
                D: {analysis_results['confidence_weights'].get('D', 0.0):.1f}
            
            Confidence Level Distribution:
        """))
        for level, count in analysis_results['confidence_distribution'].items():
            f.write(f"  {level}: {count}\n")

def main():
    """Main function to run scaffold analysis from command line."""
    parser = argparse.ArgumentParser(
        description='Analyze scaffold contributions to antimicrobial activity',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=textwrap.dedent('''
            Example Usage:
            python scaffold_analysis.py data.csv brics --weights A:1.0,B:0.7,C:0.3,D:0.1
        ''')
    )
    parser.add_argument('input', type=str, help='Path to input CSV file')
    parser.add_argument('type', choices=['brics', 'murcko'], help='Scaffold decomposition type')
    parser.add_argument('--weights', type=str, help='Custom confidence weights (e.g., A:1.0,B:0.7)')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    # Process confidence weights
    confidence_weights = {}
    if args.weights:
        try:
            confidence_weights = {
                k: float(v) for pair in args.weights.split(',') 
                for k, v in [pair.split(':')]
            }
        except:
            raise ValueError("Invalid weights format. Use 'A:1.0,B:0.7' format")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Load and preprocess data
        df = pd.read_csv(args.input)
        
        # Validate required columns
        required = {'logMIC', 'pred_logMIC'}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")
        
        # Calculate derived columns
        df['abs_error'] = np.abs(df['logMIC'] - df['pred_logMIC'])
        df['fold_diff'] = 10 ** df['abs_error']
        
        if 'confidence_level' not in df.columns:
            bins = [0, 1.5, 3, 6, np.inf]
            labels = ['A', 'B', 'C', 'D']
            df['confidence_level'] = pd.cut(df['fold_diff'], bins=bins, labels=labels, right=False)
        
        # Perform analysis
        stats_df, analysis_results = analyze_scaffolds(
            df, args.type, confidence_weights
        )
        
        # Generate output
        plot_results(stats_df, output_dir)
        save_results(stats_df, analysis_results, output_dir)
        
        print(f"Analysis completed successfully. Results saved to {output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()