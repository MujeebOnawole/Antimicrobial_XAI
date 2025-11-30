#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple
import textwrap


class ScaffoldPropertyAnalyzer:
    """Analyze physicochemical properties of scaffold categories"""
    
    PROPERTIES = {
        'MolecularWeight': Descriptors.MolWt,
        'LogP': Descriptors.MolLogP,
        'NumHDonors': Lipinski.NumHDonors,
        'NumHAcceptors': Lipinski.NumHAcceptors,
        'NumRotatableBonds': Lipinski.NumRotatableBonds,
        'TPSA': Descriptors.TPSA,
        'FractionCSP3': Lipinski.FractionCSP3,
        'NumAromaticRings': Lipinski.NumAromaticRings
    }

    def __init__(self, positive_csv: str, negative_csv: str, neutral_csv: str):
        self.dfs = {
            'positive': self._load_data(positive_csv, 'positive'),
            'negative': self._load_data(negative_csv, 'negative'),
            'neutral': self._load_data(neutral_csv, 'neutral')
        }
        self.combined_df = pd.concat(self.dfs.values())
        self.property_stats = {}
        self._check_scaffold_uniqueness()

    def _check_scaffold_uniqueness(self):
        """Check if scaffolds are unique across categories"""
        # Get sets of scaffolds for each category
        positive_scaffolds = set(self.dfs['positive']['scaffold_smiles'])
        negative_scaffolds = set(self.dfs['negative']['scaffold_smiles'])
        neutral_scaffolds = set(self.dfs['neutral']['scaffold_smiles'])
        
        # Check for overlaps
        pos_neg_overlap = positive_scaffolds.intersection(negative_scaffolds)
        pos_neu_overlap = positive_scaffolds.intersection(neutral_scaffolds)
        neg_neu_overlap = negative_scaffolds.intersection(neutral_scaffolds)
        
        # Print detailed report
        print("\nScaffold Uniqueness Analysis:")
        print("-----------------------------")
        print(f"Total scaffolds in positive category: {len(positive_scaffolds)}")
        print(f"Total scaffolds in negative category: {len(negative_scaffolds)}")
        print(f"Total scaffolds in neutral category: {len(neutral_scaffolds)}")
        print("\nOverlap Analysis:")
        print(f"Scaffolds appearing in both positive and negative: {len(pos_neg_overlap)}")
        print(f"Scaffolds appearing in both positive and neutral: {len(pos_neu_overlap)}")
        print(f"Scaffolds appearing in both negative and neutral: {len(neg_neu_overlap)}")
        
        # If any overlaps exist, print warning and the actual overlapping scaffolds
        if pos_neg_overlap or pos_neu_overlap or neg_neu_overlap:
            print("\nWARNING: Some scaffolds appear in multiple categories!")
            if pos_neg_overlap:
                print("\nScaffolds in both positive and negative:")
                for scaffold in pos_neg_overlap:
                    print(f"- {scaffold}")
            if pos_neu_overlap:
                print("\nScaffolds in both positive and neutral:")
                for scaffold in pos_neu_overlap:
                    print(f"- {scaffold}")
            if neg_neu_overlap:
                print("\nScaffolds in both negative and neutral:")
                for scaffold in neg_neu_overlap:
                    print(f"- {scaffold}")
            print("\nConsider reviewing these overlapping scaffolds before proceeding with analysis.")
        else:
            print("\nVERIFIED: All scaffolds are unique to their respective categories.")

    def _load_data(self, csv_path: str, category: str) -> pd.DataFrame:
        """Load scaffold data and add category label"""
        df = pd.read_csv(csv_path)
        df['category'] = category
        return df[['scaffold_smiles', 'category']]

    def _calculate_properties(self, smiles: str) -> Dict[str, float]:
        """Calculate physicochemical properties for a scaffold"""
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return {}
            
        return {
            prop_name: prop_func(mol)
            for prop_name, prop_func in self.PROPERTIES.items()
        }

    def analyze_properties(self):
        """Calculate and analyze properties for all scaffolds"""
        # Calculate properties
        property_data = []
        for category, df in self.dfs.items():
            for _, row in df.iterrows():
                props = self._calculate_properties(row['scaffold_smiles'])
                if props:
                    props['category'] = category
                    property_data.append(props)
        
        self.property_df = pd.DataFrame(property_data)
        
        # Perform statistical analysis
        self._perform_statistical_tests()
        



    def _perform_statistical_tests(self):
        """Enhanced statistical tests with effect sizes"""
        self.property_stats = {}
        
        for prop in self.PROPERTIES.keys():
            # Separate pairwise comparisons
            pos_data = self.property_df[self.property_df['category'] == 'positive'][prop]
            neg_data = self.property_df[self.property_df['category'] == 'negative'][prop]
            neu_data = self.property_df[self.property_df['category'] == 'neutral'][prop]
            
            # Calculate Cohen's d effect size for meaningful comparisons
            d_pos_neg = (pos_data.mean() - neg_data.mean()) / np.sqrt((pos_data.var() + neg_data.var()) / 2)
            d_pos_neu = (pos_data.mean() - neu_data.mean()) / np.sqrt((pos_data.var() + neu_data.var()) / 2)
            d_neg_neu = (neg_data.mean() - neu_data.mean()) / np.sqrt((neg_data.var() + neu_data.var()) / 2)
            
            # Original Kruskal-Wallis test
            stat, p = stats.kruskal(pos_data, neg_data, neu_data)
            
            # Store both significance and effect size
            self.property_stats[prop] = {
                'statistic': stat,
                'p_value': p,
                'test_type': 'Kruskal-Wallis',  # Added this line
                'significant': p < 0.05,
                'effect_size_pos_neg': d_pos_neg,
                'effect_size_pos_neu': d_pos_neu,
                'effect_size_neg_neu': d_neg_neu,
                'practical_significance': abs(d_pos_neg) > 0.5
            }
   
    def plot_properties(self, output_dir: Path):
        """Generate box plots for all properties"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for prop in self.PROPERTIES.keys():
            plt.figure(figsize=(10, 6))
            sns.boxplot(
                data=self.property_df,
                x='category',
                y=prop,
                hue='category',  # Add this line
                order=['positive', 'negative', 'neutral'],
                palette={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'},
                legend=False     # Add this line
            )
            plt.title(f'Distribution of {prop} by Scaffold Category')
            plt.savefig(output_dir / f'{prop}_distribution.png')
            plt.close()


    def plot_simplified_properties(self, output_dir: Path):
        """Generate more intuitive property visualizations"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for prop in self.PROPERTIES.keys():
            plt.figure(figsize=(10, 6))
            
            # Use simple bar plot with error bars
            summary = self.property_df.groupby('category')[prop].agg(['mean', 'std'])
            
            # Plot bars
            bars = plt.bar(range(len(summary)), summary['mean'], 
                        color=['#2ecc71', '#e74c3c', '#95a5a6'],
                        yerr=summary['std'],
                        capsize=5)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom')
            
            # Customize appearance
            plt.xticks(range(len(summary)), ['Positive', 'Negative', 'Neutral'])
            plt.title(f'Average {prop} by Category\nError bars show standard deviation')
            
            # Add a brief interpretation
            interpretation = self._get_interpretation(prop)
            plt.figtext(0.02, -0.1, interpretation, wrap=True, 
                    horizontalalignment='left', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{prop}_simple_distribution.png',
                    bbox_inches='tight', dpi=300)
            plt.close()

    def _get_interpretation(self, prop):
        """Get simple interpretation text for each property"""
        interpretations = {
            'MolecularWeight': 'Higher values mean larger molecules',
            'LogP': 'Higher values mean more fat-soluble/less water-soluble',
            'NumHDonors': 'More H-donors mean stronger hydrogen bonding ability',
            'NumHAcceptors': 'More H-acceptors mean stronger binding potential',
            'NumRotatableBonds': 'More rotatable bonds mean more flexibility',
            'TPSA': 'Higher values suggest better water solubility',
            'FractionCSP3': 'Higher values indicate more 3D-like structure',
            'NumAromaticRings': 'More rings suggest flatter, more rigid structure'
        }
        return interpretations.get(prop, '')
    


    def plot_property_trends(self, output_dir: Path):
        """Create a simple trend summary"""
        output_dir.mkdir(parents=True, exist_ok=True)
        # Calculate average values relative to neutral
        trends = {}
        for prop in self.PROPERTIES.keys():
            neutral_mean = self.property_df[self.property_df['category'] == 'neutral'][prop].mean()
            pos_rel = self.property_df[self.property_df['category'] == 'positive'][prop].mean() / neutral_mean
            neg_rel = self.property_df[self.property_df['category'] == 'negative'][prop].mean() / neutral_mean
            
            trends[prop] = {'positive': pos_rel, 'negative': neg_rel}
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        x = range(len(self.PROPERTIES))
        
        plt.plot(x, [trends[p]['positive'] for p in self.PROPERTIES], 
                'g-', label='Positive vs Neutral', marker='o')
        plt.plot(x, [trends[p]['negative'] for p in self.PROPERTIES], 
                'r-', label='Negative vs Neutral', marker='o')
        plt.axhline(y=1, color='gray', linestyle='--', label='Neutral baseline')
        
        plt.xticks(x, list(self.PROPERTIES.keys()), rotation=45)
        plt.ylabel('Relative to Neutral Compounds')
        plt.title('Property Trends Relative to Neutral Compounds\nValues > 1 mean higher than neutral, < 1 mean lower')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'property_trends.png')
        plt.close()

    def generate_report(self, output_dir: Path):
        """Generate statistical summary report"""
        report = [
            "Physicochemical Property Analysis Report",
            "=========================================",
            "\nSignificant Differences Between Categories (p < 0.05):"
        ]
        
        # List significant properties
        sig_props = [
            prop for prop, stats in self.property_stats.items()
            if stats['significant']
        ]
        report.append(f"- {len(sig_props)} significant properties found: {', '.join(sig_props)}")
        
        # Add detailed stats table
        report.append("\n\nDetailed Statistics:")
        stats_table = []
        for prop, stats in self.property_stats.items():
            stats_table.append([
                prop,
                stats['test_type'],
                f"{stats['statistic']:.2f}",
                f"{stats['p_value']:.4f}",
                "Yes" if stats['significant'] else "No"
            ])
            
        report.append(
            pd.DataFrame(
                stats_table,
                columns=['Property', 'Test', 'Statistic', 'p-value', 'Significant']
            ).to_string(index=False)
        )
        
        # Add interpretation
        report.append(textwrap.dedent("""
            
            Interpretation Guide:
            --------------------
            1. Significant properties (p < 0.05) suggest structural differences between categories
            2. Positive/negative comparisons indicate property trends associated with activity
            3. Check individual property plots for distribution patterns
            """))
        
        with open(output_dir / 'property_analysis_report.txt', 'w') as f:
            f.write('\n'.join(report))

    def generate_enhanced_report(self, output_dir: Path):
        """Generate more informative statistical summary"""
        report = [
            "Enhanced Physicochemical Property Analysis Report",
            "=============================================",
            "\nPractical Differences Between Categories:",
        ]
        
        # Report meaningful differences
        meaningful_props = []
        for prop, stats in self.property_stats.items():
            if stats['practical_significance']:
                meaningful_props.append(
                    f"{prop} (effect size between pos/neg: {stats['effect_size_pos_neg']:.2f})"
                )
        
        report.append(f"\nProperties with meaningful differences (effect size > 0.5):")
        if meaningful_props:
            report.extend([f"- {prop}" for prop in meaningful_props])
        else:
            report.append("No properties show practically significant differences")
        
        # Add detailed comparison table
        report.append("\n\nDetailed Property Comparisons:")
        stats_table = []
        for prop, stats in self.property_stats.items():
            stats_table.append([
                prop,
                f"{stats['effect_size_pos_neg']:.2f}",
                f"{stats['effect_size_pos_neu']:.2f}",
                f"{stats['effect_size_neg_neu']:.2f}",
                "Yes" if stats['practical_significance'] else "No"
            ])
        
        report.append(
            pd.DataFrame(
                stats_table,
                columns=['Property', 'Pos-Neg Effect', 'Pos-Neutral Effect', 
                        'Neg-Neutral Effect', 'Practically Significant']
            ).to_string(index=False)
        )
        
        # Add interpretation guide
        report.append(textwrap.dedent("""
            
            Effect Size Interpretation:
            -------------------------
            < 0.2: Negligible difference
            0.2-0.5: Small difference
            0.5-0.8: Medium difference
            > 0.8: Large difference
            
            Note: Statistical significance (p-value) alone doesn't indicate practical importance.
            Effect sizes show the magnitude of differences between groups.
            """))
        
        with open(output_dir / 'enhanced_property_analysis.txt', 'w') as f:
            f.write('\n'.join(report))

def main():
    parser = argparse.ArgumentParser(
        description='Analyze physicochemical properties of scaffold categories',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=textwrap.dedent('''
            Example Usage:
            $ python property_analysis.py positive.csv negative.csv neutral.csv -o results
            
            Output Files:
            - simple_distributions/: PNG plots with clear interpretations
            - property_trends.png: Relative trends compared to neutral
            - property_analysis_report.txt: Statistical summary
            - enhanced_property_analysis.txt: Practical significance analysis
        ''')
    )
    parser.add_argument('positive', help='CSV file for positive scaffolds')
    parser.add_argument('negative', help='CSV file for negative scaffolds')
    parser.add_argument('neutral', help='CSV file for neutral scaffolds')
    parser.add_argument('-o', '--output', default='property_analysis',
                      help='Output directory (default: property_analysis)')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = ScaffoldPropertyAnalyzer(args.positive, args.negative, args.neutral)
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Perform analysis
        print("Calculating scaffold properties...")
        analyzer.analyze_properties()
        
        # Generate visualizations
        print("Generating simplified visualizations...")
        analyzer.plot_simplified_properties(output_dir / 'simple_distributions')
        
        print("Generating trend summary...")
        analyzer.plot_property_trends(output_dir)
        
        print("Generating statistical reports...")
        analyzer.generate_report(output_dir)  # Original statistical report
        analyzer.generate_enhanced_report(output_dir)  # New enhanced report
        
        print(f"\nAnalysis complete! Results saved to: {output_dir.resolve()}")
        print("\nKey files generated:")
        print("1. simple_distributions/*.png - Easy to interpret property distributions")
        print("2. property_trends.png - Shows relative differences from neutral compounds")
        print("3. property_analysis_report.txt - Traditional statistical analysis")
        print("4. enhanced_property_analysis.txt - Practical significance analysis")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
