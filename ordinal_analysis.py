#python confidence_level.py input.csv --output_dir output
import os
import argparse
from typing import Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class MolecularConfidenceAnalyzer:
    def __init__(self, output_dir: str = "confidence_analysis_results"):
        """
        Initialize the analyzer with confidence thresholds and output directory.
        Also sets global matplotlib parameters for publication quality.
        """
        self.fold_thresholds = {
            'A': 1.5,  # Within 1.5-fold
            'B': 3.0,  # Within 3-fold 
            'C': 10.0  # Within 10-fold
        }
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set global matplotlib parameters for publication-quality figures
        plt.rcParams.update({
            'font.size': 16,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'legend.fontsize': 14,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14
        })

    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load and validate the molecular data CSV."""
        required_columns = ['COMPOUND_ID', 'logMIC', 'pred_logMIC', 'prediction_std']
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        return df
    
    def assign_confidence_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign confidence levels based on fold difference between predicted and actual values."""
        df = df.copy()
        
        # Calculate absolute fold difference
        df['fold_diff'] = np.abs(df['pred_logMIC'] - df['logMIC'])
        
        # Assign confidence levels using thresholds for A, B, and C. Anything above is D.
        conditions = [
            (df['fold_diff'] <= self.fold_thresholds['A']),
            (df['fold_diff'] <= self.fold_thresholds['B']),
            (df['fold_diff'] <= self.fold_thresholds['C'])
        ]
        choices = ['A', 'B', 'C']
        df['confidence_level'] = np.select(conditions, choices, default='D')
        
        return df

    def calculate_metrics(self, true_vals: np.ndarray, pred_vals: np.ndarray) -> Dict:
        """Calculate regression metrics for provided true and predicted values."""
        if len(true_vals) == 0 or len(pred_vals) == 0:
            return {
                'Total Samples': 0,
                'Mean (True)': np.nan,
                'Mean (Predicted)': np.nan,
                'Std Dev (True)': np.nan,
                'Std Dev (Predicted)': np.nan,
                'MAE': np.nan,
                'MSE': np.nan,
                'RMSE': np.nan,
                'R²': np.nan,
                'Correlation': np.nan,
                'MAPE (%)': np.nan
            }
            
        total_samples = len(true_vals)
        mean_true = np.mean(true_vals)
        mean_pred = np.mean(pred_vals)
        std_true = np.std(true_vals)
        std_pred = np.std(pred_vals)
        
        mae = mean_absolute_error(true_vals, pred_vals)
        mse = mean_squared_error(true_vals, pred_vals)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_vals, pred_vals)
        
        try:
            correlation = np.corrcoef(true_vals, pred_vals)[0, 1]
        except Exception:
            correlation = np.nan
            
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((true_vals - pred_vals) / true_vals)) * 100
            if np.isinf(mape) or np.isnan(mape):
                mape = np.nan
        
        return {
            'Total Samples': total_samples,
            'Mean (True)': mean_true,
            'Mean (Predicted)': mean_pred,
            'Std Dev (True)': std_true,
            'Std Dev (Predicted)': std_pred,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2,
            'Correlation': correlation,
            'MAPE (%)': mape
        }
    
    def plot_individual_confidence_levels(self, df: pd.DataFrame):
        """
        For each confidence level present in the data, create a separate plot.
        Each plot shows the predicted versus actual values with a perfect prediction line
        and annotated regression metrics.
        """
        confidence_colors = {
            'A': '#2ca02c',  # green
            'B': '#1f77b4',  # blue
            'C': '#ff7f0e',  # orange
            'D': '#d62728'   # red
        }
        
        # Use the overall min/max from the data for consistent axis limits.
        min_val = min(df['logMIC'].min(), df['pred_logMIC'].min())
        max_val = max(df['logMIC'].max(), df['pred_logMIC'].max())
        padding = (max_val - min_val) * 0.05
        
        # Loop through each unique confidence level and create a plot
        for level in sorted(df['confidence_level'].unique()):
            level_data = df[df['confidence_level'] == level]
            if level_data.empty:
                continue

            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Scatter plot of predictions vs. actual
            ax.scatter(
                level_data['pred_logMIC'],
                level_data['logMIC'],
                alpha=0.7,
                color=confidence_colors.get(level, 'gray'),
                s=50,
                label=f'n = {len(level_data)}'
            )
            
            # Plot the perfect prediction line
            ax.plot(
                [min_val - padding, max_val + padding],
                [min_val - padding, max_val + padding],
                linestyle='--',
                color='black',
                lw=2,
                label='Perfect Prediction'
            )
            
            # Calculate and display metrics
            metrics = self.calculate_metrics(
                level_data['logMIC'].values,
                level_data['pred_logMIC'].values
            )
            metrics_text = (
                f"R² = {metrics['R²']:.3f}\n"
                f"RMSE = {metrics['RMSE']:.3f}\n"
                f"MAE = {metrics['MAE']:.3f}"
            )
            ax.text(
                0.05, 0.95, metrics_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=14
            )
            
            # Set labels and title
            ax.set_title(f'Confidence Level {level} (n = {len(level_data)})', fontsize=18)
            ax.set_xlabel('Predicted logMIC', fontsize=16)
            ax.set_ylabel('Actual logMIC', fontsize=16)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlim(min_val - padding, max_val + padding)
            ax.set_ylim(min_val - padding, max_val + padding)
            ax.legend(fontsize=14)
            
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, f'confidence_level_{level}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved plot for confidence level {level} to {save_path}")

    def plot_overall_analysis(self, df: pd.DataFrame):
        """
        Create an overall scatter plot of predicted vs. actual values,
        with points colored by confidence level. The plot also includes a perfect
        prediction line and annotated overall regression metrics.
        """
        confidence_colors = {
            'A': '#2ca02c',
            'B': '#1f77b4',
            'C': '#ff7f0e',
            'D': '#d62728'
        }
        
        min_val = min(df['logMIC'].min(), df['pred_logMIC'].min())
        max_val = max(df['logMIC'].max(), df['pred_logMIC'].max())
        padding = (max_val - min_val) * 0.05
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot each confidence level separately
        for level in ['A', 'B', 'C', 'D']:
            level_data = df[df['confidence_level'] == level]
            if not level_data.empty:
                ax.scatter(
                    level_data['pred_logMIC'],
                    level_data['logMIC'],
                    alpha=0.7,
                    color=confidence_colors[level],
                    s=50,
                    label=f'Level {level} (n = {len(level_data)})'
                )
        
        # Plot the perfect prediction line
        ax.plot(
            [min_val - padding, max_val + padding],
            [min_val - padding, max_val + padding],
            linestyle='--',
            color='black',
            lw=2,
            label='Perfect Prediction'
        )
        
        # Calculate overall metrics and annotate the plot (top left)
        metrics = self.calculate_metrics(
            df['logMIC'].values,
            df['pred_logMIC'].values
        )
        metrics_text = (
            f"Overall Metrics:\n"
            f"Total Samples: {metrics['Total Samples']}\n"
            f"R² = {metrics['R²']:.3f}\n"
            f"RMSE = {metrics['RMSE']:.3f}\n"
            f"MAE = {metrics['MAE']:.3f}\n"
            f"Correlation = {metrics['Correlation']:.3f}"
        )
        ax.text(
            0.05, 0.95, metrics_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=14
        )
        
        ax.set_title('Overall Prediction Performance', fontsize=18)
        ax.set_xlabel('Predicted logMIC', fontsize=16)
        ax.set_ylabel('Actual logMIC', fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(min_val - padding, max_val + padding)
        ax.set_ylim(min_val - padding, max_val + padding)
        
        # Move the legend to the lower right to avoid overlapping the metrics annotation
        ax.legend(fontsize=14, loc='lower right')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'overall_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved overall analysis plot to {save_path}")

    def save_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate regression metrics for each confidence level (and overall) and
        save them to a CSV file.
        """
        metrics_by_level = {}
        
        # Calculate metrics for each confidence level
        for level in ['A', 'B', 'C', 'D']:
            level_data = df[df['confidence_level'] == level]
            metrics = self.calculate_metrics(
                level_data['logMIC'].values,
                level_data['pred_logMIC'].values
            )
            metrics_by_level[level] = metrics
        
        # Calculate overall metrics
        metrics_by_level['Overall'] = self.calculate_metrics(
            df['logMIC'].values,
            df['pred_logMIC'].values
        )
        
        metrics_df = pd.DataFrame(metrics_by_level).T
        csv_path = os.path.join(self.output_dir, 'confidence_metrics.csv')
        metrics_df.to_csv(csv_path)
        print(f"Saved metrics to {csv_path}")
        return metrics_by_level

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Analyze molecular prediction confidence levels.')
    parser.add_argument('input_csv', help='Path to the input CSV file')
    parser.add_argument('--output_dir', default='confidence_analysis_results',
                        help='Directory for output files (default: confidence_analysis_results)')
    args = parser.parse_args()
    
    analyzer = MolecularConfidenceAnalyzer(output_dir=args.output_dir)
    
    try:
        print(f"Loading data from {args.input_csv}...")
        df = analyzer.load_data(args.input_csv)
        print(f"Loaded {len(df)} compounds.")
        
        print("Assigning confidence levels...")
        df_with_confidence = analyzer.assign_confidence_levels(df)
        
        # Report the confidence level distribution
        print("\nConfidence level distribution:")
        total = len(df)
        for level in ['A', 'B', 'C', 'D']:
            count = len(df_with_confidence[df_with_confidence['confidence_level'] == level])
            pct = (count / total) * 100 if total > 0 else 0
            if count > 0:
                print(f"  Level {level}: {count} compounds ({pct:.1f}%)")
            else:
                print(f"  Level {level}: No compounds")
        
        print("\nGenerating individual confidence level plots...")
        analyzer.plot_individual_confidence_levels(df_with_confidence)
        
        print("\nGenerating overall analysis plot...")
        analyzer.plot_overall_analysis(df_with_confidence)
        
        print("\nCalculating and saving metrics...")
        analyzer.save_metrics(df_with_confidence)
        
        print(f"\nAnalysis complete! Results saved to {args.output_dir}")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()

