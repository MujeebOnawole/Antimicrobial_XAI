# final_eval.py

import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import glob
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score, matthews_corrcoef,
    precision_recall_curve, average_precision_score, roc_curve, confusion_matrix 
)
from scipy import stats
from model import BaseGNN
from data_module import MoleculeDataModule
from data_module import collate_fn
from logger import LoggerSetup, get_logger 
from config import Configuration
import matplotlib.pyplot as plt
import seaborn as sns
from statistical_testing import StatisticalTesting, create_diagnostic_plots
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import logging
import random
import traceback

# Initialize logger at the module level
logger = get_logger(__name__)

class FinalEvaluator:
    def __init__(self, config: Configuration):
        """Initialize final evaluator."""
        self.config = config
        self.device = config.device

        # Setup paths
        self.cv_base_dir = os.path.join(
            config.output_dir,
            f"{config.task_name}_{config.task_type}_cv_results"
        )
        self.eval_dir = os.path.join(
            config.output_dir,
            f"{config.task_name}_{config.task_type}_final_eval"
        )
        os.makedirs(self.eval_dir, exist_ok=True)

        # Setup logger
        LoggerSetup.initialize(self.eval_dir, f"{config.task_name}_final_eval")
        # Initialize your custom logger for class instance
        self._logger = get_logger(__name__)

        # Initialize data module for test set
        self.data_module = MoleculeDataModule(config)
        self.data_module.setup()
        self.test_dataset = self.data_module.test_dataset

        # Load hyperparameters from CV
        self._load_hyperparameters()

        # Results storage
        self.individual_results = []
        self.ensemble_results = None

    def _load_hyperparameters(self):
        """Load hyperparameters from CV final results."""
        final_results_file = os.path.join(
            self.cv_base_dir,
            f"{self.config.task_name}_{self.config.task_type}_final_results.json"
        )
    
        if not os.path.exists(final_results_file):
            raise FileNotFoundError(f"Final results file not found: {final_results_file}")
    
        with open(final_results_file, 'r') as f:
            cv_results = json.load(f)
            self.hyperparameters = cv_results['config']['hyperparameters']
            self.cv_results = cv_results.get('cv_results', [])


    def _load_best_model_info(self):
        """Load and validate model information from CSV with proper fold handling."""
        try:
            csv_path = os.path.join(
                self.cv_base_dir,
                f"{self.config.task_name}_{self.config.task_type}_all_run_metrics.csv"
            )
            
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Metrics CSV file not found: {csv_path}")
                
            # Read the CSV file
            metrics_df = pd.read_csv(csv_path)
            
            # Convert column names to be case-insensitive
            metrics_df.columns = [col.lower() for col in metrics_df.columns]
            
            # Adjust fold numbering if needed (add 1 to fold numbers if they start from 0)
            if 'fold' in metrics_df.columns:
                min_fold = metrics_df['fold'].min()
                max_fold = metrics_df['fold'].max()
                
                if min_fold == 0:  # If folds are zero-based
                    self._logger.info("Converting zero-based fold numbering to one-based...")
                    metrics_df['fold'] = metrics_df['fold'] + 1
                    min_fold = metrics_df['fold'].min()
                    max_fold = metrics_df['fold'].max()
                
                expected_max_fold = self.config.statistical_validation['cv_folds']
                
                if max_fold != expected_max_fold:
                    raise ValueError(
                        f"Expected maximum fold number to be {expected_max_fold}, "
                        f"but found maximum fold of {max_fold}"
                    )
                    
                self._logger.info(f"Using folds from {min_fold} to {max_fold}")
            
            # Filter for best models
            # First, check if 'finalsaved' column exists (case-insensitive)
            finalsaved_col = next((col for col in metrics_df.columns if col.lower() == 'finalsaved'), None)
            
            if finalsaved_col:
                best_models = metrics_df[metrics_df[finalsaved_col].str.lower() == 'yes'].copy()
            else:
                # If no FinalSaved column, group by CV and Fold and take best model based on metric
                metric_col = 'auc' if self.config.classification else 'rmse'
                if metric_col not in metrics_df.columns:
                    raise ValueError(f"Required metric column '{metric_col}' not found in CSV")
                    
                best_models = metrics_df.sort_values(metric_col, ascending=not self.config.classification)\
                    .groupby(['cv', 'fold']).first().reset_index()
            
            # Verify we have the expected number of best models
            expected_models = (
                self.config.statistical_validation['cv_repeats'] * 
                self.config.statistical_validation['cv_folds']
            )
            
            if len(best_models) != expected_models:
                self._logger.warning(
                    f"Expected {expected_models} best models, found {len(best_models)}. "
                    "This might indicate missing models or incorrect filtering."
                )
            
            # Add logging for debugging
            self._logger.info("\nBest models summary:")
            for _, row in best_models.iterrows():
                cv = row['cv']
                fold = row['fold']
                metric_value = row['auc' if self.config.classification else 'rmse']
                self._logger.info(f"CV{cv} Fold{fold}: {metric_value:.4f}")
            
            return best_models
            
        except Exception as e:
            self._logger.error(f"Error loading best model info from CSV: {str(e)}")
            raise


    def _load_model_checkpoint(self, cv: int, fold: int) -> BaseGNN:
        """Load model checkpoint with improved error handling and validation based on predict.py approach."""
        try:
            checkpoint_path = os.path.join(
                self.cv_base_dir,
                f"cv{cv}",
                "checkpoints",
                f"{self.config.task_name}_{self.config.task_type}_cv{cv}_fold{fold}_best.ckpt"
            )
            
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
                
            # Load checkpoint with proper error handling
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Validate checkpoint structure
            if 'hyperparameters' not in checkpoint:
                raise ValueError(f"No hyperparameters found in checkpoint for CV{cv}, Fold{fold}")
                
            hyperparams = checkpoint['hyperparameters']
            
            # Log hyperparameters for debugging
            self._logger.info(f"\nHyperparameters for CV{cv}, Fold{fold}:")
            self._logger.info("-" * 50)
            for param_name, param_value in hyperparams.items():
                self._logger.info(f"{param_name}: {param_value}")
            self._logger.info("-" * 50)
            
            # Handle classification explicitly
            if 'config' in checkpoint and 'classification' in checkpoint['config']:
                classification = checkpoint['config']['classification']
            else:
                classification = self.config.classification
                
            # Initialize model with validated hyperparameters
            model = BaseGNN(
                config=self.config,
                rgcn_hidden_feats=hyperparams['rgcn_hidden_feats'],
                ffn_hidden_feats=hyperparams['ffn_hidden_feats'],
                ffn_dropout=hyperparams['ffn_dropout'],
                rgcn_dropout=hyperparams['rgcn_dropout'],
                classification=classification,
                num_classes=2 if classification else None
            )
            
            # Load state dict
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(self.device)
            model.eval()
            
            if self.config.classification:
                metric_value = checkpoint.get('metrics', {}).get('auc', 'N/A')
                metric_str = f"(AUC={metric_value})"
            else:
                metric_value = checkpoint.get('metrics', {}).get('rmse', 'N/A')
                metric_str = f"(RMSE={metric_value})"
                
            self._logger.info(f"Successfully loaded model for CV{cv}, Fold{fold} {metric_str}")
            
            return model
            
        except Exception as e:
            self._logger.error(f"Error loading model for CV{cv} Fold{fold}: {str(e)}")
            self._logger.error(traceback.format_exc())
            raise


    def _evaluate_individual_models(self):
        """Evaluate models with improved error handling."""
        test_loader = self._get_test_loader()
        
        cv_repeats = self.config.statistical_validation['cv_repeats']
        cv_folds = self.config.statistical_validation['cv_folds']
        
        self._logger.info(f"\nEvaluating models from {cv_repeats} CV runs x {cv_folds} folds")
        successful_models = 0
        failed_models = []
        
        for cv in range(1, cv_repeats + 1):
            for fold in range(1, cv_folds + 1):
                try:
                    model = self._load_model_checkpoint(cv, fold)
                    raw_metrics = self._evaluate_on_test(model, test_loader)
                    
                    # Process metrics to ensure they're comparison-safe
                    metrics = {}
                    for key, value in raw_metrics.items():
                        if isinstance(value, dict):
                            # Store nested dictionaries with a special prefix
                            for nested_key, nested_value in value.items():
                                metrics[f"{key}_{nested_key}"] = nested_value
                        else:
                            metrics[key] = value
                    
                    metrics.update({
                        'cv': cv,
                        'fold': fold
                    })
                    
                    self.individual_results.append(metrics)
                    successful_models += 1
                    
                    metric_name = 'auc' if self.config.classification else 'rmse'
                    metric_value = float(metrics.get(metric_name, 0.0))
                    
                    self._logger.info(
                        f"CV{cv} Fold{fold}: "
                        f"{self.config.task_type.upper()}: "
                        f"{metric_value:.4f}"
                    )
                    
                except Exception as e:
                    failed_models.append((cv, fold))
                    self._logger.error(f"Failed processing CV{cv} Fold{fold}: {str(e)}")
                    self._logger.error(traceback.format_exc())
                    continue
                    
                finally:
                    if 'model' in locals():
                        del model
                        torch.cuda.empty_cache()
        
        # Log summary
        total_models = cv_repeats * cv_folds
        self._logger.info(f"\nModel Loading Summary:")
        self._logger.info(f"Successfully loaded {successful_models}/{total_models} models")
        if failed_models:
            self._logger.warning(f"Failed models: {failed_models}")
        
        if successful_models == 0:
            raise RuntimeError("No models were successfully loaded")





    def evaluate_all(self):
        """Run complete evaluation pipeline with improved error handling."""
        try:
            # Core evaluation steps
            self._evaluate_individual_models()
            self._evaluate_ensemble()
            
            # Calculate metrics and save best models for both classification and regression
            if hasattr(self, 'ensemble_results'):
                if self.config.classification:
                    ensemble_preds = np.array(self.ensemble_results.get('predictions', []))
                    ensemble_labels = np.array(self.ensemble_results.get('labels', []))
                    
                    if len(ensemble_preds) > 0 and len(ensemble_labels) > 0:
                        # Calculate ROC AUC and other metrics
                        metrics = self._calculate_metrics(ensemble_labels, ensemble_preds)
                        self.ensemble_results.update(metrics)
                        
                        # Calculate thresholds and metrics
                        thresholds, threshold_metrics = self._find_optimal_thresholds(ensemble_labels, ensemble_preds)
                        
                        # Store results
                        self.ensemble_results['thresholds'] = thresholds
                        for key, value in threshold_metrics.items():
                            self.ensemble_results[key] = value
                        
                        # Generate classification-specific plots
                        self._plot_threshold_metrics(ensemble_labels, ensemble_preds)
                        self._generate_plots()
                
                # Save best models for both classification and regression
                self._save_best_models()
            else:
                self._logger.warning("No ensemble predictions or labels available")
            
            # Statistical Analysis and plots
            try:
                statistical_testing = StatisticalTesting()
                metric = 'auc' if self.config.classification else 'rmse'
                
                # Create diagnostic plots for single model analysis
                _ = statistical_testing.create_single_model_diagnostic_plots(
                    cv_results=self.individual_results,
                    metric=metric,
                    save_path=os.path.join(self.eval_dir, 'single_model_statistical_analysis.png')
                )
                
                # Create general diagnostic plots
                create_diagnostic_plots(
                    self.individual_results,
                    metric=metric,
                    save_path=os.path.join(self.eval_dir, 'statistical_analysis.png'),
                    methods=['RGCN']
                )
                
            except Exception as e:
                self._logger.warning(f"Could not create statistical analysis plots: {str(e)}")
                self._logger.error(traceback.format_exc())
            
            # Save all results
            self._save_results()
            
            # Create diagnostic plots from metrics CSV
            try:
                csv_path = os.path.join(
                    self.cv_base_dir,
                    f"{self.config.task_name}_{self.config.task_type}_all_run_metrics.csv"
                )
                if os.path.exists(csv_path):
                    metrics_df = pd.read_csv(csv_path)
                    metrics_df.columns = [col.lower() for col in metrics_df.columns]
                    
                    metric_name = 'f1' if self.config.classification else 'rmse'
                    
                    if metric_name not in metrics_df.columns:
                        raise ValueError(f"Metric column '{metric_name}' not found in CSV")
                    
                    metrics_df[metric_name] = pd.to_numeric(metrics_df[metric_name], errors='coerce')
                    
                    if 'finalsaved' in metrics_df.columns:
                        metrics_df = metrics_df[metrics_df['finalsaved'].str.lower() == 'yes']
                    
                    cv_results = metrics_df.to_dict(orient='records')
                    
                    create_diagnostic_plots(
                        cv_results,
                        metric=metric_name,
                        save_path=os.path.join(self.eval_dir, 'statistical_analysis_from_csv.png'),
                        methods=['RGCN']
                    )
                else:
                    self._logger.warning(f"Metrics CSV file not found: {csv_path}")
                    
            except Exception as e:
                self._logger.warning(f"Could not create diagnostic plots from CSV: {str(e)}")
                self._logger.error(traceback.format_exc())
            
            # Generate final summary report
            try:
                self._generate_summary_report()
            except Exception as e:
                self._logger.error(f"Error generating summary report: {str(e)}")
                self._logger.error(traceback.format_exc())
                
            self._logger.info("\n?? Evaluation completed successfully!")
            
        except Exception as e:
            self._logger.error(f"Critical error during evaluation: {str(e)}")
            self._logger.error(traceback.format_exc())
            raise




    def _get_test_loader(self):
        """Get DataLoader for test set with proper collate function."""
        # Set worker init function for reproducibility
        def worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.training_optimization['num_workers'],
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,  # Use the imported collate_fn
            persistent_workers=True if self.config.training_optimization['num_workers'] > 0 else False
        )



    def _evaluate_on_test(self, model: BaseGNN, test_loader: DataLoader) -> Dict:
        """Evaluate model on test set with comprehensive metrics and proper error handling."""
        model.eval()
        all_preds = []
        all_labels = []
    
        try:
            with torch.no_grad():
                for batch in test_loader:
                    graphs, labels = batch
                    graphs = graphs.to(self.device)
                    labels = labels.to(self.device)
                    outputs, _ = model(graphs)
    
                    if self.config.classification:
                        preds = torch.sigmoid(outputs)
                    else:
                        preds = outputs
    
                    all_preds.extend(preds.cpu().numpy().flatten())
                    all_labels.extend(labels.cpu().numpy().flatten())
    
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
    
            # Initialize metrics dictionary
            metrics = {
                'raw_predictions': all_preds.tolist(),
                'labels': all_labels.tolist()
            }
    
            if self.config.classification:
                # Threshold-independent metrics
                auc_score = float(roc_auc_score(all_labels, all_preds))
                metrics.update({
                    'auc': auc_score,
                    'roc_auc': auc_score,  # Added for compatibility
                    'average_precision': float(average_precision_score(all_labels, all_preds)),
                    })
    
                # Calculate optimal thresholds BEFORE threshold-dependent metrics
                thresholds, threshold_metrics = self._find_optimal_thresholds(all_labels, all_preds)
                
                # Use optimal F1 threshold for MCC calculation (CONSISTENT APPROACH)
                optimal_f1_threshold = thresholds.get('f1', 0.5)
                optimal_binary_preds = (all_preds >= optimal_f1_threshold).astype(int)
                metrics['matthews_corrcoef_raw'] = float(matthews_corrcoef(all_labels, optimal_binary_preds))
    
                # Store thresholds as float values
                metrics['thresholds'] = {k: float(v) for k, v in thresholds.items()}
    
                # Store threshold metrics ensuring no nested dictionaries in comparison fields
                for threshold_name, threshold_data in threshold_metrics.items():
                    metrics[threshold_name] = threshold_data
                    # Extract confusion matrix to separate key to avoid comparison issues
                    if 'confusion_matrix' in threshold_data:
                        metrics[f"{threshold_name}_confusion_matrix"] = threshold_data.pop('confusion_matrix')
    
            else:
                # Regression metrics
                metrics.update({
                    'mse': float(mean_squared_error(all_labels, all_preds)),
                    'rmse': float(np.sqrt(mean_squared_error(all_labels, all_preds))),
                    'mae': float(mean_absolute_error(all_labels, all_preds)),
                    'r2': float(r2_score(all_labels, all_preds))
                })
    
            return metrics
    
        except Exception as e:
            self._logger.error(f"Error in model evaluation: {str(e)}")
            raise



    def _evaluate_ensemble(self):
        """Create and evaluate ensemble of all models with improved error handling."""
        try:
            if not hasattr(self, 'best_models_df'):
                self.best_models_df = self._load_best_model_info()
                
            test_loader = self._get_test_loader()
            all_predictions = []
            failed_models = []
            successful_models = 0
            
            total_models = len(self.best_models_df)
            self._logger.info(f"\nStarting ensemble evaluation of {total_models} models...")
            
            # Ensure column names are lowercase for consistency
            self.best_models_df.columns = [col.lower() for col in self.best_models_df.columns]
    
            # Iterate through best models
            for _, row in self.best_models_df.iterrows():
                try:
                    # Get values using lowercase column names
                    cv = int(row['cv'])  # Ensure CV is an integer
                    fold = int(row['fold'])  # Ensure fold is an integer
                    metric_value = row['auc' if self.config.classification else 'rmse']
                    metric_name = 'AUC' if self.config.classification else 'RMSE'
                    
                    self._logger.info(f"Loading model CV{cv} Fold{fold} ({metric_name}={metric_value:.4f})...")
                    model = self._load_model_checkpoint(cv, fold)
                    preds = self._get_predictions(model, test_loader)
                    all_predictions.append(preds)
                    successful_models += 1
                    
                except Exception as e:
                    failed_models.append((cv, fold))
                    self._logger.error(f"Failed processing CV{cv} Fold{fold}: {str(e)}")
                    self._logger.error(traceback.format_exc())
                    continue
                    
                finally:
                    if 'model' in locals():
                        del model
                        torch.cuda.empty_cache()
    
            if not all_predictions:
                raise ValueError("No models were successfully evaluated. Cannot create ensemble.")
    
            # Log summary
            self._logger.info(f"\nEnsemble Summary:")
            self._logger.info(f"Successfully processed {successful_models}/{total_models} models")
            if failed_models:
                self._logger.warning(f"Failed models: {failed_models}")
    
            # Stack predictions and average them
            all_predictions = np.stack(all_predictions, axis=0)
            ensemble_preds = np.mean(all_predictions, axis=0)
    
            # Get true labels
            labels = []
            for batch in test_loader:
                labels.extend(batch[1].numpy())
            labels = np.array(labels)
    
            # Calculate metrics and store results
            metrics = self._calculate_metrics(labels, ensemble_preds)
            
            if self.config.classification:
                # Find optimal thresholds for ensemble predictions
                thresholds, threshold_metrics = self._find_optimal_thresholds(labels, ensemble_preds)
                
                self.ensemble_results = {
                    'auc': float(metrics['auc']),
                    'average_precision': float(metrics['average_precision']),
                    'predictions': ensemble_preds.tolist(),
                    'labels': labels.tolist(),
                    'num_models': int(successful_models),
                    'failed_models': failed_models,
                    'thresholds': {k: float(v) for k, v in thresholds.items()},
                }
                
                # Add threshold metrics
                for threshold_name, threshold_data in threshold_metrics.items():
                    self.ensemble_results[threshold_name] = threshold_data
            else:
                self.ensemble_results = {
                    'rmse': float(metrics['rmse']),
                    'mae': float(metrics['mae']),
                    'r2': float(metrics['r2']),
                    'predictions': ensemble_preds.tolist(),
                    'labels': labels.tolist(),
                    'num_models': int(successful_models),
                    'failed_models': failed_models
                }
    
            metric = 'auc' if self.config.classification else 'rmse'
            self._logger.info(f"Ensemble {metric.upper()}: {self.ensemble_results[metric]:.4f}")
            
        except Exception as e:
            self._logger.error(f"Error in ensemble evaluation: {str(e)}")
            self._logger.error(traceback.format_exc())
            raise



    def _find_optimal_thresholds(self, labels: np.ndarray, preds: np.ndarray) -> Tuple[Dict[str, float], Dict[str, Dict]]:
        """Find optimal thresholds based on different criteria.
    
        Args:
            labels (np.ndarray): True binary labels.
            preds (np.ndarray): Predicted probabilities or scores.
    
        Returns:
            Tuple[Dict[str, float], Dict[str, Dict]]: 
                - thresholds: A dictionary mapping criterion names to threshold values.
                - metrics_at_thresholds: A dictionary mapping criterion names to their respective metrics.
        """
        if not self.config.classification:
            return {}, {}

        def find_threshold(y_true, y_pred, criterion='f1'):
            """Find optimal threshold based on criterion."""
            if criterion == 'f1':
                precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
                f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
                optimal_idx = np.argmax(f1_scores)
                return thresholds[optimal_idx], f1_scores[optimal_idx], None
            elif criterion == 'youdens':
                fpr, tpr, thresholds = roc_curve(y_true, y_pred)
                youdens_j = tpr - fpr
                optimal_idx = np.argmax(youdens_j)
                return thresholds[optimal_idx], youdens_j[optimal_idx], None
            elif criterion == 'eer':
                fpr, tpr, thresholds = roc_curve(y_true, y_pred)
                fnr = 1 - tpr
                optimal_idx = np.argmin(np.abs(fpr - fnr))
                return thresholds[optimal_idx], fpr[optimal_idx], fnr[optimal_idx]
            elif criterion == 'pr_break_even':
                precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
                pr_be = np.abs(precision - recall)
                optimal_idx = np.argmin(pr_be)
                return thresholds[optimal_idx], precision[optimal_idx], recall[optimal_idx]
            else:
                raise ValueError(f"Unknown criterion: {criterion}")

        try:
            # Initialize dictionaries to store thresholds and metrics
            thresholds = {'default': 0.5}
            metrics_at_thresholds = {}

            # Log data distribution
            unique_labels, counts = np.unique(labels, return_counts=True)
            self._logger.info(f"Labels distribution: {dict(zip(unique_labels, counts))}")
            self._logger.info(f"Predictions range: min={preds.min():.3f}, max={preds.max():.3f}")

            # Calculate thresholds for different criteria
            criteria = ['f1', 'youdens', 'eer', 'pr_break_even']
            for criterion in criteria:
                threshold, metric1, metric2 = find_threshold(labels, preds, criterion)
                thresholds[criterion] = float(threshold)

                # Calculate metrics at this threshold
                binary_preds = (preds >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(labels, binary_preds).ravel()

                # Avoid division by zero
                specificity = float(tn / (tn + fp + 1e-10))
                npv = float(tn / (tn + fn + 1e-10))
                balanced_accuracy = float((recall_score(labels, binary_preds, zero_division=0) + specificity) / 2)

                threshold_metrics = {
                    'threshold_value': float(threshold),
                    'accuracy': float(accuracy_score(labels, binary_preds)),
                    'precision': float(precision_score(labels, binary_preds, zero_division=0)),
                    'recall': float(recall_score(labels, binary_preds, zero_division=0)),
                    'f1': float(f1_score(labels, binary_preds, zero_division=0)),
                    'cohen_kappa': float(cohen_kappa_score(labels, binary_preds)),
                    'matthews_corrcoef': float(matthews_corrcoef(labels, binary_preds)),
                    'confusion_matrix': {
                        'tn': int(tn),
                        'fp': int(fp),
                        'fn': int(fn),
                        'tp': int(tp)
                    },
                    'specificity': specificity,
                    'npv': npv,
                    'balanced_accuracy': balanced_accuracy
                }

                metrics_at_thresholds[f'metrics_at_{criterion}'] = threshold_metrics

                # Log the values
                self._logger.info(f"\n{criterion.upper()} Threshold Analysis:")
                self._logger.info(f"Threshold: {threshold:.3f}")
                self._logger.info(f"Metrics at threshold:")
                for metric_name, value in threshold_metrics.items():
                    if isinstance(value, dict):
                        self._logger.info(f"  {metric_name}: {value}")
                    else:
                        self._logger.info(f"  {metric_name}: {value:.3f}")

            # Store optimal F1 threshold metrics as the default optimal metrics
            if 'metrics_at_f1' in metrics_at_thresholds:
                metrics_at_thresholds['optimal_threshold_metrics'] = metrics_at_thresholds['metrics_at_f1']

            # Return thresholds and metrics separately
            return thresholds, metrics_at_thresholds

        except Exception as e:
            self._logger.error(f"Error in threshold optimization: {str(e)}")
            raise



    # Configuration constants for variance-based model selection
    VARIANCE_THRESHOLD = 0.1  # 10 % coefficient of variation
    MIN_MODELS_PER_CV = 1      # Minimum models from each CV run
    TOTAL_MODELS_TO_SELECT = 5 # Total models for ensemble
    
    def _calculate_variance_statistics(self, values: List[float]) -> Dict[str, float]:
        """
        Calculate variance statistics for model selection.
        
        Variance threshold justification:
        - CV < 10 %: Literature standard for "excellent reproducibility" (Montgomery, 2019)
        - Statistical rationale: When CV < 10%, model differences are within measurement noise
        - Practical benefit: Diversity-based selection maximizes ensemble error decorrelation
        - Alternative: Manual selection required when variance indicates meaningful differences
        
        Args:
            values (List[float]): Metric values from all models
            
        Returns:
            Dict[str, float]: Statistical measures including coefficient of variation
        """
        try:
            if not values or len(values) < 2:
                return {'mean': 0.0, 'std': 0.0, 'cv': float('inf')}
                
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)  # Sample standard deviation
            cv = std_val / mean_val if mean_val != 0 else float('inf')
            
            return {
                'mean': float(mean_val),
                'std': float(std_val),
                'cv': float(cv)
            }
            
        except Exception as e:
            self._logger.error(f"Error calculating variance statistics: {str(e)}")
            return {'mean': 0.0, 'std': 0.0, 'cv': float('inf')}

    def _validate_model_availability(self, cv: int, fold: int) -> bool:
        """Verify that model checkpoint exists."""
        try:
            checkpoint_path = os.path.join(
                self.cv_base_dir,
                f"cv{cv}",
                "checkpoints",
                f"{self.config.task_name}_{self.config.task_type}_cv{cv}_fold{fold}_best.ckpt"
            )
            return os.path.exists(checkpoint_path)
        except Exception as e:
            self._logger.error(f"Error validating model availability for CV{cv} Fold{fold}: {str(e)}")
            return False


    def _select_diverse_models(self, primary_metric: str) -> List[Dict]:
        """
        Select models using diversity-based strategy when variance is low.
        
        Algorithm:
        1. Select best model from each CV run (based on primary metric)
        2. Fill remaining slots with maximum fold diversity
        
        Args:
            primary_metric (str): Primary metric for selection ('auc' for classification, 'rmse' for regression)
            
        Returns:
            List[Dict]: Selected models ensuring CV representation
        """
        try:
            # Group results by CV run
            cv_groups = {}
            for result in self.individual_results:
                cv = result['cv']
                if cv not in cv_groups:
                    cv_groups[cv] = []
                cv_groups[cv].append(result)
            
            selected_models = []
            
            # Step 1: Select best model from each CV run (with checkpoint validation)
            for cv in sorted(cv_groups.keys()):
                cv_models = cv_groups[cv]
                # Filter models with valid checkpoints
                valid_cv_models = [
                    model for model in cv_models 
                    if self._validate_model_availability(model['cv'], model['fold'])
                ]
                
                if not valid_cv_models:
                    self._logger.warning(f"No valid checkpoints found for CV{cv}")
                    continue
                
                if self.config.classification:
                    # Higher is better for AUC
                    best_model = max(valid_cv_models, key=lambda x: x.get(primary_metric, 0.0))
                else:
                    # Lower is better for RMSE
                    best_model = min(valid_cv_models, key=lambda x: x.get(primary_metric, float('inf')))
                
                selected_models.append(best_model)
                self._logger.info(f"Selected best model from CV{cv}: {primary_metric.upper()}={best_model[primary_metric]:.4f}")
            
            # Step 2: Fill remaining slots with maximum fold diversity
            remaining_slots = max(0, self.TOTAL_MODELS_TO_SELECT - len(selected_models))
            selected_identifiers = {(m['cv'], m['fold']) for m in selected_models}
            
            # Get all remaining models with valid checkpoints
            remaining_models = [
                result for result in self.individual_results 
                if (result['cv'], result['fold']) not in selected_identifiers
                and self._validate_model_availability(result['cv'], result['fold'])
            ]
            
            self._logger.info(f"Found {len(remaining_models)} valid remaining models for diversity selection")
            
            # Simple fold diversity selection: avoid duplicate (CV, Fold) pairs
            # Sort remaining models by performance
            if self.config.classification:
                remaining_models.sort(key=lambda x: x.get(primary_metric, 0.0), reverse=True)
            else:
                remaining_models.sort(key=lambda x: x.get(primary_metric, float('inf')))
            
            # Select remaining models, avoiding duplicates
            for candidate in remaining_models:
                if len(selected_models) >= self.TOTAL_MODELS_TO_SELECT:
                    break
                
                # Check if this (CV, Fold) combination is already selected
                candidate_id = (candidate['cv'], candidate['fold'])
                if candidate_id not in selected_identifiers:
                    selected_models.append(candidate)
                    selected_identifiers.add(candidate_id)
                    self._logger.info(f"Selected additional model CV{candidate['cv']} Fold{candidate['fold']}: "
                                    f"{primary_metric.upper()}={candidate[primary_metric]:.4f}")
            
            return selected_models[:self.TOTAL_MODELS_TO_SELECT]
            
        except Exception as e:
            self._logger.error(f"Error in diversity-based model selection: {str(e)}")
            return []

    def _select_best_models(self) -> List[Dict]:
        """
        Select best models using variance-based strategy.
        
        Uses AUC as primary metric (threshold-independent) to ensure consistency
        with optimal threshold approach used in cross-validation.
        
        Returns:
            List[Dict]: Selected models if variance is low, empty list if high variance
        """
        try:
            # Determine primary metric based on task type
            if self.config.classification:
                primary_metric = 'auc'  # Threshold-independent for consistency
                secondary_metrics = ['average_precision', 'matthews_corrcoef_raw']
            else:
                primary_metric = 'rmse'  # Standard error metric
                secondary_metrics = ['mae', 'r2']
            
            # Extract primary metric values for variance analysis
            primary_values = [result.get(primary_metric, float('inf') if not self.config.classification else 0.0) 
                            for result in self.individual_results]
            
            # Calculate variance statistics
            variance_stats = self._calculate_variance_statistics(primary_values)
            cv_value = variance_stats['cv']
            
            # Log variance analysis
            self._logger.info(f"Variance analysis: CV={cv_value:.4f} (threshold={self.VARIANCE_THRESHOLD:.4f})")
            
            # Decision based on variance
            if cv_value < self.VARIANCE_THRESHOLD:
                self._logger.info(f"Low variance detected - using diversity-based selection")
                self._logger.info(f"Selected models ensure representation from all CV runs")
                
                # Use diversity-based selection
                selected_models = self._select_diverse_models(primary_metric)
                
                if not selected_models:
                    self._logger.warning("Diversity-based selection failed, returning empty list")
                    return []
                
                # Format results with proper key access
                best_models = []
                for model in selected_models:
                    model_info = {
                        'cv': model['cv'],
                        'fold': model['fold'],
                        'metrics': {}
                    }
                    
                    if self.config.classification:
                        # Use metrics already calculated with optimal thresholds
                        model_info['metrics'].update({
                            'auc': float(model['auc']),
                            'average_precision': float(model['average_precision']),
                            'matthews_corrcoef': float(model.get('matthews_corrcoef_raw', 0.0)),
                            # Use optimal threshold metrics already stored
                            'tpr': float(model.get('metrics_at_f1', {}).get('recall', 0.0)),
                            'tnr': float(model.get('metrics_at_f1', {}).get('specificity', 0.0))
                        })
                    else:
                        model_info['metrics'].update({
                            'rmse': float(model['rmse']),
                            'mae': float(model['mae']),
                            'r2': float(model['r2'])
                        })
                        
                    best_models.append(model_info)
                    
                # Log final selection
                self._logger.info(f"\nSelected {len(best_models)} models using diversity-based strategy:")
                for i, model in enumerate(best_models, 1):
                    metrics = model['metrics']
                    self._logger.info(f"\nModel {i} - CV{model['cv']} Fold{model['fold']}:")
                    if self.config.classification:
                        self._logger.info(f"  AUC: {metrics['auc']:.4f}")
                        self._logger.info(f"  Average Precision: {metrics['average_precision']:.4f}")
                        self._logger.info(f"  MCC: {metrics['matthews_corrcoef']:.4f}")
                    else:
                        self._logger.info(f"  RMSE: {metrics['rmse']:.4f}")
                        self._logger.info(f"  MAE: {metrics['mae']:.4f}")
                        self._logger.info(f"  R²: {metrics['r2']:.4f}")
                        
                return best_models
                
            else:
                # High variance - skip automated selection
                self._logger.warning(f"High variance detected: CV={cv_value:.4f} exceeds threshold={self.VARIANCE_THRESHOLD:.4f}")
                self._logger.warning(f"Skipping automated model selection - manual review recommended")
                self._logger.info(f"No best_models.json will be created due to high variance")
                return []
                
        except Exception as e:
            self._logger.error(f"Error selecting best models: {str(e)}")
            self._logger.error(traceback.format_exc())
            return []

    




    def _save_best_models(self):
        """Save the selected best models to JSON with detailed metrics."""
        try:
            best_models = self._select_best_models()
            
            if not best_models:
                raise ValueError("No best models were selected")
                
            # Create best_models directory within final_eval directory
            best_models_dir = os.path.join(
                self.eval_dir,  
                "best_models"
            )
            os.makedirs(best_models_dir, exist_ok=True)
            
            # Update path to match the dependent code's expected location
            best_models_file = os.path.join(best_models_dir, 'best_models.json')
            
            self._logger.info(f"\nPreparing to save best models to: {best_models_file}")
            
            # Format model info to match loading script expectations
            model_data = {
                "models": []  
            }
            
            # Get metrics CSV for init numbers
            metrics_csv = os.path.join(
                self.cv_base_dir,
                f"{self.config.task_name}_{self.config.task_type}_all_run_metrics.csv"
            )
            
            if not os.path.exists(metrics_csv):
                raise FileNotFoundError(f"Metrics CSV file not found: {metrics_csv}")
                
            metrics_df = pd.read_csv(metrics_csv)
            metrics_df.columns = [col.lower() for col in metrics_df.columns]
            
            for model in best_models:
                cv = int(model['cv'])
                fold = int(model['fold'])
                
                # Get init number for this model
                model_metrics = metrics_df[
                    (metrics_df['cv'] == cv) & 
                    (metrics_df['fold'] == fold) &
                    (metrics_df['finalsaved'].str.lower() == 'yes')
                ]
                
                if len(model_metrics) == 0:
                    raise ValueError(f"Could not find metrics for CV{cv} Fold{fold}")
                    
                init_num = int(model_metrics['init'].iloc[0])
                
                model_info = {
                    'cv': cv,
                    'fold': fold,
                    'init': init_num,
                    'metrics': {}
                }
    
                # Add metrics based on task type
                if self.config.classification:
                    model_info['metrics'].update({
                        'auc': float(model['metrics']['auc']),
                        'tpr': float(model['metrics']['tpr']),
                        'tnr': float(model['metrics']['tnr']),
                        'matthews_corrcoef': float(model['metrics']['matthews_corrcoef']),
                        'average_precision': float(model['metrics']['average_precision'])
                    })
                else:  # regression
                    model_info['metrics'].update({
                        'rmse': float(model['metrics']['rmse']),
                        'mae': float(model['metrics']['mae']),
                        'r2': float(model['metrics']['r2'])
                    })
                
                model_data["models"].append(model_info)
    
            # Save to JSON file with proper indentation
            with open(best_models_file, 'w') as f:
                json.dump(model_data, f, indent=4)
    
            # Log results with fold diversity information
            folds_used = [model['fold'] for model in model_data["models"]]
            unique_folds = set(folds_used)
            
            self._logger.info(f"\n✓ Selected and saved best {len(model_data['models'])} models to: {best_models_file}")
            self._logger.info(f"Fold diversity achieved: {len(unique_folds)} unique folds out of {len(model_data['models'])} models")
            self._logger.info(f"Folds used: {sorted(unique_folds)}")
            self._logger.info("Best models saved (ordered by performance):")
            for i, model in enumerate(model_data["models"], 1):
                metrics = model['metrics']
                if self.config.classification:
                    self._logger.info(f"\nModel {i} - CV{model['cv']} Fold{model['fold']} Init{model['init']}:")
                    self._logger.info(f"  AUC: {metrics['auc']:.4f}")
                    self._logger.info(f"  Average Precision: {metrics['average_precision']:.4f}")
                    self._logger.info(f"  MCC: {metrics['matthews_corrcoef']:.4f}")
                else:
                    self._logger.info(f"\nModel {i} - CV{model['cv']} Fold{model['fold']} Init{model['init']}:")
                    self._logger.info(f"  RMSE: {metrics['rmse']:.4f}")
                    self._logger.info(f"  MAE: {metrics['mae']:.4f}")
                    self._logger.info(f"  R2: {metrics['r2']:.4f}")
                    
        except Exception as e:
            self._logger.error(f"Error saving best models: {str(e)}")
            self._logger.error(traceback.format_exc())

    def _generate_plots(self):
        """Generate comprehensive visualization plots for model performance."""
        plot_dir = os.path.join(self.eval_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        if self.config.classification:
            preds = np.array(self.ensemble_results['predictions'])
            labels = np.array(self.ensemble_results['labels'])
            thresholds = self.ensemble_results['thresholds']
            
            # 1. ROC Curve with threshold points
            plt.figure(figsize=(10, 6))
            fpr, tpr, roc_thresholds = roc_curve(labels, preds)
            plt.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {self.ensemble_results["auc"]:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            
            # Add threshold points
            threshold_colors = {'default': 'red', 'f1': 'green', 
                              'youdens': 'purple', 'eer': 'orange',
                              'pr_break_even': 'brown'}
            
            for name, threshold in thresholds.items():
                if name == 'pr_break_even' and not self.config.classification:
                    continue  # Skip if not classification
                threshold_idx = np.abs(roc_thresholds - threshold).argmin()
                plt.plot(fpr[threshold_idx], tpr[threshold_idx], 'o', 
                        color=threshold_colors.get(name, 'black'), 
                        label=f'{name.replace("_", " ").title()} ({threshold:.3f})')
            
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve with Optimal Thresholds')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'roc_curve.png'), bbox_inches='tight', dpi=300)
            plt.close()
            
            # 2. Precision-Recall Curve with threshold points
            plt.figure(figsize=(10, 6))
            precision, recall, pr_thresholds = precision_recall_curve(labels, preds)
            plt.plot(recall, precision, 'b-', 
                    label=f'PR (AP = {self.ensemble_results["average_precision"]:.3f})')
            
            # Add baseline
            plt.axhline(y=sum(labels)/len(labels), color='k', linestyle='--', 
                       label='Random')
            
            # Add threshold points
            for name, threshold in thresholds.items():
                if name == 'pr_break_even' and not self.config.classification:
                    continue  # Skip if not classification
                # Find closest threshold value
                if len(pr_thresholds) > 0:
                    threshold_idx = np.abs(pr_thresholds - threshold).argmin()
                    if threshold_idx < len(precision):
                        plt.plot(recall[threshold_idx], precision[threshold_idx], 'o',
                                color=threshold_colors.get(name, 'black'),
                                label=f'{name.replace("_", " ").title()} ({threshold:.3f})')
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve with Optimal Thresholds')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'pr_curve.png'), bbox_inches='tight', dpi=300)
            plt.close()
            
            # 3. Metrics comparison at different thresholds
            plt.figure(figsize=(12, 6))
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'matthews_corrcoef']
            metric_values = {metric: [] for metric in metrics}
            threshold_names = []
            
            for name in thresholds.keys():
                if f'metrics_at_{name}' in self.ensemble_results:
                    threshold_metrics = self.ensemble_results[f'metrics_at_{name}']
                    for metric in metrics:
                        metric_values[metric].append(threshold_metrics[metric])
                    threshold_names.append(name.replace('_', ' ').title())
            
            x = np.arange(len(threshold_names))
            bar_width = 0.15
            multiplier = 0
            
            for metric in metrics:
                offset = bar_width * multiplier
                plt.bar(x + offset, metric_values[metric], bar_width, 
                       label=metric.replace('_', ' ').title())
                multiplier += 1
            
            plt.xlabel('Threshold Criterion')
            plt.ylabel('Score')
            plt.title('Performance Metrics at Different Thresholds')
            plt.xticks(x + bar_width * (len(metrics)/2 - 0.5), threshold_names, rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'threshold_comparison.png'), bbox_inches='tight', dpi=300)
            plt.close()
            
            # 4. Prediction Distribution
            plt.figure(figsize=(10, 6))
            plt.hist(preds[labels == 0], bins=50, alpha=0.5, label='Negative Class', 
                    density=True, color='red')
            plt.hist(preds[labels == 1], bins=50, alpha=0.5, label='Positive Class', 
                    density=True, color='blue')
            
            # Add vertical lines for thresholds
            for name, threshold in thresholds.items():
                if name == 'pr_break_even' and not self.config.classification:
                    continue  # Skip if not classification
                plt.axvline(x=threshold, color=threshold_colors.get(name, 'black'), linestyle='--',
                           label=f'{name.replace("_", " ").title()} Threshold')
            
            plt.xlabel('Prediction Score')
            plt.ylabel('Density')
            plt.title('Prediction Score Distribution by Class')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'prediction_distribution.png'), bbox_inches='tight', dpi=300)
            plt.close()



    def _plot_threshold_metrics(self, labels: np.ndarray, preds: np.ndarray):
        """Create plot showing how metrics vary with threshold."""
        try:
            if not self.config.classification:
                return
                
            # Create directory for plots
            plot_dir = os.path.join(self.eval_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            
            # Calculate precision and recall for various thresholds
            precision, recall, thresholds_pr = precision_recall_curve(labels, preds)
            
            if len(thresholds_pr) == 0:
                self._logger.warning("No valid thresholds found. Skipping threshold metrics plot.")
                return
                
            # Calculate F1 scores
            epsilon = 1e-10
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + epsilon)
            
            if len(f1_scores) > 0:
                optimal_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds_pr[optimal_idx]
                
                # Create the plot
                plt.figure(figsize=(12, 8))
                
                # Plot metrics
                plt.plot(thresholds_pr, precision[:-1], 'b-', label='Precision', linewidth=2)
                plt.plot(thresholds_pr, recall[:-1], 'r-', label='Recall', linewidth=2)
                plt.plot(thresholds_pr, f1_scores, 'g-', label='F1 Score', linewidth=2)
                
                # Add vertical line for optimal threshold
                plt.axvline(x=optimal_threshold, color='purple', linestyle='--', 
                           label=f'Optimal Threshold = {optimal_threshold:.3f}')
                
                # Add points for metric values at optimal threshold
                plt.plot(optimal_threshold, precision[optimal_idx], 'bo', 
                        label=f'Precision = {precision[optimal_idx]:.3f}')
                plt.plot(optimal_threshold, recall[optimal_idx], 'ro', 
                        label=f'Recall = {recall[optimal_idx]:.3f}')
                plt.plot(optimal_threshold, f1_scores[optimal_idx], 'go', 
                        label=f'F1 = {f1_scores[optimal_idx]:.3f}')
                
                # Customize plot
                plt.title('Performance Metrics vs Classification Threshold', fontsize=14)
                plt.xlabel('Classification Threshold', fontsize=12)
                plt.ylabel('Score', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.tight_layout()
                
                # Save plot
                plt.savefig(os.path.join(plot_dir, 'threshold_metrics.png'), 
                            bbox_inches='tight', dpi=300)
                plt.close()
                
                # Store optimal threshold metrics
                if hasattr(self, 'ensemble_results'):
                    self.ensemble_results['optimal_threshold_metrics'] = {
                        'threshold': float(optimal_threshold),
                        'precision': float(precision[optimal_idx]),
                        'recall': float(recall[optimal_idx]),
                        'f1': float(f1_scores[optimal_idx])
                    }
                
                # Log optimal values
                self._logger.info(f"\nOptimal Threshold Analysis:")
                self._logger.info(f"Optimal Threshold: {optimal_threshold:.3f}")
                self._logger.info(f"At optimal threshold:")
                self._logger.info(f"  Precision: {precision[optimal_idx]:.3f}")
                self._logger.info(f"  Recall: {recall[optimal_idx]:.3f}")
                self._logger.info(f"  F1 Score: {f1_scores[optimal_idx]:.3f}")
            else:
                self._logger.warning("Could not compute F1 scores. Skipping threshold metrics plot.")
                
        except Exception as e:
            self._logger.error(f"Error in threshold metrics plotting: {str(e)}")
            self._logger.error(traceback.format_exc())






    def _generate_summary_report(self):
        """Generate human-readable summary report with both classification and regression metrics."""
        try:
            metric = 'auc' if self.config.classification else 'rmse'
            values = [r[metric] for r in self.individual_results]
            
            # Initialize the report list
            report = [
                "Final Evaluation Summary",
                "=" * 50,
            ]
            
            # Add basic statistics
            if values:
                report.extend([
                    f"\nIndividual Models ({len(values)} total):",
                    f"Mean {metric.upper()}: {np.mean(values):.4f} +/- {np.std(values):.4f}",
                    f"Median {metric.upper()}: {np.median(values):.4f}",
                    f"Min {metric.upper()}: {np.min(values):.4f}",
                    f"Max {metric.upper()}: {np.max(values):.4f}"
                ])
            
            # Add performance distribution and variance analysis
            try:
                cv_analysis = self.analyze_cv_performance()
                if cv_analysis:
                    variance_analysis = self.analyze_variance_components()
                    if variance_analysis:
                        report.extend([
                            f"\nDetailed Performance Analysis:",
                            f"95% Confidence Interval: [{cv_analysis['ci_lower']:.4f}, {cv_analysis['ci_upper']:.4f}]",
                            f"Shapiro-Wilk normality test p-value: {cv_analysis['normality_p_value']:.4f}",
                            "\nVariance Components:",
                            f"Between CV runs: {variance_analysis['between_cv']:.4f}",
                            f"Between folds: {variance_analysis['between_fold']:.4f}",
                            f"Residual variance: {variance_analysis['residual']:.4f}"
                        ])
            except Exception as e:
                self._logger.error(f"Error in performance analysis: {str(e)}")
            
            # Branch for classification metrics
            if self.config.classification:
                optimal_metrics = self.ensemble_results.get('metrics_at_f1', {})
                optimal_threshold = optimal_metrics.get('threshold_value')
                
                if optimal_threshold is not None:
                    report.extend([f"\nOptimal Threshold from Ensemble: {optimal_threshold:.4f}"])
                
                report.extend(["\nDetailed Individual Model Metrics:"])
                
                # Sort models by AUC for better readability
                sorted_models = sorted(self.individual_results, 
                                     key=lambda x: x.get('auc', 0.0),
                                     reverse=True)
                
                for model in sorted_models:
                    cv = model['cv']
                    fold = model['fold']
                    auc = model['auc']
                    avg_precision = model['average_precision']
                    mcc = model.get('matthews_corrcoef_raw', 0.0)
                    
                    # Calculate TPR/TNR at optimal threshold
                    if optimal_threshold is not None:
                        predictions = np.array(model['raw_predictions'])
                        labels = np.array(model['labels'])
                        binary_preds = (predictions >= optimal_threshold).astype(int)
                        tn, fp, fn, tp = confusion_matrix(labels, binary_preds).ravel()
                        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
                        
                        # Calculate balanced accuracy for display
                        balanced_acc = (tpr + tnr) / 2
                        
                        report.extend([
                            f"\nCV{cv} Fold{fold} Metrics:",
                            f"  AUC: {auc:.4f}",
                            f"  TPR: {tpr:.4f}",
                            f"  TNR: {tnr:.4f}",
                            f"  MCC: {mcc:.4f}",
                            f"  Average Precision: {avg_precision:.4f}",
                            f"  Balanced Accuracy: {balanced_acc:.4f}"
                        ])
                
            # Branch for regression metrics
            else:
                report.extend(["\nDetailed Individual Model Metrics:"])
                
                # Sort models by RMSE (ascending) for regression
                sorted_models = sorted(self.individual_results, 
                                     key=lambda x: x.get('rmse', float('inf')))
                
                # Calculate additional statistics for regression
                mae_values = [r.get('mae', float('inf')) for r in sorted_models]
                r2_values = [r.get('r2', -float('inf')) for r in sorted_models]
                
                # Add overall regression statistics
                report.extend([
                    "\nOverall Regression Statistics:",
                    f"MAE - Mean: {np.mean(mae_values):.4f}, Std: {np.std(mae_values):.4f}",
                    f"R2 - Mean: {np.mean(r2_values):.4f}, Std: {np.std(r2_values):.4f}",
                    "\nDetailed Model Performance:"
                ])
                
                for model in sorted_models:
                    cv = model['cv']
                    fold = model['fold']
                    rmse = model.get('rmse', float('inf'))
                    mae = model.get('mae', float('inf'))
                    r2 = model.get('r2', -float('inf'))
                    mse = model.get('mse', float('inf'))
                    
                    report.extend([
                        f"\nCV{cv} Fold{fold} Metrics:",
                        f"  RMSE: {rmse:.4f}",
                        f"  MAE: {mae:.4f}",
                        f"  MSE: {mse:.4f}",
                        f"  R2: {r2:.4f}"
                    ])
            
            # Add ensemble results
            if hasattr(self, 'ensemble_results'):
                report.extend(["\nEnsemble Model Performance:"])
                
                if self.config.classification:
                    report.extend([
                        f"AUC: {self.ensemble_results.get('auc', 'N/A'):.4f}"
                    ])
                    
                    if optimal_metrics:
                        report.extend([
                            f"\nAt optimal threshold {optimal_threshold:.4f}:",
                            f"  Precision: {optimal_metrics.get('precision', 'N/A'):.4f}",
                            f"  Recall: {optimal_metrics.get('recall', 'N/A'):.4f}",
                            f"  TNR: {optimal_metrics.get('specificity', 'N/A'):.4f}",
                            f"  TPR: {optimal_metrics.get('recall', 'N/A'):.4f}",
                            f"  F1: {optimal_metrics.get('f1', 'N/A'):.4f}",
                            f"  MCC: {optimal_metrics.get('matthews_corrcoef', 'N/A'):.4f}",
                            f"  Balanced Accuracy: {optimal_metrics.get('balanced_accuracy', 'N/A'):.4f}"
                        ])
                else:
                    report.extend([
                        f"RMSE: {self.ensemble_results.get('rmse', 'N/A'):.4f}",
                        f"MAE: {self.ensemble_results.get('mae', 'N/A'):.4f}",
                        f"R2: {self.ensemble_results.get('r2', 'N/A'):.4f}"
                    ])
            
            # Add best models section if available
            try:
                best_models = self._select_best_models()
                if best_models:
                    report.extend([
                        f"\nSelected Best 5 Models (Based on {'Diversity-based Selection' if self.config.classification else 'RMSE-based Selection'}):",
                        "-" * 50
                    ])
                    
                    for i, model in enumerate(best_models, 1):
                        metrics = model['metrics']
                        report.append(f"\nModel {i}:")
                        report.append(f"CV: {model['cv']}, Fold: {model['fold']}")
                        
                        if self.config.classification:
                            report.extend([
                                f"AUC: {metrics['auc']:.4f}",
                                f"TPR: {metrics['tpr']:.4f}",
                                f"TNR: {metrics['tnr']:.4f}",
                                f"MCC: {metrics['matthews_corrcoef']:.4f}",
                                f"Average Precision: {metrics['average_precision']:.4f}"
                            ])
                        else:
                            report.extend([
                                f"RMSE: {metrics['rmse']:.4f}",
                                f"MAE: {metrics['mae']:.4f}",
                                f"R2: {metrics['r2']:.4f}"
                            ])
            except Exception as e:
                self._logger.warning(f"Could not include best models section: {str(e)}")
            
            # Write report to file
            with open(os.path.join(self.eval_dir, 'summary_report.txt'), 'w') as f:
                f.write('\n'.join(report))
                
        except Exception as e:
            self._logger.error(f"Error generating summary report: {str(e)}")
            self._logger.error(traceback.format_exc())



    def _save_results(self):
        """Save comprehensive evaluation results."""
        # Prepare summary DataFrame for individual models
        summary_data = []
        for result in self.individual_results:
            base_metrics = {
                'cv': result.get('cv'),
                'fold': result.get('fold')
            }
    
            if self.config.classification:
                # Add threshold-independent metrics
                base_metrics.update({
                    'roc_auc': result.get('roc_auc', result.get('auc', 'N/A')),  # Fallback to 'auc' if 'roc_auc' is missing
                    'average_precision': result.get('average_precision', 'N/A'),
                    'matthews_corrcoef_raw': result.get('matthews_corrcoef_raw', 'N/A')
                })
    
                # Add metrics at each threshold
                for threshold_name in ['f1', 'youdens', 'eer', 'pr_break_even']:
                    metrics_key = f'metrics_at_{threshold_name}'
                    if metrics_key in result:
                        threshold_metrics = result[metrics_key]
                        for metric_name, value in threshold_metrics.items():
                            if not isinstance(value, dict):  # Skip nested structures like confusion matrix
                                base_metrics[f'{threshold_name}_{metric_name}'] = value
            else:
                # Regression metrics
                base_metrics.update({
                    'mse': result.get('mse', 'N/A'),
                    'rmse': result.get('rmse', 'N/A'),
                    'mae': result.get('mae', 'N/A'),
                    'r2': result.get('r2', 'N/A')
                })
    
            summary_data.append(base_metrics)
    
        # Create and save summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(self.eval_dir, 'individual_model_metrics.csv'), index=False)
    
        # Save ensemble results
        if self.ensemble_results:
            ensemble_df = pd.DataFrame([self.ensemble_results])
            ensemble_df.to_csv(os.path.join(self.eval_dir, 'ensemble_metrics.csv'), index=False)
    
        # Calculate confidence intervals and normality test results
        try:
            cv_performance = self.analyze_cv_performance()
            variance_components = self.analyze_variance_components()
        except Exception as e:
            self._logger.warning(f"Could not perform additional analyses for saving: {str(e)}")
            cv_performance = {}
            variance_components = {}
    
        # Save full results including curve data and statistical analyses
        results = {
            'config': {
                'task_name': self.config.task_name,
                'task_type': self.config.task_type,
                'classification': self.config.classification
            },
            'individual_results': self.individual_results,
            'ensemble_results': self.ensemble_results,
            'statistical_analysis': {
                'confidence_intervals': cv_performance,
                'variance_components': variance_components
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
        with open(os.path.join(self.eval_dir, 'full_results.json'), 'w') as f:
            json.dump(results, f, indent=4)


    def _get_predictions(self, model: BaseGNN, loader: DataLoader) -> np.ndarray:
        """Get model predictions.
        
        Args:
            model (BaseGNN): The loaded model
            loader (DataLoader): DataLoader containing the test data
            
        Returns:
            np.ndarray: Array of predictions
        """
        model.eval()
        all_preds = []
        
        with torch.no_grad():
            for batch in loader:
                # Unpack the batch - batch is a tuple of (graphs, labels) from collate_fn
                graphs, _ = batch
                
                # Move graphs to device
                graphs = graphs.to(self.device)
                
                # Get predictions
                outputs, _ = model(graphs)
                preds = torch.sigmoid(outputs) if self.config.classification else outputs
                all_preds.extend(preds.cpu().numpy().flatten())
                    
        return np.array(all_preds)



    def _calculate_metrics(self, labels: np.ndarray, predictions: np.ndarray) -> Dict:
        """Calculate evaluation metrics with improved error handling."""
        try:
            metrics = {}
            
            if self.config.classification:
                # Threshold-independent metrics
                metrics.update({
                    'auc': float(roc_auc_score(labels, predictions)),
                    'roc_auc': float(roc_auc_score(labels, predictions)),  # Add both keys for compatibility
                    'average_precision': float(average_precision_score(labels, predictions))
                })
                
                # Calculate metrics at different thresholds
                thresholds = [0.5]  # Default threshold
                for threshold in thresholds:
                    binary_preds = (predictions >= threshold).astype(int)
                    metrics.update({
                        f'accuracy_{threshold}': float(accuracy_score(labels, binary_preds)),
                        f'precision_{threshold}': float(precision_score(labels, binary_preds, zero_division=0)),
                        f'recall_{threshold}': float(recall_score(labels, binary_preds, zero_division=0)),
                        f'f1_{threshold}': float(f1_score(labels, binary_preds, zero_division=0))
                    })
            else:
                metrics.update({
                    'rmse': float(np.sqrt(mean_squared_error(labels, predictions))),
                    'mae': float(mean_absolute_error(labels, predictions)),
                    'r2': float(r2_score(labels, predictions))
                })
                
            return metrics
            
        except Exception as e:
            self._logger.error(f"Error calculating metrics: {str(e)}")
            self._logger.error(traceback.format_exc())
            raise

    def calculate_confidence_intervals(self, metric_values, confidence=0.95):
        """Calculate confidence intervals for metrics."""
        n = len(metric_values)
        mean = np.mean(metric_values)
        se = stats.sem(metric_values)
        ci = stats.t.interval(confidence, n-1, loc=mean, scale=se)
        return mean, ci

    def analyze_cv_performance(self):
        """Analyze cross-validation performance with confidence intervals."""
        metric = 'auc' if self.config.classification else 'rmse'
        values = [r[metric] for r in self.individual_results]
        
        # Calculate confidence intervals
        mean, ci = self.calculate_confidence_intervals(values)
        ci_lower, ci_upper = ci
        
        # Perform normality test
        _, p_value = stats.shapiro(values)
        
        return {
            'mean': mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std': np.std(values),
            'normality_p_value': p_value
        }




    def analyze_variance_components(self):
        """Analyze variance components from repeated CV using two-way ANOVA."""
        try:
            metric = 'auc' if self.config.classification else 'rmse'
            df = pd.DataFrame(self.individual_results)
    
            # Ensure proper data formatting
            df['cv'] = df['cv'].astype(str)
            df['fold'] = df['fold'].astype(str)
    
            # Perform two-way ANOVA
            formula = f"{metric} ~ C(cv) + C(fold)"
            model = ols(formula, data=df).fit()
            anova_table = anova_lm(model)
    
            variance_components = {
                'between_cv': float(anova_table.loc['C(cv)', 'mean_sq']),
                'between_fold': float(anova_table.loc['C(fold)', 'mean_sq']),
                'residual': float(anova_table.loc['Residual', 'mean_sq'])
            }
    
            return variance_components
        except Exception as e:
            self._logger.error(f"Error in variance analysis: {str(e)}")
            return None


    def plot_performance_distribution(self):
        """Create performance distribution plots."""
        try:
            metric = 'auc' if self.config.classification else 'rmse'
            values = [r[metric] for r in self.individual_results]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram with KDE
            sns.histplot(values, kde=True, ax=ax1, color='skyblue', edgecolor='black')
            ax1.set_title(f'{metric.upper()} Distribution')
            ax1.set_xlabel(metric.upper())
            ax1.set_ylabel('Frequency')
            
            # Box plot
            sns.boxplot(y=values, ax=ax2, color='lightgreen')
            ax2.set_title(f'{metric.upper()} Box Plot')
            ax2.set_ylabel(metric.upper())
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.eval_dir, 'performance_distribution.png'))
            plt.close()
            
            self._logger.info("Performance distribution plots saved successfully.")
        except Exception as e:
            self._logger.error(f"Error in plotting performance distribution: {str(e)}")
            self._logger.error(traceback.format_exc())



if __name__ == "__main__":
    # Initialize configuration
    config = Configuration()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Initialize evaluator
        evaluator = FinalEvaluator(config)
        evaluator.evaluate_all()
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
