# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_add_pool as sum_nodes
import pytorch_lightning as pl
from torchmetrics import AUROC, AveragePrecision
from torchmetrics.classification import (
    BinaryAccuracy, BinaryF1Score, BinaryPrecision, 
    BinaryRecall, BinarySpecificity, BinaryMatthewsCorrCoef,
    BinaryAUROC, BinaryAveragePrecision, BinaryCohenKappa
)
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
import logging
from typing import List, Tuple, Dict, Any, Optional
from logger import get_logger
from config import Configuration
import psutil 
import gc
from memory_tracker import MemoryTracker
import numpy as np 

# Get configuration
config = Configuration()

logger = get_logger(__name__)
logger.info("Starting model process...")


class WeightAndSum(nn.Module):
    """Compute importance weights for atoms and perform a weighted sum."""
    def __init__(self, in_feats):
        super(WeightAndSum, self).__init__()
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
        )

    def forward(self, g, compute_weights=True):
        if compute_weights:
            # Compute atom weights and perform weighted sum
            weight = self.atom_weighting(g.x)
            if hasattr(g, 'smask'):
                weight = weight * g.smask.unsqueeze(-1)
            weighted_feats = g.x * weight
            h_g_sum = sum_nodes(weighted_feats, g.batch)
            return h_g_sum, weight
        else:
            # Skip weight computation; perform a simple sum pooling
            h_g_sum = sum_nodes(g.x, g.batch)
            return h_g_sum, None



class RGCNLayer(nn.Module):
    """Single layer RGCN with detailed dimension tracking."""

    def __init__(self, in_feats, out_feats, num_rels=65, activation=F.relu,
                 residual=True, batchnorm=True, rgcn_dropout=0.5):
        super(RGCNLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.activation = activation
        self.graph_conv_layer = RGCNConv(
            in_channels=in_feats,
            out_channels=out_feats,
            num_relations=num_rels,
            num_bases=None
        )
        self.residual = residual
        if self.residual:
            self.res_connection = nn.Linear(in_feats, out_feats)
        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)
        self.dropout = nn.Dropout(rgcn_dropout)
        self._logger = get_logger(__name__)
        self._logger.debug(f"Initialized RGCNLayer with dropout rate: {rgcn_dropout}")

    def forward(self, g):
        """Forward pass with detailed dimension tracking."""
        # Log input dimensions
        self._logger.debug(f"RGCN layer input dimension: {g.x.shape[1]}")
        
        # Validate input dimensions
        if g.x.shape[1] != self.in_feats:
            raise ValueError(
                f"Input feature dimension mismatch in RGCN layer. "
                f"Expected {self.in_feats}, got {g.x.shape[1]}. "
                f"Layer settings: in_feats={self.in_feats}, out_feats={self.out_feats}"
            )
        
        # Apply RGCN convolution
        # INTENTIONAL DESIGN: 3-edge Relational GCN
        # Edge types: SINGLE=1, DOUBLE=2, TRIPLE=3, AROMATIC=-1 (undefined)
        # For BACKWARD COMPATIBILITY with legacy checkpoints:
        # Aromatic bonds (type -1) are remapped to SINGLE (type 1).
        # This preserves the original behavior where GetBondTypeAsDouble()
        # returned 1.5 for aromatic, which was truncated to 1 by .long().
        # Aromaticity is also captured via node features (atom.GetIsAromatic()).
        if hasattr(g, 'edge_type'):
            edge_type = g.edge_type.long()
            # Remap undefined types (-1) to SINGLE (1) for backward compatibility
            edge_type = torch.where(edge_type < 0, torch.ones_like(edge_type), edge_type)
            new_feats = self.graph_conv_layer(g.x, g.edge_index, edge_type)
        else:
            new_feats = self.graph_conv_layer(g.x, g.edge_index, None)
        self._logger.debug(f"After RGCN conv dimension: {new_feats.shape[1]}")
        
        # Apply residual connection if enabled
        if self.residual:
            res_feats = self.res_connection(g.x)
            new_feats = new_feats + res_feats
            self._logger.debug(f"After residual dimension: {new_feats.shape[1]}")
        
        # Apply batch normalization if enabled
        if self.bn:
            new_feats = self.bn_layer(new_feats)
            self._logger.debug(f"After batch norm dimension: {new_feats.shape[1]}")
        
        # Apply activation and dropout
        if self.activation is not None:
            new_feats = self.activation(new_feats)
        new_feats = self.dropout(new_feats)
        
        # Validate output dimensions
        if new_feats.shape[1] != self.out_feats:
            raise ValueError(
                f"Output feature dimension mismatch in RGCN layer. "
                f"Expected {self.out_feats}, got {new_feats.shape[1]}"
            )
        
        self._logger.debug(f"RGCN layer output dimension: {new_feats.shape[1]}")
        return new_feats





class BaseGNN(pl.LightningModule):
    def __init__(self, config: Configuration, rgcn_hidden_feats, ffn_hidden_feats, 
                 ffn_dropout, rgcn_dropout, classification=None, num_classes=None):
        """Initialize the BaseGNN model.
        
        Args:
            config (Configuration): Configuration object
            rgcn_hidden_feats: Hidden features for RGCN layers
            ffn_hidden_feats: Hidden features for FFN layers
            ffn_dropout (float): FFN dropout rate
            rgcn_dropout (float): RGCN dropout rate
            classification (bool, optional): Override classification flag. If None, uses config value.
            num_classes (int, optional): Number of classes for classification. If None, determined automatically.
        """
        super().__init__()
        
        
        # Store configuration
        self.config = config
        
        # Initialize your custom self._logger
        self._logger = get_logger(__name__)
        
        # Validate parameters before using them
        self._validate_params(rgcn_hidden_feats, ffn_hidden_feats, ffn_dropout, rgcn_dropout)
        
        # Set classification mode from config if not explicitly provided
        self._classification = classification if classification is not None else config.classification
        
        # Determine number of classes if needed
        if self._classification:
            if num_classes is not None:
                self._num_classes = num_classes
            elif hasattr(config, 'num_classes'):
                self._num_classes = config.num_classes
            elif hasattr(config, 'train_labels') and config.train_labels is not None:
                self._num_classes = len(np.unique(config.train_labels))
            else:
                self._num_classes = 2  # Default to binary classification
                self._logger.warning("Number of classes not provided, defaulting to binary classification")
        else:
            self._num_classes = None
            
        # Initialize instance attributes
        self._rgcn_hidden_feats = rgcn_hidden_feats
        self._ffn_hidden_feats = ffn_hidden_feats
        self._ffn_dropout = ffn_dropout
        self._rgcn_dropout = rgcn_dropout
        
        # Save hyperparameters
        self.save_hyperparameters(
            'rgcn_hidden_feats',
            'ffn_hidden_feats',
            'ffn_dropout',
            'rgcn_dropout',
            'classification'
        )
        
        # Store initial values for verification
        self._initial_params = {
            'rgcn_dropout': rgcn_dropout,
            'ffn_dropout': ffn_dropout,
            'rgcn_hidden_feats': rgcn_hidden_feats,
            'ffn_hidden_feats': ffn_hidden_feats,
        }
        
        # Initialize RGCN layers
        self.rgcn_gnn_layers = nn.ModuleList()
        in_feats = config.num_node_features
        
        # Build RGCN layers
        for out_feats in self._rgcn_hidden_feats:
            if isinstance(out_feats, str):
                out_feats = int(out_feats)
                
            self.rgcn_gnn_layers.append(
                RGCNLayer(
                    in_feats=in_feats,
                    out_feats=out_feats,
                    num_rels=config.num_edge_types,
                    activation=F.relu, 
                    residual=True,
                    batchnorm=True,
                    rgcn_dropout=self._rgcn_dropout
                )
            )
            in_feats = out_feats

        # Readout layer
        self.readout = WeightAndSum(in_feats)
        
        # Fully Connected Layers
        self.fc_layers1 = self._fc_layer(self._ffn_dropout, in_feats, self._ffn_hidden_feats)
        self.fc_layers2 = self._fc_layer(self._ffn_dropout, self._ffn_hidden_feats, self._ffn_hidden_feats)
        self.fc_layers3 = self._fc_layer(self._ffn_dropout, self._ffn_hidden_feats, self._ffn_hidden_feats)
        
        # Output layer - number of outputs based on task
        if self._classification:
            if self._num_classes == 2:
                out_features = 1  # For binary classification, output should be 1
            else:
                out_features = self._num_classes
        else:
            out_features = 1
        self.predict = self._output_layer(self._ffn_hidden_feats, out_features)
        
        # Initialize metrics
        self._init_metrics()
        
        # Log initialization
        self._logger.debug(f"Initialized BaseGNN with {len(self.rgcn_gnn_layers)} RGCN layers")
        if self._classification:
            self._logger.info(f"Initialized for classification with {self._num_classes} classes")
        else:
            self._logger.info("Initialized for regression")


    def verify_dropout_values(self):
        """Verify that dropout values are consistent throughout the model."""
        # Check RGCN layers
        for idx, layer in enumerate(self.rgcn_gnn_layers):
            if not hasattr(layer, 'dropout') or layer.dropout.p != self._rgcn_dropout:
                raise ValueError(
                    f"RGCN layer {idx} has incorrect dropout rate. "
                    f"Expected {self._rgcn_dropout}, got {layer.dropout.p if hasattr(layer, 'dropout') else 'None'}"
                )
        
        # Check FFN layers
        for name, module in self.named_modules():
            if isinstance(module, nn.Dropout):
                if 'fc_layers' in name and module.p != self._ffn_dropout:
                    raise ValueError(
                        f"FFN layer {name} has incorrect dropout rate. "
                        f"Expected {self._ffn_dropout}, got {module.p}"
                    )


    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, config=None, **kwargs):
        """Simplified checkpoint loading."""
        model = super().load_from_checkpoint(checkpoint_path, config=config, **kwargs)
        model.config = config  # Update config if necessary
        
        # Basic verification without loading checkpoint again
        model.verify_dropout_values()  # Keep this as it's a quick runtime check
        return model

    def configure_optimizers(self):
        """Enhanced optimizer configuration with parameter validation."""
        # Verify required parameters exist in config
        required_params = {'lr', 'weight_decay', 'scheduler_factor', 'scheduler_patience'}
        missing_params = [param for param in required_params if not hasattr(self.config, param)]
        if missing_params:
            raise ValueError(f"Missing required configuration parameters: {missing_params}")
        
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max' if self.classification else 'min',
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_mcc' if self.classification else 'val_rmse',
                'interval': 'epoch',
                'frequency': 1
            }
        }


    def _validate_params(self, rgcn_hidden_feats, ffn_hidden_feats, ffn_dropout, rgcn_dropout):
        """Validate parameters before initialization."""
        try:
            if not isinstance(ffn_dropout, float) or not 0 <= ffn_dropout <= 1:
                raise ValueError(f"Invalid ffn_dropout value: {ffn_dropout}")
            if not isinstance(rgcn_dropout, float) or not 0 <= rgcn_dropout <= 1:
                raise ValueError(f"Invalid rgcn_dropout value: {rgcn_dropout}")
                
            # Validate feature dimensions
            if isinstance(rgcn_hidden_feats, (list, tuple)):
                if not all(isinstance(x, (int, str)) for x in rgcn_hidden_feats):
                    raise ValueError("rgcn_hidden_feats must be integers or strings")
            elif not isinstance(rgcn_hidden_feats, (int, str)):
                raise ValueError("rgcn_hidden_feats must be int, str, or list/tuple of these")
                
            if not isinstance(ffn_hidden_feats, (int, str)):
                raise ValueError("ffn_hidden_feats must be an integer or string")
                
            # Validate config parameters
            required_config_params = [
                'num_node_features',
                'num_edge_types',
                'classification'
            ]
            
            missing_params = [param for param in required_config_params 
                             if not hasattr(self.config, param)]
            
            if missing_params:
                raise ValueError(f"Missing required configuration parameters: {missing_params}")
                
        except Exception as e:
            self._logger.error(f"Parameter validation failed: {str(e)}")
            raise


    def on_train_start(self):
        """Verify model configuration before training starts."""
        self.verify_dropout_values()
        self._logger.info(f"Model configuration verified successfully:")
        self._logger.info(f"RGCN dropout: {self._rgcn_dropout}")
        self._logger.info(f"FFN dropout: {self._ffn_dropout}")
        self._logger.info(f"RGCN hidden features: {self._rgcn_hidden_feats}")
        self._logger.info(f"FFN hidden features: {self._ffn_hidden_feats}")
    
    # Add property decorators for accessing the protected attributes
    @property
    def rgcn_hidden_feats(self):
        return self._rgcn_hidden_feats
    
    @property
    def ffn_hidden_feats(self):
        return self._ffn_hidden_feats
    
    @property
    def ffn_dropout(self):
        return self._ffn_dropout
    
    @property
    def classification(self):
        return self._classification
    
    @property
    def num_classes(self):
        return self._num_classes


    def _fc_layer(self, dropout, in_feats, hidden_feats):
        """Creates a fully connected layer with dropout, ReLU, and batch norm."""
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_feats)
        )

    def _output_layer(self, hidden_feats, out_feats):
        """Creates the final output layer."""
        return nn.Sequential(
            nn.Linear(hidden_feats, out_feats)
        )



    def _init_metrics(self):
        """Initialize metrics for training, validation, and testing."""
        if self.classification:
            # Training and Validation Metrics
            self.train_accuracy = BinaryAccuracy()
            self.val_accuracy = BinaryAccuracy()
            self.train_auc = AUROC(task='binary' if self.num_classes == 2 else 'multiclass',
                                   num_classes=self.num_classes)
            self.val_auc = AUROC(task='binary' if self.num_classes == 2 else 'multiclass',
                                 num_classes=self.num_classes)
            self.val_mcc = BinaryMatthewsCorrCoef()
    
            # Test Metrics
            self.test_accuracy = BinaryAccuracy()
            self.test_auc = AUROC(task='binary' if self.num_classes == 2 else 'multiclass',
                                  num_classes=self.num_classes)
            self.test_prc = AveragePrecision(task='binary' if self.num_classes == 2 else 'multiclass',
                                             num_classes=self.num_classes)
            self.test_f1 = BinaryF1Score()
            self.test_precision = BinaryPrecision()
            self.test_recall = BinaryRecall()
            self.test_specificity = BinarySpecificity()
            self.test_mcc = BinaryMatthewsCorrCoef()
            self.test_kappa = BinaryCohenKappa()
        else:
            # Regression Metrics
            self.train_rmse = None
            self.val_rmse = None
    
            # Test Metrics
            self.test_mae = MeanAbsoluteError()
            self.test_mse = MeanSquaredError()
            self.test_r2 = R2Score()




    def _log_memory_usage(self, stage: str):
        """Optimized memory logging with reduced frequency."""
        # Only log memory if debug logging is enabled
        if self._logger.getEffectiveLevel() <= logging.DEBUG:
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)
                self._logger.debug(f"{stage} - GPU Memory: {gpu_mem:.2f} GB")
            cpu_mem = psutil.virtual_memory().percent
            self._logger.debug(f"{stage} - CPU Usage: {cpu_mem}%")



    def forward(self, batch, compute_atom_weights: Optional[bool] = True):
        """Forward pass with masked features for consistent predictions."""
        # Always apply substructure mask if available - THIS IS CRITICAL
        if hasattr(batch, 'smask'):
            smask = batch.smask.to(batch.x.device).unsqueeze(-1)
            batch.x = batch.x * smask  # Zero out features of masked atoms
        
        # Process through RGCN layers
        for rgcn_layer in self.rgcn_gnn_layers:
            batch.x = rgcn_layer(batch)
        
        # Always use weighted sum for prediction consistency with training
        graph_feats, weight = self.readout(batch, compute_weights=True)
        
        # Pass through fully connected layers
        h1 = self.fc_layers1(graph_feats)
        h2 = self.fc_layers2(h1)
        h3 = self.fc_layers3(h2)
        out = self.predict(h3)
        
        return out, weight









    def training_step(self, batch, batch_idx):
        """Training step with proper handling of pos_weight parameter."""
        graphs, labels = batch
        preds, weights = self(graphs)
        
        labels_float = labels.float().unsqueeze(1)
        
        if self.classification:
            # Handle pos_weight parameter properly
            pos_weight = None
            if hasattr(self.config, 'pos_weight_multiplier') and self.config.pos_weight_multiplier is not None:
                pos_weight = torch.tensor([self.config.pos_weight_multiplier], device=labels.device)
            
            # Calculate loss with optional pos_weight
            loss = F.binary_cross_entropy_with_logits(
                preds,
                labels_float,
                pos_weight=pos_weight
            )
        
            # Compute metrics

            with torch.no_grad():
                preds_proba = torch.sigmoid(preds)
                preds_squeezed = preds_proba.squeeze()
                
                # Calculate primary metrics
                val_auc = self.val_auc(preds_squeezed, labels)
                val_accuracy = self.val_accuracy(preds_squeezed, labels)
                val_mcc = self.val_mcc(preds_squeezed, labels)  # Compute MCC here
                
                metrics = {
                    'val_loss': loss,
                    'val_auc': val_auc,
                    'val_mcc': val_mcc,
                    'val_accuracy': val_accuracy
                }
                
                self.log_dict(metrics, on_epoch=True, prog_bar=True, logger=True, 
                              batch_size=labels.size(0))
        else:
            loss = F.mse_loss(preds, labels_float)
            self.log_dict({
                'train_loss': loss,
                'train_rmse': torch.sqrt(loss)
            }, on_step=False, on_epoch=True, prog_bar=True, 
               logger=True, batch_size=labels.size(0))
        
        
        # Only clean up at the same frequency as DataModule
        if self.trainer and hasattr(self.trainer, 'datamodule'):
            batch_size = self.trainer.datamodule.batch_size
            if batch_idx % (100 * (128 // batch_size)) == 0:  # Scale cleanup frequency with batch size
                torch.cuda.empty_cache()
        
        return loss






    def validation_step(self, batch, batch_idx):
        """Optimized validation step with correct trial metric storage."""
        try:
            graphs, labels = batch
            preds, weights = self(graphs)
            labels_float = labels.float().unsqueeze(1)
            
            if self.classification:
                loss = F.binary_cross_entropy_with_logits(preds, labels_float)
                
                with torch.no_grad():
                    preds_proba = torch.sigmoid(preds)
                    preds_squeezed = preds_proba.squeeze()
                    
                    # Calculate primary metrics
                    val_auc = self.val_auc(preds_squeezed, labels)
                    val_accuracy = self.val_accuracy(preds_squeezed, labels)
                    val_mcc = self.val_mcc(preds_squeezed, labels)  # Compute MCC here
                    
                    metrics = {
                        'val_loss': loss,
                        'val_auc': val_auc,  # old Primary metric
                        'val_mcc': val_mcc,  # new primary metric
                        'val_accuracy': val_accuracy
                    }
                    
                    # Log metrics for visualization
                    self.log_dict(metrics, on_epoch=True, prog_bar=True, logger=True, 
                                batch_size=labels.size(0))
                    
                    # Store metrics in trial user attributes at the end of validation
                    if self.trainer.state.stage == 'validate':
                        try:
                            # Get the Optuna trial object properly
                            if hasattr(self, 'trial') and hasattr(self.trial, 'set_user_attr'):
                                trial = self.trial
                            elif (hasattr(self.trainer, 'callbacks') and 
                                  any(hasattr(callback, 'trial') for callback in self.trainer.callbacks)):
                                trial = next(callback.trial 
                                           for callback in self.trainer.callbacks 
                                           if hasattr(callback, 'trial'))
                            else:
                                # If we can't find the trial object, log a debug message and return
                                self._logger.debug("No trial object found for metric storage")
                                return loss
                                
                            # Store each metric, converting tensors to float values
                            for metric_name, value in metrics.items():
                                if isinstance(value, torch.Tensor):
                                    value = value.item()
                                trial.set_user_attr(metric_name, value)
                            
                            # Store current epoch
                            trial.set_user_attr('best_epoch', self.trainer.current_epoch)
                            
                        except Exception as e:
                            self._logger.debug(f"Metric storage skipped: {str(e)}")
            else:
                # Regression case
                loss = F.mse_loss(preds, labels_float)
                rmse = torch.sqrt(loss)
                
                metrics = {
                    'val_loss': loss,
                    'val_rmse': rmse  # Primary metric for regression
                }
                
                # Log metrics for visualization
                self.log_dict(metrics, on_epoch=True, prog_bar=True, logger=True, 
                             batch_size=labels.size(0))
                
                # Store metrics in trial user attributes at the end of validation
                if self.trainer.state.stage == 'validate':
                    try:
                        # Get the Optuna trial object properly
                        if hasattr(self, 'trial') and hasattr(self.trial, 'set_user_attr'):
                            trial = self.trial
                        elif (hasattr(self.trainer, 'callbacks') and 
                              any(hasattr(callback, 'trial') for callback in self.trainer.callbacks)):
                            trial = next(callback.trial 
                                       for callback in self.trainer.callbacks 
                                       if hasattr(callback, 'trial'))
                        else:
                            # If we can't find the trial object, log a debug message and return
                            self._logger.debug("No trial object found for metric storage")
                            return loss
                            
                        # Store each metric, converting tensors to float values
                        for metric_name, value in metrics.items():
                            if isinstance(value, torch.Tensor):
                                value = value.item()
                            trial.set_user_attr(metric_name, value)
                        
                        # Store current epoch
                        trial.set_user_attr('best_epoch', self.trainer.current_epoch)
                        
                    except Exception as e:
                        self._logger.debug(f"Metric storage skipped: {str(e)}")
            
            return loss
            
        except Exception as e:
            self._logger.error(f"Error in validation step {batch_idx}: {str(e)}")
            raise



    def test_step(self, batch, batch_idx):
        """Test step with comprehensive metrics for final evaluation."""
        try:
            graphs, labels = batch
            preds, weights = self(graphs)
            labels_float = labels.float().unsqueeze(1)
            
            if self.classification:
                loss = F.binary_cross_entropy_with_logits(preds, labels_float)
                
                # For metrics, apply sigmoid and squeeze predictions
                with torch.no_grad():
                    preds_proba = torch.sigmoid(preds)
                    preds_squeezed = preds_proba.squeeze()
                    
                    # Compute all metrics at once
                    metrics = {
                        'test_loss': loss,
                        'test_accuracy': self.test_accuracy(preds_squeezed, labels),
                        'test_auc': self.test_auc(preds_squeezed, labels),
                        'test_prc': self.test_prc(preds_squeezed, labels),
                        'test_f1': self.test_f1(preds_squeezed, labels),
                        'test_precision': self.test_precision(preds_squeezed, labels),
                        'test_recall': self.test_recall(preds_squeezed, labels),
                        'test_specificity': self.test_specificity(preds_squeezed, labels),
                        'test_mcc': self.test_mcc(preds_squeezed, labels)
                    }
    
                    # Add metric descriptions (logged only once at the start of testing)
                    if batch_idx == 0:
                        metric_descriptions = {
                            'test_accuracy': 'Classification Accuracy',
                            'test_auc': 'Area Under ROC Curve',
                            'test_prc': 'Precision-Recall AUC',
                            'test_f1': 'F1 Score',
                            'test_precision': 'Precision',
                            'test_recall': 'Recall/Sensitivity',
                            'test_specificity': 'Specificity',
                            'test_mcc': 'Matthews Correlation Coefficient'
                        }
                        self._logger.info("Test Metrics Description:")
                        for metric, desc in metric_descriptions.items():
                            self._logger.info(f"{metric}: {desc}")
    
                    # Log all metrics at once
                    self.log_dict(
                        metrics,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        logger=True,
                        batch_size=labels.size(0)
                    )
            
            else:
                # Regression task
                loss = F.mse_loss(preds, labels_float)
                test_rmse = torch.sqrt(loss)
                
                # Compute all regression metrics
                metrics = {
                    'test_loss': loss,
                    'test_rmse': test_rmse,
                    'test_mae': self.test_mae(preds, labels_float),
                    'test_mse': self.test_mse(preds, labels_float),
                    'test_r2': self.test_r2(preds, labels_float)
                }
                
                # Add metric descriptions for regression (only once)
                if batch_idx == 0:
                    metric_descriptions = {
                        'test_rmse': 'Root Mean Square Error',
                        'test_mae': 'Mean Absolute Error',
                        'test_mse': 'Mean Squared Error',
                        'test_r2': 'R2 Score'
                    }
                    self._logger.info("Test Metrics Description:")
                    for metric, desc in metric_descriptions.items():
                        self._logger.info(f"{metric}: {desc}")
                
                # Log all metrics at once
                self.log_dict(
                    metrics,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    batch_size=labels.size(0)
                )
            
            if batch_idx % 50 == 0:  # Log memory usage periodically
                self._log_memory_usage(f"Test Step {batch_idx}")
            
            return loss
                
        except Exception as e:
            self._logger.error(f"Error in test step {batch_idx}: {str(e)}")
            raise
        finally:
            if batch_idx % 50 == 0:  # Clean up memory periodically
                torch.cuda.empty_cache()
                gc.collect()


