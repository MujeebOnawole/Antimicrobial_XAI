# config.py

import os
import pandas as pd
import logging
import multiprocessing
from rdkit import Chem
import random
import numpy as np
import torch
from typing import Tuple, List, Dict, Any
from logger import LoggerSetup, get_logger

from sklearn.metrics import (
    accuracy_score,
    precision_score, 
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# Constants for metrics (Optional but recommended)
CLASSIFICATION_METRICS = [
    'accuracy', 'auc', 'f1', 'precision', 'recall', 
    'specificity', 'mcc', 'recall_at_precision', 'tnr_at_recall'
]

REGRESSION_METRICS = [
    'rmse', 'mae', 'r2', 'relative_rmse', 'pearson_r'
]
# Get a basic logger for config initialization
logger = get_logger(__name__)
class Configuration:
    """
    Centralized configuration class.
    Aligns with the improved build_data.py structure and functionality.
    """

    def __init__(self):

        # -----------------------------
        # 1. Task Configuration
        # -----------------------------
        self.task_name = 'combined_Sa'#'name of your input csv file without the extension' # Name of the task ('combined_Sa',-SA; 'C_albicans'-CA and 'E_coli_model_data'-EC)
        self.classification = True # Set to True for classification tasks or False for regression tasks
        self.task_type = 'classification' if self.classification else 'regression'
        self.labels_name = 'TARGET' if self.classification else 'logMIC' 
        self.smiles_name = 'PROCESSED_SMILES'
        self.compound_id_name = 'COMPOUND_ID'
        self.dataset_origin = None
        # EDGE TYPE CONFIGURATION
        # The model uses a 3-edge RGCN design: SINGLE=1, DOUBLE=2, TRIPLE=3
        # Aromatic bonds are assigned type -1 and remapped to SINGLE at runtime.
        # Aromaticity is learned from node features (atom.GetIsAromatic()).
        #
        # num_edge_types=4 is the clean setting for NEW training.
        # Legacy checkpoints (trained with 65) are auto-detected by the helper
        # which reads weight dimensions and adjusts accordingly.
        self.num_edge_types = 4
        self.num_classes = None
        self.pos_weight_multiplier = None 
        # Initialize logger as an instance attribute
        self.logger = get_logger(__name__)
        self.logger.info("Starting configuration initialization...")

 
        # Now, use self.logger instead of the global logger within the class
        self.logger.info("Logger initialized for Configuration class.")  

        # -----------------------------
        # 2. Data Paths and Structure
        # -----------------------------
        self.data_dir = '/workspace/data/graph_data'
        # Remove task_type from filename - we'll detect it from the data
        self.origin_data_path = os.path.join(self.data_dir, f'{self.task_name}.csv')
        self.output_dir = '/workspace/data/graph_data'
        os.makedirs(self.output_dir, exist_ok=True)



        
        # Add debug logging
        #self._log_paths()

        LoggerSetup.initialize(self.output_dir, self.task_name)
        # -----------------------------
        # 5. Substructure Configuration
        # -----------------------------
        self.substructure_types = ['primary']  
        self.min_substructure_size = 1
        self.max_substructure_size = 20
        self.max_substructure_issues_ratio = 0.1


        # Functional Groups SMARTS patterns (ensure no duplicates) 
        self.fg_with_ca_smart = [
            'CN(C)C(=O)C',           # N,N-dimethylacetamide
            'C(=O)O',                # Carboxylic acid
            'C(=O)OC',               # Ester
            'C(=O)[H]',              # Aldehyde
            'C(=O)N',                # Amide
            'C(=O)C',                # Ketone
            '[CH]=O',                # Aldehyde (alternative)
            'N=C=O',                 # Isocyanate
            'N=C=S',                 # Isothiocyanate
            '[N+](=O)[O-]',          # Nitro
            'N=O',                   # Nitroso
            'NO',                    # N-oxide
            'NC',                    # Primary amine
            'N=C',                   # Imine
            'N=NC',                  # Azo
            'N=N',                   # Diazo
            'N#N',                   # Azide
            'C#N',                   # Nitrile
            'S(=O)(=O)N',            # Sulfonamide
            'NS(=O)(=O)C',           # N-methylsulfonamide
            'S(=O)(=O)O',            # Sulfonic acid
            'S(=O)(=O)OC',           # Sulfonic ester
            'S(=O)(=O)C',            # Sulfone
            'S(=O)C',                # Sulfoxide
            'SC',                    # Thioether
            '[SH]',                  # Thiol
            '[F,Cl,Br,I]',           # Halogen
            'C(C)(C)C',              # t-butyl
            'C(F)(F)F',              # Trifluoromethyl
            'C#C[H]',                # Terminal alkyne
            'C1CC1',                 # Cyclopropyl
            'OCC',                   # Primary alcohol
            'OC',                    # Hydroxyl
            '[OH]',                  # Hydroxyl (alternative)
            '[NH2]',                 # Primary amine (alternative)
            '[CX4][OX2H]',           # Aliphatic hydroxyl groups
            'c[OX2H]',               # Aromatic hydroxyl groups
            '[OX2][CH3]',            # Methoxy groups -OCH3
            '[#6]=N[OH]',            # Oxime groups
            '[CX3](=O)[OX2H0][#6]',  # Esters
            '[CX3](=O)[OX2H1][CX4]', # Aliphatic carboxylic acids
            '[CX3](=O)[OX2H1][c]',   # Aromatic carboxylic acids
            '[CX3]=[OX1]',           # Carbonyl O
            '[#6][CX3](=O)[#6]',     # Ketones
            '[#6][OX2][#6]',         # Ether oxygens (including phenoxy)
            'c1ccccc1',              # Benzene rings
            '[NX4+]',                # Quaternary ammonium
            '[NX3H2]',               # Primary amines
            '[NX3H1][#6]',           # Secondary amines
            '[NX3H0]([#6])[#6]',     # Tertiary amines
            '[n]',                   # Aromatic nitrogens
            '[nH]',                  # Aromatic amines
            '[CX2]#N',               # Nitriles
            '[NX3](=O)[O-]',         # Nitro groups
            '[CX3](=O)[NX3][#6]',    # Amides
            '[CX3](=O)[NX3H2]',      # Primary amides
            '[CX4][F,Cl,Br,I]',      # Alkyl halides
            '[#16X2][#6]',           # Thioether
            '[#16X2H]',              # Thiol groups
            '[NX3][CX3](=O)[NX3]',   # Urea groups
            '[CX2H]#C',              # Terminal acetylenes
            '[CX4]',                 # Saturated carbons (e.g., alkanes)
            '[CX3]=[CX3]',           # Double-bonded carbons (alkenes)
            'c',                     # Aromatic carbons (e.g., benzene)
            '[CX3H1](=O)',           # Aldehydes (alternative)
            '[CX3](=O)[OX2][#6]',    # Ethers (alternative)
            '[OX2H]',                # Hydroxyl (general)
            '[OX2H1][CX3]=O',        # Carboxylic acid alternative
            '[CX3H0](=O)[OX2H1]',    # General carboxylic acids
            '[OX2H][CX3]=O',         # Acidic oxygen in carboxylic acids
            '[#7][CX3]=O',           # Amide (alternative)
            '[OX2][CX3H]',           # Ketone (alternative)
            '[#7X3][CX3]',           # Amide nitrogen alternative
            '[NX3H]',                # Amines, general alternative
            '[NX2]',                 # Nitrenes, general
            '[NX3H1]',               # Secondary amines alternative
            '[NX2H0]',               # Non-hydrogen bonded nitrogens
            '[NX1]',                 # Radical nitrogens
            '[SX2]',                 # Sulfur with two connections (alternative)
            '[CX3](=O)[SX1]',        # Thioesters
            '[#6][OX2]',             # Alkoxy groups (alternative to ether)
            '[CX4][NX3]',            # General alkylamines
            '[OX2][CX3H]',           # Secondary alcohols (alternative)
            '[NX3][CX4]',            # Tertiary amines (alternative)
            '[CX3](=O)[NX2]',        # Secondary amides alternative
        ]


        # Use the same patterns for without_ca version
        self.fg_without_ca_smart = self.fg_with_ca_smart.copy()


        # Validate SMARTS patterns upon initialization
        self._validate_smarts()
        # -----------------------------
        # 6. Logging Configuration
        # -----------------------------
        self.log_dir = os.path.join('logs', self.task_name)  # Directory for logs
        os.makedirs(self.log_dir, exist_ok=True)  # Create directory if it doesn't exist




        
        #  checkpoint directory
        #self.checkpoint_dir = os.path.join('checkpoints', self.task_name)
        #os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Add training optimization parameters
        self.training_optimization = {
            # DataLoader optimizations
            'num_workers': min(os.cpu_count(), 8),  # Default to 8 workers max
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 2,
            
            # Training optimizations
            'precision': 16,  # Use mixed precision by default
            'accumulate_grad_batches': 1,
            'benchmark': True,
            'deterministic': False,  # Disable for speed during hyperparameter search
            
            # Memory optimizations
            'dataset_in_memory': True,  # Try to keep dataset in GPU memory
            
            # Validation optimization
            'val_check_interval': 0.5,  # Validate less frequently during hyperparameter search
            'limit_val_batches': 1.0,  # Use full validation set by default
        }
        
        # Add performance monitoring
        self.enable_profiling = False  # Enable for debugging performance issues
        self.profile_output_dir = os.path.join(self.output_dir, 'profiling')
        
        #GPU memory optimization
        self.training_optimization.update({
            'gradient_clip_val': 1.0,  # Prevent exploding gradients
            'detect_anomaly': False,  # Disable anomaly detection for speed
            'sync_batchnorm': False,  # Disable synchronized batch norm
            'replace_sampler_ddp': False,  # Disable DDP sampler replacement
        })
        
        
        self.training_optimization.update({
            'move_metrics_to_cpu': True,  # Move metrics to CPU to free GPU memory
            'enable_progress_bar': False,  # Disable progress bar for faster training
            'enable_model_summary': False,  # Disable model summary for faster startup
            'precision': os.getenv('PRECISION', '16'),
            'num_workers': int(os.getenv('NUM_WORKERS', min(os.cpu_count(), 8))),
        })
        


        # -----------------------------
        # 8. Hyperparameter Search Space Definitions
        # -----------------------------

        

        # Updated hyperparameter search space based on latest COADD and previous ChEMBL results
        self.rgcn_hidden_feats_choices = (
            "128-128-256",  # Best performer in COADD (val_auc: 0.896)
            "256-256",      # Strong performer in both COADD (0.885) and ChEMBL
            "64-128"        # Good balance option, performed well in previous experiments
        )
        
        self.ffn_hidden_feats_choices = ("32", "64")  # Both showed strong performance in COADD
        
        # mapping dictionaries 
        self.rgcn_hidden_feats_map = {
            "128-128-256": (128, 128, 256),
            "256-256": (256, 256),
            "64-128": (64, 128)
        }
        
        self.ffn_hidden_feats_map = {
            "32": 32,
            "64": 64
        }
        
        # Dropout ranges focused on best performing values from COADD results
        self.ffn_dropout_choices = (0.1, 0.2)    # 0.1 dominated top results
        self.rgcn_dropout_choices = (0.1, 0.4)   # Both 0.1 and 0.4 showed good performance
        
        # Learning rate range based on top performing configurations
        self.lr_min = 5e-4   # Based on successful rates (~0.000501)
        self.lr_max = 1.5e-3 # Covering best performer (0.001203)
        
        # Weight decay range from successful trials
        self.weight_decay_min = 2e-6  # Around 0.000002
        self.weight_decay_max = 1e-5  # Around 0.000009
        
        # Reduced trials with focused search space
        self.n_trials = 15  # Keeping focused search space

        # -----------------------------
        # 9. Hyperparamater tuning and cross validation Configurations
        # -----------------------------    

        # Training Configuration for Robustness Testing
        self._optuna_max_epochs = 100  # Number of epochs for optuna optimization
        self._robustness_max_epochs = 150  # Number of epochs for robustness testing
        self._current_context = 'optuna'  # Default context 
        self.early_stopping_patience = 15  # Patience for early stopping
        
        # Ensemble Prediction Configurations
        self.ensemble_num_models = 10  # Utilize all 10 seeds
        
        # Seeds Configuration
        self.seed_list = [2, 17]   # using 3 seed based on prime numbers as they are jointly used in cross validation

        
        # -----------------------------
        # 15. Maximum Substructures Configuration
        # -----------------------------
        self.max_substructures = 20  # Adjust based on your requirements


        # -----------------------------
        # 10. Validation Parameters
        # -----------------------------
        self.expected_groups = {'training', 'valid', 'test'}
        self.max_molecule_size = 150  # Maximum number of atoms before warning
        self.max_memory_usage_gb = 10  # Maximum estimated memory usage before warning

        # -----------------------------
        # 11. Graph Feature Dimensions
        # -----------------------------
        self.num_node_features = 40  # Based on atom feature vector size

        # -----------------------------
        # 12. File Paths Generator
        # -----------------------------
        self.file_paths = self._generate_file_paths()

        # -----------------------------
        # 13. Weight Handling Configuration
        # -----------------------------
        self.use_pos_weight = True  # Whether to use positive class weights for imbalanced data
        self.use_weighted_sampler = True  # Whether to use weighted sampling during training
        self.class_weight_strategy = 'balanced'  # Strategy for computing class weights
        self.pos_weight_multiplier = 1.0  # Multiplier for positive class weight
        

        # -----------------------------
        # 14. Training and Optimization Configuration
        # -----------------------------
        # Base parameters (used as defaults and for adaptation)
        self.base_lr = 1e-3
        self.base_weight_decay = 1e-4
        self.base_batch_size = 128 #32
        self.base_pos_weight_multiplier = 1.0
    
        # Current parameters (will be set by analyze_dataset or manually)
        self.lr = self.base_lr
        self.weight_decay = self.base_weight_decay
        self.batch_size = int(os.getenv('BATCH_SIZE', self.base_batch_size))
        self.pos_weight_multiplier = self.base_pos_weight_multiplier
        self.num_workers = int(os.getenv('NUM_WORKERS', 0))
    
        # Adaptive training thresholds
        self.mild_imbalance_threshold = 0.3
        self.severe_imbalance_threshold = 0.1
    
        # Focal loss configuration
        self.use_focal_loss = False
        self.focal_gamma = None
        self.focal_alpha = None
    
        # Optimizer configuration
        self.optimizer_type = 'adam'
        self.use_weighted_sampler = True
        self.class_weight_strategy = 'balanced'
    
        # Learning rate scheduler
        self.scheduler_factor = 0.5
        self.scheduler_patience = 5
        self.scheduler_min_lr = 1e-6
        self.scheduler_mode = 'max' if self.classification else 'min'
        self.scheduler_threshold = 1e-4
        self.scheduler_verbose = True
    
        # Early stopping
        self.early_stop_patience = 10
        self.early_stop_min_delta = 1e-4
    
        # Other training parameters
        self.use_chirality = True
        self.train_labels = None


        # -----------------------------
        # 15. Statistical validation configuration
        # -----------------------------        

        self.statistical_validation = {
            'cv_repeats': 3,
            'cv_folds': 5,
            'statistical_test': 'tukey_hsd',
            'effect_size_threshold': 0.5,
            'classification_metrics': CLASSIFICATION_METRICS,
            'regression_metrics': REGRESSION_METRICS
            }

        # Define CV output directory with task type
        self.cv_dir = os.path.join(
            self.output_dir, 
            f'{self.task_name}_{self.task_type}_cv_results'
        )
        os.makedirs(self.cv_dir, exist_ok=True)

        # Add validation metrics functionality
        self.metric_functions = self._initialize_metric_functions()
        
        

        # Default classification parameters
        self.prediction_threshold = 0.5  # Set a default threshold
        self.use_focal_loss = False
        self.focal_gamma = None
        self.focal_alpha = None
        self.pos_weight_multiplier = None
    
    def set_task_specific_params(self):
        """Set parameters based on task type"""
        if self.task_type == 'classification':
            # Set classification-specific parameters with default threshold
            self.prediction_threshold = 0.5  # Default threshold that will be optimized by stat_val
            self.use_focal_loss = False
            self.focal_gamma = 2.0
            self.focal_alpha = 0.25
            self.pos_weight_multiplier = 1.0
        else:
            # Set regression parameters
            self.prediction_threshold = None  # Not used for regression
            self.use_focal_loss = False
            self.focal_gamma = None
            self.focal_alpha = None
            self.pos_weight_multiplier = None

    def _initialize_metric_functions(self) -> Dict:
        """Initialize dictionary of metric functions based on task type"""
        if self.classification:
            return {
                'accuracy': accuracy_score,
                'precision': precision_score,
                'recall': recall_score,
                'f1': f1_score,
                'mcc': matthews_corrcoef,
                'auc': roc_auc_score,
                'average_precision': average_precision_score
            }
        else:
            return {
                'rmse': lambda y, yhat: np.sqrt(mean_squared_error(y, yhat)),
                'mae': mean_absolute_error,
                'r2': r2_score,
                'relative_rmse': lambda y, yhat: np.sqrt(mean_squared_error(y, yhat)) / np.mean(np.abs(y)),
                'pearson_r': lambda y, yhat: np.corrcoef(y, yhat)[0, 1]
            }

    def get_metric_function(self, metric_name: str):
        """Get metric function by name"""
        return self.metric_functions.get(metric_name)

    def validate_metrics(self, metrics: Dict[str, float]) -> bool:
        """Validate metrics against thresholds"""
        if self.classification:
            required_metrics = self.statistical_validation['classification_metrics']
        else:
            required_metrics = self.statistical_validation['regression_metrics']

        # Check if all required metrics are present
        missing_metrics = [m for m in required_metrics if m not in metrics]
        if missing_metrics:
            self.logger.warning(f"Missing required metrics: {missing_metrics}")
            return False

        # Validate against thresholds
        thresholds = self.statistical_validation['metric_thresholds']
        for metric, value in metrics.items():
            min_key = f'min_{metric}'
            max_key = f'max_{metric}'
            if min_key in thresholds and value < thresholds[min_key]:
                return False
            if max_key in thresholds and value > thresholds[max_key]:
                return False

        return True


    @property
    def max_epochs(self):
        """Returns appropriate max_epochs based on context"""
        if self._current_context == 'robustness':
            return self._robustness_max_epochs
        return self._optuna_max_epochs

    def set_context(self, context: str):
        """Set the current context to determine which max_epochs to use"""
        valid_contexts = ['optuna', 'robustness']
        if context not in valid_contexts:
            raise ValueError(f"Invalid context. Must be one of: {valid_contexts}")
        self._current_context = context
        logger.info(f"Context set to: {context}, max_epochs: {self.max_epochs}")       


    def optimize_for_hyperparameter_search(self):
        """
        Adjust configuration for faster hyperparameter search.
        Call this before starting hyperparameter optimization.
        """
        # Optimize batch size if not explicitly set
        if not hasattr(self, '_original_batch_size'):
            self._original_batch_size = self.batch_size
            self.batch_size = min(128, self._original_batch_size * 2)
        
        # Optimize validation frequency
        self.training_optimization['val_check_interval'] = 0.5
        self.training_optimization['limit_val_batches'] = 0.5
        
        # Enable speed optimizations
        self.training_optimization['benchmark'] = True
        self.training_optimization['deterministic'] = False
        
        # Update early stopping for faster trials
        self._original_early_stop_patience = self.early_stop_patience
        self.early_stop_patience = max(5, self.early_stop_patience // 2)
        
        logger.info("Configuration optimized for hyperparameter search:")
        logger.info(f"- Batch size: {self.batch_size} (original: {self._original_batch_size})")
        logger.info(f"- Early stopping patience: {self.early_stop_patience} (original: {self._original_early_stop_patience})")
        logger.info(f"- Validation check interval: {self.training_optimization['val_check_interval']}")
        logger.info(f"- Validation batches: {self.training_optimization['limit_val_batches'] * 100}%")
    
    def restore_original_configuration(self):
        """
        Restore original configuration settings after hyperparameter search.
        """
        if hasattr(self, '_original_batch_size'):
            self.batch_size = self._original_batch_size
            delattr(self, '_original_batch_size')
            
        if hasattr(self, '_original_early_stop_patience'):
            self.early_stop_patience = self._original_early_stop_patience
            delattr(self, '_original_early_stop_patience')
            
        self.training_optimization['val_check_interval'] = 1.0
        self.training_optimization['limit_val_batches'] = 1.0
        self.training_optimization['deterministic'] = True
        
        logger.info("Restored original configuration settings")
    


    def get_training_kwargs(self):
        return {
            'max_epochs': self.max_epochs,
            'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
            'devices': 1,
            'precision': self.training_optimization['precision'],
            'accumulate_grad_batches': self.training_optimization['accumulate_grad_batches'],
            'benchmark': self.training_optimization['benchmark'],
            'deterministic': self.training_optimization['deterministic'],
            'val_check_interval': self.training_optimization['val_check_interval'],
            'limit_val_batches': self.training_optimization['limit_val_batches'],
            'enable_progress_bar': self.training_optimization.get('enable_progress_bar', True),
            'enable_model_summary': self.training_optimization.get('enable_model_summary', True),
            'gradient_clip_val': self.training_optimization.get('gradient_clip_val', 0.0),
            'detect_anomaly': self.training_optimization.get('detect_anomaly', False),
            'enable_checkpointing': False,  # Set to False here if desired
            # Do not include 'logger' here
        }

        
    def get_dataloader_kwargs(self):
        """
        Get kwargs for DataLoader based on current configuration.
        """
        return {
            'batch_size': self.batch_size,
            'num_workers': self.training_optimization['num_workers'],
            'pin_memory': self.training_optimization['pin_memory'],
            'persistent_workers': self.training_optimization['persistent_workers'],
            'prefetch_factor': self.training_optimization['prefetch_factor'],
        }


    def inspect_dataset(self):
        """
        Inspect the dataset to determine task type and validate columns.
        Respects the classification flag set in config.
        """
        try:
            import pandas as pd
            import numpy as np
            data = pd.read_csv(self.origin_data_path)
            
            # Check for required columns
            if self.smiles_name not in data.columns:
                raise ValueError(f"Required column {self.smiles_name} not found in dataset. "
                               f"Available columns: {', '.join(data.columns)}")
                
            if self.compound_id_name not in data.columns:
                raise ValueError(f"Required column {self.compound_id_name} not found in dataset. "
                               f"Available columns: {', '.join(data.columns)}")
    
            # Validate columns based on classification flag
            if self.classification:
                if 'TARGET' not in data.columns:
                    raise ValueError("'TARGET' column not found but classification=True in config")
                self.labels_name = 'TARGET'
            else:
                if 'logMIC' not in data.columns:
                    raise ValueError("'logMIC' column not found but classification=False in config")
                self.labels_name = 'logMIC'
    
            # Set task type based on classification flag
            self.task_type = 'classification' if self.classification else 'regression'
            
            # Store dataset
            self.dataset_origin = data
            
            logger.info(f"Dataset inspection complete:")
            logger.info(f"- Task type: {self.task_type}")
            logger.info(f"- Labels column: {self.labels_name}")
            logger.info(f"- Number of samples: {len(data)}")
            
            # Additional validation for classification task
            if self.classification:
                unique_labels = data[self.labels_name].unique()
                logger.info(f"- Unique labels: {unique_labels}")
                
                # Check if values are valid for classification
                valid_labels = all(
                    isinstance(x, (int, bool, np.integer, np.bool_)) or
                    (isinstance(x, (float, np.floating)) and float(x).is_integer())
                    for x in unique_labels
                )
                
                if not valid_labels:
                    invalid_labels = [x for x in unique_labels if not (
                        isinstance(x, (int, bool, np.integer, np.bool_)) or
                        (isinstance(x, (float, np.floating)) and float(x).is_integer())
                    )]
                    raise ValueError(
                        f"Classification labels must be integers or booleans. "
                        f"Found invalid values: {invalid_labels}"
                    )
                
                # Convert labels to integers if they're not already
                if not all(isinstance(x, (int, np.integer)) for x in unique_labels):
                    data[self.labels_name] = data[self.labels_name].astype(int)
                    logger.info("- Labels converted to integers")
                
                # Verify binary classification
                if len(unique_labels) != 2:
                    raise ValueError(
                        f"Binary classification requires exactly 2 classes. "
                        f"Found {len(unique_labels)} classes: {unique_labels}"
                    )
                
                # Sort unique labels to ensure consistent ordering
                sorted_labels = sorted(unique_labels)
                logger.info(f"- Binary classification labels: {sorted_labels}")
            else:
                # Validation for regression task
                if not data[self.labels_name].dtype.kind in 'fc':  # float or complex
                    raise ValueError(
                        f"Regression labels must be numeric. "
                        f"Found dtype: {data[self.labels_name].dtype}"
                    )
                logger.info(f"- Label range: [{data[self.labels_name].min():.2f}, {data[self.labels_name].max():.2f}]")
    
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Input file not found: {self.origin_data_path}\n"
                f"Please ensure the file exists and is named: {self.task_name}.csv\n"
                f"Expected path: {self.data_dir}"
            )
        except Exception as e:
            raise type(e)(f"Error inspecting dataset: {str(e)}")

    def get_processed_file_path(self, file_type: str, subtype: str = 'primary') -> str:
        """Get the path for a processed file with the correct naming convention."""
        # Now includes task type in the filename after it's been detected
        if not self.task_type:
            raise ValueError("Task type not set. Call inspect_dataset() first.")
            
        base_name = f"{self.task_name}_{self.task_type}_{subtype}"
        if file_type == 'meta':
            return os.path.join(self.output_dir, f"{base_name}_meta.csv")
        elif file_type == 'graphs':
            return os.path.join(self.output_dir, f"{base_name}_graphs.pt")
        elif file_type == 'smask':
            return os.path.join(self.output_dir, f"{base_name}_smask.npy")
        else:
            raise ValueError(f"Unknown file type: {file_type}")


    def get_processed_file_path(self, file_type: str, subtype: str = 'primary') -> str:
        """
        Get the path for a processed file with the correct naming convention.
        
        Args:
            file_type: Type of file ('meta', 'graphs', 'smask')
            subtype: Substructure type ('primary', 'brics', 'murcko', 'fg')
            
        Returns:
            Full path to the processed file
        """
        base_name = f"{self.task_name}_{self.task_type}_{subtype}"
        if file_type == 'meta':
            return os.path.join(self.output_dir, f"{base_name}_meta.csv")
        elif file_type == 'graphs':
            return os.path.join(self.output_dir, f"{base_name}_graphs.pt")
        elif file_type == 'smask':
            return os.path.join(self.output_dir, f"{base_name}_smask.npy")
        else:
            raise ValueError(f"Unknown file type: {file_type}")
    
    def _log_paths(self):
        """Log all important paths for debugging"""
        print("\nConfiguration Paths:")
        print(f"Task name: {self.task_name}")
        print(f"Task type: {self.task_type}")
        print(f"Data directory: {self.data_dir}")
        print(f"Original data path: {self.origin_data_path}")
        print(f"Output directory: {self.output_dir}")
        
        # Check for expected files
        meta_file = f"{self.task_name}_{self.task_type}_primary_meta.csv"
        graphs_file = f"{self.task_name}_{self.task_type}_primary_graphs.pt"
        
        expected_meta = os.path.join(self.output_dir, meta_file)
        expected_graphs = os.path.join(self.output_dir, graphs_file)
        
        print("\nExpected files:")
        print(f"Original data: {self.origin_data_path}")
        print(f"Original data exists: {os.path.exists(self.origin_data_path)}")
        print(f"Meta file: {expected_meta}")
        print(f"Meta file exists: {os.path.exists(expected_meta)}")
        print(f"Graphs file: {expected_graphs}")
        print(f"Graphs file exists: {os.path.exists(expected_graphs)}")
        
        if os.path.exists(self.output_dir):
            print("\nFiles in output directory:")
            for file in os.listdir(self.output_dir):
                print(f"- {file}")
        else:
            print("\nOutput directory does not exist!")



    def reset_to_base_parameters(self):
        """Reset adaptive parameters to their base values."""
        self.lr = self.base_lr
        self.weight_decay = self.base_weight_decay
        self.batch_size = self.base_batch_size
        self.pos_weight_multiplier = self.base_pos_weight_multiplier
        self.use_focal_loss = False
        self.focal_gamma = None
        self.focal_alpha = None
        self.scheduler_patience = 5
        self.early_stop_patience = 10




    def analyze_dataset(self, labels):
        """Analyze dataset and set appropriate parameters."""
        # Reset parameters to base values first
        self.reset_to_base_parameters()
        
        if self.classification:
            # Classification analysis
            unique_labels, counts = np.unique(labels, return_counts=True)
            total_samples = len(labels)
            n_classes = len(unique_labels)
            self.num_classes = len(unique_labels)
            self.logger.info(f"Detected {self.num_classes} classes in dataset")
            
            # Update configuration
            if self.num_classes != 2 and not hasattr(self, 'multi_class_strategy'):
                self.logger.warning(f"Found {self.num_classes} classes, but no multi-class strategy specified")
                
            # Store class distribution information
            self.class_distribution = {
                label: np.sum(labels == label) for label in unique_labels
            }              
            # Calculate imbalance ratio for classification
            min_count = np.min(counts)
            max_count = np.max(counts)
            imbalance_ratio = min_count / max_count
            
            logger.info(f"Dataset analysis:")
            logger.info(f"Total samples: {total_samples}")
            logger.debug(f"Class distribution: {dict(zip(unique_labels, counts))}")
            logger.info(f"Imbalance ratio: {imbalance_ratio:.3f}")
            
          
            # Adjust parameters based on imbalance ratio
            if imbalance_ratio < self.severe_imbalance_threshold:
                logger.info("Severe class imbalance detected. Adjusting parameters accordingly.")
                self._set_severe_imbalance_params(imbalance_ratio)
            elif imbalance_ratio < self.mild_imbalance_threshold:
                logger.info("Mild class imbalance detected. Adjusting parameters accordingly.")
                self._set_mild_imbalance_params(imbalance_ratio)
            else:
                logger.info("Dataset is relatively balanced. Using standard parameters.")
                self._set_balanced_params()
        else:
            # Regression analysis
            total_samples = len(labels)
            mean_value = np.mean(labels)
            std_value = np.std(labels)
            min_value = np.min(labels)
            max_value = np.max(labels)
            
            logger.info(f"Dataset analysis (Regression):")
            logger.info(f"Total samples: {total_samples}")
            logger.info(f"Value range: [{min_value:.3f}, {max_value:.3f}]")
            logger.info(f"Mean: {mean_value:.3f}")
            logger.info(f"Standard deviation: {std_value:.3f}")
            
            # For regression, we use standard parameters but adjust based on data distribution
            self._set_regression_params(mean_value, std_value, min_value, max_value)
        
        self._log_parameters()
    
    def _set_regression_params(self, mean, std, min_val, max_val):
        """Set parameters specifically for regression tasks."""
        # Use standard parameters as base
        self.lr = self.base_lr
        self.weight_decay = self.base_weight_decay
        self.batch_size = self.base_batch_size
        
        # Adjust learning rate based on value range
        value_range = max_val - min_val
        if value_range > 10:
            self.lr *= 0.1  # Reduce learning rate for large value ranges
        
        # Set regression-specific parameters
        self.scheduler_patience = 8  # More patience for regression
        self.early_stop_patience = 15
        
        # Disable classification-specific parameters
        self.use_focal_loss = False
        self.focal_gamma = None
        self.focal_alpha = None
        self.pos_weight_multiplier = None
        self.prediction_threshold = None

    def _set_severe_imbalance_params(self, imbalance_ratio):
        """Set parameters for severely imbalanced datasets."""
        # More aggressive learning parameters for severe imbalance
        self.lr = self.base_lr * 0.7  # Less reduction in learning rate
        self.weight_decay = self.base_weight_decay  # Keep original regularization
        self.batch_size = max(8, self.base_batch_size // 4)  # Smaller batches
        
        # More aggressive class balancing
        # Calculate pos_weight based on actual class distribution
        neg_pos_ratio = (1 - imbalance_ratio) / imbalance_ratio
        self.pos_weight_multiplier = min(neg_pos_ratio * 1.5, 15.0)  # More aggressive, higher cap
        
        # Enable focal loss with adjusted parameters
        self.use_focal_loss = True
        self.focal_gamma = 3.0  # Increased focus on hard examples
        self.focal_alpha = 0.75  # More weight on positive class
        
        # Longer training
        self.scheduler_patience = 12
        self.early_stop_patience = 20
        
        # Adjust threshold for positive predictions
        self.prediction_threshold = 0.3  # Lbased on optimal threshold
        
        
    def _set_mild_imbalance_params(self, imbalance_ratio):
        """Set parameters for mildly imbalanced datasets."""
        self.lr = self.base_lr * 0.8
        self.weight_decay = self.base_weight_decay
        self.batch_size = max(16, self.base_batch_size // 2)
        
        # Balanced weighting
        neg_pos_ratio = (1 - imbalance_ratio) / imbalance_ratio
        self.pos_weight_multiplier = min(neg_pos_ratio * 1.2, 10.0)
        
        # Use focal loss with milder parameters
        self.use_focal_loss = True
        self.focal_gamma = 2.0
        self.focal_alpha = 0.6
        
        # Moderate patience
        self.scheduler_patience = 8
        self.early_stop_patience = 15
        
        # Slightly adjusted threshold
        self.prediction_threshold = 0.4
    
    def _set_balanced_params(self):
        """Set parameters for balanced datasets."""
        self.lr = self.base_lr
        self.weight_decay = self.base_weight_decay
        self.batch_size = self.base_batch_size
        self.pos_weight_multiplier = 1.0
        
        self.use_focal_loss = False
        self.focal_gamma = None
        self.focal_alpha = None
        
        self.scheduler_patience = 5
        self.early_stop_patience = 10
        self.prediction_threshold = 0.5
    
    def _log_parameters(self):
        """Log all current parameter settings."""
        logger = logging.getLogger(__name__)
        logger.info("\nCurrent parameter settings:")
        logger.info(f"Learning rate: {self.lr}")
        logger.info(f"Weight decay: {self.weight_decay}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Pos weight multiplier: {self.pos_weight_multiplier}")
        logger.info(f"Prediction threshold: {self.prediction_threshold}")
        logger.info(f"Using focal loss: {self.use_focal_loss}")
        if self.use_focal_loss:
            logger.info(f"Focal gamma: {self.focal_gamma}")
            logger.info(f"Focal alpha: {self.focal_alpha}")
        logger.info(f"Scheduler patience: {self.scheduler_patience}")
        logger.info(f"Early stop patience: {self.early_stop_patience}")


    def update_optimizer_config(self, **kwargs):
        """
        Update optimizer configuration parameters.
        
        Args:
            **kwargs: Keyword arguments for optimizer parameters
                - lr: Learning rate
                - weight_decay: Weight decay
                - optimizer_type: Type of optimizer
                - scheduler_factor: Factor to reduce learning rate
                - scheduler_patience: Patience for scheduler
                - scheduler_min_lr: Minimum learning rate
                - early_stop_patience: Patience for early stopping
                - early_stop_min_delta: Minimum delta for early stopping
        """
        valid_params = {
            'lr', 'weight_decay', 'optimizer_type', 
            'scheduler_factor', 'scheduler_patience', 'scheduler_min_lr',
            'scheduler_threshold', 'scheduler_verbose',
            'early_stop_patience', 'early_stop_min_delta'
        }
        
        for key, value in kwargs.items():
            if key not in valid_params:
                raise ValueError(f"Invalid parameter: {key}. Valid parameters are: {valid_params}")
            
            if key == 'lr' and not isinstance(value, (int, float)):
                raise ValueError("Learning rate must be a number")
            elif key == 'weight_decay' and not isinstance(value, (int, float)):
                raise ValueError("Weight decay must be a number")
            
            setattr(self, key, value)
            
        # Log updates
        logger = logging.getLogger(__name__)
        logger.info("Updated optimizer configuration:")
        logger.info(f"  Learning rate: {self.lr}")
        logger.info(f"  Weight decay: {self.weight_decay}")
        logger.info(f"  Optimizer type: {self.optimizer_type}")
        logger.info(f"  Scheduler parameters:")
        logger.info(f"    Factor: {self.scheduler_factor}")
        logger.info(f"    Patience: {self.scheduler_patience}")
        logger.info(f"    Min LR: {self.scheduler_min_lr}")
        logger.info(f"  Early stopping parameters:")
        logger.info(f"    Patience: {self.early_stop_patience}")
        logger.info(f"    Min delta: {self.early_stop_min_delta}")



    def get_optimizer_config(self) -> Dict[str, Any]:
        """
        Get current optimizer configuration.
        
        Returns:
            Dict containing current optimizer configuration parameters
        """
        return {
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'optimizer_type': self.optimizer_type,
            'scheduler_factor': self.scheduler_factor,
            'scheduler_patience': self.scheduler_patience,
            'scheduler_min_lr': self.scheduler_min_lr,
            'scheduler_threshold': self.scheduler_threshold,
            'scheduler_verbose': self.scheduler_verbose,
            'early_stop_patience': self.early_stop_patience,
            'early_stop_min_delta': self.early_stop_min_delta
        }

    def set_train_labels(self, labels):
        """Set the training labels for pos_weight calculation."""
        self.train_labels = labels

    def _generate_file_paths(self) -> Dict[str, Dict[str, str]]:
        """Generates all required file paths for each substructure type."""
        paths = {}
        for subtype in self.substructure_types:
            paths[subtype] = {
                'graphs': os.path.join(self.output_dir, f'{self.task_name}_{subtype}_graphs.pt'),
                'meta': os.path.join(self.output_dir, f'{self.task_name}_{subtype}_meta.csv'),
                'smask': os.path.join(self.output_dir, f'{self.task_name}_{subtype}_smask.npy')
            }
        return paths

    def get_substructure_paths(self, subtype: str) -> Dict[str, str]:
        """Gets all file paths for a specific substructure type."""
        if subtype not in self.substructure_types:
            raise ValueError(f"Invalid substructure type: {subtype}")
        return self.file_paths[subtype]

    def set_seed(self, seed: int = 42) -> None:
        """Sets random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False




    def validate_files(self) -> None:
        """
        Validates the existence of required files and directories.
        Checks data paths and directories, but allows for missing processed files
        during initial data preparation.
        """
        logger.info("Validating files and directories...")
        
        # Check directories
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
            
        if not os.path.exists(self.origin_data_path):
            raise ValueError(
                f"Original data file not found: {self.origin_data_path}\n"
                f"Looking in directory: {self.data_dir}\n"
                f"Available files: {os.listdir(self.data_dir)}"
            )
            
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Created output directory: {self.output_dir}")
    
        # Only check for processed files if they should exist
        if hasattr(self, 'task_type'):
            meta_file = f"{self.task_name}_{self.task_type}_primary_meta.csv"
            graphs_file = f"{self.task_name}_{self.task_type}_primary_graphs.pt"
            
            required_files = [
                (os.path.join(self.output_dir, meta_file), "meta"),
                (os.path.join(self.output_dir, graphs_file), "graphs")
            ]
            
            missing_files = []
            for file_path, file_type in required_files:
                if not os.path.exists(file_path):
                    missing_files.append((file_path, file_type))
            
            if missing_files:
                files_str = '\n'.join(f"- {file_type}: {path}" for path, file_type in missing_files)
                logger.warning(
                    f"Some processed files are missing:\n{files_str}\n"
                    f"These will be generated during data preparation."
                )
        
        logger.info("File validation completed")
    

    def _validate_smarts(self) -> None:
        """
        Validate all SMARTS patterns to ensure they are correctly formatted.
        Provides detailed error reporting for invalid patterns.
        """
        all_smarts = self.fg_with_ca_smart + self.fg_without_ca_smart
        invalid_patterns = []
        
        for smarts in all_smarts:
            mol = Chem.MolFromSmarts(smarts)
            if mol is None:
                invalid_patterns.append(smarts)
        
        if invalid_patterns:
            error_msg = "Invalid SMARTS patterns in configuration:\n" + \
                       "\n".join(f"- {pattern}" for pattern in invalid_patterns)
            raise ValueError(error_msg)
            
        logger.info("All SMARTS patterns are valid.")
    
    def validate_configuration(self) -> None:
        """
        Validates configuration settings and parameters.
        Checks SMARTS patterns, substructure settings, and other parameters.
        """
        # Validate SMARTS patterns first (using specialized validation)
        logger.info("Validating SMARTS patterns...")
        self._validate_smarts()
        
        # Continue with other configuration validations
        logger.info("Validating other configuration settings...")
        
        # Validate substructure configuration
        if self.min_substructure_size < 1:
            raise ValueError("min_substructure_size must be at least 1")
        if self.max_substructure_size < self.min_substructure_size:
            raise ValueError("max_substructure_size must be greater than min_substructure_size")
        
        # Validate batch size and workers
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
            
        # Validate substructure_types
        valid_substructures = {'primary', 'brics', 'murcko', 'fg'}
        if not set(self.substructure_types).issubset(valid_substructures):
            raise ValueError(f"substructure_types must be subset of {valid_substructures}")
        
        logger.info("Configuration validation completed successfully")
    
    def validate(self) -> None:
        """
        Performs comprehensive validation of all configuration aspects.
        Includes both file and configuration validation.
        """
        logger.info("Starting configuration validation...")
        
        try:
            # Validate files first
            logger.info("Validating files and directories...")
            self.validate_files()
            logger.info("File validation successful")
            
            # Then validate configuration settings
            logger.info("Validating configuration settings...")
            self.validate_configuration()
            logger.info("Configuration validation successful")
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise
        
        logger.info("All validation checks passed successfully")




    def update_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        """Updates the configuration with new hyperparameters."""
        if not isinstance(hyperparameters, dict):
            raise TypeError("Hyperparameters must be provided as a dictionary")
        
        # Handle rgcn_hidden_feats specially
        if 'rgcn_hidden_feats' in hyperparameters:
            value = hyperparameters['rgcn_hidden_feats']
            # Convert string selection to actual tuple/list
            if isinstance(value, str) and value in self.rgcn_hidden_feats_map:
                self.rgcn_hidden_feats = self.rgcn_hidden_feats_map[value]
            # Handle direct tuple/list assignment
            elif isinstance(value, (tuple, list)):
                self.rgcn_hidden_feats = value
            else:
                raise ValueError(f"Invalid rgcn_hidden_feats value: {value}")
        
        # Handle ffn_hidden_feats specially
        if 'ffn_hidden_feats' in hyperparameters:
            value = hyperparameters['ffn_hidden_feats']
            # Handle both string and direct value cases
            if isinstance(value, str) and value in self.ffn_hidden_feats_map:
                self.ffn_hidden_feats = [self.ffn_hidden_feats_map[value]]
            elif isinstance(value, (int, str)):
                self.ffn_hidden_feats = [int(value)]
            elif isinstance(value, (list, tuple)):
                self.ffn_hidden_feats = list(value)
            else:
                raise ValueError(f"Invalid ffn_hidden_feats value: {value}")

        # Handle dropout values
        for dropout_key in ['ffn_dropout', 'rgcn_dropout']:
            if dropout_key in hyperparameters:
                value = hyperparameters[dropout_key]
                if isinstance(value, (int, float)) and 0 <= value <= 1:
                    setattr(self, dropout_key, float(value))
                else:
                    raise ValueError(f"{dropout_key} must be a float between 0 and 1")

        # Handle learning rate
        if 'lr' in hyperparameters:
            value = hyperparameters['lr']
            if isinstance(value, (int, float)) and value > 0:
                self.lr = float(value)
            else:
                raise ValueError("Learning rate must be a positive float")

        # Handle weight decay
        if 'weight_decay' in hyperparameters:
            value = hyperparameters['weight_decay']
            if isinstance(value, (int, float)) and value >= 0:
                self.weight_decay = float(value)
            else:
                raise ValueError("Weight decay must be a non-negative float")
                
        # Handle prediction_threshold (use only when retraining with set of  hyperparameter)
#        if 'prediction_threshold' in hyperparameters:
#            value = hyperparameters['prediction_threshold']
#            if isinstance(value, (int, float)) and 0 <= value <= 1:
#                self.prediction_threshold = float(value)
#            else:
#                raise ValueError("prediction_threshold must be a float between 0 and 1")
        
        logger.info("Updated hyperparameters:")
        for key, value in hyperparameters.items():
            logger.info(f"  {key}: {value}")
            

# Create the config instance
def get_config():
    """Get configuration instance"""
    return Configuration()

# Create the config instance only if this file is run directly
if __name__ == '__main__':
    config = Configuration()
else:
    config = get_config()


