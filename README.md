# XAI for Molecular Bioactivity Prediction

A comprehensive machine learning pipeline for predicting and explaining molecular bioactivity using Graph Neural Networks with integrated interpretability through substructure analysis.

## Project Overview

This project implements an advanced approach to molecular bioactivity prediction that combines state-of-the-art Graph Neural Networks with robust interpretability methods. The pipeline supports both classification and regression tasks, making it suitable for various molecular property prediction challenges. A key feature is its ability to provide explanations for predictions through substructure analysis and attribution scoring.

## Project Structure

The project is organized into three main components that work together to create an end-to-end pipeline for molecular property prediction and interpretation:

```
.
├── Data Processing
│   ├── prep_data.py        # Orchestrates the entire data preparation pipeline
│   │                       # Handles input validation, preprocessing, and output management
│   │
│   ├── build_data.py       # Core molecular graph construction functionality
│   │                       # Implements BRICS, Murcko, and functional group analysis
│   │                       # Contains optimized graph building algorithms
│   │
│   └── data_module.py      # PyTorch Lightning data handling
│                          # Implements efficient data loading and batching
│                          # Manages memory optimization for large datasets
│
├── Model Development
│   ├── model.py            # GNN model architecture implementation
│   │                       # Contains RGCN layers, attention mechanisms
│   │                       # Implements classification and regression heads
│   │
│   ├── hyper.py           # Hyperparameter optimization framework
│   │                       # Manages Optuna trials and study persistence
│   │                       # Implements early stopping and pruning logic
│   │
│   ├── stat_val.py        # Comprehensive statistical validation
│   │                       # Implements k-fold cross-validation
│   │                       # Handles significance testing and effect sizes
│   │
│   └── final_eval.py      # Final model evaluation and selection
│                          # Manages ensemble creation and validation
│                          # Generates comprehensive performance reports
│
└── Explainability
    ├── ordinal_analysis.py # Regression confidence calibration
    │                       # Implements precise fold-difference confidence levels:
    │                       # A: Within 1.5-fold (highest confidence)
    │                       # B: Within 3.0-fold (good confidence)
    │                       # C: Within 10.0-fold (moderate confidence)
    │                       # D: Beyond 10-fold (low confidence)
    │                       # Quantifies prediction reliability
    │
    ├── scaffold_reg.py     # Regression-specific scaffold analysis
    │                       # Analyzes scaffold contributions to numeric predictions
    │                       # Handles confidence-weighted attributions
    │
    ├── property_analysis.py # Physicochemical property analysis
    │                       # Analyzes scaffold chemical properties
    │                       # Correlates properties with predictions
    │
    └── xai_antimicrobial_helper.py # XAI visualization helper
                                   # Provides ensemble prediction and attribution averaging
                                   # Generates visualizations for publications
```

Each component is designed to work independently while maintaining seamless integration with the overall pipeline. The data processing modules prepare molecular graphs that feed into the model development modules, which in turn generate predictions and attributions for the explainability modules to analyze.

## Key Features

### Molecular Graph Construction and Analysis
- **Multiple Substructure Types:**
  - BRICS decomposition for synthetically relevant fragments
  - Murcko scaffolds for core structure analysis
  - Functional group detection
  - Primary molecular graphs
- **Optimized Memory Handling:**
  - Efficient batch processing of large datasets
  - Smart caching mechanisms
  - Memory-mapped file loading for large datasets

### Advanced Model Architecture
- **Relational Graph Convolutional Network (RGCN):**
  - Edge type-aware convolutions
  - Adaptive feature aggregation
  - Residual connections and batch normalization
- **Configurable Neural Network Components:**
  - Customizable RGCN hidden features
  - Flexible feed-forward network architecture
  - Dropout regularization with optimized rates
  - Substructure masking for attribution analysis

### Robust Training Pipeline
- **Automated Hyperparameter Optimization:**
  - Optuna-based Bayesian optimization
  - Parallel trial execution
  - Early stopping and pruning
  - Configuration persistence and recovery
- **Statistical Validation:**
  - k-fold cross-validation with repetition
  - Statistical significance testing
  - Effect size calculations
  - Confidence calibration for predictions

### Interpretability Features
- **Substructure Attribution Analysis:**
  - Quantitative contribution scoring
  - Confidence level assessment
  - Property analysis of significant scaffolds
  - Visual analysis of important fragments
  - Interactive visualization via `xai_antimicrobial_helper.py`
- **Ordinal Analysis for Regression:**
  - Confidence level categorization (A/B/C/D)
  - Property-based binning
  - Attribution confidence scoring
- **Multi-pathogen XAI Visualization:**
  - Ensemble prediction and attribution averaging
  - Internal consistency assessment (prediction-explanation agreement)
  - Interactive plots for Jupyter Notebooks


## Getting Started

### Repository and Installation

First, you will need to clone the repository and set up your environment. This project uses Python and several scientific computing libraries. Here are the detailed installation steps:

```bash
# Clone the repository
git clone https://github.com/catenate15/XAI_Bioactivity_prediction.git

# Navigate to project directory
cd XAI_Bioactivity_prediction

# Create and activate virtual environment
python -m venv venv

# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

### Pre-trained Models

Due to their large size, pre-trained model checkpoints are not included in this repository. They can be downloaded from Zenodo:

[Zenodo Link to Pre-trained Models] (Please replace with the actual link)

```



### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB RAM minimum

### Set Up Environment
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

# For CPU-only installation
pip install -r requirements.txt

# For GPU support (recommended)
pip install -r requirements.txt torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cu111.html

### Dependencies
Core requirements:
- PyTorch >= 1.9.0
- RDKit >= 2021.03.1
- PyTorch Geometric >= 2.0.0
- Optuna >= 2.10.0
- scikit-learn >= 0.24.2

## Usage Guide

### 1. Data Preparation
```bash
python prep_data.py \
    --task_name your_task \
    --dataset_path data/input.csv \
    --output_dir processed_data \
    --substructure_types primary brics murcko fg \
    --batch_size 32 \
    --force_rebuild  # Optional: rebuild all data
```

Required CSV format:
```text
PROCESSED_SMILES,COMPOUND_ID,TARGET/logMIC,group
CCc1ccc(CC)cc1,COMP001,1,training
CCCc1ccccc1,COMP002,0,training
...
```

### 2. Model Development

#### Hyperparameter Optimization
To run hyperparameter optimization, simply execute the `hyper.py` script. The number of trials and other optimization parameters are configured within `config.py`.
```bash
python hyper.py
```

#### Statistical Validation
To perform statistical validation (repeated k-fold cross-validation), execute the `stat_val.py` script. Cross-validation parameters are configured within `config.py`.
```bash
python stat_val.py
```

#### Final Evaluation
To perform the final model evaluation and ensemble analysis, execute the `final_eval.py` script. Ensemble size and evaluation metrics are configured internally.
```bash
python final_eval.py
```

### 3. Prediction and Analysis

#### Making Predictions
The `predict.py` script is used to make predictions on new data, with options for ensemble type and XAI analysis.
```bash
# Using best 5 models (default) with basic predictions
python predict.py \
    --input new_data.csv \
    --output predictions.csv \
    --ensemble best5 \
    --batch_size 32 \
    --prediction_threshold 0.5 \
    --substructure_type None # No XAI analysis for faster prediction

# Using full ensemble with Murcko scaffold XAI and resuming from checkpoint
python predict.py \
    --input large_new_data.csv \
    --output predictions_with_xai.csv \
    --ensemble full \
    --batch_size 64 \
    --prediction_threshold 0.5 \
    --substructure_type murcko \
    --resume \
    --num_workers 8 \
    --start_idx 0 \
    --end_idx 1000 # Process a subset of data
```
- `--input`: Path to the input CSV file containing SMILES strings and COMPOUND_IDs.
- `--output`: Path to save the prediction results.
- `--ensemble`: Type of ensemble to use (`best5` (default) or `full`).
- `--batch_size`: Number of molecules to process in parallel.
- `--prediction_threshold`: Classification threshold for binary predictions (if applicable).
- `--substructure_type`: Type of substructure analysis for XAI (`brics`, `murcko`, `fg`, or `None` to skip).
- `--start_idx`, `--end_idx`: Optional. Specify a range of molecules to process from the input file.
- `--resume`: Optional. If set, attempts to resume from a previous checkpoint.
- `--num_workers`: Number of worker processes for parallel processing.

#### Explainability Analysis

After generating predictions with `--substructure_type` enabled in `predict.py`, you can perform further analysis:

##### Regression Analysis (e.g., for logMIC predictions)

To analyze scaffolds with confidence weighting:
```bash
python scaffold_reg.py \
    --input predictions.csv \
    --type brics \
    --weights "A:1.0,B:0.7,C:0.3,D:0.1" \
    --output scaffold_analysis_results
```
- `--input`: Path to the predictions CSV file (output from `predict.py`).
- `--type`: Scaffold decomposition type used (`brics` or `murcko`).
- `--weights`: Optional. Custom confidence weights (e.g., "A:1.0,B:0.7").
- `-o`, `--output`: Output directory to save analysis results.

To perform ordinal analysis and visualize prediction confidence:
```bash
python ordinal_analysis.py \
    predictions.csv \
    --output_dir confidence_analysis_results
```
- `input_csv`: Positional argument. Path to the predictions CSV file.
- `--output_dir`: Output directory to save plots and metrics.

##### Physicochemical Property Analysis

To analyze physicochemical properties of different scaffold categories:
```bash
python property_analysis.py \
    positive_scaffolds.csv \
    negative_scaffolds.csv \
    neutral_scaffolds.csv \
    --output property_analysis_results
```
- `positive`, `negative`, `neutral`: Positional arguments. Paths to CSV files containing scaffolds categorized by their effect on activity.
- `-o`, `--output`: Output directory to save plots and reports.

##### XAI Visualization Helper

The `xai_antimicrobial_helper.py` script provides interactive visualization tools for analyzing individual molecules in a Jupyter Notebook or similar environment.

Example usage in a Python script or Jupyter Notebook:
```python
from xai_antimicrobial_helper import analyze_molecule, EnsembleModelManager

# Initialize the manager once
manager = EnsembleModelManager(project_root='./') # Adjust project_root if needed

# Analyze a single molecule for a specific pathogen
smiles = "CC(=O)Oc1ccccc1C(=O)O" # Aspirin
results = analyze_molecule(smiles, pathogens=['SA'], manager=manager)

# Analyze a molecule for multiple pathogens
smiles_multi = "CCOc1ccc(CC)cc1"
results_multi = analyze_molecule(smiles_multi, pathogens=['SA', 'EC', 'CA'], show_attributions=True, manager=manager)

# For batch prediction without visualization
from xai_antimicrobial_helper import batch_predict
smiles_list = ["CCOc1ccc(CC)cc1", "CC(=O)Oc1ccccc1C(=O)O"]
batch_df = batch_predict(smiles_list, pathogens=['SA', 'EC'])
print(batch_df.head())
```

## Advanced Configuration

### Memory Optimization
The pipeline includes several memory optimization features:
```python
# In config.py
self.training_optimization = {
    'num_workers': min(os.cpu_count(), 8),
    'pin_memory': True,
    'persistent_workers': True,
    'prefetch_factor': 2,
    'precision': 16  # Mixed precision training
}
```

### Custom Functional Groups
Add or modify functional group SMARTS patterns:
```python
# In config.py
self.fg_with_ca_smart = [
    'CN(C)C(=O)C',           # N,N-dimethylacetamide
    'C(=O)O',                # Carboxylic acid
    # Add custom patterns here
]
```

### Hyperparameter Search Space
Customize the search space for optimization:
```python
# In config.py
self.rgcn_hidden_feats_choices = (
    "128-128-256",  # Best for complex tasks
    "256-256",      # Balanced option
    "64-128"        # Memory-efficient option
)
```

## Output Files

### Model Training
- `hyperparameter_results_{timestamp}.csv`: Optimization results
- `best_models.json`: Selected model configurations
- `{task}_cv_results/`: Cross-validation results and statistics

### Prediction Analysis
- `predictions.csv`: Model predictions with confidence scores
- `scaffold_statistics.csv`: Detailed scaffold analysis
- `property_analysis.csv`: Physicochemical property analysis

### Visualization
- `attribution_analysis.png`: Scaffold contribution visualization
- `confidence_distribution.png`: Prediction confidence analysis
- `property_trends.png`: Physicochemical property trends

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request



