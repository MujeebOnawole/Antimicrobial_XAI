#predict.py

import os
import re
import torch
import numpy as np
import pandas as pd
import argparse
import logging
import traceback
import signal
from datetime import datetime
import time
import pickle
import math
from typing import Dict, Any, List, Tuple, Optional
from torch_geometric.loader import DataLoader as PyGDataLoader
from collections import deque
import itertools
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, SanitizeFlags, SanitizeMol
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
import concurrent.futures
import gc
import traceback
import itertools
from collections import deque
from rdkit.Chem import MolFragmentToSmiles
from torch_geometric.data import Batch, Data
import concurrent.futures
import sys
import json
from model import BaseGNN
from data_module import MoleculeDataset
from build_data import return_murcko_leaf_structure, return_brics_leaf_structure, return_fg_hit_atom ,get_scaffold_hierarchy, construct_mol_graph_from_smiles
from logger import get_logger
from config import Configuration   # use config3.py for regression and config2 for classification

# Silence RDKit kekulize warnings
RDLogger.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.warning')



logger = get_logger(__name__)
class PredictionManager:
    def __init__(self, config: Configuration, input_csv: str, device: Optional[str] = None, 
                num_workers: int = 4, ensemble_type: str = 'best5'):
        """Initialize PredictionManager with unique checkpoint file per input file."""
        # Initialize logger first
        self.logger = get_logger(__name__)
        self.logger.info("Initializing PredictionManager...")
        
        self.ensemble_type = ensemble_type
   
        # Store configuration and parameters
        self.config = config
        self.num_workers = num_workers
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

     
        
        # Generate unique checkpoint filename based on input file and current timestamp
        input_basename = os.path.splitext(os.path.basename(input_csv))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_dir = os.path.join(os.path.dirname(input_csv), 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create unique checkpoint filename using input file name and timestamp
        self.checkpoint_file = os.path.join(
            checkpoint_dir,
            f"prediction_checkpoint_{input_basename}.pkl"
        )
        self.logger.info(f"Using checkpoint file: {self.checkpoint_file}")
        
        # Initialize other attributes
        self.models = []
        self.should_terminate = False
        self.input_csv = input_csv  # Store input file path for reference
        
        # Initialize process pool for parallel processing
        if torch.cuda.is_available():
            torch.multiprocessing.set_start_method('spawn', force=True)
            
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Initialize functional group patterns if needed
        if hasattr(self.config, 'fg_with_ca_smart'):
            self._initialize_fg_patterns()
        
        # Set prediction mode
        self.config.prediction_mode = True
        
        # Load models
        self._load_models()
        
    def _initialize_fg_patterns(self):
        """Initialize functional group patterns."""
        if not hasattr(self.config, 'fg_with_ca_list'):
            self.config.fg_name_list = [f'fg_{i}' for i in range(len(self.config.fg_with_ca_smart))]
            self.config.fg_with_ca_list = [Chem.MolFromSmarts(s) for s in self.config.fg_with_ca_smart]
            self.config.fg_without_ca_list = [Chem.MolFromSmarts(s) for s in self.config.fg_without_ca_smart]

    def _signal_handler(self, signum, frame):
        """Handle termination signals gracefully."""
        self.logger.info("Termination signal received. Will exit after current batch.")
        self.should_terminate = True



    def _load_models(self):
        """Load models based on ensemble type."""
        if self.ensemble_type == 'full':
            return self._load_full_ensemble()
        else:
            return self._load_best_five()

    def _load_full_ensemble(self):
        """Load all models from CV runs."""
        cv_repeats = self.config.statistical_validation['cv_repeats']
        cv_folds = self.config.statistical_validation['cv_folds']
        
        self.logger.info(f"\nLoading all models from CV runs...")
        
        for cv_num in range(1, cv_repeats + 1):
            for fold in range(1, cv_folds + 1):
                try:
                    self._load_single_model(cv_num, fold)
                except Exception as e:
                    self.logger.error(f"Error loading model for CV{cv_num}, Fold{fold}: {str(e)}")
                    continue




    def _load_best_five(self):
        """Load best 5 models based on saved results."""
        # Search for best_models.json in the final_eval directory
        best_models_file = os.path.join(
            self.config.output_dir,
            f"{self.config.task_name}_{self.config.task_type}_final_eval",
            "best_models",
            'best_models.json'
        )
        
        try:
            with open(best_models_file, 'r') as f:
                json_data = json.load(f)
                
            if 'models' not in json_data:
                raise ValueError("JSON file does not contain 'models' key")
                
            best_models = json_data['models']
            self.logger.info(f"\nLoading top {len(best_models)} models...")
            
            for model_info in best_models:
                try:
                    if 'cv' not in model_info or 'fold' not in model_info:
                        raise ValueError(f"Model info missing required keys: {model_info}")
                        
                    cv_num = model_info['cv']
                    fold = model_info['fold']
                    
                    # Log the model being loaded with appropriate metric based on task type
                    metrics = model_info.get('metrics', {})
                    if self.config.classification:
                        metric_str = f"AUC: {metrics.get('auc', 'N/A')}"  # Remove .4f formatting
                    else:
                        metric_str = f"RMSE: {metrics.get('rmse', 'N/A')}"  # Remove .4f formatting
                    
                    self.logger.info(f"Loading model CV{cv_num} Fold{fold} ({metric_str})")
                    
                    self._load_single_model(cv_num, fold)
                    
                except Exception as e:
                    self.logger.error(f"Error loading model for CV{cv_num}, Fold{fold}: {str(e)}")
                    continue
                    
            if not self.models:
                raise ValueError("No models were successfully loaded")
                
            self.logger.info(f"Successfully loaded {len(self.models)} models")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Best models file not found: {best_models_file}")
        except Exception as e:
            self.logger.error(f"Error loading best models: {str(e)}")
            raise

    def _load_single_model(self, cv_num: int, fold: int):
        """Load a single model with proper error handling."""
        checkpoint_path = os.path.join(
            self.config.output_dir,
            f'{self.config.task_name}_{self.config.task_type}_cv_results',
            f'cv{cv_num}',
            'checkpoints',
            f"{self.config.task_name}_{self.config.task_type}_cv{cv_num}_fold{fold}_best.ckpt"
        )
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        hyperparams = checkpoint['hyperparameters']
        
        model = BaseGNN(
            config=self.config,
            rgcn_hidden_feats=hyperparams['rgcn_hidden_feats'],
            ffn_hidden_feats=hyperparams['ffn_hidden_feats'],
            ffn_dropout=hyperparams['ffn_dropout'],
            rgcn_dropout=hyperparams['rgcn_dropout'],
            classification=self.config.classification,
            num_classes=2 if self.config.classification else None
        )
        
        # Failsafe: Filter out incompatible keys (e.g., bias terms from old checkpoints)
        def filter_state_dict(state_dict, model):
            """Filter out incompatible keys from state_dict"""
            model_keys = set(model.state_dict().keys())
            filtered_dict = {}
            filtered_out = []

            for key, value in state_dict.items():
                if 'graph_conv_layer.bias' in key:
                    filtered_out.append(key)
                    continue
                if key in model_keys:
                    filtered_dict[key] = value

            if filtered_out:
                print(f"Filtered out bias keys: {filtered_out}")

            return filtered_dict

        filtered_state_dict = filter_state_dict(checkpoint['state_dict'], model)
        model.load_state_dict(filtered_state_dict, strict=False)
        model = model.to(self.device)
        model.eval()
        
        self.models.append({
            'model': model,
            'cv': cv_num,
            'fold': fold,
            'hyperparams': hyperparams
        })
    


    def save_checkpoint(self, state: Dict, checkpoint_path: str = None):
        """
        Save checkpoint for recovery.
        
        Args:
            state (Dict): State to save in the checkpoint
            checkpoint_path (str, optional): Custom path to save checkpoint. 
                                            Defaults to self.checkpoint_file.
        """
        try:
            # Use provided path or default to the standard location
            save_path = checkpoint_path if checkpoint_path else self.checkpoint_file
            
            # Add metadata to state
            state['metadata'] = {
                'input_csv': self.input_csv,
                'timestamp': datetime.now().isoformat(),
                'total_molecules': state.get('total_molecules', 0),
                'start_idx': state.get('start_idx', 0),
                'end_idx': state.get('end_idx', None)
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(state, f)
            self.logger.info(f"Saved checkpoint to {save_path}")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {str(e)}")
    


    def load_checkpoint(self, checkpoint_path: str = None) -> Dict:
        """
        Load checkpoint if exists and validate it matches current input file.
        
        Args:
            checkpoint_path (str, optional): Custom path to load checkpoint from.
                                        Defaults to self.checkpoint_file.
                                        
        Returns:
            Dict: Checkpoint data or empty template if no valid checkpoint
        """
        # Use provided path or default
        load_path = checkpoint_path if checkpoint_path else self.checkpoint_file
        
        if os.path.exists(load_path):
            try:
                with open(load_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                
                # Validate checkpoint matches current run
                metadata = checkpoint.get('metadata', {})
                if metadata.get('input_csv') != self.input_csv:
                    self.logger.warning(f"Checkpoint at {load_path} is from a different input file. Starting fresh.")
                    return {'current_idx': 0, 'results': []}
                
                # Log which specific checkpoint was loaded
                self.logger.info(f"Loaded checkpoint from {load_path}")
                
                # Return the checkpoint data
                return checkpoint
            except Exception as e:
                self.logger.error(f"Error loading checkpoint from {load_path}: {str(e)}")
                return {'current_idx': 0, 'results': []}
        else:
            # Log only if it would have been expected to exist
            if checkpoint_path:
                self.logger.info(f"No checkpoint found at {load_path}")
            return {'current_idx': 0, 'results': []}
        

    def _predict_single_model(self, model_info: Dict, graph: Data) -> Tuple[float, Optional[np.ndarray]]:
        """
        Make prediction using a single model with proper device handling.
        
        Args:
            model_info: Dictionary containing model and its metadata
            graph: PyG Data object containing molecule graph
        
        Returns:
            Tuple of (prediction, atom weights)
        """
        try:
            model = model_info['model']
            
            with torch.no_grad():
                # Create a new graph object with tensors on the correct device
                graph_device = next(model.parameters()).device
                graph_copy = Data(
                    x=graph.x.to(graph_device),
                    edge_index=graph.edge_index.to(graph_device),
                    edge_attr=graph.edge_attr.to(graph_device) if hasattr(graph, 'edge_attr') else None,
                    edge_type=graph.edge_type.to(graph_device) if hasattr(graph, 'edge_type') else None,
                    batch=None  # Will be set by Batch.from_data_list
                )
                
                if hasattr(graph, 'smask'):
                    graph_copy.smask = graph.smask.to(graph_device)
                
                # Create batch with tensor on the correct device
                model_batch = Batch.from_data_list([graph_copy])
                
                # Ensure model is in eval mode
                model.eval()
                
                # Make prediction
                pred, atom_weights = model(model_batch)
                
                # Process prediction based on task type
                if model.classification:  # Use model's classification attribute
                    pred = torch.sigmoid(pred).item()
                else:
                    pred = pred.item()
                
                # Move atom weights to CPU if they exist
                atom_weights_np = atom_weights.cpu().numpy() if atom_weights is not None else None
                
                return pred, atom_weights_np
                
        except Exception as e:
            self.logger.error(f"Error in model prediction: {str(e)}")
            return None, None
    










    def calculate_decision_framework(self, compound_attributions: List[float], predictions: List[float], threshold: float) -> Dict[str, str]:
        """
        Calculate decision framework based on ensemble agreement and fragment reliability
        
        The 4-scenario decision framework maps prediction uncertainty and explanation reliability
        to specific recommended actions:
        
        - Scenario A (HIGH Agreement + HIGH Reliability): "Trust + Use explanation for SAR"
          When models agree and explanations are reliable, trust both.
          
        - Scenario B (HIGH Agreement + LOW Reliability): "Trust prediction, validate mechanism experimentally"
          When models agree but explanations aren't reliable, trust prediction but verify mechanism.
          
        - Scenario C (LOW Agreement + HIGH Reliability): "Investigate prediction uncertainty, trust mechanism"
          When models disagree but explanations are reliable, investigate prediction but trust mechanism.
          
        - Scenario D (LOW Agreement + LOW Reliability): "Full experimental validation required"
          When both models disagree and explanations aren't reliable, validate experimentally.
        
        Args:
            compound_attributions: List of attribution values for all fragments
            predictions: List of individual model predictions
            threshold: Classification threshold for determining activity
        
        Returns:
            Dictionary with:
            - ensemble_agreement: 'HIGH' or 'LOW'
            - reliability_status: 'HIGH' or 'LOW'
            - decision_scenario: 'A', 'B', 'C', or 'D'
            - decision_scenario_description: Description of what the scenario means
            - recommended_action: Specific recommended action based on the scenario
        """
        result = {}
        
        # Calculate ensemble agreement
        binary_predictions = [1 if pred >= threshold else 0 for pred in predictions]
        all_active = all(bp == 1 for bp in binary_predictions)
        all_inactive = all(bp == 0 for bp in binary_predictions)
        
        # HIGH if all models agree, LOW otherwise
        result['ensemble_agreement'] = 'HIGH' if (all_active or all_inactive) else 'LOW'
        
        # Get consensus binary prediction
        binary_prediction = 1 if np.mean(predictions) >= threshold else 0
        
        # Calculate reliability based on fragment consistency
        if not compound_attributions:
            result['reliability_status'] = 'LOW'
        else:
            total_fragments = len(compound_attributions)
            avg_attribution = np.mean(compound_attributions)
            
            # Count fragments CONSISTENT with prediction
            if binary_prediction == 1:  # Active prediction
                consistent_fragments = sum(1 for attr in compound_attributions if attr > 0.1)
                avg_consistent = avg_attribution > 0.1
            else:  # Inactive prediction  
                consistent_fragments = sum(1 for attr in compound_attributions if attr < -0.1)
                avg_consistent = avg_attribution < -0.1
            
            pct_consistent = (consistent_fragments / total_fragments) * 100
            
            # Reliability is HIGH if:
            # 1. 100% fragments are consistent, OR
            # 2. Either ≥70% fragments are consistent OR average attribution is consistent
            if pct_consistent == 100:
                result['reliability_status'] = 'HIGH'
            elif pct_consistent >= 70 or avg_consistent:
                result['reliability_status'] = 'HIGH'
            else:
                result['reliability_status'] = 'LOW'
        
        # Determine decision scenario (A, B, C, or D) with descriptions
        if result['ensemble_agreement'] == 'HIGH' and result['reliability_status'] == 'HIGH':
            result['decision_scenario'] = 'A'
            result['decision_scenario_description'] = "HIGH Agreement + HIGH Reliability"
            result['recommended_action'] = "Trust + Use explanation for SAR"
        elif result['ensemble_agreement'] == 'HIGH' and result['reliability_status'] == 'LOW':
            result['decision_scenario'] = 'B'
            result['decision_scenario_description'] = "HIGH Agreement + LOW Reliability"
            result['recommended_action'] = "Trust prediction, validate mechanism experimentally"
        elif result['ensemble_agreement'] == 'LOW' and result['reliability_status'] == 'HIGH':
            result['decision_scenario'] = 'C'
            result['decision_scenario_description'] = "LOW Agreement + HIGH Reliability"
            result['recommended_action'] = "Investigate prediction uncertainty, trust mechanism"
        else:  # LOW agreement + LOW reliability
            result['decision_scenario'] = 'D'
            result['decision_scenario_description'] = "LOW Agreement + LOW Reliability"
            result['recommended_action'] = "Full experimental validation required"
        
        return result
    
    def calculate_reliability_confidence(self, compound_attributions: List[float], prediction: int) -> str:
        """
        Legacy function for backwards compatibility.
        Calculate XAI reliability confidence based on fragment attribution consistency
        
        Args:
            compound_attributions: List of attribution values for all fragments
            prediction: Binary prediction (1=active, 0=inactive)
        
        Returns:
            'HIGH', 'MODERATE', or 'LOW' reliability confidence
        """
        if not compound_attributions:
            return 'LOW'
        
        total_fragments = len(compound_attributions)
        avg_attribution = np.mean(compound_attributions)
        
        # Count fragments CONSISTENT with prediction (not contradictory)
        if prediction == 1:  # Active prediction
            consistent_fragments = sum(1 for attr in compound_attributions if attr > 0.1)
            avg_consistent = avg_attribution > 0.1
        else:  # Inactive prediction  
            consistent_fragments = sum(1 for attr in compound_attributions if attr < -0.1)
            avg_consistent = avg_attribution < -0.1
        
        pct_consistent = (consistent_fragments / total_fragments) * 100
        
        # Reliability Classification (based on CONSISTENCY, not contradiction)
        if pct_consistent == 100:
            return 'HIGH'  # 100% fragments consistent = highly reliable
        elif pct_consistent >= 70 or avg_consistent:
            return 'MODERATE'  # 70%+ consistent OR average consistent = moderate
        else:
            return 'LOW'  # <70% consistent = low reliability



    def predict_molecule(self, smiles: str, prediction_threshold: float = 0.4164) -> Dict:
        """
        Predict properties for a single molecule using ensemble averaging,
        and include scaffold and substituent attributions with reliability confidence.
        """
        try:
            # Construct base graph
            graph = construct_mol_graph_from_smiles(smiles, smask=[])
            if graph is None:
                return {'error': 'Failed to construct graph', 'SMILES': smiles}

            # Ensemble predictions
            predictions = []
            atom_weights_list = []
            for model_info in self.models:
                pred, atom_w = self._predict_single_model(model_info, graph)
                if pred is not None:
                    predictions.append(pred)
                    if atom_w is not None:
                        atom_weights_list.append(atom_w)
            
            if not predictions:
                return {'error': 'No valid predictions', 'SMILES': smiles}

            mean_pred = float(np.mean(predictions))
            std_pred = float(np.std(predictions))

            # Build result
            result = {
                'SMILES': smiles,
                'prediction_std': std_pred,
                'n_models': len(predictions),
                'individual_predictions': predictions  
            }
            
            if self.config.classification:
                result['ensemble_prediction'] = mean_pred
            else:
                result['pred_logMIC'] = mean_pred

            # Include atom-level weights if available
            if atom_weights_list:
                result['atom_weights'] = np.mean(atom_weights_list, axis=0).tolist()

            # Substructure analysis
            subtype = getattr(self.config, 'substructure_type', None)
            if subtype:
                # Check if molecule has rings
                mol = Chem.MolFromSmiles(smiles)
                if mol and mol.GetRingInfo().NumRings() == 0:
                    # Skip Murcko analysis for molecules without rings
                    result['xai_method'] = 'None - No ring systems'
                    result['xai_failure_reason'] = 'Murcko requires aromatic/aliphatic rings'
                else:
                    # Perform substructure analysis
                    t = subtype
                    sc_attrs, sc_smis, sc_subs = self.analyze_substructures(smiles, subtype)
                    result['xai_method'] = subtype  # Add XAI method used
                    
                    # Add scaffold information
                    for i, (attr, smi) in enumerate(zip(sc_attrs, sc_smis)):
                        result[f'{t}_substructure_{i}_attribution'] = attr
                        result[f'{t}_substructure_{i}_smiles'] = smi
                        
                        # Flatten the substituents data
                        if i < len(sc_subs):
                            for j, sub in enumerate(sc_subs[i]):
                                result[f'{t}_substituent_{i}_{j}_smiles'] = sub.get('smiles', '')
                                result[f'{t}_substituent_{i}_{j}_context'] = sub.get('context', '')
                                result[f'{t}_substituent_{i}_{j}_attribution'] = sub.get('attribution', 0.0)

                    # Apply 4-scenario decision framework if we have classifications and substructures
                    if sc_attrs and self.config.classification:
                        # Get binary prediction using the SAME threshold as the main prediction
                        binary_prediction = 1 if result.get('ensemble_prediction', 0) >= prediction_threshold else 0
                        
                        # For backwards compatibility
                        reliability = self.calculate_reliability_confidence(sc_attrs, binary_prediction)
                        result['reliability_confidence'] = reliability
                        
                        # Calculate new decision framework results
                        framework_results = self.calculate_decision_framework(
                            sc_attrs, 
                            predictions,  # Pass all individual model predictions
                            prediction_threshold
                        )
                        
                        # Add framework results to the result dictionary
                        result.update(framework_results)
                        
                        # Add detailed metrics for analysis
                        total_fragments = len(sc_attrs)
                        avg_attribution = float(np.mean(sc_attrs))
                        
                        if binary_prediction == 1:
                            consistent_fragments = sum(1 for attr in sc_attrs if attr > 0.1)
                        else:
                            consistent_fragments = sum(1 for attr in sc_attrs if attr < -0.1)
                        
                        result['reliability_fragments_total'] = total_fragments
                        result['reliability_fragments_consistent'] = consistent_fragments
                        result['reliability_consistency_pct'] = (consistent_fragments / total_fragments) * 100
                        result['reliability_avg_attribution'] = avg_attribution

            return result

        except Exception as e:
            self.logger.error(f"Error predicting molecule {smiles}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {'error': str(e), 'SMILES': smiles}





 



    def analyze_molecule_scaffolds(self, smiles: str) -> Dict:
        """Analyze molecule's scaffold hierarchy and substituents"""
        try:
            # Get scaffold hierarchy
            hierarchy = get_scaffold_hierarchy(smiles)
            
            if "error" in hierarchy:
                return {"error": hierarchy["error"]}
            
            # Create a formatted result with clear hierarchy
            result = {
                "SMILES": smiles,
                "num_scaffolds": len(hierarchy["scaffolds"]),
                "num_substituents": len(hierarchy["substituents"])
            }
            
            # Add scaffolds
            for i, scaffold in enumerate(hierarchy["scaffolds"]):
                result[f"scaffold_{i+1}_smiles"] = scaffold["smiles"]
            
            # Add substituents with attachment context
            for i, subst in enumerate(hierarchy["substituents"]):
                result[f"substituent_{i+1}_smiles"] = subst["smiles"]
                result[f"substituent_{i+1}_attachment"] = subst["attachment_type"]
                result[f"substituent_{i+1}_r_group"] = subst["r_group"]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing scaffolds for {smiles}: {str(e)}")
            return {"error": str(e), "SMILES": smiles}



    def smiles_to_mol_no_kekule(smi):
        # build without sanitization
        m = MolFromSmiles(smi, sanitize=False)
        # sanitize everything except Kekulize
        SanitizeMol(m, 
            SanitizeFlags.SANITIZE_ALL ^ SanitizeFlags.SANITIZE_KEKULIZE
        )
        return m





    def get_substructures(self, smiles: str, substructure_type: str) -> dict:
        """
        Get substructures based on type, but for Murcko only keep the top 40 largest scaffolds.
        """
        try:
            if substructure_type == 'murcko':
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return {}

                # Check if molecule has rings - explicit check before proceeding
                if mol.GetRingInfo().NumRings() == 0:
                    return {}  # Skip Murcko analysis for molecules without rings

                # build adjacency list for BFS
                graph = {
                    atom.GetIdx(): [nbr.GetIdx() for nbr in atom.GetNeighbors()]
                    for atom in mol.GetAtoms()
                }

                # get all rings as sets of atom‐indices
                ring_sets = [set(r) for r in mol.GetRingInfo().AtomRings()]
                if not ring_sets:
                    return {}

                def shortest_path(src: set, dst: set):
                    visited = set(src)
                    queue = deque([[i] for i in src])
                    while queue:
                        path = queue.popleft()
                        last = path[-1]
                        if last in dst:
                            return path
                        for nb in graph[last]:
                            if nb not in visited:
                                visited.add(nb)
                                queue.append(path + [nb])
                    return []

                # enumerate all combinations of rings 1..N
                all_scaffolds = []
                N = len(ring_sets)
                for k in range(1, N + 1):
                    for combo in itertools.combinations(range(N), k):
                        atoms = set().union(*(ring_sets[i] for i in combo))
                        # if more than one ring, connect them
                        if len(combo) > 1:
                            base = ring_sets[combo[0]]
                            for idx in combo[1:]:
                                atoms |= set(shortest_path(base, ring_sets[idx]))
                        all_scaffolds.append(atoms)

                # dedupe
                unique = []
                for s in all_scaffolds:
                    if not any(s == u for u in unique):
                        unique.append(s)

                # build indexed dict
                scaffolds = {i: sorted(s) for i, s in enumerate(unique)}

                # now **limit** to top 40 by size
                MAX_SCAFFOLDS = 40
                if len(scaffolds) > MAX_SCAFFOLDS:
                    # sort indices by scaffold‐size descending
                    sorted_idxs = sorted(
                        scaffolds.keys(),
                        key=lambda i: len(scaffolds[i]),
                        reverse=True
                    )[:MAX_SCAFFOLDS]
                    scaffolds = {i: scaffolds[i] for i in sorted_idxs}

                return scaffolds

            elif substructure_type == 'brics':
                # … your existing BRICS code …
                info = return_brics_leaf_structure(smiles)
                return info.get('substructure', {})

            elif substructure_type == 'fg':
                # … your existing FG code …
                if not hasattr(self.config, 'fg_with_ca_list'):
                    self._initialize_fg_patterns()
                fg_hits, _ = return_fg_hit_atom(
                    smiles,
                    self.config.fg_name_list,
                    self.config.fg_with_ca_list,
                    self.config.fg_without_ca_list
                )
                return {i: hits[0] for i, hits in enumerate(fg_hits) if hits}

            else:
                self.logger.warning(
                    f"No substructures for type '{substructure_type}' in {smiles}"
                )
                return {}

        except Exception as e:
            self.logger.error(f"Error extracting [{substructure_type}] for {smiles}: {e}")
            return {}







    def _extract_substituents_with_context(self, smiles: str, core_atoms: List[int]) -> List[Dict[str,Any]]:
        #print(f"DEBUG: _extract_substituents_with_context called with {len(core_atoms)} core atoms")
        """
        Identify true medicinal chemistry substituents - terminal groups attached to a core scaffold.
        
        Args:
            smiles: Molecule SMILES string
            core_atoms: List of atom indices that belong to the core scaffold
            
        Returns:
            List of dictionaries containing substituent information
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        core_set = set(core_atoms)
        all_idx = set(range(mol.GetNumAtoms()))
        non_core_idx = all_idx - core_set

        # Build atom connectivity map
        nbrs = {a.GetIdx(): [n.GetIdx() for n in a.GetNeighbors()]
                for a in mol.GetAtoms()}
        
        # Find attachment points: core atoms connected to non-core atoms
        attachment_points = {}
        for core_atom in core_set:
            for nbr in nbrs[core_atom]:
                if nbr in non_core_idx:
                    if nbr not in attachment_points:
                        attachment_points[nbr] = core_atom

        # Collect substituent fragments
        visited = set()
        true_substituents = []
        
        for start_atom in attachment_points.keys():
            if start_atom in visited:
                continue
                
            # Do a BFS to find the connected substituent fragment
            fragment = set()
            queue = deque([start_atom])
            
            while queue:
                atom_idx = queue.popleft()
                fragment.add(atom_idx)
                visited.add(atom_idx)
                
                for neighbor in nbrs[atom_idx]:
                    if neighbor in non_core_idx and neighbor not in fragment and neighbor not in visited:
                        queue.append(neighbor)
            
            # Get the attachment point
            attachment_atom = attachment_points[start_atom]
            
            # Determine if the attachment point is aromatic
            is_aromatic = mol.GetAtomWithIdx(attachment_atom).GetIsAromatic()
            context = 'aromatic' if is_aromatic else 'aliphatic'
            
            # Get SMILES for this fragment
            try:
                frag_smiles = Chem.MolFragmentToSmiles(
                    mol, atomsToUse=list(fragment), kekuleSmiles=True
                )
                
                # Filter out suspicious fragments that are likely parts of rings
                suspicious_fragments = ['CCC', 'CC', 'C', 'c']
                if frag_smiles in suspicious_fragments:
                    # Extra check: if this fragment is in a ring, skip it
                    in_ring = any(mol.GetAtomWithIdx(idx).IsInRing() for idx in fragment)
                    if in_ring:
                        continue
                    
                true_substituents.append({
                    'fragment_atoms': list(fragment),
                    'smiles': frag_smiles,
                    'attachment_atom': attachment_atom,
                    'context': context
                })
            except Exception as e:
                self.logger.warning(f"Failed to process substituent: {e}")
                
        return true_substituents


    



    def attribute_scaffolds(self, smiles: str, substructure_type: str) -> Tuple[List[float], List[str], List[List[Dict[str,Any]]]]:
        """
        Analyze substructures with adaptive scaling for attributions.
        
        Args:
            smiles: Molecule SMILES string
            substructure_type: Type of substructure to analyze ('murcko', 'brics', 'fg')
        
        Returns:
            Tuple[List[float], List[str], List[List[Dict[str,Any]]]]: 
                - Per-scaffold attribution scores
                - Scaffold SMILES
                - List of substituents (with smiles, context, attribution)
        """
        import math
        
        subs_dict = self.get_substructures(smiles, substructure_type)
        if not subs_dict:
            return [], [], []

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [], [], []

        # 1) Collect all logit differences for all substructures (scaffolds)
        scaffold_diffs = []
        scaffold_atoms = []
        scaffold_smis = []
        
        for idx, atoms in subs_dict.items():
            atoms = list(map(int, atoms))
            
            # Get scaffold SMILES (skip weird ones)
            try:
                scf_smi = Chem.MolFragmentToSmiles(
                    mol, atomsToUse=atoms, kekuleSmiles=True
                )
                if ':' in scf_smi or '*' in scf_smi:
                    continue
            except:
                continue
            
            # Collect logit differences from all models for this scaffold
            model_diffs = []
            for minfo in self.models:
                _, diff = self.calculate_attribution(
                    minfo['model'], smiles, atoms, self.device
                )
                if diff is not None:
                    model_diffs.append(diff)
            
            if not model_diffs:
                continue
                
            mean_diff = float(np.mean(model_diffs))
            scaffold_diffs.append(mean_diff)
            scaffold_atoms.append(atoms)
            scaffold_smis.append(scf_smi)
        
        if not scaffold_diffs:
            return [], [], []
        
        # 2) Determine adaptive scaling factor based on max difference
        max_diff = max(abs(diff) for diff in scaffold_diffs) + 1e-6
        alpha = 1.0 / max_diff
        
        # 3) Apply tanh with adaptive scaling to get final attributions
        scaling_factor = 1.0 / 0.76  # Since tanh(1.0) ≈ 0.76, scale to utilize full [-1,1] range
        scaffold_attrs = [math.tanh(alpha * diff) * scaling_factor for diff in scaffold_diffs]
        
        # 4) Process substituents with the same adaptive scaling
        all_subs = []
        for atoms in scaffold_atoms:
            raw_subs = self._extract_substituents_with_context(smiles, atoms)
            subs_with_attr = []
            
            for sub in raw_subs:
                sub_diffs = []
                # Calculate raw differences for each substituent using all models
                for minfo in self.models:
                    _, sub_diff = self.calculate_attribution(
                        minfo['model'], smiles, sub['fragment_atoms'], self.device
                    )
                    if sub_diff is not None:
                        sub_diffs.append(sub_diff)
                
                # Get the SMILES for the substituent        
                sub_smiles = sub['smiles']
                if ':' in sub_smiles:
                    # Replace colons with appropriate bond symbols
                    sub_smiles = sub_smiles.replace(":", "-")
                    sub['smiles'] = sub_smiles
                    
                # Apply the same adaptive scaling to substituents
                if sub_diffs:
                    mean_sub_diff = float(np.mean(sub_diffs))
                    mean_sub_attr = math.tanh(alpha * mean_sub_diff) * scaling_factor
                    subs_with_attr.append({
                        'smiles': sub['smiles'],
                        'context': sub['context'],
                        'attribution': mean_sub_attr
                    })
            
            all_subs.append(subs_with_attr)
        
        return scaffold_attrs, scaffold_smis, all_subs





    def analyze_substructures(
        self, 
        smiles: str, 
        substructure_type: str
    ) -> Tuple[List[float], List[str], List[List[Dict[str,Any]]]]:
        """
        Analyze substructures for attribution with improved batching and caching.
        
        Returns:
            Tuple of (attributions, SMILES, substituents)
        """
        import math
        
        # Use caching to avoid redundant calculations
        if not hasattr(self, '_analysis_cache'):
            self._analysis_cache = {}
            
        cache_key = f"{smiles}_{substructure_type}" 
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]
        
        try:
            # Check if molecule has rings - required for Murcko analysis
            if substructure_type == 'murcko':
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return [], [], []
                if mol.GetRingInfo().NumRings() == 0:
                    return [], [], []  # Skip Murcko for molecules without rings

            # Get substructures with optimized extraction
            subs_dict = self.get_substructures(smiles, substructure_type)
            if not subs_dict:
                return [], [], []

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [], [], []

            # Extract substructure atoms and compute SMILES once
            substructure_atoms = []
            scaffold_smis = []
            
            for idx, atoms in subs_dict.items():
                # Get scaffold SMILES
                try:
                    # Ensure atoms is a list of integers
                    atoms = list(map(int, atoms)) if not all(isinstance(a, int) for a in atoms) else atoms
                    
                    scf_smi = Chem.MolFragmentToSmiles(
                        mol, atomsToUse=atoms, kekuleSmiles=True
                    )
                    # Skip invalid SMILES
                    if ':' in scf_smi or '*' in scf_smi:
                        continue
                    substructure_atoms.append(atoms)
                    scaffold_smis.append(scf_smi)
                except Exception as e:
                    self.logger.debug(f"Error getting SMILES for substructure: {str(e)}")
                    continue
            
            if not substructure_atoms:
                return [], [], []
            
            # Calculate attributions for all substructures in parallel
            # Process each model sequentially, but all substructures in batch
            all_model_diffs = []
            
            for model_info in self.models:
                # Calculate all attributions in one batch per model
                orig_pred, attributions = self.batch_calculate_attributions(
                    model_info['model'],
                    smiles,
                    substructure_atoms,
                    self.device
                )
                
                if attributions:
                    all_model_diffs.append(attributions)
            
            # Average attribution differences across models
            scaffold_diffs = []
            for atoms in substructure_atoms:
                atom_tuple = tuple(atoms)
                
                # Collect differences for this substructure across models
                diffs = []
                for model_diffs in all_model_diffs:
                    if atom_tuple in model_diffs:
                        diff_val = model_diffs[atom_tuple]
                        if diff_val is not None:
                            diffs.append(diff_val)
                
                # Calculate average difference
                if diffs:
                    avg_diff = float(np.mean(diffs))
                    scaffold_diffs.append(avg_diff)
                else:
                    scaffold_diffs.append(0.0)
            
            if not scaffold_diffs:
                return [], [], []
            
            # Apply adaptive tanh scaling with normalization
            max_diff = max(abs(diff) for diff in scaffold_diffs) + 1e-6
            alpha = 1.0 / max_diff
            scaling_factor = 1.0  # Scale to target range of [-1, 1]
            scaffold_attrs = [math.tanh(alpha * diff) * scaling_factor for diff in scaffold_diffs]
            
            # Process substituents (simplified approach to reduce computation)
            all_subs = []
            

            # Set limits for substituents
            self.MAX_SUBSTITUENTS = 12 # Number of substituents to process
            self.MAX_SCAFFOLD_SUBSTITUENTS = 10  # Number of scaffolds to process substituents for

            # Then in analyze_substructures:
            for i, atoms in enumerate(substructure_atoms):
                # Process substituents for more scaffolds
                if i < min(self.MAX_SCAFFOLD_SUBSTITUENTS, len(substructure_atoms)):  # Changed from 3 to 10
                    try:
                        raw_subs = self._extract_substituents_with_context(smiles, atoms)
                        
                        # Process more substituents per scaffold
                        subs_to_process = raw_subs[:self.MAX_SUBSTITUENTS] if raw_subs else []  # Changed from 3 to 12
                        
                        # Skip full attribution calculation for substituents - use simplified approach
                        # This avoids the costly double model pass for each substituent
                        subs_with_attr = []
                        for sub in subs_to_process:
                            # Set attribution proportional to parent scaffold
                            attr_val = scaffold_attrs[i] * 0.5  # Simplified attribution
                            
                            sub_smiles = sub['smiles']
                            if ':' in sub_smiles:
                                # Replace colons with appropriate bond symbols
                                sub_smiles = sub_smiles.replace(":", "-")
                                sub['smiles'] = sub_smiles
                            
                            subs_with_attr.append({
                                'smiles': sub['smiles'],
                                'context': sub['context'],
                                'attribution': attr_val
                            })
                        
                        all_subs.append(subs_with_attr)
                    except Exception as e:
                        self.logger.error(f"Error processing substituents: {str(e)}")
                        all_subs.append([])
                else:
                    # Skip detailed analysis for less important scaffolds
                    all_subs.append([])
            
            # Make sure lengths match
            while len(all_subs) < len(scaffold_attrs):
                all_subs.append([])
            
            # Cache results
            result = (scaffold_attrs, scaffold_smis, all_subs)
            self._analysis_cache[cache_key] = result
            return result
            
        except Exception as e:
            self.logger.error(f"Error in substructure analysis for {smiles}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return [], [], []



    






    def process_batch(self, smiles_list: List[str], compound_ids: List[str], 
                    prediction_threshold: float = None) -> List[Dict]:
        """Process a batch of molecules in parallel with improved memory management."""
        batch_start_time = time.time()
        results = []
        
        # Clean memory before starting batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Adjust worker count based on batch size
        effective_workers = min(self.num_workers, len(smiles_list))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_smiles = {}
            
            # Submit all jobs with timeout protection
            for smiles, comp_id in zip(smiles_list, compound_ids):
                future = executor.submit(
                    self._process_molecule_with_timeout, 
                    smiles, 
                    comp_id, 
                    prediction_threshold
                )
                future_to_smiles[future] = (smiles, comp_id)
            
            # Process results as they complete
            completed = 0
            for future in concurrent.futures.as_completed(future_to_smiles):
                smiles, comp_id = future_to_smiles[future]
                completed += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Log progress every 10 molecules or at 25% intervals
                    log_interval = max(1, len(smiles_list) // 4)
                    if completed % log_interval == 0 or completed == len(smiles_list):
                        elapsed = time.time() - batch_start_time
                        per_mol = elapsed / completed
                        remaining = per_mol * (len(smiles_list) - completed)
                        
                        self.logger.info(
                            f"Processed {completed}/{len(smiles_list)} molecules "
                            f"({completed/len(smiles_list)*100:.1f}%) - "
                            f"{per_mol:.2f}s per molecule - "
                            f"Est. remaining: {self.format_time(remaining)}"
                        )
                        
                        # Periodically clean memory
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                except Exception as e:
                    self.logger.error(f"Error processing {smiles}: {str(e)}")
                    results.append({
                        'COMPOUND_ID': comp_id,
                        'SMILES': smiles,
                        'error': str(e)
                    })
        
        batch_time = time.time() - batch_start_time
        self.logger.info(
            f"Batch completed in {self.format_time(batch_time)} - "
            f"Average: {batch_time/len(smiles_list):.2f}s per molecule"
        )
        
        return results






    def _process_molecule_with_timeout(self, smiles: str, comp_id: str, prediction_threshold: float, 
                                    timeout: int = 120) -> Dict:
        """Process a single molecule with timeout protection to prevent hanging on complex molecules."""
        try:
            # Get base prediction first - NOW PASS THE THRESHOLD
            result = self.predict_molecule(smiles, prediction_threshold)
            result['COMPOUND_ID'] = comp_id
            
            # Add classification prediction if threshold is provided
            if prediction_threshold is not None and self.config.classification:
                if 'ensemble_prediction' in result:
                    result['prediction'] = 1 if result['ensemble_prediction'] >= prediction_threshold else 0
            
            # Debug logging for substructure analysis
            self.logger.debug(f"Checking substructure analysis for {smiles}")
            self.logger.debug(f"Config has substructure_type: {hasattr(self.config, 'substructure_type')}")
            if hasattr(self.config, 'substructure_type'):
                self.logger.debug(f"Substructure type value: {self.config.substructure_type}")
            
            # Check if we need ADDITIONAL substructure analysis
            # (in case predict_molecule didn't include it for some reason)
            has_substructure_data = any(key.endswith('_attribution') for key in result.keys() 
                                    if isinstance(key, str) and 'substructure' in key)
            
            if (hasattr(self.config, 'substructure_type') and 
                self.config.substructure_type and 
                'error' not in result and 
                not has_substructure_data):
                
                try:
                    self.logger.debug(f"Starting additional substructure analysis for {smiles}")
                    
                    # Check if molecule has rings before attempting Murcko analysis
                    mol = Chem.MolFromSmiles(smiles)
                    if mol and mol.GetRingInfo().NumRings() == 0:
                        # Skip Murcko analysis for molecules without rings
                        result['xai_method'] = 'None - No ring systems'
                        result['xai_failure_reason'] = 'Murcko requires aromatic/aliphatic rings'
                    else:
                        # Use a separate thread with timeout to prevent hanging
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                self.analyze_substructures,
                                smiles,
                                self.config.substructure_type
                            )
                            
                            # Wait for result with timeout
                            sc_attrs, sc_smis, sc_subs = future.result(timeout=timeout)
                            result['xai_method'] = self.config.substructure_type
                            
                            # Add scaffold information to result
                            t = self.config.substructure_type
                            
                            # Log the number of scaffolds found
                            self.logger.debug(f"Found {len(sc_attrs)} scaffolds for {smiles}")
                            
                            for i, (attr, smi) in enumerate(zip(sc_attrs, sc_smis)):
                                result[f'{t}_substructure_{i}_attribution'] = attr
                                result[f'{t}_substructure_{i}_smiles'] = smi
                                
                                # Add substituent information
                                if i < len(sc_subs):
                                    for j, sub in enumerate(sc_subs[i]):
                                        result[f'{t}_substituent_{i}_{j}_smiles'] = sub.get('smiles', '')
                                        result[f'{t}_substituent_{i}_{j}_context'] = sub.get('context', '')
                                        result[f'{t}_substituent_{i}_{j}_attribution'] = sub.get('attribution', 0.0)
                            
                            # Apply decision framework if we have scaffold data
                            if sc_attrs and self.config.classification:
                                # For backwards compatibility
                                if 'reliability_confidence' not in result:
                                    # Get binary prediction using the SAME threshold
                                    binary_prediction = 1 if result.get('ensemble_prediction', 0) >= prediction_threshold else 0
                                    
                                    # Calculate reliability confidence based on fragment consistency
                                    reliability = self.calculate_reliability_confidence(sc_attrs, binary_prediction)
                                    result['reliability_confidence'] = reliability
                                
                                # Calculate new decision framework
                                if 'individual_predictions' in result and 'decision_scenario' not in result:
                                    framework_results = self.calculate_decision_framework(
                                        sc_attrs,
                                        result['individual_predictions'],
                                        prediction_threshold
                                    )
                                    # Add framework results to the result dictionary
                                    result.update(framework_results)
                                
                                # Add detailed metrics for analysis if not already present
                                if 'reliability_fragments_total' not in result:
                                    total_fragments = len(sc_attrs)
                                    avg_attribution = float(np.mean(sc_attrs))
                                    
                                    binary_prediction = 1 if result.get('ensemble_prediction', 0) >= prediction_threshold else 0
                                    if binary_prediction == 1:
                                        consistent_fragments = sum(1 for attr in sc_attrs if attr > 0.1)
                                    else:
                                        consistent_fragments = sum(1 for attr in sc_attrs if attr < -0.1)
                                    
                                    result['reliability_fragments_total'] = total_fragments
                                    result['reliability_fragments_consistent'] = consistent_fragments
                                    result['reliability_consistency_pct'] = (consistent_fragments / total_fragments) * 100
                                    result['reliability_avg_attribution'] = avg_attribution
                    
                except concurrent.futures.TimeoutError:
                    self.logger.warning(f"Substructure analysis for {smiles} timed out after {timeout}s")
                    result['attribution_timeout'] = True
                    result['xai_method'] = self.config.substructure_type
                    result['xai_failure_reason'] = f"Timeout after {timeout}s"
                except Exception as e:
                    self.logger.error(f"Error in substructure analysis for {smiles}: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    result['attribution_error'] = str(e)
                    result['xai_method'] = self.config.substructure_type
                    result['xai_failure_reason'] = str(e)
            else:
                # Log why we're skipping additional substructure analysis
                if not hasattr(self.config, 'substructure_type'):
                    self.logger.debug(f"Config missing substructure_type attribute")
                elif not self.config.substructure_type:
                    self.logger.debug(f"Substructure type is set to None or empty")
                elif 'error' in result:
                    self.logger.debug(f"Molecule has error, skipping substructure analysis")
                elif has_substructure_data:
                    self.logger.debug(f"Substructure data already present in result")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {smiles}: {str(e)}")
            return {
                'COMPOUND_ID': comp_id,
                'SMILES': smiles,
                'error': str(e)
            }

    

    def _process_substructure_batch(self, batch_smiles: List[str], batch_ids: List[str], prediction_threshold: float) -> List[Dict]:
        """Process a batch with improved scaffold substructure analysis."""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_smiles = {}
            
            for smiles, comp_id in zip(batch_smiles, batch_ids):
                # First get the prediction
                pred_future = executor.submit(self.predict_molecule, smiles)
                future_to_smiles[pred_future] = (smiles, comp_id, "prediction")
                
                # Then analyze scaffolds
                scaffold_future = executor.submit(self.analyze_molecule_scaffolds, smiles)
                future_to_smiles[scaffold_future] = (smiles, comp_id, "scaffold")
            
            # Process all results
            predictions = {}
            scaffolds = {}
            
            for future in concurrent.futures.as_completed(future_to_smiles):
                smiles, comp_id, task_type = future_to_smiles[future]
                try:
                    result = future.result()
                    
                    if task_type == "prediction":
                        predictions[(smiles, comp_id)] = result
                    else:  # scaffold
                        scaffolds[(smiles, comp_id)] = result
                except Exception as e:
                    self.logger.error(f"Error processing {smiles} ({task_type}): {str(e)}")
                    if task_type == "prediction":
                        predictions[(smiles, comp_id)] = {
                            'COMPOUND_ID': comp_id, 
                            'SMILES': smiles,
                            'error': str(e)
                        }
                    else:
                        scaffolds[(smiles, comp_id)] = {
                            'error': str(e), 
                            'SMILES': smiles
                        }
        
        # Combine predictions with scaffold analysis
        for (smiles, comp_id) in predictions:
            combined_result = predictions[(smiles, comp_id)].copy()
            combined_result['COMPOUND_ID'] = comp_id
            
            # Add classification prediction if threshold is provided
            if prediction_threshold is not None and self.config.classification:
                if 'ensemble_prediction' in combined_result:
                    combined_result['prediction'] = 1 if combined_result['ensemble_prediction'] >= prediction_threshold else 0
                    combined_result.pop('individual_predictions', None)
            
            # Add scaffold information
            if (smiles, comp_id) in scaffolds:
                scaffold_result = scaffolds[(smiles, comp_id)]
                for key, value in scaffold_result.items():
                    if key not in ['SMILES', 'error']:  # Avoid duplicating these keys
                        combined_result[key] = value
            
            results.append(combined_result)
        
        return results










    def calculate_attribution(
        self,
        model,
        smiles: str,
        substructure: List[int],
        device: str
    ) -> Tuple[float, float]:
        """
        Calculate logit differences for attribution scores.
        This is a more efficient implementation that handles batching better.
        
        Args:
            model: The trained PyTorch model
            smiles: SMILES string
            substructure: List of atom indices to mask
            device: Device to perform computation on
            
        Returns:
            Tuple of (original_prediction, logit_difference)
        """
        # Handle the case where model is None
        if model is None:
            return None, 0.0
        
        model.eval()
        try:
            # Check if we already have multiple substructures - if so, use batch version
            if isinstance(substructure, list) and len(substructure) > 0 and isinstance(substructure[0], list):
                return self.batch_calculate_attributions(model, smiles, substructure, device)
            
            # Handle single substructure case (backwards compatibility)
            original_graph = construct_mol_graph_from_smiles(smiles, smask=[])
            masked_graph = construct_mol_graph_from_smiles(smiles, smask=substructure)
            
            if original_graph is None or masked_graph is None:
                self.logger.error(f"Could not construct graphs for SMILES: {smiles}")
                return None, None
            
            with torch.no_grad():
                # Move graphs to correct device
                original_batch = Batch.from_data_list([original_graph]).to(device)
                masked_batch = Batch.from_data_list([masked_graph]).to(device)
                
                original_pred, _ = model(original_batch)
                masked_pred, _ = model(masked_batch)
                
                if self.config.classification:
                    # Get probability for returning the original prediction
                    original_prob = torch.sigmoid(original_pred).item()
                    
                    # For attribution: Return the raw logit difference
                    logit_diff = original_pred.item() - masked_pred.item()
                    
                    return original_prob, logit_diff
                else:
                    # For regression tasks
                    original_val = original_pred.item()
                    masked_val = masked_pred.item()
                    
                    # Return the raw difference
                    raw_diff = original_val - masked_val
                    
                    return original_val, raw_diff
                    
        except Exception as e:
            self.logger.error(f"Error in attribution calculation: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None, None

    def batch_calculate_attributions(
        self,
        model,
        smiles: str,
        substructures: List[List[int]],
        device: str
    ) -> Tuple[float, Dict[Tuple[int, ...], float]]:
        """
        Calculate attributions for multiple substructures with a single model pass.
        
        Args:
            model: The trained model
            smiles: SMILES string
            substructures: List of substructure atom index lists 
            device: Device to perform calculation on
            
        Returns:
            Tuple of (original_prediction, {substructure_tuple: logit_diff})
        """
        model.eval()
        try:
            # Construct original graph once
            original_graph = construct_mol_graph_from_smiles(smiles, smask=[])
            if original_graph is None:
                return None, {}
                
            # Process original prediction
            with torch.no_grad():
                original_batch = Batch.from_data_list([original_graph]).to(device)
                original_pred, _ = model(original_batch)
                
                # Create masked graphs for all substructures
                masked_graphs = []
                valid_indices = []
                
                for i, atoms in enumerate(substructures):
                    try:
                        masked_graph = construct_mol_graph_from_smiles(smiles, smask=atoms)
                        if masked_graph is not None:
                            masked_graphs.append(masked_graph)
                            valid_indices.append(i)
                    except Exception as e:
                        self.logger.debug(f"Error constructing graph for substructure {i}: {str(e)}")
                        continue
                
                # Return empty if no valid masked graphs
                if not masked_graphs:
                    return original_pred.item(), {}
                    
                # Process each masked graph individually to avoid OOM issues
                attributions = {}
                
                # Process in smaller batches if there are many substructures
                batch_size = 5  # Process 5 substructures at a time
                for i in range(0, len(masked_graphs), batch_size):
                    batch_graphs = masked_graphs[i:i+batch_size]
                    batch_indices = valid_indices[i:i+batch_size]
                    
                    # Create batch of graphs
                    masked_batch = Batch.from_data_list([g.to(device) for g in batch_graphs])
                    
                    # Get predictions in a single forward pass
                    masked_preds, _ = model(masked_batch)
                    
                    # Process attributions
                    for j, (batch_idx, orig_idx) in enumerate(zip(range(len(batch_graphs)), batch_indices)):
                        substructure = tuple(substructures[orig_idx])  # Convert to tuple for dict key
                        if self.config.classification:
                            logit_diff = original_pred.item() - masked_preds[j].item()
                            attributions[substructure] = logit_diff
                        else:
                            diff = original_pred.item() - masked_preds[j].item()
                            attributions[substructure] = diff
                
                if self.config.classification:
                    return torch.sigmoid(original_pred).item(), attributions
                else:
                    return original_pred.item(), attributions
                    
        except Exception as e:
            self.logger.error(f"Error in batch attribution calculation: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None, {}








    def process_dataset(
        self,
        input_csv: str,
        output_csv: str,
        batch_size: int,
        prediction_threshold: float,
        substructure_type: str,
        start_idx: int = 0,
        end_idx: int = None,
        resume: bool = False
    ):
        """Process a specific subset of the dataset with optimized batch processing."""
        
        try:
            if input_csv != self.input_csv:
                raise ValueError("Input file mismatch. Please initialize a new PredictionManager.")

            # Store prediction threshold for consistency across all predictions
            self.prediction_threshold = prediction_threshold
            self.logger.info(f"Using prediction threshold: {self.prediction_threshold}")

            # Handle None or "None" values consistently 
            if substructure_type == "None":
                substructure_type = None
                
            # Explicitly set substructure_type on config
            if substructure_type is not None:
                self.config.substructure_type = substructure_type
                self.logger.info(f"Substructure analysis type set to: {self.config.substructure_type}")
            else:
                self.config.substructure_type = None
                self.logger.warning("No substructure analysis will be performed (substructure_type is None)")

            # Initialize caches
            if not hasattr(self, '_substructure_cache'):
                self._substructure_cache = {}
            if not hasattr(self, '_analysis_cache'):
                self._analysis_cache = {}

            # Start timing
            global_start_time = time.time()

            # Read full file to get total count
            try:
                if input_csv.lower().endswith('.parquet'):
                    df_full = pd.read_parquet(input_csv)
                else:
                    df_full = pd.read_csv(input_csv)
                df_full.columns = df_full.columns.str.strip()
            except Exception as e:
                raise ValueError(f"Error reading input file {input_csv}: {str(e)}")

            self.logger.info(f"Available columns: {df_full.columns.tolist()}")

            # Validate required columns
            required_columns = ['SMILES', 'COMPOUND_ID']
            missing_columns = [col for col in required_columns if col not in df_full.columns]
            if missing_columns:
                raise ValueError(f"Required columns {missing_columns} not found. Available columns: {df_full.columns.tolist()}")

            # Clean and validate data
            empty_smiles = df_full['SMILES'].isna().sum()
            empty_ids = df_full['COMPOUND_ID'].isna().sum()
            if empty_smiles > 0 or empty_ids > 0:
                self.logger.warning(f"Found {empty_smiles} empty SMILES and {empty_ids} empty COMPOUND_IDs")

            df_full = df_full.dropna(subset=['SMILES', 'COMPOUND_ID'])
            
            full_dataset_size = len(df_full)
            if full_dataset_size == 0:
                raise ValueError("No valid molecules found in input file after cleaning")
            
            # Determine subset range
            if end_idx is None or end_idx >= full_dataset_size:
                end_idx = full_dataset_size - 1
                
            if start_idx < 0:
                start_idx = 0
                
            # Extract only our subset for processing
            df = df_full.iloc[start_idx:end_idx+1].reset_index(drop=True)
            total_molecules = len(df)
            
            self.logger.info(f"Processing subset of {total_molecules} molecules (indices {start_idx} to {end_idx}) out of {full_dataset_size} total")
            
            # Use a shard-specific checkpoint for better isolation
            shard_id = ""
            if output_csv:
                # Try to extract shard ID from output filename
                filename = os.path.basename(output_csv)
                if "shard" in filename:
                    shard_id = "_shard" + filename.split("shard")[-1].split(".")[0]
                    
            # Create shard-specific checkpoint path
            checkpoint_path = self.checkpoint_file
            if shard_id:
                checkpoint_dir = os.path.dirname(self.checkpoint_file)
                checkpoint_base = os.path.basename(self.checkpoint_file)
                checkpoint_name, checkpoint_ext = os.path.splitext(checkpoint_base)
                checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}{shard_id}{checkpoint_ext}")
                
            self.logger.info(f"Using checkpoint file: {checkpoint_path}")
            
            # Load checkpoint data if resuming
            all_results = []
            current_idx = 0
            
            if resume and os.path.exists(checkpoint_path):
                try:
                    with open(checkpoint_path, 'rb') as f:
                        checkpoint = pickle.load(f)
                        
                    # Extract only relevant results for our range
                    saved_results = checkpoint.get('results', [])
                    if saved_results:
                        # Only include results that match our molecule IDs
                        subset_ids = set(df['COMPOUND_ID'].tolist())
                        all_results = [r for r in saved_results if r.get('COMPOUND_ID') in subset_ids]
                        
                        # Determine how many we've already processed
                        current_idx = len(all_results)
                        
                        self.logger.info(f"Resuming from checkpoint with {current_idx} molecules already processed")
                except Exception as e:
                    self.logger.error(f"Error loading checkpoint: {str(e)}")
                    self.logger.warning("Starting from beginning")
                    current_idx = 0
                    all_results = []
            else:
                self.logger.info("Starting fresh processing")
                current_idx = 0
                all_results = []

            # Save initial checkpoint
            self.save_checkpoint({
                'current_idx': current_idx,
                'results': all_results,
                'total_molecules': total_molecules,
                'start_idx': start_idx,
                'end_idx': end_idx
            }, checkpoint_path)

            # Processing time tracking
            start_time = time.time()
            last_save_time = start_time
            
            # Clean memory before starting
            try:
                self._clean_memory()
            except AttributeError:
                self.logger.warning("_clean_memory method not found, using fallback memory cleanup")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            # Calculate remaining molecules to process
            remaining_molecules = total_molecules - current_idx
            if remaining_molecules <= 0:
                self.logger.info(f"All {total_molecules} molecules in this range already processed")
            else:
                self.logger.info(f"Processing {remaining_molecules} remaining molecules")
            
            # Determine batch count for logging
            total_batches = (remaining_molecules + batch_size - 1) // batch_size

            # Main processing loop with batch-level optimization
            batch_num = 1
            subset_idx = current_idx
            
            while subset_idx < total_molecules:
                if self.should_terminate:
                    self.logger.info("Termination requested. Saving checkpoint...")
                    self.save_checkpoint({
                        'current_idx': subset_idx,
                        'results': all_results,
                        'total_molecules': total_molecules,
                        'start_idx': start_idx,
                        'end_idx': end_idx
                    }, checkpoint_path)
                    break

                # Get current batch from our subset
                batch_start_time = time.time()
                batch_end_idx = min(subset_idx + batch_size, total_molecules)
                batch_df = df.iloc[subset_idx:batch_end_idx]
                batch_smiles = batch_df['SMILES'].tolist()
                batch_ids = batch_df['COMPOUND_ID'].tolist()
                
                batch_size_actual = len(batch_smiles)

                self.logger.info(f"Processing batch {batch_num}/{total_batches} ({batch_size_actual} molecules)")
                
                # Log GPU memory before processing
                if torch.cuda.is_available():
                    try:
                        gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)
                        gpu_max = torch.cuda.max_memory_allocated() / (1024 ** 3)
                        self.logger.info(f"GPU memory: current {gpu_mem:.2f} GB, peak {gpu_max:.2f} GB")
                    except Exception as e:
                        self.logger.warning(f"Could not log GPU memory: {str(e)}")
                
                # Process the batch
                batch_results = []
                try:
                    batch_results = self.process_batch(
                        batch_smiles, 
                        batch_ids, 
                        prediction_threshold
                    )
                    
                    # Verify we got results
                    if not batch_results:
                        self.logger.warning(f"Batch {batch_num} returned no results! Trying one by one...")
                        # Fallback to one-by-one processing
                        for smiles, comp_id in zip(batch_smiles, batch_ids):
                            try:
                                result = self.predict_molecule(smiles, prediction_threshold)
                                result['COMPOUND_ID'] = comp_id
                                if self.config.classification and 'ensemble_prediction' in result:
                                    result['prediction'] = 1 if result['ensemble_prediction'] >= prediction_threshold else 0
                                batch_results.append(result)
                            except Exception as e:
                                self.logger.error(f"Error processing {smiles}: {str(e)}")
                                batch_results.append({
                                    'COMPOUND_ID': comp_id,
                                    'SMILES': smiles,
                                    'error': str(e)
                                })
                except Exception as e:
                    self.logger.error(f"Error processing batch: {str(e)}")
                    self.logger.debug(traceback.format_exc())
                    # Try individual molecules as a fallback
                    self.logger.info("Attempting to process molecules individually...")
                    for smiles, comp_id in zip(batch_smiles, batch_ids):
                        try:
                            result = self.predict_molecule(smiles, prediction_threshold)
                            result['COMPOUND_ID'] = comp_id
                            if self.config.classification and 'ensemble_prediction' in result:
                                result['prediction'] = 1 if result['ensemble_prediction'] >= prediction_threshold else 0
                            batch_results.append(result)
                        except Exception as mol_e:
                            self.logger.error(f"Error processing {smiles}: {str(mol_e)}")
                            batch_results.append({
                                'COMPOUND_ID': comp_id,
                                'SMILES': smiles,
                                'error': str(mol_e)
                            })

                # Update results and save progress
                if batch_results:
                    self.logger.info(f"Adding {len(batch_results)} results from batch {batch_num}")
                    all_results.extend(batch_results)
                else:
                    self.logger.warning(f"Batch {batch_num} produced no results!")
                    # Add empty results to avoid getting stuck
                    for smiles, comp_id in zip(batch_smiles, batch_ids):
                        all_results.append({
                            'COMPOUND_ID': comp_id, 
                            'SMILES': smiles,
                            'error': 'Failed batch processing with no specific error'
                        })
                
                # Calculate and log timing information
                current_time = time.time()
                batch_time = current_time - batch_start_time
                elapsed_time = current_time - start_time
                molecules_processed = subset_idx + len(batch_results)
                
                avg_time_per_mol = batch_time / batch_size_actual
                overall_avg_time = elapsed_time / molecules_processed if molecules_processed > 0 else 0
                
                remaining_mols = total_molecules - molecules_processed
                estimated_time = remaining_mols * overall_avg_time if remaining_mols > 0 else 0
                
                self.logger.info(
                    f"Batch {batch_num} completed in {self.format_time(batch_time)} - "
                    f"{avg_time_per_mol:.2f}s per molecule"
                )
                
                self.logger.info(
                    f"Progress: {molecules_processed}/{total_molecules} molecules "
                    f"({(molecules_processed / total_molecules) * 100:.1f}%)"
                )
                
                self.logger.info(
                    f"Overall average: {overall_avg_time:.2f}s per molecule - "
                    f"Estimated time remaining: {self.format_time(estimated_time)}"
                )
                
                # Save checkpoint and temporary results periodically
                if (current_time - last_save_time) >= 300 or batch_end_idx == total_molecules:
                    self.save_checkpoint({
                        'current_idx': subset_idx + len(batch_results),
                        'results': all_results,
                        'total_molecules': total_molecules,
                        'start_idx': start_idx,
                        'end_idx': end_idx
                    }, checkpoint_path)
                    
                    # Save partial results
                    temp_file = f"{output_csv}.temp"
                    pd.DataFrame(all_results).to_csv(temp_file, index=False)
                    self.logger.info(f"Saved checkpoint and temporary results to {temp_file}")
                    last_save_time = current_time
                
                # Clean memory after batch
                try:
                    self._clean_memory()
                except AttributeError:
                    self.logger.debug("Using fallback memory cleanup after batch")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                
                # Update for next iteration
                subset_idx = batch_end_idx
                batch_num += 1

            # Save final results with proper formatting
            if all_results:
                # Log number of results before saving
                self.logger.info(f"Saving {len(all_results)} total results")
                
                # Check for substructure data in results
                if hasattr(self.config, 'substructure_type') and self.config.substructure_type:
                    pattern = f"{self.config.substructure_type}_substructure"
                    has_substructure = any(any(pattern in key for key in result.keys()) for result in all_results if isinstance(result, dict))
                    if has_substructure:
                        self.logger.info(f"Results contain {self.config.substructure_type} substructure data")
                    else:
                        self.logger.warning(f"No {self.config.substructure_type} substructure data found in results!")
                
                self._save_final_results(all_results, output_csv, substructure_type)
                
                # Log overall statistics
                total_time = time.time() - global_start_time
                overall_speed = len(all_results) / total_time if total_time > 0 else 0
                
                self.logger.info("=" * 50)
                self.logger.info(f"Processing completed in {self.format_time(total_time)}")
                self.logger.info(f"Average processing speed: {overall_speed:.2f} molecules/second")
                self.logger.info(f"Final results saved to: {output_csv}")
                self.logger.info("=" * 50)
                
        except Exception as e:
            self.logger.error(f"Error processing dataset: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Try to save progress even on error
            if locals().get('all_results') and locals().get('current_idx'):
                error_output = f"{output_csv}.error_partial"
                try:
                    pd.DataFrame(all_results).to_csv(error_output, index=False)
                    self.logger.info(f"Saved partial results to {error_output}")
                except:
                    self.logger.error("Failed to save partial results")
            raise


    



    def _process_simple_batch(self, batch_smiles, batch_ids, prediction_threshold):
        """Process a batch without substructure analysis."""
        batch_results = []
        for smiles, comp_id in zip(batch_smiles, batch_ids):
            try:
                result = self.predict_molecule(smiles, prediction_threshold)
                result['COMPOUND_ID'] = comp_id
                
                if self.config.classification and 'ensemble_prediction' in result:
                    result['prediction'] = 1 if result['ensemble_prediction'] >= prediction_threshold else 0
                elif not self.config.classification and 'ensemble_prediction' in result:
                    result['pred_logMIC'] = result.pop('ensemble_prediction')
                    
                batch_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing compound ID {comp_id}: {str(e)}")
                self.logger.debug(traceback.format_exc())
                batch_results.append({
                    'COMPOUND_ID': comp_id,
                    'SMILES': smiles,
                    'error': str(e)
                })
                
        return batch_results
    








    def _save_final_results(self, all_results: List[Dict], output_csv: str, substructure_type: str):
        """
        Save final results to file with proper column ordering,
        with flattened substituent columns and reliability confidence columns.
        """
        # Create DataFrame from results
        final_df = pd.DataFrame(all_results)
        
        # Log initial data size
        self.logger.info(f"Initial dataframe has {len(final_df)} rows and {len(final_df.columns)} columns")

        # Standard column mapping
        mapping = {'compound_id': 'COMPOUND_ID', 'smiles': 'SMILES', 'pred_logmic': 'pred_logMIC'}
        final_df.rename(columns=mapping, inplace=True)

        # Remove atom weights to avoid serialization issues
        if 'atom_weights' in final_df.columns:
            self.logger.info("Removing atom_weights column to avoid serialization issues")
            final_df = final_df.drop('atom_weights', axis=1)

        # Base columns with reliability confidence columns
        if self.config.classification:
            base = [
                'COMPOUND_ID', 'SMILES', 'ensemble_prediction', 'prediction', 'prediction_std', 
                'n_models', 'individual_predictions',
                # Decision framework columns
                'ensemble_agreement', 'reliability_status', 'decision_scenario', 'decision_scenario_description', 'recommended_action',
                # Legacy and detailed reliability metrics
                'reliability_confidence', 'reliability_fragments_total', 'reliability_fragments_consistent', 
                'reliability_consistency_pct', 'reliability_avg_attribution',
                # XAI method information
                'xai_method', 'xai_failure_reason'
            ]
        else:
            base = [
                'COMPOUND_ID', 'SMILES', 'pred_logMIC', 'prediction_std', 'n_models', 'individual_predictions',
                # Decision framework columns
                'ensemble_agreement', 'reliability_status', 'decision_scenario', 'decision_scenario_description', 'recommended_action',
                # Legacy and detailed reliability metrics
                'reliability_confidence', 'reliability_fragments_total', 'reliability_fragments_consistent', 
                'reliability_consistency_pct', 'reliability_avg_attribution',
                # XAI method information
                'xai_method', 'xai_failure_reason'
            ]

        # Parse and sort columns for scaffolds and substituents
        sc_columns = []  # For scaffold columns
        sub_columns = []  # For substituent columns
        
        if substructure_type:
            t = substructure_type
            
            # Extract all scaffold columns
            scaffold_attr_pattern = re.compile(f'{t}_substructure_(\d+)_attribution')
            scaffold_smiles_pattern = re.compile(f'{t}_substructure_(\d+)_smiles')
            
            # First get all scaffold attribution and smiles columns
            scaffold_attr_cols = []
            scaffold_smiles_cols = []
            
            for col in final_df.columns:
                attr_match = scaffold_attr_pattern.match(col)
                if attr_match:
                    idx = int(attr_match.group(1))
                    scaffold_attr_cols.append((idx, col))
                    continue
                    
                smiles_match = scaffold_smiles_pattern.match(col)
                if smiles_match:
                    idx = int(smiles_match.group(1))
                    scaffold_smiles_cols.append((idx, col))
                    continue
            
            # Sort scaffold columns by index
            scaffold_attr_cols.sort()
            scaffold_smiles_cols.sort()
            
            # Add to scaffold columns list
            sc_columns = [col for _, col in scaffold_attr_cols] + [col for _, col in scaffold_smiles_cols]
            
            # Extract all substituent columns
            sub_pattern = re.compile(f'{t}_substituent_(\d+)_(\d+)_(smiles|context|attribution)')
            sub_cols_dict = {}
            
            for col in final_df.columns:
                sub_match = sub_pattern.match(col)
                if sub_match:
                    sc_idx = int(sub_match.group(1))
                    sub_idx = int(sub_match.group(2))
                    prop = sub_match.group(3)
                    
                    # Create a composite key for sorting
                    key = (sc_idx, sub_idx, prop)
                    sub_cols_dict[key] = col
            
            # Sort substituent columns first by scaffold index, then by substituent index
            # For each substituent, order as smiles, context, attribution
            prop_order = {'smiles': 0, 'context': 1, 'attribution': 2}
            sorted_sub_keys = sorted(sub_cols_dict.keys(), 
                                    key=lambda k: (k[0], k[1], prop_order.get(k[2], 3)))
            
            sub_columns = [sub_cols_dict[k] for k in sorted_sub_keys]
        
        # Combine all columns in logical order
        ordered_columns = base + sc_columns + sub_columns
        
        # Add error column if present
        if 'error' in final_df.columns:
            ordered_columns.append('error')
        
        # Keep only columns that actually exist in the dataframe
        final_columns = [col for col in ordered_columns if col in final_df.columns]
        
        # Log the columns we found for debugging
        self.logger.info(f"Column types found: {list(final_df.columns)}")
        if len(sc_columns) > 0:
            self.logger.info(f"Found {len(sc_columns)} scaffold columns")
        if len(sub_columns) > 0:
            self.logger.info(f"Found {len(sub_columns)} substituent columns")
        
        # Check for reliability columns
        reliability_cols = [col for col in final_df.columns if 'reliability_' in col]
        if reliability_cols:
            self.logger.info(f"Found {len(reliability_cols)} reliability columns: {reliability_cols}")
        else:
            self.logger.warning("No reliability columns found in results!")
        
        # Reindex the dataframe
        final_df = final_df.reindex(columns=final_columns)

        # Convert individual_predictions to string for faster serialization
        if 'individual_predictions' in final_df.columns:
            final_df['individual_predictions'] = final_df['individual_predictions'].astype(str)        
        
        # Make sure we actually have data to save
        if len(final_df) == 0:
            self.logger.error("No data to save! DataFrame is empty.")
            # Save an empty file with just column headers as a fallback
            final_df = pd.DataFrame(columns=final_columns)
            final_df.to_csv(f"{output_csv}.empty", index=False)
            raise ValueError("No data to save - check your processing steps")
        
        # Save the results
        try:
            if output_csv.lower().endswith('.parquet'):
                # Check for problematic columns before saving
                for col in final_df.columns:
                    if final_df[col].dtype == 'object':
                        sample = final_df[col].dropna().iloc[0] if not final_df[col].dropna().empty else None
                        if isinstance(sample, (list, np.ndarray)):
                            self.logger.warning(f"Column {col} contains complex objects - converting to string")
                            final_df[col] = final_df[col].apply(lambda x: str(x) if x is not None else None)
                
                # Now save the parquet
                final_df.to_parquet(output_csv, index=False)
            else:
                final_df.to_csv(output_csv, index=False)
            
            self.logger.info(f"Results saved to {output_csv}")
            
            # Verify file size
            if os.path.exists(output_csv):
                file_size = os.path.getsize(output_csv)
                self.logger.info(f"Output file size: {file_size} bytes")
                if file_size == 0:
                    self.logger.error("Output file has zero size! Falling back to CSV.")
                    final_df.to_csv(output_csv + ".fallback.csv", index=False)
        except Exception as e:
            self.logger.error(f"Error saving to {output_csv}: {str(e)}")
            
            # Fallback to CSV if Parquet fails
            if output_csv.lower().endswith('.parquet'):
                csv_output = output_csv.replace('.parquet', '.csv')
                try:
                    final_df.to_csv(csv_output, index=False)
                    self.logger.info(f"Fallback: Results saved to {csv_output}")
                except Exception as csv_e:
                    self.logger.error(f"Failed to save even as CSV: {str(csv_e)}")
                    # Last resort - save the error partial
                    final_df.to_csv(f"{output_csv}.error_partial", index=False)

        # Cleanup temp
        tmp = f"{output_csv}.temp"
        if os.path.exists(tmp):
            os.remove(tmp)
            self.logger.info(f"Removed temporary file {tmp}")

    def _clean_memory(self):
        """Explicitly clean up memory to prevent OOM errors."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()            
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.0f} minutes"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{hours:.0f}h {minutes:.0f}m"




def main():
    print("🔍 DEBUG: Starting main() function")
    
    # Initialize config and logger
    config = Configuration()
    logger = get_logger(__name__)
    print("🔍 DEBUG: Config and logger initialized")

    # Argument parsing (use your former working version)
    parser = argparse.ArgumentParser(description='Run predictions on molecules.')
    default_input = os.path.join(config.data_dir, 'S_aureus_clean_ECDB.csv')  # Use your working file
    parser.add_argument('--input_csv', type=str, default=default_input, help='Path to input file')
    parser.add_argument('--output_csv', type=str, default=None, help='Path to save prediction results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None, help='Device to use')
    parser.add_argument('--prediction_threshold', type=float, default=0.4164, help='Threshold for classification') #  0.461 for E.coli, 0.4164 for S_aureus and 0.523 for C_albicans
    parser.add_argument('--substructure_type', type=str, choices=['brics','murcko','fg', None], default='murcko')
    parser.add_argument('--ensemble', type=str, choices=['best5','full'], default='best5')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index (inclusive)')
    parser.add_argument('--end_idx', type=int, default=None, help='End index (inclusive)')
    parser.add_argument('--resume', action='store_true', help='Resume from last saved position')
    args = parser.parse_args()
    print(f"🔍 DEBUG: Arguments parsed successfully")

    # Determine device with robust check
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'
    if args.device == 'cuda':
        try:
            if torch.cuda.device_count() == 0:
                raise RuntimeError('No CUDA devices available')
        except Exception as e:
            logger.warning(f'CUDA not usable ({e}); falling back to CPU')
            args.device = 'cpu'
    print(f"🔍 DEBUG: Device set to: {args.device}")

    logger.info('='*60)
    logger.info(f'Starting Prediction Process on device: {args.device}')
    logger.info('='*60)

    # Read and slice data
    print("🔍 DEBUG: About to read CSV...")
    if args.input_csv.lower().endswith('.csv'):
        df = pd.read_csv(args.input_csv)
    else:
        df = pd.read_parquet(args.input_csv)
    print(f"🔍 DEBUG: CSV read successfully - {len(df)} rows")
    
    total = len(df)
    # clamp indices  
    if args.end_idx is None or args.end_idx >= total:
        args.end_idx = total - 1
    args.start_idx = max(0, args.start_idx)
    if args.start_idx > args.end_idx:
        logger.error(f'Start index {args.start_idx} > end index {args.end_idx}')
        sys.exit(1)
    # slice
    df = df.iloc[args.start_idx:args.end_idx+1].reset_index(drop=True)
    subset_size = len(df)
    logger.info(f'Processing rows {args.start_idx}-{args.end_idx} ({subset_size}/{total} molecules)')
    print(f"🔍 DEBUG: Data sliced - {subset_size} molecules to process")

    # Prepare output path (use simpler logic like your former version)
    if args.output_csv is None:
        base = os.path.splitext(os.path.basename(args.input_csv))[0]
        task = 'classification' if config.classification else 'regression'
        ext = '.csv' if args.input_csv.lower().endswith('.csv') else '.parquet'
        args.output_csv = os.path.join(os.path.dirname(args.input_csv), f'{base}_prediction_{task}_{args.substructure_type}{ext}')
    
    out_dir = os.path.dirname(args.output_csv)
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f'Output will be written to {args.output_csv}')
    print(f"🔍 DEBUG: Output path prepared: {args.output_csv}")

    # Run PredictionManager with CPU fallback on model-load failure
    print("🔍 DEBUG: About to initialize PredictionManager...")
    print("🔍 DEBUG: This is where it might hang - model loading...")
    
    try:
        mgr = PredictionManager(config=config, input_csv=args.input_csv, device=args.device, 
                              num_workers=args.num_workers, ensemble_type=args.ensemble)
        print("🔍 DEBUG: ✅ PredictionManager initialized successfully!")
    except Exception as e:
        print(f"🔍 DEBUG: ❌ PredictionManager init failed: {e}")
        import traceback
        traceback.print_exc()
        
        # If CUDA model load fails, retry on CPU
        logger.warning(f'Model init failed on {args.device}: {e}. Retrying on CPU')
        args.device = 'cpu'
        print("🔍 DEBUG: Retrying on CPU...")
        mgr = PredictionManager(config=config, input_csv=args.input_csv, device='cpu', 
                              num_workers=args.num_workers, ensemble_type=args.ensemble)
        print("🔍 DEBUG: ✅ PredictionManager initialized on CPU!")

    print("🔍 DEBUG: About to call process_dataset...")
    try:
        mgr.process_dataset(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            batch_size=args.batch_size,
            prediction_threshold=args.prediction_threshold,
            substructure_type=args.substructure_type,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            resume=args.resume
        )
        print("🔍 DEBUG: ✅ process_dataset completed!")
        
        # done marker
        shard = os.path.splitext(os.path.basename(args.output_csv))[0].split('_')[-1]
        done_file = os.path.join(out_dir, f'shard{shard}.done')
        open(done_file, 'w').write(datetime.now().isoformat())
        logger.info(f'Created done marker: {done_file}')
        logger.info('Prediction completed successfully')
        print("🔍 DEBUG: ✅ Everything completed successfully!")
    except Exception as e:
        print(f"🔍 DEBUG: ❌ process_dataset failed: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f'Prediction failed: {e}')
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()