"""
xai_antimicrobial_helper.py

Streamlined helper module for multi-pathogen antimicrobial XAI visualization.
Provides ensemble prediction, ensemble attribution averaging, and internal
consistency assessment for trustworthy explainable AI.

Key Features:
- Ensemble prediction averaging across 5 models per pathogen
- Ensemble attribution averaging for reliable explanations
- Simple binary internal consistency (prediction-explanation agreement)
- Support for S. aureus (SA), E. coli (EC), C. albicans (CA)
- Cyan/Orange color scheme with intensity-based shading
"""

import os
import sys
import json
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import display, HTML, SVG

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class PathogenConfig:
    """Configuration for each pathogen model."""
    code: str
    name: str
    task_name: str
    threshold: float
    color: str
    model_dir: str

PATHOGEN_CONFIGS = {
    'SA': PathogenConfig('SA', 'S. aureus', 'S_aureus', 0.4164, '#DC143C', 'SA_model'),
    'EC': PathogenConfig('EC', 'E. coli', 'E_coli', 0.4610, '#1E90FF', 'EC_model'),
    'CA': PathogenConfig('CA', 'C. albicans', 'C_albicans', 0.5230, '#228B22', 'CA_model'),
}

# Color scheme: Cyan for positive, Orange for negative
COLORS = {
    'positive_base': (0.0, 0.5, 1.0),     # Blue/Cyan - increases activity
    'negative_base': (1.0, 0.5, 0.0),     # Orange - decreases activity
    'neutral': (0.7, 0.7, 0.7),           # Gray - minimal impact
}


def get_intensity_color(attribution: float, base_positive: Tuple, base_negative: Tuple) -> Tuple:
    """
    Get color with intensity based on attribution magnitude.
    Deeper shade = higher attribution magnitude.
    """
    if abs(attribution) < 0.05:
        return (*COLORS['neutral'], 0.3)

    intensity = min(0.9, 0.3 + abs(attribution) * 1.2)

    if attribution > 0:
        base = base_positive
    else:
        base = base_negative

    white_mix = 1.0 - intensity
    r = base[0] * intensity + 1.0 * white_mix
    g = base[1] * intensity + 1.0 * white_mix
    b = base[2] * intensity + 1.0 * white_mix

    return (r, g, b, intensity)


# ==============================================================================
# ENSEMBLE MODEL MANAGER
# ==============================================================================

class EnsembleModelManager:
    """
    Manages ensemble models for multi-pathogen prediction with averaged attributions.
    Uses best_models.json to load the 5 best models per pathogen.
    """

    def __init__(self, project_root: str = None, device: str = None):
        self.project_root = project_root or os.getcwd()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.loaded_managers = {}
        self._setup_paths()

    def _setup_paths(self):
        for path in [self.project_root, os.path.join(self.project_root, 'Models')]:
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)

    def _find_checkpoint(self, model_dir: str, cv: int, fold: int) -> Optional[str]:
        """Find checkpoint file by walking directories."""
        pattern = f"cv{cv}_fold{fold}_best.ckpt"

        for root, _, files in os.walk(model_dir):
            for fn in files:
                if fn.endswith(pattern) or (f"cv{cv}" in fn and f"fold{fold}" in fn and fn.endswith("_best.ckpt")):
                    return os.path.join(root, fn)
        return None

    def load_pathogen_model(self, pathogen_code: str) -> bool:
        """Load ensemble models for a specific pathogen using best_models.json."""
        if pathogen_code in self.loaded_managers:
            return True

        config = PATHOGEN_CONFIGS.get(pathogen_code)
        if not config:
            print(f"  Unknown pathogen code: {pathogen_code}")
            return False

        model_dir = os.path.join(self.project_root, config.model_dir)

        if not os.path.isdir(model_dir):
            print(f"  Model directory not found: {model_dir}")
            return False

        best_models_file = os.path.join(model_dir, 'best_models.json')
        if not os.path.isfile(best_models_file):
            print(f"  best_models.json not found in {model_dir}")
            return False

        try:
            from config import get_config
            from model import BaseGNN

            cfg = get_config()
            cfg.classification = True
            cfg.task_type = 'classification'
            cfg.task_name = config.task_name
            cfg.substructure_type = 'murcko'
            cfg.output_dir = model_dir
            cfg.data_dir = model_dir

            with open(best_models_file, 'r') as f:
                json_data = json.load(f)

            if 'models' not in json_data:
                print(f"  Invalid best_models.json format")
                return False

            best_models = json_data['models'][:5]
            loaded_models = []

            for model_info in best_models:
                cv = model_info['cv']
                fold = model_info['fold']

                checkpoint_path = self._find_checkpoint(model_dir, cv, fold)

                if not checkpoint_path:
                    continue

                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    hyperparams = checkpoint['hyperparameters']
                    state_dict = checkpoint['state_dict']

                    # Detect num_edge_types from checkpoint weights for backward compatibility
                    # Legacy checkpoints may have different num_relations than config
                    detected_num_edge_types = cfg.num_edge_types
                    for key, tensor in state_dict.items():
                        if 'graph_conv_layer.weight' in key and len(tensor.shape) == 3:
                            detected_num_edge_types = tensor.shape[0]
                            break

                    # Temporarily override config for model creation
                    original_num_edge_types = cfg.num_edge_types
                    cfg.num_edge_types = detected_num_edge_types

                    model = BaseGNN(
                        config=cfg,
                        rgcn_hidden_feats=hyperparams['rgcn_hidden_feats'],
                        ffn_hidden_feats=hyperparams['ffn_hidden_feats'],
                        ffn_dropout=hyperparams['ffn_dropout'],
                        rgcn_dropout=hyperparams['rgcn_dropout'],
                        classification=cfg.classification,
                        num_classes=2 if cfg.classification else None
                    )

                    # Restore original config
                    cfg.num_edge_types = original_num_edge_types
                    model_keys = set(model.state_dict().keys())
                    filtered_dict = {k: v for k, v in state_dict.items()
                                   if k in model_keys and 'graph_conv_layer.bias' not in k}

                    model.load_state_dict(filtered_dict, strict=False)
                    model = model.to(self.device)
                    model.eval()

                    loaded_models.append({
                        'model': model,
                        'cv': cv,
                        'fold': fold,
                        'hyperparams': hyperparams
                    })
                except Exception as e:
                    import traceback
                    print(f"    Error loading cv{cv} fold{fold}: {str(e)}")
                    traceback.print_exc()
                    continue

            if loaded_models:
                self.loaded_managers[pathogen_code] = {
                    'models': loaded_models,
                    'config': cfg,
                    'threshold': config.threshold
                }
                print(f"  Loaded {len(loaded_models)} models for {config.name}")
                return True
            else:
                print(f"  No models loaded for {config.name}")
                return False

        except Exception as e:
            print(f"  Error loading {config.name}: {str(e)}")
            return False

    def predict_ensemble(self, smiles: str, pathogen_code: str) -> Dict[str, Any]:
        """
        Make ensemble prediction with averaged attributions and simple consistency check.
        """
        if pathogen_code not in self.loaded_managers:
            if not self.load_pathogen_model(pathogen_code):
                return {'error': f'Failed to load model for {pathogen_code}'}

        manager_data = self.loaded_managers[pathogen_code]
        models = manager_data['models']
        cfg = manager_data['config']
        threshold = manager_data['threshold']
        pathogen_config = PATHOGEN_CONFIGS[pathogen_code]

        try:
            from build_data import construct_mol_graph_from_smiles
            from torch_geometric.data import Batch, Data

            graph = construct_mol_graph_from_smiles(smiles, smask=[])
            if graph is None:
                return {'error': 'Failed to construct molecular graph', 'smiles': smiles}

            # Ensemble predictions
            predictions = []
            atom_weights_list = []

            for model_info in models:
                model = model_info['model']

                try:
                    with torch.no_grad():
                        graph_device = next(model.parameters()).device

                        graph_copy = Data(
                            x=graph.x.to(graph_device),
                            edge_index=graph.edge_index.to(graph_device),
                            edge_attr=graph.edge_attr.to(graph_device) if hasattr(graph, 'edge_attr') else None,
                            edge_type=graph.edge_type.to(graph_device) if hasattr(graph, 'edge_type') else None,
                        )

                        if hasattr(graph, 'smask'):
                            graph_copy.smask = graph.smask.to(graph_device)

                        batch = Batch.from_data_list([graph_copy])
                        pred, atom_weights = model(batch)

                        pred_val = torch.sigmoid(pred).item()
                        predictions.append(pred_val)

                        if atom_weights is not None:
                            atom_weights_list.append(atom_weights.cpu().numpy())
                except Exception:
                    continue

            if not predictions:
                return {'error': 'No valid predictions', 'smiles': smiles}

            # Ensemble averaged prediction
            mean_pred = float(np.mean(predictions))
            std_pred = float(np.std(predictions))
            is_active = mean_pred >= threshold

            # Get ensemble averaged attributions using substructure analysis
            scaffold_attrs = []
            scaffold_data = []

            try:
                from build_data import return_murcko_leaf_structure

                murcko_info = return_murcko_leaf_structure(smiles)
                substructures = murcko_info.get('substructure', {})

                if substructures:
                    for sub_idx, atoms in substructures.items():
                        if not atoms:
                            continue

                        # Get masked predictions from ALL models (ensemble attribution)
                        masked_preds = []
                        for model_info in models:
                            model = model_info['model']
                            try:
                                with torch.no_grad():
                                    masked_graph = construct_mol_graph_from_smiles(smiles, smask=atoms)
                                    if masked_graph is None:
                                        continue

                                    graph_device = next(model.parameters()).device
                                    masked_copy = Data(
                                        x=masked_graph.x.to(graph_device),
                                        edge_index=masked_graph.edge_index.to(graph_device),
                                        edge_attr=masked_graph.edge_attr.to(graph_device) if hasattr(masked_graph, 'edge_attr') else None,
                                        edge_type=masked_graph.edge_type.to(graph_device) if hasattr(masked_graph, 'edge_type') else None,
                                    )
                                    if hasattr(masked_graph, 'smask'):
                                        masked_copy.smask = masked_graph.smask.to(graph_device)

                                    batch = Batch.from_data_list([masked_copy])
                                    pred, _ = model(batch)
                                    masked_preds.append(torch.sigmoid(pred).item())
                            except:
                                continue

                        if masked_preds:
                            # Ensemble attribution: base ensemble pred - masked ensemble pred
                            masked_mean = np.mean(masked_preds)
                            attribution = mean_pred - masked_mean
                            scaffold_attrs.append(attribution)

                            mol = Chem.MolFromSmiles(smiles)
                            try:
                                sub_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=atoms, kekuleSmiles=True)
                            except:
                                sub_smiles = ""

                            scaffold_data.append({
                                'idx': sub_idx,
                                'smiles': sub_smiles,
                                'atoms': atoms,
                                'attribution': attribution
                            })
            except Exception:
                pass

            # Simple binary internal consistency check
            consistency = self._check_internal_consistency(
                prediction=mean_pred,
                threshold=threshold,
                attributions=scaffold_attrs
            )

            result = {
                'smiles': smiles,
                'pathogen': pathogen_config.name,
                'pathogen_code': pathogen_code,
                'prediction': mean_pred,
                'prediction_std': std_pred,
                'threshold': threshold,
                'classification': 'ACTIVE' if is_active else 'INACTIVE',
                'n_models': len(predictions),
                'internal_consistency': consistency,
                'scaffold_data': scaffold_data,
                'atom_weights': np.mean(atom_weights_list, axis=0).tolist() if atom_weights_list else None,
            }

            return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'smiles': smiles}

    def _check_internal_consistency(
        self,
        prediction: float,
        threshold: float,
        attributions: List[float]
    ) -> Dict[str, Any]:
        """
        Simple binary internal consistency check using sign-matching criterion.

        Internal consistency assesses whether explanatory evidence logically supports
        the corresponding prediction. We compute the mean attribution across all
        explanatory units and determine whether this net explanatory stance aligns
        with the predicted class direction:

        - AGREE: Active prediction (prob >= threshold) + positive mean attribution
        - AGREE: Inactive prediction (prob < threshold) + negative mean attribution
        - DISAGREE: Otherwise
        """
        result = {
            'agreement': 'N/A',
            'mean_attribution': None,
            'prediction_direction': None,
            'attribution_polarity': None,
        }

        if not attributions:
            result['agreement'] = 'N/A'
            return result

        # Compute mean attribution across all explanatory units
        mean_attr = float(np.mean(attributions))
        result['mean_attribution'] = round(mean_attr, 4)

        # Determine prediction direction
        is_active = prediction >= threshold
        result['prediction_direction'] = 'ACTIVE' if is_active else 'INACTIVE'

        # Determine attribution polarity
        result['attribution_polarity'] = 'POSITIVE' if mean_attr > 0 else 'NEGATIVE'

        # Simple sign-matching criterion
        # Active prediction should have positive mean attribution
        # Inactive prediction should have negative mean attribution
        if (is_active and mean_attr > 0) or (not is_active and mean_attr < 0):
            result['agreement'] = 'AGREE'
        else:
            result['agreement'] = 'DISAGREE'

        return result


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def draw_molecule_with_attribution(
    mol,
    highlight_atoms: List[int] = None,
    attribution: float = None,
    size: Tuple[int, int] = (600, 400),
    title: str = None
) -> str:
    """Draw molecule with intensity-based cyan/orange coloring."""
    if mol.GetNumConformers() == 0:
        rdDepictor.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DSVG(*size)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2.5

    if highlight_atoms and attribution is not None:
        color = get_intensity_color(
            attribution,
            COLORS['positive_base'],
            COLORS['negative_base']
        )

        atom_colors = {}
        bond_colors = {}

        for atom_idx in highlight_atoms:
            if atom_idx < mol.GetNumAtoms():
                atom_colors[atom_idx] = color

        for i, atom1 in enumerate(highlight_atoms):
            for atom2 in highlight_atoms[i+1:]:
                if atom1 < mol.GetNumAtoms() and atom2 < mol.GetNumAtoms():
                    bond = mol.GetBondBetweenAtoms(atom1, atom2)
                    if bond:
                        bond_colors[bond.GetIdx()] = color

        drawer.DrawMolecule(
            mol,
            highlightAtoms=list(atom_colors.keys()),
            highlightBonds=list(bond_colors.keys()),
            highlightAtomColors=atom_colors,
            highlightBondColors=bond_colors
        )
    else:
        drawer.DrawMolecule(mol)

    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def create_prediction_card(result: Dict[str, Any]) -> str:
    """Create HTML card with simple AGREE/DISAGREE consistency indicator."""
    if 'error' in result:
        return f"""
        <div style='background:#fee; border:2px solid #c00; padding:15px; border-radius:8px; margin:10px 0;'>
            <h3 style='color:#c00; margin:0;'>Error</h3>
            <p>{result['error']}</p>
        </div>
        """

    config = PATHOGEN_CONFIGS.get(result.get('pathogen_code', 'SA'))
    is_active = result['classification'] == 'ACTIVE'
    consistency = result.get('internal_consistency', {})

    if is_active:
        bg_color = '#e8f5e9'
        border_color = '#4caf50'
        text_color = '#2e7d32'
        status_icon = '&#x2713;'
    else:
        bg_color = '#ffebee'
        border_color = '#f44336'
        text_color = '#c62828'
        status_icon = '&#x2717;'

    # Simple AGREE/DISAGREE indicator
    agreement = consistency.get('agreement', 'N/A')
    if agreement == 'AGREE':
        agree_color = '#4caf50'
        agree_icon = '&#x2713;'
        agree_bg = '#e8f5e9'
    elif agreement == 'DISAGREE':
        agree_color = '#f44336'
        agree_icon = '&#x2717;'
        agree_bg = '#ffebee'
    else:
        agree_color = '#9e9e9e'
        agree_icon = '?'
        agree_bg = '#f5f5f5'

    mean_attr = consistency.get('mean_attribution', 'N/A')
    attr_polarity = consistency.get('attribution_polarity', 'N/A')

    html = f"""
    <div style='background:{bg_color}; border:2px solid {border_color}; padding:20px; border-radius:10px; margin:10px 0; font-family:Arial, sans-serif;'>
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <div>
                <h2 style='color:{config.color}; margin:0 0 10px 0;'>{config.name}</h2>
                <div style='font-size:28px; font-weight:bold; color:{text_color};'>
                    {status_icon} {result['classification']}
                </div>
            </div>
            <div style='text-align:right;'>
                <div style='font-size:14px; color:#666;'>Ensemble Prediction</div>
                <div style='font-size:24px; font-weight:bold;'>{result['prediction']:.4f}</div>
                <div style='font-size:12px; color:#999;'>(&plusmn; {result['prediction_std']:.4f}, n={result['n_models']})</div>
                <div style='font-size:12px; color:#999;'>Threshold: {result['threshold']}</div>
            </div>
        </div>

        <hr style='border:none; border-top:1px solid #ddd; margin:15px 0;'>

        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <div style='display:flex; align-items:center; gap:15px;'>
                <div>
                    <span style='font-weight:bold;'>Prediction-Explanation Agreement:</span>
                </div>
                <div style='background:{agree_bg}; border:2px solid {agree_color}; border-radius:8px; padding:8px 16px;'>
                    <span style='color:{agree_color}; font-weight:bold; font-size:18px;'>
                        {agree_icon} {agreement}
                    </span>
                </div>
            </div>
            <div style='text-align:right; font-size:12px; color:#666;'>
                Mean Attribution: {mean_attr}<br>
                Polarity: {attr_polarity}
            </div>
        </div>
    </div>
    """
    return html


def visualize_attributions(
    smiles: str,
    result: Dict[str, Any],
    max_scaffolds: int = 5
) -> None:
    """Visualize molecular attributions with intensity-based cyan/orange coloring."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Invalid SMILES - cannot visualize")
        return

    scaffold_data = result.get('scaffold_data', [])

    if not scaffold_data:
        svg = draw_molecule_with_attribution(mol, title="Molecule Structure")
        display(HTML(f"<div style='text-align:center;'>{svg}</div>"))
        return

    scaffold_data.sort(key=lambda x: abs(x['attribution']), reverse=True)
    scaffold_data = scaffold_data[:max_scaffolds]

    print(f"\n Top {len(scaffold_data)} Scaffolds by Attribution:")

    html_parts = ["<div style='display:flex; flex-wrap:wrap; gap:20px; justify-content:center;'>"]

    for scaffold in scaffold_data:
        attr = scaffold['attribution']
        atoms = scaffold.get('atoms', [])

        color = get_intensity_color(attr, COLORS['positive_base'], COLORS['negative_base'])

        if attr > 0.1:
            impact = "Increases activity"
            border_color = "#00bcd4"
        elif attr < -0.1:
            impact = "Decreases activity"
            border_color = "#ff9800"
        else:
            impact = "Minimal impact"
            border_color = "#9e9e9e"

        svg = draw_molecule_with_attribution(
            mol,
            highlight_atoms=atoms,
            attribution=attr,
            size=(350, 280)
        )

        rgb = f"rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})"

        html_parts.append(f"""
        <div style='border:2px solid {border_color}; border-radius:8px; padding:10px; background:#fafafa;'>
            <div style='text-align:center; margin-bottom:5px;'>
                <span style='display:inline-block; width:14px; height:14px; background:{rgb}; border-radius:50%; margin-right:5px; border:1px solid #333;'></span>
                <b>Attribution: {attr:.3f}</b>
            </div>
            <div style='font-size:11px; color:#666; text-align:center; margin-bottom:5px;'>{impact}</div>
            {svg}
            <div style='font-size:10px; color:#999; text-align:center; word-break:break-all; max-width:350px;'>
                {scaffold['smiles'][:50]}{'...' if len(scaffold['smiles']) > 50 else ''}
            </div>
        </div>
        """)

    html_parts.append("</div>")
    display(HTML(''.join(html_parts)))


def create_summary_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Create summary DataFrame from multi-pathogen results."""
    rows = []
    for code, result in results.items():
        if 'error' not in result:
            consistency = result.get('internal_consistency', {})
            rows.append({
                'Pathogen': result.get('pathogen', code),
                'Prediction': round(result.get('prediction', 0), 4),
                'Std Dev': round(result.get('prediction_std', 0), 4),
                'Classification': result.get('classification', 'N/A'),
                'Threshold': result.get('threshold', 'N/A'),
                'Agreement': consistency.get('agreement', 'N/A'),
                'Mean Attr': consistency.get('mean_attribution', 'N/A'),
            })

    return pd.DataFrame(rows)


# ==============================================================================
# MAIN VISUALIZATION FUNCTION
# ==============================================================================

def analyze_molecule(
    smiles: str,
    pathogens: List[str] = None,
    show_attributions: bool = True,
    manager: EnsembleModelManager = None
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze a molecule against selected pathogens with full XAI visualization.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Invalid SMILES string")
        return {'error': 'Invalid SMILES'}

    if pathogens is None:
        pathogens = ['SA', 'EC', 'CA']

    valid_pathogens = [p for p in pathogens if p in PATHOGEN_CONFIGS]
    if not valid_pathogens:
        print("No valid pathogen codes provided. Use 'SA', 'EC', or 'CA'.")
        return {'error': 'Invalid pathogen codes'}

    if manager is None:
        manager = EnsembleModelManager()

    print("="*70)
    print(f" SMILES: {smiles}")
    print(f" Atoms: {mol.GetNumAtoms()} | Bonds: {mol.GetNumBonds()}")
    print("="*70)

    svg = draw_molecule_with_attribution(mol, size=(500, 350))
    display(HTML(f"<div style='text-align:center; margin:20px 0;'>{svg}</div>"))

    results = {}

    print("\n Making ensemble predictions...")
    for code in valid_pathogens:
        result = manager.predict_ensemble(smiles, code)
        results[code] = result

    print("\n" + "="*70)
    print(" PREDICTION RESULTS (Ensemble Averaged)")
    print("="*70)

    for code in valid_pathogens:
        result = results[code]
        html = create_prediction_card(result)
        display(HTML(html))

    if len(valid_pathogens) > 1:
        print("\n Summary Table:")
        df = create_summary_table(results)
        display(df)

    if show_attributions:
        for code in valid_pathogens:
            result = results[code]
            if 'error' not in result:
                config = PATHOGEN_CONFIGS[code]
                print(f"\n" + "-"*70)
                print(f" Attribution Analysis: {config.name} (Ensemble Averaged)")
                print("-"*70)
                visualize_attributions(smiles, result)

    # Color legend
    display(HTML("""
    <div style='margin-top:20px; padding:15px; background:#f5f5f5; border-radius:8px; font-size:12px;'>
        <b>Color Legend:</b> (Deeper shade = higher magnitude)
        <div style='margin-top:10px; display:flex; gap:30px;'>
            <div>
                <span style='display:inline-block; width:16px; height:16px; background:rgb(0,128,255); border-radius:3px; vertical-align:middle;'></span>
                <span style='margin-left:5px;'>Cyan/Blue = Positive (increases activity)</span>
            </div>
            <div>
                <span style='display:inline-block; width:16px; height:16px; background:rgb(255,128,0); border-radius:3px; vertical-align:middle;'></span>
                <span style='margin-left:5px;'>Orange = Negative (decreases activity)</span>
            </div>
            <div>
                <span style='display:inline-block; width:16px; height:16px; background:rgb(178,178,178); border-radius:3px; vertical-align:middle;'></span>
                <span style='margin-left:5px;'>Gray = Neutral (minimal impact)</span>
            </div>
        </div>
    </div>
    """))

    return results


# ==============================================================================
# QUICK ACCESS FUNCTIONS
# ==============================================================================

def quick_predict(smiles: str, pathogen: str = 'SA') -> Dict[str, Any]:
    """Quick single-pathogen prediction without visualization."""
    manager = EnsembleModelManager()
    return manager.predict_ensemble(smiles, pathogen)


def batch_predict(
    smiles_list: List[str],
    pathogens: List[str] = None
) -> pd.DataFrame:
    """Batch prediction for multiple molecules."""
    if pathogens is None:
        pathogens = ['SA', 'EC', 'CA']

    manager = EnsembleModelManager()

    for code in pathogens:
        manager.load_pathogen_model(code)

    rows = []
    for i, smiles in enumerate(smiles_list):
        print(f"Processing {i+1}/{len(smiles_list)}...", end='\r')

        for code in pathogens:
            result = manager.predict_ensemble(smiles, code)

            if 'error' not in result:
                consistency = result.get('internal_consistency', {})
                rows.append({
                    'SMILES': smiles,
                    'Pathogen': result.get('pathogen'),
                    'Prediction': result.get('prediction'),
                    'Std': result.get('prediction_std'),
                    'Classification': result.get('classification'),
                    'Agreement': consistency.get('agreement'),
                    'Mean_Attr': consistency.get('mean_attribution'),
                })
            else:
                rows.append({
                    'SMILES': smiles,
                    'Pathogen': PATHOGEN_CONFIGS[code].name,
                    'Prediction': None,
                    'Classification': 'ERROR',
                    'Error': result.get('error')
                })

    print(f"Processed {len(smiles_list)} molecules" + " "*20)
    return pd.DataFrame(rows)
