# build_data.py

import os
import traceback
import pandas as pd
import functools
from rdkit import Chem
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdRGroupDecomposition
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem
from rdkit.Chem import BRICS
import numpy as np
import torch
from torch_geometric.data import Data
import itertools
import logging
import random
from typing import List, Tuple, Dict, Any, Optional
from logger import get_logger
from config import Configuration, config   
#from config2 import Configuration, config 
# Initialize logger
logger = get_logger(__name__)

# Utility Functions
def return_fg_without_c_i_wash(fg_with_c_i, fg_without_c_i):
    fg_without_c_i_wash = []
    for fg_with_c in fg_with_c_i:
        for fg_without_c in fg_without_c_i:
            if set(fg_without_c).issubset(set(fg_with_c)):
                fg_without_c_i_wash.append(list(fg_without_c))
    return fg_without_c_i_wash

def return_fg_hit_atom(smiles, fg_name_list, fg_with_ca_list, fg_without_ca_list):
    mol = Chem.MolFromSmiles(smiles)
    hit_at = []
    hit_fg_name = []
    all_hit_fg_at = []
    for i in range(len(fg_with_ca_list)):
        fg_with_c_i = mol.GetSubstructMatches(fg_with_ca_list[i])
        fg_without_c_i = mol.GetSubstructMatches(fg_without_ca_list[i])
        fg_without_c_i_wash = return_fg_without_c_i_wash(fg_with_c_i, fg_without_c_i)
        if len(fg_without_c_i_wash) > 0:
            hit_at.append(fg_without_c_i_wash)
            hit_fg_name.append(fg_name_list[i])
            all_hit_fg_at += fg_without_c_i_wash

    sorted_all_hit_fg_at = sorted(all_hit_fg_at, key=lambda fg: len(fg), reverse=True)

    remain_fg_list = []
    for fg in sorted_all_hit_fg_at:
        if fg not in remain_fg_list:
            if len(remain_fg_list) == 0:
                remain_fg_list.append(fg)
            else:
                i = 0
                for remain_fg in remain_fg_list:
                    if set(fg).issubset(set(remain_fg)):
                        break
                    else:
                        i += 1
                if i == len(remain_fg_list):
                    remain_fg_list.append(fg)

    hit_at_wash = []
    hit_fg_name_wash = []
    for j in range(len(hit_at)):
        hit_at_wash_j = []
        for fg in hit_at[j]:
            if fg in remain_fg_list:
                hit_at_wash_j.append(fg)
        if len(hit_at_wash_j) > 0:
            hit_at_wash.append(hit_at_wash_j)
            hit_fg_name_wash.append(hit_fg_name[j])
    return hit_at_wash, hit_fg_name_wash

def getAllBricsBondSubset(BricsBond, max_subsets=10000):
    """
    Generate all possible combinations of BRICS bonds up to a maximum limit.
    Args:
        BricsBond: List of BRICS bonds
        max_subsets: Maximum number of combinations to generate
    Returns:
        List of bond subset combinations
    """
    all_brics_bond_subset = []
    N = len(BricsBond)
    
    # Generate all possible combinations using binary counting
    for i in range(2 ** N):
        brics_bond_subset = []
        for j in range(N):
            if (i >> j) % 2:
                brics_bond_subset.append(BricsBond[j])
        if len(brics_bond_subset) > 0:
            all_brics_bond_subset.append(brics_bond_subset)
        if len(all_brics_bond_subset) > max_subsets:
            logger.warning(f"Reached maximum subset limit of {max_subsets}")
            break
    return all_brics_bond_subset



def get_fragment_atoms(mol, start_atom, exclude=None):
    """Get connected atoms in a fragment."""
    if exclude is None:
        exclude = []
        
    visited = set([start_atom])
    to_visit = set([start_atom])
    exclude = set(exclude)
    
    while to_visit:
        atom_idx = to_visit.pop()
        atom = mol.GetAtomWithIdx(atom_idx)
        
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if (neighbor_idx not in visited and 
                neighbor_idx not in exclude):
                visited.add(neighbor_idx)
                to_visit.add(neighbor_idx)
    
    return list(visited)

def return_brics_leaf_structure(smiles):
    """
    BRICS decomposition focusing on medicinally relevant synthetic breaks.
    Prioritizes breaks at common medicinal chemistry connection points
    like amides, amines, esters, aromatic linkages, and ethers.

    Returns a dictionary with:
    - 'substructure': A dict of fragments indexed by integer keys.
    - 'reaction_centers': A dict containing the atoms where the chosen break occurs.
    - 'brics_bond_types': A list of the chosen break type(s).
    """
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Priority BRICS rules for medicinal chemistry
    priority_breaks = {
        '7': 'amide',      
        '5': 'amine',      
        '6': 'ester',      
        '4': 'aromatic',   
        '3': 'ether'
    }

    # Find all BRICS bonds
    brics_bonds = list(BRICS.FindBRICSBonds(m))
    
    all_brics_substructure_subset = dict()

    if brics_bonds:
        # Evaluate and sort BRICS bonds by defined priority
        prioritized_bonds = []
        for (atom1, atom2), (break_type1, break_type2) in brics_bonds:
            # Determine priority
            p1 = int(break_type1) if break_type1 in priority_breaks else 99
            p2 = int(break_type2) if break_type2 in priority_breaks else 99
            priority = min(p1, p2)

            # Only consider if at least one of the break_types is in priority_breaks
            if priority < 99:
                prioritized_bonds.append((priority, (atom1, atom2), (break_type1, break_type2)))

        # Sort by priority, lowest is highest priority
        prioritized_bonds.sort()

        if prioritized_bonds:
            # Take the top-priority bond
            _, (atom1, atom2), (break_type1, break_type2) = prioritized_bonds[0]
            chosen_break_type = break_type1 if break_type1 in priority_breaks else break_type2

            # Extract fragments after this priority break
            fragment1 = get_fragment_atoms(m, atom1, exclude=[atom2])
            fragment2 = get_fragment_atoms(m, atom2, exclude=[atom1])
            
            substrate_idx = {
                0: fragment1,
                1: fragment2
            }

            # Reaction centers: the atoms at the break
            reaction_centers = {0: [atom1, atom2]}

            # Store chosen BRICS bond type(s)
            # Using a list here, but you could store more detail if needed
            brics_bond_types = [chosen_break_type]

            all_brics_substructure_subset['substructure'] = substrate_idx
            all_brics_substructure_subset['reaction_centers'] = reaction_centers
            all_brics_substructure_subset['brics_bond_types'] = brics_bond_types
        else:
            # No priority breaks found; fallback to whole molecule
            substrate_idx = {0: [x for x in range(m.GetNumAtoms())]}
            all_brics_substructure_subset['substructure'] = substrate_idx
            all_brics_substructure_subset['reaction_centers'] = {}
            all_brics_substructure_subset['brics_bond_types'] = []
    else:
        # No BRICS bonds at all; just return the whole molecule as a single substructure
        substrate_idx = {0: [x for x in range(m.GetNumAtoms())]}
        all_brics_substructure_subset['substructure'] = substrate_idx
        all_brics_substructure_subset['reaction_centers'] = {}
        all_brics_substructure_subset['brics_bond_types'] = []

    return all_brics_substructure_subset

def return_pure_murcko_scaffold_structures(smiles: str) -> Dict[int, List[int]]:
    """
    Enumerate all Murcko core scaffolds (full → individual rings) without using scaffoldgraph.
    Returns a mapping {i: atom_index_list} for each pure scaffold.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    
    # Get the Murcko scaffold (rings and linkers only)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    
    # Generate clean scaffold SMILES
    scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=False)
    
    # Get all sub-scaffolds
    all_scaffold_smiles = _all_murcko_smiles(scaffold_smiles)
    scaff_mols = []
    
    # Convert SMILES to molecules
    for smi in all_scaffold_smiles:
        try:
            scaff_mol = Chem.MolFromSmiles(smi)
            if scaff_mol is not None:
                scaff_mols.append(scaff_mol)
        except:
            continue
    
    # Map each scaffold to its atoms in the original molecule
    scaff_dict = {}
    for i, scf in enumerate(scaff_mols):
        match = mol.GetSubstructMatch(scf)
        if match:
            scaff_dict[i] = list(match)
    
    return scaff_dict


# 1) Cache recursive enumeration of ALL sub‑scaffolds:
@functools.lru_cache(maxsize=None)
def _all_murcko_smiles(smiles: str):
    """
    Return a tuple of SMILES for every Murcko core from full -> single ring.
    Cached so each unique input is done once.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ()
    # full scaffold - remove includeChirality parameter
    full = MurckoScaffold.GetScaffoldForMol(mol)  # Remove includeChirality=False
    seen = {Chem.MolToSmiles(full, isomericSmiles=False)}
    queue = [full]
    while queue:
        scf = queue.pop(0)
        for ring in scf.GetRingInfo().AtomRings():
            rw = Chem.RWMol(scf)
            for idx in sorted(ring, reverse=True):
                try: rw.RemoveAtom(idx)
                except: pass
            new = rw.GetMol()
            try:
                Chem.SanitizeMol(new)
                smi = Chem.MolToSmiles(new, isomericSmiles=False)
            except:
                continue
            if smi and smi not in seen:
                seen.add(smi)
                queue.append(new)
    return tuple(seen)

# 2) Cache the full decomposition into scaffolds + tagged R‑groups:
@functools.lru_cache(maxsize=None)
def return_murcko_leaf_structure(smiles: str) -> Dict[str, Dict[int, Any]]:
    m = Chem.MolFromSmiles(smiles)
    scaffolds = _all_murcko_smiles(smiles)
    substructure = {}
    for i, scf_smi in enumerate(scaffolds):
        scf_mol = Chem.MolFromSmiles(scf_smi)
        match = m.GetSubstructMatch(scf_mol)
        if match:
            substructure[i] = list(match)

    # R‑group decomposition on *full* scaffold only
    full = Chem.MolFromSmiles(scaffolds[0]) if scaffolds else None
    substituents = {}
    if full:
        try:
            params = rdRGroupDecomposition.RGroupDecompositionParameters()
            params.RGroupLabelling = None
            # Use RGroupDecompose directly instead of RGroupDecomposition
            groups, fails = rdRGroupDecomposition.RGroupDecompose([full], [m], asRows=True, options=params)
            
            idx = 0
            for row in groups:
                for rname, frag in row.items():
                    if rname == 'Core': continue
                    # Tag context by whether attachment point is aromatic
                    try:
                        attach = next(a for a in frag.GetAtoms() if a.GetSymbol()=='*')
                        nbr = attach.GetNeighbors()[0]
                        ctx = 'aromatic' if nbr.GetIsAromatic() else 'aliphatic'
                        frag_smi = Chem.MolToSmiles(frag, isomericSmiles=False)
                        submol = Chem.MolFromSmiles(frag_smi)
                        match = m.GetSubstructMatch(submol)
                        if match:
                            substituents[idx] = (list(match), f"{rname}_{ctx}")
                            idx += 1
                    except (StopIteration, IndexError):
                        continue
        except Exception as e:
            logger.error(f"Error in R-group decomposition: {str(e)}")

    return {'substructure': substructure, 'substituents': substituents}
	
def get_scaffold_hierarchy(smiles):
    """
    Get hierarchical scaffold decomposition with substituents.
    Returns a dictionary with hierarchical scaffolds and substituents.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": "Invalid SMILES"}
    
    # Get all hierarchical scaffolds
    scaffolds = _all_murcko_smiles(smiles)
    scaffold_mols = [Chem.MolFromSmiles(s) for s in scaffolds]
    
    # Get substituents for each scaffold
    result = {
        "scaffolds": [],
        "substituents": []
    }
    
    for i, scaffold_smi in enumerate(scaffolds):
        scaffold_mol = Chem.MolFromSmiles(scaffold_smi)
        result["scaffolds"].append({
            "smiles": scaffold_smi,
            "level": i,
            "mol": scaffold_mol
        })
        
        # Get substituents for this scaffold
        try:
            params = rdRGroupDecomposition.RGroupDecompositionParameters()
            params.RGroupLabelling = None
            # Use RGroupDecompose directly instead of RGroupDecomposition
            groups, fails = rdRGroupDecomposition.RGroupDecompose([scaffold_mol], [mol], asRows=True, options=params)
            
            for row in groups:
                for rname, frag in row.items():
                    if rname == 'Core': continue
                    try:
                        # Tag context by whether attachment point is aromatic
                        attach = next(a for a in frag.GetAtoms() if a.GetSymbol()=='*')
                        nbr = attach.GetNeighbors()[0]
                        ctx = 'aromatic' if nbr.GetIsAromatic() else 'aliphatic'
                        frag_smi = Chem.MolToSmiles(frag, isomericSmiles=False)
                        
                        result["substituents"].append({
                            "smiles": frag_smi,
                            "parent_scaffold": scaffold_smi,
                            "attachment_type": ctx,
                            "r_group": rname,
                            "mol": frag
                        })
                    except (StopIteration, IndexError):
                        continue
        except Exception as e:
            logger.error(f"Error in scaffold decomposition: {str(e)}")
            continue
    
    return result


def analyze_molecule_with_scaffolds(smiles):
    """
    Analyze a molecule and provide detailed information about its scaffolds and substituents.
    """
    hierarchy = get_scaffold_hierarchy(smiles)
    
    if "error" in hierarchy:
        return hierarchy
    
    # Prepare a detailed report
    report = {
        "molecule_smiles": smiles,
        "num_scaffolds": len(hierarchy["scaffolds"]),
        "num_substituents": len(hierarchy["substituents"]),
        "scaffolds": [],
        "substituents": []
    }
    
    # Add scaffold details
    for i, scaffold in enumerate(hierarchy["scaffolds"]):
        report["scaffolds"].append({
            "level": i + 1,
            "smiles": scaffold["smiles"],
            "num_atoms": scaffold["mol"].GetNumAtoms() if scaffold["mol"] else 0
        })
    
    # Add substituent details
    for i, subst in enumerate(hierarchy["substituents"]):
        report["substituents"].append({
            "id": i + 1,
            "smiles": subst["smiles"],
            "parent_scaffold": subst["parent_scaffold"],
            "attachment_type": subst["attachment_type"],
            "r_group": subst["r_group"]
        })
    
    return report
	
# 3) Rewrite the Murcko builder to use that cache:
def build_mol_graph_data_for_murcko(dataset_smiles, labels_name, smiles_name):
    dataset = []
    df = dataset_smiles
    for i, smiles in enumerate(df[smiles_name]):
        y = df[labels_name].iloc[i]
        grp = df['group'].iloc[i]
        try:
            info = return_murcko_leaf_structure(smiles)
            # scaffolds first
            for scf_idx, atoms in info['substructure'].items():
                g = construct_mol_graph_from_smiles(smiles, smask=atoms)
                dataset.append([smiles, g, y, grp, atoms, i, f"scaffold_{scf_idx}"])
            # then substituents
            for sub_idx, (atoms, tag) in info['substituents'].items():
                g = construct_mol_graph_from_smiles(smiles, smask=atoms)
                dataset.append([smiles, g, y, grp, atoms, i, tag])
        except Exception as e:
            logger.error(f"Murcko build failed for {smiles}: {e}")
    return dataset

def atom_features(atom, use_chirality=True):
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        [
            'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As',
            'Se', 'Br', 'Te', 'I', 'At', 'other'
        ]) + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                  Chem.rdchem.HybridizationType.SP3D2, 'other']) + [atom.GetIsAromatic()]
    results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'), ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]

    return np.array(results)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def etype_features(bond, use_chirality=True):
    features = []
    
    # Bond type
    bond_type = bond.GetBondType()
    features.extend([
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC
    ])
    
    # Other features
    features.append(bond.GetIsConjugated())
    features.append(bond.IsInRing())
    
    if use_chirality:
        stereo = str(bond.GetStereo())
        features.extend([
            stereo == "STEREONONE",
            stereo == "STEREOANY",
            stereo == "STEREOZ",
            stereo == "STEREOE"
        ])
    
    return features  # This will be a list of boolean values



def construct_mol_graph_from_smiles(smiles, smask):
    """Construct molecular graph from SMILES with enhanced dimension validation."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        edge_index = []
        edge_attr = []
        edge_type = []
        x = []
        
        # Collect atom features
        for atom in mol.GetAtoms():
            atom_feats = atom_features(atom)
            x.append(atom_feats)
            
        # Log the original feature dimension
        original_dim = len(x[0]) if x else 0
        logger.debug(f"Original atom feature dimension: {original_dim}")
        
        # Collect bond information
        # INTENTIONAL DESIGN: 3-edge Relational GCN
        # Edge types: SINGLE=1, DOUBLE=2, TRIPLE=3
        # Aromatic bonds exist in graph but have UNDEFINED type (-1)
        # which gets filtered out during message passing in the model.
        # Aromaticity is instead captured via node features (atom.GetIsAromatic()).
        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.extend([[u, v], [v, u]])
            bond_features = etype_features(bond)
            edge_attr.extend([bond_features, bond_features])

            # Explicit edge type assignment for clean, reproducible behavior
            if bond.GetIsAromatic():
                bond_type = -1  # UNDEFINED - excluded from typed message passing
            elif bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                bond_type = 1
            elif bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                bond_type = 2
            elif bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                bond_type = 3
            else:
                bond_type = 1  # Default to SINGLE for any other bond type

            edge_type.extend([bond_type, bond_type])
        
        # Convert to tensors
        x = torch.tensor(np.array(x), dtype=torch.float)
        edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
        edge_type = torch.tensor(np.array(edge_type), dtype=torch.float)
        
        # Ensure x has exactly 40 features with detailed logging
        if x.shape[1] != 40:
            logger.warning(
                f"Node features dimension mismatch for SMILES {smiles}. "
                f"Current dimension: {x.shape[1]}, Required: 40"
            )
            
            if x.shape[1] > 40:
                logger.debug("Truncating features to first 40 dimensions")
                x = x[:, :40]
            else:
                logger.debug(f"Padding features with zeros (adding {40 - x.shape[1]} dimensions)")
                padding = torch.zeros(x.shape[0], 40 - x.shape[1])
                x = torch.cat([x, padding], dim=1)
            
            logger.info(f"Final feature dimension after adjustment: {x.shape[1]}")
        
        # Validate final dimensions, uncomment to debug
        #assert x.shape[1] == 40, f"Feature dimension is {x.shape[1]}, expected 40" 
        
        # Create and validate the graph data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type)
        
        # IMPROVED: Only add smask if atoms are actually being masked
        if smask and len(smask) > 0:
            # Create mask tensor where masked atoms get 0, others get 1
            num_atoms = data.x.shape[0]
            mask = torch.ones(num_atoms, dtype=torch.float32)
            for atom_idx in smask:
                if atom_idx < num_atoms:
                    mask[atom_idx] = 0.0
            data.smask = mask
            logger.debug(f"Created smask with {torch.sum(mask == 0.0)} masked atoms")
        # If smask is empty or None, don't add smask attribute at all
        
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        
        return data
        
    except Exception as e:
        logger.error(f"Error constructing graph for SMILES {smiles}: {str(e)}")
        raise

        
def build_mol_graph_data(dataset_smiles, labels_name, smiles_name):
    dataset_gnn = []
    failed_molecule = []
    labels = dataset_smiles[labels_name]
    split_index = dataset_smiles['group']
    smilesList = dataset_smiles[smiles_name]
    molecule_number = len(smilesList)
    
    for i, smiles in enumerate(smilesList):
        try:
            g_rgcn = construct_mol_graph_from_smiles(smiles, smask=[])
            molecule = [smiles, g_rgcn, labels.iloc[i], split_index.iloc[i]]  # Use iloc for safe indexing
            dataset_gnn.append(molecule)
            logger.info(f'{i + 1}/{molecule_number} molecule is transformed to mol graph! {len(failed_molecule)} is transformed failed!')
        except Exception as e:
            logger.error(f'{smiles} is transformed to mol graph failed: {e}')
            failed_molecule.append(smiles)
    
    logger.info(f'{failed_molecule}({len(failed_molecule)}) is transformed to mol graph failed!')
    return dataset_gnn


def build_mol_graph_data_for_brics(dataset_smiles, labels_name, smiles_name):
    """
    Build molecular graphs with enhanced BRICS decomposition.
    """
    dataset_gnn = []
    failed_molecule = []
    labels = dataset_smiles[labels_name]
    split_index = dataset_smiles['group']
    smilesList = dataset_smiles[smiles_name]
    molecule_number = len(smilesList)
    
    for i, orig_smiles in enumerate(smilesList):
        try:
            substructure_dir = return_brics_leaf_structure(orig_smiles)
            
            # Create masks for both fragments and reaction centers
            atom_mask = []
            brics_substructure_mask = []
            reaction_center_mask = []
            
            # Process regular fragments
            for _, substructure in substructure_dir['substructure'].items():
                brics_substructure_mask.append(substructure)
                atom_mask = atom_mask + substructure
            
            # Process reaction centers
            for _, center in substructure_dir['reaction_centers'].items():
                reaction_center_mask.append(center)
            
            # Combine masks for processing
            smask = [[x] for x in range(len(atom_mask))] + brics_substructure_mask + reaction_center_mask
            
            # Create graphs for each mask
            for j, smask_i in enumerate(smask):
                try:
                    g = construct_mol_graph_from_smiles(orig_smiles, smask=smask_i)
                    
                    # Store additional information in the graph
                    g.brics_bond_types = substructure_dir['brics_bond_types']
                    g.is_reaction_center = j >= len(atom_mask) + len(brics_substructure_mask)
                    
                    molecule = [orig_smiles, g, labels.iloc[i], split_index.iloc[i], smask_i, i]
                    dataset_gnn.append(molecule)
                    
                    logger.info(f'{j + 1}/{len(smask)}, {i + 1}/{molecule_number} molecule processed')
                except Exception as e:
                    logger.error(f'{orig_smiles} with smask {smask_i} failed: {e}')
                    failed_molecule.append(orig_smiles)
        except Exception as e:
            logger.error(f'Error processing molecule {orig_smiles}: {e}')
            failed_molecule.append(orig_smiles)
    
    logger.info(f'{len(failed_molecule)} molecules failed processing')
    return dataset_gnn
    


def build_mol_graph_data_for_fg(dataset_smiles, labels_name, smiles_name, config: Configuration):
    dataset_gnn = []
    failed_molecule = []
    labels = dataset_smiles[labels_name]
    split_index = dataset_smiles['group']
    smilesList = dataset_smiles[smiles_name]
    molecule_number = len(smilesList)
    
    # Get SMARTS patterns from config
    fg_with_ca_smarts = config.fg_with_ca_smart
    fg_without_ca_smarts = config.fg_without_ca_smart
    fg_name_list = [f'fg_{i}' for i in range(len(fg_with_ca_smarts))]
    
    # Precompile SMARTS patterns
    fg_with_ca_list = [Chem.MolFromSmarts(smarts) for smarts in fg_with_ca_smarts]
    fg_without_ca_list = [Chem.MolFromSmarts(smarts) for smarts in fg_without_ca_smarts]
    
    for i, orig_smiles in enumerate(smilesList):
        try:
            hit_at_wash, hit_fg_name_wash = return_fg_hit_atom(orig_smiles, fg_name_list, fg_with_ca_list, fg_without_ca_list)
            atom_mask = []  # Added to match BRICS/Murcko pattern
            fg_substructure_mask = []  # Added to match BRICS/Murcko pattern
            
            # Construct mask list similar to BRICS/Murcko
            for fg_atoms in hit_at_wash:
                fg_substructure_mask.append(fg_atoms)
                atom_mask = atom_mask + fg_atoms
            smask = fg_substructure_mask  # Match BRICS/Murcko pattern
            
            for j, smask_i in enumerate(smask):
                try:
                    g = construct_mol_graph_from_smiles(orig_smiles, smask=smask_i)  # Removed config parameter to match others
                    molecule = [orig_smiles, g, labels.iloc[i], split_index.iloc[i], smask_i, i]  # Same 6-element structure
                    dataset_gnn.append(molecule)
                    logger.info(f'{j + 1}/{len(smask)}, {i + 1}/{molecule_number} molecule is transformed to mol graph! {len(failed_molecule)} is transformed failed!')
                except Exception as e:
                    logger.error(f'{orig_smiles} with smask {smask_i} is transformed to mol graph failed: {e}')
                    failed_molecule.append(orig_smiles)
        except Exception as e:
            logger.error(f'Error processing molecule {orig_smiles}: {e}')
            failed_molecule.append(orig_smiles)
    
    logger.info(f'{failed_molecule}({len(failed_molecule)}) is transformed to mol graph failed!')
    return dataset_gnn

def save_dataset(
    dataset: Any,
    sub_type: str,
    config: Configuration,
    graph_path: Optional[str] = None,
    meta_path: Optional[str] = None,
    smask_save_path: Optional[str] = None
) -> None:
    """
    Saves the dataset and its corresponding metadata while maintaining compatibility 
    with the old output format.

    Args:
        dataset (Any): The dataset object to save.
        sub_type (str): The substructure type ('primary', 'brics', 'murcko', 'fg').
        config (Configuration): Configuration object containing paths and task details.
        graph_path (Optional[str]): Custom path to save the graph data.
        meta_path (Optional[str]): Custom path to save the meta data.
        smask_save_path (Optional[str]): Custom path to save the smask data (for non-primary types).
    """
    if not dataset:
        logger.info(f"No {sub_type} molecules were successfully processed.")
        return

    # Determine task type based on configuration
    task_type = 'classification' if config.classification else 'regression'

    if sub_type == 'primary':
        # Handle primary molecules (4 elements)
        smiles, g_pyg, labels, split_index = map(list, zip(*dataset))
        graph_labels = {'labels': torch.tensor(labels)}

        # Create DataFrame matching old format using config values
        split_index_pd = pd.DataFrame({
            'smiles': smiles,
            'group': split_index,
            config.compound_id_name: config.dataset_origin[config.compound_id_name],
            config.labels_name: config.dataset_origin[config.labels_name]
        })

        # Save files with custom or default paths
        if graph_path is None:
            graph_filename = f"{config.task_name}_{task_type}_{sub_type}_graphs.pt"
            graph_path = os.path.join(config.output_dir, graph_filename)
        if meta_path is None:
            meta_filename = f"{config.task_name}_{task_type}_{sub_type}_meta.csv"
            meta_path = os.path.join(config.output_dir, meta_filename)
        
        split_index_pd.to_csv(meta_path, index=False)
        torch.save((g_pyg, graph_labels), graph_path)
        
    else:
        # Handle BRICS, Murcko, FG (6 elements)
        smiles, g_pyg, labels, split_index, smask, orig_indices = map(list, zip(*dataset))
        graph_labels = {'labels': torch.tensor(labels)}

        # Create DataFrame matching old format using config values
        split_index_pd = pd.DataFrame({
            'smiles': smiles,
            'group': split_index,
            config.compound_id_name: [config.dataset_origin[config.compound_id_name].iloc[i] for i in orig_indices],
            config.labels_name: [config.dataset_origin[config.labels_name].iloc[i] for i in orig_indices]
        })

        # Save files with custom or default paths
        if graph_path is None:
            graph_filename = f"{config.task_name}_{task_type}_{sub_type}_graphs.pt"
            graph_path = os.path.join(config.output_dir, graph_filename)
        if meta_path is None:
            meta_filename = f"{config.task_name}_{task_type}_{sub_type}_meta.csv"
            meta_path = os.path.join(config.output_dir, meta_filename)
        if smask_save_path is None:
            smask_filename = f"{config.task_name}_{task_type}_{sub_type}_smask.npy"
            smask_save_path = os.path.join(config.output_dir, smask_filename)
        
        split_index_pd.to_csv(meta_path, index=False)
        torch.save((g_pyg, graph_labels), graph_path)
        np.save(smask_save_path, np.array(smask, dtype=object))

    logger.info(f'{sub_type.capitalize()} molecules graph and metadata saved successfully!')

def main():
    # Initialize Configuration
    config = Configuration()
    config.validate()  # Ensure configurations are valid

    # Set random seed for reproducibility
    config.set_seed(seed=42)  # You can choose any seed value

    # Load your dataset and set it in config
    dataset_smiles = pd.read_csv(config.origin_data_path)
    config.dataset_origin = dataset_smiles  # Add this line
    
    # Build datasets
    logger.info("Building primary molecule graphs...")
    dataset_gnn_primary = build_mol_graph_data(dataset_smiles, config.labels_name, config.smiles_name)
    
    if dataset_gnn_primary:  # Add check for successful processing
        logger.info("Saving primary molecule graphs and metadata...")
        save_dataset(dataset_gnn_primary, 'primary', config)
    
    logger.info("Building BRICS molecule graphs...")
    dataset_gnn_brics = build_mol_graph_data_for_brics(dataset_smiles, config.labels_name, config.smiles_name)
    
    if dataset_gnn_brics:  # Add check for successful processing
        logger.info("Saving BRICS molecule graphs and metadata...")
        save_dataset(dataset_gnn_brics, 'brics', config)
    
    logger.info("Building Murcko molecule graphs...")
    dataset_gnn_murcko = build_mol_graph_data_for_murcko(dataset_smiles, config.labels_name, config.smiles_name)
    
    if dataset_gnn_murcko:  # Add check for successful processing
        logger.info("Saving Murcko molecule graphs and metadata...")
        save_dataset(dataset_gnn_murcko, 'murcko', config)
    
    # Fix FG function call to include config
    logger.info("Building FG molecule graphs...")
    dataset_gnn_fg = build_mol_graph_data_for_fg(dataset_smiles, config.labels_name, config.smiles_name, config)
    
    if dataset_gnn_fg:  # Add check for successful processing
        logger.info("Saving FG molecule graphs and metadata...")
        save_dataset(dataset_gnn_fg, 'fg', config)
    
    logger.info("All datasets and metadata have been saved successfully.")

if __name__ == "__main__":
    main()
