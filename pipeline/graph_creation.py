import numpy as np
from pysmiles import read_smiles
from tqdm import tqdm

import rdkit
from rdkit import Chem

import numpy as np

import torch
import torch_geometric
from torch_geometric.data import Data


def get_node_features(atom):
    features = []
    
    # -1: leave unchanged
    #  0: add OHE
    #  1: replace with OHE
    
    features.append((atom.GetAtomicNum(), 1, [6, 8, 7, 16, 17, 9]))    
    features.append((atom.GetChiralTag(), 1, [rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                                              rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                                              rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW]))
    
    features.append((atom.GetDegree(), 0, [2, 3, 1]))
    features.append((atom.GetTotalDegree(), 0, [3, 4, 2, 1]))
    
    features.append((atom.GetExplicitValence(), 0, [4, 1, 2, 3]))
    features.append((atom.GetImplicitValence(), 0, [0, 1, 2, 3]))
    features.append((atom.GetTotalValence(), 0, [4, 2, 3, 1, 6]))
    
    features.append((atom.GetFormalCharge(), 0, [0, 1, -1]))
    features.append((atom.GetHybridization(), 1, [rdkit.Chem.rdchem.HybridizationType.SP3,
                                                  rdkit.Chem.rdchem.HybridizationType.SP2]))
    
    features.append((atom.GetNumExplicitHs(), 0, [0, 1]))
    features.append((atom.GetNumImplicitHs(), 0, [0, 1, 2, 3]))
    features.append((atom.GetTotalNumHs(), 0, [0, 1, 2, 3]))
    
    # features.append((atom.GetMass(), -1))
    # features.append((atom.GetNumRadicalElectrons(), 0))

    features.append((atom.GetIsAromatic(), -1))
    features.append((atom.IsInRing(), -1))
    # features.append((atom.HasOwningMol(), -1))
    
    result = []
    for feature in features:
        value, mode = feature[0], feature[1]
        if mode < 1:
            result.append(float(value))
        if mode > -1:
            result.extend([float(value == item) for item in feature[2]])
            
    return result


def get_edge_features(bond):
    features = []
    
    # -1: leave unchanged
    #  0: add OHE
    #  1: replace with OHE
    
    features.append((bond.GetBondType(), 1, [rdkit.Chem.rdchem.BondType.SINGLE,
                                             rdkit.Chem.rdchem.BondType.DOUBLE,
                                             rdkit.Chem.rdchem.BondType.AROMATIC]))    
    features.append((bond.GetIsAromatic(), -1))
    features.append((bond.GetIsConjugated(), -1))
    
        
    result = []
    for feature in features:
        value, mode = feature[0], feature[1]
        if mode < 1:
            result.append(float(value))
        if mode > -1:
            result.extend([float(value == item) for item in feature[2]])
            
    return result


def build_graph(molecule):
    atoms, bonds = list(molecule.GetAtoms()), list(molecule.GetBonds())
    node_features = np.asarray([get_node_features(atom) for atom in atoms], dtype=np.float32)
    
    edge_indices, edge_features = [], []
    for bond in bonds:
        index = bond.GetBeginAtomIdx()
        jndex = bond.GetEndAtomIdx()
        edge_indices.append([index, jndex])
        edge_indices.append([jndex, index])
        
        tmp = get_edge_features(bond)
        edge_features.append(tmp)
        edge_features.append(tmp)
    
    edge_indices = np.asarray(edge_indices, dtype=np.int64).T
    edge_features = np.asarray(edge_features, dtype=np.float32)
    
    return Data(x=torch.FloatTensor(node_features),
                edge_index=torch.LongTensor(edge_indices),
                edge_attr=torch.FloatTensor(edge_features), empty=len(bonds) == 0)


def build_graphs(smiles):
    molecules = [Chem.MolFromSmiles(smile) for smile in tqdm(smiles)]
    graphs = [build_graph(molecule) for molecule in tqdm(molecules)]
    return graphs