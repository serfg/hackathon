import numpy as np
from pysmiles import read_smiles
from tqdm import tqdm
from rdkit import Chem

import torch
import torch_geometric
from torch_geometric.data import Data


class Encoder():
    def __init__(self, features, print_unique = True):
        reshaped = [[el] for el in features[0]]
        for i in range(1, len(features)):
            for j in range(len(reshaped)):
                reshaped[j].append(features[i][j])
        self.unique = [list(np.unique(values)) for values in reshaped]
        #if (print_unique):
        #    for i in range(len(self.unique)):
        #        print(i, self.unique[i])
                
                
    def transform(self, features, show_progress = True):
        result = []
        for vector in tqdm(features, disable = not show_progress):
            now = []
            for i in range(len(vector)):
                current = np.zeros(len(self.unique[i]))
                current[self.unique[i].index(vector[i])] = 1.0
                now.append(current)
            now = np.concatenate(now, axis = 0)
            result.append(now)
        return np.array(result)
    

def get_single_raw_node_features(atom):
    now = []
    now.append(atom.GetAtomicNum())
    now.append(atom.GetChiralTag())
    now.append(atom.GetTotalDegree())
    now.append(atom.GetFormalCharge())
    now.append(atom.GetTotalNumHs())
    now.append(atom.GetNumRadicalElectrons())
    now.append(atom.GetHybridization())
    now.append(atom.GetIsAromatic())
    now.append(atom.IsInRing())
    return now


def get_raw_node_features(mol):
    features = []
    for atom in mol.GetAtoms():
        features.append(get_single_raw_node_features(atom))
    return features


def get_single_raw_bond_features(bond):
    now = []
    now.append(str(bond.GetBondType()))
    now.append(str(bond.GetStereo()))
    now.append(bond.GetIsConjugated())
    return now


def get_raw_bond_features(mol):
    features = []
    for bond in mol.GetBonds():
        features.append(get_single_raw_bond_features(bond))
    return features


def mol_2_pytorch_geometric(mol, node_encoder, bond_encoder, extra_features=None):
    node_features = node_encoder.transform(get_raw_node_features(mol), show_progress = False)
    if extra_features is not None:
        node_features = np.concatenate([node_features, extra_features], axis=1)
    #print("node features: ", node_features.shape)
    
    edge_indices, edge_attrs = [], []
    
    num_bond_features = None
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_features = bond_encoder.transform([get_single_raw_bond_features(bond)], show_progress = False)
        num_bond_features = len(bond_features[0])
        #print("bond features: ", bond_features[0].shape)
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [bond_features[0], bond_features[0]]
    
    if len(edge_attrs) > 0:
        
   
        edge_index = torch.tensor(edge_indices)
        edge_index = edge_index.t().to(torch.long).view(2, -1)
        edge_attrs = torch.tensor(edge_attrs, dtype=torch.long).view(-1, num_bond_features)
        ''' print(edge_attrs.shape)
        print(edge_index.shape)
        print(node_features.shape)
        print(edge_index)'''
        # Sort indices.
  
        perm = (edge_index[0] * node_features.shape[0] + edge_index[1]).argsort()
        #print("perm: ", perm.shape)
        #print(edge_index[0] * node_features.shape[0] + edge_index[1])
        edge_index, edge_attrs = edge_index[:, perm], edge_attrs[perm]
        data = Data(x=torch.FloatTensor(node_features), edge_index=edge_index, edge_attr=edge_attrs, empty = False)
    else:
        data = Data(x = torch.FloatTensor(node_features), empty = True)
    #print(edge_index.shape)
    #print(edge_attrs.shape)
    
    return data


def build_graphs(smiles, extra_features_table=None):
    molecules = [Chem.MolFromSmiles(smile) for smile in tqdm(smiles)]
    
    extra_features_dict, extra_features = dict(), None
    if extra_features_table is not None:
        for row in extra_features_table.iterrows():
            key, features = row[0], row[1].values
            extra_features_dict[key] = np.asarray(features, dtype=np.float32)            
        
        extra_features = []
        for molecule in tqdm(molecules):
            symbols = list(map(lambda x: x.GetSymbol(), list(molecule.GetAtoms())))
            features = np.asarray([extra_features_dict[symbol] for symbol in symbols], dtype=np.float32)
            extra_features.append(features)
    
    train_raw_bond_features = []
    for mol in molecules:
        train_raw_bond_features = train_raw_bond_features + get_raw_bond_features(mol)
    bond_encoder = Encoder(train_raw_bond_features)
    train_bond_features = bond_encoder.transform(train_raw_bond_features)
    
    train_raw_node_features = []
    for mol in molecules:
        train_raw_node_features = train_raw_node_features + get_raw_node_features(mol)
    node_encoder = Encoder(train_raw_node_features)
    train_node_features = node_encoder.transform(train_raw_node_features)
    
    graphs = [mol_2_pytorch_geometric(molecule, node_encoder, bond_encoder, extra_features=extra_features_line) 
              for molecule, extra_features_line in zip(tqdm(molecules), extra_features)]
    return graphs