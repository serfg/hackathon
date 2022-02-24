import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import Data

from openbabel import pybel
pybel.ob.obErrorLog.SetOutputLevel(0)
from ase import Atoms


class MoleculePybel(object):
    def __init__(self, smiles, fixed=False):
        self.molecule = pybel.readstring('smi', smiles)
        self.fixed = fixed
        self.remake3d()
        
        
    def remake3d(self):
        self._process_molecule()
        self._transform_diagonal()
        self.augment()
        
       
    def get_ase(self):
        mol = self.molecule.OBMol
        positions, atomic_nums = [], []
        num_atoms = mol.NumAtoms()
        for index in range(num_atoms):
            atom = mol.GetAtomById(index)
            positions.append([atom.GetX(), atom.GetY(), atom.GetZ()])
            atomic_nums.append(atom.GetAtomicNum())
        return Atoms(positions = positions, numbers = atomic_nums)
        
    def _process_molecule(self):
        self.molecule.addh()
        self.molecule.make3D('gaff', steps=128)
        mol = self.molecule.OBMol
        
        num_atoms, num_bonds = mol.NumAtoms(), mol.NumBonds()
        atoms = np.zeros(num_atoms, dtype=object)
        masses = np.zeros(num_atoms, dtype=float)
        matrix = np.zeros((num_atoms, 3), dtype=float)
        bonds = np.zeros((num_bonds, 2), dtype=int)
        orders = np.zeros(num_bonds, dtype=int)
        
        node_features, edge_features = [], []

        for index in range(num_atoms):
            atom = mol.GetAtomById(index)
            atoms[index] = atom.GetType()
            masses[index] = atom.GetAtomicMass()
            matrix[index] = [atom.GetX(), atom.GetY(), atom.GetZ()]
            
            features = []
            features.append((atom.GetAtomicNum(), [1, 6, 8, 7, 9, 16, 17]))
            features.append((atom.GetTotalDegree(), [1, 3, 4, 2]))
            features.append((atom.GetTotalValence(), [1, 4, 2, 3, 6]))
            features.append((atom.GetFormalCharge(), [0, 1, -1]))
            features.append((atom.GetHeteroDegree(), [0, 1, 2, 3]))
            features.append((atom.GetHvyDegree(), [1, 2, 3, 4, 0]))
            features.append((atom.GetHyb(), [1, 2, 3]))
            
            features.append(atom.GetPartialCharge())            
            features.append(atom.HasAlphaBetaUnsat())
            
            features.append(atom.IsAromatic())
            features.append(atom.IsHbondAcceptor())
            features.append(atom.IsHbondDonor())
            features.append(atom.IsInRing())
            features.append(atom.IsHeteroatom())
            features.append(atom.IsAromatic())
            features.append(1.0)
            
            node_features_atom = []
            for feature in features:
                if not isinstance(feature, tuple) == 1:
                    node_features_atom.append(float(feature))
                    continue
                value, keys = feature[0], feature[1]                
                tmp = [float(value == key) for key in keys]
                node_features_atom.extend(tmp)                
            node_features.append(node_features_atom)
            
        node_features = np.asarray(node_features, dtype=np.float32)
        
        for index in range(num_bonds):
            bond = mol.GetBondById(index)
            bonds[index] = [bond.GetBeginAtom().GetId(), bond.GetEndAtom().GetId()]
            
            features = []
            features.append((bond.GetBondOrder(), [1, 2]))
            features.append(bond.IsAromatic())
            features.append(bond.IsInRing())
            features.append(bond.IsRotor())
            features.append(1.0)
            
            edge_features_bond = []
            for feature in features:
                if not isinstance(feature, tuple) == 1:
                    edge_features_bond.append(float(feature))
                    continue
                value, keys = feature[0], feature[1]                
                tmp = [float(value == key) for key in keys]
                edge_features_bond.extend(tmp)                
            edge_features.append(edge_features_bond)
            
        edge_features = np.asarray(edge_features, dtype=np.float32)

        self.atoms, self.masses, self.matrix, self.bonds, self.node_features, self.edge_features =\
            atoms, masses, matrix, bonds, node_features, edge_features
    
    
    @staticmethod
    def _calc_center_mass(masses, matrix):
        return np.sum(masses[:, None] * matrix, axis=0) / np.sum(masses)
    
    
    @staticmethod
    def _calc_intertia_moment(masses, matrix):
        tensor = np.zeros((3, 3), dtype=masses.dtype)
        for index in range(3):
            indices = [item for item in range(3) if item != index]
            tensor[index, index] = np.sum(masses[:, None] * matrix[:, indices] * matrix[:, indices])
            for jndex in range(index + 1, 3):
                tensor[index, jndex] = tensor[jndex, index] = -np.sum(masses * matrix[:, index] * matrix[:, jndex])
        return tensor
    
    
    def _transform_diagonal(self):
        mean = self._calc_center_mass(self.masses, self.matrix)
        self.matrix -= mean[None]
        
        moment = self._calc_intertia_moment(self.masses, self.matrix)
        values, vectors = np.linalg.eigh(moment)
        self.matrix = np.dot(vectors.T, self.matrix.T).T
        
        self.inertias = self._calc_intertia_moment(self.masses, self.matrix).diagonal()
        self.jnertias = np.sum(self.inertias) - 2.0 * self.inertias
        assert self.jnertias[0] >= self.jnertias[1] and self.jnertias[1] >= self.jnertias[2], str(self.jnertias)

        
    def augment(self):
        self.matrix_plus = self.matrix.copy()
        if self.fixed:
            return
        
        for index in range(3):
            if np.random.randint(2):
                self.matrix_plus[:, index] *= -1.0
                
        alpha_bound = np.pi * self.jnertias[1] / (2.0 * self.jnertias[0])
        theta_bound = np.pi * self.jnertias[2] / (2.0 * self.jnertias[1])
        
        if not np.isfinite(alpha_bound):
            alpha_bound = np.pi / 2.0
        if not np.isfinite(theta_bound):
            theta_bound = np.pi / 2.0
        
        alpha = np.random.uniform(low=-alpha_bound, high=alpha_bound)
        theta = np.random.uniform(low=-theta_bound, high=theta_bound)
        
        cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        
        m = np.asarray([[cos_theta + (1.0 - cos_theta) * cos_alpha * cos_alpha,
                         (1.0 - cos_theta) * cos_alpha * sin_alpha,
                         sin_theta * sin_alpha],
                        [(1.0 - cos_theta) * cos_alpha * sin_alpha,
                         cos_theta + (1.0 - cos_theta) * sin_alpha * sin_alpha,
                         -sin_theta * cos_alpha],
                        [-sin_theta * sin_alpha, sin_theta * cos_alpha, cos_theta]])
        self.matrix_plus = np.dot(m, self.matrix_plus.T).T
        
        
    def get_graph(self):
        return Data(x=torch.FloatTensor(self.node_features),
                    edge_index=torch.LongTensor(self.bonds.T),
                    edge_attr=torch.FloatTensor(self.edge_features), empty=len(self.bonds) == 0)
    
    
    def build_tensor(self, **kwargs):
        return self._build_tensor(self, **kwargs)
    
    
    @staticmethod
    def _build_tensor(molecule, diameter=16.0, size=128,
                      atoms_radius=0.5, bonds_radius=0.25,
                      rotate=True, shift=True,
                      use_torch=False, clip=False, device=torch.device('cuda:0')):
        def _atom_density(tensor, radius, use_torch=False):
            radius2 = radius * radius
            values = (1.0 - tensor / radius2)
            signs = tensor < radius2
            result = values * (signs.float() if use_torch else signs)
            return result ** 0.5


        def _bond_density(tensor, radius, use_torch=False):
            radius2 = radius * radius
            values = (1.0 - tensor / radius2)
            signs = tensor < radius2
            result = values * (signs.float() if use_torch else signs)
            return result ** 0.5

        node_features, edge_features = molecule.node_features, molecule.edge_features

        step = diameter / size
        x_grid = np.arange(-(size - 1) // 2, (size - 1) // 2 + 1) * step
        y_grid = np.arange(-(size - 1) // 2, (size - 1) // 2 + 1) * step

        coords = np.asarray(np.meshgrid(x_grid, y_grid, indexing='ij'))
        tensor = np.zeros((node_features.shape[1] + edge_features.shape[1], size, size), dtype=np.float64)

        atoms, matrix, bonds = molecule.atoms, molecule.matrix_plus[:, :2], molecule.bonds
        if rotate:
            phi = np.random.uniform(low=-np.pi, high=np.pi)
            cos_phi, sin_phi = np.cos(phi), np.sin(phi)
            m = np.asarray([[cos_phi, -sin_phi], [sin_phi, cos_phi]])
            matrix = np.dot(m, matrix.T).T
        if shift:
            delta = np.random.uniform(-step * size // 8, step * size // 8, size=2)
            matrix += delta[None]

        if use_torch:
            coords, tensor, matrix = torch.FloatTensor(coords), torch.FloatTensor(tensor), torch.FloatTensor(matrix)
            coords, tensor, matrix = coords.to(device), tensor.to(device), matrix.to(device)
            node_features, edge_features = torch.FloatTensor(node_features), torch.FloatTensor(edge_features)
            node_features, edge_features = node_features.to(device), edge_features.to(device)
        else:
            coords, tensor, matrix = coords.astype(np.float32), tensor.astype(np.float32), matrix.astype(np.float32)

        for atom, feature, vector in zip(atoms, node_features, matrix):
            deltas_tensor = coords - vector[:, None, None]
            squared_distances_tensor = (deltas_tensor * deltas_tensor).sum(**{'dim' if use_torch else 'axis':0})
            density_tensor = _atom_density(squared_distances_tensor, atoms_radius, use_torch=use_torch)
            tensor[:len(feature)] += density_tensor[None] * feature[:, None, None]

        for bond, feature in zip(bonds, edge_features):
            dir_vector = matrix[bond[1]] - matrix[bond[0]]
            a, b = dir_vector[1], -dir_vector[0]
            c = -(a * matrix[bond[0]][0] + b * matrix[bond[0]][1])
            distances_tensor = (a * coords[0] + b * coords[1] + c)
            squared_distances_tensor = distances_tensor * distances_tensor / (a * a + b * b)

            deltas_tensor = coords - matrix[bond[0]][:, None, None]
            mask_tensor = (deltas_tensor * dir_vector[:, None, None]).sum(**{'dim' if use_torch else 'axis':0})
            density_mask_tensor = (mask_tensor >= 0.0) * (mask_tensor <= (dir_vector * dir_vector).sum())
            if use_torch:
                density_mask_tensor = density_mask_tensor.float()

            density_tensor = _bond_density(squared_distances_tensor, bonds_radius, use_torch=use_torch)
            tensor[-len(feature):] += density_tensor[None] * density_mask_tensor[None] * feature[:, None, None]

        if clip:
            tensor = eval('torch' if use_torch else 'np').clip(tensor, 0.0, 1.0)
        return tensor.cpu().numpy() if use_torch else tensor


def build_molecules_pybel(smiles, fixed=False):
    return [MoleculePybel(item, fixed=fixed) for item in tqdm(smiles)]