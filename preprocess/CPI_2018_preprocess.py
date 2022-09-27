"""
All code are adapted from the authors' original github
https://github.com/masashitsubaki/CPI_prediction
"""


import pickle
from collections import defaultdict

import numpy as np
from rdkit import Chem


def create_atoms(mol, atom_dict):
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_i_j_bond_dict(mol, bond_dict):
    """Create a dictionary, in which each key is a node ID
    and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs."""
    i_j_bond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_j_bond_dict[i].append((j, bond))
        i_j_bond_dict[j].append((i, bond))
    return i_j_bond_dict


def extract_fingerprints(atoms, i_j_bond_dict, radius, fingerprint_dict, edge_dict):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_j_edge_dict = i_j_bond_dict
        fingerprints = []

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            fingerprints = []
            for i, j_edge in i_j_edge_dict.items():
                # i -- node ID
                # j_edge -- [(connected_node1_ID, bond_type_ID), ...]

                # neighbors -- [(connected_atom_ID, bond_type_ID), ...]
                # we replacing node_IDs by atom_IDs
                # node_IDs == i == encode atoms in whole molecule, and each atom will have unique ID;
                # whereas atom_IDs == nodes[j] == encode atom types
                # e.g. in CCCCC each C will have unique node_ID but same atom_ID
                neighbors = [(nodes[j], edge) for j, edge in j_edge]

                # fingerprint == tuple(atom_ID, tuple( (connected_atom_ID, bond_type_ID), (..., ...) ))
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                # if two nodes have absolutely the same neighbors, they will have the same fingerprint
                # and the same fingerprint_ID from fingerprint_dict[fingerprint]
                fingerprints.append(fingerprint_dict[fingerprint])

            nodes = fingerprints

            """Also update each edge ID considering two nodes
            on its both sides."""
            _i_j_edge_dict = defaultdict(lambda: [])
            for i, j_edge in i_j_edge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_j_edge_dict[i].append((j, edge))
            i_j_edge_dict = _i_j_edge_dict

            # Then, if R > 1, we calculate next-level neighbors

    return np.array(fingerprints)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)


def split_sequence(sequence, ngram, word_dict):
    sequence = '-' + sequence + '='
    words = [word_dict[sequence[i:i+ngram]]
             for i in range(len(sequence)-ngram+1)]
    return np.array(words)


def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)


def cpi_preprocess(dataset, radius=2, ngram=3):
    """
    Takes dataset instance and preprocesses it for CPI_prediction2018 model.
    All new features are written back to dataset using the `` add_feature`` method.
    """

    # Dataset must have "SMILES", "Sequence" and "Label" features to perform preprocessing.
    for feat in ["SMILES", "Sequence", "Label"]:
        if feat not in dataset.return_options:
            print(f"Dataset must contain {feat} feature to properly encode features for CPI_prediction2018 model.")

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    word_dict = defaultdict(lambda: len(word_dict))

    compounds, adjacency_list, proteins = {}, {}, {}

    old_return_type = dataset.return_type
    dataset.return_type = ['SMILES', 'Sequence', 'Label']
    dataset.mode = 'all'

    # TODO: Is it OK that I use dataset.features dictionary
    # TODO: or is it better to implement dataset.all_drugs and dataset.all_proteins?
    # unique_drugs = dataset.all_drugs  # all IDs of unique drugs
    unique_drugs = list(set(dataset.features["SMILES"]))  # all unique drug SMILES
    # unique_proteins = dataset.all_proteins  # all IDs of unique proteins
    unique_proteins = list(set(dataset.features["Sequence"]))  # all unique protein sequences

    """Encode unique drugs and proteins"""
    for smiles in unique_drugs:
        # Iterate over all unique Drugs to encode them all

        # Consider hydrogen.
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        # convert SMILES to np.array of numbers, ex. [0 1 2 1 1 1 3 3]
        # each number corresponds to a different atom (encodings stated in atom_dict).
        # Also, aromatic atoms (e.g. C's in cyclo-hexane) saved in dict as ('C', 'aromatic')...
        # ...and because of that ('C') and ('C', 'aromatic') will be encoded by different numbers
        atoms = create_atoms(mol, atom_dict)

        # dictionary {node_ID: [(connected_node1_ID, bond_type_ID), (connected_node2_ID, bond_type_ID), ...]}
        # where bond_type_ID is a number that depends on a bond type (single, double)
        i_j_bond_dict = create_i_j_bond_dict(mol, bond_dict)

        # get fingerprint -- np.array() of some integers 
        fingerprints = extract_fingerprints(atoms, i_j_bond_dict, radius, fingerprint_dict, edge_dict)
        compounds[smiles] = fingerprints

        # The elements of the matrix indicate whether pairs of vertices are adjacent or not in the graph
        adjacency = create_adjacency(mol)
        adjacency_list[smiles] = adjacency

    for sequence in unique_proteins:
        # Iterate over all unique proteins to encode them all

        # Split each Target Sequence into parts of size *ngram* (default=3)
        # and then encode each ngram-plet (e.g. triplets like AAG, TGC, etc.) by ID
        words = split_sequence(sequence, ngram, word_dict)
        proteins[sequence] = words

    """Encode all drugs and proteins in dataset to obtain new feature lists"""

    all_compounds, all_adjacency, all_proteins = [], [], []

    for smiles, sequence, _ in dataset:
        compound = compounds[smiles]
        all_compounds.append(compound)

        adjacency = adjacency_list[smiles]
        all_adjacency.append(adjacency)

        words = proteins[sequence]
        all_proteins.append(words)

    dataset.return_type = old_return_type

    """Add features to dataset."""
    dataset.add_feature(feat_name="compounds", feat_values=all_compounds)
    dataset.add_feature(feat_name="adjacencies", feat_values=all_adjacency)
    dataset.add_feature(feat_name="proteins", feat_values=all_proteins, save=True)
    dataset.fingerprint_dict = fingerprint_dict
    dataset.word_dict = word_dict


    """Save processed data to .pickle files."""
    # dir_input = dataset.processed_folder
    # os.makedirs(dir_input, exist_ok=True)
    # dump_dictionary(compounds, os.path.join(dir_input, 'compounds.pickle'))
    # dump_dictionary(adjacency, os.path.join(dir_input, 'adjacency.pickle'))
    # dump_dictionary(proteins, os.path.join(dir_input, 'proteins.pickle'))
    # dump_dictionary(fingerprint_dict, os.path.join(dir_input, 'fingerprint_dict.pickle'))
    # dump_dictionary(word_dict, os.path.join(dir_input, 'word_dict.pickle'))

    """Return result dataset"""
    return dataset
