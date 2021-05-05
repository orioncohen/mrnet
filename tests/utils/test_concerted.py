# coding: utf-8
import io
import os

import pickle
import copy
import numpy as np
import pandas as pd
import scipy.sparse

from monty.serialization import dumpfn, loadfn
from networkx.readwrite import json_graph

from pymatgen.util.testing import PymatgenTest
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender

from mrnet.core.mol_entry import MoleculeEntry
from mrnet.core.reactions import RedoxReaction
from mrnet.network.reaction_network import ReactionPath, ReactionNetwork
from mrnet.network.reaction_generation import ReactionGenerator

from mrnet.utils.concerted import (
    construct_reaction_dataframe,
    get_reaction_indices,
    square_matrix,
    construct_matrix
    validate_concerted_rxns
)

try:
    import openbabel as ob
except ImportError:
    ob = None

test_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "test_files",
    "reaction_network_files",
)


class TestConcertedUtilities(PymatgenTest):
    @classmethod
    # this code is largely copied from the ReactionNetwork setup, how can I reuse it?
    def setUpClass(cls):
        # if ob:
        #     with open(os.path.join(test_dir, "unittest_RN_build.pkl"), "rb") as input:
        #         cls.RN_build = pickle.load(input)
        with open(
                os.path.join(
                    test_dir, "identify_concerted_via_intermediate_unittest_RN.pkl"
                ),
                "rb",
        ) as input:
            cls.RN_build = pickle.load(input)

    def test_setup(self):
        return
        # RN = copy.deepcopy(self.RN_build)
        #
        # # run calc
        # PR_record = RN.build_PR_record()
        # reactant_record = RN.build_reactant_record()
        #
        # # assert
        # self.assertEqual(len(PR_record[0]), 42)
        # self.assertEqual(PR_record[44], ["165+PR_44,434"])
        # self.assertEqual(len(PR_record[529]), 0)
        # self.assertEqual(len(PR_record[556]), 104)
        # self.assertEqual(len(PR_record[564]), 165)
        #
        # self.assertEqual(len(reactant_record[0]), 43)
        # self.assertCountEqual(
        #     reactant_record[44], ["44+PR_165,434", "44,43", "44,40+556"]
        # )
        # self.assertEqual(len(reactant_record[529]), 0)
        # self.assertEqual(len(reactant_record[556]), 104)
        # self.assertEqual(len(reactant_record[564]), 167)

    def test_get_reaction_indices(self):
        RN = self.RN_build
        rxn_dataframe = construct_reaction_dataframe(RN)
        data, coords, node_index = get_reaction_indices(RN, rxn_dataframe)

        self.assertEqual(len(RN.reactions) * 2 + len(RN.entries_list), len(node_index))
        self.assertEqual(len(RN.reactions) * 2, len(rxn_dataframe))

        coord_list = list(coords)
        self.assertIn((node_index[116814], node_index[(0, 'A')]), coord_list)
        self.assertIn((node_index[(459, 'B')], node_index[115885]), coord_list)
        self.assertIn((node_index[120769], node_index[(1053, 'A')]), coord_list)

    def test_construct_reaction_dataframe(self):
        RN = self.RN_build
        df = construct_reaction_dataframe(RN)
        new_index = pd.MultiIndex.from_tuples(df['node_id'])
        new_df = df.set_index(new_index)
        self.assertEqual(len(RN.reactions) * 2, len(df))

        rxn_0A = new_df.loc[(0, 'A')]
        self.assertAlmostEqual(rxn_0A['delta_g'], 4.74991782078232)
        self.assertEqual(rxn_0A['node_id'], (0, 'A'))
        self.assertEqual(rxn_0A['rxn_type'], 'One electron oxidation')
        self.assertEqual(rxn_0A['reactants'][0], 116814)
        self.assertEqual(rxn_0A['products'][0], 116813)

        rxn_459B = new_df.loc[(459, 'B')]
        self.assertAlmostEqual(rxn_459B['delta_g'], -5.505757600154084)
        self.assertEqual(rxn_459B['node_id'], (459, 'B'))
        self.assertEqual(rxn_459B['rxn_type'], 'Molecular formation from one new bond A+B -> C')
        self.assertEqual(rxn_459B['reactants'][0], 116817)
        self.assertEqual(rxn_459B['products'][0], 115885)

        rxn_1053A = new_df.loc[(1053, 'A')]
        self.assertAlmostEqual(rxn_1053A['delta_g'], 2.0872563796547183)
        self.assertEqual(rxn_1053A['node_id'], (1053, 'A'))
        self.assertEqual(rxn_1053A['rxn_type'], 'Molecular decomposition breaking one bond A -> B+C')
        self.assertEqual(rxn_1053A['reactants'][0], 120769)
        self.assertEqual(rxn_1053A['products'][1], 116813)

    def test_construct_matrix(self):
        RN = self.RN_build
        rxn_dataframe = construct_reaction_dataframe(RN)
        data, coords, node_index = get_reaction_indices(RN, rxn_dataframe)
        A = construct_matrix(data, coords, len(node_index))
        self.assertEqual(A.shape, (len(node_index), len(node_index)))
        A = A.todense()
        self.assertEqual(A[node_index[116814], node_index[(0, 'A')]], 1)
        self.assertEqual(A[node_index[(459, 'B')], node_index[115885]], 1)
        self.assertEqual(A[node_index[120769], node_index[(1053, 'A')]], 1)

    def test_square_matrix(self):
        # Test with two small 3x3 examples first
        A = scipy.sparse.coo_matrix(np.array(
            [[1,1,0], [0,1,0], [1,1,1]]))
        result = square_matrix(A)
        expected = np.array(
            [[1,2,0], [0,1,0], [2,3,1]])
        np.testing.assert_array_equal(result.todense(), expected)

        A = scipy.sparse.coo_matrix(np.array(
            [[1,1,0], [1,0,1], [0,1,1]]))
        result = square_matrix(A)
        expected = np.array(
            [[2,1,1], [1,2,1], [1,1,2]])
        np.testing.assert_array_equal(result.todense(), expected)

        # Test with random matrices up to size 100x100
        for s in range(1, 100):
            A = np.random.randint(2, size=s*s).reshape(s, s)
            expected = np.matmul(A, A)
            result = square_matrix(scipy.sparse.coo_matrix(A))
            np.testing.assert_array_equal(result.todense(), expected)

        # Test with the matrix from the reaction network example
        RN = self.RN_build
        rxn_dataframe = construct_reaction_dataframe(RN)
        data, coords, node_index = get_reaction_indices(RN, rxn_dataframe)
        A = construct_matrix(data, coords, len(node_index))
        result = square_matrix(A)
        A_csr = A.tocsr()
        expected = (A_csr * A_csr).todense()
        np.testing.assert_array_equal(result.todense(), expected)
        

    def test_get_rxn_subspace(self):
        RN = self.RN_build

    def test_validate_concerted_rxns(self):
        RN = self.RN_build
        dense_adjacency = np.array([[0, 0, 0, 1, 1, 1],
                                   [0, 0, 0, 1, 1, 1],
                                   [0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0]])
        node_ids = [(0, 'A'), (0, 'B'), (1, 'A'), (2, 'A'), (3, 'A'), (4, 'A')]
        delta_g = [0.5, -0.5, -1, -1, -1, -0.2]
        rxn_types = ['', '', '', '', '', '']
        reactants = [[10, 11], [14, 15], [12], [13, 14], [14], [15]]
        products = [[14, 15], [10, 11], [14], [16], [17], [18]]
        rxn_dataframe = pd.DataFrame({
            'node_ids': node_ids,
            'delta_g': delta_g,
            'rxn_types': rxn_types,
            'reactants': reactants,
            'products': products
        })
        valid_adjacency = validate_concerted_rxns(dense_adjacency, rxn_dataframe)
        correct_matrix = np.array([[0, 0, 0, 0, 1, 1],
                                   [0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0]])
        self.assertArrayEqual(valid_adjacency, correct_matrix)

    def test_rxn_matrix_to_list(self):
        RN = self.RN_build
