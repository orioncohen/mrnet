# coding: utf-8
import io
import os

import pickle
import copy

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
    # this code is
    def setUpClass(cls):
        if ob:
            with open(os.path.join(test_dir, "unittest_RN_build.pkl"), "rb") as input:
                cls.RN_build = pickle.load(input)

    def test_setup(self):
        RN = copy.deepcopy(self.RN_build)

        # run calc
        PR_record = RN.build_PR_record()
        reactant_record = RN.build_reactant_record()

        # assert
        self.assertEqual(len(PR_record[0]), 42)
        self.assertEqual(PR_record[44], ["165+PR_44,434"])
        self.assertEqual(len(PR_record[529]), 0)
        self.assertEqual(len(PR_record[556]), 104)
        self.assertEqual(len(PR_record[564]), 165)

        self.assertEqual(len(reactant_record[0]), 43)
        self.assertCountEqual(
            reactant_record[44], ["44+PR_165,434", "44,43", "44,40+556"]
        )
        self.assertEqual(len(reactant_record[529]), 0)
        self.assertEqual(len(reactant_record[556]), 104)
        self.assertEqual(len(reactant_record[564]), 167)

    def test_get_reaction_indices(self):
        RN = self.RN_build

    def test_construct_reaction_dataframe(self):
        RN = self.RN_build

    def test_construct_reaction_dataframe(self):
        RN = self.RN_build

    def test_square_matrix(self):
        RN = self.RN_build

    def test_get_rxn_subspace(self):
        RN = self.RN_build

    def test_validate_concerted_rxns(self):
        RN = self.RN_build

    def test_rxn_matrix_to_list(self):
        RN = self.RN_build
