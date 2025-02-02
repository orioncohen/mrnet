# coding: utf-8
import io
import os
import sys
import unittest
import copy
import pickle
from itertools import permutations
from ast import literal_eval

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


class TestReactionGenerator(PymatgenTest):
    def test_reaction_generator(self):

        molecule_entries = loadfn(os.path.join(test_dir, "ronalds_MoleculeEntry.json"))
        reaction_generator = ReactionGenerator(molecule_entries)

        reaction_set = set()
        counter = 0

        for reaction in reaction_generator:
            reaction_set.add(frozenset(reaction[0] + reaction[1]))
            counter += 1

        self.assertEqual(counter, 119)
        self.assertEqual(len(reaction_set), 106)


class TestReactionPath(PymatgenTest):
    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_characterize_path(self):
        # set up input variables
        with open(
            os.path.join(test_dir, "unittest_RN_before_characterize_path.pkl"), "rb"
        ) as input:
            RN = pickle.load(input)

        solved_PRs = loadfn(
            os.path.join(test_dir, "unittest_characterize_path_old_solved_PRs_IN.json")
        )
        path = loadfn(os.path.join(test_dir, "unittest_characterize_path_path_IN.json"))

        # run calc
        path_instance = ReactionPath.characterize_path(
            path, "softplus", RN.graph, solved_PRs
        )

        # assert
        self.assertEqual(path_instance.byproducts, [356, 548])
        self.assertEqual(path_instance.unsolved_prereqs, [])
        self.assertEqual(path_instance.solved_prereqs, [556, 46])
        self.assertEqual(path_instance.cost, 12.592087913497771)
        self.assertEqual(path_instance.pure_cost, 0.0)
        self.assertEqual(path_instance.hardest_step_deltaG, None)
        self.assertEqual(
            path_instance.path,
            [
                456,
                "456+556,424",
                424,
                "424,423",
                423,
                "423,420",
                420,
                "420,356+543",
                543,
                "46+543,15",
                15,
                "15,13",
                13,
                "13,1+548",
                1,
                "1,2",
                2,
            ],
        )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_characterize_path_final(self):

        # set up input variables
        with open(os.path.join(test_dir, "unittest_RN_pr_solved.pkl"), "rb") as input:
            RN_pr_solved = pickle.load(input)

        # perform calc
        # print(RN_pr_solved.PRs[2], RN_pr_solved.PRs[2])
        path_class = ReactionPath.characterize_path_final(
            RN_pr_solved.PRs[2][456].path,
            RN_pr_solved.weight,
            RN_pr_solved.graph,
            RN_pr_solved.solved_PRs,
            RN_pr_solved.PRs,
            RN_pr_solved.PR_byproducts,
        )

        # assert
        self.assertEqual(path_class.byproducts, [356, 548, 182])
        self.assertEqual(path_class.solved_prereqs, [556, 46])
        self.assertEqual(path_class.all_prereqs, [556, 46])
        self.assertEqual(path_class.cost, 12.592087913497771)
        self.assertEqual(
            path_class.path,
            [
                456,
                "456+556,424",
                424,
                "424,423",
                423,
                "423,420",
                420,
                "420,356+543",
                543,
                "46+543,15",
                15,
                "15,13",
                13,
                "13,1+548",
                1,
                "1,2",
                2,
            ],
        )
        self.assertEqual(path_class.overall_free_energy_change, 10.868929712195717)
        self.assertEqual(path_class.pure_cost, 12.592087913497771)
        self.assertEqual(path_class.hardest_step_deltaG, 4.018404627691188)


class TestReactionNetwork(PymatgenTest):
    @classmethod
    def setUpClass(cls):
        if ob:
            EC_mg = MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir, "EC.xyz")), OpenBabelNN()
            )
            cls.EC_mg = metal_edge_extender(EC_mg)

            LiEC_mg = MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir, "LiEC.xyz")), OpenBabelNN()
            )
            cls.LiEC_mg = metal_edge_extender(LiEC_mg)

            LEDC_mg = MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir, "LEDC.xyz")), OpenBabelNN()
            )
            cls.LEDC_mg = metal_edge_extender(LEDC_mg)

            LEMC_mg = MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir, "LEMC.xyz")), OpenBabelNN()
            )
            cls.LEMC_mg = metal_edge_extender(LEMC_mg)

            cls.LiEC_reextended_entries = []
            entries = loadfn(os.path.join(test_dir, "LiEC_reextended_entries.json"))
            for entry in entries:
                if "optimized_molecule" in entry["output"]:
                    mol = entry["output"]["optimized_molecule"]
                else:
                    mol = entry["output"]["initial_molecule"]
                E = float(entry["output"]["final_energy"])
                H = float(entry["output"]["enthalpy"])
                S = float(entry["output"]["entropy"])
                mol_entry = MoleculeEntry(
                    molecule=mol,
                    energy=E,
                    enthalpy=H,
                    entropy=S,
                    entry_id=entry["task_id"],
                )
                if mol_entry.formula == "Li1":
                    if mol_entry.charge == 1:
                        cls.LiEC_reextended_entries.append(mol_entry)
                else:
                    cls.LiEC_reextended_entries.append(mol_entry)

            # dumpfn(cls.LiEC_reextended_entries, "unittest_input_molentries.json")

            with open(os.path.join(test_dir, "unittest_RN_build.pkl"), "rb") as input:
                cls.RN_build = pickle.load(input)

            with open(
                os.path.join(test_dir, "unittest_RN_pr_solved.pkl"), "rb"
            ) as input:
                cls.RN_pr_solved = pickle.load(input)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_add_reactions(self):

        # set up RN
        RN = ReactionNetwork.from_input_entries(
            self.LiEC_reextended_entries, electron_free_energy=-2.15
        )

        # set up input variables
        EC_0_entry = None
        EC_minus_entry = None

        # print(RN.entries["C3 H4 O3"].keys())

        for entry in RN.entries["C3 H4 O3"][10][0]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_0_entry = entry
                break
        for entry in RN.entries["C3 H4 O3"][10][-1]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_minus_entry = entry
                break

        redox = RedoxReaction(EC_0_entry, EC_minus_entry)
        redox.electron_free_energy = -2.15
        redox_graph = redox.graph_representation()

        # run calc
        RN.add_reaction(redox_graph)

        # assert (use set to be ordering-agnostic)
        self.assertEqual(set(RN.graph.nodes), set(["456,455", 456, 455, "455,456"]))
        self.assertEqual(
            set(RN.graph.edges),
            set(
                [("456,455", 455), (456, "456,455"), (455, "455,456"), ("455,456", 456)]
            ),
        )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_build(self):
        # set up RN
        RN = ReactionNetwork.from_input_entries(
            self.LiEC_reextended_entries, electron_free_energy=-2.15
        )

        # perfrom calc
        RN.build()

        # assert
        EC_ind = None
        LEDC_ind = None
        LiEC_ind = None
        for entry in RN.entries["C3 H4 Li1 O3"][12][1]:
            if self.LiEC_mg.isomorphic_to(entry.mol_graph):
                LiEC_ind = entry.parameters["ind"]
                break
        for entry in RN.entries["C3 H4 O3"][10][0]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_ind = entry.parameters["ind"]
                break
        for entry in RN.entries["C4 H4 Li2 O6"][17][0]:
            if self.LEDC_mg.isomorphic_to(entry.mol_graph):
                LEDC_ind = entry.parameters["ind"]
                break
        Li1_ind = RN.entries["Li1"][0][1][0].parameters["ind"]
        self.assertEqual(len(RN.entries_list), 569)
        self.assertEqual(EC_ind, 456)
        self.assertEqual(LEDC_ind, 511)
        self.assertEqual(Li1_ind, 556)
        self.assertEqual(LiEC_ind, 424)

        self.assertEqual(len(RN.graph.edges), 19824)
        self.assertEqual(len(RN.graph.nodes), 7415)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_build_PR_record(self):
        # set up RN
        RN = copy.deepcopy(self.RN_build)
        # run calc
        PR_record = RN.build_PR_record()
        # assert
        self.assertEqual(len(PR_record[0]), 42)
        self.assertEqual(PR_record[44], [(165, "44+165,434")])
        self.assertEqual(len(PR_record[529]), 0)
        self.assertEqual(len(PR_record[556]), 104)
        self.assertEqual(len(PR_record[564]), 165)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_build_reactant_record(self):

        # set up RN
        RN = copy.deepcopy(self.RN_build)

        # run calc
        reactant_record = RN.build_reactant_record()

        # assert
        self.assertEqual(len(reactant_record[0]), 43)
        self.assertCountEqual(
            reactant_record[44], [(44, "44+165,434"), (44, "44,43"), (44, "44,40+556")]
        )
        self.assertEqual(len(reactant_record[529]), 0)
        self.assertEqual(len(reactant_record[556]), 104)
        self.assertEqual(len(reactant_record[564]), 167)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_solve_prerequisites(self):
        # set up RN
        RN = copy.deepcopy(self.RN_build)
        RN.build_PR_record()
        # set up input variables

        EC_ind = None

        for entry in RN.entries["C3 H4 O3"][10][0]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_ind = entry.parameters["ind"]
            break
        Li1_ind = RN.entries["Li1"][0][1][0].parameters["ind"]

        # perfrom calc
        PRs_calc, old_solved_PRs = RN.solve_prerequisites(
            [EC_ind, Li1_ind], weight="softplus"
        )

        # assert
        with open(
            os.path.join(test_dir, "unittest_RN_pr_solved_PRs.pkl"), "rb"
        ) as input:
            PR_paths = pickle.load(input)
        print(PRs_calc[12])
        print(PR_paths[12])

        for node in PR_paths:
            for start in PR_paths[node]:
                try:
                    self.assertEqual(
                        PRs_calc[node][start].all_prereqs,
                        PR_paths[node][start].all_prereqs,
                    )
                    self.assertEqual(
                        PRs_calc[node][start].byproducts,
                        PR_paths[node][start].byproducts,
                    )
                    self.assertEqual(
                        PRs_calc[node][start].solved_prereqs,
                        PR_paths[node][start].solved_prereqs,
                    )
                    self.assertEqual(
                        PRs_calc[node][start].unsolved_prereqs,
                        PR_paths[node][start].unsolved_prereqs,
                    )
                    # below code (similar to many other functions, is responsible for tweaking old strings to match current representation)
                    for i in range(
                        len(PR_paths[node][start].full_path)
                        if PR_paths[node][start].full_path != None
                        else 0
                    ):  # iterate over all nodes in the path
                        path = PR_paths[node][start].full_path[i]
                        if isinstance(
                            path, str
                        ):  # for string nodes, remove PR from the node name
                            trimmed_pr = path.replace("PR_", "")
                        else:
                            trimmed_pr = path
                        # order of reactants no longer enforced by PR, so just make sure all reactants are present in the reaction rather than worrying about exact order
                        try:
                            self.assertEqual(
                                trimmed_pr, PRs_calc[node][start].full_path[i]
                            )
                        except AssertionError:
                            rct_path = trimmed_pr.split(",")[0].split("+")
                            rct_calc = (
                                PRs_calc[node][start]
                                .full_path[i]
                                .split(",")[0]
                                .split("+")
                            )
                            self.assertCountEqual(rct_path, rct_calc)
                    for i in range(
                        len(PR_paths[node][start].path)
                        if PR_paths[node][start].path != None
                        else 0
                    ):
                        path = PR_paths[node][start].path[i]
                        if isinstance(path, str):
                            trimmed_pr = path.replace("PR_", "")
                        else:
                            trimmed_pr = path
                        try:
                            self.assertEqual(trimmed_pr, PRs_calc[node][start].path[i])
                        except AssertionError:
                            rct_path = trimmed_pr.split(",")[0].split("+")
                            rct_calc = (
                                PRs_calc[node][start].path[i].split(",")[0].split("+")
                            )
                            self.assertCountEqual(rct_path, rct_calc)

                    if PRs_calc[node][start].cost != PR_paths[node][start].cost:
                        self.assertAlmostEqual(
                            PRs_calc[node][start].cost,
                            PR_paths[node][start].cost,
                            places=2,
                        )
                    if (
                        PRs_calc[node][start].pure_cost
                        != PR_paths[node][start].pure_cost
                    ):
                        self.assertAlmostEqual(
                            PRs_calc[node][start].pure_cost,
                            PR_paths[node][start].pure_cost,
                            places=2,
                        )
                except KeyError:
                    self.assertTrue(False)
                    print("Node: ", node)
                    print("Start: ", start)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_find_path_cost(self):
        # set up RN

        with open(
            os.path.join(test_dir, "unittest_RN_pr_ii_4_before_find_path_cost.pkl"),
            "rb",
        ) as input:
            RN_pr_ii_4 = pickle.load(input)

        # set up input variables
        EC_ind = None
        for entry in RN_pr_ii_4.entries["C3 H4 O3"][10][0]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_ind = entry.parameters["ind"]
                break
        for entry in RN_pr_ii_4.entries["C4 H4 Li2 O6"][17][0]:
            if self.LEDC_mg.isomorphic_to(entry.mol_graph):
                LEDC_ind = entry.parameters["ind"]
                break
        Li1_ind = RN_pr_ii_4.entries["Li1"][0][1][0].parameters["ind"]

        loaded_cost_from_start_str = loadfn(
            os.path.join(test_dir, "unittest_find_path_cost_cost_from_start_IN.json")
        )

        old_solved_PRs = loadfn(
            os.path.join(test_dir, "unittest_find_path_cost_old_solved_prs_IN.json")
        )

        loaded_min_cost_str = loadfn(
            os.path.join(test_dir, "unittest_find_path_cost_min_cost_IN.json")
        )

        with open(
            os.path.join(test_dir, "unittest_find_path_cost_PRs_IN.pkl"), "rb"
        ) as input:
            loaded_PRs = pickle.load(input)

        loaded_cost_from_start = {}
        for node in loaded_cost_from_start_str:
            loaded_cost_from_start[int(node)] = {}
            for start in loaded_cost_from_start_str[node]:
                loaded_cost_from_start[int(node)][
                    int(start)
                ] = loaded_cost_from_start_str[node][start]

        loaded_min_cost = {}
        for node in loaded_min_cost_str:
            loaded_min_cost[int(node)] = loaded_min_cost_str[node]

        # perform calc
        PRs_cal, cost_from_start_cal, min_cost_cal = RN_pr_ii_4.find_path_cost(
            [EC_ind, Li1_ind],
            RN_pr_ii_4.weight,
            old_solved_PRs,
            loaded_cost_from_start,
            loaded_min_cost,
            loaded_PRs,
        )

        # assert
        self.assertEqual(cost_from_start_cal[456][456], 0.0)
        self.assertEqual(cost_from_start_cal[556][456], "no_path")
        self.assertEqual(cost_from_start_cal[0][456], 7.291618376702763)
        self.assertEqual(cost_from_start_cal[6][556], 3.76319246637113)
        self.assertEqual(cost_from_start_cal[80][456], 9.704537989094758)

        self.assertEqual(min_cost_cal[556], 0.0)
        self.assertEqual(min_cost_cal[1], 8.01719987083261)
        self.assertEqual(min_cost_cal[4], 5.8903562531872256)
        self.assertEqual(min_cost_cal[148], 2.5722550049918964)

        self.assertEqual(PRs_cal[556][556].path, [556])
        self.assertEqual(PRs_cal[556][456].path, None)
        self.assertEqual(PRs_cal[29][456].path, None)
        self.assertEqual(PRs_cal[313], {})

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_identify_solved_PRs(self):

        # set up RN
        with open(
            os.path.join(
                test_dir, "unittest_RN_pr_ii_4_before_identify_solved_PRs.pkl"
            ),
            "rb",
        ) as input:
            RN_pr_ii_4 = pickle.load(input)

        # set up input variables
        cost_from_start_IN_str = loadfn(
            os.path.join(
                test_dir, "unittest_identify_solved_PRs_cost_from_start_IN.json"
            )
        )
        solved_PRs = loadfn(
            os.path.join(test_dir, "unittest_identify_solved_PRs_solved_PRs_IN.json")
        )
        with open(
            os.path.join(test_dir, "unittest_find_path_cost_PRs_IN.pkl"), "rb"
        ) as input:
            PRs = pickle.load(input)

        cost_from_start = {}
        for node in cost_from_start_IN_str:
            cost_from_start[int(node)] = {}
            for start in cost_from_start_IN_str[node]:
                cost_from_start[int(node)][int(start)] = cost_from_start_IN_str[node][
                    start
                ]

        # perform calc
        (
            solved_PRs_cal,
            new_solved_PRs_cal,
            cost_from_start_cal,
        ) = RN_pr_ii_4.identify_solved_PRs(PRs, solved_PRs, cost_from_start)

        # assert
        self.assertEqual(len(solved_PRs_cal), 166)
        self.assertEqual(len(cost_from_start_cal), 569)
        self.assertEqual(cost_from_start_cal[456][556], "no_path")
        self.assertEqual(cost_from_start_cal[556][556], 0.0)
        self.assertEqual(cost_from_start_cal[2][556], 9.902847048351545)
        self.assertEqual(cost_from_start_cal[7][456], 3.5812151003313524)
        self.assertEqual(cost_from_start_cal[30][556], "no_path")

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_update_edge_weights(self):

        # set up RN

        with open(
            os.path.join(
                test_dir, "unittest_RN_pr_ii_4_before_update_edge_weights.pkl"
            ),
            "rb",
        ) as input:
            RN_pr_ii_4 = pickle.load(input)

        # set up input variables
        min_cost_str = loadfn(
            os.path.join(test_dir, "unittest_update_edge_weights_min_cost_IN.json")
        )
        with open(
            os.path.join(test_dir, "unittest_update_edge_weights_orig_graph_IN.pkl"),
            "rb",
        ) as input:
            orig_graph = pickle.load(input)

        min_cost = {}
        for key in min_cost_str:
            temp = min_cost_str[key]
            min_cost[int(key)] = temp

        # perform calc
        attrs_cal = RN_pr_ii_4.update_edge_weights(min_cost, orig_graph)

        # assert
        self.assertEqual(len(attrs_cal), 6143)
        self.assertEqual(
            attrs_cal[(556, "456+556,421")]["softplus"], 0.24363920804933614
        )
        self.assertEqual(attrs_cal[(41, "41+556,42")]["softplus"], 0.26065563056500646)
        self.assertEqual(
            attrs_cal[(308, "308+556,277")]["softplus"], 0.08666835894406484
        )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_final_PR_check(self):
        with open(os.path.join(test_dir, "unittest_RN_pr_solved.pkl"), "rb") as input:
            RN_pr_solved = pickle.load(input)

        # set up RN
        RN = copy.deepcopy(RN_pr_solved)

        # perform calc
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        RN.final_PR_check(RN.PRs)
        output = new_stdout.getvalue()
        sys.stdout = old_stdout

        # assert
        self.assertTrue(output.__contains__("No path found from any start to PR 30"))
        self.assertTrue(
            output.__contains__("NOTE: Matching prereq and byproduct found! 542")
        )
        self.assertTrue(output.__contains__("No path found from any start to PR 513"))
        self.assertTrue(output.__contains__("No path found from any start to PR 539"))

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_find_or_remove_bad_nodes(self):
        # set up RN
        RN = copy.deepcopy(self.RN_build)

        # set up input variables
        LEDC_ind = None
        LiEC_ind = None
        EC_ind = None

        for entry in RN.entries["C3 H4 O3"][10][0]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_ind = entry.parameters["ind"]
                break

        for entry in RN.entries["C4 H4 Li2 O6"][17][0]:
            if self.LEDC_mg.isomorphic_to(entry.mol_graph):
                LEDC_ind = entry.parameters["ind"]
                break

        for entry in RN.entries["C3 H4 Li1 O3"][12][1]:
            if self.LiEC_mg.isomorphic_to(entry.mol_graph):
                LiEC_ind = entry.parameters["ind"]
                break

        Li1_ind = RN.entries["Li1"][0][1][0].parameters["ind"]

        nodes = [LEDC_ind, LiEC_ind, Li1_ind, EC_ind]

        # perform calc & assert
        bad_nodes_list = RN.find_or_remove_bad_nodes(nodes, remove_nodes=False)
        self.assertEqual(len(bad_nodes_list), 231)
        print(set(bad_nodes_list))
        self.assertTrue(
            {"511,108+112", "46+556,34", "199+556,192", "456,399+543", "456,455"}
            <= set(bad_nodes_list)
        )

        bad_nodes_pruned_graph = RN.find_or_remove_bad_nodes(nodes, remove_nodes=True)
        # self.assertEqual(len(bad_nodes_pruned_graph.nodes), 10254)
        # self.assertEqual(len(bad_nodes_pruned_graph.edges), 22424)
        for node_ind in nodes:
            self.assertEqual(bad_nodes_pruned_graph[node_ind], {})

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_valid_shortest_simple_paths(self):
        with open(os.path.join(test_dir, "unittest_RN_pr_solved.pkl"), "rb") as input:
            RN_pr_solved = pickle.load(input)

        RN = copy.deepcopy(RN_pr_solved)

        EC_ind = None
        LEDC_ind = None

        for entry in RN.entries["C3 H4 O3"][10][0]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_ind = entry.parameters["ind"]
                break
        for entry in RN.entries["C4 H4 Li2 O6"][17][0]:
            if self.LEDC_mg.isomorphic_to(entry.mol_graph):
                LEDC_ind = entry.parameters["ind"]
                break

        paths = RN.valid_shortest_simple_paths(EC_ind, LEDC_ind)
        p = [
            [
                456,
                "456+556,424",
                424,
                "424,423",
                423,
                "423,420",
                420,
                "41+420,511",
                511,
            ],
            [
                456,
                "456+556,424",
                424,
                "424,423",
                423,
                "423,420",
                420,
                "420,41+164",
                41,
                "41+420,511",
                511,
            ],
            [
                456,
                "456,455",
                455,
                "455,448",
                448,
                "448,51+164",
                51,
                "51+556,41",
                41,
                "41+420,511",
                511,
            ],
            [
                456,
                "456+556,421",
                421,
                "421,424",
                424,
                "424,423",
                423,
                "423,420",
                420,
                "41+420,511",
                511,
            ],
            [
                456,
                "456+556,421",
                421,
                "421,424",
                424,
                "424,423",
                423,
                "423,420",
                420,
                "420,41+164",
                41,
                "41+420,511",
                511,
            ],
            [
                456,
                "456,455",
                455,
                "455,448",
                448,
                "448+556,420",
                420,
                "420,41+164",
                41,
                "41+420,511",
                511,
            ],
            [
                456,
                "456,455",
                455,
                "455,448",
                448,
                "448+556,420",
                420,
                "41+420,511",
                511,
            ],
            [
                456,
                "456,455",
                455,
                "455+556,423",
                423,
                "423,420",
                420,
                "41+420,511",
                511,
            ],
            [
                456,
                "456,455",
                455,
                "455+556,423",
                423,
                "423,420",
                420,
                "420,41+164",
                41,
                "41+420,511",
                511,
            ],
            [
                456,
                "456+556,424",
                424,
                "424,423",
                423,
                "423,420",
                420,
                "420,419",
                419,
                "41+419,510",
                510,
                "510,511",
                511,
            ],
        ]
        ind = 0
        paths_generated = []
        for path in paths:
            if ind == 10:
                break
            else:
                paths_generated.append(path)
                ind += 1
        print(paths_generated)
        self.assertCountEqual(p, paths_generated)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_find_paths(self):
        with open(os.path.join(test_dir, "unittest_RN_pr_solved.pkl"), "rb") as input:
            RN_pr_solved = pickle.load(input)

        # set up RN
        RN = copy.deepcopy(RN_pr_solved)

        # set up input variables
        EC_ind = None
        LEDC_ind = None

        for entry in RN.entries["C3 H4 O3"][10][0]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_ind = entry.parameters["ind"]
                break
        for entry in RN.entries["C4 H4 Li2 O6"][17][0]:
            if self.LEDC_mg.isomorphic_to(entry.mol_graph):
                LEDC_ind = entry.parameters["ind"]
                break
        Li1_ind = RN.entries["Li1"][0][1][0].parameters["ind"]

        PR_paths_calculated, paths_calculated, top_paths_list = RN.find_paths(
            [EC_ind, Li1_ind], LEDC_ind, weight="softplus", num_paths=10
        )

        if 420 in paths_calculated[0]["all_prereqs"]:
            self.assertEqual(paths_calculated[0]["byproducts"], [164])
        elif 41 in paths_calculated[0]["all_prereqs"]:
            self.assertEqual(paths_calculated[0]["byproducts"], [164, 164])

        self.assertAlmostEqual(paths_calculated[0]["cost"], 2.3135953094636403, 5)
        self.assertAlmostEqual(
            paths_calculated[0]["overall_free_energy_change"], -6.2399175587598394, 5
        )
        self.assertAlmostEqual(
            paths_calculated[0]["hardest_step_deltaG"], 0.37075842588456, 5
        )

        for path in paths_calculated:
            self.assertTrue(abs(path["cost"] - path["pure_cost"]) < 0.000000001)

    def test_mols_w_cuttoff(self):
        with open(os.path.join(test_dir, "unittest_RN_pr_solved.pkl"), "rb") as input:
            RN_pr_solved = pickle.load(input)

        RN_loaded = copy.deepcopy(RN_pr_solved)

        mols_to_keep, pruned_entries_list = ReactionNetwork.mols_w_cuttoff(
            RN_loaded, 0, build_pruned_network=False
        )

        self.assertEqual(len(mols_to_keep), 236)

    def test_parse_reaction_node(self):

        nodes = ["19+32,673", "41,992", "1+652,53+40", "4,6+5"]
        node_prod_react = []

        for node in nodes:
            r_p = ReactionNetwork.parse_reaction_node(node)
            node_prod_react.append(r_p)

        self.assertListEqual(
            node_prod_react,
            [([19, 32], [673]), ([41], [992]), ([1, 652], [40, 53]), ([4], [5, 6])],
        )

    def test_generate_node_string(self):

        react_prod = [
            ([19, 32], [673]),
            ([41], [992]),
            ([1, 652], [40, 53]),
            ([4], [5, 6]),
        ]
        node_strings = []

        for rxn in react_prod:
            node_str = ReactionNetwork.generate_node_string(rxn[0], rxn[1])
            node_strings.append(node_str)

        self.assertListEqual(
            node_strings, ["19+32,673", "41,992", "1+652,40+53", "4,5+6"]
        )

    def test_build_matrix(self):

        with open(
            os.path.join(
                test_dir, "identify_concerted_via_intermediate_unittest_RN.pkl"
            ),
            "rb",
        ) as input:
            RN_loaded = pickle.load(input)

        with open(os.path.join(test_dir, "RN_matrix_build.pkl"), "rb") as handle:
            loaded_matrix = pickle.load(handle)

        with open(
            os.path.join(test_dir, "RN_matrix_inverse_build.pkl"), "rb"
        ) as handle:
            loaded_inverse_matrix = pickle.load(handle)

        RN_loaded.build_matrix()

        def matrix_assert(computed, loaded):
            for r in loaded:  # iterate over reactants (rows)
                for p in loaded[r]:  # iterate over products (cols)
                    try:
                        for k in range(
                            len(loaded[r][p])
                        ):  # iterate over reactions (element, list form)
                            pr_free = loaded[r][p][k][0].replace("PR_", "")  # node name
                            assert computed[r][p][k][1] == loaded[r][p][k][1]  # weight
                            assert (
                                computed[r][p][k][2] == loaded[r][p][k][2]
                            )  # rxn type (elem/concert)
                            try:
                                assert computed[r][p][k][0] == pr_free
                            except AssertionError:
                                match = False
                                for perm in permutations(
                                    pr_free.split(",")[0].split("+")
                                ):
                                    permuted_rxn = (
                                        "+".join(perm) + "," + pr_free.split(",")[1]
                                    )
                                    match = permuted_rxn == computed[r][p][k][0]
                                    if match:
                                        break
                                assert match
                    except KeyError:
                        print(loaded[i])

            return loaded

        matrix_assert(RN_loaded.matrix, loaded_matrix)
        matrix_assert(RN_loaded.matrix_inverse, loaded_inverse_matrix)

    def test_build_matrix_from_reactions(self):
        with open(
            os.path.join(
                test_dir, "identify_concerted_via_intermediate_unittest_RN.pkl"
            ),
            "rb",
        ) as input:
            RN_loaded = pickle.load(input)

        RN_loaded.build_matrix_from_reactions(RN_loaded)

    def test_concerted_reaction_filter(self):
        r, r_node = ReactionNetwork.concerted_reaction_filter("6,2+7", "2+1,3")
        self.assertEqual([[1, 6], [3, 7]], r)
        self.assertEqual([[1, 6], [3, 7], ["6,2+7", "2+1,3"]], r_node)
        r, r_node = ReactionNetwork.concerted_reaction_filter("2+1,3+10", "6,2+7")
        self.assertEqual(r, None)
        self.assertEqual(r_node, None)

    def test_identify_concerted_rxns_via_intermediates(self):
        # load RN
        with open(
            os.path.join(
                test_dir, "identify_concerted_via_intermediate_unittest_RN.pkl"
            ),
            "rb",
        ) as input:
            RN_loaded = pickle.load(input)

        RN_loaded.matrix = None
        (
            v3_unique_iter1,
            v3_all_iter1,
        ) = RN_loaded.identify_concerted_rxns_via_intermediates(
            RN_loaded, single_elem_interm_ignore=[], update_matrix=True
        )
        unique_reactions = loadfn(
            os.path.join(
                test_dir,
                "identify_concerted_rxns_via_intermediates_v3_all_unique_reactions.json",
            )
        )
        self.assertEqual(len(v3_unique_iter1), len(unique_reactions))
        v3_unique_computed_set = set(map(lambda x: repr(x), v3_unique_iter1))
        v3_unique_loaded_set = set(map(lambda x: repr(x), unique_reactions))
        self.assertEqual(v3_unique_computed_set, v3_unique_loaded_set)
        reactions_with_intermediates = loadfn(
            os.path.join(
                test_dir,
                "identify_concerted_rxns_via_intermediates_v3_with_intermediate_nodes.json",
            )
        )
        self.assertEqual(len(v3_all_iter1), len(reactions_with_intermediates))
        for i in range(len(reactions_with_intermediates)):
            for j in range(len(reactions_with_intermediates[i])):
                for k in range(len(reactions_with_intermediates[i][j])):
                    for l in range(len(reactions_with_intermediates[i][j][k])):
                        if isinstance(reactions_with_intermediates[i][j][k][l], str):
                            el = reactions_with_intermediates[i][j][k][l]
                            el = el.replace("PR_", "")
                            sorted_rcts = sorted(el.split(",")[0].split("+"))
                            sorted_pros = sorted(el.split(",")[1].split("+"))
                            reactions_with_intermediates[i][j][k][l] = (
                                "+".join(sorted_rcts) + "," + "+".join(sorted_pros)
                            )
        for i in range(len(v3_all_iter1)):
            for j in range(len(v3_all_iter1[i])):
                for k in range(len(v3_all_iter1[i][j])):
                    for l in range(len(v3_all_iter1[i][j][k])):
                        if isinstance(v3_all_iter1[i][j][k][l], str):
                            el = v3_all_iter1[i][j][k][l]
                            sorted_rcts = sorted(el.split(",")[0].split("+"))
                            sorted_pros = sorted(el.split(",")[1].split("+"))
                            v3_all_iter1[i][j][k][l] = (
                                "+".join(sorted_rcts) + "," + "+".join(sorted_pros)
                            )
        v3_all_computed_set = set(map(lambda x: repr(x), v3_all_iter1))
        v3_all_loaded_set = set(map(lambda x: repr(x), reactions_with_intermediates))
        try:
            self.assertEqual(v3_all_computed_set, v3_all_loaded_set)
        except AssertionError:
            c_minus_l = v3_all_computed_set - v3_all_loaded_set
            l_minus_c = v3_all_loaded_set - v3_all_computed_set
            self.assertEqual(len(c_minus_l), len(l_minus_c))
            print(len(c_minus_l))

    def test_identify_concerted_rxns_for_specific_intermediate(self):

        with open(
            os.path.join(
                test_dir, "identify_concerted_via_intermediate_unittest_RN.pkl"
            ),
            "rb",
        ) as input:
            RN_loaded = pickle.load(input)
        RN_loaded.matrix = None
        (
            reactions,
            reactions_with_nodes,
        ) = RN_loaded.identify_concerted_rxns_for_specific_intermediate(
            RN_loaded.entries_list[1],
            RN_loaded,
            single_elem_interm_ignore=[],
            update_matrix=False,
        )
        unique_reactions = set(map(lambda x: repr(x), reactions))
        self.assertEqual(len(unique_reactions), 6901)

    def test_add_concerted_rxns(self):
        with open(
            os.path.join(
                test_dir, "identify_concerted_via_intermediate_unittest_RN.pkl"
            ),
            "rb",
        ) as input:
            RN_loaded = pickle.load(input)

        reactions = loadfn(
            os.path.join(test_dir, "add_concerted_unittest_rxn_list.json")
        )

        RN_loaded.add_concerted_rxns(RN_loaded, reactions)

        # self.assertEqual(len(RN_loaded.graph.nodes), 61600)
        # self.assertEqual(len(RN_loaded.graph.edges), 181703)
        self.assertTrue("6+181,8+178" in RN_loaded.graph.nodes)
        self.assertAlmostEqual(
            RN_loaded.graph.nodes["6+181,8+178"]["free_energy"], -4.47641198
        )


if __name__ == "__main__":
    unittest.main()
