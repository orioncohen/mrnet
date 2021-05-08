from mrnet.network.reaction_network import (
    ReactionNetwork
)


import os
import pickle
from monty.serialization import loadfn

benchmarking_dir = "/Users/orioncohen/software/mrnet/concerted_benchmarking/"

with open(
        os.path.join(
            benchmarking_dir, "large_reaction_network.pkl"
        ),
        "rb",
) as input:
    RN_loaded = pickle.load(input)

adjacency_matrix, rxn_dataframe, node_index =  RN_loaded.build_matrix_from_reactions(RN_loaded)
print('hi')