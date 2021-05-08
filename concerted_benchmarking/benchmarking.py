from mrnet.network.reaction_network import (
    ReactionNetwork
)
from mrnet.utils.concerted import (
    square_matrix_scipy
)
import os
import pickle
from monty.serialization import loadfn
import time

benchmarking_dir = os.getcwd()

with open(
        os.path.join(
            benchmarking_dir, "large_reaction_network.pkl"
        ),
        "rb",
) as input:
    RN_loaded = pickle.load(input)

start_time = time.time()
adjacency_matrix, rxn_dataframe, node_index = RN_loaded.build_matrix_from_reactions(RN_loaded)
print('Matrix built at: ', time.time() - start_time)
squared_matrix = square_matrix_scipy(adjacency_matrix)
print('Matrix squared at: ', time.time() - start_time)
