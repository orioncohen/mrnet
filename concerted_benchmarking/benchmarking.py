from mrnet.network.reaction_network import (
    ReactionNetwork
)
from mrnet.utils.concerted import (
    square_matrix_scipy,
    validate_concerted_rxns
)
import os
import sys
import pickle
from monty.serialization import loadfn
import time

benchmarking_dir = os.getcwd()

keyword = sys.argv[1]

if keyword == 'large':
    network_file = "large_reaction_network.pkl"
elif keyword == 'small':
    network_file = "small_reaction_network.pkl"
else:
    raise Exception("benchmarking.py takes 'large' or 'small' as a keyword to "
                    "select which network to benchmark.")

with open(
        os.path.join(
            benchmarking_dir, network_file
        ),
        "rb",
) as input:
    RN_loaded = pickle.load(input)

start_time = time.time()
adjacency_matrix, rxn_dataframe, node_index = RN_loaded.build_matrix_from_reactions(RN_loaded)
print('Matrix built at: ', time.time() - start_time)
squared_matrix = square_matrix_scipy(adjacency_matrix)
print('Matrix squared at: ', time.time() - start_time)
valid_reactions = validate_concerted_rxns(squared_matrix, rxn_dataframe)
print('Matrix validated at: ', time.time() - start_time)