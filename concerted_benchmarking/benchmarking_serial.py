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

# keyword = sys.argv[1]
keyword = 'small'
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
unique_rxns, all_rxns = ReactionNetwork.identify_concerted_rxns_via_intermediates(RN_loaded, single_elem_interm_ignore=[])
print('Concerted identified at: ', time.time() - start_time)
RN_concerted = ReactionNetwork.add_concerted_rxns(RN_loaded, unique_rxns)
print('Concerted added at: ', time.time() - start_time)