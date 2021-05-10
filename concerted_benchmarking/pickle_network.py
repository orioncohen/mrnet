from mrnet.network.reaction_network import (
    ReactionNetwork
)
import os
import pickle
from monty.serialization import loadfn

benchmarking_dir = os.getcwd()

entries = loadfn(os.path.join(benchmarking_dir, "mrnet_all_of_entries_16334.json"))
RN = ReactionNetwork.from_input_entries(entries)
RN.build()
pickle_in = open(
    os.path.join(benchmarking_dir, benchmarking_dir + "large_reaction_network.pkl"),
    "wb",
)
pickle.dump(RN, pickle_in)
pickle_in.close()
