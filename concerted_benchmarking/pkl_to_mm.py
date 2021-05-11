import pickle
import scipy.io
from mrnet.utils.concerted import construct_matrix
from mrnet.utils.concerted import construct_reaction_dataframe
from mrnet.utils.concerted import get_reaction_indices

with open("large_reaction_network.pkl", "rb") as f:
    rn = pickle.load(f)

rxn_dataframe = construct_reaction_dataframe(rn)
data, coords, node_index = get_reaction_indices(rn, rxn_dataframe)
A = construct_matrix(data, coords, len(node_index))
scipy.io.mmwrite("large_RN.mm", A)
