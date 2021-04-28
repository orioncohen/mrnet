import pandas as pd
import itertools


def get_reaction_indices(RN, rxn_dataframe):
    """
    given a RN and a data_frame this will the sparse non-zero entries of the
    graph in COO format. The data will be a list of values, the coords will
    be a list of tuples, and the node_index is a map between absolute node_id's
    and relative indices in the matrix.

    :param RN:
    :param rxn_dataframe:
    :return: data, coords, node_index
    """
    # reactions are indexed first, then nodes
    rxn_index = {row['node_id']: i
                 for i, row in rxn_dataframe.iterrows()}
    offset = len(rxn_index)
    mol_index = {mol.entry_id: mol.parameters['ind'] + offset
                 for mol in RN.entries_list}
    node_index = {**rxn_index, **mol_index}
    i = []
    j = []
    data = []
    for rxn_index, rxn in rxn_dataframe.iterrows():
        in_edges = itertools.product(rxn['reactants'], [rxn['node_id']])
        out_edges = itertools.product([rxn['node_id']], tuple(rxn['products']))
        edges = itertools.chain(in_edges, out_edges)
        for edge in edges:
            i.append(node_index[edge[0]])
            j.append(node_index[edge[1]])
            data.append(1)
    coords = zip(i, j)
    return data, coords, node_index


def construct_reaction_dataframe(RN):
    """
    given a RN, this constructs a dataframe with the minimal information needed to characterize
    whether or not a concerted rxn is valid. Each reaction object is split into its forward and
    reverse (A and B) components and both are added to the dataframe separately.

    :param RN:
    :return: rxn_dataframe
    """
    node_id = []
    delta_g = []
    rxn_type = []
    reactants = []
    products = []
    for i, rxn in enumerate(RN.reactions):
        # append rxn A
        node_id.append((rxn.parameters['ind'], 'A'))
        delta_g.append(rxn.free_energy_A)
        rxn_type.append(rxn.rxn_type_A)
        reactants.append(rxn.reactant_ids)
        products.append(rxn.product_ids)
        # append rxn B
        node_id.append((rxn.parameters['ind'], 'B'))
        delta_g.append(rxn.free_energy_B)
        rxn_type.append(rxn.rxn_type_B)
        reactants.append(rxn.product_ids)
        products.append(rxn.reactant_ids)
    # map to dict to easily transform to df
    col_names = {'node_id': node_id,
                'delta_g': delta_g,
                'rxn_type': rxn_type,
                'reactants': reactants,
                'products': products}
    rxn_dataframe = pd.DataFrame(col_names)
    return rxn_dataframe


# TODO for Chloe: make the matrix you need
def construct_matrix(data, coords):
    """
    This will construct a matrix of arbitrary form from just the data and coords. I have left this
    open ended so that Chloe can choose whatever format is best.

    :param matrix:
    :return:
    """


# TODO for Chloe: implement PyCu matrix multiplication
def square_matrix(matrix):
    """
    this should square a matrix using PyCu. Should run on GPU!

    :param matrix:
    :return:
    """
    return


def get_rxn_subspace(squared_adjacency_matrix, reaction_dataframe):
    """
    this should create a submatrix with only the reaction subspace in the squared adjacency matrix.
    Perhaps it should not be an independent function.

    :param squared_adjacency_matrix:
    :param reaction_dataframe:
    :return:
    """
    return


# TODO for Atsushi: map a validation kernel over a matrix with
def validate_concerted_rxns(rxn_adjacency_matrix, rxn_dataframe):
    """
    This should map a validation kernel written in Cython or Numba over the whole rxn_adjacency_matrix
    and return a new matrix with all entries invalid entries removed. Should run on GPU!

    :param rxn_adjacency_matrix:
    :param rxn_dataframe:
    :return:
    """
    return


def rxn_matrix_to_list(RN, valid_concerted_matrix, rxn_dataframe):
    """
    This should take the RN, the reaction matrix, the reaction data_frame and create a nice
    new list of concerted reactions. Maybe less parameters are required.

    :param RN:
    :param valid_concerted_matrix:
    :param rxn_dataframe:
    :return:
    """
    return
