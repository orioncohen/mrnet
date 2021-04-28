import pandas as pd
import itertools


def get_reaction_indices(RN, rxn_dataframe):
    """
    given a RN and a data_frame this will return a list of all the indices of all the edges
    in the adjacency matrix.

    :param RN:
    :param rxn_dataframe:
    :return:
    """
    return


def construct_reaction_dataframe(RN):
    """
    given a RN, this should construct a dataframe with the minimal information needed to characterize
    whether or not a concerted rxn is valid.

    :param RN:
    :return:
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
