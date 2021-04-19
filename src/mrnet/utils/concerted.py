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
    return


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
