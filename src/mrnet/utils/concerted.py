import pandas as pd
import itertools
import scipy.sparse
import cupy
import cupyx.scipy.sparse


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


def construct_matrix(data, coords, size):
    """
    This will construct a matrix of arbitrary form from just the data and coords. I have left this
    open ended so that Chloe can choose whatever format is best.

    scipy COO sparse matrix format

    Note: scipy matrix can be converted to matrix market format by
    scipy.io.mmwrite

    :param data: list of nonzero values
    :param coords: list of tuples that indicate row and col indices
    :param size: the number of rows/columns in the square matrix
    :return: resulting matrix in scipy sparse COO format
    """
    row_idx, col_idx = zip(*coords)
    return scipy.sparse.coo_matrix((data, (row_idx, col_idx)),
            shape=(size, size))


def square_matrix(matrix):
    """
    this should square a matrix using PyCu. Should run on GPU!

    :param matrix: scipy sparse coo matrix
    :return: matrix squared, also in scipy sparse coo format
    """
    data = cupy.array(matrix.data, dtype=float)
    row = cupy.array(matrix.row, dtype=float)
    col = cupy.array(matrix.col, dtype=float)
    matrix_csr = cupyx.scipy.sparse.coo_matrix((
        data, (row, col)), shape=matrix.shape).tocsr()
    return (matrix_csr * matrix_csr).tocoo().get()


def square_matrix_scipy(matrix):
    """
    barebone scipy version of matmul

    :param matrix: scipy sparse coo matrix
    :return: matrix squared, also in scipy sparse coo format
    """
    matrix_csr = matrix.tocsr()
    return (matrix_csr * matrix_csr).tocoo()


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
    # this is the validation kernel that we need to map over the sparse matrix
    # it might be advantageous to somehow write it up in c?
    # TODO for Atsushi: consider writing these operations in C or C++ with Cython?
    def validate_rxn_pair(rxn1_index, rxn2_index):
        # using inequalities to confirm thermodynamic favorability
        delta_g_1 = rxn_dataframe.iloc[rxn1_index]['delta_g']
        delta_g_2 = rxn_dataframe.iloc[rxn2_index]['delta_g']
        if delta_g_1 < 0 or delta_g_1 < delta_g_2:
            return 0

        # TODO for Orion: add logic for type checking

        # checking to ensure the reaction and products are not too long
        all_reactants = set(rxn_dataframe.iloc[rxn1_index]['reactants']) | \
                        set(rxn_dataframe.iloc[rxn2_index]['reactants'])
        all_products = set(rxn_dataframe.iloc[rxn1_index]['products']) | \
                       set(rxn_dataframe.iloc[rxn2_index]['products'])
        n_products = len(all_products - all_reactants)
        n_reactants = len(all_reactants - all_products)
        if n_products > 2 or n_reactants > 2 \
                or n_products == 0 or n_reactants == 0:
            return 0
        return 1

    # TODO for Atsushi: make this work for a sparse matrix and accelerate it with Numba
    # the for loop I wrote is very specific to dense matrices! it should be
    # made to work for sparse matrices too!
    for i in range(len(rxn_adjacency_matrix)):
        for j in range(len(rxn_adjacency_matrix)):
            if i == j:
                continue
            rxn_adjacency_matrix[i, j] = validate_rxn_pair(i, j)
    return rxn_adjacency_matrix


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
