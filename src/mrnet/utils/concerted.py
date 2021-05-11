import pandas as pd
import itertools

import scipy.sparse
try:
    import cupy
    import cupyx.scipy.sparse
except ImportError:
    cupy = None
    print("WARNING: The concerted module can only be used in a CUDA supported environment.")

from numba import cuda
import math
import itertools
from itertools import accumulate
import numpy as np



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


# TODO for Atsushi: map a validation kernel over a matrix with
def validate_concerted_rxns_ori(rxn_adjacency_matrix, rxn_dataframe):
    """
    This should map a validation kernel written in Cython or Numba over the whole rxn_adjacency_matrix
    and return a new matrix with all entries invalid entries removed.
    This also shrinks the matrix to select only the reaction subspace.
    Should run on GPU!

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

    # TODO for Atsushi: accelerate this with Numba!
    rxn_dim = len(rxn_dataframe)
    rx = rxn_adjacency_matrix  # this could be a copy but that would be slow?
    rx.resize((rxn_dim, rxn_dim))

    row = []
    col = []
    data = []
    for num, i, j in zip(range(len(rx.data)), rx.row, rx.col):
        if i != j and validate_rxn_pair(i, j):
            row.append(i)
            col.append(j)
            data.append(1)

    return scipy.sparse.coo_matrix((data, (row, col)), shape=rx.shape)



# TODO for Atsushi: map a validation kernel over a matrix with
def validate_concerted_rxns(rxn_adjacency_matrix, rxn_dataframe):
    """
    This should map a validation kernel written in Cython or Numba over the whole rxn_adjacency_matrix
    and return a new matrix with all entries invalid entries removed.
    This also shrinks the matrix to select only the reaction subspace.
    Should run on GPU!

    :param rxn_adjacency_matrix:
    :param rxn_dataframe:
    :return:
    """

    # device function
    @cuda.jit(device=True)
    def prod_set_2(list_1, list_2):
        count = 0
        for i1 in list_1:
            for i2 in list_2:
                if i1 == i2:
                    count = count + 1
        return count

    @cuda.jit(device=True)
    def prod_set_3(list_1, list_2, list_3):
        count = 0
        for i1 in list_1:
            for i2 in list_2:
                for i3 in list_3:
                    if i1 == i2 and i2==i3:
                        count = count + 1
        return count

    @cuda.jit(device=True)
    def prod_set_4(list_1, list_2, list_3, list_4):
        count = 0
        for i1 in list_1:
            for i2 in list_2:
                for i3 in list_3:
                    for i4 in list_4:
                        if i1 == i2 and i2==i3 and i3==i4:
                            count = count + 1
        return count


    # this is the validation kernel that we need to map over the sparse matrix
    # it might be advantageous to somehow write it up in c?
    # TODO for Atsushi: consider writing these operations in C or C++ with Cython?

    # cuda kernel
    @cuda.jit    
    def validate_rxn_pair(d_rxn_df_deltag, d_list_reacts, d_list_prodct, d_indx_reacts, d_indx_prodct, d_rxn_mat_row, d_rxn_mat_col, d_rxn_mat_data):

        abs_pos = cuda.grid(1)
        if abs_pos < d_rxn_mat_col.size:

            rxn1_index = d_rxn_mat_row[abs_pos]
            rxn2_index = d_rxn_mat_col[abs_pos]

            val = 1 # concerted reaction is valid

            # using inequalities to confirm thermodynamic favorability --------------------
            # delta_g_1 = d_rxn_dataframe.iloc[rxn1_index]['delta_g']
            # delta_g_2 = d_rxn_dataframe.iloc[rxn2_index]['delta_g']
            delta_g_1 = d_rxn_df_deltag[rxn1_index]
            delta_g_2 = d_rxn_df_deltag[rxn2_index]
            if delta_g_1 < 0 or delta_g_1 < delta_g_2:
                val = 0 # concerted reaction is invalid

            # TODO for Orion: add logic for type checking ---------------------------------
            # 
            # 

            # checking to ensure the reaction and products are not too long ---------------
            # all_reactants = set(d_rxn_dataframe.iloc[rxn1_index]['reactants']) | \
            #                 set(d_rxn_dataframe.iloc[rxn2_index]['reactants'])
            # all_products  = set(d_rxn_dataframe.iloc[rxn1_index]['products'])  | \
            #                 set(d_rxn_dataframe.iloc[rxn2_index]['products'])
            # n_products  = len(all_products - all_reactants)
            # n_reactants = len(all_reactants - all_products)


            # note: we cannot use "set" inside the kernel
            # dynamic memory allocation inside a kernel is disabled

            rct1 = d_list_reacts[d_indx_reacts[rxn1_index]:d_indx_reacts[rxn1_index+1]]
            rct2 = d_list_reacts[d_indx_reacts[rxn2_index]:d_indx_reacts[rxn2_index+1]]
            pdt1 = d_list_prodct[d_indx_prodct[rxn1_index]:d_indx_prodct[rxn1_index+1]]
            pdt2 = d_list_prodct[d_indx_prodct[rxn2_index]:d_indx_prodct[rxn2_index+1]]

            p_ab = prod_set_2(rct1,rct2)
            p_cd = prod_set_2(pdt1,pdt2)
            p_ac = prod_set_2(rct1,pdt1)
            p_ad = prod_set_2(rct1,pdt2)
            p_bc = prod_set_2(rct2,pdt1)
            p_bd = prod_set_2(rct2,pdt2)

            p_abc = prod_set_3(rct1,rct2,pdt1)
            p_abd = prod_set_3(rct1,rct2,pdt2)
            p_acd = prod_set_3(rct1,pdt1,pdt2)
            p_bcd = prod_set_3(rct2,pdt1,pdt2)

            p_abcd = prod_set_4(rct1,rct2,pdt1,pdt2)

            itsxn = p_ac + p_ad + p_bc + p_bd - p_abc - p_abd - p_acd - p_bcd + p_abcd

            n_reactants  = len(rct1) + len(rct2) - p_ab - itsxn
            n_products   = len(pdt1) + len(pdt2) - p_cd - itsxn

            if n_products > 2 or n_reactants > 2 \
                    or n_products == 0 or n_reactants == 0:
                val = 0 # concerted reaction is invalid

            # output result -------------------------------------------------------------
            d_rxn_mat_data[abs_pos] = val            

    ### end of the kernel (validate_rxn_pair)



    # TODO for Atsushi: accelerate this with Numba!
    rxn_dim = len(rxn_dataframe)
    rx = rxn_adjacency_matrix  # this could be a copy but that would be slow?
    rx.resize((rxn_dim, rxn_dim))


    # reshape data for cuda kernel
    list_reacts = rxn_dataframe['reactants'].tolist()
    list_prodct = rxn_dataframe['products'].tolist()
    indx_reacts = [len(v) for v in list_reacts]
    indx_prodct = [len(v) for v in list_prodct]
    indx_reacts = list(accumulate(indx_reacts))
    indx_prodct = list(accumulate(indx_prodct))
    indx_reacts.insert(0,0)
    indx_prodct.insert(0,0)
    list_reacts = list(itertools.chain.from_iterable(list_reacts))
    list_prodct = list(itertools.chain.from_iterable(list_prodct))

    # send data from host to device
    d_rxn_df_deltag = cuda.to_device(rxn_dataframe['delta_g'].tolist())
    d_list_reacts   = cuda.to_device(list_reacts)
    d_list_prodct   = cuda.to_device(list_prodct)    
    d_indx_reacts   = cuda.to_device(indx_reacts)
    d_indx_prodct   = cuda.to_device(indx_prodct)

    d_rxn_mat_row   = cuda.to_device(rx.row)
    d_rxn_mat_col   = cuda.to_device(rx.col)
    d_rxn_mat_data  = cuda.device_array(rxn_dim)

    # determine thread/block size
    threadsperblock = 128
    blockspergrid   = int(math.ceil(rxn_dim / threadsperblock))

    #GPU kernel
    validate_rxn_pair[blockspergrid, threadsperblock] \
    (d_rxn_df_deltag, d_list_reacts, d_list_prodct, d_indx_reacts, d_indx_prodct, d_rxn_mat_row, d_rxn_mat_col, d_rxn_mat_data)

    # send back data to host device
    data = d_rxn_mat_data.copy_to_host()

    # create sparce matrix
    output = scipy.sparse.coo_matrix((data, (rx.row, rx.col)), shape=rx.shape)
    output.eliminate_zeros()

    return output


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
