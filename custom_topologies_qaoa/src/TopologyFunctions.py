# Copyright (c) 
# SPDX-License-Identifier: GPL-2.0
# coding: utf-8

import numpy as np
import itertools
import networkx as nx
import collections

#####################################################################################################
### General helper functions
#####################################################################################################


def get_qubits_for_coupling_map(coupling_map):
    return set(sum(coupling_map, []))

def add_qubit_connection(coupling_map, index1, index2):
    coupling_map.append([index1, index2])
    coupling_map.append([index2, index1])
    return coupling_map

def remove_qubit_connection(coupling_map, index1, index2):
    for ind in range(len(coupling_map)):
        if coupling_map[ind][0] == index1 and coupling_map[ind][1] == index2:
            coupling_map.pop(ind)
            coupling_map.pop(ind) #since the first part is removed, all indices after that are shifted by 1
            break
    return coupling_map

def get_adjacency_matrix_from_graph(G): #if two qubits need to be connected, there is a 1, the diagonal is 0. Since we are dealing with undirected graphs, the matrix is symmetric.
    n_qubits = len(list(G.nodes())) 
    edges=list(G.edges())
    ad=np.zeros((n_qubits, n_qubits))
    for pair in edges:
        ad[pair[0], pair[1]]=1
        ad[pair[1], pair[0]]=1
    return ad


# get edge pairs with a given distance 
def get_eligible_edges(coupling_map, coupling_map_distances, distance):
    qubits = set(sum(coupling_map, []))
    eligible_edges = []
    for (i, j) in itertools.combinations(qubits, 2):
        if coupling_map_distances[i][j] == distance:
            eligible_edges.append((i, j))
    return eligible_edges

#create nx graph from coupling map
def get_nx_graph(base_map):
    qubits = get_qubits_for_coupling_map(base_map)
    g = nx.Graph()
    g.add_nodes_from(list(qubits))
    g.add_edges_from(base_map)
    return g


# extract mean execution time for each native gate for a certain IBM-Q-Backend. 
def get_gate_times_from_backend(backend):

    """
    returns a dictionary: {gate name, mean gate time, std of mean gate time}
    """

    config = backend.configuration()
    gate_set=config.basis_gates

    prop_dict = backend.properties().to_dict()
    gate_times={}

    # Gate times in ns 
    for gate in gate_set:
        gate_times_tmp=[]
        for j in range(len(prop_dict['gates'])):

            if prop_dict['gates'][j]['gate'] == gate and prop_dict['gates'][j]['gate'] != 'reset':
                #print(prop_dict['gates'][j]['gate'] )
                gate_times_tmp.append(prop_dict['gates'][j]['parameters'][1]['value'])
            gate_times[gate] = round(np.mean(gate_times_tmp),3), round(np.std(gate_times_tmp),3)

            if prop_dict['gates'][j]['gate']==gate and prop_dict['gates'][j]['gate'] == 'reset':
                gate_times_tmp.append(prop_dict['gates'][j]['parameters'][0]['value'])
            gate_times[gate] = round(np.mean(gate_times_tmp),3), round(np.std(gate_times_tmp),3)

    return gate_times




#####################################################################################################
### IBM-Q Heavy-hex Topology
#####################################################################################################

# The heavy hex lattice topology consists of a sequence of rows with 3 qubit cells. These cells can be decomposed in an top and bottom line of qubits,
# as well as intermediate qubits connecting the top and bottom row, and thereby serving as the borders of the unit cells.


#####################################################################################################
# Create and extend IBM-Q Topologies 
#####################################################################################################



def create_heavy_hex_IBMQ(n_unit_rows, n_columns):

    """
    # Get heavy-hex lattcie topology for IBM-Q. 
    # Parameters: number of (unit cell) rows and columns (unit cells per row) of the coupling map to be created.
    # Output: 
    # coupling_map: Coupling map of the new topology 
    # n_qubits_new: Number of qubits in the new topology 
    #
    # The coupling map being created follows the scheme of the IBM-Q Hummingbird and Washington, having two "loose ends" in the first and lastb row. 
    """

    coupling_map = []
    #number of qubits per row and intermediate qubits in the final topology. IN the first and last row, there is one qubit less. 
    n_qubits_row = 5 + (n_columns-1)*4 + 2
    n_qubits_intermediate = n_columns + 1
    n_rows = n_unit_rows + 1 #number of qubit rows = number of unit cell rows + 1

    for row in range(n_rows):
        if row == 0:
            leftmost_qubit_index = 0
        if 0 < row: 
            leftmost_qubit_index = (n_qubits_row - 1 + n_qubits_intermediate) + (row-1)*(n_qubits_intermediate + n_qubits_row )

    #first add the row
        current_qubit = leftmost_qubit_index

        #first and last row have one qubit less
        if row == 0 or row == n_rows - 1: 
            qubit_line_length = n_qubits_row - 1
        else:
            qubit_line_length = n_qubits_row

        for i in range(qubit_line_length - 1):
            coupling_map = add_qubit_connection(coupling_map, current_qubit, current_qubit+1)
            current_qubit += 1


    #then add the intermediate line below, except for the last row.
        current_qubit += 1
        starting_index_next_row = current_qubit + n_qubits_intermediate
        if row < n_rows - 1:
            for j in range(n_qubits_intermediate):
                if row % 2 == 0:
                    upper_qubit_index = leftmost_qubit_index + j*4
                    lower_qubit_index = starting_index_next_row + j*4

                if row % 2 == 1:
                    upper_qubit_index = leftmost_qubit_index + j*4 + 2
                    lower_qubit_index = starting_index_next_row + j*4 + 2
                
                if row == n_rows - 2: #the lower indices in the last row are shifted since one qubit on the left is missing
                    lower_qubit_index = starting_index_next_row + j*4 + 1

                coupling_map = add_qubit_connection(coupling_map, upper_qubit_index, current_qubit)
                coupling_map = add_qubit_connection(coupling_map, current_qubit, lower_qubit_index)
                #print(current_qubit, upper_qubit_index, lower_qubit_index)
                current_qubit += 1

    return coupling_map


def get_extended_heavy_hex_IBMQ(coupling_map_initial, extra_rows, extra_columns):

    """
    # Get extended heavy-hex lattcie topology for IBM-Q. 
    # Parameters: coupling map of the initial topology, number of rows and columns (unit cells) that should be added.
    # Output: 
    # coupling_map: Coupling map of the new topology 
    # n_qubits_new: NUmber of qubits in the new topology 
    # n_rows: Number of rows in teh new topology (number of unit cell rows = n_rows -1) 
    # n_cells_per_row: Number of unit cells per row ~ number of columns
    # important: If the original IBM-Q Washington is used as basis, the missing qubit connections between [8,9] and [109, 114] need to be added by calling "repair_IBMQ_Washington_topology"
    """

    """
    Step 1: Characterize the current topology
    """
    
    if len(coupling_map_initial) > 0:
        n_qubits_initial=len(get_qubits_for_coupling_map(coupling_map_initial))
        ad=get_adjacency_matrix_from_graph(get_nx_graph(coupling_map_initial))

        # qubits in the first and last row. Find qubits which have only one connection, to get the number of qubits per row
        loose_ends=[]
        for l in range(len(ad)):
            line=ad[l,:]
            if np.round(np.sum(line)).astype(int) == 1: 
                loose_ends.append(l)

        #number of unit cells per row in the current topology = N. Number of qubits per row = n. Number of qubits per row for N unit cells = 5 + (N-1)*4. In the first and last row, there are N unit cells + 1 qubit. 
        #In all others, N qubit cells + 2 qubits
        n_unit_cells_hor = np.round((loose_ends[0] - 5)/4 + 1).astype(int)
        n_unit_cells_vert = np.round((n_qubits_initial - (4*n_unit_cells_hor + 1))/(5*n_unit_cells_hor + 4)).astype(int)


        # number of qubits per line in the current coupling map. In the first line, there is one qubit less (and we start counting from 0), therefore we add +2
        n_qubits_row_initial = loose_ends[0] + 2
        n_qubits_intermediate_initial = n_unit_cells_hor + 1  #+1 to include the border of the unit cell

    if len(coupling_map_initial) == 0:
        n_unit_cells_hor = 0
        n_unit_cells_vert = 0
        n_qubits_row_initial = 3
        n_qubits_intermediate_initial = 1


    """
    Step 2: Create extended topology from scratch to stick to the numbering scheme
    """
    coupling_map = []
    #number of qubits per row and intermediate qubits in the final topology. There will be 4 qubits added per row and 1 intermediate qubit for each unit cell that is added to the right
    n_qubits_row = n_qubits_row_initial + 4*extra_columns
    n_qubits_intermediate = n_qubits_intermediate_initial + extra_columns

    #number of rows in the final topology. For each unit cell row, 1 normal and 1 intermediate qubit row is added.
    n_rows = n_unit_cells_vert + 1 + extra_rows


    for row in range(n_rows):
        if row == 0:
            leftmost_qubit_index = 0
        if 0 < row: 
            leftmost_qubit_index = (n_qubits_row - 1 + n_qubits_intermediate) + (row-1)*(n_qubits_intermediate + n_qubits_row )

    #first add the row
        current_qubit = leftmost_qubit_index

        #first and last row have one qubit less
        if row == 0 or row == n_rows - 1: 
            qubit_line_length = n_qubits_row - 1
        else:
            qubit_line_length = n_qubits_row

        for i in range(qubit_line_length - 1):
            coupling_map = add_qubit_connection(coupling_map, current_qubit, current_qubit+1)
            current_qubit += 1


    #then add the intermediate line below, except for the last row.
        current_qubit += 1
        starting_index_next_row = current_qubit + n_qubits_intermediate
        if row < n_rows - 1:
            for j in range(n_qubits_intermediate):
                if row % 2 == 0:
                    upper_qubit_index = leftmost_qubit_index + j*4
                    lower_qubit_index = starting_index_next_row + j*4

                if row % 2 == 1:
                    upper_qubit_index = leftmost_qubit_index + j*4 + 2
                    lower_qubit_index = starting_index_next_row + j*4 + 2
                
                if row == n_rows - 2: #the lower indices in the last row are shifted since one qubit on the left is missing
                    lower_qubit_index = starting_index_next_row + j*4 + 1

                coupling_map = add_qubit_connection(coupling_map, upper_qubit_index, current_qubit)
                coupling_map = add_qubit_connection(coupling_map, current_qubit, lower_qubit_index)
                #print(current_qubit, upper_qubit_index, lower_qubit_index)
                current_qubit += 1
    n_qubits_new = current_qubit + 1
    n_cells_per_row = n_unit_cells_vert + extra_rows

    return coupling_map, n_qubits_new, n_rows, n_cells_per_row



# for IBM-Q Washington: Remove qubit connections between certain qubits to represent the real topology
def remove_conn_IBMQ_Washington(coupling_map_initial):
    coupling_map = coupling_map_initial.copy()
    indices = [[8, 9], [109, 114]]
    for inds in indices:
        remove_qubit_connection(coupling_map, inds[0], inds[1])
    return coupling_map
    

def repair_IBMQ_Washington_topology(coupling_map):
    ## Repairing the broken connections for the IBM-Q Washington QPU
    coupling_map = add_qubit_connection(coupling_map, 8, 9)
    coupling_map = add_qubit_connection(coupling_map, 109, 114)
    return coupling_map


def get_number_qubits_heavy_hex(n_unit_rows, n_columns):
    return 1 + 4*n_unit_rows + 4*n_columns + 5*n_unit_rows*n_columns



#####################################################################################################
# Plotting functions to use plot_coupling_map for heavy-hex topology
#####################################################################################################


"""
Functions for plotting the coupling map of the heavy hex topology with the qiskit built-in function "plot_coupling_map".
This function requires 
1) A list of qubit coordinates
2) A coupling map containing each connection only once
"""

def get_qubit_coordinates_heavy_hex(n_unit_rows, n_columns, zoom):
    # Zoom: Factor to multiply all coordinates with to modify the appearance of the plot. 

    # number of qubits per line in the current coupling map. In the first line, there is one qubit less (and we start counting from 0)
    n_qubits_row = 5 + (n_columns-1)*4 + 2
    n_qubits_intermediate = n_columns + 1  #+1 to include the border of the unit cell

    #number of rows in the current topology, including only normal rows
    n_rows = n_unit_rows + 1

    qubit_coordinates = []

    for row in range(n_rows):
        y0 = 2*row
    #first add the row

        #first and last row have one qubit less
        if row == 0 or row == n_rows - 1: 
            qubit_line_length = n_qubits_row - 1
        else:
            qubit_line_length = n_qubits_row

        # the actual coordinates are transposed since the plot_coupling_map function uses [y, -x] as coordinates for plotting.
        if row == n_rows -1 and n_unit_rows > 1:
            for i in range(qubit_line_length):
                qubit_coordinates.append([y0*zoom, (i+1)*zoom])
        if row == n_rows -1 and n_unit_rows == 1: #if there is only one unit cell, the positions are shifted to the left in the last row. 
            for i in range(qubit_line_length):
                qubit_coordinates.append([y0*zoom, (i)*zoom])
        if row < n_rows -1 and n_unit_rows > 1:
            for i in range(qubit_line_length):
                qubit_coordinates.append([y0*zoom, i*zoom])
        if row < n_rows -1 and n_unit_rows == 1:
            for i in range(qubit_line_length):
                qubit_coordinates.append([y0*zoom, (i+1)*zoom]) #if there is only one unit cell, the positions are shifted by 1 in the first row to avoid an x-position of -1 in the last row


    #then add the intermediate line below, except for the last row.
        if row < n_rows - 1:
            for j in range(n_qubits_intermediate):
                if row % 2 == 0 and n_unit_rows > 1:
                    qubit_coordinates.append([(y0+1)*zoom, (j*4)*zoom])
                if row % 2 == 0 and n_unit_rows == 1:
                    qubit_coordinates.append([(y0+1)*zoom, (j*4+1)*zoom])
                if row % 2 == 1:
                    qubit_coordinates.append([(y0+1)*zoom, (j*4 + 2)*zoom])

    return qubit_coordinates



def get_coupling_map_single_heavy_hex(coupling_map):

    qubits = set(sum(coupling_map, []))

    cmap_single=[]
    for (i, j) in itertools.combinations(qubits, 2):
        if [i,j] in coupling_map: 
            cmap_single.append([i,j])

    return cmap_single



#####################################################################################################
### Analyze and increase coupling density for arbitrary topology
#####################################################################################################


#calculates the coupling density = number of connections / number of all possible connections (for all-to-all-conn.)
def get_coupling_density(base_map):
    number_of_nodes = len(set(sum(base_map, [])))
    max_edges = (number_of_nodes)*(number_of_nodes-1) 
    density = len(base_map)/max_edges

    return density

# calculates the mean number of nearest neighbours for all qubits
def get_num_NN(base_map):
    qubits = set(sum(base_map, []))
    distance = 1 #get number of qubits having distance 1 to each qubit = nearest neighbors
    g = get_nx_graph(base_map)
    coupling_map_distances = dict(nx.all_pairs_shortest_path_length(g))

    num_NNs=[]
    for q in qubits:
        num_NNs.append(collections.Counter(list(coupling_map_distances[q].values()))[distance])
    return np.mean(num_NNs)


# increase coupling density
def increase_coupling_density(base_map, density):
    
    if density == 0:
        return base_map
    
    number_of_nodes = len(set(sum(base_map, [])))
    ## Note: IBM-Q requires a list of directed edges as a coupling map
    max_edges = (number_of_nodes)*(number_of_nodes-1)  #count edges in both directions, i.e. (1,0) and (0,1)
    coupling_map = base_map.copy()
    
    g = get_nx_graph(coupling_map)
    coupling_map_distances = dict(nx.all_pairs_shortest_path_length(g))
    distance = 2 # Start with edges connecting nodes of distance 2
    eligible_edges = get_eligible_edges(base_map, coupling_map_distances, distance) #get all qubits which have the specified distance of 2
    
    while (len(coupling_map)/max_edges) < density:
        nodes = eligible_edges[np.random.choice(np.arange(len(eligible_edges)), size=1, replace=False)[0]]
        coupling_map.append([nodes[0], nodes[1]])
        coupling_map.append([nodes[1], nodes[0]])
        eligible_edges.remove(nodes)
        # Check if no more elements of the current minimum distance exist
        if not eligible_edges:
            # If so, increase the considered distance and fetch a new list of eligible edges
            distance = distance + 1
            eligible_edges = get_eligible_edges(base_map, coupling_map_distances, distance)
        
    return coupling_map



