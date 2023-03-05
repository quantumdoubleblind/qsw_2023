# Copyright (c) 
# SPDX-License-Identifier: GPL-2.0
# coding: utf-8

import numpy as np
import io
import itertools
import networkx as nx
from qiskit.exceptions import MissingOptionalLibraryError




#####################################################################################################
### Generate Graph and QUBO for MaxCut problem
#####################################################################################################



def generate_graph_from_density(num_nodes, density):

    """
    Input: 
    num_nodes: number of nodes  
    density: coupling density = number of edges / maximum number of edges

    Output:
    Graph G
    """


    G = nx.Graph()
    nodes=np.linspace(0, num_nodes-1, num_nodes).astype(int)
    G.add_nodes_from(nodes)

    #Randomly set edges with the given density
    max_edge_list = list(itertools.combinations(nodes, 2)) # all egdes that would be possible
    max_edges = len(max_edge_list)
    num_edges = np.round(max_edges*density).astype(int)

    edges=[]
    for j in range(num_edges):
        edge=max_edge_list[np.random.choice(np.arange(len(max_edge_list)), size=1, replace=False)[0]]
        edges.append(edge)
        max_edge_list.remove(edge)

    G.add_edges_from(edges)

    return G




def get_qubo_maxcut(G):
    """
    Parameters:
    G: Graph with nodes and edges

    Output: QUBO-matrix
    """

    n_qubits=len(list(G.nodes))

    H=np.zeros((n_qubits, n_qubits))
    for pair in list(G.edges()):
        H[pair[0],pair[0]]+=-1
        H[pair[1],pair[1]]+=-1
        H[pair[0],pair[1]]+=2
    return H



