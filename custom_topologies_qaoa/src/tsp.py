import itertools
import numpy as np
import networkx as nx


class TSP:
    # b = border penalty
    # a = constraint penalty
    # A * sum_(v=1)_n(1- sum_(j=1)_N x_v,j)^2 + A * sum_(j=1)_n(1- sum_(v=1)_N x_v,j)^2 + B * sum_uv isin E W_uv
    # sum_(j=1)_N x_u,jx_v,j +1
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def generate_qubo(self, distance_matrix, G):
        """
        Parameters:
        G: Graph with nodes and edges
        distance_matrix: encodes the weights of each edge. Can be symmetric or asymmetric for an undirected or directed graph. 

        Penalty terms:
        a: Penalty for one edge being travelled twice. Each vertex v can only be in one position j in the list of travelled vertices.
        b: Prefactor for weight-term: sum weights of all edges along the travelled path

        QUBO formulation taken from Lucas (arXiv:1302.5843v3)

        bitstrings: x_vj = 1 if node v appears at position j in the path. j,v \in [1, N]
        [x_00, x_01, ..., x_0N, x_10, x_11, ..., x_NN]
        List all possible positions for each node within a block.

        Important for weight-terms: Having nodes u --> v or v --> u is encoded by different index pairs (i.e., x_00 and x_11 vs. x_10 and x_01) and thus always has to be differentiated.
        Part of the solution is to find out if node u or node v comes first.
        The difference between directed and undirected graphs only materializes in the values of the distance matrix at these index pairs which is same or different.
        """

        N = len(list(G.nodes))
        n_qubits = N ** 2

        pairs = list(itertools.combinations(range(N), 2))

        H = np.zeros((n_qubits, n_qubits))
        offset = 2 * self.a * N
        # diagonal entries: Originate from both sums (giving a factor of 2)
        np.fill_diagonal(H, -2 * self.a)

        # mixed terms within a block
        for v in range(N):
            for j in range(len(pairs)):
                ind1 = N * v + pairs[j][0]
                ind2 = N * v + pairs[j][1]
                H[ind1, ind2] += 2 * self.a

                # mixed terms between blocks
        for j in range(N):
            for v in range(len(pairs)):
                ind1 = j + N * pairs[v][0]
                ind2 = j + N * pairs[v][1]
                H[ind1, ind2] += 2 * self.a

                # loop through all pairs and differentiate if they are part of the edge set or not
        for pair in pairs:
            for j in range(N):
                ind1 = N * pair[0] + j
                ind2 = N * pair[1] + j + 1
                if j == N - 1:
                    ind2 = N * pair[1] + 0  # last index corresponds to first one since the graph is closed

                if pair in list(G.edges()):
                    H[ind1, ind2] += self.b * distance_matrix[pair[0], pair[1]]
                if pair not in list(G.edges()):
                    H[ind1, ind2] += self.a

                ind1 = N * pair[0] + j + 1
                ind2 = N * pair[1] + j
                if j == N - 1:
                    ind1 = N * pair[0] + 0  # last index corresponds to first one since the graph is closed

                if pair in list(G.edges()):
                    H[ind1, ind2] += self.b * distance_matrix[pair[1], pair[0]]
                if pair not in list(G.edges()):
                    H[ind1, ind2] += self.a

        return H

    def generate_graph_from_density(self, num_nodes, density):

        """
        Input: 
        num_nodes: number of nodes  
        density: coupling density = number of edges / maximum number of edges

        Output:
        Graph G
        """
        G = nx.Graph()
        nodes = np.linspace(0, num_nodes - 1, num_nodes).astype(int)
        G.add_nodes_from(nodes)

        # Randomly set edges with the given density
        max_edge_list = list(itertools.combinations(nodes, 2))  # all egdes that would be possible
        max_edges = len(max_edge_list)
        num_edges = np.round(max_edges * density).astype(int)

        edges = []
        for j in range(num_edges):
            edge = max_edge_list[np.random.choice(np.arange(len(max_edge_list)), size=1, replace=False)[0]]
            edges.append(edge)
            max_edge_list.remove(edge)
        G.add_edges_from(edges)

        return G

    def generate_problem_symm(self, n, graph_density):
        dist_matrix = np.zeros((n, n))
        for pair in list(itertools.combinations(range(n), 2)):
            num = np.random.random_sample()
            dist_matrix[pair[0], pair[1]] = num

        # generate problem graph with given graph density:
        G = self.generate_graph_from_density(n, graph_density)
        return dist_matrix, G

    def generate_problem(self, n, graph_density):
        dist_matrix = np.random.random(size=(n, n))
        np.fill_diagonal(dist_matrix, 0)

        # generate problem graph with given graph density:
        G = self.generate_graph_from_density(n, graph_density)
        return dist_matrix, G


