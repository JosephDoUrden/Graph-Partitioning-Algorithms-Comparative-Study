import networkx as nx
import numpy as np
import time
import random

# Multi-level Graph Partitioning Algorithm
def multi_level_graph_partitioning(G, k, levels):
    if levels == 0:
        return partition_graph(G, k)

    G_coarse = coarsen_graph(G)
    partition_coarse = multi_level_graph_partitioning(G_coarse, k, levels - 1)
    partition_fine = refine_partition(G, G_coarse, partition_coarse)
    
    return partition_fine

def coarsen_graph(G):
    # Edge contraction coarsening strategy
    contracted_nodes = random.sample(list(G.nodes), len(G) // 2)

    # Create a new graph for coarsening
    G_coarse = nx.Graph()

    # Add nodes corresponding to contracted nodes with initialized 'contraction' attribute
    for u in contracted_nodes:
        G_coarse.add_node(u, contraction=[])

    for u in contracted_nodes:
        neighbors = list(G.neighbors(u))

        # Add edges to the coarsened graph
        for v in neighbors:
            if v not in contracted_nodes:
                # Add nodes if not already present in G_coarse
                if v not in G_coarse.nodes:
                    G_coarse.add_node(v, contraction=[])

                G_coarse.add_edge(u, v)

                # Update contraction information
                G_coarse.nodes[u]['contraction'].append(v)
                G_coarse.nodes[v]['contraction'].append(u)

    return G_coarse

def partition_graph(G, k):
    nodes = list(G.nodes)
    random.shuffle(nodes)
    partition = {nodes[i]: i % k for i in range(len(nodes))}
    
    return partition

def refine_partition(G, G_coarse, partition_coarse):
    partition_fine = {}
    for node_coarse, part in partition_coarse.items():
        nodes_fine = G_coarse.nodes[node_coarse]['contraction']
        partition_fine.update({node_fine: part for node_fine in nodes_fine})
    
    return partition_fine

# Spectral Bisection Algorithm
def spectral_bisection(G):
    Laplacian_matrix = calculate_laplacian_matrix(G)
    eigenvalues, eigenvectors = calculate_eigenvalues_and_vectors(Laplacian_matrix)

    fiedler_vector = eigenvectors[:, 1]
    partition = {node: 0 if fiedler_vector[i] >= 0 else 1 for i, node in enumerate(G.nodes)}

    return partition

def calculate_laplacian_matrix(G):
    Laplacian_matrix = nx.linalg.laplacianmatrix.laplacian_matrix(G).toarray()
    return Laplacian_matrix

def calculate_eigenvalues_and_vectors(matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return eigenvalues, eigenvectors

# Louvain Algorithm
def louvain_algorithm(G):
    partition = initialize_partition(G)
    prev_partition = None
    
    while partition != prev_partition:
        prev_partition = partition
        partition = bottom_up_phase(G, partition)
        partition = top_down_phase(G, partition)
    
    return partition

def initialize_partition(G):
    return {node: i for i, node in enumerate(G.nodes)}

def bottom_up_phase(G, partition):
    for node in G.nodes:
        best_community = find_best_community(G, node, partition)
        partition[node] = best_community
    
    return partition

def top_down_phase(G, partition):
    communities = set(partition.values())
    
    for community in communities:
        subgraph = G.subgraph([node for node, comm in partition.items() if comm == community])
        sub_partition = bottom_up_phase(subgraph, {})
        
        for node, sub_community in sub_partition.items():
            partition[node] = community + sub_community
    
    return partition

def find_best_community(G, node, partition):
    current_community = partition.get(node, 0)  # Use 0 as the default community if the node is not in the partition
    neighbor_communities = set(partition.get(neighbor, 0) for neighbor in G.neighbors(node))
    
    max_gain = 0
    best_community = current_community
    
    for neighbor_community in neighbor_communities:
        gain = calculate_modularity_gain(G, partition, node, current_community, neighbor_community)
        if gain > max_gain:
            max_gain = gain
            best_community = neighbor_community
    
    return best_community

def calculate_modularity_gain(G, partition, node, current_community, neighbor_community):
    m = G.number_of_edges()
    
    # Use 0 as the default degree if the node is not in the partition
    deg_in_community = sum(G.degree(neighbor) for neighbor in G.neighbors(node) if partition.get(neighbor, 0) == neighbor_community)
    
    deg_total = G.degree(node)
    
    # Use 0 as the default degree if the node is not in the partition
    deg_current_community = sum(G.degree(neighbor) for neighbor in G.neighbors(node) if partition.get(neighbor, 0) == current_community)
    
    q = (deg_in_community / (2 * m)) - ((deg_total * deg_current_community) / (2 * m * m))
    
    return q

# Compare the algorithms on a random graph
G = nx.random_graphs.erdos_renyi_graph(100, 0.1)

# Multi-level Graph Partitioning
start_time = time.time()
partition_multi_level = multi_level_graph_partitioning(G, 3, 2)
time_multi_level = time.time() - start_time
excess_multi_level = ((len(set(partition_multi_level.values())) - 1) / nx.node_connectivity(G) - 1) * 100

# Spectral Bisection
start_time = time.time()
partition_spectral = spectral_bisection(G)
time_spectral = time.time() - start_time
excess_spectral = ((len(set(partition_spectral.values())) - 1) / nx.node_connectivity(G) - 1) * 100

# Louvain Algorithm
start_time = time.time()
partition_louvain = louvain_algorithm(G)
time_louvain = time.time() - start_time
excess_louvain = ((len(set(partition_louvain.values())) - 1) / nx.node_connectivity(G) - 1) * 100

# Print results
print("Results:")
print(f"Multi-level Graph Partitioning Excess: {abs(excess_multi_level):.2f}% | Time: {time_multi_level:.6f} seconds")
print(f"Spectral Bisection Excess: {abs(excess_spectral):.2f}% | Time: {time_spectral:.6f} seconds")
print(f"Louvain Algorithm Excess: {abs(excess_louvain):.2f}% | Time: {time_louvain:.6f} seconds")
