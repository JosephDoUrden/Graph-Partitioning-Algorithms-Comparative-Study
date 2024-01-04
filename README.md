# Graph Partitioning Algorithms Comparative Study

## Abstract

This research investigates and compares three prominent graph partitioning algorithms: Multi-level Graph Partitioning, Spectral Bisection, and the Louvain Algorithm. The study aims to provide a comprehensive understanding of their efficiency, scalability, and partition quality through empirical analysis.

## 1. Introduction

Graph partitioning is a complex problem with applications in network analysis, where the goal is to divide a graph's vertices into subsets while minimizing inter-partition edges. This project explores heuristic solutions to the NP-hard Graph Partitioning Problem (GPP) using three algorithms.

## 2. Problem Definition

The Graph Partitioning Problem (GPP) involves minimizing the number of edges crossing between subsets when dividing the vertices of an undirected graph into a specified number of subsets. GPP is NP-hard, necessitating heuristic approaches for efficient solutions.

## 3. Algorithms

### 3.1 Multi-level Graph Partitioning Algorithm

#### 3.1.1 Description

The Multi-level Graph Partitioning Algorithm employs coarsening, solving, and refining processes. It operates hierarchically, enhancing efficiency on large graphs.

#### 3.1.2 Computational Complexity

The time complexity is O(|N| * log(|N|)), where |N| is the number of nodes, achieved through meticulous coarsening.

### 3.2 Spectral Bisection Algorithm

#### 3.2.1 Description

Spectral Bisection utilizes the Fiedler vector, derived from the Laplacian matrix's second-smallest eigenvalue. The algorithm minimizes inter-partition edges based on the Fiedler vector's values.

#### 3.2.2 Computational Complexity

Calculating the Laplacian matrix is O(|E| + |V|), and eigenvalue/eigenvector computations are O(|V|^3) for dense graphs and O(|E| * log(|V|)) for sparse graphs.

### 3.3 Louvain Algorithm

#### 3.3.1 Description

The Louvain Algorithm optimizes the modularity score of a partition through a greedy bottom-up phase and a top-down phase.

#### 3.3.2 Computational Complexity

Known for scalability, the time complexity is typically linear or near-linear in the number of nodes and edges.

## 4. Implementation

The algorithms are implemented in Python using the NetworkX library for graph manipulation and NumPy for numerical operations. Synthetic graphs are generated through the Erdos-Renyi model for experimental analysis.

## 5. Results

### 5.1 Multi-level Graph Partitioning

- **Partition Quality:** 33.33% Excess
- **Execution Time:** 0.000956 seconds

### 5.2 Spectral Bisection

- **Partition Quality:** 66.67% Excess
- **Execution Time:** 0.005841 seconds

### 5.3 Louvain Algorithm

- **Partition Quality:** 300.00% Excess
- **Execution Time:** 0.076925 seconds

## 6. Comparative Analysis

A detailed comparative analysis provides insights into each algorithm's strengths and weaknesses. Multi-level Graph Partitioning excels with hierarchical graphs, Spectral Bisection is effective for graphs with a distinct spectral gap, while the Louvain Algorithm demonstrates scalability on extensive networks.

## 7. Conclusion

This exploration contributes nuanced insights into graph partitioning algorithms, supported by empirical evidence. The findings facilitate a deeper understanding of their capabilities, guiding potential optimization efforts for practical applicability.

## 8. Future Work

Future work may involve algorithm refinement, optimization strategies exploration, and application to real-world datasets for comprehensive validation in diverse network scenarios.

## 9. References

1. B. Goodarzi, F. Khorasani, V. Sarkar and D. Goswami, "High Performance Multilevel Graph Partitioning on GPU," 2019 International Conference on High Performance Computing & Simulation (HPCS), Dublin, Ireland, 2019, pp. 769-778, doi: 10.1109/HPCS48598.2019.9188120.
2. W. Zhou and J. Huang, "A Fast Multi-level Algorithm for Drawing Large Undirected Graphs," 2008 International Conference on Internet Computing in Science and Engineering, Harbin, China, 2008, pp. 110-117, doi: 10.1109/ICICSE.2008.27.
