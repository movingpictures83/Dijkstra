# GPUDijkstra
# Language: CUDA
# Input: TXT
# Output: TXT
# Tested with: PluMA 1.0, CUDA 10

Dijkstra's Algorithm on the GPU.

Original authors: Lazaro Fernandez, Damian Niebler, Jessica Silva

The plugin accepts as input a TXT file of keyword-value pairs:
matrix: TSV (tab-separated) values for the network, as an adjacency matrix
N: Number of nodes in the network

The program will output a TXT file containing each node, and the length of the shortest path from the source.
