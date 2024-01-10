# Graph-attention-network-GAT-for-node-classification
Implemented a GNN and GAT model for node prediction on the Cora dataset (consisting of 2,708 scientific papers classified into 7 classes, with a citation network comprising 5,429 links and binary word vectors of size 1,433). 

Build the model
GAT takes as input a graph (namely an edge tensor and a node feature tensor) and outputs [updated] node states. The node states are, for each target node, neighborhood aggregated information of N-hops (where N is decided by the number of layers of the GAT). Importantly, in contrast to the graph convolutional network (GCN) the GAT makes use of attention machanisms to aggregate information from neighboring nodes (or source nodes). In other words, instead of simply averaging/summing node states from source nodes (source papers) to the target node (target papers), GAT first applies normalized attention scores to each source node state and then sums.

(Multi-head) graph attention layer
The GAT model implements multi-head graph attention layers. The MultiHeadGraphAttention layer is simply a concatenation (or averaging) of multiple graph attention layers (GraphAttention), each with separate learnable weights W. The GraphAttention layer does the following:

Consider inputs node states h^{l} which are linearly transformed by W^{l}, resulting in z^{l}.

For each target node:

Computes pair-wise attention scores a^{l}^{T}(z^{l}_{i}||z^{l}_{j}) for all j, resulting in e_{ij} (for all j). || denotes a concatenation, _{i} corresponds to the target node, and _{j} corresponds to a given 1-hop neighbor/source node.
Normalizes e_{ij} via softmax, so as the sum of incoming edges' attention scores to the target node (sum_{k}{e_{norm}_{ik}}) will add up to 1.
Applies attention scores e_{norm}_{ij} to z_{j} and adds it to the new target node state h^{l+1}_{i}, for all j.
