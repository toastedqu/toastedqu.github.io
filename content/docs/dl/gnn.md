---
title : "Graph Neural Networks"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: false
images: []
weight: 4
---
# Graph
Def: a triplet $\mathcal{G}=(\mathcal{V},\mathcal{E},\mathcal{W})$
- Vertices: set of $n$ labels - $\mathcal{V}=\{1,\cdots,n\}$
- Edges: ordered pairs of label $(i,j)\in\mathcal{E}$ - $j\rightarrow i$
- Weights: $w_{ij}\in\mathbb{R}$: numbers associated to edge $(i,j)$ - influence strength of $j$ on $i$

Properties:
- Symmetric/Undirected: $(i,j)\in\mathcal{E}, (j,i)\in\mathcal{E}, w_{ij}=w_{ji}$
- Unweighted: $w_{ij}=1\ \ \forall(i,j)\in\mathcal{E}$

## Representations
**Adjacency Matrices**: sparse matrix $A$ with nonzero entries - $A_{ij}=w_{ij}\ \ \forall(i,j)\in\mathcal{E}$
- If symmetric: $A=A^T$
- If unweighted: $A_{ij}=1\ \ \forall(i,j)\in\mathcal{E}$

**Neighborhood**: set of nodes that influence $i$ - $n(i)=\{j:(i,j)\in\mathcal{E}\}$

**Degree**: sum of weights of its incident edges - $d_i=\sum_{j\in n(i)}w_{ij}$

**Degree Matrix**: diagonal matrix $D=\text{diag}(AI)$ with $D_{ii}=d_i$

**Laplacian Matrix**: weights relative to nodes' degrees $L=D-A$
- Non-diagonal entries: $L_{ij}=-w_{ij}$
- Diagonal entries: $L_{ij}=d_i$

**Normalized Adjacency Matrix**: weights relative to nodes' degrees $\bar{A}=D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$ with $\bar{A}\_{ij}=\frac{w_{ij}}{\sqrt{d_{i}d_{j}}}$

**Normalized Laplacian Matrix**: $\bar{L}=D^{-\frac{1}{2}}LD^{-frac{1}{2}}=I-\bar{A}$

**Graph Shift Operator**: a stand-in for any matrix representation of graph
- All representations:
    - Adjacency matrix
    - Laplacian matrix
    - Normalized adjacency
    - Normalized Laplacian
- Symmetric: $S=S^T$

**Graph Signal**: a vector $\textbf{x}\in\mathbb{R}^n$ in which component $x_i$ is associated with note $i$
- If the graph is intrinsic to the signal, we write $(S,\textbf{x})$
- The graph is an expectation of proximity/similarity between signal components

**Graph Signal Diffusion**: diffused signal $\textbf{y}=S\textbf{x}$. where $y_i=\sum_{j\in n(i)}w_{ij}x_j$
- Stronger weights contribute more to diffusion output
- Codifies a local operation where components are mixed with components of neighboring nodes

**Diffusion Sequence**
- Recursive ver: $\textbf{x}^{(k+1)}=S\textbf{x}^{(k)}$ (best for implementation)
- Power ver: $\textbf{x}^{(k)}=S^k\textbf{x}$ (best for analysis)
- $k$th element of diffusion sequence $x^{(k)}$ diffuses info to $k$-hop neighborhoods
- used for graph convolution

## Graph Convolutional Filter
**Graph Filter**: a polynomial on $S$: $H(S)=\sum_{k=0}^{\infty}h_kS^k$

**Graph Convolution**: $\textbf{y}=H(S)\textbf{x}=\sum_{k=0}^{\infty}h_kS^k\textbf{x}$