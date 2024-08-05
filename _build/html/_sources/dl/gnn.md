---
title : "GNN"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: false
images: []
weight: 15
---
This page is heavily conditioned on the course "[ESE 5140 Graph Neural Networks](https://gnn.seas.upenn.edu/)" by Prof. Alejandro Ribeiro at UPenn.

# Graph
Def: a triplet $ \mathcal{G}=(\mathcal{V},\mathcal{E},\mathcal{W}) $
- Vertices: set of $ n$ labels - $\mathcal{V}=\{1,\cdots,n\} $
- Edges: ordered pairs of label $ (i,j)\in\mathcal{E}$ - $j\rightarrow i $
- Weights: $ w_{ij}\in\mathbb{R}$: numbers associated to edge $(i,j)$ - influence strength of $j$ on $i $

Properties:
- Symmetric/Undirected: $ (i,j)\in\mathcal{E}, (j,i)\in\mathcal{E}, w_{ij}=w_{ji} $
- Unweighted: $ w_{ij}=1\ \ \forall(i,j)\in\mathcal{E} $

## Graph Shift Operator
Def: a stand-in for any matrix representation of graph

Key property: Symmetric: $ S=S^T $

Notions:
- **Neighborhood**: set of nodes that influence $ i$: $n(i)=\\{j:(i,j)\in\mathcal{E}\\} $
- **Degree**: sum of weights of its incident edges: $ d_i=\sum_{j\in n(i)}w_{ij} $
- **Degree Matrix**: diagonal matrix $ D=\text{diag}(AI)$ with $D_{ii}=d_i $

Types:
- **Adjacency Matrices**: sparse matrix $ A $ with nonzero entries
$$
A_{ij}=w_{ij}\ \ \forall(i,j)\in\mathcal{E}
$$
    - If symmetric: $ A=A^T $
    - If unweighted: $ A_{ij}=1\ \ \forall(i,j)\in\mathcal{E} $
- **Normalized Adjacency Matrix**: weights relative to nodes' degrees
$$
\bar{A}=D^{-\frac{1}{2}}AD^{-\frac{1}{2}}
$$
    - entries: $ \bar{A}\_{ij}=\frac{w_{ij}}{\sqrt{d_{i}d_{j}}} $
- **Laplacian Matrix**: weights relative to nodes' degrees
$$
L=D-A
$$
    - Non-diagonal entries: $ L_{ij}=-w_{ij} $
    - Diagonal entries: $ L_{ij}=d_i $
- **Normalized Laplacian Matrix**:
$$
\bar{L}=D^{-\frac{1}{2}}LD^{-\frac{1}{2}}=I-\bar{A}
$$

## Graph Signal
**Graph Signal**: a vector $ \textbf{x}\in\mathbb{R}^n$ in which component $x_i$ is associated with note $i $
- If the graph is intrinsic to the signal, we write $ (S,\textbf{x}) $
- The graph is an expectation of proximity/similarity between signal components

**Graph Signal Diffusion**: diffused signal $ \textbf{y}=S\textbf{x}$. where $y_i=\sum_{j\in n(i)}w_{ij}x_j $
- Stronger weights contribute more to diffusion output
- Codifies a local operation where components are mixed with components of neighboring nodes

**Diffusion Sequence**:
- Recursive ver: $ \textbf{x}^{(k+1)}=S\textbf{x}^{(k)} $ (best for implementation)
- Power ver: $ \textbf{x}^{(k)}=S^k\textbf{x} $ (best for analysis)
- $ k$th element of diffusion sequence $x^{(k)}$ diffuses info to $k $-hop neighborhoods
- used for graph convolution

## Graph Convolutional Filter
**Graph Filter**: a polynomial for linear processing of graph signals
$$
H(S)=\sum_{k=0}^{\infty}h_kS^k
$$
- $ S $: graph shift operator
- $ \textbf{h}=\\{h_k\\}\_{k=0}^\infty $: filter coefficients 

**Graph Convolution**: apply filter $ H(S)$ to signal $\textbf{x} $
$$
\textbf{y}=h_{\star S}\textbf{x}=H(S)\textbf{x}=\sum_{k=0}^{\infty}h_kS^k\textbf{x}
$$
- Convolution = shift + scale + sum
- Graph convolution = weighted linear combination of diffusion sequence (i.e., shift register)
- Properties:
    - Globalization: aggregate info from local to global neighborhoods
    - Transferability: the same filter $ \textbf{h} $ can be executed in multiple graphs

**Time Convolution**: linear combination of time-shifted inputs
$$\begin{align*}
&y_n=\sum_{k=0}^{K-1}h_kx_{n-k}\\\\
&\textbf{y}=\sum_{k=0}^{K-1}h_kS^k\textbf{x}
\end{align*}$$
- Time signals are representable as graph signals on a line graph $ (S,\textbf{x}) $.
    - nodes = data points
    - edges = temporal relationships between adjacent data points
- Time shift can be interpreted as multiplications of adjacency matrix $ S $.

## Graph Fourier Transform
Notions:
- Eigenvectors & Eigenvalues: $ Sv_i=\lambda_iv_i $
- Eigenvector matrix: $ V=[\textbf{v}_1,\cdots,\textbf{v}_n] $
    - $ V^TV=I $
- Eigenvalue matrix: $ \Lambda=\text{diag}([\lambda_1;\cdots;\lambda_n]) $
    - Ordered real eigenvalues: $ \lambda_0\leq\lambda_1\leq\cdots\leq\lambda_n $
- Eigenvector decomposition: $ S=V\Lambda V^T $

**Graph Fourier Transform**: Given $ S$, the projection of $S$ of graph signal $\textbf{x} $ on the eigenspace is:
$$
\tilde{\textbf{x}}=V^T\textbf{x}
$$
- $ \tilde{\textbf{x}}$: graph frequency representation of $\textbf{x} $

**Inverse Graph Fourier Transform**: Given $ S$, the recovery of original signal $\textbf{x} $ from the eigenspace is:
$$
\tilde{\tilde{\textbf{x}}}=V\tilde{\textbf{x}}=VV^T\textbf{x}=I\textbf{x}=\textbf{x}
$$

**Graph Frequency Represention of Graph Filters**: consider graph filter $ \textbf{h}$, graph signal $\textbf{x}$, and filtered signal $\textbf{y}$, the GFTs $\tilde{\textbf{x}}=V^T\textbf{x}$ and $\tilde{\textbf{y}}=V^T\textbf{y} $ are related by
$$
\tilde{\textbf{y}}=\sum_{k=0}^{\infty}h_k\Lambda^k\tilde{\textbf{x}}
$$
- Graph convolutions are pointwise in GFT domain: $ \tilde{y}_i=\tilde{h}(\lambda_i)\tilde{x}_i $
- **Frequency Response of a Graph Filter**:
$$
\tilde{h}(\lambda)=\sum_{k=0}^{\infty}h_k\lambda^k
$$
    - = the same polynomial that defines the graph filter, but on scalar variable $ \lambda $
    - Independent of the graph
        - Role of the graph: to determine the eigenvalues on which the response is instantiated
        - Eigenvectors determine the meaning of the frequencies

# Graph Neural Network

## Learning with Graph Signals