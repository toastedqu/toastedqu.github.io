---
title : "Unsupervised Learning"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: false
images: []
weight: 400
---
Common Notations:
- $m$: #samples in the input batch
- $n$: #features in the input sample
- $i$: $i$th sample
- $j$: $j$th feature
- $k$: class $k$
- $X=[\mathbf{x}_1,\cdots,\mathbf{x}_m]^T$: input matrix of shape $(m,n)$ (add $\textbf{1}$ if bias is needed)
- $\mathbf{y}=[y_1,\cdots,y_{m}]^T$: output vector of shape $(m,1)$
- $\textbf{w}=[w_1,\cdots,w_n]$ (add $b$ if bias is needed)
- $X,Y$: (occasionally) used to represent input feature and label as random variables

Each subsection roughly follows this order:
- Idea
- Usage
- Background
- Assumption
- Model/Algorithm
- Prediction
- Objective (mainly Loss Function)
- Optimization (mainly Parameter Estimation)
- Pros
- Cons
- (extra)

# Dimensionality Reduction

## Singular Value Decomposition
Idea: Decompose a data matrix into a rotating, scaling, and second rotating matrix in a vector space

Usage: Null space determination, Pseudoinverse calculation, Least squares model fitting, Eigenwords, Recommender Systems
- **Pseudo-inverse**: find the "inverse" of a rectangular matrix.
    - e.g., LinReg: $X^+=(X^TX)^{-1}X^T$
- **Power Method**: find the largest eigenvalue/eigenvector by continuously multiplying this matrix on an arbitrary vector.
- **Eigenword**: project high-dimensional context to low-dimensional space, assuming **distributional similarity** (i.e., words with similar contexts have similar meanings).
    - Similar words are close in this low-dimensional space.
    - Left singular vectors = Eigenwords (i.e., word embeddings)
    - Right singular vectors $\times$ Context = Eigentokens (i.e., contextual embeddings)
    - Word sense disambiguation: estimate contextual embedding (state vector) for a word using right singular vectors.
    - Word2vec is NOT contextualized.

Background: 
- **Positive Semi-Definite** (PSD): A symmetric matrix $X$ is PSD if $\forall\textbf{z}\in\mathbb{R}^n,\textbf{z}\geq\textbf{0}: \textbf{z}^TX\textbf{z}\geq 0$.
- Size of Matrix: $||X||_F=\sqrt{\sum\_{i=1}^m\sum\_{j=1}^n|x\_{ij}|^2}=\sqrt{\sum\_{i=1}^{\min(m,n)}\sigma_i^2(X)}$, where $\sigma_i(X)$ is the $i$th eigenvalue.
- Properties: 
    - $X^TX=V(D^TD)V^T=\sum_{i=1}^n(D_{ii})^2\textbf{v}_i\textbf{v}_i^T$
    - $XX^T=U(DD^T)U^T=\sum_{i=1}^m(D_{ii})^2\textbf{v}_i\textbf{v}_i^T$
    - $X^+=VD^{-1}U^T$

Model/Algorithm:
$$
X=UDV^T=\sum_{i=1}^{\min(m,n)}D_{ii}\textbf{u}_i\textbf{v}_i^T
$$
- $U$: left singular vectors (**orthonormal**: $U^TU=I$), shape $m\times m$ 
- $D$: singular values (**diagonal**), shape $m\times n$
- $V$: right singular vectors (**orthonormal**: $V^TV=I$), shape $n\times n$
- $\textbf{u}_i\textbf{v}_i^T$: outer product of $i$th (column) unit vectors
- $D_{ii}$: importance/strength of $i$th outer product matrix

(**Thin SVD**): use the best rank $k$ approximation to represent the matrix: $X\approx U_kD_kV_k^T$

Pros:
- Wide range of applications
- Data compression

Cons:
- Scale variant
- High computational cost

## Principal Component Analysis
Idea:
- Project the original data onto an orthonormal basis $\\{\textbf{u}_1,\cdots,\textbf{u}_k\\}$ of smaller dimensions, while covering maximal variance among features.

Background:
- Properties of orthonormal basis: $\textbf{u}_i^T\textbf{u}_j=0,\textbf{u}_i^T\textbf{u}_i=1$.
- Any sample vector can be expressed in terms of coefficients (scores) on eigenvectors (loadings): $\textbf{x}_i=\sum\_{j=1}^nz\_{ij}\textbf{u}_j$.
- The projected point (coefficients) of the original sample onto the new basis can be calculated inversely: $z\_{ij}=\textbf{x}_i^T\textbf{u}_j$.
- $D_{ii}=\sqrt{\Lambda_{ii}}$

Assumptions:
- All PCs are linear combinations of original features. (otherwise Kernel PCA)
- Variance is a measure of how important a feature is.

Model/Algorithm:
1. Center/Demean the data into $X_c$ where each row is:
$$
\textbf{x}_i\leftarrow\textbf{x}_i-\frac{1}{m}\sum\_{i=1}^m\textbf{x}_i
$$
2. Calculate the covariance matrix in the observation space:
$$
\Sigma=\text{Cov}(X_c,X_c)=X_c^TX_c=\sum\_{i=1}^m(\textbf{x}_i-\bar{\textbf{x}})(\textbf{x}_i-\bar{\textbf{x}})^T
$$
3. Diagonalize the covariance matrix to find its corresponding eigenvalues (PC strength) and orthonormal eigenvectors (PCs) via Spectral Theorem:
$$
\Sigma=Q\Lambda Q^T
$$
4. Sort eigenvectors in $Q$ based on eigenvalues $\Lambda$ in descending order. Select and normalize the strongest $k$ eigenvectors.
5. A projected point would look like:
$$
\textbf{z}_i=((\textbf{x}_i-\bar{\textbf{x}})^T\textbf{u}_1,\cdots,(\textbf{x}_i-\bar{\textbf{x}})^T\textbf{u}_k)
$$

(PCA via SVD):
1. Center/Demean the data into $X_c$.
2. Compute SVD: $X_c=UDV^T$.
3. Select $k$ rows of $V^T$ (the right singular matrix) with largest singular values as principal loadings.
4. A projected point would look like: $\textbf{z}_i=((\textbf{x}_i-\bar{\textbf{x}})^T\textbf{v}_1,\cdots,(\textbf{x}_i-\bar{\textbf{x}})^T\textbf{v}_k)$

Prediction/Reconstruction:
1. Reconstruct the original point via inverse mapping:
$$
\hat{\textbf{x}}_i=\bar{\textbf{x}}+\sum\_{j=1}^kz\_{ij}\textbf{u}_j
$$
*The original point can be fully reconstructed if $k=n$:
$$
\textbf{x}_i=\bar{\textbf{x}}+\sum\_{j=1}^nz\_{ij}\textbf{u}_j
$$

Objective: minimize distortion (i.e., maximize variance in new coordinates)
- **Distortion**:
$$\begin{align*}
\text{Distortion}_k=||X-ZU^T||_F&=\sum\_{i=1}^m||\textbf{x}_i-\hat{\textbf{x}}_i||_2^2 \\\\
&=\sum\_{i=1}^m\sum\_{j=k+1}^{n}z\_{ij}^2\\\\
&=m\sum\_{j=k+1}^{n}\textbf{u}_j^T\Sigma\textbf{u}_j\\\\
&=m\sum\_{j=k+1}^{n}\lambda_j
\end{align*}$$
- **Variance**: (the variance of projected points)
$$
\text{Variance}_k=m\sum\_{j=1}^{k}\textbf{u}_j^T\Sigma\textbf{u}_j=m\sum\_{j=1}^{k}\lambda_j
$$
- Minimizing distortion = Maximizing variance:
$$
\text{Variance}_k+\text{Distortion}_k=m\sum\_{j=1}^{n}\lambda_j=m\cdot\text{trace}(\Sigma)
$$

Optimization: Using eigenvectors/eigenvalues as PCs/PC scores guarantee minimization.

Pros:
- Guarantee removal of correlated features (All PCs are orthogonal to each other)
- Reduce overfitting for other models
- Improve visualization for high-dimensional data
- Robust to outliers and noisy data

Cons:
- Scale invariant (must demean)
- Less interpretability of new features/PCs
- Require standardization
- Potential information loss if PCs/$\\#$PCs are not selected carefully
- (weak) If any of the assumptions fail, then PCA fails.
    - This can be solved with Kernel PCA, which can effectively learn nonlinear dimensionality reductions.
- Very situational (e.g. cannot be applied on NLP because 1) covariance matrix is useless 2) it breaks sparse structure of words)

## Independent Component Analysis
tbd

## Autoencoder
Idea: Use unsupervised NNs to learn latent representations via reconstruction. (Semi-supervised learning)

Model: NN
- **Denoising AE**: add noise to the input and try to output the original image (to avoid perfect fitting)

Objective: minimize reconstruction error

Optimization: see DL

Pros:
- = Nonlinear PCA (PCA = Linear Manifold)
- Offer embeddings that can be used in supervised learning
- Better performance than PCA in general

Cons:
- High computational cost


## Variational Autoencoder
tbd

# Clustering

## K-Means

Idea: Hard clustering (clustering with deterministic results)

Model/Algorithm:
1. Init centroids $\boldsymbol{\mu}_1, \cdots, \boldsymbol{\mu}_K\in\mathbb{R}^n$.
2. Find cluster for each data point:
$$
c_i=\arg\min_k||\mathbf{x}_i-\boldsymbol{\mu}_k||^2\quad(r\_{ik}=\textbf{1}\\{c_i=k\\})
$$
3. Update centroids as the mean of all data points in the current cluster:
$$
\boldsymbol{\mu}_k=\frac{\sum\_{i=1}^{m}r\_{ik}\mathbf{x}_i}{\sum\_{i=1}^{m}r\_{ik}}
$$
4. Repeat Step 2-3 until convergence (i.e., $\boldsymbol{\mu}_j$ remain unchanged).

Objective: reconstruction error: $\mathcal{L}=\sum_{i=1}^m\sum_{k=1}^Kr\_{ik}||\textbf{x}_i-\boldsymbol{\mu}_k||_2^2$

Pros:
- Simple. Interpretable
- Guarantee convergence
- Flexible re-training
- Generalize to any type/shape/size of clusters
- Suitable for large datasets
- Time complexity: $O(kmn)$

Cons:
- Scale variant
- Numerical features only
- Manual hyperparameter choice: $k$
- Inconsistent: Sensitive to centroid initialization
- Sensitive to outliers and noisy data by including them
- Bad performance on high-dimensional data (distance metric works poorly)
- Hard clustering (assume 100% in the designated cluster)

Code:
```python
def distance(v1,v2,metric_type='L2'):
    if metric_type == "L0":
        return np.count_nonzero(v1-v2)
    if metric_type == "L1":
        return np.sum(np.abs(v1-v2))
    if metric_type == "L2":
        return np.sqrt(np.sum(np.square(v1-v2)))
    if metric_type == "Linf":
        return np.max(np.abs(v1-v2))

class KMeans:
    def __init__(self, n_clusters=8, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        
    def fit(self, X_train):
        # 1) Init centroids uniformly
        x_min, x_max = np.min(X_train, axis=0), np.max(X_train, axis=0)
        self.centroids = [np.random.uniform(x_min, x_max) for _ in range(self.n_clusters)]
        
        # 4) Repeat Step 2-3
        i = 0
        centroids_cache = self.centroids
        while i < self.max_iter and np.not_equal(self.centroids, centroids_cache).any():
            # 2) Find cluster for each data point
            clustered_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                distances = [distance(x, centroid) for centroid in self.centroids]
                clustered_points[np.argmin(distances)].append(x)
            
            # 3) Update centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in clustered_points]
            i += 1
```

## Gaussian Mixture Model
Idea: clustering but deterministic $\textbf{1}\\{c_i=k\\}$ $\rightarrow$ stochastic $\mathbb{E}[\textbf{1}\\{z_i=k\\}]=P(z_i=k|\textbf{x}_i)$ (denoted as $r\_{ik}$)

Assumptions:
- There exists a latent variable $z\in\\{1,\cdots,K\\}$ representing the index of the Gaussian distribution in the mixture.

Model: Generative: $P(\mathbf{x}_i,z_i)=P(\mathbf{x}_i|z_i)P(z_i)$, where
- $z_i\sim\text{Multinomial}(\boldsymbol{\phi})$, $\boldsymbol{\phi}\in\mathbb{R}^{K}$
- $(\mathbf{x}_i|z_i=k)\sim N(\boldsymbol{\mu}_k,\Sigma_k)$, $\boldsymbol{\mu}_k\in\mathbb{R}^n,\Sigma_k\in\mathbb{R}^{n\times n}$

Optimization: EM
1. Init distributions for prior $P(z_i)$ and likelihood $P(\mathbf{x}_i|z_i)$
2. E-step: estimate $P(z_i=k|\textbf{x}_i)$ given params $(\boldsymbol{\phi},\boldsymbol{\mu}_k,\Sigma_k)$:
$$
r\_{ik}=P(z_i=k|\textbf{x}_i)=\frac{P(\textbf{x}_i|z_i=k)P(z_i=k)}{\sum\_{k=1}^{K}P(\textbf{x}_i|z_i=k)P(z_i=k)}
$$
3. M-step: estimate params via MLE given $P(z_i=j|\textbf{x}_i)$:
$$\begin{align*}
\phi_k&=\frac{1}{m}\sum\_{i=1}^mr\_{ik}\\\\
\boldsymbol{\mu}_k&=\frac{\sum\_{i=1}^mr\_{ik}\textbf{x}_i}{\sum\_{i=1}^mr\_{ik}}
\end{align*}$$
4. Repeat Step 2-3 until convergence.

Pros:
- More robust to outliers and noisy data
- Flexible to a great variety of shapes of data points
- Soft clustering
- Weighted distance instead of absolute distance in k-means

Cons:
- High computational cost