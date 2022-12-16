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
# Dimensionality Reduction

## Principal Component Analysis

Idea: transform data points into smaller dimensions while covering maximal variance among features.

Algorithm:
1. Standardize data:
$$
x_j^{(i)}\leftarrow\frac{x_j^{(i)}-\mu_j}{\sigma_j}
$$
2. Calculate the covariance matrix in the observation space:
$$
\Sigma=\text{Cov}(X,X)
$$
3. Diagonalize the covariance matrix to find its corresponding orthonormal eigenvectors (via Spectral Theorem):
$$
\Sigma=Q\Lambda Q^T
$$
4. Sort eigenvectors in $Q$ based on eigenvalues $\Lambda$ in descending order. Select and normalize the strongest 2 (or more) eigenvectors.
5. Transform all data points using the selected eigenvectors as the new basis.

Pros:
- Guarantee removal of correlated features (All PCs are orthogonal to each other).
- Reduce overfitting for other ML models.
- Improve visualization for high-dimensional data.

Cons:
- Less interpretability of new features (i.e., PCs)
- Require standardization.
- Potential information loss if PCs are not selected carefully.
- The assumption that PCs are linear combinations of original features. If they are NOT, then gg.
- The assumption that variance is a measure of how important a dimension is. If it is not, then gg.

## Independent Component Analysis

# Clustering

## K-Means

Idea: repeatedly update cluster centroids until convergence.

Terminology:
- Cluster: a group of data points with a common label/property
- Centroid: the center of a cluster

Algorithm:
1. Init centroids $\boldsymbol{\mu}_1, \cdots, \boldsymbol{\mu}_k\in\mathbb{R}^n$.
2. Find cluster for each data point:
$$
c^{(i)}=\arg\min_j||\mathbf{x}^{(i)}-\boldsymbol{\mu}_j||^2
$$
3. Update centroids as the mean of all data points in the current cluster:
$$
\boldsymbol{\mu}_j=\frac{1}{\sum_{i=1}^{m}1\{c^{(i)}=j\}}\sum_{i=1}^{m}1\{c^{(i)}=j\}\mathbf{x}^{(i)}
$$
4. Repeat Step 2-3 until convergence (i.e., $\boldsymbol{\mu}_j$ remain unchanged).

Pros:
- Easy to implement and interpret.
- Guarantee convergence.
- Flexible re-training.
- Generalize to any type/shape/size of clusters.
- Suitable for large datasets.
- Time complexity: $O(kmn)$.

Cons:
- Scale variant.
- Numerical features only.
- Manual hyperparameter choice: $k$.
- Inconsistent: sensitive to initialization of centroids.
- Sensitive to outliers and noisy data by including them.
- Sensitive to high-dimensional data (distance metric works poorly).
- Hard clustering (assume 100% in the designated cluster).

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

Idea: use a mixture of Gaussian distributions to represent the data distribution.

Algorithm (EM):
1. Init distributions for prior $P(z^{(i)})$ and likelihood $P(\mathbf{x}^{(i)}|z^{(i)})$
2. E-step: Init $w_j^{(i)}$ as the probability that $z^{(i)}$ points to the $j$th Gaussian distribution:
$$
w_j^{(i)}=P(z^{(i)}=j|x^{(i)};\boldsymbol{\phi},\boldsymbol{\mu}_j,\Sigma_j)=\frac{P(x^{(i)}|z^{(i)}=j;\boldsymbol{\mu}_j,\Sigma_j)P(z^{(i)}=j;\boldsymbol{\phi})}{\sum_{k=1}^{K}P(x^{(i)}|z^{(i)}=k;\boldsymbol{\mu}_k,\Sigma_k)P(z^{(i)}=k;\boldsymbol{\phi})}=\text{E}[1\{z^{(i)}=j\}]
$$
3. M-step: update params via MLE on all $w_j^{(i)}$:
$$\begin{align*}
\boldsymbol{\phi}_j&=\frac{1}{m}\sum_{i=1}^{m}w_j^{(i)}\\\\
\boldsymbol{\mu}_j&=\frac{1}{\sum_{i=1}^{m}w_j^{(i)}}\sum_{i=1}^{m}w_j^{(i)}x^{(i)}
\end{align*}$$
4. Repeat Step 2-3 until convergence.
    

Assumptions:
- There exists a latent variable $z\in\{1,\cdots,K\}$ representing the index of the Gaussian distribution in the mixture.
- $P(\mathbf{x}^{(i)},z^{(i)})=P(\mathbf{x}^{(i)}|z^{(i)})P(z^{(i)})$, where
    - $z^{(i)}\sim\text{Multinomial}(\boldsymbol{\phi})$, $\boldsymbol{\phi}\in\mathbb{R}^{K}$
    - $\mathbf{x}^{(i)}|z^{(i)}=j\sim N(\boldsymbol{\mu}_j,\Sigma_j)$, $\boldsymbol{\mu}_j\in\mathbb{R}^n$

Pros:
- More robust to outliers and noisy data
- Flexible to a great variety of shapes of data points
- Allow for soft clustering
- Weighted distance instead of pure distance in k-means

Cons:
- High computational cost