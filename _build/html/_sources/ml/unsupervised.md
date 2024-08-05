---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Unsupervised Learning

## Clustering
Clustering groups similar samples together with no prior knowledge.

### K-Means
Idea: Hard clustering (clustering with deterministic results)

Model/Algorithm:
1. Init centroids $ \boldsymbol{\mu}_1, \cdots, \boldsymbol{\mu}_K\in\mathbb{R}^n $.
2. Find cluster for each data point:
$$
c_i=\arg\min_k||\mathbf{x}_i-\boldsymbol{\mu}_k||^2\quad(r\_{ik}=\textbf{1}\\{c_i=k\\})
$$
3. Update centroids as the mean of all data points in the current cluster:
$$
\boldsymbol{\mu}_k=\frac{\sum\_{i=1}^{m}r\_{ik}\mathbf{x}_i}{\sum\_{i=1}^{m}r\_{ik}}
$$
4. Repeat Step 2-3 until convergence (i.e., $ \boldsymbol{\mu}_j $ remain unchanged).

Objective: reconstruction error: $ \mathcal{L}=\sum_{i=1}^m\sum_{k=1}^Kr\_{ik}||\textbf{x}_i-\boldsymbol{\mu}_k||_2^2 $

Optimization:
- Objective minimization: Greedy (because this objective is NP-hard to optimize)
- Hyperparameter tuning: choose the elbow point in the "reconstruction error vs $ \\## $clusters" graph.

Pros:
- Simple. Interpretable
- Guarantee convergence in a finite number of iterations
- Flexible re-training
- Generalize to any type/shape/size of clusters
- Suitable for large datasets
- Time complexity: $ O(kmn) $

Cons:
- Scale variant
- Numerical features only
- Manual hyperparameter choice: $ k $
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
        ## 1) Init centroids uniformly
        x_min, x_max = np.min(X_train, axis=0), np.max(X_train, axis=0)
        self.centroids = [np.random.uniform(x_min, x_max) for _ in range(self.n_clusters)]
        
        ## 4) Repeat Step 2-3
        i = 0
        centroids_cache = self.centroids
        while i < self.max_iter and np.not_equal(self.centroids, centroids_cache).any():
            ## 2) Find cluster for each data point
            clustered_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                distances = [distance(x, centroid) for centroid in self.centroids]
                clustered_points[np.argmin(distances)].append(x)
            
            ## 3) Update centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in clustered_points]
            i += 1
```

### Gaussian Mixture Model
Idea: clustering but deterministic $ r\_{ik}=\textbf{1}\\{c_i=k\\}$ $\rightarrow$ stochastic $r\_{ik}=\mathbb{E}[\textbf{1}\\{z_i=k\\}]=P(z_i=k|\textbf{x}_i) $.

Background: Shapes of clusters are determined by the properties of their covariance matrices.
- **Shared Spherical**: #params=1, same $ \sigma^2$ for all features, all features independent, same $\Sigma $ for all clusters.
    - K-means = GMM with $ N(\boldsymbol{\mu}_k,\sigma^2I) $
- **Spherical**: #params=$ k$, same $\sigma^2$ for all features, all features independent, diff $\Sigma_k $ for all clusters.
- **Shared Diagonal**: #params=$ n$, diff $\sigma_k^2$ for all features, all features independent, same $\Sigma $ for all clusters.
- **Diagonal**: #params=$ kn$, diff $\sigma_k^2$ for all features, all features independent, diff $\Sigma_k $ for all clusters. 
- **Shared Full Covariance**: #params=$ \frac{n(n+1)}{2}$, same $\Sigma $ for all clusters. 
- **Full Covariance**: #params=$ \frac{kn(n+1)}{2}$, diff $\Sigma_k $ for all clusters. 

Assumptions:
- There exists a latent variable $ z\in\\{1,\cdots,K\\} $ representing the index of the Gaussian distribution in the mixture.

Model: $ P(\mathbf{x}_i,z_i)=P(\mathbf{x}_i|z_i)P(z_i) $, where
- $ z_i\sim\text{Multinomial}(\boldsymbol{\phi})$, $\boldsymbol{\phi}\in\mathbb{R}^{K} $
- $ (\mathbf{x}_i|z_i=k)\sim N(\boldsymbol{\mu}_k,\Sigma_k)$, $\boldsymbol{\mu}_k\in\mathbb{R}^n,\Sigma_k\in\mathbb{R}^{n\times n} $

#params: $ (K-1)+Kn+K\frac{n(n+1)}{2} $

Optimization: EM
1. Init distributions for prior $ P(z_i)$ and likelihood $P(\mathbf{x}_i|z_i) $
2. E-step: estimate $ P(z_i=k|\textbf{x}_i)$ given params $(\boldsymbol{\phi},\boldsymbol{\mu}_k,\Sigma_k) $:
$$
r\_{ik}=P(z_i=k|\textbf{x}_i)=\frac{P(\textbf{x}_i|z_i=k)P(z_i=k)}{\sum\_{k=1}^{K}P(\textbf{x}_i|z_i=k)P(z_i=k)}
$$
3. M-step: estimate params via MLE given $ P(z_i=k, \textbf{x}_i) $:
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

## Dimensionality Reduction
Dimensionality reduction reduces dimensionality (i.e., #features) of input data.

We need dimensionality reduction because:
- **Curse of Dimensionality**: high-dimensional data has high sparsity.
- **Computational Efficiency**: high-dimensional data requires more computational resources (time & space).
- **Overfitting**: high-dimensional data are prone to overfitting.
- **Visualization**: we cannot visualize any data beyond 3D.
- **Performance**: high-dimensional data are prone to have more noises, which are reducible by selecting the most important features.
- **Interpretability**: only the most relevant features matter.



### Singular Value Decomposition
**Model**:
$$
X=UDV^T=\sum_{i=1}^{\text{rank}(X)}D_{ii}\textbf{u}_i\textbf{v}_i^T
$$
- Input:
    - $ X\in\mathbb{R}^{m\times n} $: arbitrary input matrix
- Output (Matrix ver.):
    - $ U\in\mathbb{R}^{m\times m} $: left singular vectors (i.e., final rotation)
    - $ D\in\mathbb{R}^{m\times n} $: singular values (i.e., scaling)
    - $ V\in\mathbb{R}^{n\times n} $: right singular vectors (i.e., initial rotation)
    - $ \text{rank}(X)=\min(m,n) $: rank
- Output (Vector ver.):
    - $ \textbf{u}_i\textbf{v}_i^T$: outer product matrix of $i $th column unit vectors
    - $ D_{ii}$: importance/strength of $i $th outer product matrix
- Note:
    - rotation matrices are **orthonormal**: $ U^TU=I, V^TV=I $
    - scaling matrix is **diagonal**: $ D=\text{diag}(\sigma_1,\cdots,\sigma_{\text{rank}(X)})$, where $\sigma_i=\sqrt{\lambda_i} $

**Idea**: An arbitrary matrix = "unit matrix $ \rightarrow$ initial rotation $\rightarrow$ scaling $\rightarrow $ final rotation"
- Think of an arbitrary $ 2\times 2 $ matrix on a 2D plane. We can reconstruct this matrix with a unit disk (i.e., the 2 unit vectors along the coordinates) via the following steps:
    1. Rotate the unit vectors by $ V^T $.
    2. Scale the rotated unit disk into an ellipse by $ D $.
    3. Rotate the ellipse by $ U $.

**Properties**:
- $ X^TX=V(D^TD)V^T=\sum_{i=1}^n(D_{ii})^2\textbf{v}_i\textbf{v}_i^T $ (i.e., right singular vectors = eigenvectors of covariance matrix)
- $ XX^T=U(DD^T)U^T=\sum_{i=1}^m(D_{ii})^2\textbf{u}_i\textbf{u}_i^T $ (i.e., left singular vectors = eigenvectors of outer product matrix)
- Pseudo-inverse: $ X^+=VD^{-1}U^T\in\mathbb{R}^{n\times m} $
- If $ X$ is a positive semi-definite matrix, then $\sigma_{i}=\lambda_{i} $ (i.e., singular value = eigenvalue).
    - Positive Semi-Definite: A symmetric matrix $ X$ s.t. $\forall\textbf{z}\in\mathbb{R}^n,\textbf{z}\geq\textbf{0}: \textbf{z}^TX\textbf{z}\geq 0 $.

**Applications**:
- **Simplify OLS in regression**: when calculating the "inverse" of a rectangular matrix, $ (X^TX)^{-1}X^T\approx X^+ $.
- **Low-rank matrix approximation**: define a lower rank $ k$ & approximate $X\approx U_kD_kV_k^T $.
- **Eigenword**: project high-dimensional context to low-dimensional space, assuming distributional similarity.
    - Distributional similarity: words with similar contexts have similar meanings.
    - Distance-based similarity measure: similar words are close in this low-dimensional space.
    - Eigenwords: left singular vectors (i.e., word embeddings).
    - Eigentokens: right singular vectors $ \times $ Context (i.e., contextual embeddings).
    - Word sense disambiguation: estimate contextual embedding for a word with right singular vectors.
- **PCA**: see next.



### Principal Component Analysis
**Model (Original ver.)**:
1. Calculate the covariance matrix of $ X$ in the observation space: $\Sigma=\text{Cov}(X,X)=X^TX $.
2. Diagonalize the covariance matrix via Spectral Theorem: $ \Sigma=V\Lambda V^T $.
    - $ \Lambda $: eigenvalues (i.e., PC strength / sample variance of projection)
    - $ V $: orthonormal eigenvectors (i.e., PCs)
3. Sort eigenvectors in $ V$ based on eigenvalues in $\Lambda $ in descending order.
4. Select & normalize the strongest $ k$ eigenvectors $\\{\textbf{v}_1,\cdots,\textbf{v}_k\\}$, where $k $ is a hyperparameter.
    - Now $ \Lambda\in\mathbb{R}^{k\times k}, V\in\mathbb{R}^{n\times k} $.
5. Project $ X$ onto a new space based on the $k$ eigenvectors: $Z=XV $
    - Each projected point is: $ \textbf{z}_i=(\textbf{x}_i^T\textbf{v}_1,\cdots,\textbf{x}_i^T\textbf{v}_k) $

**Model (SVD ver.)**:
1. Compute SVD: $ X=UDV^T $.
2. Select $ k$ rows of $V^T $ (the right singular matrix) with the largest singular values as PCs.
3. Project the original dataset $ X$ onto a new space based on the $k $ eigenvectors.

**Idea**: We can project the input data onto an orthonormal basis $ \\{\textbf{v}_1,\cdots,\textbf{v}_k\\} $ of smaller dimensions while covering maximal variance among features.
- **Orthonormal basis**: $ \textbf{v}_i^T\textbf{v}_j=0,\textbf{v}_i^T\textbf{v}_i=1 $.
- Ideally, we lose minimal info (represented by minimal variance) while successfully reducing the dimensionality.

**Assumptions**:
- Input matrix $ X $ is centered/standardized on the sample space, unless data is sparse.
    - $ \bar{\textbf{x}}=\frac{1}{m}\sum_{i=1}^{m}\textbf{x}_i=\textbf{0} $
    - $ s\_j^2=\frac{1}{m}\sum\_{i=1}^{m}x\_{ij}^2=1 $ (optional but strongly recommended)
- PCs = linear combinations of original features. (if not, then Kernel PCA)
- Variance = a measure of feature importance.

**Optimization**: minimize distortion (i.e., maximize variance in new coordinates)
- **Distortion**:
$$\begin{align*}
\text{Distortion}_k=||X-ZV^T||_F&=\sum\_{i=1}^m||\textbf{x}_i-\hat{\textbf{x}}_i||_2^2 \\\\
&=\sum\_{i=1}^m\sum\_{j=k+1}^{n}z\_{ij}^2\\\\
&=m\sum\_{j=k+1}^{n}\textbf{v}_j^T\Sigma\textbf{v}_j\\\\
&=m\sum\_{j=k+1}^{n}\lambda_j
\end{align*}$$
- **Variance** (of projected points):
$$
\text{Variance}_k=m\sum\_{j=1}^{k}\textbf{v}_j^T\Sigma\textbf{v}_j=m\sum\_{j=1}^{k}\lambda_j
$$
- **Minimizing distortion = Maximizing variance**:
$$
\text{Variance}_k+\text{Distortion}_k=m\sum\_{j=1}^{n}\lambda_j=m\cdot\text{trace}(\Sigma)
$$

**Inference/Reconstruction**: Any sample vector can be approximated in terms of coefficients (scores) on eigenvectors (loadings).
1. Predict the original point via inverse mapping: $ \hat{\textbf{x}}_i=\sum\_{j=1}^kz\_{ij}\textbf{v}_j $
    - The projected new points $ z_i $s can still correlate with each other. Only their basis are independent. Therefore, variance is still not 1 even if you standardize data.
2. The original point can be fully reconstructed if $ k=n $:
$ \textbf{x}_i=\bar{\textbf{x}}+\sum\_{j=1}^nz\_{ij}\textbf{u}_j $

**Pros**:
- Guarantee removal of correlated features (PCs are orthogonal)
- Reduce overfitting for supervised learning
- Improve visualization for high-dimensional data
- Robust to outliers & noisy data

**Cons**:
- Scale invariant
- Low interpretability of new features (i.e., PCs)
- Potential info loss if PCs & $ \\## $PCs are not selected carefully
- Situational (e.g. cannot be applied on NLP because 1) covariance matrix is useless 2) it breaks sparse structure of words)
- (weak) If any assumption fails, PCA fails (solvable by Kernel PCA)


<!-- ### Independent Component Analysis
Idea: find an embedding so that different features are "deconfounded" (i.e., as independent as possible from each other).

tbd -->



### Autoencoder
Idea: Use unsupervised NNs to **learn latent representations** via reconstruction. (Semi-supervised learning)
- The goal of AE is NOT to reconstruct the input as accurately as possible but to LEARN major features from it. Reconstruction is only an objective for the learning process.

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


### Variational Autoencoder
tbd

<!-- ## Bayesian Belief Networks -->
<!-- The core of generative models in comparison to discriminative models is that the generative model **GENERATES** samples, which many newbies like me overlooked at the very beginning.

Discriminative models predict label given sample features, but Generative models uses a completely different thinking process, where we
1) propose/calculate the prior of labels $ P(Y) $, using relevant params for the prior distribution,
2) calculate the likelihood of the current combination of values of sample features given the label $ P(X_1,\cdots,X_n|Y) $, using relevant params for the likelihood distribution,
3) calculate $ P(X,Y)=P(Y)P(X_1,\cdots,X_n|Y) $ for generation;
    
    calculate $ P(Y|X)\propto P(Y)P(X_1,\cdots,X_n|Y) $ for discrimination.

During training, we estimate the params which maximize the combination of prior distribution $ \times $ likelihood distribution.

During Prediction, we directly use those params to either generate samples or compute label for the given sample.

The following models are already abandoned in practice because of DL, but the ideas behind them are still important for forming a deep understanding of ML.

Note that Naive Bayes and GMM are also generative models. This section is more like a miscellaneous collection of Bayesian-Network-based models. -->

<!-- ### Bayesian Network
Idea: compact specification of full joint distributions with CI assertions using Conditional Probability Table (CPT)

Background:
- CI properties:
    - Symmetry: $ (X\perp Y|Z)\rightarrow(Y\perp X|Z) $
    - Decomposition: $ (X\perp Y,W|Z)\rightarrow(X\perp Y|Z) $
    - Weak union: $ (X\perp Y,W|Z)\rightarrow(X\perp Y|Z,W) $
    - Contraction: $ (X\perp W|Y,Z),(X\perp Y|Z)\rightarrow(X\perp Y,W|Z) $
    - $ P(X|Y)+P(X|\neg Y)\neq1 $ (they are NOT related)
    - $ P(X|Y)+P(\neg X|Y)=1 $
- **Active Trail** (in an acyclic graph): for each consecutive triplet in the trail:
    - $ X\rightarrow Y\rightarrow Z$ and $Y $ is NOT observed.
    - $ X\leftarrow Y\rightarrow Z$ and $Y $ is NOT observed.
    - $ X\rightarrow Y\leftarrow Z$ and $Y $ or one of its descendants IS observed.
- If there is NO active trail between $ X$ and $Y $, then they are CI. (i.e., **D-separation**)
- NB = basic BayesNet with 1 parent (label) and multiple children (features)
- Complexity scales exponentially with #parents; Complexity scales linearly with #children.

Assumption: A variable $ X $ is independent of its non-descendants given its parents (Local Markov Assumption)

Model/Algorithm: Graph (built from data/people, or automatic search)

Objective: NLL or 0-1

Optimization:
- Annealing (if all vars are observable):
    1. Estimate params from data.
    2. Randomly change the net structure by one link.
    3. Re-estimate params.
    4. Accept change if lower loss, else repeat Step 2-4.
- EM (if hidden vars exist):
    1. Assuming priors onto the latent variables, estimate params from data (CPT).
    2. With the params (probabilities), estimate expected values of the latent variables.

Pros:
- Guaranteed to be a consistent specification
- Clear visualization of conditional independence (a compact representation of joint distributions)
- Nets that capture causality tend to be sparser
- Easy estimation when everything is observable and when the net structure is available

Cons:
- There is still no universal method for constructing BayesNet from data (require serious search if net structure is unknown)
- Fail to define cyclic relationships
- bad performance on high-dimensional data


### Latent Dirichlet Allocation
Usage: Topic Modeling

Assumption: CI (just like NB)

Background:
- Multinomial (like Binomial) distribution models the outcomes of a series of i.i.d. experiments (e.g., dice rolling).
- Dirichlet (like Beta) distribution models probability vectors.
- We do not know anything about the probabilities of each outcome at the beginning (i.e., the params of Multinomial distribution), so we use Dirichlet distribution to offer us a prior over these params.
- Therefore, Dirichlet (like Beta) distribution is a conjugate prior for Multinomial (like Binomial) distribution.

Model/Algorithm: For each document $ d $,
1. Choose its topic distribution $ \boldsymbol{\theta}_d\sim\text{Dirichlet}(\alpha)$, where $\theta\_{dk}=p(\text{topic}=k|\text{document}=d) $
2. For each word $ w_j$ in $d $:
    1. Choose this word's topic $ z_{dj}\sim\text{Multinomial}(\boldsymbol{\theta}_d) $
    2. Choose a word $ w_j\sim\text{Multinomial}(\beta_{z_{dj}})$ where $\beta_{z_{dj}}=p(w_j|z_{dj}) $

Objective: 0-1

Optimization: EM (Variational EM in practice)
- E-step: Compute $ p(\boldsymbol{\theta},\textbf{z}|d;\alpha,\boldsymbol{\beta})$ (posterior of hidden vars $(\boldsymbol{\theta},\textbf{z})$ given each document $d $)
- M-step: Estimate params $ (\alpha,\boldsymbol{\beta}) $ given posterior estimates

Naive Bayes vs LDA:
- Naive Bayes assumes each doc is on a single topic.
- LDA allows each doc to be a mixture of topics (i.e., each word can be on a different topic).

Pros: 
- Discovery of implicit topics that were not explicit in the documents
- Applicable on semi-supervised learning
- Sometimes dimensionality reduction

Cons:
- High computational cost (Param Estimation is a bit messy)
- Low interpretability

### Hidden Markov Model
Idea: Hidden Markov Chain + Observed variables

Usage: Seq2Seq Synthesis (Speech recognition, POS Tagging, Named Entity Recognition, etc.)
- Evaluation: compute $ P(X)$ given $X=[x_1,\cdots,x_T]$ and $(A,B,\pi) $.
- Decoding: find the best $ S=[s_1,\cdots,s_T]$ which best explains the observations given $X=[x_1,\cdots,x_T]$ and $(A,B,\pi) $.
    - i.e., $ \arg\max\_{[s_1,\cdots,s_T]}\prod P(x_i|s_i)P(s_i|s_{i-1}) $, where the first term is from emission matrix, and the second term is from transition matrix.
- Learning: estimate $ (A,B,\pi)$ which maximize $P(X;A,B,\pi) $.

Background:
- Transition Matrix: specifying transition probabilities from one state to another.
- Emission Matrix: specifying the probabilities of each observed outcome to occur given each hidden state.

Assumptions:
- Markov Assumption: $ P(X_t|X_{t-1},\cdots,X_1)=P(X_t|X_{t-1}) $.
- CI Assumption: $ S_t$ D-separates all $X\in\textbf{X}\_{<t}$ from all $X\in\textbf{X}\_{>t} $.
    - The hidden state at time $ t$ D-separates all emissions/observations at times before $t$ from all emissions/observations at times after $t $.
    - The future is independent of the past given the present.
- Stationarity: Transition matrix and emission probabilities stay the same over time.

Model:
1. Start in some initial state $ s_i$ with probability $p(s_i)=\pi $.
2. Move to a new state $ s_j$ with probability $p(s_j|s_i)=a_{ij}$, where $a_{ij}$ is a cell value in transition matrix $A $.
3. Emit an observation $ x_v$ with probability $p(x_v|s_i)=b_{iv}$, where $b_{iv}$ is a cell value in emission matrix $B $.


Pros:
- Can handle inputs of variable lengths
- Efficient learning
- Wild range of applications (until DL bloomed)

Cons:
- A large number of unstructured params
- Limited by Markov Assumption
- No dependency between hidden states
- Completely destroyed by DL -->