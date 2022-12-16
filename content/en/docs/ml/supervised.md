---
title : "Supervised Learning"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: false
images: []
weight: 300
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
- Inference
- Objective (mainly Loss Function)
- Optimization (mainly Parameter Estimation)
- Pros
- Cons
- (extra)

# Linear Models
- All linear models are parametric.

Types:

* Linear Regression

$$
\hat{y_i}=\sum_{j=1}^{n}{w_jx_{ij}}
$$

* Generalized Linear Models ($f$: link function; $\mathbf{w}^T\mathbf{x}$: logits)

$$
\hat{y_i}=f\left(\sum_{j=1}^{n}{w_jx_{ij}}\right)
$$

* Basis Transformation ($\phi_j(\mathbf{x}_i)$: transformation of non-linear inputs into linear inputs)

$$
\hat{y_i}=\sum_{j=1}^{d}{w_j\phi_j(\mathbf{x}_i)}
$$

## Linear Regression
Idea: fit a linear hyperplane between features and labels.

Assumptions:
- **Linearity**: The underlying relationship between $\textbf{y}$ and $X$ is linear.
- **Independence**: $\varepsilon_i$ is independent of each other.
- **Normality**: $\varepsilon_i$ follows Gaussian distribution.
- **Non-Collinearity**: No/Minimal explanatory variables correlate with each other.
- **Homoskedasticity**: The variance of all noises is the same constant $\sigma^2$.

Model:
$$\begin{align*}
&\text{Frequentist:} &&\mathbf{y}=X\mathbf{w}+\boldsymbol{\varepsilon},\varepsilon_i\sim N(0,\sigma^2) \\\\
&\text{Bayesian:}    &&p(\mathbf{y}|X,w)=N(X\mathbf{w},\sigma^2)
\end{align*}$$

Inference:
$$
\hat{y}_i=\mathbf{w}^T\mathbf{x}_i
$$

Objective:
- Loss: MSE
- Regularization: L1, L2, ElasticNet, L0

Optimization:

$$\begin{aligned}
&\text{OLS/MLE}:\ &&\hat{\mathbf{w}}=(X^TX)^{-1}X^T\mathbf{y} \\\\
&\text{Ridge}:\ &&\hat{\mathbf{w}}=(X^TX+\lambda I)^{-1}X^T\mathbf{y} \\\\
&\text{Lasso}:\ &&\frac{\partial \mathcal{L}_B}{\partial w_j}=\frac{1}{m}\sum\_{i=1}\^{m}[x\_{ij}(\hat{y}_i-y_i)]+\lambda\cdot\text{sign}(w_j) \\\\
&\text{ElasticNet}:\ &&\frac{\partial \mathcal{L}_B}{\partial w_j}=\frac{1}{m}\sum\_{i=1}\^{m}[x\_{ij}(\hat{y}_i-y_i)]+\lambda r_1\cdot\text{sign}(w_j)+(1-\lambda r_1)w_j
\end{aligned}$$

Pros:
- Simple and Interpretable
- Scale invariant
- Consistent and Unbiased (OLS/MLE ver.)

Cons:
- Sensitive to outliers
- Limited to assumptions (if any assumption fails, LinReg fails)

Time complexity:
- Train:
    - Exact Solution: $O(n^2(m+n))$
    - Gradient Descent: $O(mn)$
- Test: $O(n)$

Code:
```python
class LinearRegression:
    def __init__(self):
        self.w = None
    
    def fit(self, X, y, intercept=False):
        if intercept: X = np.hstack(np.ones((X.shape[0],1)),X)
        self.w = (np.linalg.inv(X.T @ X) @ X.T @ y).reshape(-1)

    def predict(self, X):
        return np.dot(self.w, X.T)
```

## Logistic Regression
Idea: use sigmoid/softmax as link function for linear regression for classification.

Model:
$$\begin{align*}
&\text{Binary}: &&P(y_i=1|\mathbf{x}_i,\mathbf{w})=\sigma(\mathbf{w}^T\mathbf{x}_i)=\frac{1}{1+\exp{(-\mathbf{w}^T\mathbf{x}_i)}}=\frac{\exp{(\mathbf{w}^T\mathbf{x}_i)}}{1+\exp{(\mathbf{w}^T\mathbf{x}_i)}}\\\\
&\text{Multiclass}: &&P(y_i=k|\mathbf{x}_i,W)=\text{softmax}(W^T\mathbf{x}_i)=\frac{\exp{(\mathbf{w}_k^T\mathbf{x}_i)}}{\sum\_{k=1}^{K}{\exp{(\mathbf{w}_k^T\mathbf{x}_i)}}}
\end{align*}$$

Inference:
$$
\hat{y}_i=\arg\max_k\hat{p}\_{ik}
$$

Objective:
- Loss: Cross Entropy
- Regularization: L1, L2, ElasticNet

Optimization: Gradient Descent
$$
\frac{\partial \mathcal{L}}{\partial w_{jk}}=\frac{1}{m}\sum_{i=1}\^{m}[x\_{ij}(\hat{p}_{ik}-y_i)]
$$

Pros:
- Scale invariant
- Easy expansion to multiclass
- Coefficients resemble feature importance
- Easy gradient calculation

Cons:
- Named regression but can only work for discrete classification
- bad performance when $n>>m$
- bad performance for nonlinear cases (assume linearity by log odds)

Time Complexity: Train: $O(mn)$; Test: $O(n)$

Space Complexity: $O(n)$

## Support Vector Machine
Idea: choose a decision boundary that maximizes the soft margin between classes.

Background:
- **Primal vs Dual**: Primal form operates in the feature space $X^TX$, while Dual form operates in the sample space $XX^T$ (i.e., Kernel Matrix)
    - The params in Primal and Dual are transferable: $\textbf{w}=\sum_{i=1}^m\alpha_iy_i\textbf{x}_i$
- **Hard margin vs Soft margin**: Hard margin does NOT accept any misclassification (thus prone to overfitting), while Soft margin allows some misclassifications (thus regularization).

Model/Inference: linear classifier
$$
\hat{y}_i=\text{sign}(\textbf{w}^T\phi(\textbf{x}_i))
$$

Objective: Hinge Loss
- Primal (Linear):
$$\begin{align*}
\min\_{w,\xi}\quad & \frac{1}{2} ||\textbf{w}||^2 + C \sum\_{i=1}^m \xi_i \\\\
\text{s.t.}\quad & y_i(\textbf{w}^T\textbf{x}_i) \geq 1-\xi_i\\\\
& \xi_i \geq 0
\end{align*}$$
- Primal (Kernel):
$$\begin{align*}
\min\_{w,\xi}\quad & \frac{1}{2} ||\textbf{w}||^2 + C \sum\_{i=1}^m \xi_i \\\\
\text{s.t.}\quad & y_i(\textbf{w}^T\phi(\textbf{x}_i)) \geq 1-\xi_i\\\\
& \xi_i \geq 0
\end{align*}$$
- Dual (Linear):
$$\begin{align*}
\max\_{\alpha\geq 0}\quad & \sum_{i=1}^m \alpha_i - \frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m y_i y_j \alpha_i \alpha_j \mathbf{x}_i^T \mathbf{x}_j \\\\
\text{s.t.}\quad & \sum\_{i=1}^n \alpha_i y_i = 0\\\\
& \alpha_i\leq C
\end{align*}$$
- Dual (Kernel):
$$\begin{align*}
\max\_{\alpha\geq 0}\quad & \sum_{i=1}^m \alpha_i - \frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m y_i y_j \alpha_i \alpha_j k(\mathbf{x}_i, \mathbf{x}_j) \\\\
\text{s.t.}\quad & \sum\_{i=1}^n \alpha_i y_i = 0\\\\
& \alpha_i\leq C
\end{align*}$$

Optimization: Decomposition (quadratic programming), Closed-form, GD, etc.

Pros:
- Good performance on high-dimensional and non-linearly separable data (with Kernel trick)
- Good generalization to unseen data
- Guaranteed convexity. Easy to optimize
- Low memory cost (only support vectors matter)

Cons:
- High computational cost, especially 1) with Kernels 2) when sample size is too large 3) with Multiclass classification (no native support for it; need 1v1 or 1-v-rest strategies)
- Low interpretability: No probability estimates
- Bad performance when $n>>m$.
- Bad performance on large datasets (i.e., $m>>0$).
- Sensitive to outliers and noisy data
- Sensitive to overlapping classes (i.e., classes which share the same parts of some feature values)

Time Complexity:
- Train: $O(m^2)$
- Test: $O(kn)$, where $k=$ #support vectors.

# Local Learning

## K Nearest Neighbors
Idea: generate label for a sample from its $K$ nearest neighbors.

Model/Algorithm:
1. Calculate distance between sample point and every training point.
2. Find the $K$ nearest neighbors with minimal distances.
3. Take the majority vote and output it as the label for the sample point.

Pros:
- No training: instance-based learning (i.e., lazy learner)
- Seamless data augmentation at any step
- Simple and interpretable

Cons:
- Scale variant
- High computational cost for large datasets
- High computational cost + Low variance in distance measure for high-dimensional data
- Sensitive to noisy data, missing values, and outliers

Time Complexity: Test: $O(kmn)$

Space Complexity: $O(mn)$

Code:

```python
###### Scratch ######
def distance(metric_type,v1,v2):
    if metric_type == "L0":
        return np.count_nonzero(v1-v2)
    if metric_type == "L1":
        return np.sum(np.abs(v1-v2))
    if metric_type == "L2":
        return np.sqrt(np.sum(np.square(v1-v2)))
    if metric_type == "Linf":
        return np.max(np.abs(v1-v2))
    
def KNN(X_train,y_train,samples,K=5,metric_type="L2"):
    def KNN_for_single_sample(K,metric_type,X_train,y_train,sample):
        # Calculate dist between each X_train[i] and sample
        dis_vec = np.array([distance(metric_type,X_train[i],sample) for i in range(len(X_train))])
        
        # Find index of top-K neighbors
        ids = np.argsort(dis_vec)[:K]
        
        # Find neighbors' labels from y_train
        neighbors = list(y_train[ids])
        
        # Return majority vote for the sample
        return max(set(neighbors),key=neighbors.count)
    
    return [KNN_for_single_sample(K,metric_type,X_train,y_train,sample) for sample in samples]
```

## Kernel Regression

Model/Algorithm:

$$\begin{align*}
&\text{Regression}: &&\hat{y}=\frac{\sum\_{i=1}^{m}{k(\mathbf{x},\mathbf{x}_i)y_i}}{\sum\_{i=1}^{m}{k(\mathbf{x},\mathbf{x}_i)}}\\\\
&\text{Binary Classification}: &&\hat{y}=\text{sign}(\sum\_{i=1}^{m}{k(\mathbf{x},\mathbf{x}_i)y_i})
\end{align*}$$

Pros:
- Save computation cost (for SVM) since no need to actually convert features to higher-dimensional data to find nonlinear patterns.
- Easy to test whether a proposed kernel is valid by finding arbitrary 2 points with negative determinant (i.e., negative eigenvalue).
- Can be extended on literally any data type.

Cons:
- Hard to choose the suitable kernel.
- Hard to comprehend what kernels learned exactly.
- Easy overfitting.

KNN vs KernelReg: 

<center>

|               KNN              |                   Kernel                  |
|:------------------------------:|:-----------------------------------------:|
|         distance metric        |              kernel function              |
|         $K$ neighbors          |               all neighbors               |
| same impact from all neighbors | weighted impact favoring closer neighbors |
|          Scale variant         |               Scale variant               |

</center>

# Decision Tree
Idea: build a tree where each node is a feature split to classify data points into different leaf outputs.

Background:
- Information gain: $IG(Y|X)=H(Y)-H(Y|X)$
    - $H(\cdot)$: Impurity measure
        - Gini: $H(Y)=\sum_{y}{p_y(1-p_y)}$
        - Entropy: $H(Y)=-\sum_{y}{p_y\log_2{p_y}}$
- Entropy: a measure of uncertainty
    - Conditional entropy (Average): $H(Y|X)=\sum_{x}{P(X=x)H(Y|X=x)}$
    - Specific conditional entropy: $H(Y|X=x)=-\sum_{y}{P(Y=y|X=x)\log_2{P(Y=y|X=x)}}$

Model/Algorithm:
1. Calculate info gain for each feature.
2. Select the feature that maximizes info gain as the decision threshold.
3. Split data based on the decision. Repeat Step 1-2 until stop.

Pros:
- Scale invariant
- Low computational cost
- Can handle multiclass classification
- Interpretable and Easy to validate (easy model visualization)

Cons:
- Prone to overfitting (perfect fit on smaller sample size)
- Bad performance with class imbalance (biased toward most frequently occurring class)
- Sensitive to noisy data, missing value, and outliers
- Discrete predictions only
- Difficult to find globally optimal decisions (only support locally optimal decisions at each node)
- Bad performance overall

Time complexity:
- Train: $O(mn\log{m})$
- Test: $O(d)$, where $d=$ depth. (Ideally $O(\log{m})$ if balanced binary tree)

Code:
```python
###### SCRATCH ######
# This scratch is only meant for binary classification.

def IG(X_train,y_train,feature_index,impurity="entropy"):
    # Compute impurity
    def H(probs):
        if impurity=="entropy":
            return -np.sum(np.multiply(probs[probs!=0],np.log2(probs[probs!=0])))
        if impurity=="gini":
            return np.sum(np.multiply(probs,1-probs))
    
    # Calculate H(Y)
    m = len(y_train)
    p_y_1 = np.count_nonzero(y_train)/m
    p_y = np.array([1-p_y_1,p_y_1])
    H_y = H(p_y)
    
    x_col = X_train[:,feature_index].reshape(-1)
    y_col = y_train.reshape(-1)
    p_y_x = {}  # for each value of the selected feature, store its y count. 
    p_x = {}    # for each value of the selected feature, store its own count.
    
    # Count all occurrences of all values of the selected feature, together with their corresponding y value counts.
    for i in range(m):
        p_x[x_col[i]] = p_x.get(x_col[i],0)+1
        if x_col[i] not in p_y_x:
            p_y_x[x_col[i]] = [0,0]
        if y_col[i] == 0:
            p_y_x[x_col[i]][0] += 1
        elif y_col[i] == 1:
            p_y_x[x_col[i]][1] += 1
            
    # Calculate H(Y|X=v)
    H_y_x_specs = {} # for each value v of the selected feature, store H(Y|X=v).
    x_total_count = sum(p_x.values())
    for key in p_y_x:
        y_total = sum(p_y_x[key])
        p_x[key] /= x_total_count   # normalize counts to P(X=v)
        p_y_x[key] = [p_y_x[key][0]/y_total, p_y_x[key][1]/y_total] # normalize counts to P(Y=u|X=v)
        H_y_x_specs[key] = H(np.array(p_y_x[key])) 
        
    # Calculate H(Y|X)
    H_y_x = sum([p_x[key]*H_y_x_specs[key] for key in p_x])
    
    # Return IG
    return H_y-H_y_x

def select_feature_max_IG(X_train,y_train,impurity="entropy"):
    IG_cache = [IG(X_train,y_train,i,impurity="entropy") for i in range(X_train.shape[1])]
    return IG_cache.index(max(IG_cache))

# to be continued
```

# Ensemble Methods
- **Bootstrapping**: randomly select $fm$ samples with replacement from the original training set into subsets, where $f$ is the fraction of samples to bootstrap.
- **Bagging** (bootstrap aggregation): bootstrap, get bunch of weak models trained on separate subsets individually, and aggregate their predictions via mean or majority.
- **Boosting**: train multiple models sequentially and dependently.
    - Converge exponentially with #iterations.
    - Cons: Highly sensitive to outliers and noisy data.
- Common Pros of Ensemble Methods:
    - Scale invariant
    - Reduce overfitting
    - Greater performance
    - Can handle high-dimensional data efficiently
- Common Cons of Ensemble Methods: 
    - High computational cost
    - Low interpretability

## Random Forest
Idea: Bagging with Decision Trees

Model/Algorithm: 
1. Bootstrap.
2. Create a full decision tree (no pruning). On each node, randomly select $\sqrt{n}$ features from the bootstrapped subset. Find the best split.
3. Repeat 1-2 to create a random forest till #tree limit.
4. Use out-of-bag samples to determine the accuracy of each tree.

Inference: Take a majority/average vote of all trees.

Pros:
- Reduce overfitting in decision tree & much more accurate than a single decision tree
- Flexible to both categorical & numerical outputs
- Automatically handle missing values by dropping or filling with median/mode
- Robust to outliers and noisy data
- Best used in banking and healthcare

Cons:
- Worse performance than Boosting in general

Time Complexity:
- Train: $O(kmn\log{m})$, where $k=$ #trees
- Test: $O(kd)$, where $d=$ max depth

## AdaBoost

Idea: train a bunch of stumps (weak learners) sequentially and take a weighted majority vote.

Model/Algorithm:
1. Init all samples with equal sample weight.
2. Find optimal feature for first stump. Calculate total error. Calculate amount of say:
$$
\text{Amount of Say}=\frac{1}{2}\log{\frac{1-Err_\text{tot}}{Err_\text{tot}}}
$$
3. Modify sample weights for incorrectly and correctly predicted samples as follows:
$$\begin{align*}
w_\text{incorrect}&\leftarrow w_\text{incorrect}\cdot e^\text{Amount of Say}\\\\
w_\text{correct}&\leftarrow w_\text{correct}\cdot e^{-\text{Amount of Say}}\\\\
\end{align*}$$
4. Normalize all sample weights.
5. Select new samples based on new sample weights as probabilities with replacement to generate a new training set.
6. Give equal sample weights to all samples in the new training set.
7. Repeat Steps 1-6 till #stumps reach limit.

Inference: Take a weighted majority vote using the Amount of Say from each stump.

Objective: Exponential Loss

Pros:
- Reduce overfitting more than bagging (because parameters are not optimized jointly but stagewise)
- Fewer hyperparameters than other models

Cons:
- Sensitive to outliers and noisy data
- Slower and generally worse performance than Gradient Boosting

Random Forest vs AdaBoost:
<center>

|         Random Forest         |                 AdaBoost                 |
|:-----------------------------:|:----------------------------------------:|
|   No tree shape requirement   | Each tree is a stump (1 node + 2 leaves) |
| Each tree has equal influence |      Stumps have weighted influence      |
|    Each tree is independent   |  Each stump is dependent of its previous one |

</center>

## Gradient Boosting
Idea: train a bunch of fixed-size trees to fit residuals sequentially. take a weighted majority vote.

Model/Algorithm:
1. Init a leaf as the average of $\mathbf{y}$. Calculate residuals.
2. Train a tree (#leaves < $m$) on the residuals. Scale it with a fixed learning rate.
3. Combine leaf (and previous trees) and current tree to make new predictions on the same data. Calculate new residuals.
4. Repeat Steps 2-3 till limit.

Optimization (Hyperparams):
- #stages: $T$
- bag size (fraction): $f$
- learning rate: $\eta$
- tree depth: $d$

Pros:
- Great performance in general
- Fast inference
- High flexibility (multiple hyperparameters to tune, multiple loss functions to use)
- Can handle missing data
- Reduce overfitting by using a small learning rate and incorporate bootstrapping

Cons:
- Sensitive to outliers and noisy data (may cause overfitting by overemphasizing them)
- Require a large grid search for hyperparameter tuning
- Longer training due to sequential building


AdaBoost vs Gradient Boosting:

<center>

|         AdaBoost         |       Gradient Boosting       |
|:------------------------:|:-----------------------------:|
|     Init with a stump    | Init with a leaf of $\bar{y}$ |
|          Stumps          |        Fixed-size trees       |
| Scale stumps differently |      Scale trees equally      |
| Train on $y$ |      Train on residuals      |

</center>


## XGBoost
Idea: use a unique tree to make decisions based on similarity scores.

Background:
- **Pre-sort algorithm**: sort samples by the feature value, then split linearly.
- **Histogram-based algorithm**: bucket continuous feature values into discrete bins.
- **Level-wise tree growth** (BFS)

Model/Algorithm:
1. Calculate residuals for all samples based on current prediction. Calculate similarity score for the root node of a new tree.
$$
\text{Similarity}=\frac{\sum_{i=1}^{m_r}{r_i^2}}{m_r+\lambda}
$$
2. Find similarity gain of each possible split. Choose the split with the max gain at root node.
$$
\text{Gain}=\text{Similarity}\_\text{left}+\text{Similarity}\_\text{right}-\text{Similarity}\_\text{root}
$$
3. Repeat Step 2 till limit. Prune branches bottom-up by checking whether the gain of the branch is higher than a predefined threshold $\gamma$. If it is higher, stop. If it is lower, prune it, move on to the next branch.
4. Define the output value for each leaf of this tree.
$$
\text{Output}=\frac{\sum_{i=1}^{m_r}{r_i}}{m_r+\lambda}
$$
5. Define the predicted value for each sample.
$$
\hat{y}_i=\bar{y}+\eta\sum{\text{Tree}(\mathbf{x}_i)} 
$$
6. Repeat Steps 1-5 till limit.

Optimization (hyperparams):
- Regularization: $\lambda$ (the higher $\lambda$, the lower gain, thus easier to prune.)
- Prune threshold: $\gamma$ ($\gamma=0$ prunes negative gains.)
- Learning rate: $\eta$

Pros:
- Perform well when #features is small
- The Pros of Gradient Boosting

Cons:
- Cannot handle categorical features (must do encoding)
- Bad performance on sparse and unstructured data
- The Cons of Gradient Boosting

## LightGBM

Idea: Gradient Boosting + GOSS + EFB

Background:
- **GOSS (Gradient-based One-Side Sampling)**: focus more on under-trained samples without changing the original data distribution.
    1. Sort all samples based on abs(gradient). Select top $\alpha$% samples as the samples with large gradients. Keep them.
    2. Randomly sample $b$% of the remaining samples with small gradients. Amplify them with a constant $\frac{1-a}{b}$.
- **EFB (Exclusive Feature Bundling)**: bundle mutually exclusive features together into much fewer dense features by conflict rate (more nonzero values lead to higher probability of conflicts).
    1. Sort features based on conflicts in a descending order. Assign each to an existing bundle with a small conflict or create a new bundle.
    2. Merge exclusive features in the same bundle. If two features have joint ranges, add an offset value so that the two features can be merged into one range.
- **Leaf-wise tree growth**: choose the leaf with max delta loss to grow. (DFS)

Pros:
- Can handle categorical features
- Much faster and efficient training both in time and space
- Higher accuracy than most other boosting algorithms, especially on large datasets

Cons:
- Overfitting (Because of leaf-wise splitting, the tree can be much more complex) 
- Bad performance on small datasets

Gradient Boosting Comparisons:
<center>

|                      |                                       XGBoost                                       |                                         CatBoost                                        |                                   LightGBM                                  |
|:--------------------:|:-----------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------:|
|      Tree growth     |                          Asymmetric level-wise tree growth                          |                            Symmetric growth                             |                               Asymmetric leaf-wise tree growth                              |
|         Split        |                                 Pre-sort + Histogram                                |                                          Greedy                                         |                                  GOSS + EFB                                 |
|  Numerical features  |                                       Support                                       |                                         Support                                         |                                   Support                                   |
| Categorical features | Need external encoding into numerical features<br>Cannot interpret ordinal category |                          Support with default encoding methods                          |   Support with default encoding methods<br>Can interpret ordinal category   |
|     Text features    |                                          NO                                         | YES by converting them to numerical features<br>via bag-of-words, BM-25, or Naive Bayes |                                      NO                                     |
|    Missing values    |     Interpret as NaN<br>Assign to side that reduces loss the most in each split     |                          Interpret as NaN<br>Process as min/max                         | Interpret as NaN<br>Assign to side that reduces loss the most in each split |

</center>

# Online Learning


## Least Mean Squares

## Perceptron

# Generative Models

The core of generative models in comparison to discriminative models is that the generative model **GENERATES** samples, which many newbies like me overlooked at the very beginning.

Discriminative models predict label given sample features, but Generative models uses a completely different thinking process, where we
1) propose/calculate the prior of labels $P(Y)$, using relevant params for the prior distribution,
2) calculate the likelihood of the current combination of values of sample features given the label $P(X_1,\cdots,X_n|Y)$, using relevant params for the likelihood distribution,
3) calculate $P(X,Y)=P(Y)P(X_1,\cdots,X_n|Y)$ for generation;
    
    calculate $P(Y|X)\propto P(Y)P(X_1,\cdots,X_n|Y)$ for discrimination.

During training, we estimate the params which maximize the combination of prior distribution $\times$ likelihood distribution.

During inference, we directly use those params to either generate samples or compute label for the given sample.

The following models are already abandoned in practice because of neural nets, but the ideas behind them are still important for forming a deep understanding of ML.

## Naive Bayes

Assumption: Features are **conditionally independent** of each other given the label:
$$
P(x_{i1},\cdots,x_{in}|y_i=k)=\prod_{j=1}^{n}P(x_{ij}|y_i=k)
$$

Model/Algorithm:
1. Build a feature space (e.g., vocabulary) $\mathcal{V}$ from training set (where each feature (e.g., word) takes a YES/NO binary value in text classification)
2. Estimate $P(k)$ for all $k\in\\{1,\cdots,K\\}$:
$$
P(k)=\frac{\\#\\{Y=k\\}}{m}
$$
where $\\#\\{Y=k\\}$ is #samples of class $k$.
3. Estimate $P(X_j=v|k)$ for all $v\in\mathcal{V}(X_j)$ for all $X_j\in\mathcal{V}$:
$$\begin{align*}
&\text{MLE}: &&P(X_j=v|k)=\frac{\\#\\{X_j=v,Y=k\\}}{\\#\\{Y=k\\}}\\\\
&\text{MAP (Laplace Smoothing)}: &&P(X_j=v|k)=\frac{\\#\\{X_j=v,Y=k\\}+1}{\\#\\{Y=k\\}+V}
\end{align*}$$
where $\\#\\{X_j=v,Y=k\\}$ is #occurrences of feature-value pair $X_j=v$ in all samples of class $k$.

Inference:
$$
\hat{y}_i=\arg\max_k\prod\_{k\in\\{1,\cdots,K\\}}P(k)\prod\_{x\_{ij}\in\mathbf{x}_i}P(x\_{ij}|k)
$$

#params: 
- Formula: $P(a_1,\cdots,a_n|b_1,\cdots,b_m)$: $(\prod_{i=1}^{n}|a_i|-1)\prod_{i=1}^{m}|b_j|$
- Joint distribution if all binary: $P(x_1,\cdots,x_n|y)$: $(2^n-1)\cdot2$
- NB if all binary: $\prod_{j=1}^{n}P(x_j|y)$: $2n$ (a significant reduction in #params)
- NB if arbitrary: $\prod_{j=1}^{n}P(x_j|y)$: $O((|X|-1)n)$ (again, a significant reduction in #params)

Objective: 0-1

Optimization: none

Pros: 
- Easy, Fast, Simple
- Significant reduction in #params with CI assumption
- Robust to outliers and noisy data
- Low computational cost
- No overfitting (only overfit in a sense of never seeing a feature value in a test set)
- Scale invariant
- Can handle real-time prediction very easily
- Widely used in spam detection or text classification in general

Cons:
- Assumption of feature independence fails in real life
- Very low accuracy in terms of output probability estimates

## Latent Dirichlet Allocation

Usage: Topic Modeling

Background:
- Multinomial (like Binomial) distribution models the outcomes of a series of i.i.d. experiments (e.g., dice rolling).
- Dirichlet (like Beta) distribution models probability vectors.
- We do not know anything about the probabilities of each outcome at the beginning (i.e., the params of Multinomial distribution), so we use Dirichlet distribution to offer us a prior over these params.
- Therefore, Dirichlet (like Beta) distribution is a conjugate prior for Multinomial (like Binomial) distribution.

Model/Algorithm: For each document $d$,
1. Choose its topic distribution $\boldsymbol{\theta}_d\sim\text{Dirichlet}(\alpha)$, where $\theta\_{dk}=p(\text{topic}=k|\text{document}=d)$
2. For each word $w_j$ in $d$:
    1. Choose this word's topic $z_{dj}\sim\text{Multinomial}(\boldsymbol{\theta}_d)$
    2. Choose a word $w_j\sim\text{Multinomial}(\beta_{z_{dj}})$ where $\beta_{z_{dj}}=p(w_j|z_{dj})$

Objective: 0-1

Optimization: EM (Variational EM in practice)
- E-step: Compute $p(\boldsymbol{\theta},\textbf{z}|d;\alpha,\boldsymbol{\beta})$ (posterior of hidden vars $(\boldsymbol{\theta},\textbf{z})$ given each document $d$)
- M-step: Estimate params $(\alpha,\boldsymbol{\beta})$ given posterior estimates

Naive Bayes vs LDA:
- Naive Bayes assumes each doc is on a single topic.
- LDA allows each doc to be a mixture of topics (i.e., each word can be on a different topic).

Pros: 
- Discovery of implicit topics that were not explicit in the documents
- Applicable on semi-supervised learning

Cons:
- High computational cost (Param Estimation is a bit messy)
- Low interpretability

## Hidden Markov Model

Usage: Seq2Seq Synthesis (Speech recognition, POS Tagging, Named Entity Recognition, etc.)

Assumptions:
- Markov Assumption: $P(X_t|X_{t-1},\cdots,X_1)=P(X_t|X_{t-1})$
- Stationarity: Transition matrix and emission probabilities stay the same over time.

Model:
1. Start in some initial state $s_i$ with probability $p(s_i)=\pi$.
2. Move to a new state $s_j$ with probability $p(s_j|s_i)=a_{ij}$, where $a_{ij}$ is a cell value in transition matrix $A$.
3. Emit an observation $x_v$ with probability $p(x_v|s_i)=b_{iv}$, where $b_{iv}$ is a lookup value in emission probability function $B(x_v,s_i)$.

Optimization:
1. Evaluation: compute $P(X)$ given $X=[x_1,\cdots,x_T]$ and $(A,B,\pi)$.
2. Decoding: find the best $S=[s_1,\cdots,s_T]$ which best explains the observations given $X=[x_1,\cdots,x_T]$ and $(A,B,\pi)$.
3. Learning: estimate $(A,B,\pi)$ which maximize $P(X|A,B,\pi)$.

Pros:
- Can handle inputs of variable lengths
- Efficient learning
- Wild range of applications (until DL bloomed)

Cons:
- A large number of unstructured params
- Limited by Markov Assumption
- No dependency between hidden states
- Completely destroyed by DL


## Bayesian Network

Usage: compact specification of full joint distributions with CI assertions using Conditional Probability Table (CPT)

Background:
- CI properties:
    - Symmetry: $(X\perp Y|Z)\rightarrow(Y\perp X|Z)$
    - Decomposition: $(X\perp Y,W|Z)\rightarrow(X\perp Y|Z)$
    - Weak union: $(X\perp Y,W|Z)\rightarrow(X\perp Y|Z,W)$
    - Contraction: $(X\perp W|Y,Z),(X\perp Y|Z)\rightarrow(X\perp Y,W|Z)$
    - $P(X|Y)+P(X|\neg Y)\neq1$ (they are NOT related)
    - $P(X|Y)+P(\neg X|Y)=1$
- **Active Trail** (in an acyclic graph): for each consecutive triplet in the trail:
    - $X\rightarrow Y\rightarrow Z$ and $Y$ is NOT observed.
    - $X\leftarrow Y\rightarrow Z$ and $Y$ is NOT observed.
    - $X\rightarrow Y\leftarrow Z$ and $Y$ or one of its descendants IS observed.
- If there is NO active trail between $X$ and $Y$, then they are CI.
- D-separation: block a trail.
- NB = basic BayesNet with 1 parent (label) and multiple children (features)
- Complexity scales exponentially with #parents; Complexity scales linearly with #children.

Assumption: A variable $X$ is independent of its non-descendants given its parents (Local Markov Assumption)

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