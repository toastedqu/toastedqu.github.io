# ML Models

Common Notations:

- $m$: #samples
- $n$: #features
- $X=[\mathbf{x}_1,\cdots,\mathbf{x}_m]^T$: input matrix of shape $(m,n)$
- $\mathbf{y}=[y_1,\cdots,y_{m}]^T$: output vector of shape $(m,1)$


```python
### Imports ###
import math
import io
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
```


```python
### Data ###
from sklearn.datasets import make_blobs
X_train, y_train = make_blobs(n_samples=10000, n_features=10, centers=5, random_state=0)
```

# Supervised Learning

## Linear Models

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

### Linear Regression

Assumption:

$$\begin{align*}
&\text{Frequentist:} &&\mathbf{y}=X\mathbf{w}+\boldsymbol{\varepsilon},\varepsilon_i\sim N(0,\sigma^2) \\\\
&\text{Bayesian:}    &&\mathbf{y}=N(X\mathbf{w},\sigma^2)
\end{align*}$$

- **Linearity**: The underlying relationship between $\textbf{y}$ and $X$ is linear.
- **Independence**: $\varepsilon_i$ is independent of each other.
- **Normality**: $\varepsilon_i$ follows Gaussian distribution.
- **No Collinearity**: No/Minimal explanatory variables correlate with each other.
- **Homoskedasticity**: The variance of all noises is the same constant $\sigma^2$.

Model:

$$
\hat{y}_i=\mathbf{w}^T\mathbf{x}_i
$$

Loss: MSE (Mean Squared Error)

Minimizer:

$$\begin{aligned}
&\text{OLS/MLE}:\ &&\hat{\mathbf{w}}=(X^TX)^{-1}X^T\mathbf{y} \\\\
&\text{Ridge}:\ &&\hat{\mathbf{w}}=(X^TX+\lambda I)^{-1}X^T\mathbf{y} \\\\
&\text{Lasso}:\ &&\frac{\partial \mathcal{L}_B}{\partial w_j}=\frac{1}{B}\sum\_{i=1}\^{B}[x\_{ij}(\hat{y}_i-y_i)]+\lambda\text{Heavi}(w_j) \\\\
&\text{ElasticNet}:\ &&\frac{\partial \mathcal{L}_B}{\partial w_j}=\frac{1}{B}\sum\_{i=1}\^{B}[x\_{ij}(\hat{y}_i-y_i)]+\lambda_1\text{Heavi}(w_j)+2\lambda_2w_j
\end{aligned}$$

Pros:
- Scale invariant.
- Consistent and unbiased.

Cons:
- Sensitive to outliers.
- If any assumption fails, linear regression fails.

Time complexity:
- Train:
    - Exact Solution: $O(n^2(m+n))$
    - Gradient Descent: $O(mn)$
- Test: $O(n)$

#### Code


```python
def LinearRegression(X,y,intercept=False):
    if intercept: X = np.hstack(np.ones((X.shape[0],1)),X)
    return (np.linalg.inv(X.T @ X) @ X.T @ y).reshape(-1)
```


```python
###### Sklearn ######
from sklearn.linear_model import Lasso

reg = Lasso(alpha=0.01,fit_intercept=False,max_iter=1000)
reg.fit(X_train,y_train)

y_train_pred = reg.predict(X_train)
train_acc = accuracy_score(y_train_pred,y_train)
```

### L0 Regression

$$
\mathbf{y}=X\mathbf{w}+\boldsymbol{\varepsilon},\varepsilon_i\sim N(0,\sigma^2),w_j\sim\text{Spike and Slab}
$$

MAP:

$$\begin{align*}
\hat{\mathbf{w}}&=\arg\min_{\mathbf{w}}{||y_i-\textbf{x}_i\textbf{w}||_2^2+\lambda||\textbf{w}||_0}\\\\
\end{align*}$$

Solution: Search
- Streamwise Regression:
    1. Init model. Init $Err_0=||y_i||_2^2$.
    2. For $j$ in range(1,n+1):
        1. Add feature $\mathbf{x}_j$ to model.
        2. If $Err=||y_i-\sum_{j\in\text{model}}{w_{j}x_{ij}}||_2^2+\lambda||\textbf{w}_{\text{model}}||_0 <\ Err_{j-1}$:
            1. Keep $\mathbf{x}_j$.
            2. $Err_j = Err$
        3. Else:
            1. $Err_j = Err_{j-1}$
    - Pros: low cost: $O(n)$.
    - Cons: no guarantee to find optimal solution. order of features matters.

- Stepwise Regression:
    1. Init model. Init $Err_0=||y_i||_2^2$.
    2. While True (n loops):
        1. Try to add each of all remaining features $\mathbf{x}_k$ to model.
        2. Pick the feature with $Err=\min(||y_i-\sum_{j\in\text{model}}{w_{j}x_{ij}}||_2^2+\lambda||\textbf{w}_\text{model}||_0$:
        3. If $Err<Err_0$:
            1. Add feature to model.
            2. $Err_0 = Err$
        4. Else:
            1. Break
    - Pros: much more likely to find optimal solution.
    - Cons: high cost: $O(mn)$. overfitting. multicollinearity.

- Stagewise Regression:
    1. Init model. Init $Err_0=||y_i||_2^2$. Init cache $\textbf{w}$.
    2. While True (n loops):
        1. Try to add each of all remaining features $\mathbf{x}_k$ to model.
        2. Pick the feature with $Err=\min(||r_i-w_{k}x_{ik}||_2^2+\lambda||\textbf{w}||_0$, where $r_i=y_i-\sum_{j\in\text{model}}{w_jx_{ij}}$:
        3. If $Err<Err_0$:
            1. Add feature to model. Add $w_{k}$ to cache.
            2. $Err_0 = Err$
        4. Else:
            1. Break
    - Pros: faster than stepwise regression because of no need to create new long models each time. no multicollinearity. used for boosting.
    - Cons: high cost: $O(mn)$.


Pros:
- Sparse & Feature selection: ignore trash features.
- Can handle $n>>m$ cases, while L2 cannot.
- Stagewise Regression can reduce overfitting effectively.

Cons:
- Inconsistent and biased.
- Very weak for reducing overfitting.
- No guarantee to find optimal solution. (Not convex)
- No weight reduction.
- Scale variant.
- High computational cost. (Search algorithm)
- Need to select perfect $\lambda=\left(\frac{\sigma^2}{\gamma^2}\right)$.

Idea: use link functions to do nonlinear regression, where $\mathbf{w}^TX$ becomes the logits.

$$
\hat{y_i}=f\left(\sum_{j=1}^{n}{w_jx_{ij}}\right)
$$

### Logistic Regression

Binary:
$$
P(y_i=1|\mathbf{x}_i,\mathbf{w})=\sigma(\mathbf{x}_i\mathbf{w})=\frac{1}{1+\exp{(-\mathbf{w}^T\mathbf{x}_i)}}=\frac{\exp{(\mathbf{w}^T\mathbf{x}_i)}}{1+\exp{(\mathbf{w}^T\mathbf{x}_i)}}
$$

- Loss (MLE):
$$
\mathcal{L}=\sum_{(\textbf{x}_i,y_i)\in D}{\left(-y_i\log{\sigma(\mathbf{x}_i\mathbf{w})}-(1-y)\log{(1-\sigma(\mathbf{x}_i\mathbf{w}))}\right)}
$$

- Loss (MAP, L2 penalty):
$$
\mathcal{L}=\mathcal{L}_{MLE}+\frac{1}{2\gamma^2}||\mathbf{w}||_2^2
$$

Multi:
$$
P(y_i=k|\mathbf{x}_i,\mathbf{w})=\text{softmax}(\mathbf{x}_i\mathbf{w})=\frac{\exp{(\mathbf{w}_k^T\mathbf{x}_i)}}{\sum_{k=1}^{K}{\exp{(\mathbf{w}_k^T\mathbf{x}_i)}}}
$$

- Loss (Cross-Entropy, MLE):
$$
\mathcal{L}=\sum_{(\textbf{x}_i,y_i)\in D}{\sum_{k=1}^{K}{\textbf{1}[y_i=k]\log{\frac{\exp{(\mathbf{w}_k^T\mathbf{x}_i)}}{\sum_{k=1}^{K}{\exp{(\mathbf{w}_k^T\mathbf{x}_i)}}}}}}
$$


Solution: Gradient Descent

Pros:
- Easy multiclass, which is equivalent to binary.
- Coefficient resemble feature importance.
- Easy gradient calculation.

Cons:
- Named regression but can only work for discrete classification.
- Cannot work when $n>>m$.
- Assume linearity by log odds. Cannot work for nonlinear cases.

Time Complexity:
- Train: $O(mn)$
- Test: $O(n)$

Space Complexity: $O(n)$


```python
###### Sklearn ######
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(penalty="l2",fit_intercept=False,)
clf.fit(X_train,y_train)

y_train_pred = clf.predict(X_train)
train_acc = accuracy_score(y_train_pred,y_train)
```

### Radial Basis Function

$$
\phi_j(\mathbf{x})=\exp{\left(-\frac{||\mathbf{x}-\mu_j||_2^2}{c}\right)}
$$

Steps:
1. Cluster points $\mu_j$ with k-means clustering.
2. Pick a width $c=2\sigma^2$ for all the Gaussian pdfs $N(\mu_j,\sigma^2)$ at each cluster.
3. Fit a linear regression.

Usage:
- $d<n$: dimensionality reduction
- $d>n$: convert nonlinear problem to linear
- $d=n$: switch to a dual representation

Pros:

Cons:
- Scale variant.
- Need to find perfect $c$. Low $c$ leads to overfitting. High $c$ leads to learning nothing (different centroids may cover each other, which is horrible).

## Local Learning

### K Nearest Neighbors

Idea: use the majority label from nearest neighbors as predicted label.

Algorithm:
1. Calculate distance between sample point and every training point.
2. Find the $K$ nearest neighbors with minimal distances.
3. Take the majority vote and output it as the label for the sample point.

Norm:
- Properties:
    - $L_p(\textbf{x})=0\leftrightarrow \textbf{x}=\textbf{0}$
    - $L_p(\textbf{x}+\textbf{y})\leq L_p(\textbf{x})+L_p(\textbf{y})$
    - $L_p(c\textbf{x})=|c|L_p(\textbf{x})\ \ \ \forall c\in\mathbb{R}$
- Types:
    - $L_p$: $||\textbf{x}||_p=\left(\sum_i{|x_i|^p}\right)^\frac{1}{p}$
    - $L_0$: $||\textbf{x}||_0=\#x_i:x_i>0,x_i\in\textbf{x}$ 
    - $L_1$: $||\textbf{x}||_1=\sum_i{|x_i|}$
    - $L_2$: $||\textbf{x}||_2=\sqrt{\sum_i{(x_i)^2}}$
    - $L_\infty$: $||\textbf{x}||_\infty=\max{\{|x_i|:x_i\in\textbf{x}\}}$

Pros:
- Non-parametric.
- No training: instance-based learning (i.e., lazy learner).
- Seamless data augmentation: at any step.
- Easy to implement. (no model, just method.)

Cons:
- Scale variant.
- Bad for large dataset: high computational cost.
- Bad for high dimension: high computational cost and low variance in distance measure.
- Sensitive to noisy data, missing values, and outliers.

Time Complexity:
- Train: n/a
- Test: $O(kmn)$

Space Complexity: $O(mn)$

##### Code:


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


```python
###### Sklearn ######
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=5,
                           weights='uniform',   # uniform or distance weighting
                           p=2,                 # Lp distance
                           metric='minkowski')  # distance metric
clf.fit(X_train,y_train)

y_train_pred = clf.predict(X_train)
train_acc = accuracy_score(y_train_pred,y_train)
```

### Kernel Methods

Kernel Regression:

$$\begin{align*}
\text{Regression}: &\hat{y}=\frac{\sum_{i=1}^{m}{k(\mathbf{x},\mathbf{x}_i)y_i}}{\sum_{i=1}^{m}{k(\mathbf{x},\mathbf{x}_i)}}\\\\
\text{Binary Classification}: &\hat{y}=\text{sign}(\sum_{i=1}^{m}{k(\mathbf{x},\mathbf{x}_i)y_i})
\end{align*}$$

- Steps:
    1. Calculate kernel function (similarity) between input sample feature and training data features.
    2. Estimate $y$ with the above formulae.
    3. Use cross validation to tune hyperparameters (kernel width in most cases).

Kernel:
- Def: measure of similarity between 2 vectors.
- $\mathbf{K}=k(\mathbf{x},\mathbf{x}')$ is positive semi-definite.
    - PSD: $\mathbf{K}=\sum_{i=1}^{m}{\lambda_i\mathbf{z}_i\mathbf{z}_i^T}$.
    - $\lambda_i$: non-negative real eigenvalues.
    - $\mathbf{z}_i$: real eigenvectors.
- $k(\mathbf{x},\mathbf{x}')=ck_1(\mathbf{x},\mathbf{x}'),c>0$
- $k(\mathbf{x},\mathbf{x}')=k_1(\mathbf{x},\mathbf{x}')+k_2(\mathbf{x},\mathbf{x}')$
- $k(\mathbf{x},\mathbf{x}')=k_1(\mathbf{x},\mathbf{x}')+k_2(\mathbf{x},\mathbf{x}')$
- $k(\mathbf{x},\mathbf{x}')=q(k_1(\mathbf{x},\mathbf{x}'))$, where $q(\cdot)$ is polynomial func with positive coeffs.
- $k(\mathbf{x},\mathbf{x}')=f(\mathbf{x})k_1(\mathbf{x},\mathbf{x}')f(\mathbf{x}')$
- $k(\mathbf{x},\mathbf{x}')=\mathbf{x}^T\mathbf{A}\mathbf{x}'$
- $k(\mathbf{x},\mathbf{x}')=\phi(\mathbf{x})^T\phi(\mathbf{x}')$

Pros:
- Save computation cost (for SVM) since no need to actually convert features to higher-dimensional data to find nonlinear patterns.
- Easy to test whether a proposed kernel is valid by finding arbitrary 2 points with negative determinant (i.e., negative eigenvalue).
- Can be extended on literally any data type.

Cons:
- Hard to choose the suitable kernel.
- Hard to comprehend what kernels learned exactly.
- Easy overfitting.

### Summary

<center>

|               KNN              |                   Kernel                  |
|:------------------------------:|:-----------------------------------------:|
|         distance metric        |              kernel function              |
|         $K$ neighbors          |               all neighbors               |
| same impact from all neighbors | weighted impact favoring closer neighbors |
|          Scale variant         |               Scale variant               |

</center>

## Decision Tree

Idea: build a tree where each node is a feature split to classify data points into different leaf outputs.

Algorithm:
1. Calculate info gain for each feature.
2. Select the feature that maximizes info gain as the decision threshold.
3. Split data based on the decision. Repeat Step 1-2 until stop.

Information gain:
$$
IG(Y|X)=H(Y)-H(Y|X)
$$
- $X,Y$: random vars.
- $H(\cdot)$: Impurity measure:
    - Gini: $H(Y)=\sum_{y}{p_y(1-p_y)}$
    - Entropy: $H(Y)=-\sum_{y}{p_y\log_2{p_y}}$
        - (Average) Conditional entropy: $H(Y|X)=\sum_{x}{P(X=x)H(Y|X=x)}$
        - Specific conditional entropy: $H(Y|X=x)=-\sum_{y}{P(Y=y|X=x)\log_2{P(Y=y|X=x)}}$
            - $y\in\mathcal{Y}$: a possible value for $Y$.
            - $\mathcal{Y}$: set of all possible values for $Y$.
            - $x\in\mathcal{X}$: a possible value for $X$.
            - $\mathcal{X}$: set of all possible values for $X$
    

Pros:
- Non-parametric.
- No feature scaling required. (i.e., scale invariant)
- Low computation cost for prediction: $O(\log{m})$ (if balanced binary tree).
- Can handle multi-class classification.
- Easy to interpret and validate. Easy model visualization.
<!-- - Good performance in practice even if assumptions are violated by the data generator model. -->

Cons:
- Easy overfitting, especially when $n$ is large.
- Sensitive to noisy data, missing value, and outliers.
- Discrete predictions only.
- Difficult to find globally optimal decisions. Current modules only support locally optimal decisions at each node.
- High computation cost for training: $O(mn\log{m})$
- Inaccurate in practice.

Time complexity:
- Train: $O(mn\log{m})$
- Test: $O(d)$, where $d=$ depth. (Ideally $O(\log{m})$)

#### Code


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


```python
###### Sklearn #######
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion="entropy",       # impurity measure
                             splitter="best",           # best or random
                             max_depth=None,
                             min_samples_split=2,       # min #samples required to split a node
                             min_samples_leaf=1,        # min #samples required at each leaf
                             max_leaf_nodes=None,
                             min_impurity_decrease=0.0)  # min impurity decrease required to split a node
clf.fit(X_train,y_train)

y_train_pred = clf.predict(X_train)
train_acc = accuracy_score(y_train_pred,y_train)
```

## Support Vector Machine

Separable Primal:
$$\begin{align*}
\min_{\mathbf{w},b}\ &{\frac{1}{2}||\textbf{w}||_2} \\\\
\text{s.t.}\ &y_i(\textbf{w}^T\textbf{x}_i+b)\geq 1\\\\ 
\end{align*}$$

Separable Dual:


Hypothesis:
$$
h_w(x)=\begin{cases}
1 & \text{if}\ \mathbf{w}^T\mathbf{x}_i\geq0 \\\\
0 & \text{if}\ \mathbf{w}^T\mathbf{x}_i<0 \\\\
\end{cases}
$$

Loss: Hinge Loss

Terminology:
- Support vector: points closest to the boundary.

Steps:
1. Use kernel trick to calculate higher-dimensional relationships for all pairs of observations in the training set.
2. Determine a hyperplane decision boundary and its soft margin for the (pseudo-)high-dimensional data points with cross validation.

Pros:
- Can handle outliers.
- Can handle overlapping classes (multiple classes share same parts of feature values).
- Guaranteed convexity. Easy to optimize.
- Work well with high-dimensional data.
- Low memory cost (only support vectors matter)

Cons:
- High time cost, especially with Kernels when data set is too large.
- Low interpretability: no probability estimates are given.
- Bad performance when $n>>m$.
- Bad performance on large datasets (i.e., $m>>0$).
- Bad performance with noisy data (when target classes overlap).

Time Complexity:
- Train: $O(m^2)$
- Test: $O(kn)$, where $k=$ #support vectors.

## Ensemble Methods

Terminology:

- **Bagging**: create a bootstrapped dataset by randomly select samples with replacement from training set.
- **Boosting**: train multiple models sequentially and dependently to improve accuracy.

### Random Forest

Training:
1. Create a bootstrapped dataset by randomly select samples (with replacement) from training set.
2. Create a full decision tree (no pruning) on a randomly selected subset of features of the bootstrapped dataset. (size determined by out-of-bag error in step 4.)
3. Repeat 1-2 to create a random forest till #tree limit.
4. Use out-of-bag samples to determine the accuracy of each tree.

Prediction:
1. Take a majority vote / average vote from all trees.

Pros:
- Reduce overfitting in decision tree $\rightarrow$ much more accurate than a single decision tree.
- Flexible to both categorical and continuous outputs.
- Scale invariant.
- Automatically handle missing values by dropping or filling with median/mode.
- Robust of outliers and noisy data.
- Best used in banking and healthcare.

Cons:
- High computation cost.
- High training time.
- Low interpretability.

Time Complexity:
- Train: $O(kmn\log{m})$, where $k=$ #trees
- Test: $O(kd)$, where $d=$ max depth


```python
###### Sklearn ######
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10,
                             criterion="entropy",
                             max_depth=None,
                             min_samples_split=2,
                             min_samples_leaf=1,
                             max_leaf_nodes=None,
                             min_impurity_decrease=0.0,
                             bootstrap=True,    # if false, use full dataset
                             max_samples=0.5)   # fraction of data when bootstrap is true
clf.fit(X_train,y_train)

y_train_pred = clf.predict(X_train)
train_acc = accuracy_score(y_train_pred,y_train)
```

### AdaBoost

Idea: train a bunch of stumps (weak learners) sequentially and take a weighted majority vote.

Training:
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

Prediction:
1. Take a weighted majority vote using the Amount of Say from each stump.

Random Forest vs AdaBoost

<center>

|         Random Forest         |                 AdaBoost                 |
|:-----------------------------:|:----------------------------------------:|
|   No tree shape requirement   | Each tree is a stump (1 node + 2 leaves) |
| Each tree has equal influence |      Stumps have weighted influence      |
|    Each tree is independent   |  Each stump is dependent of its previous |

</center>

Pros:
- Reduce overfitting because parameters are not optimized jointly but stagewise.
- Much fewer hyperparameters than most other algorithms.
- Boosting converges exponentially with #iterations.

Cons:
- Extremely sensitive to outliers and noisy data.
- Slower than XGBoost.
- No bagging.


```python
###### Sklearn ######
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(base_estimator=None,   # base model. default decision tree with max_depth=1 (i.e., stump)
                         n_estimators=50,
                         learning_rate=1.0)     # a higher learning rate sequentially increase the contribution of each stump.
clf.fit(X_train,y_train)

y_train_pred = clf.predict(X_train)
train_acc = accuracy_score(y_train_pred,y_train)
```

### Gradient Boosting

Idea: train a bunch of fixed-size trees to fit residuals sequentially. take a weighted majority vote.

Training:
1. Init a leaf as the average of $\mathbf{y}$. Calculate residuals.
2. Train a tree (#leaves < $m$) on the residuals. Scale it with a fixed learning rate.
3. Combine leaf (and previous trees) and current tree to make new predictions on the same data. Calculate new residuals.
4. Repeat Steps 2-3 till limit.

Essential Hyperparameters:
- #stages: $T$
- bag size (fraction): $f$
- learning rate: $\eta$
- tree depth: $d$

AdaBoost vs Gradient Boosting

<center>

|         AdaBoost         |       Gradient Boosting       |
|:------------------------:|:-----------------------------:|
|     Init with a stump    | Init with a leaf of $\bar{y}$ |
|          Stumps          |        Fixed-size trees       |
| Scale stumps differently |      Scale trees equally      |
| Train on $y$ |      Train on residuals      |

</center>

Pros:
- Very accurate.
- Fast prediction.
- High flexibility: multiple hyperparameters to tune, multiple loss functions to use.
- Can handle missing data.
- Reduce overfitting by using a small learning rate and bagging.

Cons:
- Highly sensitive to outliers and noisy data.
- May cause overfitting by overemphasizing outliers and noisy data.
- High computation cost.
- Low interpretability.
- Require a large grid search for hyperparameter tuning.
- Longer training due to sequential building.

#### Code


```python
from sklearn.ensemble import GradientBoostingClassifier
gbrt = GradientBoostingClassifier(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)
gbrt.fit(X_train, y_train)

y_train_pred = gbrt.predict(X_train)
train_acc = accuracy_score(y_train_pred,y_train)
```

### XGBoost

Idea: use a unique tree to make decisions based on similarity scores.

Training:
1. Calculate residuals for all samples based on current prediction. Calculate similarity score for the root node of a new tree.
$$
\text{Similarity}=\frac{\sum_{i=1}^{m_r}{r_i^2}}{m_r+\lambda}
$$
2. Find similarity gain of each possible split. Choose the split with the max gain at root node.
$$
\text{Gain}=\text{Similarity}_\text{left}+\text{Similarity}_\text{right}-\text{Similarity}_\text{root}
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

Essential Hyperparameters:
- Regularization: $\lambda$ (the higher $\lambda$, the lower gain, thus easier to prune.)
- Prune threshold: $\gamma$ ($\gamma=0$ prunes negative gains.)
- Learning rate: $\eta$

Key points:
- Pre-sort-based algorithm: sort samples by the feature value, then split linearly.
- Histogram-based algorithm: bucket continuous feature values into discrete bins.
- Level-wise tree growth (BFS)

Pros:
- Perform well when #features is small.

Cons:
- Cannot handle categorical features (must do encoding)
- Perform poor on sparse and unstructured data.
- Sensitive to outliers and noisy data.

#### Code


```python
from xgboost import XGBClassifier
xgb = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100,
                    silent=True, objective='binary:logistic', booster='gbtree')
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])

y_train_pred = xgb.predict(X_train)
train_acc = accuracy_score(y_train_pred,y_train)
```

### LightGBM

Idea:
- GOSS (Gradient-based One-Side Sampling): focus more on under-trained samples without changing the original data distribution.
    1. Sort all samples based on abs(gradient). Select top $\alpha$% samples as the samples with large gradients. Keep them.
    2. Randomly sample $b$% of the remaining samples with small gradients. Amplify them with a constant $\frac{1-a}{b}$.
- EFB (Exclusive Feature Bundling): bundle mutually exclusive features together into much fewer dense features by conflict rate (more nonzero values lead to higher probability of conflicts).
    1. Sort features based on conflicts in a descending order. Assign each to an existing bundle with a small conflict or create a new bundle.
    2. Merge exclusive features in the same bundle. If two features have joint ranges, add an offset value so that the two features can be merged into one range.

Key points:
- Leaf-wise tree growth: choose the leaf with max delta loss to grow. (DFS)

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

Pros:
- Can handle categorical features by taking the input of feature names.
- Much faster and efficient training both in time and space.
- High accuracy than most other boosting algorithms, especially on large datasets.

Cons:
- Overfitting: because of leaf-wise splitting, the tree can be much more complex. Thus poor performance on small dataset.

#### Code


```python
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier()
lgbm.fit(X_train,y_train)

y_train_pred = lgbm.predict(X_train)
train_acc = accuracy_score(y_train_pred,y_train)
```

### Naive Bayes

Pros: 
- Easy, Fast, Simple
- Robust to outliers and noisy data.
- Low computation cost
- No overfitting
- Scale invariant
- Much more efficient than SVM or neural networks on Large datasets.
- Can handle real-time prediction very easily.

Cons:
- Assumption of feature independence fails in real life.
- Very low accuracy in terms of output probability estimates.

# Unsupervised Learning

## Dimensionality Reduction

### Principal Component Analysis

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

### Independent Component Analysis

## Clustering

### K-Means

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

##### Code:


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

### Gaussian Mixture Model

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


