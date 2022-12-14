---
title : "Concepts"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: false
images: []
weight: 100
---
# Components of ML
1. **Feature Representation**
2. **Model Architecture**
    - Linear vs Nonlinear
    - Parametric vs Non-parametric
    - Discriminative vs Generative
3. **Objective Function**
    - Loss function
    - Regularization
4. **Optimization Method**
    - Parameter estimation
    - Hyperparameter tuning
    - Model selection

# MLE & MAP
- Minimize loss = Maximize likelihood
- Regularization = Adding prior belief
- MAP = MLE + Prior
<!-- 
## Types

**Supervised vs Unsupervised**:

<center>

|             Supervised            |                                Unsupervised                                |
|:---------------------------------:|:--------------------------------------------------------------------------:|
|              labeled              |                                  unlabeled                                 |
|             train/test split            |                                  full data                                 |
|             prediction            |                                  analysis                                  |
| classification<br>+<br>regression | clustering<br>+<br>dimentionality reduction |

</center>

**Discriminative vs Generative**:

<center>

|             Discriminative            |                                Generative                                |
|:---------------------------------:|:--------------------------------------------------------------------------:|
|              estimate $P(Y\|X)$ as some functional form given $D$             |          estimate $P(Y\|X)=\frac{P(Y)P(X\|Y)}{P(X)}$             |
|             separate classes            |               separate classes + modeling data points + generate data points                                 |
|             robust to outliers in general            |          sensitive to outliers in general                 |
|  frequentist | Bayesian |
| more suitable for supervised | more suitable for unsupervised |
| computationally cheap | computationally expensive |
| need more data to train | need less data to train due to strong prior as bias |
| less friendly with missing data | easy marginalization over missing data |
| more accurate in general | less accurate in general due to violation of conditional independence |

</center>

## MLE/MAP

MLE: maximize likelihood

\begin{align*}
\hat{\mathbf{w}}&=\arg\max_{\mathbf{w}}{P(D|\mathbf{w})}\\
&=\arg\max_{\mathbf{w}}{\prod_{(\textbf{x}_i,y_i)\in D}{P(y_i|\textbf{x}_i,\textbf{w})}}\\
&=\arg\max_{\mathbf{w}}{\sum_{(\textbf{x}_i,y_i)\in D}{\log{P(y_i|\textbf{x}_i,\textbf{w})}}}\\
\end{align*}

MAP: maximize posterior

\begin{align*}
\hat{\mathbf{w}}&=\arg\max_{\mathbf{w}}{P(\mathbf{w}|D)}\\
&=\arg\max_{\mathbf{w}}{\prod_{(\textbf{x}_i,y_i)\in D}{P(y_i|\textbf{x}_i,\textbf{w})P(\textbf{w})}}\\
&=\arg\max_{\mathbf{w}}{\sum_{(\textbf{x}_i,y_i)\in D}{\log{P(y_i|\textbf{x}_i,\textbf{w})}+\log{P(\textbf{w})}}}\\
\end{align*}

Why MAP not MLE?
- MAP allows us to use existing domain knowledge (via prior)
- MAP prevents overfitting of parameters to data

Why log-likelihood?
- log makes optimization much easier
- log prevents value overflow

## Bias-Variance Trade-off

\begin{align*}
&\text{Bias}(\hat{w})=E[\hat{w}-w]=E[\hat{w}]-E[w]\\
&\text{Var}(\hat{w})=E[(\hat{w}-E[\hat{w}])^2]\\
\end{align*}

**Bias**: how much our average model predictions differ from ground truth over different training sets. (i.e., model predictive power)
- High bias: oversimplified model; underfitting; high error on train & test.
- Models with Low bias: KNN, Decision Tree, SVM
- Models with High bias: Linear models
- Validation set error = unbiased estimator of true error.

**Variance**: how much our estimates change due to changes in training data. (i.e., model sensitivity)
- High variance: overcomplex model; overfitting; low error on train & high error on test.

**Trade-off**: when you have low/high bias, it is inevitable to have high/low variance.

- Test error = Variance + Bias^2 + Noise

    \begin{equation*}
    E_{x,y,D}[(\hat{y}(x;D)-y)^2]=E_{x,D}[(\hat{y}(x;D)-\bar{\hat{y}}(x))^2]+E_{x}[(\bar{\hat{y}}(x)-\bar{y}(x))^2]+E_{x,y}[(\bar{y}(x)-y)^2]
    \end{equation*}

    - $D$: training set
    - $x,y$: test set

-  Graph:
<center>
<img src="/images/ml_concepts/bv_tradeoff.jpg" width="500"/>
</center>

## Hyperparamater Tuning

### Cross Validation

CV: evaluate how the outcomes will generalize to independent datasets.

## Confusion Matrix

**Confusion Matrix**: a 2x2 matrix which compares actual classes against predicted classes to see model performance.

<center>

|       |  P | N  |
|:-----:|:--:|----|
| **P** | TP | FN |
| **N** | FP | TN |

</center>

- Notations:
    - T&F = whether prediction matches actual
    - P&N = predicted class
    - FP = type 1 error
    - FN = type 2 error
- Metrics:
    - **Error Rate**: $\frac{FP+FN}{P+N}$
    - **Accuracy**: $\frac{TP+TN}{P+N}$
    - **Specificity**: TN rate among negative actual values: $\frac{TN}{FP+TN}$
    - **1-Specificity (FPR)**: FP rate among negative actual values: $\frac{FP}{FP+TN}$
    - **Precision**: TP rate among positive predicted values: $\frac{TP}{TP+FP}$
    - **Recall**: TP rate among positive actual values: $\frac{TP}{TP+FN}$
    - **F-score**: harmonic mean of Precision and Recall: $\frac{(1+\beta^2)\cdot\text{precision}\cdot\text{recall}}{(\beta^2\cdot\text{precision})+\text{recall}}$
    - **F1-score**: $\frac{2\cdot\text{precision}\cdot\text{recall}}{\text{precision}+\text{recall}}$
- **ROC** (Receiver Operating Curve): plot of TPR vs FPR
- **AUC** (Area Under Curve): area under ROC curve

## Vanishing/Exploding Gradient

**Gradient**: $\frac{\partial\mathcal{L}}{\partial w}$, specifically on $w$.

**Vanishing**: When backprop towards input layer, the gradients get smaller and smaller and approach zero which eventually leaves the weights of the front layers nearly unchanged. $\rightarrow$ gradient descent never converges to optimum.
- Causes:
    - Sigmoid or similar activation funcs. They have 0 gradient when abs(input) is large enough.
    - Gradients at the back are consistently less than 1.0. Therefore the chain reaction approaches 0.
- Symptoms:
    - Params at the back change a lot, while params at the front barely change.
    - Some model weights become 0.
    - The model learns very slowly, and training stagnate at very early iterations.

**Exploding**: in some cases, gradients get larger and larger and eventually causes very large weight updates to the front layers $\rightarrow$ gradient descent diverges.
- Causes:
    - Bad weight initialization. They cause large loss and therefore large gradients.
    - Gradients at the back are consistently larger than 1.0. Therefore the chain reaction approaches $\infty$.
- Symptoms:
    - Params grow exponentially.
    - Some model weights become NaN.
    - The model learns crazily, and the changes in params/loss make no sense.

Solutions:
- Proper Weight Inits (e.g., Xavier, Glorot, He.)
    - All layer outputs should have equal variance as input samples.
    - All gradients should have equal variance.
- Proper Activation Funcs (e.g., ReLU, LReLU, ELU, SELU, etc.)
    - Gradient = 1 for positive inputs.
- Batch Normalization
    - normalize inputs to ideally $N(0,1)$ before passing them to the layer.
- Gradient Clipping
    - Clip gradient with max & min thresholds. Any value beyond will be clipped back to the threshold.


<center>

| Model | Type | Accuracy | Speed (train) | Speed (test) | Interpretability | Scale Invariant | 
|:-|:-:|:-:|:-:|:-:|:-:|:-:|
| Linear Regression | Regression | NO | YES | YES | YES | YES w/o regularization<br>NO with regularization |
| Logistic Regression | Classification | NO | YES | YES | YES | YES w/o regularization<br>NO with regularization |
| Naive Bayes | Classification | NO | YES | YES | YES | YES |
| K-Nearest Neighbors | Both | NO | - | YES on small dataset<br>NO on large dataset | YES | NO |
| Decision Tree | Both | NO | YES | YES | YES | YES |
| Linear SVM | Both | NO | YES | YES | YES | NO |
| Kernel SVM | Both | YES | YES on small dataset<br>NO on large dataset | NO | NO | NO |
| Random Forest | Both | YES | NO | NO | NO | YES |
| Boosting | Both | YES | NO | NO | NO | YES |
| Neural Networks | Both | YES | NO | NO | NO | NO |

</center> -->

# Distance and Similarity

## Norm
Properties:
- $L_p(\textbf{x})=0\leftrightarrow \textbf{x}=\textbf{0}$
- $L_p(\textbf{x}+\textbf{y})\leq L_p(\textbf{x})+L_p(\textbf{y})$
- $L_p(c\textbf{x})=|c|L_p(\textbf{x})\ \ \ \forall c\in\mathbb{R}$

Types:
- $L_p$: $||\textbf{x}||_p=\left(\sum_i{|x_i|^p}\right)^\frac{1}{p}$
- $L_0$: $||\textbf{x}||_0=\\#x_i:x_i>0,x_i\in\textbf{x}$ 
- $L_1$: $||\textbf{x}||_1=\sum_i{|x_i|}$
- $L_2$: $||\textbf{x}||_2=\sqrt{\sum_i{(x_i)^2}}$
- $L_\infty$: $||\textbf{x}||_\infty=\max{\{|x_i|:x_i\in\textbf{x}\}}$

## Kernel
Def: measure of similarity between 2 vectors.

Properties:
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

# Loss

Loss is a measure of difference between predicted output $\hat{y}_i$ and actual output $y_i$.

Unreduced loss:

$$
\mathcal{L}=[\mathcal{l}_1,\cdots,\mathcal{l}_m]
$$

Reduced loss:

$$
\mathcal{L}(\mathbf{y},\mathbf{\hat{y}})=\begin{cases}
\text{sum}(\mathcal{L})=\sum_{i=1}^{m}l_i\\\\
\text{mean}(\mathcal{L})=\frac{1}{m}\sum_{i=1}^{m}l_i
\end{cases}
$$

## Regression

### L1 (Absolute Error)

$$
\mathcal{l}_i=|y_i-\hat{y}_i|
$$

Pros:
- Robust to outliers

Cons:
- Non-differentiable
- Less related to the underlying assumption of Gaussian distribution on the errors (for most problems)

### L2 (Squared Error)

$$
\mathcal{l}_i=(y_i-\hat{y}_i)^2
$$

Pros:
- Penalize large errors more heavily
- Differentiable
- Closely related to the underlying assumption of Gaussian distribution on the errors (for most problems)

Cons:
- Sensitive to outliers

### Huber

$$
l_i(\delta)=\begin{cases}
\frac{1}{2}(y_i-\hat{y_i})^2 & \text{if }|y_i-\hat{y_i}|\leq\delta\\\\
\delta|y_i-\hat{y_i}|-\frac{1}{2}\delta^2 & \text{if }|y_i-\hat{y_i}|>\delta
\end{cases}
$$

## Classification

### 0-1 (Heaviside)

$$
l_i=\begin{cases}
0 & \text{if }y_i\hat{y_i}>0\\\\
1 & \text{if }y_i\hat{y_i}<0
\end{cases}, y_i\in\\{-1,1\\}
$$

Usage: rare

Cons:
- Penalize all incorrect predictions equally
- Non-convex
- Non-differentiable


```python
def misclassification(v):
    return np.vectorize(lambda x: 0 if x>0 else 1)(v)
```

### Hinge

$$
l_i=\max{(0,1-y_i\hat{y}_i)}, y_i\in\\{-1,1\\}
$$

Usage: SVM

Pros:
- Allow misclassification (penalize more heavily for confident incorrect predictions than for uncertain predictions)
- No penalty on confident correct predictions
- Convex

Cons:
- Non-differentiable at 0

```python
def hinge_loss(v):
    return np.vectorize(lambda x: max(0,1-x))(v)
```

### Logistic

$$
l_i=\frac{1}{\log{2}}\log{(1+e^{-y_i\hat{y_i}})}, y_i\in\\{-1,1\\}
$$

Usage: LogReg

Pros:
- Convex
- Differentiable

Cons:
- Penalize confidently correct predictions
- Penalize confidently incorrect predictions less compared to other losses

```python
def logistic_loss(v):
    return 1/np.log(2)*np.log(1+np.exp(-v))
```

### Exponential

$$
l_i=e^{-y_i\hat{y}_i}, y_i\in\\{-1,1\\}
$$

Usage: AdaBoost

Pros:
- Convex
- Differentiable
- Much heavier penalty on much stronger misclassification (effective for weight assignments to examples for AdaBoost)

Cons:
- Penalize confidently correct predictions

```python
def exponential_loss(v):
    return np.exp(-v)
```

### Squared

$$
l_i=(1-y_i\hat{y}_i)^2, y_i\in\\{-1,1\\}
$$

Usage: rare

```python
def squared_loss(v):
    return np.square(1-v)
```

Visualization of all the above losses:
<!-- 

```python
import numpy as np
import matplotlib.pyplot as plt

v = np.linspace(-2,2,1000)
plt.plot(v, squared_loss(v), label="squared")
plt.plot(v, logistic_loss(v), label="logistic")
plt.plot(v, exponential_loss(v), label="exponential")
plt.plot(v, hinge_loss(v), label="hinge")
plt.plot(v, misclassification(v), label="misclassified")
plt.xlabel('$y_i\hat{y}_i$')
plt.ylabel('l_i')
plt.ylim(0,4)
plt.legend()
plt.show()
``` -->

<center>
<img src="/images/ml_concepts/loss.png" width="500"/>
</center>

### Cross Entropy

$$\begin{align*}
&\text{Binary}: &&\mathcal{l}_i=-[y_i\log{\hat{y}_i}+(1-y_i)\log{(1-\hat{y}_i)}], y_i\in\\{0,1\\} \\\\
&\text{Multiclass}: &&l_i=-\sum\_{k=1}^K\textbf{1}[y_i=k]\log{\left(\frac{\exp{(\hat{y}\_{ik})}}{\sum\_{c=1}^{K}{\exp{(\hat{y}\_{ic})}}}\right)}, y_i\in\\{0,\cdots,K\\}
\end{align*}$$

Usage: most (= Log loss for binary, with a much more prevalent label definition)

Pros:
- Extremely easy to compute (all terms become 0 except one)

Cons:
- Sensitive to imbalanced datasets

### Kullback-Leibler Divergence (Relative Entropy)

$$
l_i=y_i\log\frac{y_i}{\hat{y}_i}
$$

Usage: anywhere necessary to calculate difference between true and predicted probability distributions (i.e., information loss)

# Regularization

- Regularization reduces overfitting by adding our prior belief onto loss minimization (i.e., MAP estimate). Such prior belief is associated with our expectation of test error.
- Regularization makes a model inconsistent, biased, and scale variant.
- Regularization requires extra hyperparameter tuning.

## Explicit (Penalty)

### L1

$$\begin{align*}
&\text{Frequentist:}\ &&\lambda||\mathbf{w}||_1\\\\
&\text{Bayesian:}\    &&w_j\sim\text{Laplace}(0,\gamma^2)
\end{align*}$$

Pros:
- Sparse $\leftarrow$ Feature selection (reduce weights of trash features to 0)
- Robust to outliers
- Equal weight shrinkage ($\frac{\partial L_1}{\partial w_{j}}=1$)
- Effective with multicollinearity
- Able to handle $n>>m$ cases

Cons:
- No closed-form solution (high computational cost with LinReg)
- Need to select perfect $\lambda=\frac{\sigma^2}{\gamma^2}$

### L2

$$\begin{align*}
&\text{Frequentist:}\ &&\lambda||\mathbf{w}||_2^2\\\\
&\text{Bayesian:}\    &&w_j\sim N(0,\gamma^2)
\end{align*}$$

Pros:
- More/Less shrinkage on larger/smaller weights ($\frac{\partial L_2}{\partial w_{j}}=2w_{j}$)
- Effective with multicollinearity
- Closed-form solution. Low computational cost with LinReg

Cons:
- No sparsity $\leftarrow$ No feature selection
- Not robust to outliers
- Not able to handle $n>>m$ cases
- Need to select perfect $\lambda=\frac{\sigma^2}{\gamma^2}$

### L0

$$\begin{align*}
&\text{Frequentist:} &&\lambda||\mathbf{w}||_0\\\\
&\text{Bayesian:} &&w_j\sim\text{Spike and Slab}
\end{align*}$$

Optimization: Search
- **Streamwise Regression**:
    1. Init model. Init $Err_0=||y_i||_2^2$.
    2. For $j$ in range(1,n+1):
        1. Add feature $\mathbf{x}_j$ to model.
        2. If $Err=||y_i-\sum\_{j\in\text{model}}{w_{j}x\_{ij}}||_2^2+\lambda||\textbf{w}\_{\text{model}}||_0 <\ Err\_{j-1}$:
            1. Keep $\mathbf{x}_j$.
            2. $Err_j = Err$
        3. Else:
            1. $Err_j = Err_{j-1}$
    - Pros: low cost: $O(n)$
    - Cons: no guarantee to find optimal solution. order of features matters

- **Stepwise Regression**:
    1. Init model. Init $Err_0=||y_i||_2^2$.
    2. While True (n loops):
        1. Try to add each of all remaining features $\mathbf{x}_k$ to model.
        2. Pick the feature with $Err=\min(||y_i-\sum\_{j\in\text{model}}{w_{j}x\_{ij}}||_2^2+\lambda||\textbf{w}\_\text{model}||_0$:
        3. If $Err<Err_0$:
            1. Add feature to model.
            2. $Err_0 = Err$
        4. Else:
            1. Break
    - Pros: much more likely to find optimal solution
    - Cons: high cost: $O(mn)$. overfitting. multicollinearity

- **Stagewise Regression**:
    1. Init model. Init $Err_0=||y_i||_2^2$. Init cache $\textbf{w}$.
    2. While True (n loops):
        1. Try to add each of all remaining features $\mathbf{x}_k$ to model.
        2. Pick the feature with $Err=\min(||r_i-w_{k}x\_{ik}||_2^2+\lambda||\textbf{w}||_0)$, where $r_i=y_i-\sum\_{j\in\text{model}}{w_jx\_{ij}}$:
        3. If $Err<Err_0$:
            1. Add feature to model. Add $w_{k}$ to cache.
            2. $Err_0 = Err$
        4. Else:
            1. Break
    - Pros: faster than stepwise regression because of no need to create new long models each time. no multicollinearity. used for boosting
    - Cons: high cost: $O(mn)$

Cons:
- Severe limitations in optimization methods
- Extreme computational cost