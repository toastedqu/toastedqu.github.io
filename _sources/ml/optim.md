---
title : "Optimization"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: false
images: []
weight: 5
---
# Loss
Loss is a measure of difference between predicted output $ \hat{y}_i$ and actual output $y_i $.
- Unreduced loss: $ \mathcal{L}=[\mathcal{l}_1,\cdots,\mathcal{l}_m] $
- Reduced loss: $\mathcal{L}(\mathbf{y},\mathbf{\hat{y}})=\begin{cases}
\text{sum}(\mathcal{L})=\sum_{i=1}^{m}l_i\\\\
\text{mean}(\mathcal{L})=\frac{1}{m}\sum_{i=1}^{m}l_i
\end{cases}$ (can impose sample weights as well)

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

Pros:
- A mixture of L1 and L2 losses

Cons:
- Less used than either L1 or L2

## Classification

### 0-1 (Heaviside)

$$
l_i=\begin{cases}
0 & \text{if }y_i\hat{y_i}>0\\\\
1 & \text{if }y_i\hat{y_i}<0
\end{cases}, y_i\in\\{-1,1\\}
$$

Usage: Naive Bayes

Cons:
- Penalize all incorrect predictions equally
- Non-convex
- Non-differentiable


<!-- ```python
def misclassification(v):
    return np.vectorize(lambda x: 0 if x>0 else 1)(v)
``` -->

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

<!-- ```python
def hinge_loss(v):
    return np.vectorize(lambda x: max(0,1-x))(v)
``` -->

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

<!-- ```python
def logistic_loss(v):
    return 1/np.log(2)*np.log(1+np.exp(-v))
``` -->

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

<!-- ```python
def exponential_loss(v):
    return np.exp(-v)
``` -->

### Squared

$$
l_i=(1-y_i\hat{y}_i)^2, y_i\in\\{-1,1\\}
$$

Usage: rare

<!-- ```python
def squared_loss(v):
    return np.square(1-v)
``` -->

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
plt.xlabel('$ y_i\hat{y}_i $')
plt.ylabel('l_i')
plt.ylim(0,4)
plt.legend()
plt.show()
``` -->

<center>
<img src="/images/ml/loss.png" width="500"/>
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
- Regularization has a weaker effect on the model (params) as the sample size increases, because more info become available from the data.

## Explicit (Penalty)

### L1

$$\begin{align*}
&\text{Frequentist:}\ &&\lambda||\mathbf{w}||_1\\\\
&\text{Bayesian:}\    &&w_j\sim\text{Laplace}(0,\gamma^2)
\end{align*}$$

Pros:
- Sparse $ \leftarrow $ Feature selection (reduce weights of trash features to 0)
- Robust to outliers
- Equal weight shrinkage ($ \frac{\partial L_1}{\partial w_{j}}=1 $)
- Effective with multicollinearity
- Able to handle $ n>>m $ cases

Cons:
- No closed-form solution (high computational cost with LinReg)
- Need to select perfect $ \lambda=\frac{\sigma^2}{\gamma^2} $

### L2

$$\begin{align*}
&\text{Frequentist:}\ &&\lambda||\mathbf{w}||_2^2\\\\
&\text{Bayesian:}\    &&w_j\sim N(0,\gamma^2)
\end{align*}$$

Pros:
- More/Less shrinkage on larger/smaller weights ($ \frac{\partial L_2}{\partial w_{j}}=2w_{j} $)
- Effective with multicollinearity
- Closed-form solution. Low computational cost with LinReg

Cons:
- No sparsity $ \leftarrow $ No feature selection
- Not robust to outliers
- Not able to handle $ n>>m $ cases
- Need to select perfect $ \lambda=\frac{\sigma^2}{\gamma^2} $

### Elastic Net
$$
\lambda(r\_{\text{L1}}||\mathbf{w}||_1+(1-r\_{\text{L1}})||\mathbf{w}||_2^2)
$$
- $ r\_{\text{L1}} $: L1 ratio

Pros:
- A mixture of L1 and L2 penalties

Cons:
- Less used than either L1 or L2

### L0

$$\begin{align*}
&\text{Frequentist:} &&\lambda||\mathbf{w}||_0\\\\
&\text{Bayesian:} &&w_j\sim\text{Spike and Slab}
\end{align*}$$

Optimization: Search
- **Streamwise Regression**:
    1. Init model. Init $ Err_0=||y_i||_2^2 $.
    2. For $ j $ in range(1,n+1):
        1. Add feature $ \mathbf{x}_j $ to model.
        2. If $ Err=||y_i-\sum\_{j\in\text{model}}{w_{j}x\_{ij}}||_2^2+\lambda||\textbf{w}\_{\text{model}}||_0 <\ Err\_{j-1} $:
            1. Keep $ \mathbf{x}_j $.
            2. $ Err_j = Err $
        3. Else:
            1. $ Err_j = Err_{j-1} $
    - Pros:
        - Low computational cost: $ O(n) $
    - Cons:
        - No guarantee to find optimal solution
        - Order of features matters
<br><br>

- **Stepwise Regression**:
    1. Init model. Init $ Err_0=||y_i||_2^2 $.
    2. While True (n loops):
        1. Try to add each of all remaining features $ \mathbf{x}_k $ to model.
        2. Pick the feature with $ Err=\min(||y_i-\sum\_{j\in\text{model}}{w_{j}x\_{ij}}||_2^2+\lambda||\textbf{w}\_\text{model}||_0 $:
        3. If $ Err<Err_0 $:
            1. Add feature to model.
            2. $ Err_0 = Err $
        4. Else:
            1. Break
    - Pros: 
        - More likely to find optimal solution than streamwise
    - Cons: 
        - High computational cost: $ O(mn) $
        - Overfitting
        - Multicollinearity
<br><br>

- **Stagewise Regression**:
    1. Init model. Init $ Err_0=||y_i||_2^2$. Init cache $\textbf{w} $.
    2. While True (n loops):
        1. Try to add each of all remaining features $ \mathbf{x}_k $ to model.
        2. Pick the feature with $ Err=\min(||r_i-w_{k}x\_{ik}||_2^2+\lambda||\textbf{w}||_0)$, where $r_i=y_i-\sum\_{j\in\text{model}}{w_jx\_{ij}} $:
        3. If $ Err<Err_0 $:
            1. Add feature to model. Add $ w_{k} $ to cache.
            2. $ Err_0 = Err $
        4. Else:
            1. Break
    - Pros:
        - Faster than stepwise regression (no need to create new long models each time)
        - No multicollinearity
        - Used for **boosting**
    - Cons: 
        - High computational cost: $ O(mn) $

Common Pros:
- Explicit feature selection $ \rightarrow $ Sparsity

Common Cons:
- Severe limitations in optimization methods
- Extreme computational cost


## Implicit
tbd





# Parameter Optimization
The word "Learning" in ML is literally just parameter estimation (i.e., objective optimization). Pure statistics.

## Gradient Descent
$$
\boldsymbol{\theta}\leftarrow\boldsymbol{\theta}-\eta\nabla\mathcal{L}(\boldsymbol{\theta})
$$

Types:
- Stochastic GD: update params for each single sample
- Mini-Batch GD: update params for each mini-batch of samples
- Batch GD: update params after observing all samples

Pros:
- Extremely widely used

Cons:
- Cannot find global minimum if
    - bad step size: too big then unstable; too small then taking too long
    - non-convex objective: stuck at local minima or saddle point

### Adagrad
Idea: tune the learning rate so that the model learns slowly from frequent features but pay special attention to rare but informative features.
$$
\eta_{tj}=\frac{\eta}{\sqrt{\sum_{\tau=1}^t(\frac{\partial\mathcal{L}_k}{\partial{\theta_j}})^2}+\epsilon}
$$

## Expectation-Maximization
Idea: learn model (do MLE/MAP on params) with latent variables (NOT explicit for GMM).

Method: 
1. Init params $ \Theta$ for prior $P(z)$ and likelihood $P(\mathcal{D}|z) $
2. E-step: estimate $ P(z|\mathcal{D})$ given params $\Theta $.
    - Estimate the expected value of the latent variables given the current estimates of the model params and the data.
3. M-step: estimate $ \Theta$ via "$\arg\max_\Theta P(z|\mathcal{D})$" (MLE) or "$\arg\max_\Theta P(z,\mathcal{D}) $" (MAP).
    - Estimate the model params through MLE or MAP on the expected value of the complete data likelihood.
4. Repeat Step 2-3 until convergence.

Pros:
- Offer estimates for both latent variables and model params

Cons:
- NOT guaranteed to find optimum





# Hyperparameter Tuning
tbd

# Evaluation
In regression, MAE/MSE/RMSE is used for both training and evaluation.

In classification, there are various metrics mostly centered at the idea of **confusion matrix**.

## Confusion Matrix
Confusion Matrix is a $ K\times K $ matrix which visualizes actual classes against predicted classes.

An example of confusion matrix for binary classification:

<center>

|       | True P | True N |
|:-----:|:--:|----|
| Predicted **P** | TP | FP |
| Predicted **N** | FN | TN |

</center>

- Notations:
    - T&F = whether prediction matches actual
    - P&N = predicted class
    - FP = type 1 error
    - FN = type 2 error
- Metrics:
    - **Error Rate**: $ \frac{FP+FN}{P+N} $
    - **Accuracy**: $ \frac{TP+TN}{P+N} $
    - **Specificity**: TN rate among negative actual values: $ \frac{TN}{FP+TN} $
    - **1-Specificity (FPR)**: FP rate among negative actual values: $ \frac{FP}{FP+TN} $
    - **Precision**: TP rate among positive predicted values: $ \frac{TP}{TP+FP} $
    - **Recall**: TP rate among positive actual values: $ \frac{TP}{TP+FN} $
    - **F-score**: harmonic mean of Precision and Recall: $ \frac{(1+\beta^2)\cdot\text{precision}\cdot\text{recall}}{(\beta^2\cdot\text{precision})+\text{recall}} $
    - **F1-score**: $ \frac{2\cdot\text{precision}\cdot\text{recall}}{\text{precision}+\text{recall}} $
- **ROC** (Receiver Operating Curve): plot of TPR vs FPR
- **AUC** (Area Under Curve): area under ROC curve





# Distance & Similarity

## Norm & Distance
Norm is NOT a distance measure but a size measure. It offers insights for many prevalent distance measures.

Properties:
- $ L_p(\textbf{x})=0\leftrightarrow \textbf{x}=\textbf{0} $
- $ L_p(\textbf{x}+\textbf{y})\leq L_p(\textbf{x})+L_p(\textbf{y}) $
- $ L_p(c\textbf{x})=|c|L_p(\textbf{x})\ \ \ \forall c\in\mathbb{R} $

Types:
- $ L_p$: $||\textbf{x}||_p=\left(\sum_i{|x_i|^p}\right)^\frac{1}{p} $
- $ L_0$: $||\textbf{x}||_0=\\#x_i:x_i>0,x_i\in\textbf{x} $ 
- $ L_1$: $||\textbf{x}||_1=\sum_i{|x_i|} $
- $ L_2$: $||\textbf{x}||_2=\sqrt{\sum_i{(x_i)^2}} $
- $ L_\infty$: $||\textbf{x}||_\infty=\max{\{|x_i|:x_i\in\textbf{x}\}} $

## Cosine Similarity
tbd


## Kernel
Def: measure of similarity between 2 vectors.

Properties:
- $ \mathbf{K}=k(\mathbf{x},\mathbf{x}') $ is positive semi-definite.
    - PSD: $ \mathbf{K}=\sum_{i=1}^{m}{\lambda_i\mathbf{z}_i\mathbf{z}_i^T} $.
    - $ \lambda_i $: non-negative real eigenvalues.
    - $ \mathbf{z}_i $: real eigenvectors.
- $ k(\mathbf{x},\mathbf{x}')=ck_1(\mathbf{x},\mathbf{x}'),c>0 $
- $ k(\mathbf{x},\mathbf{x}')=k_1(\mathbf{x},\mathbf{x}')+k_2(\mathbf{x},\mathbf{x}') $
- $ k(\mathbf{x},\mathbf{x}')=k_1(\mathbf{x},\mathbf{x}')+k_2(\mathbf{x},\mathbf{x}') $
- $ k(\mathbf{x},\mathbf{x}')=q(k_1(\mathbf{x},\mathbf{x}'))$, where $q(\cdot) $ is polynomial func with positive coeffs.
- $ k(\mathbf{x},\mathbf{x}')=f(\mathbf{x})k_1(\mathbf{x},\mathbf{x}')f(\mathbf{x}') $
- $ k(\mathbf{x},\mathbf{x}')=\mathbf{x}^T\mathbf{A}\mathbf{x}' $
- $ k(\mathbf{x},\mathbf{x}')=\phi(\mathbf{x})^T\phi(\mathbf{x}') $