---
title : "Data"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: false
images: []
weight: 2
---
Notations:

- $m$: #samples
- $n$: #features
- $X=[\mathbf{x}_1,\cdots,\mathbf{x}_m]^T$: input matrix of shape $(m,n)$
- $\mathbf{y}=[y_1,\cdots,y_{m}]^T$: output vector of length $m$

# Cleaning

1. **Remove unwanted observations**: e.g., duplicates, irrelevant obs, etc.
2. **Fix structural errors**: e.g., typos, mislabels, inconsistency, etc.
3. **Manage unwanted outliers**: remove outliers if the model is less robust to them; keep outliers if the model is robust enough.
4. **Handle missing data** drop them, or replace them with values based on past obs.
    - (Missingness is informative in itself, so be careful.)

&nbsp;

# Transformation
## Standardization
$$
X_\text{new}=\frac{X-\bar{X}}{\sigma_X}
$$

Pros:
- Remove mean and/or scale data to unit variance. (i.e., $x_i\sim N(0,1)$)

Cons:
- Highly sensitive to outliers (outliers can greatly impact empirical mean and empirical std).
- Destroy sparsity since center might be shifted.


## Min-Max Scaling

$$\begin{align*}
&x\in[0,1]: &&X_\text{new}=\frac{X-\min{(X)}}{\max{(X)}-\min{(X)}}\\\\
&x\in[\min,\max]: &&X_\text{new}=\frac{X-\min{(X)}}{\max{(X)}-\min{(X)}}(\text{max}-\text{min})+\text{min}
\end{align*}$$

Pros:
- Can scale each $x_i$ into a range of your choice.

Cons:
- Highly sensitive to outliers.
- Destroy sparsity since center might be shifted.

## Max-Abs Scaling
$$
X_\text{new}=\frac{X}{\max{(|X|)}}
$$

Pros:
- Preserve signs of each $x_i$.
- Preserve sparsity since no shift is applied.
- Scale each $x_i$ into a range of $[-1,1]$ ($[-1,0)$ for neg entries, $(0,1]$ for pos entries).

Cons:
- Highly sensitive to outliers.

## Robust Scaling
$$
X_\text{new}=\frac{X-\text{med}(X)}{Q_{75\\%}(X)-Q_{25\\%}(X)}
$$

Pros:
- Robust to outliers.

Cons:
- Destroy sparsity since center might be shifted.

## Normalization
$$
X_\text{new}=\frac{X}{\text{norm}(X)}
$$

Pros:
- Scale individual samples to their unit norms.
- Can choose l1, l2, or max as $\text{norm}(\cdot)$.
    - l1: $\sum_j{|x_{ij}|}$
    - l2: $\sqrt{\sum_j{x_{ij}^2}}$

## Quantile Transform

- Original form
    $$
    X_\text{new}=Q^{-1}(F(X))
    $$

    - $Q^{-1}$: quantile func (i.e., PPF [percent-point func], inverse of CDF)
    - $F$: empirical CDF


- Uniform outputs:
$$
X_\text{new}=F_U^{-1}(F(X))\in[0,1]
$$

- Gaussian outputs:
$$
X_\text{new}=F_N^{-1}(F(X))\sim N(0,1)
$$

Pros:
- Robust to outliers (and literally collapse them).
- Non-parametric.

Cons:
- Distort linear correlations between diff features.
- Only works well when you have a sufficiently large amount of samples.


## Power Transform

- Yeo-Johnson Transform
    $$
    \mathbf{x}_i^{(\lambda)}=\begin{cases}
    \frac{(\mathbf{x}_i+1)^\lambda-1}{\lambda} & \text{if }\lambda\neq0,\mathbf{x}_i\geq0 \\\\
    \ln{(\mathbf{x}_i+1)}                      & \text{if }\lambda=0,\mathbf{x}_i\geq0 \\\\
    \frac{1-(1-\mathbf{x}_i)^{2-\lambda}}{2-\lambda} & \text{if }\lambda\neq2,\mathbf{x}_i<0 \\\\
    -\ln{(1-\mathbf{x}_i)}                           & \text{if }\lambda=2,\mathbf{x}_i<0
    \end{cases}
    $$
    
    - $\lambda$ is determined by MLE.



- Box-Cox Transform
    $$
    \mathbf{x}\_i^{(\lambda)}=\begin{cases}
    \frac{\mathbf{x}_i^\lambda-1}{\lambda} & \text{if }\lambda\neq0 \\\\
    \ln{(\mathbf{x}_i)} & \text{if }\lambda=0
    \end{cases}
    $$
    
    - Only applicable when $\mathbf{x}_i>0$.
    
Pros:
- Map any data to Gaussian distribution (i.e., stabilize variance and minimize skewness).
- Useful against heteroskedasticity (i.e., non-const variance).
- Sklearn's PowerTransformer further converts data to $N(0,1)$ by default.

Cons:
- Distort linear correlations between diff features.

&nbsp;

# Imputation
There are 2 types of **Missing Data**:
- MaR (Missing at Random)
- MNaR (Missing Not at Random). 

Most imputations are done on MaR. Imputation = EM:
- Repeat:
    - Estimate the missing data
    - Estimate the params

## Simple Imputation
Assume no multicollinearity:
- **Zero imputation**
- **Mean imputation** (usually better than Zero)
- **Majority imputation**

## Complex Imputation
- **Regression imputation**: fit missing feature on other features (assume multicollinearity)
    - This is NOT necessarily better than simple imputations because assumption can fail.
- **Indicator addition**: add 0-1 indicators for each feature on whether this feature is missing (0: present; 1: absent) (feature size is doubled)
- **Category addition**: for categorical features, add one more category called "missing" to represent missing values (straightforward, much better than numerical features)
- **Unsupervised Learning**: if there are lots of categories and/or lots of features, use clustering or dimensionality reduction.