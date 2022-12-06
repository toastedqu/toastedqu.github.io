---
title : "System Design"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: false
images: []
weight: 400
---
## Clarify Requirements

- Data
    - What size? small $\rightarrow$ simpler model; large $\rightarrow$ complex model
    - What is $y$? Regression/Classification/Clustering
    - What is $x$?
- Hardware Constraints
    - Time limit?
    - Space limit?
- Objective
    - Accuracy or Speed? small models $\rightarrow$ faster; large models $\rightarrow$ accurate
    - Will we retrain after eval?

## Determine Metrics

- Offline Metrics:
    - i.e., training/testing metrics
    - e.g., AUC, precision/F1, R^2, MSE, etc.
- Online Metrics:
    - i.e., eval metrics
    - e.g., click rate, active hours, etc.
- Non-functional metrics:
    - Training speed
    - Scalability to large datasets
    - Extensibility to new techniques
    - Convenience for deployment

## Input & Data

- Target variable $y$
    - Explicit: a user action that directly indicates the label/value of $y$. (e.g., "bought", "liked") (BEST)
    - Implicit: a user action that might potentially relate to $y$. (e.g., "save for later", long review, etc.) (extra)
- Features $x$
    - Identify different features for different systems.
    - Feature engineering
        - Train/Test split
        - Missing values/Outliers
        - Scaling
        - Balance pos/neg samples (e.g., up-sampling, under-sampling, SMOTE)
    - Feature selection
        - Use some models like trees, L1, L0, etc.
        - Unnecessary for large models.
- Extra concerns
    - Sample range: are we sampling from a large enough subset of demographics?
    - Privacy/law: anonymize? remove some features violating privacy?
    - Data accessibility:
        - tabular: SQL
        - images/videos: GCP

## Model

- Order:
    - A baseline model with no ML component
        - e.g., majority vote, max/min, mean, etc.
    - Traditional ML models (typically small & fast)
    - Advanced models (typically large & slow)
- Model Explanation
    - Idea & Procedure (rough)
    - Key hyperparameters
    - Loss/Optimization objective
    - Pros & Cons

## Output & Serving

- Online A/B Testing
- Where to run inference
- Monitoring performance
- Biases/Misuses of model
- Retraining frequency

## Recommender Systems

Notations:
- $i$: product $i$
- $j$: user $j$
- $r(i,j)$: whether user $j$ has rated product $i$
- $m^{(j)}$: #products rated by user $j$
- $y^{(i,j)}$: rating by user $j$ on movie $i$
- $\theta^{(j)}$: param vector for user $j$
- $x^{(i)}$: feature vector for product $i$
- $\hat{y}^{(i,j)}=\theta^{(j)T}x^{(i)}$

Content-based recommendation:
- Used when content features $x^{(i)}$ are directly available.
- LinReg L2 Objective:
    $$
    \mathcal{L}=\frac{1}{2m^{(j)}}\sum_{j=1}^{n_u}\sum_{i:r(i,j)=1}{\left(\theta^{(j)T}x^{(i)}-y^{(i,j)}\right)^2}+\frac{\lambda}{2m^{(j)}}\sum_{j=1}^{n_u}||\theta^{(j)}||_2^2
    $$

User-based recommendation:
- Used when user vectors $\theta^{(j)}$ are directly available.
- LinReg L2 Objective:
    $$
    \mathcal{L}=\frac{1}{2m^{(j)}}\sum_{i=1}^{n_p}\sum_{j:r(i,j)=1}{\left(\theta^{(j)T}x^{(i)}-y^{(i,j)}\right)^2}+\frac{\lambda}{2m^{(j)}}\sum_{i=1}^{n_p}||x^{(i)}||_2^2
    $$

Collaborative Filtering:
- 2 problems:
    - Given $x^{(1)},\cdots,x^{(n_p)}$, estimate $\theta^{(1)},\cdots,\theta^{(n_u)}$.
    - Given $\theta^{(1)},\cdots,\theta^{(n_u)}$, estimate $x^{(1)},\cdots,x^{(n_p)}$.
- Back&Forth Switch (BAD): guess $\theta$, learn $x$, update $\theta$, update $x$, update $\theta$, update $x$, ......
- Simultaneous Objective:
    $$
    J(X,\Theta)=\frac{1}{2}\sum_{(i,j):r(i,j)=1}{\left(\theta^{(j)T}x^{(i)}-y^{(i,j)}\right)^2}+\frac{\lambda}{2m^{(j)}}\sum_{i=1}^{n_p}||x^{(i)}||_2^2+\frac{\lambda}{2m^{(j)}}\sum_{j=1}^{n_u}||\theta^{(j)}||_2^2
    $$
    - $X=[x^{(1)},\cdots,x^{(n_p)}]$, shape $(n_p,n)$
    - $\Theta=[\theta^{(1)},\cdots,\theta^{(n_u)}]$, shape $(n_u,n)$
- Algorithm:
    1. Init $x^{(1)},\cdots,x^{(n_p)},\theta^{(1)},\cdots,\theta^{(n_u)}$ to small random values.
    2. Minimize $J(X,\Theta)$ with gradient descent or smth else.

        \begin{align*}
        &x_k^{(i)}\leftarrow x_k^{(i)}-\alpha\left(\sum_{j:r(i,j)=1}{\left(\theta^{(j)T}x^{(i)}-y^{(i,j)}\right)\theta_k^{(j)}}+\lambda x_k^{(i)}\right)\\
        &\theta_k^{(j)}\leftarrow \theta_k^{(j)}-\alpha\left(\sum_{i:r(i,j)=1}{\left(\theta^{(j)T}x^{(i)}-y^{(i,j)}\right)x_k^{(i)}}+\lambda\theta_k^{(j)}\right)\\
        \end{align*}
    3. Given a user with params $\theta$ and a product with learned features $x$, predict rating of $\theta^Tx$.
