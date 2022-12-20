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
This section includes fundamental concepts in ML.

# Components of ML
1. **Feature Representation**
2. **Model Structure**
3. **Objective Function**
    - Loss function
    - Regularization
4. **Optimization Method**
    - Parameter estimation
    - Hyperparameter tuning

# Types of ML
- **Supervised vs Unsupervised vs Reinforcement**:
    - Supervised learning uses labeled data and focuses on prediction.
    - Unsupervised learning uses unlabeled data and focuses on analysis.
        - Semi-supervised learning uses a small portion of labeled data and a large portion of unlabeled data. It focuses mostly on prediction on future test data or current unlabeled data (since manual labeling is not always available).
    - Reinforcement learning uses past experiences and focuses on finding optimal strategies to maximize long-term rewards.
- **Batch vs Online**:
    - Batch learning uses a bunch of samples at each training step.
    - Online learning uses a single sample at each training step.
- **Parametric vs Nonparametric**:
    - Parametric models have parameters that can be estimated by:
        - $\arg\max_{\Theta}P(\mathcal{D}|\Theta)$ (MLE)
        - $\arg\max_{\Theta}P(\Theta|\mathcal{D})$ (MAP)
    - Nonparametric models do not have parameters.
- **Discriminative vs Generative**:
    - Discriminative models directly compute $P(Y|X)$ or predict label given sample features.
    - Generative models can be used for both sample generation and label prediction by computing:
        - $P(X,Y)=P(Y)P(X|Y)$ for generation
        - $P(Y|X)\propto P(Y)P(X_1,\cdots,X_n|Y)$ for label prediction

<center>

|   Discriminative            |                Generative    |
|:-----------------------------------:|:---------------------------:|
| more suitable for supervised | more suitable for unsupervised |
| computationally cheap | computationally expensive |
| need more data to train | need less data to train due to strong prior as bias |
| less friendly with missing data | easy marginalization over missing data |
| more accurate in general | less accurate in general (violation of CI assumption) |

</center>

# MLE & MAP
$$\begin{align*}
&\text{MLE}: &&\arg\max_{\Theta}P(\mathcal{D}|\Theta)\\\\
&\text{MAP}: &&\arg\max_{\Theta}P(\Theta|\mathcal{D})
\end{align*}$$

- Minimize loss = Maximize likelihood
- Regularization = Adding prior belief
- MAP = MLE + Prior

# Bias-Variance Trade-off

$$\begin{align*}
&\text{Bias}(\hat{\boldsymbol{\theta}})=\mathbb{E}[\hat{\boldsymbol{\theta}}-\boldsymbol{\theta}]=\mathbb{E}[\hat{\boldsymbol{\theta}}]-\mathbb{E}[\boldsymbol{\theta}]\\\\
&\text{Var}(\hat{\boldsymbol{\theta}})=\mathbb{E}[(\hat{\boldsymbol{\theta}}-\mathbb{E}[\hat{\boldsymbol{\theta}}])^2]\\\\
\end{align*}$$

**Bias**: how much our average model predictions differ from ground truth over different training sets. (i.e., model predictive power)
- High bias: oversimplified model $\rightarrow$ underfitting $\rightarrow$ high error on train & test.
- Models with Low bias: KNN, Decision Tree, SVM.
- Models with High bias: Linear models.
- Validation set error = unbiased estimator of true error.

**Variance**: how much our estimates change due to changes in training data. (i.e., model sensitivity)
- High variance: overcomplex model $\rightarrow$ overfitting $\rightarrow$ low error on train & high error on test.

**Trade-off**: when you have low/high bias, it is inevitable to have high/low variance.

- Test error = Variance + Bias^2 + Noise
$$
\mathbb{E}\_{\textbf{x},y,\mathcal{D}}[(\hat{y}(\textbf{x};\mathcal{D})-y)^2]=\mathbb{E}\_{\textbf{x},\mathcal{D}}[(\hat{y}(\textbf{x};\mathcal{D})-\bar{\hat{y}}(\textbf{x}))^2]+\mathbb{E}\_{\textbf{x}}[(\bar{\hat{y}}(\textbf{x})-\bar{y}(\textbf{x}))^2]+\mathbb{E}\_{\textbf{x},y}[(\bar{y}(\textbf{x})-y)^2]
$$
    - $\mathcal{D}$: training set
    - $(\textbf{x},y)$: test sample

<center>
<img src="/images/ml_concepts/bv_tradeoff.jpg" width="500"/>
</center>