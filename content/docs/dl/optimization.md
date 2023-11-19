---
title : "Optimization"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: false
images: []
weight: 300
---
Optimization means the adjustment of params to minimize/maximize an objective function. In DL, it involves 5 key components:
- **Loss Function**: Difference between predicted output $\hat{y}$ and actual output $y$.
- **Gradient Descent**: Iteratively use loss gradient to update params to reduce loss. While other optimization methods exist, GD and its variations are the best.
- **Learning Rate**: Step size taken during each iteration, controlling convergence and stability of GD.
- **Epochs**: #times to go through the entire dataset.
- **Batch Size**: #samples in a batch, which impacts how often params are updated.

Basic formula:
$$
w_t\leftarrow w_{t-1}-\eta g_t
$$

Notations:
- $w_t$: param
- $\eta$: learning rate
- $g_t$: gradient

&nbsp;

## Gradient Descent
$$\begin{align*}
&\text{Basic ver.:}             &&g_t=\nabla_w\mathcal{L}(w_{t-1})\\\\
&\text{L2 regularization ver.:} &&g_t=\nabla_w\mathcal{L}(w_{t-1})+\lambda w_{t-1}\\\\
&\text{Momentum ver.:}          &&g_t\leftarrow\beta g_{t-1}+(1-\beta)g_t
\end{align*}$$

Notations:
- $\mathcal{L}$: loss
- $\lambda$: L2 penalty weight
- $\beta$: momentum weight
    - larger $\rightarrow$ smoother updates due to more past gradients involved
    - typical values: 0.8, 0.9, 0.999

Types:
- **Stochastic GD**: update params after each sample
- **Mini-Batch GD**: update params after each mini-batch of samples
- **Batch GD**: update params after the entire dataset

&nbsp;

## RMSProp
