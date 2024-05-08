---
title: "Hi👋"
description: ""
summary: ""
url: "/"
date: 2023-09-07T16:12:03+02:00
lastmod: 2023-09-07T16:12:03+02:00
draft: false
weight: 1
toc: true
---
I'm Renyi Qu. Welcome to my minimalist AI handbook.

Each concept is decomposed in the following structure:
- **Why**: Why do we need it? What's the motivation? What's the problem?
- **What**: What is it?
    - **Mechanism**: What's the mechanism?
    - **Components**: What are the components of it?
        - **Params**: What params? What do they do?
        - **Hyperparams**: What hyperparams? What do they do?
- **How**: How do we use it?
    - **Training**: How do we train it?
        - **Objective**: What objective are we aiming at?
        - **Optimization**: How do we get to the objective?
    - **Inference**: How do we apply it?
- **When**: When can we use it? What are the assumptions/conditions?
- **Where**: Where can we apply it?
- **Pros & Cons**: What should we be aware of when we use it?

The following notations are commonly used (section-specific notations take higher priority):
- {{< math >}}$ []/\textbf{a} ${{</ math>}}: vector
- {{< math >}}$ A ${{</ math>}}: matrix
- {{< math >}}$ \{\}/\mathcal{A}/\mathbb{A} ${{</ math>}}: set
- {{< math >}}$ || ${{</ math>}}: norm  (for continuous vectors) / count (for discrete vectors)
- {{< math >}}$ \# ${{</ math>}}: count
- {{< math >}}$ \hat{\ \ } ${{</ math>}}: estimator
- {{< math >}}$ m ${{</ math>}}: #samples in the input batch
- {{< math >}}$ n ${{</ math>}}: #features in the input sample
- {{< math >}}$ c ${{</ math>}}: #classes in the training set
- {{< math >}}$ i ${{</ math>}}: sample index
- {{< math >}}$ j ${{</ math>}}: feature index
- {{< math >}}$ k ${{</ math>}}: class index
- {{< math >}}$ \mathcal{D} ${{</ math>}}: training set
- {{< math >}}$ \mathcal{D}_y=\{y_i:i\in\{1,\cdots,m\}\} ${{</ math>}}: class set
- {{< math >}}$ \mathcal{D}_{jk}=\{x_{ij}:y_i=k\} ${{</ math>}}: all values of {{< math >}}$j${{</ math>}}th feature for samples from {{< math >}}$k${{</ math>}}th class
- {{< math >}}$ X=[\mathbf{x}_1,\cdots,\mathbf{x}_m]^T${{</ math>}}: input matrix of shape {{< math >}}$(m,n)${{</ math>}} (add {{< math >}}$\textbf{1}${{</ math>}} if bias is needed)
- {{< math >}}$ \mathbf{y}=[y_1,\cdots,y_{m}]^T${{</ math>}}: output vector of shape {{< math >}}$(m,1)${{</ math>}}
- {{< math >}}$ \textbf{w}=[w_1,\cdots,w_n]${{</ math>}}: params (add {{< math >}}$b${{</ math>}} if bias is needed)