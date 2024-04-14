---
title: "Intro"
description: ""
summary: ""
url: "/"
date: 2023-09-07T16:12:03+02:00
lastmod: 2023-09-07T16:12:03+02:00
draft: false
weight: 0
toc: true
---
Hi, I'm Renyi Qu. This is my handbook for AI-related stuff.

Each method is discussed in the following structure:
- **What?**: What's the idea of the method? What does it do?
- **Why?**: Why do we need it? What problems does it solve?
- **Where?**: In which application domains can we apply it?
- **When?**: When can we use it? What assumptions/conditions does it require?
- **How?**: How does it work? What's the architecture or algorithm of it?
- **Training**: How do we train it?
    - **Params**: What parameters does it have?
    - **Hyperparams**: What hyperparameters does it have? What do they do?
    - **Objective**: What loss functions are we using for it? What about regularization?
    - **Optimization**: How do we choose the optimal parameters for it?
    - **Complexity**: What's the computational cost of it?
- **Inference**: How do we use it? (Evaluation metrics will be discussed separately.)
- **Pros & Cons**: What should we be aware of when we use it?

General notations (section-specific notations take higher priority):
- {{< math >}}$ [] ${{</ math>}}: vector
- {{< math >}}$ \{\} ${{</ math>}}: set/sequence
- {{< math >}}$ || ${{</ math>}}: norm  (for a continuous vector) / count (for a discrete vector)
- {{< math >}}$ \# ${{</ math>}}: count
- {{< math >}}$ \hat{\ \ } ${{</ math>}}: estimator
- {{< math >}}$ m ${{</ math>}}: #samples in the input batch
- {{< math >}}$ n ${{</ math>}}: #features in the input sample
- {{< math >}}$ K ${{</ math>}}: #classes in the training set
- {{< math >}}$ i $: $ i ${{</ math>}}th sample
- {{< math >}}$ j$: $j ${{</ math>}}th feature
- {{< math >}}$ k$: $k ${{</ math>}}th class
- {{< math >}}$ \mathcal{D} ${{</ math>}}: training set
- {{< math >}}$ \mathcal{D}_y=\\{y_i:i\in\\{1,\cdots,m\\}\\} ${{</ math>}}: all labels
- {{< math >}}$ \mathcal{D}\_{jk}=\\{x\_{ij}:y_i=k\\}$: all values of $j$th feature for samples from $k ${{</ math>}}th class
- {{< math >}}$ X=[\mathbf{x}_1,\cdots,\mathbf{x}_m]^T$: input matrix of shape $(m,n)$ (add $\textbf{1} ${{</ math>}} if bias is needed)
- {{< math >}}$ \mathbf{y}=[y_1,\cdots,y_{m}]^T$: output vector of shape $(m,1) ${{</ math>}}
- {{< math >}}$ \textbf{w}=[w_1,\cdots,w_n]$: params (add $b ${{</ math>}} if bias is needed)