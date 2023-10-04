---
title : "Layer"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: false
images: []
weight: 200
---
A layer is fundamentally a function that transforms input $X$ into output $Y$.

(Note: I believe RNN is dead, so I won't waste time on it.)

# Basics

## Linear

$$
Y=\sigma(XW^T+\textbf{b})
$$

Idea: linear transformation

Notations:
- $X$: input tensor of shape $(*, H_{in})$
- $Y$: output tensor of shape $(*, H_{out})$
- $W$: weight matrix of shape $(H_{out}, H_{in})$
- $\textbf{b}$: bias vector of size $(H_{out})$

## Dropout

$$
Y=\mathrm{Dropout}(X, p)
$$

Idea: randomly make some elements become 0 with specified probability

Notations:
- $X$: input tensor of arbitrary shape
- $Y$: output tensor of the input shape
- $p\in(0,1)$: dropout probability

## Normalization

### Batch Normalization

$$
Y=\gamma\frac{X-E_m[X]}{\sqrt{\mathrm{Var}_m[X]+\epsilon}}+\beta
$$

Idea: normalize each feature independently across all samples

Notations:
- $X$: input tensor of shape $(m,n)$
- $Y$: output tensor of shape $(m,n)$
- $m$: (hyperparam) batch size
- $n$: (hyperparam) #features
- $\epsilon$: (hyperparam) tiny value to avoid zero division
- $\gamma$: (learnable param) init as 1
- $\beta$: (learnable param) init as 0

Pros:
- Allow us to use higher learning rates
- Allow us to care less about param initialization

Cons:
- Dependent on batch size $\rightarrow$ ineffective for small batches

### Layer Normalization

$$
Y=\gamma\frac{X-E_n[X]}{\sqrt{\mathrm{Var}_n[X]+\epsilon}}+\beta
$$

Idea: normalize each sample independently across all features

Notations:
- $X$: input tensor of shape $(m,*)$
- $Y$: output tensor of shape $(m,*)$
- $m$: (hyperparam) batch size
- $\epsilon$: (hyperparam) tiny value to avoid zero division
- $\gamma$: (learnable param) init as 1
- $\beta$: (learnable param) init as 0

Pros:
- Same as BN
- Applicable on small batches

# CNN Layers

## Convolutional

$$
Y_{ij}=\sum_{c=0}^{C_{in}-1}W_{jc}\ast X_{ic}+\textbf{b}_j
$$

Idea: convolution

Notations:
- $X$: input tensor of shape $(m,C_{in},H_{in},W_{in})$
- $Y$: output tensor of shape $(m,C_{out},H_{out},W_{out})$
- $W$: filter weights of shape $(C_{out}, C_{in}, H_{filt}, W_{filt})$
- $\textbf{b}$: filter bias of shape $(C_{out})$
- $C_{in}, C_{out}$: #channels of input, #channels of output
- $(H_{out},W_{out})=\left(\lfloor\frac{H_{in}-H_{filt}+2p}{s}+1\rfloor, \lfloor\frac{W_{in}-W_{filt}+2p}{s}+1\rfloor\right)$, height and width of output (image)
- $p$: padding size
- $s$: stride size
- $m$: (hyperparam) batch size
- $i\in[1,m]$: sample index
- $j\in[1,C_{out}]$: out channel index

## Pooling

## Max Pooling

$$
Y_{ijhw}=\max_{u\in[0,H_{filt}-1]}\max_{v\in[0,W_{filt}-1]}X_{ij,H_{filt}*h+u,W_{filt}*w+u}
$$

Idea: pool images by selecting the max element in each filter window (the equation is stupid, just visualize it)

Notations:
- $X$: input tensor of shape $(m,C,H_{in},W_{in})$
- $Y$: output tensor of shape $(m,C,H_{out},W_{out})$
- $m$: (hyperparam) batch size
- $C$: #channels
- $(H_{out},W_{out})=\left(\lfloor\frac{H_{in}+2p}{H_{filt}}\rfloor, \lfloor\frac{W_{in}+2p}{W_{filt}}\rfloor\right)$ (stride size is filter size in pooling)
- $p$: padding size

## Average Pooling

$$
Y_{ijhw}=\frac{1}{H_{filt}W_{filt}}\sum_{u=0}^{H_{filt}-1}\sum_{v=0}^{W_{filt}-1}X_{ij,H_{filt}*h+u,W_{filt}*w+u}
$$

Idea: pool images by selecting the max element in each filter window (the equation is stupid, just visualize it)

Notations:
- $X$: input tensor of shape $(m,C,H_{in},W_{in})$
- $Y$: output tensor of shape $(m,C,H_{out},W_{out})$
- $m$: (hyperparam) batch size
- $C$: #channels
- $(H_{out},W_{out})=\left(\lfloor\frac{H_{in}+2p}{H_{filt}}\rfloor, \lfloor\frac{W_{in}+2p}{W_{filt}}\rfloor\right)$ (stride size is filter size in pooling)
- $p$: padding size