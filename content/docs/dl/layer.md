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

&nbsp;

## Dropout
$$
Y=\mathrm{Dropout}(X, p)
$$

Idea: randomly make some elements become 0 with specified probability

Notations:
- $X$: input tensor of arbitrary shape
- $Y$: output tensor of the input shape
- $p\in(0,1)$: dropout probability

&nbsp;

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

&nbsp;

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

&nbsp;

&nbsp;

# CNN Layers
## Convolution
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

&nbsp;

## Pooling
### Max Pooling
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

&nbsp;

### Average Pooling
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

&nbsp;

&nbsp;

# RNN Layers
## RNN

<center>
<img src="/images/dl/RNN.png" width="400"/>
</center>

$$
h_t=\tanh(x_tW_{xh}^T+h_{t-1}W_{hh}^T)
$$

Idea: **recurrence** - maintain a hidden state that captures information about previous inputs in the sequence

Notations:
- $x_t$: input at time $t$ of shape $(m,H_{in})$
- $h_t$: hidden state at time $t$ of shape $(D,m,H_{out})$
- $W_{xh}$: weight matrix of shape $(H_{out},H_{in})$ if initial layer, else $(H_{out},DH_{out})$
- $W_{hh}$: weight matrix of shape $(H_{out},H_{out})$
- $H_{in}$: input size, #features in $x_t$
- $H_{out}$: hidden size, #features in $h_t$
- $m$: batch size
- $D$: $=2$ if bi-directional else $1$

Cons:
- Short-term memory: hard to carry info from earlier steps to later ones if long seq
- Vanishing gradient: gradients in earlier parts become extremely small if long seq

&nbsp;

## GRU

<center>
<img src="/images/dl/GRU.png" width="400"/>
</center>

$$\begin{align*}
&r_t=\sigma(x_tW_{xr}^T+h_{t-1}W_{hr}^T) \\\\
&z_t=\sigma(x_tW_{xz}^T+h_{t-1}W_{hz}^T) \\\\
&\tilde{h}\_t=\tanh(x_tW_{xn}^T+r_t\odot(h_{t-1}W_{hn}^T)) \\\\
&h_t=(1-z_t)\odot\tilde{h}\_t+z_t\odot h_{t-1}
\end{align*}$$

Idea: Gated Recurrent Unit - use 2 gates to address long-term info propagation issue in RNN:
1. **Reset gate**: determine how much of $h_{t-1}$ should be ignored when computing $\tilde{h}\_t$.
2. **Update gate**: determine how much of $h_{t-1}$ should be retained for $h_t$.
3. **Candidate**: calculate candidate $\tilde{h}\_t$ with reset $h_{t-1}$.
4. **Final**: calculate weighted average between candidate $\tilde{h}\_t$ and prev state $h_{t-1}$ with the retain ratio.

Notations:
- $r_t$: reset gate at time $t$ of shape $(m,H_{out})$
- $z_t$: update gate at time $t$ of shape $(m,H_{out})$
- $\tilde{h}\_t$: candidate hidden state at time $t$ of shape $(m,H_{out})$
- $\odot$: element-wise product

&nbsp;

## LSTM

<center>
<img src="/images/dl/LSTM.png" width="400"/>
</center>

$$\begin{align*}
&i_t=\sigma(x_tW_{xi}^T+h_{t-1}W_{hi}^T) \\\\
&f_t=\sigma(x_tW_{xf}^T+h_{t-1}W_{hf}^T) \\\\
&\tilde{c}\_t=\tanh(x_tW_{xc}^T+h_{t-1}W_{hc}^T) \\\\
&c_t=f_t\odot c_{t-1}+i_t\odot \tilde{c}\_t \\\\
&o_t=\sigma(x_tW_{xo}^T+h_{t-1}W_{ho}^T) \\\\
&h_t=o_t\odot\tanh(c_t)
\end{align*}$$

Idea: Long Short-Term Memory - use 3 gates:
1. **Input gate**: determine what new info from $x_t$ should be added to cell state $c_t$.
2. **Forget gate**: determine what info from prev cell $c_{t-1}$ should be forgotten.
3. **Candidate cell**: create a new candidate cell from $x_t$ and $h_{t-1}$.
4. **Update cell**: use $i_t$ and $f_t$ to combine prev and new candidate cells.
5. **Output gate**: determine what info from curr cell $c_t$ should be added to output $h_t$.
6. **Final**: simply apply $o_t$ to activated cell $c_t$.

Notations:
- $i_t$: input gate at time $t$ of shape $(m,H_{out})$
- $f_t$: forget gate at time $t$ of shape $(m,H_{out})$
- $c_t$: cell state at time $t$ of shape $(m,H_{cell})$
- $o_t$: output gate at time $t$ of shape $(m,H_{out})$
- $H_{cell}$: cell hidden size (in most cases same as $H_{out}$)

&nbsp;

&nbsp;

# Transformer

<center>
<img src="/images/dl/transformer.png" width="500"/>
</center>

## Positional Encoding
$$\begin{align*}
\text{PE}\_{(pos,2i)}&=\sin(\frac{pos}{10000^{\frac{2i}{d_\text{model}}}})\\\\
\text{PE}\_{(pos,2i+1)}&=\cos(\frac{pos}{10000^{\frac{2i}{d_\text{model}}}})
\end{align*}$$

Idea: encode the sequence order info into the token embeddings.

## Scaled Dot-Product Attention

<center>
<img src="/images/dl/scaled_dot_product_attention.png" width="200"/>
</center>

$$
\text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

Idea: A common question that most people have about this is not HOW, but WHY?
1. Why?
    - The end goal is **Attention** - "Which parts of the sentence should we focus on?"
    - We want to **capture the most relevant info** in the sentence.
    - And we also want to **keep track of all info** in the sentence as well, just with different weights.
    - We want to create **contextualized representations** of the sentence.
    - Therefore, attention mechanism - we want to assign different attention scores to each token.
2. Preliminaries:
    - **Query (Q)**: Think of it like a **question** about a token - "How important is this token in the context of the whole sentence?"
    - **Key (K)**: Think of it like a piece of **unique info** about a token - "Here's something unique about this token."
    - **Value (V)**: Think of it like the **actual meaning** of a token - "Here's the content about this token."
3. First, we **compare the similarity** between the **Q** of one word and the **K** of every other word.
    - The more similar they are, the more attention we should give to that word for the queried word.
4. Second, we **scale them down** by the square root of vector dimensionality to avoid the similarity scores being too large.
5. Third, we use softmax to convert the attention scores into a **probability distribution**.
    - softmax sums up to 1 and emphasizes important attention weights (and reduces the impact of negligible ones).
6. Fourth, we get a **weighted combination** of all words, for each queried word, as the final attention score.

&nbsp;

## Multi-Head Attention

<center>
<img src="/images/dl/mha.png" width="300"/>
</center>

$$\begin{align*}
\text{MultiHead}(Q,K,V)&=\text{Concat}(\text{head}_1,\cdots,\text{head}_h)W^O \\\\
\text{head}_i&=\text{Attention}(QW_i^Q,KW_i^K,VW_i^V)
\end{align*}$$

Idea: basically a weighted average of multiple attention heads.
- The sole reason is that multiple heads perform better than a single head.
- Masked MHA is basically masking the succeeding tokens off, because they can't be seen during decoding.

Notations:
- $W_i^Q\in\mathbb{R}^{d_\text{model}\times d_k},W_i^K\in\mathbb{R}^{d_\text{model}\times d_k},W_i^V\in\mathbb{R}^{d_\text{model}\times d_v}$: learnable linear projection params.
- $W^O\in\mathbb{R}^{d_\text{model}\times hd_v}$: learnable linear combination weights.
- $h=8, d_k=d_v=\frac{d_\text{model}}{h}=64$ in the original paper.

&nbsp;

## Postion-wise Feed-Forward Networks
$$
\text{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2
$$

Idea: 2 linear transformations with ReLU in between.

Notations: $d_\text{model}=512, d_\text{FF}=2048$