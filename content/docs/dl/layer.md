---
title : "Layer"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: false
images: []
weight: 12
---
A layer is a function that transforms input {{<math>}}$X${{</math>}} into output {{<math>}}$Y${{</math>}}.

# Basics
## Linear
{{< math class=text-center >}}$$
Y=XW^T+\textbf{b}
$${{</ math >}}

What: linear transformation

Why: Universal Approximation Theorem

Notations:
- {{< math >}}$ X$: input tensor of shape $(*, H_{in}) ${{</ math>}}
- {{< math >}}$ Y$: output tensor of shape $(*, H_{out}) ${{</ math>}}
- {{< math >}}$ W$: weight matrix of shape $(H_{out}, H_{in}) ${{</ math>}}
- {{< math >}}$ \textbf{b}$: bias vector of size $(H_{out}) ${{</ math>}}

## Dropout
$$
Y=\mathrm{Dropout}(X, p)
$$

Idea: randomly make some elements become 0 with specified probability

Notations:
- {{< math >}}$ X ${{</ math>}}: input tensor of arbitrary shape
- {{< math >}}$ Y ${{</ math>}}: output tensor of the input shape
- {{< math >}}$ p\in(0,1) ${{</ math>}}: dropout probability

## Residual Block (ResNet)
$$
Y\leftarrow Y+X
$$

Idea: add input to output
- reduce vanishing gradient problem
- allow parametrization for the identity function {{< math >}}$ f(X)=X ${{</ math>}}
- add complexity in the simplest way

## Normalization
### Batch Normalization
$$
Y=\gamma\frac{X-E_m[X]}{\sqrt{\mathrm{Var}_m[X]+\epsilon}}+\beta
$$

Idea: normalize each feature independently across all samples

Notations:
- {{< math >}}$ X$: input tensor of shape $(m,n) ${{</ math>}}
- {{< math >}}$ Y$: output tensor of shape $(m,n) ${{</ math>}}
- {{< math >}}$ m ${{</ math>}}: (hyperparam) batch size
- {{< math >}}$ n ${{</ math>}}: (hyperparam) #features
- {{< math >}}$ \epsilon ${{</ math>}}: (hyperparam) tiny value to avoid zero division
- {{< math >}}$ \gamma ${{</ math>}}: (learnable param) init as 1
- {{< math >}}$ \beta ${{</ math>}}: (learnable param) init as 0

Pros:
- Allow us to use higher learning rates
- Allow us to care less about param initialization

Cons:
- Dependent on batch size {{< math >}}$ \rightarrow ${{</ math>}} ineffective for small batches

### Layer Normalization
$$
Y=\gamma\frac{X-E_n[X]}{\sqrt{\mathrm{Var}_n[X]+\epsilon}}+\beta
$$

Idea: normalize each sample independently across all features

Notations:
- {{< math >}}$ X$: input tensor of shape $(m,*) ${{</ math>}}
- {{< math >}}$ Y$: output tensor of shape $(m,*) ${{</ math>}}
- {{< math >}}$ m ${{</ math>}}: (hyperparam) batch size
- {{< math >}}$ \epsilon ${{</ math>}}: (hyperparam) tiny value to avoid zero division
- {{< math >}}$ \gamma ${{</ math>}}: (learnable param) init as 1
- {{< math >}}$ \beta ${{</ math>}}: (learnable param) init as 0

Pros:
- Same as BN
- Applicable on small batches

&nbsp;

# CNN
## Convolution
$$
Y_{ij}=\sum_{c=0}^{C_{in}-1}W_{jc}\ast X_{ic}+\textbf{b}_j
$$

Idea: convolution

Notations:
- {{< math >}}$ X$: input tensor of shape $(m,C_{in},H_{in},W_{in}) ${{</ math>}}
- {{< math >}}$ Y$: output tensor of shape $(m,C_{out},H_{out},W_{out}) ${{</ math>}}
- {{< math >}}$ W$: filter weights of shape $(C_{out}, C_{in}, H_{filt}, W_{filt}) ${{</ math>}}
- {{< math >}}$ \textbf{b}$: filter bias of shape $(C_{out}) ${{</ math>}}
- {{< math >}}$ C_{in}, C_{out} ${{</ math>}}: #channels of input, #channels of output
- {{< math >}}$ (H_{out},W_{out})=\left(\lfloor\frac{H_{in}-H_{filt}+2p}{s}+1\rfloor, \lfloor\frac{W_{in}-W_{filt}+2p}{s}+1\rfloor\right) ${{</ math>}}, height and width of output (image)
- {{< math >}}$ p ${{</ math>}}: padding size
- {{< math >}}$ s ${{</ math>}}: stride size
- {{< math >}}$ m ${{</ math>}}: (hyperparam) batch size
- {{< math >}}$ i\in[1,m] ${{</ math>}}: sample index
- {{< math >}}$ j\in[1,C_{out}] ${{</ math>}}: out channel index

## Pooling
### Max Pooling
$$
Y_{ijhw}=\max_{u\in[0,H_{filt}-1]}\max_{v\in[0,W_{filt}-1]}X_{ij,H_{filt}*h+u,W_{filt}*w+u}
$$

Idea: pool images by selecting the max element in each filter window (the equation is stupid, just visualize it)

Notations:
- {{< math >}}$ X$: input tensor of shape $(m,C,H_{in},W_{in}) ${{</ math>}}
- {{< math >}}$ Y$: output tensor of shape $(m,C,H_{out},W_{out}) ${{</ math>}}
- {{< math >}}$ m ${{</ math>}}: (hyperparam) batch size
- {{< math >}}$ C ${{</ math>}}: #channels
- {{< math >}}$ (H_{out},W_{out})=\left(\lfloor\frac{H_{in}+2p}{H_{filt}}\rfloor, \lfloor\frac{W_{in}+2p}{W_{filt}}\rfloor\right) ${{</ math>}} (stride size is filter size in pooling)
- {{< math >}}$ p ${{</ math>}}: padding size

### Average Pooling
$$
Y_{ijhw}=\frac{1}{H_{filt}W_{filt}}\sum_{u=0}^{H_{filt}-1}\sum_{v=0}^{W_{filt}-1}X_{ij,H_{filt}*h+u,W_{filt}*w+u}
$$

Idea: pool images by selecting the max element in each filter window (the equation is stupid, just visualize it)

Notations:
- {{< math >}}$ X$: input tensor of shape $(m,C,H_{in},W_{in}) ${{</ math>}}
- {{< math >}}$ Y$: output tensor of shape $(m,C,H_{out},W_{out}) ${{</ math>}}
- {{< math >}}$ m ${{</ math>}}: (hyperparam) batch size
- {{< math >}}$ C ${{</ math>}}: #channels
- {{< math >}}$ (H_{out},W_{out})=\left(\lfloor\frac{H_{in}+2p}{H_{filt}}\rfloor, \lfloor\frac{W_{in}+2p}{W_{filt}}\rfloor\right) ${{</ math>}} (stride size is filter size in pooling)
- {{< math >}}$ p ${{</ math>}}: padding size

## NiN block (Network in Network)
Idea: 1{{< math >}}$ \times ${{</ math>}}1 convs (+ global average pooling)
- significantly improve computational efficiency while keeping the matrix size
- and at the same time add local nonlinearities across channel activations

&nbsp;

# RNN
## RNN

<center>
<img src="/images/dl/RNN.png" width="400"/>
</center>

$$
h_t=\tanh(x_tW_{xh}^T+h_{t-1}W_{hh}^T)
$$

Idea: **recurrence** - maintain a hidden state that captures information about previous inputs in the sequence

Notations:
- {{< math >}}$ x_t$: input at time $t$ of shape $(m,H_{in}) ${{</ math>}}
- {{< math >}}$ h_t$: hidden state at time $t$ of shape $(D,m,H_{out}) ${{</ math>}}
- {{< math >}}$ W_{xh}$: weight matrix of shape $(H_{out},H_{in})$ if initial layer, else $(H_{out},DH_{out}) ${{</ math>}}
- {{< math >}}$ W_{hh}$: weight matrix of shape $(H_{out},H_{out}) ${{</ math>}}
- {{< math >}}$ H_{in}$: input size, #features in $x_t ${{</ math>}}
- {{< math >}}$ H_{out}$: hidden size, #features in $h_t ${{</ math>}}
- {{< math >}}$ m ${{</ math>}}: batch size
- {{< math >}}$ D$: $=2$ if bi-directional else $1 ${{</ math>}}

Cons:
- Short-term memory: hard to carry info from earlier steps to later ones if long seq
- Vanishing gradient: gradients in earlier parts become extremely small if long seq

## GRU

<center>
<img src="/images/dl/GRU.png" width="400"/>
</center>

{{< math class=text-center >}}$$\begin{align*}
&r_t=\sigma(x_tW_{xr}^T+h_{t-1}W_{hr}^T) \\\\
&z_t=\sigma(x_tW_{xz}^T+h_{t-1}W_{hz}^T) \\\\
&\tilde{h}\_t=\tanh(x_tW_{xn}^T+r_t\odot(h_{t-1}W_{hn}^T)) \\\\
&h_t=(1-z_t)\odot\tilde{h}\_t+z_t\odot h_{t-1}
\end{align*}$${{< /math >}}

Idea: Gated Recurrent Unit - use 2 gates to address long-term info propagation issue in RNN:
1. **Reset gate**: determine how much of {{< math >}}$ h_{t-1}$ should be ignored when computing $\tilde{h}\_t ${{</ math>}}.
2. **Update gate**: determine how much of {{< math >}}$ h_{t-1}$ should be retained for $h_t ${{</ math>}}.
3. **Candidate**: calculate candidate {{< math >}}$ \tilde{h}\_t$ with reset $h_{t-1} ${{</ math>}}.
4. **Final**: calculate weighted average between candidate {{< math >}}$ \tilde{h}\_t$ and prev state $h_{t-1} ${{</ math>}} with the retain ratio.

Notations:
- {{< math >}}$ r_t$: reset gate at time $t$ of shape $(m,H_{out}) ${{</ math>}}
- {{< math >}}$ z_t$: update gate at time $t$ of shape $(m,H_{out}) ${{</ math>}}
- {{< math >}}$ \tilde{h}\_t$: candidate hidden state at time $t$ of shape $(m,H_{out}) ${{</ math>}}
- {{< math >}}$ \odot ${{</ math>}}: element-wise product

## LSTM

<center>
<img src="/images/dl/LSTM.png" width="400"/>
</center>

{{< math class=text-center >}}$$\begin{align*}
&i_t=\sigma(x_tW_{xi}^T+h_{t-1}W_{hi}^T) \\\\
&f_t=\sigma(x_tW_{xf}^T+h_{t-1}W_{hf}^T) \\\\
&\tilde{c}\_t=\tanh(x_tW_{xc}^T+h_{t-1}W_{hc}^T) \\\\
&c_t=f_t\odot c_{t-1}+i_t\odot \tilde{c}\_t \\\\
&o_t=\sigma(x_tW_{xo}^T+h_{t-1}W_{ho}^T) \\\\
&h_t=o_t\odot\tanh(c_t)
\end{align*}$${{< /math >}}

Idea: Long Short-Term Memory - use 3 gates:
1. **Input gate**: determine what new info from {{< math >}}$ x_t$ should be added to cell state $c_t ${{</ math>}}.
2. **Forget gate**: determine what info from prev cell {{< math >}}$ c_{t-1} ${{</ math>}} should be forgotten.
3. **Candidate cell**: create a new candidate cell from {{< math >}}$ x_t$ and $h_{t-1} ${{</ math>}}.
4. **Update cell**: use {{< math >}}$ i_t$ and $f_t ${{</ math>}} to combine prev and new candidate cells.
5. **Output gate**: determine what info from curr cell {{< math >}}$ c_t$ should be added to output $h_t ${{</ math>}}.
6. **Final**: simply apply {{< math >}}$ o_t$ to activated cell $c_t ${{</ math>}}.

Notations:
- {{< math >}}$ i_t$: input gate at time $t$ of shape $(m,H_{out}) ${{</ math>}}
- {{< math >}}$ f_t$: forget gate at time $t$ of shape $(m,H_{out}) ${{</ math>}}
- {{< math >}}$ c_t$: cell state at time $t$ of shape $(m,H_{cell}) ${{</ math>}}
- {{< math >}}$ o_t$: output gate at time $t$ of shape $(m,H_{out}) ${{</ math>}}
- {{< math >}}$ H_{cell}$: cell hidden size (in most cases same as $H_{out} ${{</ math>}})

&nbsp;

&nbsp;

# Transformer

<center>
<img src="/images/dl/transformer.png" width="500"/>
</center>

**What**: Transformer exploits self-attention mechanisms for sequential data.

**Why**:
- **long-range dependencies**: directly model relationships between any two positions in the sequence regardless of their distance, whereas RNNs struggle with tokens that are far apart.
- **parallel processing**: process all tokens in parallel, whereas RNNs process them in sequence.
- **flexibility**: can be easily modified and transferred to various structures and tasks.

**Where**: NLP, CV, Speech, Time Series, Generative tasks, ...

**When**:
- sequential data independence: a sequence can be processed in parallel to a certain extent.
- importance of contextual relationships
- importance of high-dimensional representations
- sufficient data & sufficient computational resources

**How**:
- All layers used in transformer:
    - [Positional Encoding](#positional-encoding)
    - [Residual Connection](#residual-block-resnet)
    - [Layer Normalization](#layer-normalization)
    - [Position-wise Feed-Forward Networks](#postion-wise-feed-forward-networks)
    - [Multi-Head Attention](#multi-head-attention)
- Each sublayer follows this structure: {{< math >}}$ \text{LayerNorm}(x+\text{Sublayer}(x)) ${{</ math>}}
- **Input**: input/output token embeddings {{< math >}}$ \xrightarrow{\text{PE}} ${{</ math>}} input for Encoder/Decoder
- **Encoder**: input {{< math >}}$ \xrightarrow{\text{MHA}}$$\xrightarrow{\text{FFN}}$ $K$&$V ${{</ math>}} for Decoder
- **Decoder**: output {{< math >}}$ \xrightarrow{\text{Masked MHA}}$ $Q$ + Encoder's $K$&$V$ $\xrightarrow{\text{MHA}}$$\xrightarrow{\text{FFN}} ${{</ math>}} Decoder embeddings
- **Output**: Decoder embeddings {{< math >}}$ \xrightarrow{\text{Linear}}$ embeddings shaped for token prediction $\xrightarrow{\text{Softmax}} ${{</ math>}} token probabilities

**Training**:
- **Parameters**:
    - Encoder: {{< math >}}$ \text{\\#params}=h\cdot d\_{\text{model}}\cdot (2d\_k+d\_v)+2\cdot d\_{\text{model}}\cdot d\_{\text{ff}} ${{</ math>}}
    - Decoder: {{< math >}}$ \text{\\#params}=2\cdot h\cdot d\_{\text{model}}\cdot (2d\_k+d\_v)+2\cdot d\_{\text{model}}\cdot d\_{\text{ff}} ${{</ math>}}
- **Hyperparameters**:
    - #layers
    - hidden size
    - #heads
    - learning rate (& warm-up steps)

**Inference**:
1. Process input tokens in parallel via encoder.
2. Generate output tokens sequentially via decoder.

**Pros**:
- high computation efficiency (training & inference)
- high performance
- wide applicability

**Cons**:
- require sufficient computation resources
- require sufficient large-scale data

&nbsp;

## Positional Encoding
**What**: Positional Encoding encodes sequence order info of tokens into embeddings.

**Why**: So that the model can still make use of the sequence order info since no recurrence/convolution is available for it.

**Where**: After tokenization & Before feeding into model.

**When**: The hypothesis that **relative positions** allow the model to learn to attend easier holds.

**How**: sinusoid with wavelengths from a geometric progression from {{< math >}}$ 2\pi$ to $10000\cdot2\pi ${{</ math>}}
{{< math class=text-center >}}$$\begin{align*}
\text{PE}\_{(pos,2i)}&=\sin(\frac{pos}{10000^{\frac{2i}{d_\text{model}}}})\\\\
\text{PE}\_{(pos,2i+1)}&=\cos(\frac{pos}{10000^{\frac{2i}{d_\text{model}}}})
\end{align*}$${{< /math >}}
- {{< math >}}$ pos ${{</ math>}}: absolute position of the token
- {{< math >}}$ i ${{</ math>}}: dimension
- For any fixed offset {{< math >}}$ k$, $PE_{pos+k}$ is a linear function of $PE_{pos} ${{</ math>}}.

**Pros**:
- allow model to extrapolate to sequence lengths longer than the training sequences

**Cons**: ???

&nbsp;

## Scaled Dot-Product Attention

<center>
<img src="/images/dl/scaled_dot_product_attention.png" width="200"/>
</center>

**What**: An effective & efficient variation of self-attention.

**Why**: 
- The end goal is **Attention** - "Which parts of the sentence should we focus on?"
- We want to **capture the most relevant info** in the sentence.
- And we also want to **keep track of all info** in the sentence as well, just with different weights.
- We want to create **contextualized representations** of the sentence.
- Therefore, attention mechanism - we want to assign different attention scores to each token.

**When**: 
- **linearity**: Relationship between tokens can be captured via linear transformation.
- **Position independence**: Relationship between tokens are independent of positions (fixed by Positional Encoding).

**How**:
$$
\text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

- Preliminaries:
    - **Query (Q)**: a **question** about a token - "How important is this token in the context of the whole sentence?"
    - **Key (K)**: a piece of **unique identifier** about a token - "Here's something unique about this token."
    - **Value (V)**: the **actual meaning** of a token - "Here's the content about this token."
- Procedure:
    1. **Compare the similarity** between the **Q** of one word and the **K** of every other word.
        - The more similar, the more attention we should give to that word for the queried word.
    2. **Scale down** by {{< math >}}$ \sqrt{d_k} ${{</ math>}} to avoid the similarity scores being too large.
        - Dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients.
        - They grow large because, if {{< math >}}$ q,k\sim N(0,1)$, then $qk=\sum_{i=1}^{d_k}q_ik_i\sim N(0,d_k) ${{</ math>}}.
    3. Convert the attention scores into a **probability distribution**.
        - Softmax sums up to 1 and emphasizes important attention weights (and reduces the impact of negligible ones).
    4. Calculate the **weighted combination** of all words, for each queried word, as the final attention score.

**Pros**:
- significantly higher computational efficiency (time & space) than additive attention

**Cons**: 
- outperformed by additive attention if without scaling for large values of {{< math >}}$ d_k ${{</ math>}}

&nbsp;

## Multi-Head Attention

<center>
<img src="/images/dl/mha.png" width="300"/>
</center>

**What**: A combination of multiple scaled dot-product attention heads in parallel.
- Masked MHA: mask the succeeding tokens off because they can't be seen during decoding.

**Why**: To allow the model to jointly attend to info from different representation subspaces at different positions.

**When**: The assumption of independence of attention heads holds.

**How**: 
{{< math class=text-center >}}$$\begin{align*}
\text{MultiHead}(Q,K,V)&=\text{Concat}(\text{head}_1,\cdots,\text{head}_h)W^O \\\\
\text{head}_i&=\text{Attention}(QW_i^Q,KW_i^K,VW_i^V)
\end{align*}$${{< /math >}}
- {{< math >}}$ W_i^Q\in\mathbb{R}^{d_\text{model}\times d_k},W_i^K\in\mathbb{R}^{d_\text{model}\times d_k},W_i^V\in\mathbb{R}^{d_\text{model}\times d_v} ${{</ math>}}: learnable linear projection params.
- {{< math >}}$ W^O\in\mathbb{R}^{d_\text{model}\times hd_v} ${{</ math>}}: learnable linear combination weights.
- {{< math >}}$ h=8, d_k=d_v=\frac{d_\text{model}}{h}=64 ${{</ math>}} in the original paper.

**Pros**:
- better performance than single head

**Cons**: ???

## Postion-wise Feed-Forward Networks
**What**: 2 linear transformations with ReLU in between.

**Why**: Just like 2 convolutions with kernel size 1.

**How**:
$$
\text{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2
$$
- {{< math >}}$ d_\text{model}=512 ${{</ math>}}
- {{< math >}}$ d_\text{FF}=2048 ${{</ math>}}