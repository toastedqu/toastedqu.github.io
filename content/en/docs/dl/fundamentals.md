---
title : "Fundamentals"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
# draft: true
images: []
weight: 200
---
Very loosely written. Just a toolbox.

# Layers
A layer is fundamentally a function that transforms input $X$ into output $Y$.

(Note: I believe RNN is dead, so I won't waste time on it.)

## Basic Layers

### Linear

$$
Y=\sigma(XW^T+\textbf{b})
$$

Idea: linear transformation

Notations:
- $X$: input tensor of shape $(*, H_{in})$
- $Y$: output tensor of shape $(*, H_{out})$
- $W$: weight matrix of shape $(H_{out}, H_{in})$
- $\textbf{b}$: bias vector of size $(H_{out})$

### Dropout

$$
Y=\mathrm{Dropout}(X, p)
$$

Idea: randomly make some elements become 0 with specified probability

Notations:
- $X$: input tensor of arbitrary shape
- $Y$: output tensor of the input shape
- $p\in(0,1)$: dropout probability

### Normalization

#### Batch Normalization

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

#### Layer Normalization

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

## CNN Layers

### Convolutional

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

### Pooling

#### Max Pooling

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

#### Average Pooling

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

# Activation
An activation function adds nonlinearity to the output of a layer (linear in most cases) to enhance complexity.

Honestly, this field is growing so rapidly that [ReLU](#relu) and [Softmax](#softmax) are the only activation functions that matter nowadays.

Notations:
- $z$: input (applied element-wise so shape doesn't matter)

## Binary-like

### Sigmoid

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Idea:
-  $\sigma(z)\in(0,1)$ and $\sigma(0)=0.5$.

Pros:
-  imitation of the firing rate of a neuron, 0 if too negative and 1 if too positive.
-  smooth gradient.

Cons: 
-  vanishing gradient: gradients rapidly shrink to 0 along backprop as long as any input is too positive or too negative.
-  non-zero centric bias $\rightarrow$ non-zero mean activations.
-  computationally expensive.

### Tanh

$$
\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}
$$

Idea:
- $\tanh(z)\in(-1,1)$ and $\tanh(0)=0$.
    
Pros: 
- zero-centered
- imitation of the firing rate of a neuron, -1 if too negative and 1 if too positive.
- smooth gradient.
    
Cons:
- vanishing gradient.
- computationally expensive.

## Linear Units (Rectified)

### ReLU

$$
\mathrm{ReLU}(z)=\max{(0,z)}
$$

Name: Rectified Linear Unit

Idea:
- convert negative linear outputs to 0.
    
Pros:
- no vanishing gradient
- activate fewer neurons
- much less computationally expensive compared to sigmoid and tanh.
    
Cons:
- dying ReLU: if most inputs are negative, then most neurons output 0 $\rightarrow$ no gradient for such neurons $\rightarrow$ no param update $\rightarrow$ they die. (NOTE: A SOLVABLE DISADVANTAGE)

    - Cause 1: high learning rate $\rightarrow$ too much subtraction in param update $\rightarrow$ weight too negative $\rightarrow$ input for neuron too negative.
    - Cause 2: bias too negative $\rightarrow$ input for neuron too negative.

-  activation explosion as $z\rightarrow\infty$. (NOTE: NOT A SEVERE DISADVANTAGE SO FAR)
    


### LReLU

$$
\mathrm{LReLU}(z)=\max{(\alpha z,z)}
$$


Name: Leaky Rectified Linear Unit

Params: 
- $\alpha\in(0,1)$: hyperparam (negative slope), default 0.01. 
    
Idea:
- scale negative linear outputs by $\alpha$.
    
Pros:
- no dying ReLU.
    
Cons: 
- slightly more computationally expensive than ReLU.
- activation explosion as $z\rightarrow\infty$.


### PReLU

$$
\mathrm{PReLU}(z)=\max{(\alpha z,z)}
$$

Name: Parametric Rectified Linear Unit

Params: 
- $\alpha\in(0,1)$: learnable parameter (negative slope), default 0.25.
    
Idea:
- scale negative linear outputs by a learnable $\alpha$.
    
Pros: 
- a variable, adaptive parameter learned from data.
    
Cons: 
- slightly more computationally expensive than LReLU.
- activation explosion as $z\rightarrow\infty$.



### RReLU

$$
\mathrm{RReLU}(z)=\max{(\alpha z,z)}
$$


Name: Randomized Rectified Linear Unit

Params:     
- $\alpha\sim\mathrm{Uniform}(l,u)$: a random number sampled from a uniform distribution.
- $l,u$: hyperparams (lower bound, upper bound)
    
Idea:
- scale negative linear outputs by a random $\alpha$.
    
Pros: 
- reduce overfitting by randomization.
    
Cons: 
- slightly more computationally expensive than LReLU.
- activation explosion as $z\rightarrow\infty$.
    
## Linear Units (Exponential)

### ELU

$$
\mathrm{ELU}(z)=\begin{cases}
z & \mathrm{if}\ z\geq0 \\\\
\alpha(e^z-1) & \mathrm{if}\ z<0
\end{cases}
$$


Name: Exponential Linear Unit

Params:
- $\alpha$: hyperparam, default 1.
    
Idea:
- convert negative linear outputs to the non-linear exponential function above.
    
Pros: 
- mean unit activation is closer to 0 $\rightarrow$ reduce bias shift (i.e., non-zero mean activation is intrinsically a bias for the next layer.)
- lower computational complexity compared to batch normalization.
- smooth to $-\alpha$ slowly with smaller derivatives that decrease forwardprop variation.
- faster learning and higher accuracy for image classification in practice.
    
Cons: 
- slightly more computationally expensive than ReLU.
- activation explosion as $z\rightarrow\infty$.
    


### SELU

$$
\mathrm{SELU}(z)=\lambda\begin{cases}
z & \mathrm{if}\ z\geq0 \\
\alpha(e^z-1) & \mathrm{if}\ z<0
\end{cases}
$$


Name: Scaled Exponential Linear Unit

Params:
- $\alpha$: hyperparam, default 1.67326.
- $\lambda$: hyperparam (scale), default 1.05070.
    
Idea:
- scale ELU.
    
Pros: 
- self-normalization $\rightarrow$ activations close to zero mean and unit variance that are propagated through many network layers will converge towards zero mean and unit variance.
    
Cons:
- more computationally expensive than ReLU.
- activation explosion as $z\rightarrow\infty$.
    


### CELU

$$
\mathrm{CELU}(z)=\begin{cases}
z & \mathrm{if}\ z\geq0\\
\alpha(e^{\frac{z}{\alpha}}-1) & \mathrm{if}\ z<0
\end{cases}
$$


Name: Continuously Differentiable Exponential Linear Unit

Params:
- $\alpha$: hyperparam, default 1.
    
Idea:
- scale the exponential part of ELU with $\frac{1}{\alpha}$ to make it continuously differentiable.
    
Pros:
- smooth gradient due to continuous differentiability (i.e., $\mathrm{CELU}'(0)=1$).
    
Cons:
- slightly more computationally expensive than ELU.
- activation explosion as $z\rightarrow\infty$.
    
## Linear Units (Others)

### GELU

$$
\mathrm{GELU}(z)=z*\Phi(z)=0.5z(1+\tanh{[\sqrt{\frac{2}{\pi}}(z+0.044715z^3)]})
$$


Name: Gaussian Error Linear Unit

Idea:
- weigh each output value by its Gaussian cdf.
    
Pros: 
- throw away gate structure and add probabilistic-ish feature to neuron outputs.
- seemingly better performance than the ReLU and ELU families, SOTA in transformers.
    
Cons:    
- slightly more computationally expensive than ReLU.
- lack of practical testing at the moment.
    


### SiLU

$$
\mathrm{SiLU}(z)=z*\sigma(z)
$$

Name: Sigmoid Linear Unit

Idea:   
- weigh each output value by its sigmoid value.
    
Pros: 
- throw away gate structure.
- seemingly better performance than the ReLU and ELU families.
    
Cons: 
- worse than GELU.
    


### Softplus

$$
\mathrm{softplus}(z)=\frac{1}{\beta}\log{(1+e^{\beta z})}
$$


Idea:
- smooth approximation of ReLU.
    
Pros: 
- differentiable and thus theoretically better than ReLU. 
    
Cons: 
- empirically far worse than ReLU in terms of computation and performance.
    


## Multiclass

### Softmax

$$
\mathrm{softmax}(z_i)=\frac{\exp{(z_i)}}{\sum_j{\exp{(z_j)}}}
$$


Idea:
- convert each value $z_i$ in the output tensor $\mathbf{z}$ into its corresponding exponential probability s.t. $\sum_i{\mathrm{softmax}(z_i)}=1$.
    
Pros: 
- your single best choice for multiclass classification.
    
Cons: 
- mutually exclusive classes (i.e., one input can only be classified into one class.)

### Softmin

$$
\mathrm{softmin}(z_i)=\mathrm{softmax}(-z_i)=\frac{\exp{(-z_i)}}{\sum_j{\exp{(-z_j)}}}
$$

Idea:
- reverse softmax.
    
Pros: 
- suitable for multiclass classification.
    
Cons:
- why not softmax.