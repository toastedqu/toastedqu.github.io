---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Activation
An activation function adds nonlinearity to the output of a layer (linear in most cases) to enhance complexity.

[ReLU](#relu) and [Softmax](#softmax) are SOTA.

Notations:
- $ z $: input (element-wise)

## Binary-like

### Sigmoid

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Idea:
-  $ \sigma(z)\in(0,1)$ and $\sigma(0)=0.5 $.

Pros:
-  imitation of the firing rate of a neuron, 0 if too negative and 1 if too positive.
-  smooth gradient.

Cons: 
-  vanishing gradient: gradients rapidly shrink to 0 along backprop as long as any input is too positive or too negative.
-  non-zero centric bias $ \rightarrow $ non-zero mean activations.
-  computationally expensive.

### Tanh

$$
\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}
$$

Idea:
- $ \tanh(z)\in(-1,1)$ and $\tanh(0)=0 $.
    
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
- dying ReLU: if most inputs are negative, then most neurons output 0 $ \rightarrow$ no gradient for such neurons $\rightarrow$ no param update $\rightarrow $ they die. (NOTE: A SOLVABLE DISADVANTAGE)

    - Cause 1: high learning rate $ \rightarrow$ too much subtraction in param update $\rightarrow$ weight too negative $\rightarrow $ input for neuron too negative.
    - Cause 2: bias too negative $ \rightarrow $ input for neuron too negative.

-  activation explosion as $ z\rightarrow\infty $. (NOTE: NOT A SEVERE DISADVANTAGE SO FAR)
    

### LReLU

$$
\mathrm{LReLU}(z)=\max{(\alpha z,z)}
$$


Name: Leaky Rectified Linear Unit

Params: 
- $ \alpha\in(0,1) $: hyperparam (negative slope), default 0.01. 
    
Idea:
- scale negative linear outputs by $ \alpha $.
    
Pros:
- no dying ReLU.
    
Cons: 
- slightly more computationally expensive than ReLU.
- activation explosion as $ z\rightarrow\infty $.


### PReLU

$$
\mathrm{PReLU}(z)=\max{(\alpha z,z)}
$$

Name: Parametric Rectified Linear Unit

Params: 
- $ \alpha\in(0,1) $: learnable parameter (negative slope), default 0.25.
    
Idea:
- scale negative linear outputs by a learnable $ \alpha $.
    
Pros: 
- a variable, adaptive parameter learned from data.
    
Cons: 
- slightly more computationally expensive than LReLU.
- activation explosion as $ z\rightarrow\infty $.



### RReLU

$$
\mathrm{RReLU}(z)=\max{(\alpha z,z)}
$$


Name: Randomized Rectified Linear Unit

Params:     
- $ \alpha\sim\mathrm{Uniform}(l,u) $: a random number sampled from a uniform distribution.
- $ l,u $: hyperparams (lower bound, upper bound)
    
Idea:
- scale negative linear outputs by a random $ \alpha $.
    
Pros: 
- reduce overfitting by randomization.
    
Cons: 
- slightly more computationally expensive than LReLU.
- activation explosion as $ z\rightarrow\infty $.
    
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
- $ \alpha $: hyperparam, default 1.
    
Idea:
- convert negative linear outputs to the non-linear exponential function above.
    
Pros: 
- mean unit activation is closer to 0 $ \rightarrow $ reduce bias shift (i.e., non-zero mean activation is intrinsically a bias for the next layer.)
- lower computational complexity compared to batch normalization.
- smooth to $ -\alpha $ slowly with smaller derivatives that decrease forwardprop variation.
- faster learning and higher accuracy for image classification in practice.
    
Cons: 
- slightly more computationally expensive than ReLU.
- activation explosion as $ z\rightarrow\infty $.
    


### SELU

$$
\mathrm{SELU}(z)=\lambda\begin{cases}
z & \mathrm{if}\ z\geq0 \\
\alpha(e^z-1) & \mathrm{if}\ z<0
\end{cases}
$$


Name: Scaled Exponential Linear Unit

Params:
- $ \alpha $: hyperparam, default 1.67326.
- $ \lambda $: hyperparam (scale), default 1.05070.
    
Idea:
- scale ELU.
    
Pros: 
- self-normalization $ \rightarrow $ activations close to zero mean and unit variance that are propagated through many network layers will converge towards zero mean and unit variance.
    
Cons:
- more computationally expensive than ReLU.
- activation explosion as $ z\rightarrow\infty $.
    


### CELU

$$
\mathrm{CELU}(z)=\begin{cases}
z & \mathrm{if}\ z\geq0\\
\alpha(e^{\frac{z}{\alpha}}-1) & \mathrm{if}\ z<0
\end{cases}
$$


Name: Continuously Differentiable Exponential Linear Unit

Params:
- $ \alpha $: hyperparam, default 1.
    
Idea:
- scale the exponential part of ELU with $ \frac{1}{\alpha} $ to make it continuously differentiable.
    
Pros:
- smooth gradient due to continuous differentiability (i.e., $ \mathrm{CELU}'(0)=1 $).
    
Cons:
- slightly more computationally expensive than ELU.
- activation explosion as $ z\rightarrow\infty $.
    
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
- convert each value $ z_i$ in the output tensor $\mathbf{z}$ into its corresponding exponential probability s.t. $\sum_i{\mathrm{softmax}(z_i)}=1 $.
    
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