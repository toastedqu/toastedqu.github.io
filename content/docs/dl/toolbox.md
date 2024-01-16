---
title : "Toolbox"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: false
images: []
weight: 3
---
This page is a collection of tools that are widely used in model construction and experimentation, including
- Regularization
- Optimization
- Activation

&nbsp;

# Regularization
Regularization reduces overfitting by adding our prior belief onto loss minimization (i.e., MAP estimate). Such prior belief is associated with our expectation of test error.
- Regularization makes a model inconsistent, biased, and scale variant.
- Regularization requires extra hyperparameter tuning.
- Regularization has a weaker effect on the model (params) as the sample size increases, because more info become available from the data.

## Early Stopping
Idea: stop the training process when no visible improvement is observed on validation data but on training data.

Pros: 
- faster training & higher computational efficiency

Cons:
- premature stopping -> underfitting
- sensitivity to hyperparams
- inconsistency in results if stopped at different points

## Penalty
Idea: add penalty terms in the loss function to force NN weights to be small. (see [Penalty](../../ml/toolbox/#regularization))

Pros:
- help keep NN simple
- improve model robustness

Cons:
- oversimplification

## Data Augmentation
Idea: expand training data (mostly used in CV)

CV:
- Position: rotate, flip, zoom, translate/shift, shear, scale, cut, erase, etc.
- Color: brightness, contrast, jittering, noise injection, channel shuffle, grid distortion, etc.

Pros:
- improve translation invariance
- improve model robustness
- improve generalization

Cons:
- high computational cost
- may include artifacts

## Dropout
Idea: randomly drop out some neurons during training.

Pros:
- ensemble learning effect
- improve training computational efficiency
- handle correlated neurons
- reduce sensitivity to weight initialization

Cons:
- interference with learning -> slower convergence rate
- sensitive to hyperparam (dropout prob)
- unnecessary if sufficient training data on simpler NNs

## Normalization
Idea: normalize inputs to a layer with zero mean and unit variance (across samples or features) (see [Normalization](../layer/#normalization))

Pros:
- improve convergence & stabilize learning
- allow higher learning rates
- sequence independence (layer norm)
- batch size independence (layer norm)

Cons:
- less computational efficiency
- sequence dependency (batch norm)
- batch size dependency (batch norm)

&nbsp;

# Optimization
Optimization means the adjustment of params to minimize/maximize an objective function. In DL, it involves 5 key components:
- **Loss Function**: Difference between predicted output $\hat{y}$ and actual output $y$.
- **Gradient Descent**: Iteratively use loss gradient to update params to reduce loss. While other optimization methods exist, GD and its variations are the best.
- **Learning Rate**: Step size taken during each iteration, controlling convergence and stability of GD.
- **Epochs**: #times to go through the entire dataset.
- **Batch Size**: #samples in a batch, which impacts how often params are updated.

## Gradient Descent
$$\begin{align*}
&\text{Basic ver.:}             &&g_t=\nabla_w\mathcal{L}(w_{t-1})\\\\
&\text{L2 regularization ver.:} &&g_t=\nabla_w\mathcal{L}(w_{t-1})+\lambda w_{t-1}\\\\
&\text{Weight update:} && w_t=w_{t-1}-\eta g_t\\\\
\end{align*}$$

Notations:
- $w_t$: param
- $\eta$: learning rate
- $g_t$: gradient
- $\mathcal{L}$: loss
- $\lambda$: L2 penalty weight

Types:
- **Stochastic GD**: update params after each sample
- **Mini-Batch GD**: update params after each mini-batch of samples
- **Batch GD**: update params after the entire dataset

Pros:
- simple

Cons:
- stuck in local minima or saddle points
- sensitive to learning rate

## Momentum
$$\begin{align*}
&v_t=\beta v_{t-1}+(1-\beta)g_t\\\\
&w_t=w_{t-1}-\eta v_t
\end{align*}$$

Notations:
- $\beta$: momentum weight
    - larger $\rightarrow$ smoother updates due to more past gradients involved
    - typical values: 0.8, 0.9, 0.999

Idea: moving average of past gradients

Pros:
- accelerate convergence
- reduce oscillations & noises
- escape local minima & saddle points

Cons:
- sensitive to hyperparams
- overshooting: the weight update jumps over the global minimum

## NAG
$$\begin{align*}
&v_t=\beta v_{t-1}+\nabla_w\mathcal{L}(w_{t-1}-\beta v_{t-1})\\\\
&w_t=w_{t-1}-\eta v_t
\end{align*}$$

Name: Nesterov Accelerated Gradient

Idea: momentum but look ahead to make an informed update

Pros:
- further accelerate convergence, espeically near minima
- further reduce overshooting
- more accurate weight updates in rapidly changing regions
- improve robustness to hyperparams

Cons:
- implementation complexity
- low computational efficiency
- still sensitive to learning rate

## AdaGrad
$$\begin{align*}
&v_t=v_{t-1}+g_t^2\\\\
&w_t=w_{t-1}-\frac{\eta}{\sqrt{v_t}+\epsilon}g_t
\end{align*}$$

Notations:
- $\epsilon$: small number to ensure no division by 0.

Name: Adaptive Gradient Algorithm

Idea: adapt learning rate for each param

Pros:
- adaptive learning rate -> improve robustness
- efficient for sparse data (where some features have larger gradients than others)

Cons:
- small learning rate for frequently occurring features -> slow convergence or premature stopping

## Adadelta
$$\begin{align*}
&v_t=\beta v\_{t-1}+(1-\beta)g_t^2\\\\
&\Delta w_t=-\frac{\sqrt{\Delta w_{t-1}^2+\epsilon}}{\sqrt{v_t}+\epsilon}g_t\\\\
&w_t=w\_{t-1}+\Delta w_t
\end{align*}$$

Idea: address small learning rate in AdaGrad by using a window of past gradients to normalize updates

Pros:
- introduced moving average in adagrad to adapt to changes more effectively
- no need for learning rate initialization
- robust to varying gradients

Cons:
- Far too complicated

## RMSProp
$$\begin{align*}
&v_t=\beta v_{t-1}+(1-\beta)g_t^2\\\\
&w_t=w_{t-1}-\frac{\eta}{\sqrt{v_t}+\epsilon}g_t
\end{align*}$$

Name: Root Mean Square Propagation

Idea: Momentum + AdaGrad

Pros:
- simple implementation
- no accumulation of update history

Cons:
- worse than Adam

## Adam
$$\begin{align*}
&m_t=\beta_1m_{t-1}+(1-\beta_1)g_t\\\\
&v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2\\\\
&\hat{m}_t=\frac{m_t}{1-\beta_1^t}\\\\
&\hat{v}_t=\frac{v_t}{1-\beta_2^t}\\\\
&w_t=w\_{t-1}-\frac{\eta}{\sqrt{\hat{v}_t}+\epsilon}\hat{m}_t
\end{align*}$$

Notations:
- $m_t$: first moment (adaptive gradient)
- $v_t$: second moment (adaptive learning rate)
- $\hat{m}_t, \hat{v}_t$: bias-corrected moments

Name: Adaptive Moment Estimation

Idea: adaptive learning rates for both momentum & gradient

Pros:
- SOTA
- bias correction
- extreme adaptivity
- extreme convergence speed
- extremely robust to noisy or sparse gradients

Cons:
- sensitive to hyperparams (3 hyperparams to tune)

## AdamW
$$
w_t=w_{t-1}-\frac{\eta}{\sqrt{\hat{v}_t}+\epsilon}\hat{m}_t-\lambda w\_{t-1}
$$

Idea: Adam + Weight Decay

Pros:
- Well, weight decay so regularization

&nbsp;

# Activation
An activation function adds nonlinearity to the output of a layer (linear in most cases) to enhance complexity.

[ReLU](#relu) and [Softmax](#softmax) are SOTA.

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