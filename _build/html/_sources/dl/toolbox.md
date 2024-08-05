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

# Toolbox
This page is a collection of tools that are widely used in model construction and experimentation, including
- Regularization
- Optimization
- Activation

## Regularization
Regularization reduces overfitting by adding our prior belief onto loss minimization (i.e., MAP estimate). Such prior belief is associated with our expectation of test error.
- Regularization makes a model inconsistent, biased, and scale variant.
- Regularization requires extra hyperparameter tuning.
- Regularization has a weaker effect on the model (params) as the sample size increases, because more info become available from the data.

### Early Stopping
Idea: stop the training process when no visible improvement is observed on validation data but on training data.

Pros: 
- faster training & higher computational efficiency

Cons:
- premature stopping -> underfitting
- sensitivity to hyperparams
- inconsistency in results if stopped at different points

### Penalty
Idea: add penalty terms in the loss function to force NN weights to be small. (see [Penalty](../../ml/toolbox/#regularization))

Pros:
- help keep NN simple
- improve model robustness

Cons:
- oversimplification

### Data Augmentation
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

### Dropout
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

### Normalization
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



## Optimization
Optimization means the adjustment of params to minimize/maximize an objective function. In DL, it involves 5 key components:
- **Loss Function**: Difference between predicted output $ \hat{y}$ and actual output $y $.
- **Gradient Descent**: Iteratively use loss gradient to update params to reduce loss. While other optimization methods exist, GD and its variations are the best.
- **Learning Rate**: Step size taken during each iteration, controlling convergence and stability of GD.
- **Epochs**: #times to go through the entire dataset.
- **Batch Size**: #samples in a batch, which impacts how often params are updated.

### Gradient Descent
$$\begin{align*}
&\text{Basic ver.:}             &&g_t=\nabla_w\mathcal{L}(w_{t-1})\\\\
&\text{L2 regularization ver.:} &&g_t=\nabla_w\mathcal{L}(w_{t-1})+\lambda w_{t-1}\\\\
&\text{Weight update:} && w_t=w_{t-1}-\eta g_t\\\\
\end{align*}$$

Notations:
- $ w_t $: param
- $ \eta $: learning rate
- $ g_t $: gradient
- $ \mathcal{L} $: loss
- $ \lambda $: L2 penalty weight

Types:
- **Stochastic GD**: update params after each sample
- **Mini-Batch GD**: update params after each mini-batch of samples
- **Batch GD**: update params after the entire dataset

Pros:
- simple

Cons:
- stuck in local minima or saddle points
- sensitive to learning rate

### Momentum
$$\begin{align*}
&v_t=\beta v_{t-1}+(1-\beta)g_t\\\\
&w_t=w_{t-1}-\eta v_t
\end{align*}$$

Notations:
- $ \beta $: momentum weight
    - larger $ \rightarrow $ smoother updates due to more past gradients involved
    - typical values: 0.8, 0.9, 0.999

Idea: moving average of past gradients

Pros:
- accelerate convergence
- reduce oscillations & noises
- escape local minima & saddle points

Cons:
- sensitive to hyperparams
- overshooting: the weight update jumps over the global minimum

### NAG
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

### AdaGrad
$$\begin{align*}
&v_t=v_{t-1}+g_t^2\\\\
&w_t=w_{t-1}-\frac{\eta}{\sqrt{v_t}+\epsilon}g_t
\end{align*}$$

Notations:
- $ \epsilon $: small number to ensure no division by 0.

Name: Adaptive Gradient Algorithm

Idea: adapt learning rate for each param

Pros:
- adaptive learning rate -> improve robustness
- efficient for sparse data (where some features have larger gradients than others)

Cons:
- small learning rate for frequently occurring features -> slow convergence or premature stopping

### Adadelta
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

### RMSProp
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

### Adam
$$\begin{align*}
&m_t=\beta_1m_{t-1}+(1-\beta_1)g_t\\\\
&v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2\\\\
&\hat{m}_t=\frac{m_t}{1-\beta_1^t}\\\\
&\hat{v}_t=\frac{v_t}{1-\beta_2^t}\\\\
&w_t=w\_{t-1}-\frac{\eta}{\sqrt{\hat{v}_t}+\epsilon}\hat{m}_t
\end{align*}$$

Notations:
- $ m_t $: first moment (adaptive gradient)
- $ v_t $: second moment (adaptive learning rate)
- $ \hat{m}_t, \hat{v}_t $: bias-corrected moments

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

### AdamW
$$
w_t=w_{t-1}-\frac{\eta}{\sqrt{\hat{v}_t}+\epsilon}\hat{m}_t-\lambda w\_{t-1}
$$

Idea: Adam + Weight Decay

Pros:
- Well, weight decay so regularization