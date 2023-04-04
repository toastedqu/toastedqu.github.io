---
title : "Introduction"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: true
images: []
weight: 100
---
The following notations will be used throughout this section unless specified. **bold** letters refer to vectors, and CAP letters refer to matrices in general.
- $m$: #samples in the input batch (i.e., batch size)
- $n_0$: #features in the input sample
- $n_l$: #neurons of the $l$th hidden layer (i.e., hidden size)
- $L$: #hidden layers
- $i$: sample index
- $j$: feature index
- $k$: class index
- $l$: hidden layer index
- $\mathcal{D}$: (training) dataset
- $\Theta$: set of params
- $X$: input samples of shape $(m,\cdots,n_0)$
- $H^{[l]}$: hidden outputs of shape $(m,\cdots,n_l)$
- $W$: weight matrix of shape $()$
- $\textbf{W}^{[l]}=[\textbf{w}_1^{[l]},\cdots,\textbf{w}\_{n^{[l]}}^{[l]}]$: weight matrix of shape $(n^{[l-1]}, n^{[l]})$ (concat $\textbf{b}$ if bias is needed)
- $\mathbf{y}=[y_1,\cdots,y_{m}]^T$: actual label vector of shape $(m,1)$
- $\hat{\mathbf{y}}=[\hat{y}_1,\cdots,\hat{y}_m]^T$: predicted label vector of shape $(m,1)$
- $\textbf{z}$: latent variable(s), shape dependent on situations
- ($X,Y$: occasionally used to represent feature and label as random variables)

Each subsection roughly follows the order below:
- Idea
- Usage
- Background
- Assumption
- Model/Algorithm
- Prediction
- Objective (mainly loss)
- Optimization (mainly param estimation)
- Pros
- Cons
- (extra)

# Neural Network

Most neural networks consist of 2 processes:
- Forward Pass: at each layer, calculate $H^{[l]}=$

<center><img src="../../images/DL/NN.png" width="400"/></center>

**Input Matrix**:

$$\begin{equation}
X=\begin{bmatrix}
x_1^{(1)} & \cdots & x_1^{(m)} \\
\vdots & \ddots & \vdots \\
x_{n_x}^{(1)} & \cdots & x_{n_x}^{(m)}
\end{bmatrix}=\begin{bmatrix}
x^{(1)} & \cdots & x^{(m)}
\end{bmatrix}\quad\quad\quad X\in\mathbb{R}^{n_x\times m}
\end{equation}$$

- $x_j^{(i)}$: the $j$th feature of the $i$th training example
- $m$: # training examples: each column vector of $x$ represents one training example
- $n_x$: # input features: each row vector of $x$ represents one type of input feature

for easier understanding in this session, we use one training example / input vector at each training step:

$$\begin{equation}
x^{(i)}=\begin{bmatrix}
x_1^{(i)} \\ \vdots \\ x_{n_x}^{(i)}
\end{bmatrix}\quad\quad\quad x^{(i)}\in\mathbb{R}^{n_x}
\end{equation}$$

**Output Vector**:

$$\begin{equation}
\hat{Y}=\begin{bmatrix}
\hat{y}^{(1)} & \cdots & \hat{y}^{(m)}
\end{bmatrix}\quad\quad\quad \hat{Y}\in\mathbb{R}^{m}
\end{equation}$$

- $\hat{y}^{(i)}$: the predicted output value of the $i$th training example

for easier understanding in this session, we assume that there is only one output value for each training example. The output vector in the training set is denoted without the "$\hat{}$" symbol.

**Weight Matrix**:

$$\begin{equation}
W^{[k]}=\begin{bmatrix}
w_{1,1}^{[k]} & \cdots & w_{1,n_{k-1}}^{[k]} \\
\vdots & \ddots & \vdots \\
w_{n_k,1}^{[k]} & \cdots & w_{n_k,n_{k-1}}^{[k]}
\end{bmatrix}=\begin{bmatrix}
w_1^{[k]} \\ \vdots \\ w_{n_k}^{[k]}
\end{bmatrix}\quad\quad\quad W^{[k]}\in\mathbb{R}^{n_k\times n_{k-1}}
\end{equation}$$

- $w_{j,l}^{[k]}$: the weight value for the $l$th input at the $j$th node on the $k$th layer
- $n_k$: # nodes/neurons on the $k$th layer (the current layer)
- $n_{k-1}$: # nodes/neurons on the $k-1$th layer (the previous layer)

**Bias Vector**:

$$\begin{equation}
b^{[k]}=\begin{bmatrix}
b_1^{[k]} \\ \vdots \\ b_{n_k}^{[k]}
\end{bmatrix}\quad\quad\quad b^{[k]}\in\mathbb{R}^{n_k}
\end{equation}$$

**Linear Combination**:

$$\begin{equation}
z_j^{[k]}=w_j^{[k]}\cdot a^{[k-1]}+b_j^{[k]} \quad\quad\quad z_j^{[k]}\in\mathbb{R}^{n_k}
\end{equation}$$

- $z_j^{[k]}$: the unactivated output value from the $j$th node of the $k$th layer

**Activation**:

$$\begin{equation}
a^{[k]}=\begin{bmatrix}
a_1^{[k]} \\ \vdots \\ a_{n_k}^{[k]}
\end{bmatrix}=\begin{bmatrix}
g(z_1^{[k]}) \\ \vdots \\ g(z_{n_k}^{[k]})
\end{bmatrix}\quad\quad\quad a^{[k]}\in\mathbb{R}^{n_k}
\end{equation}$$

- $g(z)$: Activation function (to add **nonlinearity**)

## Activation Functions

(Blame github pages for not supporting colspan/rowspan)
<table>
    <thead>
        <tr style="text-align: center">
            <th>Sigmoid</th>
            <th>Tanh</th>
            <th>ReLU</th>
            <th>Leaky ReLU</th>
        </tr>
    </thead>
    <tbody style="text-align: center">
        <tr>
            <td>$g(z)=\frac{1}{1+e^{-z}}$</td>
            <td>$g(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}$</td>
            <td>$g(z)=\max{(0,z)}$</td>
            <td>$g(z)=\max{(\varepsilon z,z)}$</td>
        </tr>
        <tr>
            <td><img src="../../images/DL/sigmoid.png" width="100"/></td>
            <td><img src="../../images/DL/tanh.png" width="100"/></td>
            <td><img src="../../images/DL/relu.png" width="100"/></td>
            <td><img src="../../images/DL/leakyrelu.png" width="100"/></td>
        </tr>
        <tr>
            <td><small>$g'(z)=g(z)\cdot (1-g(z))$</small></td>
            <td><small>$g'(z)=1-(g(z))^2$</small></td>
            <td><small>$$g'(z)=\begin{cases} 0&z<0 \\ 1&z>0\end{cases}$$</small></td>
            <td><small>$$g'(z)=\begin{cases} \varepsilon&z<0 \\ 1&z>0\end{cases}$$</small></td>
        </tr>
        <tr>
            <td><small>centered at $y=0.5$<br>$\Rightarrow$only good for binary classification</small></td>
            <td><small>centered at $y=0$<br>$\Rightarrow$better than sigmoid in many cases</small></td>
            <td><small>faster computing<br><strike>vanishing gradient</strike><br>model sparsity (some neurons can be inactivated)</small></td>
            <td><small>faster computing<br><strike>vanishing gradient</strike><br>model sparsity (some neurons can be inactivated)</small></td>
        </tr>
        <tr>
            <td>$|z|\uparrow\uparrow \rightarrow\frac{da}{dz}\approx 0$<br>$\Rightarrow$ vanishing gradient</td>
            <td>$|z|\uparrow\uparrow \rightarrow\frac{da}{dz}\approx 0$<br>$\Rightarrow$ vanishing gradient</td>
            <td>too many neurons get inactivated<br>$\Rightarrow$dying ReLU</td>
            <td>$\varepsilon$ usually set to 0.01<br><strike>dying ReLU</strike><br>widely used on Kaggle</td>
        </tr>
    </tbody>
</table>

- Why need activation funcs? To add nonlinearity.
    1. Suppose $g(z)=z$ (i.e. $\nexists g(z)$)
    2. $\Longrightarrow z^{[1]}=w^{[1]}x+b^{[1]}$
    3. $\Longrightarrow z^{[2]}=w^{[2]}a^{[1]}+b^{[2]}=(w^{[2]}w^{[1]})x+(w^{[2]}b^{[1]}+b^{[2]})=w'x+b'$
    4. This is just linear regression. Hidden layers exist for no reason.

## Training
<a name="fp"></a>
**Forward Propagation**

<center><img src="../../images/DL/fp.png" width="500"/></center>

<a name="bp"></a>
**Backward Propagation**

<center><img src="../../images/DL/bp.png" width="500"/></center>
  
<a name="fbss"></a>
**Example: Forward & Backward Step: Stochastic**: 2 nodes & 3 inputs & no bias

- Forward Step: 

$$\begin{equation}
\begin{bmatrix}
w_{1,1} & w_{1,2} & w_{1,3} \\
w_{2,1} & w_{2,2} & w_{2,3}
\end{bmatrix}\begin{bmatrix}
x_1 \\ x_2 \\ x_3
\end{bmatrix}=\begin{bmatrix}
z_1 \\ z_2
\end{bmatrix}
\end{equation}$$

- Backward Step: 

$$\begin{equation}
\frac{\partial{\mathcal{L}}}{\partial{W}}=\begin{bmatrix}
\frac{\partial{\mathcal{L}}}{\partial{w_{1,1}}} & \frac{\partial{\mathcal{L}}}{\partial{w_{1,2}}} & \frac{\partial{\mathcal{L}}}{\partial{w_{1,3}}} \\
\frac{\partial{\mathcal{L}}}{\partial{w_{2,1}}} & \frac{\partial{\mathcal{L}}}{\partial{w_{2,2}}} & \frac{\partial{\mathcal{L}}}{\partial{w_{2,3}}}
\end{bmatrix}=\begin{bmatrix}
\frac{\partial{\mathcal{L}}}{\partial{z_1}}x_1 & \frac{\partial{\mathcal{L}}}{\partial{z_1}}x_2 & \frac{\partial{\mathcal{L}}}{\partial{z_1}}x_3 \\
\frac{\partial{\mathcal{L}}}{\partial{z_2}}x_1 & \frac{\partial{\mathcal{L}}}{\partial{z_2}}x_2 & \frac{\partial{\mathcal{L}}}{\partial{z_2}}x_3
\end{bmatrix}=\frac{\partial{\mathcal{L}}}{\partial{Z}}x^T
\end{equation}$$
  
<a name="fbsb"></a>
**Example: Forward & Backward Step: Mini-batch**: 2 nodes & 3 inputs & bias & 2 training examples

- Forward Step: 

$$\begin{equation}
\begin{bmatrix}
w_{1,1} & w_{1,2} & w_{1,3} \\
w_{2,1} & w_{2,2} & w_{2,3}
\end{bmatrix}\begin{bmatrix}
x_1^{(1)} & x_1^{(2)} \\ 
x_2^{(1)} & x_2^{(2)} \\ 
x_3^{(1)} & x_3^{(2)}
\end{bmatrix}+\begin{bmatrix}
b_1 \\ b_2
\end{bmatrix}=\begin{bmatrix}
z_1^{(1)} & z_1^{(2)} \\
z_2^{(1)} & z_2^{(2)}
\end{bmatrix}
\end{equation}$$

- Backward Step: 

<center><small>$$\begin{equation}
\frac{\partial{\mathcal{L}}}{\partial{W}}=\begin{bmatrix}
\frac{\partial{\mathcal{L}}}{\partial{w_{1,1}}} & \frac{\partial{\mathcal{L}}}{\partial{w_{1,2}}} & \frac{\partial{\mathcal{L}}}{\partial{w_{1,3}}} \\
\frac{\partial{\mathcal{L}}}{\partial{w_{2,1}}} & \frac{\partial{\mathcal{L}}}{\partial{w_{2,2}}} & \frac{\partial{\mathcal{L}}}{\partial{w_{2,3}}}
\end{bmatrix}=\begin{bmatrix}
\frac{\partial{\mathcal{L}}}{\partial{z_1^{(1)}}}x_1^{(1)}+\frac{\partial{\mathcal{L}}}{\partial{z_1^{(2)}}}x_1^{(2)} & \frac{\partial{\mathcal{L}}}{\partial{z_1^{(1)}}}x_2^{(1)}+\frac{\partial{\mathcal{L}}}{\partial{z_1^{(2)}}}x_2^{(2)} & \frac{\partial{\mathcal{L}}}{\partial{z_1^{(1)}}}x_3^{(1)}+\frac{\partial{\mathcal{L}}}{\partial{z_1^{(2)}}}x_3^{(2)} \\
\frac{\partial{\mathcal{L}}}{\partial{z_2^{(1)}}}x_1^{(1)}+\frac{\partial{\mathcal{L}}}{\partial{z_2^{(2)}}}x_1^{(2)} & \frac{\partial{\mathcal{L}}}{\partial{z_2^{(1)}}}x_2^{(1)}+\frac{\partial{\mathcal{L}}}{\partial{z_2^{(2)}}}x_2^{(2)} & \frac{\partial{\mathcal{L}}}{\partial{z_2^{(1)}}}x_3^{(1)}+\frac{\partial{\mathcal{L}}}{\partial{z_2^{(2)}}}x_3^{(2)} \\
\end{bmatrix}=\frac{\partial{\mathcal{L}}}{\partial{Z}}X^T
\end{equation}$$</small></center>

$$\begin{equation}
\frac{\partial{\mathcal{L}}}{\partial{b}}=\begin{bmatrix}
\frac{\partial{\mathcal{L}}}{\partial{b_1}} \\ \frac{\partial{\mathcal{L}}}{\partial{b_2}}
\end{bmatrix}=\begin{bmatrix}
\frac{\partial{\mathcal{L}}}{\partial{z_1^{(1)}}}+\frac{\partial{\mathcal{L}}}{\partial{z_1^{(2)}}} \\ 
\frac{\partial{\mathcal{L}}}{\partial{z_2^{(1)}}}+\frac{\partial{\mathcal{L}}}{\partial{z_2^{(2)}}}
\end{bmatrix}=\sum_{i=1}^{2}{\frac{\partial{\mathcal{L}}}{\partial{z^{(i)}}}}
\end{equation}$$

<a name="rd"></a>
**Reverse Differentiation**: a simple procedure summarized for a clearer understanding of backprop from Node A to Node B:
1. Find one single path of "A$\rightarrow$B"
2. Multiply all edge derivatives
3. Add the multiple to the overall derivative
4. Repeat 1-3  
  
e.g.  
Path 1:

<center><img src="../../images/DL/rd1.png" width="500"/></center>

Path 2:

<center><img src="../../images/DL/rd2.png" width="500"/></center>

Path 3:

<center><img src="../../images/DL/rd3.png" width="500"/></center>

And so on ......  
<center><strong><i>Reverse Differentiation $\times$ Backward Step = Backward Propagation</i></strong></center>

## Gradient Descent

$$\begin{equation}
W := W-\alpha\frac{\partial\mathcal{L}}{\partial W}
\end{equation}$$

1. **Stochastic GD** (using 1 training example for each GD step)

    $$\begin{align}
    \mathcal{L}(\hat{Y},Y)&=\frac{1}{2}(\hat{Y_i}-Y_i)^2 \\
    W&=W-\alpha\frac{\partial\mathcal{L}}{\partial W}
    \end{align}$$

2. **Mini-batch GD** (using mini-batches of size $m'\ (\text{s.t.}\ m=km', k\in Z)$ for each GD step)

    $$\begin{align}
    \mathcal{L}(\hat{Y},Y)&=\frac{1}{2}\sum_{i=1}^{m'}{(\hat{Y_i}-Y_i)^2} \\
    W&=W-\alpha\frac{\partial\mathcal{L}}{\partial W}
    \end{align}$$

3. **Batch GD** (using the whole training set for each GD step)

    $$\begin{align}
    \mathcal{L}(\hat{Y},Y)&=\frac{1}{2}\sum_{i=1}^{m}{(\hat{Y_i}-Y_i)^2} \\
    W&=W-\alpha\frac{\partial\mathcal{L}}{\partial W}
    \end{align}$$

# Improvements on Neural Networks {#imp}

## Train/Test Split

- Dataset = training set + development/validation set + test set
- Split ratio:
    - old era: 70/0/30%, 60/20/20%, ...
    - big data era: 98/1/1%, 99.5/0.4/0.1%, 99.5/0.5/0%, ... \\
    (trend: testset as small as possible)
- All 3 subsets should come from the exact same distribution (<strike>mismatch</strike>)

## Initialization

- $W$ should be initialized with **small random values** to break symmetry (to make sure that different hidden nodes can learn different things)
- $b$ can be initialized to **zeros** ($\because$ symmetry is still broken when $W$ is randomly initialized)
- Different initializations $\rightarrow$ different results
- Refer to [keras documentation](https://keras.io/initializers/) for initializers.

## Data Fitting

**Underfitting**:

<center><img src="../../images/DL/uf.png" width="300"/></center>

**Proper fitting**:

<center><img src="../../images/DL/nof.png" width="300"/></center>

**Overfitting**:

<center><img src="../../images/DL/of.png" width="300"/></center>
&nbsp;

Tradeoff: *train error* vs *validation error*:
- ***train err* too small $\longrightarrow$ high variance (overfitting)**  
(e.g. train err = 1%; val err = 11%)
- ***train err* too big $\longrightarrow$ high bias (underfitting)**  
(e.g. train err = 17%; val err = 16%)
- ***train err* too big & *val err* even bigger $\longrightarrow$ both probs**   
(e.g. train err = 17%; val err = 34%)
- ***train err* too small & *val err* also small $\longrightarrow$ congratulations!**  
(e.g. train err = 0.5%; val err = 1%)
&nbsp;  
&nbsp;  

<a name="pro"></a>
**The Procedure**:

<center><img src="../../images/DL/fit.png" width="500"/></center>

## Regularization

Idea: add a regularization term to the original loss function:

$$\begin{equation}
\mathcal{J}(w,b)=\frac{1}{m}\sum_{i=1}^{m}{\mathcal{L}(\hat{y}^{(i)},y^{(i)})}+\frac{\lambda}{2m}f(w)
\end{equation}$$

- $\lambda$: regularization parameter
- $f(w)$: regularization on $w$

How does regularization prevent overfitting?

- set $\lambda$ as big as possible $\Rightarrow w^{[l]}\approx 0$ $\Rightarrow z^{[l]}\approx 0$ $\Rightarrow$ as if some hidden nodes don't exist any more
- $\Rightarrow$ less complexity $\Rightarrow$ variance $\downarrow$

&nbsp;  
**Regularization on LogReg**:
- <a name="L2"></a>**L2 Regularization**:

$$\begin{equation}
\mathcal{J}(w,b)=\frac{1}{m}\sum_{i=1}^{m}{\mathcal{L}(\hat{y}^{(i)},y^{(i)})}+\frac{\lambda}{2m}\|w\|^2_2 \\
\|w\|^2_2=\sum_{j=1}^{n_x}w_j^2=w^Tw
\end{equation}$$

- <a name="L1"></a>**L1 Regularization**:

$$\begin{equation}
\mathcal{J}(w,b)=\frac{1}{m}\sum_{i=1}^{m}{\mathcal{L}(\hat{y}^{(i)},y^{(i)})}+\frac{\lambda}{2m}\|w\|_1 \\
\|w\|_1=\sum_{j=1}^{n_x}{|w|}
\end{equation}$$

<a name="nnreg"></a>**Regularization on NN**:

$$\begin{equation}
\mathcal{J}(W^{[k]},b^{[k]})=\frac{1}{m}\sum_{i=1}^{m}{\mathcal{L}(\hat{y}^{(i)},y^{(i)})}+\frac{\lambda}{2m}\sum_{l=1}^L{\|W^{[l]}\|^2_F}
\end{equation}$$

- Frobenius Norm: 

$$\begin{equation}
\|W^{[l]}\|^2_F=\sum_{i=1}^{n_{l-1}}\sum_{j=1}^{n_l}{(w_{ij}^{[l]})^2}
\end{equation}$$

- Weight Decay on GD:

$$\begin{align}
W^{[l]}&:=w^{[l]}-\alpha\cdot\frac{\partial{\mathcal{L}}}{\partial{W^{[l]}}} \\
&=w^{[l]}-\alpha\cdot\Big(\frac{\partial{\mathcal{L}}}{\partial{W^{[l]}}}(\text{original})+\frac{\lambda}{m}W^{[l]}\Big)
\end{align}$$

&nbsp;  
<a name="dp"></a>**Dropout**: each node has a probability to be kicked out of the NN ($\Rightarrow$ NN becomes smaller & simpler) [only used in training] 

1. Make a **Boolean** matrix corresponding to the matrix of activation values:

    $$\begin{align}
    A^{[k]}&=\begin{bmatrix}
    a_{11}^{[k]} & \cdots & a_{1m}^{[k]} \\
    \vdots & \ddots & \vdots \\ 
    a_{n_k1}^{[k]} & \cdots & a_{n_km}^{[k]}
    \end{bmatrix}\quad\quad\quad A^{[k]}\in\mathbb{R}^{n_k\times m} \\ \\
    B^{[k]}&=\begin{bmatrix}
    b_{11}^{[k]} & \cdots & b_{1m}^{[k]} \\
    \vdots & \ddots & \vdots \\ 
    b_{n_k1}^{[k]} & \cdots & b_{n_km}^{[k]}
    \end{bmatrix}\quad\quad\quad B^{[k]}\in\mathbb{R}^{n_k\times m}
    \end{align}$$

    where $b_{ji}^{[k]}\in\\{\text{True}, \text{False}\\}$. The Boolean values are assigned randomly based on a keep-probability $p$ (can be chosen differently for diff layers).  
    
2. Multiply both matrices element-wise:

    $$\begin{equation}
    A^{[k]}=A^{[k]}* B^{[k]}
    \end{equation}$$
    
    so that some activation values are now zero (they are kicked out of the neural network) 
    
3. Invert the matrix element-wise:

    $$\begin{equation}
    A^{[k]}=A^{[k]}/p
    \end{equation}$$
    
    to ensure consistency in activation values

<br/>
<a name="da"></a>**Data Augmentation**: modify the dataset to get more data (mostly used in Computer Vision) [Benefit: a very low-cost regularization]

Examples:
- flip picture
- slight rotation
- zoom in/out
- distortions
- ...

<br/>
<a name="da"></a>**Early Stopping**: stop the training iterations in the middle

Why do we stop in the middle? 

<center><img src="../../images/DL/es.png" width="400"/></center>

The goal of our training is NOT to finish training BUT to find the optimal weight parameters that minimizes the cost/error.  
  
As shown in the figure, sometimes we should just stop in the middle with the minimal validation error instead of keeping the training going to get overfitting.

<br/>
<a name="og"></a>**Orthogonalization**: implement controls that only affect **ONE single component** of your algorithms performance at a time

<br/>
<a name="norm"></a>**Feature Scaling (normalization)**: normalize inputs for higher efficiency 

1. Set to zero mean:

    $$\begin{align}
    \mu&=\frac{1}{m}\sum_{i=1}^{m}{x^{(i)}} \\
    x&=x-\mu
    \end{align}$$
    
2. Normalize variance:

    $$\begin{align}
    \sigma^2&=\frac{1}{m}\sum_{i=1}^{m}{x^{(i)}\text{**}2}\quad\quad \text{**: element-wise squaring} \\
    x&=x/\sigma^2
    \end{align}$$

<br/>
<a name="gc"></a>**Gradient Checking**

- Why?

    Backprop is a very complex system of mathematical computations. It is very possible that there might be some miscalculation or bugs in these tremendous differentiations, even though the entire training appears as if it's working properly.  
    
    Gradient Checking is the approach to prevent such issue by checking if each gradient is calculated properly. 

    
- Equation

    $$\begin{equation}
    \frac{\partial{\mathcal{J}}}{\partial{w}}=\lim_{\varepsilon\rightarrow 0}\frac{\mathcal{J}(w+\varepsilon)-\mathcal{J}(w-\varepsilon)}{2\varepsilon}\approx\frac{\mathcal{J}(w+\varepsilon)-\mathcal{J}(w-\varepsilon)}{2\varepsilon}
    \end{equation}$$

- Implementation: Calculate the difference between actual gradient and approximated gradient to see if the difference is reasonable:

    $$\begin{equation}
    \text{diff}=\frac{||g-g'||_ 2}{||g||_ 2+||g'||_ 2}
    \end{equation}$$

## Optimization
<a name="mbgd"></a>
**Mini-Batch Gradient Descent**

- Why?

    To allow faster and more efficient computing when there is a large number of training examples (e.g. $m=10000000$)
    
- Implementation (see [gradient descent](../../DL/ANN/#gd) for more details)

    $$\begin{align}
    \mathcal{L}(\hat{Y},Y)&=\frac{1}{2}\sum_{i=1}^{m'}{(\hat{Y_i}-Y_i)^2} \\
    W&=W-\alpha\frac{\partial\mathcal{L}}{\partial W}
    \end{align}$$

- Performance
    - BGD vs MBGD
    
    <center><img src="../../images/DL/bgdvsmbgd.png" width="500"/></center>
    
    - BGD vs SGD
    
        <center><img src="../../images/DL/bgdvssgd.png" width="400"/></center>  
        <br/>
        BGD: large steps, low noise, too long per iteration  
        SGD: small steps, insane noise, lose vectorization  
        MBGD: in between $\rightarrow$ optimal in most cases  

<br/>
**Gradient Descent with Momentum**

- <a name="ema"></a>**Exponentially Weighted (Moving) Average**

    - Intuition

        <center><img src="../../images/DL/ema.png" width="500"/></center>  
        <br/>
        The blue dots represent the raw data points, while the red and green curves represent the two EMAs of the blue dots. As clearly indicated by the figure, EMA is used to reduce the huge oscillation of such time-series data.
        
    - Formula

        $$\begin{equation}
        V_t=\beta V_{t-1}+(1-\beta)\theta_t
        \end{equation}$$
        
        - $\theta_t$: the original time-series data point at time $t$
        - $V_t$: the EMA data point at time $t$
        - $\beta$: an indicator of how many time units (e.g. days) this algorithm is approximately averaging over:
        
            $$\begin{equation}
            \text{#time units}=\frac{1}{1-\beta}
            \end{equation}$$
        
            e.g. $\beta=0.9 \rightarrow$ average over 10 days; $\beta=0.96 \rightarrow$ average over 25 days
    - Performance: easy computation + one-line code + memory efficiency
        
    - <a name="bc"></a>**Bias Correction**
    
        Assume $\beta=0.99$:
        
        $$\begin{align}
        &V_0=0 \\
        &V_1=0.99 V_0+0.01\theta_1=0.01\theta_1 \\
        &V_2=0.99 V_2+0.01\theta_2=0.099\theta_1+0.01\theta_2 \\
        &...
        \end{align}$$
        
        Notice that $V_1 \& V_2$ are very tiny portions of $\theta_1 \& \theta_2$, meaning that they do not accurately represent the actual data points.  
        
        Thus, it is necessary to rescale the early EMA values, with the following formula:
        
        $$\begin{equation}
        V_t:=\frac{V_t}{1-\beta^t}
        \end{equation}$$
        
        In the later calculations, bias correction is not so necessary.
        
- <a name="m"></a>**Momentum**: application of EMA in GD

    1. Compute $dW,db$ on the current MB
    2. Compute EMA
        
        $$\begin{align}
        &V_{dW}:=\beta V_{dW}+(1-\beta)dW \\
        &V_{db}:=\beta V_{db}+(1-\beta)db
        \end{align}$$
        
    3. Compute GD
    
        $$\begin{align}
        &W:=W-\alpha V_{dW} \\
        &b:=b-\alpha V_{db}
        \end{align}$$
        
    $\beta$ is often chosen as $0.9$ in GD with Momentum.
    
    Why named "momentum"? Think of $dW$ as acceleration, $V_{dW}$ as velocity, and $\beta$ as friction.
    
- Performance

    <center><img src="../../images/DL/momentum.png" width="500"/></center>

    Red steps represent Momentum, while blue steps represent normal GD.
    
    Slower learning vertically + Faster learning horizontally
    
    $\rightarrow$ Momentum is always better than SGD
    
<br/>
<a name="rmsprop"></a>**RMSprop (Root Mean Square Propagation)**

- Intuition: a modified version of GD with Momentum

    Why? To further minimize the oscillation of GD and maximize the speed of convergence.

- Steps:

    1. Compute $dW,db$ on the current MB
    2. Compute RMS step
        
        $$\begin{align}
        &S_{dW}:=\beta S_{dW}+(1-\beta)dW^2 \\
        &S_{db}:=\beta S_{db}+(1-\beta)db^2
        \end{align}$$
        
        where $dW^2=dW* dW$
        
    3. Compute GD
    
        $$\begin{align}
        &W:=W-\alpha \frac{dW}{\sqrt{S_{dW}}+\varepsilon} \\
        &b:=b-\alpha \frac{db}{\sqrt{S_{db}}+\varepsilon}
        \end{align}$$
        
    $\varepsilon$ is added to ensure $\text{denominator}\neq0$ (normally $\varepsilon=10^{-8}$)
    
<br/>
<a name="adam"></a>**Adam**

- Intuition: **Momentum + RMSprop**

- Steps:

    1. Compute $dW,db$ on the current MB
    2. Compute Momentum:
        
        $$\begin{align}
        &V_{dW}:=\beta_1 V_{dW}+(1-\beta_1)dW \\
        &V_{db}:=\beta_1 V_{db}+(1-\beta_1)db
        \end{align}$$
        
        Compute RMSprop:
    
        $$\begin{align}
        &S_{dW}:=\beta_2 S_{dW}+(1-\beta_2)dW^2 \\
        &S_{db}:=\beta_2 S_{db}+(1-\beta_2)db^2
        \end{align}$$
        
    3. Bias Correction:
    
        $$\begin{align}
        &V_{dW}:=\frac{V_{dW}}{1-\beta_1^t}, V_{db}:=\frac{V_{db}}{1-\beta_1^t} \\
        &S_{dW}:=\frac{S_{dW}}{1-\beta_2^t}, S_{db}:=\frac{S_{db}}{1-\beta_2^t}
        \end{align}$$
        
    4. Compute GD:
    
        $$\begin{align}
        &W:=W-\alpha \frac{V_{dW}}{\sqrt{S_{dW}}+\varepsilon} \\
        &b:=b-\alpha \frac{V_{db}}{\sqrt{S_{db}}+\varepsilon}
        \end{align}$$
        
        Hyperparameter choices:
        
        - $\alpha$: depends
        - $\beta_1: 0.9$
        - $\beta_2: 0.999$
        - $\varepsilon: 10^{-8}$    
    
<br/>
<a name="lrd"></a>**Learning Rate Decay**

- Intuition: as $\alpha$ slowly decreases, training steps become smaller $\rightarrow$ oscillating closely around the minimum (instead of jumping over the minimum)

- Main Method:

    $$\begin{equation}
    \alpha=\frac{1}{1+r_{\text{decay}}\cdot \text{#epoch}}\cdot\alpha_0
    \end{equation}$$
    
    where 1 epoch means passing through data once.
    
    Normally, $\alpha_0=0.2,r_{\text{decay}}=1$
    
- Other Methods:
    
    - Exponential Decay:
    
        $$\begin{equation}
        \alpha=0.95^{\text{#epoch}}\cdot\alpha_0
        \end{equation}$$
        
    - Root Decay:
    
        $$\begin{equation}
        \alpha=\frac{k}{\sqrt{\text{#epoch}}}\cdot\alpha_0
        \end{equation}$$
        
    - Discrete Staircase:
    
        <center><img src="../../images/DL/staircase.png" width="150"/></center>
        
    - Manual Decay
    
<br/>
<a name="po"></a>**Problems with optimization**

As learnt in Calculus, no matter how we try to find the optimum, we always have problems:

- Local Optima: we get stuck in local optima instead of moving to global optima
- Saddle Points: we find GD=0 at saddle points before we find global optima
- Plateau: long saddle that makes learning super slow

## Hyperparameter Tuning

**Intuition**: try to find the optimal hyperparameter for the NN

**List of Hyperparameters** (in the order of priority)
- Tier 1: $\alpha$
- Tier 2: #hidden units, MB size
- Tier 3: #layers, $\alpha$ decay
- Tier 4: $\beta_1$, $\beta_2$, $\varepsilon$

**Random Picking**: e.g. $n^{[l]}\in [50,100], L\in [2,4]$

**Appropriate Scale**: e.g. $\alpha\in [0.0001,1]$ is obviously NOT an appropriate scale, because 90% of the values are in $[0.1,1]$.

Instead, $\alpha\in[0.0001,1]_ {\text{log}}$ is an appropriate scales because the random picking is equally distributed on the log scale.

e.g. for $\beta\in[0.9,0.999]$, the code implementation should be
- $r\in[-3,-1]$
- $\beta=1-10^r$

## Batch Normalization

**Intuition**: Feature scaling normalizes the inputs to speed up learning for the 1st layer. Similarly, can we normalize $a^{[l-1]}$ to train $W^{[l]} \& b^{[l]}$ faster? Obviously.

**Implementation**:

1. Calculate mean & variance

    $$\begin{align}
    \mu&=\frac{1}{m}\sum_{i=1}^{m}{z^{[l](i)}} \\
    \sigma^2&=\frac{1}{m}\sum_{i=1}^{m}{(z^{[l](i)}-\mu)^2}
    \end{align}$$
    
2. Normalize Node Output:

    $$\begin{equation}
    z_{\text{norm}}^{[l](i)}=\gamma\frac{z^{[l](i)}-\mu}{\sqrt{\sigma^2+\varepsilon}}+\beta
    \end{equation}$$
    
    - $\gamma\ \&\ \beta$ = learnable parameters
    - $\gamma\neq\sqrt{\sigma^2+\varepsilon}$ and $\beta\neq\mu$
    - Make sure to add $\gamma\ \&\ \beta$ to the dictionary of parameter updates during coding
    - Batch Normalization eliminates $b^{[l]}$ during $\mu$ calculation