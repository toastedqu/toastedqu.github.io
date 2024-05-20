---
title : "Supervised Learning"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: false
images: []
weight: 4
---
# Linear Models

<!-- ## Linear Discriminant Analysis -->

## Naive Bayes
**Why**: provide quick & fairly accurate predictions

**What**: the simplest generative, parametric classifier solely based on Bayes' Theorem
{{<math class=text-center >}}$$p(\mathbf{x},y)=p(y)p(\mathbf{x}|y)=p(y)\prod_jp(x_j|y)$${{</math>}}
- Background (#Params):
    - General: {{<math>}}$\#[p(a_1,\cdots,a_n|b_1,\cdots,b_m)]=(\prod_{j=1}^{n}|a_j|-1)\prod_{i=1}^{m}|b_i|${{</math>}}
    - Joint (binary): {{<math>}}$\#[p(\mathbf{x}|y)]=(2^n-1)\cdot2${{</math>}}
    - Independent (binary): {{<math>}}$ \#[\prod_jp(x_j|y)]=2n ${{</math>}}
- Assumption: **Conditional Independence** of features given label
{{<math class=text-center >}}$$p(\mathbf{x}|y=k)=\prod_{j=1}^np(x_j|y=k)$${{</math>}}
- Params:
    - {{<math>}}$\pi_k=p(y=k)${{</math>}}: prior probability for class {{<math>}}$k${{</math>}}
    - {{<math>}}$w_{jk}${{</math>}}: conditional density param for class {{<math>}}$k${{</math>}} and feature {{<math>}}$j${{</math>}}

**How**: 
- Training:
    - Objective:
        - Loss: 0-1
        - Regularization: Laplace smoothing
    - Optimization:
        - MLE:
{{< math >}}$$\begin{align*}
&\text{Likelihood}: &&p(\mathcal{D}|\mathbf{w})=\prod_{i=1}^mp(y_i|\mathbf{\pi})\prod_{j=1}^n\prod_{k=1}^cp(x_{ij}|w_{jk})^{\mathbf{1}(y_i=k)}\\
&\text{Log-likelihood}: &&\log p(\mathcal{D}|\mathbf{w})=\log p(\mathcal{D}_y|\boldsymbol{\pi})+\sum_{k=1}^c\sum_{j=1}^n\log p(\mathcal{D}_{jk}|w_{jk})\\
&\text{MLE for }\boldsymbol{\pi}: &&\hat{\pi}_k=\frac{m_k}{m}\\
&\text{MLE for }\mathbf{w}\text{ (discrete)}: &&\hat{w}_{jkl}=\frac{m_{jkl}}{m_k}\\
&\text{MLE for }\mathbf{w}\text{ (continuous)}: &&\hat{\mu}_{jk}=\frac{1}{m_k}\sum_{i:y_i=k}x_{ij}\\
&  &&\hat{\sigma}_{jk}^2=\frac{1}{m_k}\sum_{i:y_i=k}(x_{ij}-\hat{\mu}_{jk})^2
\end{align*}$${{</math>}}
            - {{<math>}}$m_k${{</math>}}: #samples of class {{<math>}}$k${{</math>}}
            - {{<math>}}$m_{jkl}=\sum_{i=1}^m\mathbf{1}(x_{ij}=l,y_i=k)${{</math>}}: #samples of class {{<math>}}$k${{</math>}} with feature {{<math>}}$j=l${{</math>}}
        - MAP:
{{< math >}}$$\begin{align*}
&\text{Prior}: &&p(\boldsymbol{\pi})=\text{Dir}(\boldsymbol{\pi}|\boldsymbol{\alpha}), p(w_{jk})=\text{Dir}(w_{jk}|\beta_{jk})\\
&\text{Posterior}: &&p(\boldsymbol{\pi},\mathbf{w}|\mathcal{D})=\text{Dir}(\boldsymbol{\pi}|\tilde{\boldsymbol{\alpha}})\prod_{j=1}^n\prod_{k=1}^c\text{Dir}(w_{jk}|\tilde{\beta}_{jk})\\
&\text{MAP for }\boldsymbol{\pi}: &&\hat{\pi}_k=\frac{m_k+\alpha_k}{m+\sum_{k'}\alpha_{k'}}\\
&\text{MAP for }\mathbf{w}\text{ (discrete)}: &&\hat{w}_{jkl}=\frac{m_{jkl}+\beta_{jkl}}{m_k+\beta_k}
\end{align*}$${{</math>}}
            - {{<math>}}$ \tilde{\alpha}_k=\alpha_k+m_k${{</math>}}: shifted Dirichlet param for {{<math>}}$\pi_k${{</math>}} (conjugate prior + likelihood)
            - {{<math>}}$ \tilde{\beta}_{jkl}=\beta_{jkl}+N_{jkl}${{</math>}}: shifted Dirichlet param for {{<math>}}$w_{jkl}${{</math>}} (conjugate prior + likelihood)
            - **Laplace smoothing**: {{<math>}}$\beta_{jkl}=1,\beta_k=L${{</math>}}
- Inference:
    {{< math class=text-center >}}$$\begin{align*}
    &\text{Likelihood}: &&p(y=k|\mathbf{x},\mathbf{w})=\frac{p(y=k|\boldsymbol{\pi})\prod_jp(x_j|y=k,w_{jk})}{\sum_cp(y=c|\boldsymbol{\pi})\prod_jp(x_j|y_i=c,w_{jc})}\\
    &\text{Posterior}: &&p(y=k|\mathbf{x},\mathcal{D})\propto p(y=k|\mathcal{D})\prod_jp(x_j|y=k,\mathcal{D})=\hat{\pi}_k\prod_j\prod_l\hat{w}_{jkl}^{\mathbf{1}(x_j=l)}
    \end{align*}$${{</math>}}

**Pros**:
- Easy
- Scale invariant
- High computational efficiency (significantly low #params)
- Robust to outliers and noisy data
- No overfitting (unless feature value missing in training set)
- Can handle real-time prediction
- High accuracy on small datasets (> LogReg)

**Cons**:
- Assumption fails in real life
- Low accuracy on large datasets (< LogReg)
- Low accuracy in terms of probability estimates (<< LogReg)

&nbsp;

<!-- All linear models are parametric. -->

<!-- Types:

* Linear Regression
$$
\hat{y_i}=\sum_{j=1}^{n}{w_jx_{ij}}
$$

* Generalized Linear Models ({{<math>}}$ f$: link function; $\mathbf{w}^T\mathbf{x} ${{</math>}}: logits)
$$
\hat{y_i}=f\left(\sum_{j=1}^{n}{w_jx_{ij}}\right)
$$

* Basis Transformation ({{<math>}}$ \phi_j(\mathbf{x}_i) ${{</math>}}: transformation of non-linear inputs into linear inputs)
$$
\hat{y}_i=\sum_{j=1}^dw_j\phi_j(\mathbf{x}_i)
$$ -->

## Linear Regression
**What**: Linear regression fits a linear hyperplane between features and labels.

**When**: 
- **Linearity**: The underlying relationship between {{<math>}}$ \textbf{y}$ and $X ${{</math>}} is linear.
- **Independence**: {{<math>}}$ \varepsilon_i ${{</math>}} is independent of each other.
- **Normality**: {{<math>}}$ \varepsilon_i ${{</math>}} follows Gaussian distribution.
- **Non-Collinearity**: No/Minimal explanatory variables correlate with each other.
- **Homoskedasticity**: The variance of all noises is the same constant {{<math>}}$ \sigma^2 ${{</math>}}.

**How**: 
{{< math class=text-center >}}$$\begin{align*}
&\text{Frequentist:} &&\mathbf{y}=X\mathbf{w}+\boldsymbol{\varepsilon},\varepsilon_i\sim N(0,\sigma^2) \\\\
&\text{Bayesian:}    &&p(\mathbf{y}|X,\mathbf{w})=N(X\mathbf{w},\sigma^2)
\end{align*}$${{</math>}}

**Training**:
- **Hyperparameters**: 
- **Objective**:
    - Loss: MSE
    - Regularization: L1, L2, ElasticNet, L0
- **Optimization**:
$$\begin{aligned}
&\text{OLS/MLE}:\ &&\hat{\mathbf{w}}=(X^TX)^{-1}X^T\mathbf{y} \\\\
&\text{Ridge}:\ &&\hat{\mathbf{w}}=(X^TX+\lambda I)^{-1}X^T\mathbf{y} \\\\
&\text{Lasso}:\ &&\frac{\partial \mathcal{L}_B}{\partial w_j}=\frac{1}{m}\sum_{i=1}\^{m}[x_{ij}(\hat{y}_i-y_i)]+\lambda\cdot\text{sign}(w_j) \\\\
&\text{ElasticNet}:\ &&\frac{\partial \mathcal{L}_B}{\partial w_j}=\frac{1}{m}\sum_{i=1}\^{m}[x_{ij}(\hat{y}_i-y_i)]+\lambda r_1\cdot\text{sign}(w_j)+\lambda(1-r_1)w_j
\end{aligned}$$

**Inference**:
$$
\hat{y}_i=\mathbf{w}^T\mathbf{x}_i
$$

**Pros**:

**Cons**: 

Objective:


Optimization:


Pros:
- Simple and Interpretable
- Scale invariant
- Consistent and Unbiased (OLS/MLE ver.)

Cons:
- Sensitive to outliers
- Limited to assumptions (if any assumption fails, LinReg fails)

Time complexity:
- Train:
    - Exact Solution: {{<math>}}$ O(n^2(m+n)) ${{</math>}}
    - Gradient Descent: {{<math>}}$ O(mn) ${{</math>}}
- Test: {{<math>}}$ O(n) ${{</math>}}

<!-- Code:
```python
class LinearRegression:
    def __init__(self):
        self.w = None
    
    def fit(self, X, y, intercept=False):
        if intercept: X = np.hstack(np.ones((X.shape[0],1)),X)
        self.w = (np.linalg.inv(X.T @ X) @ X.T @ y).reshape(-1)

    def predict(self, X):
        return np.dot(self.w, X.T)
``` -->
&nbsp;

## Logistic Regression
Idea: use sigmoid/softmax as link function for linear regression for classification.

Model:
{{< math class=text-center >}}$$\begin{align*}
&\text{Binary}: &&P(y_i=1|\mathbf{x}_i,\mathbf{w})=\sigma(\mathbf{w}^T\mathbf{x}_i)=\frac{1}{1+\exp{(-\mathbf{w}^T\mathbf{x}_i)}}=\frac{\exp{(\mathbf{w}^T\mathbf{x}_i)}}{1+\exp{(\mathbf{w}^T\mathbf{x}_i)}}\\\\
&\text{Multiclass}: &&P(y_i=k|\mathbf{x}_i,W)=\text{softmax}(W^T\mathbf{x}_i)=\frac{\exp{(\mathbf{w}_k^T\mathbf{x}_i)}}{\sum_{k=1}^{K}{\exp{(\mathbf{w}_k^T\mathbf{x}_i)}}}
\end{align*}$${{</math>}}

Prediction:
$$
\hat{y}_i=\arg\max_k\hat{p}_{ik}
$$

Objective:
- Loss: Cross Entropy
- Regularization: L1, L2, ElasticNet

Optimization:
$$
\frac{\partial \mathcal{L}}{\partial w_{jk}}=\frac{1}{m}\sum_{i=1}\^{m}[x_{ij}(\hat{p}_{ik}-y_i)]
$$

Pros:
- Scale invariant
- Easy expansion to multiclass
- Coefficients resemble feature importance
- Easy gradient calculation

Cons:
- bad performance when {{<math>}}$ n>>m ${{</math>}}
- bad performance for nonlinear cases (assume linearity by log odds)
- prioritize correctly classifying the more prevalent class, even if it means misclassifying the less prevalent class

Time Complexity: Train: {{<math>}}$ O(mn)$; Test: $O(n) ${{</math>}}

Space Complexity: {{<math>}}$ O(n) ${{</math>}}

<!-- ## Principal Component Regression
Idea: PCA + LinReg (Semi-supervised learning)

Model:
1. Do PCA on {{<math>}}$ X$ to get scores $Z$ and loadings $V ${{</math>}}.
2. Do OLS on projected points {{<math>}}$ Z ${{</math>}}:
$$
\hat{\textbf{w}}=(Z^TZ)^{-1}Z^TY
$$

Prediction:
1. Get projection: {{<math>}}$ \hat{\textbf{z}}=V^T\textbf{x} ${{</math>}}
2. Get label: {{<math>}}$ \hat{y}=\textbf{w}\hat{\textbf{z}} ${{</math>}}

Pros:
- Great performance with high-dimensional data (via Dimensionality Reduction)
- Better performance than LinReg in many situations (via filtering out irrelevant/noisy features)
- Simple

Cons:
- Make LinReg scale variant
- Sensitive to choices of PCs
- Bad performance with nonlinear relationships (might be solvable via Kernel PCR but then there would be no need to use PCA at all) -->
&nbsp;

# Support Vector Machine
Idea: choose a decision boundary that maximizes the soft margin between classes.

Preliminaries:
- **Primal vs Dual**
    - Primal operates in the feature space {{<math>}}$ X^TX ${{</math>}}
    - Dual operates in the sample space {{<math>}}$ XX^T ${{</math>}} (i.e., Kernel Matrix)
    - Params in Primal & Dual are transferable: {{<math>}}$ \textbf{w}=\sum_{i=1}^m\alpha_iy_i\textbf{x}_i ${{</math>}}
    - Dual > Primal:
        - = Weighted combination of support vectors
        - Sparsity
        - Kernel Trick
- **Hard margin vs Soft margin**
    - Hard margin does NOT accept any misclassification {{<math>}}$ \rightarrow ${{</math>}} prone to overfitting
    - Soft margin allows some misclassifications {{<math>}}$ \rightarrow ${{</math>}} regularization
- Support vectors are 1) on the margin 2) on the wrong side 3) within the margin.
- In linearly separable case, the decision boundary with the maximal margin is unique.

Model/Prediction: linear classifier
$$
\hat{y}_i=\text{sign}(\textbf{w}^T\phi(\textbf{x}_i))
$$

Objective: Hinge Loss + L2 Penalty (can use other losses/penalties but rare)
- Primal (Linear):
{{< math class=text-center >}}$$\begin{align*}
\min_{w,\xi}\quad & \frac{1}{2} ||\textbf{w}||^2 + C \sum_{i=1}^m \xi_i \\\\
\text{s.t.}\quad & y_i(\textbf{w}^T\textbf{x}_i) \geq 1-\xi_i\\\\
& \xi_i \geq 0
\end{align*}$${{</math>}}
- Primal (Kernel):
{{< math class=text-center >}}$$\begin{align*}
\min_{w,\xi}\quad & \frac{1}{2} ||\textbf{w}||^2 + C \sum_{i=1}^m \xi_i \\\\
\text{s.t.}\quad & y_i(\textbf{w}^T\phi(\textbf{x}_i)) \geq 1-\xi_i\\\\
& \xi_i \geq 0
\end{align*}$${{</math>}}
- Dual (Linear):
{{< math class=text-center >}}$$\begin{align*}
\max_{\alpha\geq 0}\quad & \sum_{i=1}^m \alpha_i - \frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m y_i y_j \alpha_i \alpha_j \mathbf{x}_i^T \mathbf{x}_j \\\\
\text{s.t.}\quad & \sum_{i=1}^n \alpha_i y_i = 0\\\\
& \alpha_i\leq C
\end{align*}$${{</math>}}
- Dual (Kernel):
{{< math class=text-center >}}$$\begin{align*}
\max_{\alpha\geq 0}\quad & \sum_{i=1}^m \alpha_i - \frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m y_i y_j \alpha_i \alpha_j k(\mathbf{x}_i, \mathbf{x}_j) \\\\
\text{s.t.}\quad & \sum_{i=1}^n \alpha_i y_i = 0\\\\
& \alpha_i\leq C
\end{align*}$${{</math>}}

Optimization:
- Param Estimation: Decomposition (quadratic programming), Closed-form, GD, etc.
- Hyperparam Tuning: {{<math>}}$ C$ tells SVM how much misclassification to avoid. The larger $C$, a smaller-margin hyperplane will be chosen. The smaller $C ${{</math>}}, a larger-margin hyperplane will be chosen.

Pros:
- Good performance on high-dimensional and non-linearly separable data (with Kernel trick)
- Good generalization to unseen data
- Guaranteed convexity. Easy to optimize
- Low memory cost (only support vectors matter)

Cons:
- High computational cost, especially 1) with Kernels 2) when sample size is too large 3) with Multiclass classification (no native support for it; need 1v1 or 1-v-rest strategies)
- Low interpretability: No probability estimates
- Bad performance when {{<math>}}$ n>>m ${{</math>}}.
- Bad performance on large datasets (i.e., {{<math>}}$ m>>0 ${{</math>}}).
- Sensitive to outliers and noisy data
- Sensitive to overlapping classes (i.e., classes which share the same parts of some feature values)

Time Complexity:
- Train: {{<math>}}$ O(m^2) ${{</math>}}
- Test: {{<math>}}$ O(kn)$, where $k= ${{</math>}} #support vectors.

&nbsp;

&nbsp;

# Local Learning

## K Nearest Neighbors
Idea: label a sample based on its {{<math>}}$ K ${{</math>}} nearest neighbors.

Model:
1. Calculate distance between sample point and every training point.
2. Find the {{<math>}}$ K ${{</math>}} nearest neighbors with minimal distances.
3. Take the majority vote and output it as the label for the sample point.

Pros:
- No training: instance-based learning (i.e., lazy learner)
- Seamless data augmentation at any step
- Simple and interpretable

Cons:
- Scale variant
- High computational cost for large datasets
- High computational cost + Low variance in distance measure for high-dimensional data
- Sensitive to noisy data, missing values, and outliers

Time Complexity: Test: {{<math>}}$ O(kmn) ${{</math>}}

Space Complexity: {{<math>}}$ O(mn) ${{</math>}}

<!-- Code:
```python
###### Scratch ######
def distance(metric_type,v1,v2):
    if metric_type == "L0":
        return np.count_nonzero(v1-v2)
    if metric_type == "L1":
        return np.sum(np.abs(v1-v2))
    if metric_type == "L2":
        return np.sqrt(np.sum(np.square(v1-v2)))
    if metric_type == "Linf":
        return np.max(np.abs(v1-v2))
    
def KNN(X_train,y_train,samples,K=5,metric_type="L2"):
    def KNN_for_single_sample(K,metric_type,X_train,y_train,sample):
        # Calculate dist between each X_train[i] and sample
        dis_vec = np.array([distance(metric_type,X_train[i],sample) for i in range(len(X_train))])
        
        # Find index of top-K neighbors
        ids = np.argsort(dis_vec)[:K]
        
        # Find neighbors' labels from y_train
        neighbors = list(y_train[ids])
        
        # Return majority vote for the sample
        return max(set(neighbors),key=neighbors.count)
    
    return [KNN_for_single_sample(K,metric_type,X_train,y_train,sample) for sample in samples]
``` -->
&nbsp;

## Kernel Method
**Model (Mercer's Theorem)**:
$$
\forall\phi:\mathbb{R}^n\rightarrow\mathbb{R}^p\ \exists k:\mathbb{R}^{n\times n}\rightarrow\mathbb{R}\ \text{ s.t. }\ k(\mathbf{x},\mathbf{x}')=\phi(\mathbf{x})^T\phi(\mathbf{x}')
$$
- {{<math>}}$ \mathbf{x}\in\mathbb{R}^n ${{</math>}}: input sample
- {{<math>}}$ \phi:\mathbb{R}^n\rightarrow\mathbb{R}^p ${{</math>}}: feature map from one space to another (typically a higher dimensional space)
- {{<math>}}$ k:\mathbb{R}^{n\times n}\rightarrow\mathbb{R} ${{</math>}}: kernel function 

**Idea**: Allow using linear models for non-linear samples, without transforming data into a higher dimensional space (i.e., without computing {{<math>}}$ \phi(\cdot) ${{</math>}}).

**Assumptions**: The kernel function must satisfy 2 conditions:
- **Symmetry**: {{<math>}}$ k(\mathbf{x},\mathbf{x}')=k(\mathbf{x}',\mathbf{x}) ${{</math>}}
- **Positive-Definite**: {{<math>}}$ \forall\mathbf{x}_1,\cdots,\mathbf{x}_m\in\mathbb{R}^n\ \forall c_1\cdots c_m\in\mathbb{R}:\ \sum_{i=1}^{m}\sum_{j=1}^{m}c_ic_jk(\mathbf{x}_i,\mathbf{x}_j)\geq0 ${{</math>}}

**Types of Kernels**:
| Type | Formula |
|:-----|:--------|
| Linear | {{<math>}}$ k(\mathbf{x}_i,\mathbf{x}_j)=\mathbf{x}_i^T\mathbf{x}_j ${{</math>}} |
| Polynomial | {{<math>}}$ k(\mathbf{x}_i,\mathbf{x}_j)=(\mathbf{x}_i^T\mathbf{x}_j+c)^d, c\geq0 ${{</math>}} |git
| RBF (Radial Basis Function) | {{<math>}}$ k(\mathbf{x}_i,\mathbf{x}_j)=\exp\left(-\frac{\|\|\mathbf{x}_i-\mathbf{x}_j\|\|^2}{2\sigma^2}\right) ${{</math>}} |

<!-- {{< math class=text-center >}}$$\begin{align*}
&\text{Regression}: &&\hat{y}=\frac{\sum_{i=1}^{m}{k(\mathbf{x},\mathbf{x}_i)y_i}}{\sum_{i=1}^{m}{k(\mathbf{x},\mathbf{x}_i)}}\\\\
&\text{Binary Classification}: &&\hat{y}=\text{sign}(\sum_{i=1}^{m}{k(\mathbf{x},\mathbf{x}_i)y_i})
\end{align*}$${{</math>}} -->

**Pros**:
- Low computational cost (relative to feature mapping calculation)
- Applicable to all data type
- Easy validation of kernel by finding arbitrary 2 points with negative determinant (i.e., negative eigenvalue)
- Consider all samples as each sample's neighbors

**Cons**:
- Scale variant
- Overfitting
- Low interpretability of kernels
- High difficulty in kernel selection
- Biased toward closer samples

<!-- KNN vs KernelReg: 

<center>

|               KNN              |                   Kernel                  |
|:------------------------------:|:-----------------------------------------:|
|         distance metric        |              kernel function              |
|         {{<math>}}$ K ${{</math>}} neighbors          |               all neighbors               |
| same impact from all neighbors | weighted impact favoring closer neighbors |
|          Scale variant         |               Scale variant               |

</center> -->

&nbsp;

&nbsp;

# Decision Tree
Idea: build a tree where each node is a feature split to classify data points into different leaf outputs.

Preliminaries:
- **Information gain**: {{<math>}}$ IG(Y|X)=H(Y)-H(Y|X) ${{</math>}}
    - {{<math>}}$ H(\cdot) ${{</math>}}: Impurity measure
        - Gini: {{<math>}}$ H(Y)=\sum_{y}{p_y(1-p_y)} ${{</math>}}
        - Entropy: {{<math>}}$ H(Y)=-\sum_{y}{p_y\log_2{p_y}} ${{</math>}}
- **Entropy**: a measure of uncertainty
    - Conditional entropy (Average): {{<math>}}$ H(Y|X)=\sum_{x}{P(X=x)H(Y|X=x)} ${{</math>}}
    - Specific conditional entropy: {{<math>}}$ H(Y|X=x)=-\sum_{y}{P(Y=y|X=x)\log_2{P(Y=y|X=x)}} ${{</math>}}

Model:
1. Calculate info gain for each feature.
2. Select the feature that maximizes info gain as the decision threshold.
3. Split data based on the decision. Repeat Step 1-2 until stop.

Pros:
- Scale invariant
- Low computational cost
- Can handle multiclass classification
- Interpretable and Easy to validate (easy model visualization)

Cons:
- Prone to overfitting (perfect fit on smaller sample size)
- Bad performance with class imbalance (biased toward most frequently occurring class)
- Sensitive to noisy data, missing value, and outliers
- Discrete predictions only
- Difficult to find globally optimal decisions (only support locally optimal decisions at each node)
- Bad performance overall

Time complexity:
- Train: {{<math>}}$ O(mn\log{m}) ${{</math>}}
- Test: {{<math>}}$ O(d)$, where $d=$ depth. (Ideally $O(\log{m}) ${{</math>}} if balanced binary tree)

<!-- Code:
```python
###### SCRATCH ######
# This scratch is only meant for binary classification.

def IG(X_train,y_train,feature_index,impurity="entropy"):
    # Compute impurity
    def H(probs):
        if impurity=="entropy":
            return -np.sum(np.multiply(probs[probs!=0],np.log2(probs[probs!=0])))
        if impurity=="gini":
            return np.sum(np.multiply(probs,1-probs))
    
    # Calculate H(Y)
    m = len(y_train)
    p_y_1 = np.count_nonzero(y_train)/m
    p_y = np.array([1-p_y_1,p_y_1])
    H_y = H(p_y)
    
    x_col = X_train[:,feature_index].reshape(-1)
    y_col = y_train.reshape(-1)
    p_y_x = {}  # for each value of the selected feature, store its y count. 
    p_x = {}    # for each value of the selected feature, store its own count.
    
    # Count all occurrences of all values of the selected feature, together with their corresponding y value counts.
    for i in range(m):
        p_x[x_col[i]] = p_x.get(x_col[i],0)+1
        if x_col[i] not in p_y_x:
            p_y_x[x_col[i]] = [0,0]
        if y_col[i] == 0:
            p_y_x[x_col[i]][0] += 1
        elif y_col[i] == 1:
            p_y_x[x_col[i]][1] += 1
            
    # Calculate H(Y|X=v)
    H_y_x_specs = {} # for each value v of the selected feature, store H(Y|X=v).
    x_total_count = sum(p_x.values())
    for key in p_y_x:
        y_total = sum(p_y_x[key])
        p_x[key] /= x_total_count   # normalize counts to P(X=v)
        p_y_x[key] = [p_y_x[key][0]/y_total, p_y_x[key][1]/y_total] # normalize counts to P(Y=u|X=v)
        H_y_x_specs[key] = H(np.array(p_y_x[key])) 
        
    # Calculate H(Y|X)
    H_y_x = sum([p_x[key]*H_y_x_specs[key] for key in p_x])
    
    # Return IG
    return H_y-H_y_x

def select_feature_max_IG(X_train,y_train,impurity="entropy"):
    IG_cache = [IG(X_train,y_train,i,impurity="entropy") for i in range(X_train.shape[1])]
    return IG_cache.index(max(IG_cache))

# to be continued
``` -->

&nbsp;

&nbsp;

# Ensemble Methods
- **Bootstrapping**: randomly select {{<math>}}$ fm$ samples with replacement from the original training set into subsets, where $f ${{</math>}} is the fraction of samples to bootstrap.
- **Bagging** (**b**ootstrap **agg**regat**ing**): aggregate a bunch of weak models trained on bootstrapped subsets individually.
- **Boosting**: train multiple models sequentially and dependently.
    - Converge exponentially with #iterations.
    - Cons: Highly sensitive to outliers and noisy data.
- Common Pros of Ensemble Methods:
    - Scale invariant
    - Reduce overfitting
    - Greater performance
    - Can handle high-dimensional data efficiently
- Common Cons of Ensemble Methods: 
    - High computational cost
    - Low interpretability

&nbsp;

## Random Forest
Idea: Bagging with Decision Trees

Model: 
1. Bootstrap.
2. Create a full decision tree (no pruning). On each node, randomly select {{<math>}}$ \sqrt{n} ${{</math>}} features from bootstrapped subset. Find the best split.
3. Repeat 1-2 to create a random forest till #tree reaches limit.
4. Use out-of-bag samples to determine the accuracy of each tree.

Prediction: Take a majority/average vote of all trees.

Objective:
- Loss: any
- Regularization: increase #trees

Pros:
- Reduce overfitting in decision tree & much more accurate than a single decision tree
- Flexible to both categorical & numerical outputs
- Automatically handle missing values by dropping or filling with median/mode
- Robust to outliers and noisy data
- Best used in banking and healthcare

Cons:
- Worse performance than Boosting in general

Time Complexity:
- Train: {{<math>}}$ O(kmn\log{m})$, where $k= ${{</math>}} #trees
- Test: {{<math>}}$ O(kd)$, where $d= ${{</math>}} max depth

&nbsp;

## AdaBoost

Idea: train a bunch of stumps (weak learners) sequentially and take a weighted majority vote.

Model:
1. Init all samples with equal sample weight.
2. Find optimal feature for first stump. Calculate total error. Calculate amount of say:
$$
\text{Amount of Say}=\frac{1}{2}\log{\frac{1-Err_\text{tot}}{Err_\text{tot}}}
$$
3. Modify sample weights for incorrectly and correctly predicted samples as follows:
{{< math class=text-center >}}$$\begin{align*}
w_\text{incorrect}&\leftarrow w_\text{incorrect}\cdot e^\text{Amount of Say}\\\\
w_\text{correct}&\leftarrow w_\text{correct}\cdot e^{-\text{Amount of Say}}\\\\
\end{align*}$${{</math>}}
4. Normalize all sample weights.
5. Select new samples based on new sample weights as probabilities with replacement to generate a new set.
6. Give equal sample weights to all samples in the new training set.
7. Repeat Steps 1-6 till #stumps reaches limit.

Prediction: Take a weighted majority vote using the Amount of Say from each stump.

Objective: Exponential Loss

Pros:
- Reduce overfitting more than bagging (because parameters are not optimized jointly but stagewise)
- Fewer hyperparameters than other models

Cons:
- Guaranteed perfect fitting (training error=0) if infinite epochs
- Sensitive to outliers and noisy data
- Slower and generally worse performance than Gradient Boosting

Random Forest vs AdaBoost:
<center>

|         Random Forest         |                 AdaBoost                 |
|:-----------------------------:|:----------------------------------------:|
|   No tree shape requirement   | Each tree is a stump (1 node + 2 leaves) |
| Each tree has equal influence |      Stumps have weighted influence      |
|    Each tree is independent   |  Each stump is dependent of its previous one |

</center>

&nbsp;

## Gradient Boosting
Idea: train a bunch of fixed-size trees to fit residuals sequentially. take a weighted majority vote.

Model:
1. Init a constant-value leaf as initial prediction (average for reg; log-odds for cls)
2. Train a tree (#leaves < {{<math>}}$ m ${{</math>}}) to predict the **negative loss gradient** w.r.t. curr ensemble's predictions for each sample in the training data.
    - **Pseudo-residuals**: negative gradients represent how far off curr predictions are from actual targets
    - The weak learner aims to capture the patterns in the errors made by curr ensemble.
3. Combine leaf (+ prev trees) + curr tree (scaled with a learning rate) to make new predictions on the same data. Calculate new negative loss gradients.
4. Repeat Steps 2-3 till #trees reaches limit.

Objective:
- Loss: arbitrary
- Regularization:
    - smaller learning rate (i.e., multiplicative shrinking of the weight on the weak learner)
    - bootstrapping

Optimization (Hyperparams):
- #stages: {{<math>}}$ T ${{</math>}}
- bag size (fraction): {{<math>}}$ f ${{</math>}}
- learning rate: {{<math>}}$ \eta ${{</math>}}
- tree depth: {{<math>}}$ d ${{</math>}}

Pros:
- Great performance in general
- Fast Prediction
- High flexibility (multiple hyperparameters to tune, multiple loss functions to use)
- Can handle missing data
- Reduce overfitting by using a small learning rate and incorporate bootstrapping

Cons:
- Sensitive to outliers and noisy data (may cause overfitting by overemphasizing them)
- Require a large grid search for hyperparameter tuning
- Longer training due to sequential building


AdaBoost vs Gradient Boosting:

<center>

|         AdaBoost         |       Gradient Boosting       |
|:------------------------:|:-----------------------------:|
|     Init with a stump    | Init with a leaf of {{<math>}}$ \bar{y} ${{</math>}} |
|          Stumps          |        Fixed-size trees       |
| Scale stumps differently |      Scale trees equally      |
| Train on {{<math>}}$ y ${{</math>}} |      Train on residuals      |

</center>

&nbsp;

## XGBoost
Idea: use a unique tree to make decisions based on similarity scores.

Preliminaries:
- **Pre-sort algorithm**: sort samples by the feature value, then split linearly.
- **Histogram-based algorithm**: bucket continuous feature values into discrete bins.
- **Level-wise tree growth** (BFS)

Model:
1. Calculate residuals for all samples based on current prediction. Calculate similarity score for the root node of a new tree.
$$
\text{Similarity}=\frac{\sum_{i=1}^{m_r}{r_i^2}}{m_r+\lambda}
$$
2. Find similarity gain of each possible split. Choose the split with the max gain at root node.
$$
\text{Gain}=\text{Similarity}_\text{left}+\text{Similarity}_\text{right}-\text{Similarity}_\text{root}
$$
3. Repeat Step 2 till limit. Prune branches bottom-up by checking whether the gain of the branch is higher than a predefined threshold {{<math>}}$ \gamma ${{</math>}}. If it is higher, stop. If it is lower, prune it, move on to the next branch.
4. Define the output value for each leaf of this tree.
$$
\text{Output}=\frac{\sum_{i=1}^{m_r}{r_i}}{m_r+\lambda}
$$
5. Define the predicted value for each sample.
$$
\hat{y}_i=\bar{y}+\eta\sum{\text{Tree}(\mathbf{x}_i)} 
$$
6. Repeat Steps 1-5 till limit.

Optimization (hyperparams):
- Regularization: {{<math>}}$ \lambda$ (the higher $\lambda ${{</math>}}, the lower gain, thus easier to prune.)
- Prune threshold: {{<math>}}$ \gamma$ ($\gamma=0 ${{</math>}} prunes negative gains.)
- Learning rate: {{<math>}}$ \eta ${{</math>}}

Pros:
- Perform well when #features is small
- The Pros of Gradient Boosting

Cons:
- Cannot handle categorical features (must do encoding)
- Bad performance on sparse and unstructured data
- The Cons of Gradient Boosting

&nbsp;

## LightGBM

Idea: Gradient Boosting + GOSS + EFB

Preliminaries:
- **GOSS (Gradient-based One-Side Sampling)**: focus more on under-trained samples without changing the original data distribution.
    1. Sort all samples based on abs(gradient). Select top {{<math>}}$ \alpha ${{</math>}}% samples as the samples with large gradients. Keep them.
    2. Randomly sample {{<math>}}$ b$% of the remaining samples with small gradients. Amplify them with a constant $\frac{1-a}{b} ${{</math>}}.
- **EFB (Exclusive Feature Bundling)**: bundle mutually exclusive features together into much fewer dense features by conflict rate (more nonzero values lead to higher probability of conflicts).
    1. Sort features based on conflicts in a descending order. Assign each to an existing bundle with a small conflict or create a new bundle.
    2. Merge exclusive features in the same bundle. If two features have joint ranges, add an offset value so that the two features can be merged into one range.
- **Leaf-wise tree growth**: choose the leaf with max delta loss to grow. (DFS)

Pros:
- Can handle categorical features
- Much faster and efficient training both in time and space
- Higher accuracy than most other boosting algorithms, especially on large datasets

Cons:
- Overfitting (Because of leaf-wise splitting, the tree can be much more complex) 
- Bad performance on small datasets

Gradient Boosting Comparisons:
<center>

|                      |                                       XGBoost                                       |                                         CatBoost                                        |                                   LightGBM                                  |
|:--------------------:|:-----------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------:|
|      Tree growth     |                          Asymmetric level-wise tree growth                          |                            Symmetric growth                             |                               Asymmetric leaf-wise tree growth                              |
|         Split        |                                 Pre-sort + Histogram                                |                                          Greedy                                         |                                  GOSS + EFB                                 |
|  Numerical features  |                                       Support                                       |                                         Support                                         |                                   Support                                   |
| Categorical features | Need external encoding into numerical features<br>Cannot interpret ordinal category |                          Support with default encoding methods                          |   Support with default encoding methods<br>Can interpret ordinal category   |
|     Text features    |                                          NO                                         | YES by converting them to numerical features<br>via bag-of-words, BM-25, or Naive Bayes |                                      NO                                     |
|    Missing values    |     Interpret as NaN<br>Assign to side that reduces loss the most in each split     |                          Interpret as NaN<br>Process as min/max                         | Interpret as NaN<br>Assign to side that reduces loss the most in each split |

</center>

&nbsp;

&nbsp;

# Online Learning
In comparison to batch learning which can be extremely expensive for big datasets, online learning is easy in a map-reduce environment.

Online learning is like SGD where we learn only on a single sample each time.

Common Pros:
- Extreme flexibility
- Extremely fast in terms of reaching optimum
- The only door towards continual learning so far

Common Cons:
- Catastrophic forgetting (The model forgets what it learnt before)
- Highly noisy convergence (might not converge due to frequent updates)

&nbsp;

## Least Mean Squares
Idea: Fit LinReg on each observation sequentially.

Model: LinReg

Objective: {{<math>}}$ \mathcal{L}=(y_i-\textbf{w}^T\textbf{x}_i)^2 ${{</math>}}

Optimization: SGD
$$
\textbf{w}_{i+1}=\textbf{w}_i-\frac{\eta}{2}\frac{\partial\mathcal{L}}{\partial\textbf{w}_i}=\textbf{w}_i+\eta(y_i-\textbf{w}^T\textbf{x}_i)\textbf{x}_i
$$
- This algorithm is guaranteed to converge for {{<math>}}$ \eta\in(0,\lambda_{\max})$, where $\lambda_{\max}$ is the largest eigenvalue of $X^TX ${{</math>}}.
- The convergence rate is proportional to {{<math>}}$ \frac{\lambda_{\min}}{\lambda_{\max}}$ (i.e., ratio of extreme eigenvalues of $X^TX ${{</math>}}).

&nbsp;

## Perceptron
Idea: Fit a linear classifier on each observation sequentially.

Model: linear classifier
- Binary: {{<math>}}$ \hat{y}_i=\text{sign}(\textbf{w}^T\textbf{x}_i) ${{</math>}}
- Multiclass: {{<math>}}$ \hat{y}_i=\arg\max_k\textbf{w}_k^T\textbf{x}_i ${{</math>}}

Objective: {{<math>}}$ \mathcal{L}=(y_i-\hat{y}_i)^2 ${{</math>}}

Optimization: SGD
- Binary:
$$
\textbf{w}_{i+1}=\textbf{w}_i+\frac{1}{2}(y_i-\text{sign}(\textbf{w}^T\textbf{x}_i))\textbf{x}_i=\textbf{w}_i+y_i\textbf{x}_i
$$
    - {{<math>}}$ \eta=\frac{1}{2} ${{</math>}}
    - If we get it correct, no update at all because the residual is 0.
    - If we get it wrong, drop weights for negative samples and raise weights for positive samples.
- Multiclass: Similar to Binary but raise the weight vector for the actual class and reduce the weight vector for the predicted wrong class.

Pros:
- Guaranteed to converge to a solution if samples are **linearly separable**
    - #mistakes before convergence is always less than {{<math>}}$ \frac{\max_i||\textbf{x}_i||_2}{\gamma} ${{</math>}}.
    - Numerator: size of the biggest sample.
    - Denominator: margin of the decision boundary ({{<math>}}$ \gamma>0$ if linearly separable; $\gamma<y_i\textbf{w}_*^T\textbf{x}_i ${{</math>}})

Cons:
- Highly unstable and bounce around if samples are not linearly separable

Variations:
- **Voted Perceptron**: Perceptron but keep track of all the intermediate models and take a majority vote during prediction.
- **Averaged Perceptron**: Voted Perceptron but take an average vote during prediction.
- Pros:
    - Better generalization performance than perceptron
    - Same training time as perceptron
    - Nearly as good as SVM
    - Can use the Kernel Trick to replace dot product with Kernel
- Cons:
    - Higher memory cost
    - Higher inference cost
- Further variations: different ways to tune {{<math>}}$ \eta$. (standard chooses $\eta=1$, alternatives chooses $\eta ${{</math>}} to maximize margin)

&nbsp;

## Passive Aggressive
Idea: Perceptron but minimizing **hinge loss** (i.e., maximize margin)

Model: Perceptron

Objective: Hinge loss
$$
\mathcal{L}=\begin{cases}
0 & \text{ if }y_i\textbf{w}^T\textbf{x}_i\geq1 \\\\
1-y_i\textbf{w}^T\textbf{x}_i & \text{ if }y_i\textbf{w}^T\textbf{x}_i<1
\end{cases}
$$

Optimization: Margin-Infused Relaxed Algorithm (MIRA)
- If correct classification with a margin of at least 1, no change.
- If wrong,
$$
\textbf{w}_{i+1}=\textbf{w}_i+\frac{\mathcal{L}}{||\textbf{x}_i||^2}y_i\textbf{x}_i
$$
- MIRA attempts to make the smallest changes to the weight vector(s) by moving the hyperplane to include the new sample point onto the margin, therefore maximizing the margin for the entire dataset.

# Pros & Cons Summary
| | Scale Invariance | Robustness | Consistency | Generalization | 