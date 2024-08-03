---
title : "Reinforcement Learning"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: true
images: []
weight: 500
---
RL is not my major focus, so I include its very basics as part of the ML handbook instead of making a separate one.

Some specifications:
- **Model-based vs Model-free**: Model-based RL assumes a model of the environment, while Model-free RL does not.
- **On-policy vs Off-policy**: On-policy learning follows the target policy ($ \pi_\text{behavior}=\pi_\text{target}$), while Off-policy learning follows a different policy from the target policy ($\pi_\text{behavior}\neq\pi_\text{target} $).
- **Episodic vs Sequential**:
    - Episodic environment: a discrete process where RL is divided into a series of **independent** episodes (i.e., epochs). The agent updates its policy AFTER finishing each episode. The goal is to maximize cumulative reward over all episodes, thus **discount factor is usually set to 1**.
    - Sequential environment: a continuous process where each current action affects future actions. The agent updates it policy DURING the process. The goal is to maximize the expected future rewards of the current process, thus discount factor matters.

Overview:
<center>
<img src="/images/rl/overview.png" width="600"/>
</center>


# Model-based

## Markov Decision Process (MDP)

Assumption: 
- Markov Property: $ P(s_{t+1}|s_t,\cdots,s_0)=P(s_{t+1}|s_t) $
- Stationarity: The underlying specification of transition model and reward structure is fixed.

Specification:
- State space: $ \mathcal{S} $
- Action space: $ \mathcal{A}(s) $
- Transition probability (i.e., model): $ p(s'|s,a)=\sum_{r\in\mathcal{R}}p(s',r|s,a) $
- Reward: $ r(s,a,s')=\frac{\sum_{r\in\mathcal{R}}rp(s',r|s,a)}{\sum_{r\in\mathcal{R}}p(s',r|s,a)} $
- Discount factor: $ \gamma\in[0,1] $

Goal: Find policy $ a_t=\pi(s_t) $ to maximize long-term reward:
$$
G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}
$$

Policy Attributes:
- Policy (deterministic/stochastic): $ \pi(a|s) $
- Value (i.e., state value): 
$$
V_\pi(s)=\mathbb{E}_\pi[G_t|s_t=s]=\sum_a\pi(a|s)\sum\_{s',r}p(s',r|s,a)[r+\gamma V\_\pi(s')], \forall s\in\mathcal{S}
$$
- Q-value (i.e., state-action value):
$$
Q_\pi(s,a)=\mathbb{E}_\pi[G_t|s_t=s,a_t=a]=\sum\_{s',r}p(s',r|s,a)[r+\gamma V\_\pi(s')], \forall s\in\mathcal{S}, \forall a\in\mathcal{A}
$$
- Optimal Policy (deterministic): $ \pi\_{\*}(s)=\arg\max\_{a}Q\_{\*}(s,a) $
- Optimal Value:
$$
V_{\*}(s)=\max_\pi V_\pi(s)=\max_a Q_{\*}(s,a)=\max_a \sum_{s',r}p(s',r|s,a)[r+\gamma V_{\*}(s')]
$$
- Optimal Q-value:
$$
Q_{\*}(s,a)=\max_\pi Q_\pi(s,a)=\sum_{s',r}p(s',r|s,a)[r+\gamma\max_{a'}Q_{\*}(s')]
$$

Dynamic Programming:
- **Policy Evaluation**: compute $ V_\pi$ for input policy $\pi$ (time complexity: $O(|\mathcal{S}|^2|\mathcal{A}|) $)
    - Init $ V(s); \pi(s); \epsilon; \Delta=0 $
    - Repeat until $ \Delta<\epsilon$ (i.e., convergence: $\lim_{k\rightarrow\infty}\\{V_k\\}=V_\pi $):
        - For $ s\in\mathcal{S} $:
            1. $ V_\text{prev}\leftarrow V(s) $
            2. $ V(s)\leftarrow \sum_a\pi(a|s)\sum\_{s',r}p(s',r|s,a)[r+\gamma V\_\pi(s')] $
            3. $ \Delta\leftarrow\max(\Delta,|V_\text{prev}-V(s)|) $
- **Policy Improvement**: update policy $ \pi $ (each update guarantees a strictly better policy)
    - Init *policy_stable=True*
    - For $ s\in\mathcal{S} $:
        1. $ a_\text{prev}\leftarrow\pi(s) $
        2. $ \pi(s)\rightarrow\arg\max_{a\in\mathcal{A}(s)}\sum_{s',r}p(s',r|s,a)[r+\gamma V(s)] $
        3. If $ a_\text{prev}\neq\pi(s) $, *policy_stable=False* 
- **Policy Iteration**: Policy Evaluation + Policy Improvement
    - Repeat:
        1. Policy Evaluation
        2. Policy Improvement
        3. If *policy_stable=True*, return $ \pi $





# Model-free
in Model-free RL, we have no information about the environment model (i.e., transition probabilities and reward structure). Therefore, we will have to explore on our own.

## Exploration-Exploitation Trade-off
There is a trade-off between exploring new strategies and exploiting current knowledge of the environment. Finding the perfect balance in various situations is challenging and sometimes impossible.

$ \epsilon $-greedy:
- With probability $ \epsilon $, execute a random action.
- With probability $ 1-\epsilon $, execute a greedy action.
- A simple annealing schedule: $ \epsilon_t=\frac{n_0}{n_0+\text{visits}(s_t)}$, where $n_0 $ is a hyperparam.

## Temporal Difference Learning
Idea: learn from current predictions rather than waiting till termination. (a weighted average between previous and current values)

### TD(0)
Algorithm (TD(0). i.e., one-step look-ahead):
- Init $ V(s_{\text{end}})=0; V(s); \pi(s); \alpha\in(0,1]$ ($\forall s\in\mathcal{S} $)
- Repeat (For each episode):
    - Init $ s $
    - For step in episode:
        1. $ a\leftarrow\pi(s) $
        2. $ r,s'\leftarrow s,a $
        3. $ V(s)\leftarrow V(s)+\alpha[r+\gamma V(s')-V(s)] $
        4. $ s\leftarrow s' $
    - If $ s=s_\text{end} $: return

### SARSA
Idea: On-policy TD(0) using Q-value. ($ \epsilon $-greedy for action choice and future evaluation)

Algorithm:
- Init $ Q(s_{\text{end}},\cdot)=0; Q(s,a)$ ($\forall s\in\mathcal{S}\ \forall a\in\mathcal{A}(s) $)
- Repeat (For each episode):
    - Init $ s $
    - $ a\leftarrow\pi_Q(s)$, where $\pi_Q(s)$ is policy derived from $Q(s,\cdot)$ (e.g., $\epsilon $-greedy)
    - For step in episode:
        1. $ r,s'\leftarrow s,a $
        2. $ a'\leftarrow\pi_Q(s') $
        3. $ Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma Q(s',a')-Q(s,a)] $
        4. $ s\leftarrow s'; a\leftarrow a' $
    - If $ s=s_\text{end} $: return


### Q-Learning
Idea: Off-policy TD(0). ($ \epsilon $-greedy for action choice, greedy for future evaluation)

Algorithm:
- Init $ Q(s_{\text{end}},\cdot)=0; Q(s,a)$ ($\forall s\in\mathcal{S}\ \forall a\in\mathcal{A}(s) $)
- Repeat (For each episode):
    - Init $ s $
    - For step in episode:
        1. $ a\leftarrow\pi_Q(s)$, where $\pi_Q(s)$ is policy derived from $Q(s,\cdot)$ (e.g., $\epsilon $-greedy)
        2. $ r,s'\leftarrow s,a $
        3. $ Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma\max_{a'}Q(s',a')-Q(s,a)] $
        4. $ s\leftarrow s' $
    - If $ s=s_\text{end} $: return

Pros:
- Low variance

Cons:
- High bias
- Sensitive to initial Q values
- Easy online learning
- Necessary for non-episodic tasks
- Faster convergence than MC on stochastic tasks

## Monte Carlo
Idea: Estimate expected reward by sampling.

Algorithm:
- Repeat:
    1. Sample an episode following the current policy (from $ s_0$ to $s_\text{end}$ or end when $t=T$). Obtain a return $G_t $ of the episode.
        - MC uses empirical mean return instead of expected return, starting from $ s_t$ or $(s_t,a_t) $.
        - $ V_\pi(s_t)$ = average of returns following all the visits to $s_t $ in a set of episodes
        - $ Q_\pi(s_t,a_t)$ = average of returns following all the visits to $(s_t,a_t) $ in a set of episodes
    2. Update policy with average of $ [G_1,\cdots,G_N] $

Pros:
- Lower bias compared to Q-learning
- Less sensitive to initial Q values
- Monte Carlo Tree Search is widely used in the most successful game playing methods

Cons:
- Higher variance compared to Q-learning
- High computational cost for long episodes
- Limited to episodic tasks
- Slower convergence on stochastic tasks

## Deep Q-Learning
Idea: Represent $ Q(s,a) $ by a neural network.

Model:
- Input: $ s $
- Output: $ Q(s,:)$ of size $|\mathcal{A}(s)| $
- Problem: Instability (i.e., rapid changes) in Q function can cause it to diverge
- Solution: Use 2 networks:
    - Q-network: regularly updated, provide value for $ Q(s,a) $
    - Target network: occasionally updated, provide value for $ Q(s',a') $

Algorithm (DQN):
- Init weights $ W$ for NN (i.e., Q function); $\mathcal{D} $ as replay memory
- Repeat (For each episode):
    - Init $ s $
    - For step in episode:
        1. $ a\leftarrow\pi_Q(s)$, where $\pi_Q(s)$ is policy derived from $Q(s,\cdot)$ (e.g., $\epsilon $-greedy)
        2. $ r,s'\leftarrow s,a $
        3. $ \mathcal{D}$.append(($s,a,r,s' $))
        4. $ s\leftarrow s' $
        5. Sample random minibatches of $ \\{(s_i,a_i,r_i,s_{i+1})\\}_{i=1}^{m}$ from $\mathcal{D} $
        6. $ y_i\leftarrow\begin{cases}r_j &\text{ if }s_{i+1}=s_\text{end} \\\\ r_j+\gamma\max_{a'}Q(s_{i+1},a') &\text{ if }s_{i+1}\neq s_\text{end}\end{cases} $
        3. GD on $ (y_i-Q(s_i,a_i;W))^2 $

Prediction: $ \pi(s)=\arg\max_a\hat{Q}(s,a) $

Objective: MSE: $ [r+\gamma\max_{a'}Q(s',a')-Q(s,a)]^2 $

Optimization: GD

Pros:
- Can play certain game(s) better than humans

Cons: 
- Poor generalization to even slightly different games

<!--
## Hyperparamater Tuning

### Cross Validation

CV: evaluate how the outcomes will generalize to independent datasets.

## Vanishing/Exploding Gradient

**Gradient**: $ \frac{\partial\mathcal{L}}{\partial w}$, specifically on $w $.

**Vanishing**: When backprop towards input layer, the gradients get smaller and smaller and approach zero which eventually leaves the weights of the front layers nearly unchanged. $ \rightarrow $ gradient descent never converges to optimum.
- Causes:
    - Sigmoid or similar activation funcs. They have 0 gradient when abs(input) is large enough.
    - Gradients at the back are consistently less than 1.0. Therefore the chain reaction approaches 0.
- Symptoms:
    - Param at the back change a lot, while params at the front barely change.
    - Some model weights become 0.
    - The model learns very slowly, and training stagnate at very early iterations.

**Exploding**: in some cases, gradients get larger and larger and eventually causes very large weight updates to the front layers $ \rightarrow $ gradient descent diverges.
- Causes:
    - Bad weight initialization. They cause large loss and therefore large gradients.
    - Gradients at the back are consistently larger than 1.0. Therefore the chain reaction approaches $ \infty $.
- Symptoms:
    - Params grow exponentially.
    - Some model weights become NaN.
    - The model learns crazily, and the changes in params/loss make no sense.

Solutions:
- Proper Weight Inits (e.g., Xavier, Glorot, He.)
    - All layer outputs should have equal variance as input samples.
    - All gradients should have equal variance.
- Proper Activation Funcs (e.g., ReLU, LReLU, ELU, SELU, etc.)
    - Gradient = 1 for positive inputs.
- Batch Normalization
    - normalize inputs to ideally $ N(0,1) $ before passing them to the layer.
- Gradient Clipping
    - Clip gradient with max & min thresholds. Any value beyond will be clipped back to the threshold.


<center>

| Model | Type | Accuracy | Speed (train) | Speed (test) | Interpretability | Scale Invariant | 
|:-|:-:|:-:|:-:|:-:|:-:|:-:|
| Linear Regression | Regression | NO | YES | YES | YES | YES w/o regularization<br>NO with regularization |
| Logistic Regression | Classification | NO | YES | YES | YES | YES w/o regularization<br>NO with regularization |
| Naive Bayes | Classification | NO | YES | YES | YES | YES |
| K-Nearest Neighbors | Both | NO | - | YES on small dataset<br>NO on large dataset | YES | NO |
| Decision Tree | Both | NO | YES | YES | YES | YES |
| Linear SVM | Both | NO | YES | YES | YES | NO |
| Kernel SVM | Both | YES | YES on small dataset<br>NO on large dataset | NO | NO | NO |
| Random Forest | Both | YES | NO | NO | NO | YES |
| Boosting | Both | YES | NO | NO | NO | YES |
| Neural Networks | Both | YES | NO | NO | NO | NO |

</center> -->



<!-- 
### Radial Basis Function

$$
\phi_j(\mathbf{x})=\exp{\left(-\frac{||\mathbf{x}-\mu_j||_2^2}{c}\right)}
$$

Steps:
1. Cluster points $ \mu_j $ with k-means clustering.
2. Pick a width $ c=2\sigma^2$ for all the Gaussian pdfs $N(\mu_j,\sigma^2) $ at each cluster.
3. Fit a linear regression.

Usage:
- $ d<n $: dimensionality reduction
- $ d>n $: convert nonlinear problem to linear
- $ d=n $: switch to a dual representation

Pros:

Cons:
- Scale variant.
- Need to find perfect $ c$. Low $c$ leads to overfitting. High $c $ leads to learning nothing (different centroids may cover each other, which is horrible). -->
