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
- **On-policy vs Off-policy**: On-policy learning follows the target policy ({{<math>}}$ \pi_\text{behavior}=\pi_\text{target}$), while Off-policy learning follows a different policy from the target policy ($\pi_\text{behavior}\neq\pi_\text{target} ${{</math>}}).
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
- Markov Property: {{<math>}}$ P(s_{t+1}|s_t,\cdots,s_0)=P(s_{t+1}|s_t) ${{</math>}}
- Stationarity: The underlying specification of transition model and reward structure is fixed.

Specification:
- State space: {{<math>}}$ \mathcal{S} ${{</math>}}
- Action space: {{<math>}}$ \mathcal{A}(s) ${{</math>}}
- Transition probability (i.e., model): {{<math>}}$ p(s'|s,a)=\sum_{r\in\mathcal{R}}p(s',r|s,a) ${{</math>}}
- Reward: {{<math>}}$ r(s,a,s')=\frac{\sum_{r\in\mathcal{R}}rp(s',r|s,a)}{\sum_{r\in\mathcal{R}}p(s',r|s,a)} ${{</math>}}
- Discount factor: {{<math>}}$ \gamma\in[0,1] ${{</math>}}

Goal: Find policy {{<math>}}$ a_t=\pi(s_t) ${{</math>}} to maximize long-term reward:
$$
G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}
$$

Policy Attributes:
- Policy (deterministic/stochastic): {{<math>}}$ \pi(a|s) ${{</math>}}
- Value (i.e., state value): 
$$
V_\pi(s)=\mathbb{E}_\pi[G_t|s_t=s]=\sum_a\pi(a|s)\sum\_{s',r}p(s',r|s,a)[r+\gamma V\_\pi(s')], \forall s\in\mathcal{S}
$$
- Q-value (i.e., state-action value):
$$
Q_\pi(s,a)=\mathbb{E}_\pi[G_t|s_t=s,a_t=a]=\sum\_{s',r}p(s',r|s,a)[r+\gamma V\_\pi(s')], \forall s\in\mathcal{S}, \forall a\in\mathcal{A}
$$
- Optimal Policy (deterministic): {{<math>}}$ \pi\_{\*}(s)=\arg\max\_{a}Q\_{\*}(s,a) ${{</math>}}
- Optimal Value:
$$
V_{\*}(s)=\max_\pi V_\pi(s)=\max_a Q_{\*}(s,a)=\max_a \sum_{s',r}p(s',r|s,a)[r+\gamma V_{\*}(s')]
$$
- Optimal Q-value:
$$
Q_{\*}(s,a)=\max_\pi Q_\pi(s,a)=\sum_{s',r}p(s',r|s,a)[r+\gamma\max_{a'}Q_{\*}(s')]
$$

Dynamic Programming:
- **Policy Evaluation**: compute {{<math>}}$ V_\pi$ for input policy $\pi$ (time complexity: $O(|\mathcal{S}|^2|\mathcal{A}|) ${{</math>}})
    - Init {{<math>}}$ V(s); \pi(s); \epsilon; \Delta=0 ${{</math>}}
    - Repeat until {{<math>}}$ \Delta<\epsilon$ (i.e., convergence: $\lim_{k\rightarrow\infty}\\{V_k\\}=V_\pi ${{</math>}}):
        - For {{<math>}}$ s\in\mathcal{S} ${{</math>}}:
            1. {{<math>}}$ V_\text{prev}\leftarrow V(s) ${{</math>}}
            2. {{<math>}}$ V(s)\leftarrow \sum_a\pi(a|s)\sum\_{s',r}p(s',r|s,a)[r+\gamma V\_\pi(s')] ${{</math>}}
            3. {{<math>}}$ \Delta\leftarrow\max(\Delta,|V_\text{prev}-V(s)|) ${{</math>}}
- **Policy Improvement**: update policy {{<math>}}$ \pi ${{</math>}} (each update guarantees a strictly better policy)
    - Init *policy_stable=True*
    - For {{<math>}}$ s\in\mathcal{S} ${{</math>}}:
        1. {{<math>}}$ a_\text{prev}\leftarrow\pi(s) ${{</math>}}
        2. {{<math>}}$ \pi(s)\rightarrow\arg\max_{a\in\mathcal{A}(s)}\sum_{s',r}p(s',r|s,a)[r+\gamma V(s)] ${{</math>}}
        3. If {{<math>}}$ a_\text{prev}\neq\pi(s) ${{</math>}}, *policy_stable=False* 
- **Policy Iteration**: Policy Evaluation + Policy Improvement
    - Repeat:
        1. Policy Evaluation
        2. Policy Improvement
        3. If *policy_stable=True*, return {{<math>}}$ \pi ${{</math>}}





# Model-free
in Model-free RL, we have no information about the environment model (i.e., transition probabilities and reward structure). Therefore, we will have to explore on our own.

## Exploration-Exploitation Trade-off
There is a trade-off between exploring new strategies and exploiting current knowledge of the environment. Finding the perfect balance in various situations is challenging and sometimes impossible.

{{<math>}}$ \epsilon ${{</math>}}-greedy:
- With probability {{<math>}}$ \epsilon ${{</math>}}, execute a random action.
- With probability {{<math>}}$ 1-\epsilon ${{</math>}}, execute a greedy action.
- A simple annealing schedule: {{<math>}}$ \epsilon_t=\frac{n_0}{n_0+\text{visits}(s_t)}$, where $n_0 ${{</math>}} is a hyperparam.

## Temporal Difference Learning
Idea: learn from current predictions rather than waiting till termination. (a weighted average between previous and current values)

### TD(0)
Algorithm (TD(0). i.e., one-step look-ahead):
- Init {{<math>}}$ V(s_{\text{end}})=0; V(s); \pi(s); \alpha\in(0,1]$ ($\forall s\in\mathcal{S} ${{</math>}})
- Repeat (For each episode):
    - Init {{<math>}}$ s ${{</math>}}
    - For step in episode:
        1. {{<math>}}$ a\leftarrow\pi(s) ${{</math>}}
        2. {{<math>}}$ r,s'\leftarrow s,a ${{</math>}}
        3. {{<math>}}$ V(s)\leftarrow V(s)+\alpha[r+\gamma V(s')-V(s)] ${{</math>}}
        4. {{<math>}}$ s\leftarrow s' ${{</math>}}
    - If {{<math>}}$ s=s_\text{end} ${{</math>}}: return

### SARSA
Idea: On-policy TD(0) using Q-value. ({{<math>}}$ \epsilon ${{</math>}}-greedy for action choice and future evaluation)

Algorithm:
- Init {{<math>}}$ Q(s_{\text{end}},\cdot)=0; Q(s,a)$ ($\forall s\in\mathcal{S}\ \forall a\in\mathcal{A}(s) ${{</math>}})
- Repeat (For each episode):
    - Init {{<math>}}$ s ${{</math>}}
    - {{<math>}}$ a\leftarrow\pi_Q(s)$, where $\pi_Q(s)$ is policy derived from $Q(s,\cdot)$ (e.g., $\epsilon ${{</math>}}-greedy)
    - For step in episode:
        1. {{<math>}}$ r,s'\leftarrow s,a ${{</math>}}
        2. {{<math>}}$ a'\leftarrow\pi_Q(s') ${{</math>}}
        3. {{<math>}}$ Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma Q(s',a')-Q(s,a)] ${{</math>}}
        4. {{<math>}}$ s\leftarrow s'; a\leftarrow a' ${{</math>}}
    - If {{<math>}}$ s=s_\text{end} ${{</math>}}: return


### Q-Learning
Idea: Off-policy TD(0). ({{<math>}}$ \epsilon ${{</math>}}-greedy for action choice, greedy for future evaluation)

Algorithm:
- Init {{<math>}}$ Q(s_{\text{end}},\cdot)=0; Q(s,a)$ ($\forall s\in\mathcal{S}\ \forall a\in\mathcal{A}(s) ${{</math>}})
- Repeat (For each episode):
    - Init {{<math>}}$ s ${{</math>}}
    - For step in episode:
        1. {{<math>}}$ a\leftarrow\pi_Q(s)$, where $\pi_Q(s)$ is policy derived from $Q(s,\cdot)$ (e.g., $\epsilon ${{</math>}}-greedy)
        2. {{<math>}}$ r,s'\leftarrow s,a ${{</math>}}
        3. {{<math>}}$ Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma\max_{a'}Q(s',a')-Q(s,a)] ${{</math>}}
        4. {{<math>}}$ s\leftarrow s' ${{</math>}}
    - If {{<math>}}$ s=s_\text{end} ${{</math>}}: return

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
    1. Sample an episode following the current policy (from {{<math>}}$ s_0$ to $s_\text{end}$ or end when $t=T$). Obtain a return $G_t ${{</math>}} of the episode.
        - MC uses empirical mean return instead of expected return, starting from {{<math>}}$ s_t$ or $(s_t,a_t) ${{</math>}}.
        - {{<math>}}$ V_\pi(s_t)$ = average of returns following all the visits to $s_t ${{</math>}} in a set of episodes
        - {{<math>}}$ Q_\pi(s_t,a_t)$ = average of returns following all the visits to $(s_t,a_t) ${{</math>}} in a set of episodes
    2. Update policy with average of {{<math>}}$ [G_1,\cdots,G_N] ${{</math>}}

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
Idea: Represent {{<math>}}$ Q(s,a) ${{</math>}} by a neural network.

Model:
- Input: {{<math>}}$ s ${{</math>}}
- Output: {{<math>}}$ Q(s,:)$ of size $|\mathcal{A}(s)| ${{</math>}}
- Problem: Instability (i.e., rapid changes) in Q function can cause it to diverge
- Solution: Use 2 networks:
    - Q-network: regularly updated, provide value for {{<math>}}$ Q(s,a) ${{</math>}}
    - Target network: occasionally updated, provide value for {{<math>}}$ Q(s',a') ${{</math>}}

Algorithm (DQN):
- Init weights {{<math>}}$ W$ for NN (i.e., Q function); $\mathcal{D} ${{</math>}} as replay memory
- Repeat (For each episode):
    - Init {{<math>}}$ s ${{</math>}}
    - For step in episode:
        1. {{<math>}}$ a\leftarrow\pi_Q(s)$, where $\pi_Q(s)$ is policy derived from $Q(s,\cdot)$ (e.g., $\epsilon ${{</math>}}-greedy)
        2. {{<math>}}$ r,s'\leftarrow s,a ${{</math>}}
        3. {{<math>}}$ \mathcal{D}$.append(($s,a,r,s' ${{</math>}}))
        4. {{<math>}}$ s\leftarrow s' ${{</math>}}
        5. Sample random minibatches of {{<math>}}$ \\{(s_i,a_i,r_i,s_{i+1})\\}_{i=1}^{m}$ from $\mathcal{D} ${{</math>}}
        6. {{<math>}}$ y_i\leftarrow\begin{cases}r_j &\text{ if }s_{i+1}=s_\text{end} \\\\ r_j+\gamma\max_{a'}Q(s_{i+1},a') &\text{ if }s_{i+1}\neq s_\text{end}\end{cases} ${{</math>}}
        3. GD on {{<math>}}$ (y_i-Q(s_i,a_i;W))^2 ${{</math>}}

Prediction: {{<math>}}$ \pi(s)=\arg\max_a\hat{Q}(s,a) ${{</math>}}

Objective: MSE: {{<math>}}$ [r+\gamma\max_{a'}Q(s',a')-Q(s,a)]^2 ${{</math>}}

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

**Gradient**: {{<math>}}$ \frac{\partial\mathcal{L}}{\partial w}$, specifically on $w ${{</math>}}.

**Vanishing**: When backprop towards input layer, the gradients get smaller and smaller and approach zero which eventually leaves the weights of the front layers nearly unchanged. {{<math>}}$ \rightarrow ${{</math>}} gradient descent never converges to optimum.
- Causes:
    - Sigmoid or similar activation funcs. They have 0 gradient when abs(input) is large enough.
    - Gradients at the back are consistently less than 1.0. Therefore the chain reaction approaches 0.
- Symptoms:
    - Param at the back change a lot, while params at the front barely change.
    - Some model weights become 0.
    - The model learns very slowly, and training stagnate at very early iterations.

**Exploding**: in some cases, gradients get larger and larger and eventually causes very large weight updates to the front layers {{<math>}}$ \rightarrow ${{</math>}} gradient descent diverges.
- Causes:
    - Bad weight initialization. They cause large loss and therefore large gradients.
    - Gradients at the back are consistently larger than 1.0. Therefore the chain reaction approaches {{<math>}}$ \infty ${{</math>}}.
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
    - normalize inputs to ideally {{<math>}}$ N(0,1) ${{</math>}} before passing them to the layer.
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
1. Cluster points {{<math>}}$ \mu_j ${{</math>}} with k-means clustering.
2. Pick a width {{<math>}}$ c=2\sigma^2$ for all the Gaussian pdfs $N(\mu_j,\sigma^2) ${{</math>}} at each cluster.
3. Fit a linear regression.

Usage:
- {{<math>}}$ d<n ${{</math>}}: dimensionality reduction
- {{<math>}}$ d>n ${{</math>}}: convert nonlinear problem to linear
- {{<math>}}$ d=n ${{</math>}}: switch to a dual representation

Pros:

Cons:
- Scale variant.
- Need to find perfect {{<math>}}$ c$. Low $c$ leads to overfitting. High $c ${{</math>}} leads to learning nothing (different centroids may cover each other, which is horrible). -->
