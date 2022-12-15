---
title : "Reinforcement Learning"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: false
images: []
weight: 500
---
RL is not my major focus, so I include its very basics as part of the ML handbook instead of making a separate one.

Some specifications:
- **Model-based vs Model-free**: Model-based RL assumes a model of the environment, while Model-free RL does not.
- **On-policy vs Off-policy**: On-policy learning follows the target policy, while Off-policy learning follows a different policy from the target policy.

# Model-based

## Markov Decision Process (MDP)

Assumption: 
- Markov Property: $P(s_{t+1}|s_t,\cdots,s_0)=P(s_{t+1}|s_t)$
- Stationarity: The underlying specification of transition model and reward structure is fixed.

Specification:
- State space: $\mathcal{S}$
- Action space: $\mathcal{A}(s)$
- Transition probability (i.e., model): $p(s'|s,a)=\sum_{r\in\mathcal{R}}p(s',r|s,a)$
- Reward: $r(s,a,s')=\frac{\sum_{r\in\mathcal{R}}rp(s',r|s,a)}{\sum_{r\in\mathcal{R}}p(s',r|s,a)}$
- Discount factor: $\gamma\in[0,1]$

Goal: Find policy $a_t=\pi(s_t)$ to maximize long-term reward:
$$
G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}
$$

Policy Attributes:
- Policy (deterministic/stochastic): $\pi(a|s)$
- Value (i.e., state value): 
$$
V_\pi(s)=\mathbb{E}_\pi[G_t|s_t=s]=\sum_a\pi(a|s)\sum\_{s',r}p(s',r|s,a)[r+\gamma V\_\pi(s')], \forall s\in\mathcal{S}
$$
- Q-value (i.e., state-action value):
$$
Q_\pi(s,a)=\mathbb{E}_\pi[G_t|s_t=s,a_t=a]=\sum\_{s',r}p(s',r|s,a)[r+\gamma V\_\pi(s')], \forall s\in\mathcal{S}, \forall a\in\mathcal{A}
$$
- Optimal Policy (deterministic): $\pi\_{\*}(s)=\arg\max\_{a}Q\_{\*}(s,a)$
- Optimal Value:
$$
V_{\*}(s)=\max_\pi V_\pi(s)=\max_a Q_{\*}(s,a)=\max_a \sum_{s',r}p(s',r|s,a)[r+\gamma V_{\*}(s')]
$$
- Optimal Q-value:
$$
Q_{\*}(s,a)=\max_\pi Q_\pi(s,a)=\sum_{s',r}p(s',r|s,a)[r+\gamma\max_{a'}Q_{\*}(s')]
$$

Dynamic Programming:
- **Policy Evaluation**: compute $V_\pi$ for input policy $\pi$
    - Init $V(s); \pi(s); \epsilon; \Delta=0$
    - Repeat until $\Delta<\epsilon$ (i.e., convergence: $\lim_{k\rightarrow\infty}\\{V_k\\}=V_\pi$):
        - For $s\in\mathcal{S}$:
            1. $V_\text{prev}\leftarrow V(s)$
            2. $V(s)\leftarrow \sum_a\pi(a|s)\sum\_{s',r}p(s',r|s,a)[r+\gamma V\_\pi(s')]$
            3. $\Delta\leftarrow\max(\Delta,|V_\text{prev}-V(s)|)$
- **Policy Improvement**: update policy $\pi$ (each update guarantees a strictly better policy)
    - Init *policy_stable=True*
    - For $s\in\mathcal{S}$:
        1. $a_\text{prev}\leftarrow\pi(s)$
        2. $\pi(s)\rightarrow\arg\max_{a\in\mathcal{A}(s)}\sum_{s',r}p(s',r|s,a)[r+\gamma V(s)]$
        3. If $a_\text{prev}\neq\pi(s)$, *policy_stable=False* 
- **Policy Iteration**: Policy Evaluation + Policy Improvement
    - Repeat:
        1. Policy Evaluation
        2. Policy Improvement
        3. If *policy_stable=True*, return $\pi$





# Model-free
in Model-free RL, we have no information about the environment model (i.e., transition probabilities and reward structure). Therefore, we will have to explore on our own.

## Exploration-Exploitation Trade-off
There is a trade-off between exploring new strategies and exploiting current knowledge of the environment. Finding the perfect balance in various situations is challenging and sometimes impossible.

$\epsilon$-greedy:
- With probability $\epsilon$, execute a random action.
- With probability $1-\epsilon$, execute a greedy action.
- A simple annealing schedule: $\epsilon_t=\frac{n_0}{n_0+\text{visits}(s_t)}$, where $n_0$ is a hyperparam.

## Temporal Difference Learning
Idea: learn from current predictions rather than waiting till termination.

Algorithm (TD(0). i.e., one-step look-ahead):
- Init $V(s_{\text{end}})=0; V(s); \pi(s); \alpha\in(0,1]$ ($\forall s\in\mathcal{S}$)
- Repeat (For each episode):
    - Init $s$
    - For step in episode:
        1. $a\leftarrow\pi(s)$
        2. $r,s'\leftarrow s,a$
        3. $V(s)\leftarrow V(s)+\alpha[r+\gamma V(s')-V(s)]$
        4. $s\leftarrow s'$
    - If $s=s_\text{end}$: return

## SARSA
Idea: On-policy TD(0) using Q-value.

Algorithm:
- Init $Q(s_{\text{end}},\cdot)=0; Q(s,a)$ ($\forall s\in\mathcal{S}\ \forall a\in\mathcal{A}(s)$)
- Repeat (For each episode):
    - Init $s$
    - $a\leftarrow\pi_Q(s)$, where $\pi_Q(s)$ is policy derived from $Q(s,\cdot)$ (e.g., $\epsilon$-greedy)
    - For step in episode:
        1. $r,s'\leftarrow s,a$
        2. $a'\leftarrow\pi_Q(s')$
        3. $Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma Q(s',a')-Q(s,a)]$
        4. $s\leftarrow s'; a\leftarrow a'$
    - If $s=s_\text{end}$: return


## Q-Learning
Idea: Off-policy TD(0).

Algorithm:
- Init $Q(s_{\text{end}},\cdot)=0; Q(s,a)$ ($\forall s\in\mathcal{S}\ \forall a\in\mathcal{A}(s)$)
- Repeat (For each episode):
    - Init $s$
    - For step in episode:
        1. $a\leftarrow\pi_Q(s)$, where $\pi_Q(s)$ is policy derived from $Q(s,\cdot)$ (e.g., $\epsilon$-greedy)
        2. $r,s'\leftarrow s,a$
        3. $Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma\max_{a'}Q(s',a')-Q(s,a)]$
        4. $s\leftarrow s'$
    - If $s=s_\text{end}$: return


## Monte Carlo

## Deep Q-Learning