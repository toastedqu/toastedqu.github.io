---
title : "Model-based RL"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: false
images: []
weight: 300
---

## Markov Decision Process (MDP)

Specification:

- State space: $\mathcal{S}$
- Action space: $\mathcal{A}$
- Transition probability (i.e., model): $p(s'|s,a)=P(S_{t+1}=s'|S_t=s,A_t=a)=\sum_{r\in\mathcal{R}}p(s',r|s,a)$
- Reward: $r(s,a,s')=\mathbb{E}[R_{t+1}|S_t=s,A_t=a,S_{t+1}=s']=\frac{\sum_{r\in\mathcal{R}}rp(s',r|s,a)}{\sum_{r\in\mathcal{R}}p(s',r|s,a)}$
- Discount factor: $\gamma\in[0,1]$

Goal: Find optimal policy $a_t=\pi(s_t)$ to maximize long-term reward:
$$
G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}
$$

Policy-specific attributes:

- Bellman's Equation ($\forall\pi(a|s)$, including stochastic)
\begin{align*}
V_\pi(s)&=E_\pi[G_t|s_t=s]=\sum_{a}\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma V_\pi(s')], \forall s\in\mathcal{S}\\
Q_\pi(s,a)&=E_\pi[G_t|s_t=s,a_t=a]=\sum_{s',r}p(s',r|s,a)[r+\gamma V_\pi(s')], \forall s\in\mathcal{S}, \forall a\in\mathcal{A}
\end{align*}
- Bellman's Optimality Equation ($\forall\pi_*(s)$, only deterministic)
\begin{align*}
V_*(s)&=\max_{a\in\mathcal{A}(s)}\sum_{s',r}p(s',r|s,a)[r+\gamma V_*(s')], \forall s\in\mathcal{S}\\
Q_*(s,a)&=\sum_{s',r}p(s',r|s,a)[r+\gamma\max_{a'}Q_*(s')], \forall s\in\mathcal{S}, \forall a\in\mathcal{A}
\end{align*}

Dynamic Programming:

- **Policy Evaluation**: compute $V_\pi$ for input policy $\pi$
    - Init $V(s)\in\mathbb{R}; \pi(s)\in\mathcal{A}(s); \Delta=0; \epsilon$
    - Repeat until $\Delta<\epsilon$ (i.e., convergence: $\{V_k\}\rightarrow V_\pi$ as $k\rightarrow\infty$):
        - for $s\in\mathcal{S}$:
            1. $V_\text{prev}\leftarrow V(s)$
            2. $V(s)\leftarrow$ Bellman's Equation
            3. $\Delta\leftarrow\max(\Delta,|V_\text{prev}-V(s)|)$
<br><br>
- **Policy Improvement**: update policy $\pi$ (each update guarantees a strictly better policy than before)
    - Init *policy_stable=True*
    - for $s\in\mathcal{S}$:
        1. $a_\text{prev}\leftarrow\pi(s)$
        2. $\pi(s)\rightarrow\arg\max_{a\in\mathcal{A}(s)}\sum_{s',r}p(s',r|s,a)[r+\gamma V(s)]$
        3. *policy_stable=False* if $a_\text{prev}\neq\pi(s)$
<br><br>
- **Policy Iteration**: Policy Evaluation + Policy Improvement
    - Repeat:
        1. Policy Evaluation
        2. Policy Improvement
        3. If *policy_stable=True*, return $\pi$
