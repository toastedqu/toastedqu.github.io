---
title : "Recommender Systems"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: true
images: []
weight: 41
---
Objective: suggest relevant items to users by leveraging feature engineering techniques on user preferences, item features, interactions.

&nbsp;

Categories:
- **Personalized**: predict the next item for an individual user based on their past preferences and behaviors
    - **Collaborative Filtering**: make automatic predictions (Filtering) about the interests of a user by collecting preferences from many users (Collaborative) 
    - **Content-based**: classify items based on their attribute information and provide recommendations accordingly
    - **Knowledge-based**: use knowledge extracted from user profiles or public data to provide more suitable recommendations
    - **Hybrid**: a mix of the above
- **Group**: provide a recommendation that alleviates conflicts within the group based on its collective preferences

&nbsp;

Evaluation:
- **Rating Based Indicators**
- **Item Based Indicators**
- **Commercial Viability**
    - **Robustness**: the ability to maintain accurate and effective recommendations in the face of varying or unexpected data
    - **Data bias**: systematic deviations in data that may result in unfair or inaccurate recommendations
    - **Faireness**: provide unbiased and equitable recommendations for all users regardless of demographic or personal characteristics

&nbsp;

# Collaborative Filtering
Def: make automatic predictions (**Filtering**) about the interests of a user by collecting preferences from many users (**Collaborative**)
- Preferences can be obtained from users' historical interactions (e.g., browsing history, purchased items, etc.)

Cons:
- **Cold Start**: when a new user/item pops up, there is limited to no existing data or history for it.
- **Grey Sheep**: users/items that do not fit into the traditional categories/clusters in a recommendation system may not be recommended properly.

&nbsp;

## Memory-based CF
Idea:
1. Calculate similarity between users/items with a Nearest Neighbor-based method.
2. Generate recommendations for users based on sorted similarity values

Pros:
- Simple, direct, efficient in small-scale scenarios
- Able to capture user behavior patterns and preferences based on historical data

Cons:
- Sparsity in the interaction matrix makes the prediction more susceptible to errors
- Huge computational cost in similarity calculation in large-scale scenarios

### K-NN
Algorithm (K-NN):
1. Find all products {{< math >}}$ i$ that user $u ${{</ math>}} has rated.
2. Find the top {{< math >}}$ k$ most similar products to the target product $j ${{</ math>}} using K-NN
{{< math class=text-center >}}$$\begin{align*}
&\text{Euclidean Distance}: &&d(\textbf{y}_i,\textbf{y}_j)=\frac{\sum\_{u\in U(i,j)}(y\_{ui}-y\_{uj})^2}{|U(i,j)|} \\\\
&\text{Cosine Similarity}: &&\cos(\textbf{y}_i,\textbf{y}_j)=\frac{\sum\_{u\in U(i,j)}(y\_{ui}y\_{uj})}{||\textbf{y}_i|| ||\textbf{y}_j||}
\end{align*}$${{< /math >}}
where {{< math >}}$ U(i,j)$ is the set of users who have rated both product $i$ and $j$. Store these products in $N(j,u)$ so that it is the set of $k$ most similar products to product $j$ for user $u ${{</ math>}}.
3. Average the ratings of these products to obtain estimated rating of user {{< math >}}$ u$ on the new product $j ${{</ math>}}:
$$
\hat{y}\_{uj}=\frac{1}{k}\sum\_{i\in N(j,u)}y\_{ui}
$$
- Cons: Each movie in the top-{{< math >}}$ k ${{</ math>}} list is weighted equally.

<br><br>

Soft K-NN: same algorithm but using {{< math >}}$ s_{ij}$ (the similarity score between $i$ and $j ${{</ math>}}):
$$
\hat{y}\_{uj}=\frac{\sum\_{i\in N(j,u)}s\_{ij}y\_{ui}}{\sum\_{i\in N(j,u)}s\_{ij}}
$$
<br><br>

K-NN with baseline off: same algorithm but subtracting off {{< math >}}$ b\_{uj}$ (the baseline rating. e.g., mean rating of user $u$, mean rating of product $j ${{</ math>}})):
$$
\hat{y}\_{uj}=b\_{uj}+\frac{\sum\_{i\in N(j,u)}s\_{ij}(y\_{ui}-b\_{ui})}{\sum\_{i\in N(j,u)}s\_{ij}}
$$
- Pros:
    - Reduce bias (some users may just on average give higher scores,and they want to distill rating as a relative metric for how much a user liked a movie)

- Cons: 
    - Similar products may be redundant
    - Less similar products got more shrunk towards the baseline

<br><br>

K-NN + Regression: same algorithm but using regression weights {{< math >}}$ w\_{ij}$ instead of similarity (weights measure how much the rating of $i$ tells you about the rating of $j ${{</ math>}}):
$$
\hat{y}\_{uj}=b\_{uj}+\sum\_{i\in N(j,u)}w\_{ij}(y\_{ui}-b\_{ui})
$$

- Cons: need to find {{< math >}}$ w_{ij}$ via seeing other user ratings $y_{vi}$ from users $v$ $\rightarrow$ need to compare every user to find most similar users $\rightarrow ${{</ math>}} high computational cost

&nbsp;

## Model-based CF
Idea: map user representations {{< math >}}$ s_u$, item representations $s_i$, and reviews $t_{ui}$ into a continuous vector space via NN to predict rating $\hat{r} ${{</ math>}}:
$$
\hat{r}=NN(s_u,s_i,t_{ui}|\theta)
$$

Objective: 
- MSE if rating data is explicit.
- NLL if rating data is implicit.

### Matrix Factorization
Problem: Sparsity in user-item interaction data is a huge issue in real-world recommendation tasks

Solution: Matrix Factorization
- e.g., Factorization Machine, Non-Negative Matrix Factorization, etc.

Idea: Factor the rating matrix {{< math >}}$ R ${{</ math>}} into user matrix and product matrix of the same hidden space.

Model:
$$
R=PQ^T
$$
- {{< math >}}$ P$: user matrix of shape $m\times h$, where $m$ is $\\#$users and $h$ is $\\# ${{</ math>}}hidden factors (just like PC scores)
- {{< math >}}$ Q$: product matrix of shape $n\times h$, where $n$ is $\\# ${{</ math>}}hidden factors (just like PCs/loadings)

Prediction: Given a user with params {{< math >}}$ \textbf{p}_u$ and a product with learned features $\textbf{q}_i$, predict rating $\textbf{p}_u\textbf{q}_i^T ${{</ math>}}.

Objective: minimize reconstruction error + L2 penalty:
$$
\mathcal{L}=\sum\_{(u,i)\in K}[(y\_{ui}-\textbf{p}_u\textbf{q}_i^T)^2+\lambda(||\textbf{p}_u||_2^2+||\textbf{q}_i||_2^2)]
$$
- Further regularizations: 
    - Non-Negative Matrix Factorization (NNMF): force all elements of {{< math >}}$ P\\&Q ${{</ math>}} non-negative. (canNOT be an alternative to PCA because not orthogonal!)
    - Locally weighted matrix factorization: {{< math >}}$ \mathcal{L}=\sum\_{(u,i)\in K}[s\_{ij}(y\_{ui}-\textbf{p}_u\textbf{q}_i^T)^2+\lambda(||\textbf{p}_u||_2^2+||\textbf{q}_i||_2^2)] ${{</ math>}}

Optimization:
- Alternating least squares (BAD): fix {{< math >}}$ P$, solve $Q$, fix $Q$, solve $P ${{</ math>}}, ...
- **Collaborative Filtering**: simultaneously estimate both parameters with GD:
    1. Init {{< math >}}$ P\\&Q ${{</ math>}} to small random values.
    2. GD (need to modify notifications):
    {{< math class=text-center >}}$$\begin{align*}
    &x_k^{(i)}\leftarrow x_k^{(i)}-\alpha\left(\sum_{j:r(i,j)=1}{\left(\theta^{(j)T}x^{(i)}-y^{(i,j)}\right)\theta_k^{(j)}}+\lambda x_k^{(i)}\right)\\\\
    &\theta_k^{(j)}\leftarrow \theta_k^{(j)}-\alpha\left(\sum_{i:r(i,j)=1}{\left(\theta^{(j)T}x^{(i)}-y^{(i,j)}\right)x_k^{(i)}}+\lambda\theta_k^{(j)}\right)
    \end{align*}$${{< /math >}}

Cons:
- High computational cost
- Cannot handle out-of-sample users and products (i.e., new ones)

&nbsp;

## Context-aware CF
Idea: incorporate contextual info (e.g., time, location, social status, mood, weather, day type, language, etc.) to get better recommendations

Problem: contextual features are dynamic and complex.

Solution: use latent context embeddings

Types:
- **Pre-filtering**: filter out irrelevant data before feeding context into the system
- **Post-filtering**: apply context to the system after standard recommendation
- **Contextual modeling**: integrate context into recommendation system directly

&nbsp;

# Content-based RS
Def: treat recommendations as a user-specific classification problem by learning a classifier based on item features with a focus on 2 types of info:
- user preference model
- user interaction history

Examples of content features: metadata, item descriptions, full-text indexing of textual items, user-generated content, visual features, multimedia features, etc.

**Heterogeneous Information Network** (HIN): a unique graph network which incorporates different item types and relationships.

Recommendation methods:
- **Clustering**: Bayesian
- **Ranking**: similarity measure

Pros:
- Eliminates cold start
- Eliminates sparsity issue
- Provides interpretability

&nbsp;

# Knowledge-based RS
Def: use domain knowledge, instead of user-item interaction history, to make recommendations in complex domains where items are not typically purchased.

Pros:
- Eliminates cold start
- Eliminates grey sheep
- Versatility in a wide range of domains (as long as domain knowledge exists)
- Pretty much the current SOTA

## Knowledge Graph-based RS
Idea: use an entity-relation-entity triple {{< math >}}$ (h,r,t) ${{</ math>}} to assist RS

Usages:
- Knowledge graph embedding
- Path pattern learning
- Hybrid embedding

## Path-based RS
Idea: use the connectivity patterns between entities (measured by semantic similarity between users/items via different meta-paths) in a knowledge graph to enrich the side info in users/items.

&nbsp;

# Conversational RS (NEW)
Idea: get real-time user preferences via dialogues.

## Preliminaries
### Dialogue Systems
- Problem:
    - Input: conversation history (+ background knowledge)
    - Output: response utterance (+ actions with external tools)
- Goal:
    - **Naturalness**: talk like how humans talk
    - **Specialization**: perform exceptionally well in its specialized scenarios
    - **Automation**: help solve user issues with no manual effort from both sides
- Taxonomy:
    - Objective:
        - **Open-domain**: chat about any topic, focus on engagement and interactiveness
        - **Goal-oriented**: chat about a specific field of topics, focus on guidance for task completion
    - Architecture:
        - **Modular**: a pipeline of modules, each with a specific functionality
        - **End-to-End**: a single model
    - Interaction: (S: system; U: user; A: active; P: passive; E: engage)
        - **SAUP**: system asks clarifying questions; user responds directly
        - **SAUE**: system asks questions; user responds; allow chit-chat
        - **SAUA**: both sides can lead; allow chit-chat
        - **UASP**: user asks questions; system responds
    - Length:
        - Single-turn: QA, prompting
        - Multi-turn: conversation
        - Perpetual: conversation while memorizing info about the user and past conversations

### CRS
- Def: support users in achieving recommendation-related goals through a multi-turn dialogue
    - Categories: Goal-oriented + Multi-turn
- Input: conversation history (+ user preferences + item knowledge + domain knowledge)
- Output: response utterance (+ explanation + item recommendation)

&nbsp;

## Modular CRS
- **Input Processing module**:
    - Input: generic user input (in various forms: natural language, formatted texts (e.g., forms), buttons/clicks, voice, handwritings, multi-language, ...)
    - Output: user intent + dialogue state update
    - Procedure: acquire input -> decode input -> process info
    - Directions: NLP & NLU
        - **Intent Recognition**: the informative need of the user (e.g., ask for recommendation, provide preferences, provide feedbacks, ask questions about items, ...)
        - **Named Entity Recognition**: tne informative elements (e.g., names, locations, characteristics, items, ...)
        - **Sentiment Analysis**: the tone of the message
- **User Modeling module**:
    - Input: content of user input + external knowledge base (+ relevant entities + sentiment + dialogue state)
    - Output: user profile representation (+ dialogue state update)
    - Procedure: extract preferences/needs from input
        - Objective properties: item characteristics from exogenous knowledge sources (i.e., knowledge graphs) based on **Entity Linking**
        - Subjective properties: personal feelings about the item based on **Aspect Extraction** (from reviews)
- **Recommendation module**:
    - When to recommend: depends on the interaction strategy and the dialogue flexibility
        - Ask "good" questions (to guide recommendation)
        - Decide when enough prerferences are collected
    - What to recommend: aware of knowledge from previous utterances, background knowledge, item features, ...
- **Output Generation module**:
    - Template-based Generation:
        - Pros: easy to implement, very straightforward and effective
        - Cons: static, repetitive
    - Dynamic Generation:
        - Pros: dynamic
        - Cons: poorly significant answers
    - Directions:
        - Concept explanation
        - Retrieval-based answer generation

&nbsp;

## End-to-End CRS
One single neural network handles everything. Any LM/LLM can be trained to do this.
- Pros: simple, flexible
- Cons: requires training, hard to debug

&nbsp;

## Evaluation
What makes a CRS "good"?
- Accurate recommendations
- Fewer interaction turns
- Engaging interactions
- Realistic & Natural utterances

What is the goal for CRS evaluation?
- Research goals: high prediction accuracy, high effectiveness of algorithm, ability to mimic humans, ...
- Business goals: more sales, more engagements, more profits, less costs, ...

Current evaluation paradigms:
- Online studies: experimental research with users via A/B Testing
- Offline studies: accuracy evaluation of single components, simulation of user behavior
- Qualitative studies: mainly based on human annotations

Current evaluation dimensions:
- Effectiveness: do we get better recommendations with CRS?
    - metrics: task completion rate, hit rate, choice satisfaction, accuracy metrics, ...
- Efficiency: is CRS faster than other RSs?
    - metrics: interaction time, #interactions in each conversation, cognitive effort, perceived effort, ...
- Quality: is the conversation natural, fluent, and relevant?
    - metrics: perplexity, BLEU, ROGUE, usability, misrecognition rate, quality of dialogue, user control, ...

# Hybrid RS
Def: a mix of all of the above
