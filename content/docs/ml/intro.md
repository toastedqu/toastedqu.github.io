---
title : "Introduction"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: false
images: []
weight: 1
---

# Overview
A typical ML project consists of 4 parts:

<center>
<img src="/images/ml/system_design_flow.png"/>
</center>

## Problem
Stage 1: **Understanding**: what's the purpose? what inputs? what outputs? (rough idea)

Stage 2: **Scope**: what are the constraints? (smaller {{< math >}}$ \rightarrow$ simpler; larger $\rightarrow ${{</ math>}} complex)
- **Data constraints**
    - #samples?
    - #features?
    - feature types? 
- **Model constraints**
    - performance-first / quality-first?
    - single general model / multiple specific models?
    - interpretability?
    - retrainability?
- **Resource constraints**
    - time? (training, inference, retraining, project duration, etc.)
    - computing resources? (training, inference, retraining, local/cloud, etc.)

Stage 3: **Evaluation**: how do we know if our problem is properly solved?
- **User experience**: user interaction? recent user experience report? company intention for users? personalization?
- **Evaluation metrics**:
    - **Offline**: MSE, P/R/F1, AUC, R^2, ...
    - **Online**: usage time, usage frequency, click rate, ...

&nbsp;

## Data
Stage 1: **Type**
- **Features**: user, content, context
- **Targets**: explicit (direct indicators), implicit (indirect indicators)

Stage 2: **Availability**
- **Available**: What is available? How much is available? Annotated? How good are the annotations?
- **Unavailable**: What is unavailable? How expensive is data annotation? How many annotators for each sample? How to resolve annotators' disagreements? Is it possible to generate these data automatically (e.g., ChatGPT, rule-based generation, etc.)
- **Privacy**: What user data can we access? How do we access? How do we get user feedback on the system? Online/Periodic data? Anonymity? 
- **Logistics**: Where are the data (local/cloud)? What data structures? How big? What biases exist in such data?

Stage 3: **Processing**

&nbsp;

## Modeling
Stage 1: Baseline method (no model): random benchmark, average, etc.

Stage 2: Easy model

Stage 3: Hard model

For each model, specify:
- Functionality
- Assumptions (if any)
- Pros & Cons
- Objective & Optimization (i.e., loss & training)

&nbsp;

## Production
Production is completely different from experiments. The following factors need to be taken into account:
1. **Inference location**:
    - **local** (user's phone/PC): low latency, require their memory/storage
    - **server** (ours or cloud): low memory/storage usage, high latency, privacy concerns
2. **Feature serving**:
    - **batch features** should be handled offline and served online (i.e., need daily/weekly jobs for data generation/collection)
    - **real-time features** should be handled and served at request time (i.e., need to create a feature store to look up features at serve time; caching may be necessary) {{< math >}}$ \rightarrow ${{</ math>}} watch out for scalability and latency.
3. **Performance Monitoring**: latency, biases, data drift, CPU load, memory, ...
4. **Retrain Frequency**
5. **Online A/B Testing**

### Online A/B Testing
1. **Goal**: Define the objective. (e.g., improving click-through rates, increasing sign-up rates, etc.)
    - **Significance level** ({{< math >}}$ \alpha ${{</ math>}}): threshold of whether the observed difference between the control and treatment groups is statistically significant or occurred by chance.
        - Common values: 0.05 and 0.01
        - A lower {{< math >}}$ \alpha ${{</ math>}} makes it more challenging to detect difference
    - **Power** ({{< math >}}$ 1-\beta$): probability of correctly rejecting $H_0 ${{</ math>}} when it is false, meaning the ability to detect a meaningful difference when it exists
        - Common value: 80%
        - A higher power requires larger sample sizes to achieve
    - *notes:
        - {{< math >}}$ \alpha=P$(FP), Type I error: probability of rejecting $H_0 ${{</ math>}} when it is true.
        - {{< math >}}$ \beta=P$(FN), Type II error: probability of not rejecting $H_0 ${{</ math>}} when it is false.

<t>

2. **Variation**: Generate 2/more versions of whatever element we want to test. (e.g., 2 different designs of a button; blue round button to **control group** A; green square button to **treatment group** B)

<t>

3. **Traffic**: Calculate required sample size per variation
{{< math >}}$ $n=2\times\left(\frac{Z_{\frac{\alpha}{2}}+Z_{\beta}}{\text{MDE}}\right)^2\times p(1-p)$ ${{</ math>}}
    - **Baseline conversion rate** ({{< math >}}$ p ${{</ math>}}): the rate at which a desired action/event occurs in the control group
    - **Minimum Detectable Effect** ({{< math >}}$ \text{MDE} ${{</ math>}}): smallest difference in the conversion rate that you want to be able to detect as statistically significant
        - e.g., if {{< math >}}$ p=10\\%$ and we want a minimum improvement of $2\\%$, then $\text{MDE}=20\\% ${{</ math>}}
        - Both groups should be statistically similar in terms of characteristics for a fair comparison

<t>

4. **Splitting**: Randomly assign users into control & treatment groups 
    - **User-level**: the user consistently experience the same variation throughout their entire interaction.
        - Pros: useful when the tested variations are expected to have a long-lasting impact on user experience, reducing potential biases or confusion due to inconsistency.
    - **Request-level**: the system randomly determines the variation to be shown at each request made by the user, regardless of previous assignments.
        - Pros: useful for measuring immediate or session-specific effects of the tested variations, allowing us to capture any potential context-dependent or short-term effects.

<t>

5. **Measurement & Analysis**: Track and record user interactions, events, or conversions for both the control and treatment groups.  Compare the performance of the control and treatment groups using statistical analysis. Determine if there is a statistically significant difference in the metrics you are measuring between the variations.

&nbsp;

&nbsp;

# Q&A
## Types of ML
**Supervised vs Unsupervised**
- **Supervised**: **labeled** - learn a mapping from input data to output labels
- **Unsupervised**: **unlabeled** - find patterns/structures/relationships in the data with no labeled output
- **Semi-supervised**: **partially labeled** - use labeled data to guide learning, use unlabeled data to uncover patterns or improve model performance
- **Active**: **continuously labeled** - add most informative samples for labeling in an iterative and adaptive manner

**Parametric vs Nonparametric**
|  | Parametric | Nonparametric |
|:-:|:---------:|:--------------:|
| What | estimate trainable params from data  | learn patterns based on data alone |
| When | prior functional form/shape of data distribution | - |
| Where | when we have strong prior knowledge of data distribution | when we have complex/unknown data distribution, or EDA |
| Pros | high interpretability<br>low data requirement<br>high computational efficiency | high versatility |

**Generative vs Discriminative**
|  | Generative | Discriminative |
|:-:|:---------:|:--------------:|
| Model | {{< math >}}$ p(\mathbf{x},y)$ | $p(y\|\mathbf{x}) ${{</ math>}} |
| Pros | easy to fit<br>can handle missing features<br>can handle unlabeled data<br>can fit classes separately<br>better generalization | high accuracy<br>allow preprocessing<br>calibrated probability estimates |

&nbsp;

## Ensemble
Def: combine a bunch of smaller models together for prediction
- Types:
    - **Voting**: majority vote (cls) or average vote (reg)
    - **Bagging**: train multiple models with bootstrapped subsets of the same data
    - **Boosting**: train sequential models where each new model focuses on the data points that the previous models got wrong. 
    - **Stacking**: train a meta-model which takes the predictions of multiple base models as input and makes the final prediction (like an emperor).
- Q: Why does ensembling independently trained models generally improve performance?
- A: 
    - **Variance Reduction**: average out different errors on different data subsets {{< math >}}$ \rightarrow ${{</ math>}} more stable and accurate predictions 
    - **Generalization**: reduce bias in the same way above {{< math >}}$ \rightarrow$ better balance between bias and variance $\rightarrow ${{</ math>}} better generalization
    - **Diversity**: if each model has its own unique strength, the result combined would be way better
    - **Robustness to Noise & Outliers**: because they are averaged out

&nbsp;

## Occam's Razor
Def: simpler solutions are better.
- Q: Why does it matter?
- A: This is the fundamental principle which guides the direction of industrial ML:
    - **Model Selection**: If a simpler model reaches comparable performance as a complex model, the simpler model wins because of its higher interpretability, fewer computational resources, faster training and easiness to debug/retrain.
    - **Feature Selection**: If a smaller set of features reaches comparable performance as a larger set, the smaller set wins because of its higher interpretability, fewer computational resources, faster training and easiness to debug/retrain.
    - **Generalization**: Overfitting is a core issue in ML. Regularization and simpler configs for hyperparameters are best start points to reduce overfitting and lead to better generalization toward unseen data.
    - **Ensemble methods**: Aggregating simpler models reaches better performance than complex structures in many cases.

&nbsp;

- Q: Why is deep learning so popular now, despite Occam's Razor?
- A: The rapid growth of technology opened doors for us to do more DL stuff:
    - **Big Data**: Complex structures work better with larger #data
    - **Computational Resources**: Nvidia is the godfather of AI development (and also a demon who led gamers to financial misery)
    - **Transformer**: This thing is somehow OP and no one knows why, even till this day.
    - **Transfer Learning**: You can pretrain a gigantic general-purpose model and then finetune it on specialized tasks, which is also OP.
    - **Open-Source Communities**: Unlike CloseAI, most AI researchers/engineers are actually open about their projects and thus driving the development of complex NNs. (e.g., PyTorch, HuggingFace, etc.)
    - **Funding**: Everyone on the Wall Street wants to put a penny on an AI startup nowadays.

&nbsp;

## Production
- Q: Why does an ML model’s performance degrade in production?
- A: Far too many unperceivable factors in lab experiments exist in the real world...
    - **Data Drift**: Production data is likely different from training data. Changes in data distribution can lead to a mismatch between the model's assumptions and the real-world data.
    - **Concept Drift**: The relationship betwee features and target can change over time, especially in a dynamic environment.
    - **Feature Drift**: New features or feature transformation may occur irl. If feature engineering doesn't reflect these changes, RIP.
    - **Data Quality**: Even if somehow the training data and production data follow the same distribution, missing values, outliers, and noisy data are common irl.
    - **Scaling & Latency**: In production, the model may need to handle a much larger volume of data and respond quickly to enormous requests.
    - **Model Versioning**: Mismatches between R&D models and deployed models can introduce inconsistencies.
    - **Monitoring & Maintenance**: Even if you manage everything correctly by this stage, you simply have no idea when and where a random bug just appears for whatever reason.

&nbsp;

- Q: What problems might we run into when deploying large machine learning models?
- A: Resource issues, latency at high data throughput, scalability, model storage, network status, frequent model updates, vulnerability to adversarial attacks, privacy concerns, compatibility, complexity in operational overhead (management of such models), regulatory compliance, low interpretability, etc.