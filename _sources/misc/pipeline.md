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

# Overview
An AI project consists of 4 parts:

```{image} ../images/ml/system_design_flow.png
:align: center
```

## Problem
1. **Overview**: What problem? Why? What inputs & outputs? (rough idea)
2. **Scope**: What constraints? 
    - **Data constraints**: #samples, #features, feature types, etc.
    - **Model constraints**: priority (performance/quality), model type (single general/multiple specific), interpretability, retrainability, etc.
    - **Resource constraints**: time (training, inference, project duration, etc.), computation (training, inference, local/cloud, etc.)
3. **Evaluation**: Is the problem solved? How do we know?
    - **Auto metrics**: offline (MSE, P/R/F1, etc.), online (usage time, usage frequency, click rate, etc.)
    - **Human metrics**: user interaction, recent reports, company intention for users, personalization, etc.

## Data
1. **Type**
    - **Features**: user, content, context, etc.
    - **Targets**: explicit (direct indicators), implicit (indirect indicators)
2. **Availability**
    - **Available/Unavailable**: What is available/unavailable? How much is available/unavailable? 
    - **Annotation**: Are they Annotated? How good are the annotations? How expensive to annotate the rest? How to resolve annotators' disagreements? Is auto-annotation feasible (e.g., ChatGPT, rule-based generation, etc.)?
    - **Privacy**: What user data can we access? How do we obtain them? Can we use online/periodic data? Do we need anonymity?
    - **Logistics**: Where are the data (local/cloud)? What data structures? How big? What biases in it?
3. [**Processing (ETL)**](../ml/data.md)
4. [**Feature Engineering**](../ml/unsupervised.md)

## Modeling
NOTE: For each model, specify:
- Why (Motivation)
- What (Functionality)
- How (Objective & Optimization)
- When (Assumptions if any)
- Pros & Cons

Procedure:
1. Baseline ~~model~~: stats (mean, median, mode), random benchmarks, etc.
2. Easy model
3. Hard model
4. Experiment, Evaluation & Ablation Study

## Production
NOTE: Production $\neq$ Experiment. Performance degrades in production because of uncertainty:
- **Data Drift**: production data $\neq$ training data $\rightarrow$ assumption fails
- **Feature Drift**: new features / feature transformations $\rightarrow$ feature engineering pipeline fails
- **Concept Drift**: Relationship between features and target variable can change over time, especially in a dynamic environment.
- **Data Quality**: missing values, outliers, noisy data, etc.
- **Model Versioning**: mismatches between R&D models & deployed models
- **Scaling & Latency**: Models IRL need to handle a significantly larger volume of data and to respond significantly faster to enormous requests.
- **Ethics**: adversarial attacks, privacy concerns, regulatory compliance, interpretability, etc.
- **Others**: Even if everything is flawless, a random error may just appear for no reason (e.g., network issues)

Consider the following factors for production:
1. **Inference location**:
    - **local** (phone/PC): high memory/storage usage, low latency
    - **server** (ours/cloud): low memory/storage usage, high latency, privacy concerns
2. **Feature serving**:
    - **batch features** should be handled offline & served online
        - need daily/weekly jobs for auto data generation/collection
    - **real-time features** should be handled & served at request time (scalability & latency = priority)
        - need a feature store to look up features at serve time
        - caching may be necessary
3. **Performance Monitoring**: errors, latency, biases, data drift, CPU load, memory, etc.
4. [**Online A/B Testing**](#online-a-b-testing)
5. **Retrain Frequency**



# Concepts
## Types of ML
- Supervision
    - **Supervised** (labeled): learn a mapping from input data to output labels
    - **Unsupervised** (unlabeled): find patterns/structures/relationships in data with no labeled output
    - **Semi-supervised** (partially labeled): use labeled data to guide learning, use unlabeled data to uncover patterns or improve model performance
    - **Active** (continuously labeled): add most informative samples for labeling in an iterative and adaptive manner
- Parameterization
    - **Parametric**: estimate trainable params from data
        - Assume prior functional form/shape of data distribution
        - Used when we have strong prior knowledge
        - Pros: high interpretability, low data requirement, high computational efficiency
    - **Nonparametric**: learn patterns based on data alone
        - Assume nothing
        - Used in EDA / when we have unknown data distribution
        - Pros: high versatility
- Prediction
    - **Generative**: model $p(x,y)$
        - Pros: easy to fit, handle missing features, handle unlabeled data, fit classes separately, better generalization
        - Cons: low computational efficiency, sensitive to incorrect assumptions
    - **Discriminative**: model $p(y|x)$
        - Pros: high accuracy, allow preprocessing, provide calibrated probability estimates, robust to irrelevant features
        - Cons: data-dependent, limited application scope
- **Ensemble**: combine a bunch of smaller models together for prediction
    - Types:
        - **Voting**: majority vote (cls) / average vote (reg)
        - **Bagging**: train multiple models with bootstrapped subsets of the same data
        - **Boosting**: train sequential models where each new model focuses on the samples that the previous models got wrong. 
        - **Stacking**: train a meta-model which takes the predictions of multiple base models as input and makes the final prediction as output.
    - Performance improvement (relative to single models):
        - **Variance Reduction**: average out different errors on different subsets $\rightarrow$ higher accuracy & stability
        - **Generalization**: reduce bias $\rightarrow$ better generalization
        - **Diversity**: each model has its own unique strength
        - **Robustness to Noise & Outliers**: they are averaged out

## Occam's Razor
"Simpler solutions are better."
- Why better?
    - higher interpretability, fewer computational resources, faster training and easiness to debug/retrain
- Why does it matter?
    - **Model Selection**: If a simpler model reaches comparable performance as a complex model, the simpler model wins.
    - **Feature Selection**: If a smaller feature set reaches comparable performance as a larger feature set, the smaller set wins.
    - **Generalization**: Overfitting is a core issue in ML. Simpler configs for hyperparameters are best start points to reduce overfitting and lead to better generalization for unseen data.
    - **Ensemble methods**: Aggregating simpler models reaches better performance than complex models in many cases.
- Why deep learning if Occam's Razor holds?
    - **Big Data**: Complex structures work better with larger data
    - **Computational Resources**: Thx Nvidia
    - **Transformer**: Thx Google
    - **Transfer Learning**: We can pretrain a gigantic general-purpose model and finetune it on specialized tasks to reach significantly better performance

## Online A/B Testing
1. **Define Objective**: improving click-through rates, increasing sign-up rates, etc.
    - **Significance level** $(\alpha)$: threshold of whether the observed difference between control & treatment is statistically significant
        - $\alpha=P(FP)$ (i.e., Type I error): probability of rejecting $H_0$ when it is true.
        - Common values: 0.05 and 0.01
        - A lower $\alpha$ makes it more challenging to detect difference.
    - **Power** $(1-\beta)$: probability of rejecting $H_0$ when it is false (i.e., the ability to detect a meaningful difference when it exists)
        - $\beta=P(FN) (i.e., Type II error): probability of not rejecting $H_0$ when it is false.
        - Common value: 80%
        - A higher power requires larger sample sizes to achieve.
2. **Create Variations**: generate 2/more versions of the element we want to test
    - e.g., 2 different designs of a button: blue round button $\rightarrow$ **control group** A; green square button $\rightarrow$ **treatment group** B
3. **Calculate Traffic**: calculate required sample size per variation 
$$m=2\times\left(\frac{Z_{\frac{\alpha}{2}}+Z_{\beta}}{\text{MDE}}\right)^2\times p(1-p)$$
    - $p$ (**Baseline conversion rate**): the occurrence rate of a desired event in the control group
    - $\text{MDE}$ (**Minimum Detectable Effect**): smallest difference in the conversion rate that we want to be able to detect as statistically significant
        - e.g., if $p=10\%$ and we want a minimum improvement of $2\%$, then $\text{MDE}=20\%$
        - Both groups should be statistically similar in terms of characteristics for a fair comparison
4. **Splitting**: randomly assign users into control & treatment groups 
    - **User-level**: The user consistently experience the same variation throughout their entire interaction.
        - Pros: useful when the tested variations are expected to have a long-lasting impact on user experience, reducing potential biases or confusion due to inconsistency
    - **Request-level**: The system randomly determines the variation to be shown at each request made by the user, regardless of previous assignments.
        - Pros: useful for measuring immediate or session-specific effects of the tested variations, allowing us to capture any potential context-dependent or short-term effects
5. **Measurement & Analysis**: 
    1. Track & record user interactions, events, or conversions for both the control and treatment groups.
    2. Compare the performance of control & treatment groups using statistical analysis.
    3. Determine if there is a statistically significant difference in the metrics we are measuring between the tested variations.