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
A typical ML project consists of 4 parts:

<center>
<img src="/images/ml/system_design_flow.png"/>
</center>

Problem, Data, and Production are far more important than Modeling.

# Problem
Stage 1: **Understanding**: what's the purpose? what inputs? what outputs? (rough idea)

Stage 2: **Scope**: what are the constraints? (smaller $\rightarrow$ simpler; larger $\rightarrow$ complex)
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

Stage 3: **Evaluation**: how do we know if our model meets our needs?
- **User experience**: user interaction? recent user experience report? company intention for users? personalization?
- **Evaluation metrics**:
    - **Offline**: MSE, P/R/F1, AUC, R^2, ...
    - **Online**: usage time, usage frequency, click rate, ...

&nbsp;

# Data
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

# Modeling
Stage 1: Baseline method (no model): random benchmark, average, etc.

Stage 2: Easy model

Stage 3: Hard model

For each model, specify:
- Functionality
- Assumptions (if any)
- Pros & Cons
- Objective & Optimization (i.e., loss & training)

&nbsp;

# Production
Production is completely different from experiments. The following factors need to be taken into account:
1. **Inference location**:
    - **local** (user's phone/PC): low latency, require their memory/storage
    - **server** (ours or cloud): low memory/storage usage, high latency, privacy concerns
2. **Feature serving**:
    - **batch features** should be handled offline and served online (i.e., need daily/weekly jobs for data generation/collection)
    - **real-time features** should be handled and served at request time (i.e., need to create a feature store to look up features at serve time; caching may be necessary) $\rightarrow$ watch out for scalability and latency.
3. **Performance Monitoring**: latency, biases, data drift, CPU load, memory, ...
4. **Retrain Frequency**
5. **Online A/B Testing**

&nbsp;

## Online A/B Testing
1. **Goal**: Define the objective. (e.g., improving click-through rates, increasing sign-up rates, etc.)
    - **Significance level** ($\alpha$): threshold of whether the observed difference between the control and treatment groups is statistically significant or occurred by chance.
        - Common values: 0.05 and 0.01
        - A lower $\alpha$ makes it more challenging to detect difference
    - **Power** ($1-\beta$): probability of correctly rejecting $H_0$ when it is false, meaning the ability to detect a meaningful difference when it exists
        - Common value: 80%
        - A higher power requires larger sample sizes to achieve
    - *notes:
        - $\alpha=P$(FP), Type I error: probability of rejecting $H_0$ when it is true.
        - $\beta=P$(FN), Type II error: probability of not rejecting $H_0$ when it is false.

<t>

2. **Variation**: Generate 2/more versions of whatever element we want to test. (e.g., 2 different designs of a button; blue round button to **control group** A; green square button to **treatment group** B)

<t>

3. **Traffic**: Calculate required sample size per variation
$$n=2\times\left(\frac{Z_{\frac{\alpha}{2}}+Z_{\beta}}{\text{MDE}}\right)^2\times p(1-p)$$
    - **Baseline conversion rate** ($p$): the rate at which a desired action/event occurs in the control group
    - **Minimum Detectable Effect** ($\text{MDE}$): smallest difference in the conversion rate that you want to be able to detect as statistically significant
        - e.g., if $p=10\\%$ and we want a minimum improvement of $2\\%$, then $\text{MDE}=20\\%$
        - Both groups should be statistically similar in terms of characteristics for a fair comparison

<t>

4. **Splitting**: Randomly assign users into control & treatment groups 
    - **User-level**: the user consistently experience the same variation throughout their entire interaction.
        - Pros: useful when the tested variations are expected to have a long-lasting impact on user experience, reducing potential biases or confusion due to inconsistency.
    - **Request-level**: the system randomly determines the variation to be shown at each request made by the user, regardless of previous assignments.
        - Pros: useful for measuring immediate or session-specific effects of the tested variations, allowing us to capture any potential context-dependent or short-term effects.

<t>

5. **Measurement & Analysis**: Track and record user interactions, events, or conversions for both the control and treatment groups.  Compare the performance of the control and treatment groups using statistical analysis. Determine if there is a statistically significant difference in the metrics you are measuring between the variations.