---
title : "Introduction"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: false
images: []
weight: 100
---
ML system design consists 4 parts:

<center>
<img src="/images/ml/system_design_flow.png"/>
</center>

# Problem

1. Reword the problem: inputs? outputs? why doing this?
2. Focus on **user experience**: user POV? user interaction? recent user experience report? company intention for users?
3. Clarify the problem scope
    - **Data constraints**: #samples? #features? feature types? (just get a rough idea at this step; smaller -> simpler; larger -> complex)
    - **Model constraints**: performance or quality? interpretability? retrainability? single general or multiple specific?
    - **Resource constraints**: time? computing resources? local or cloud?
4. Determine the evaluation metrics
    - **Offline**: MSE, P/R/F1, AUC, R^2, ...
    - **Online**: usage time, usage frequency, click rate, ...

# Data

1. Identify **features** ($X$) and **target** ($Y$)
    - 3 types of features: user, content, context
    - 2 types of target: explicit (direct indicators), implicit (indirect indicators)
2. Confirm data availability
    - **Available**: What is available? How much is available? Annotated? How good are the annotations?
    - **Unavailable**: What is unavailable? How expensive is data annotation? How many annotators for each sample? How to resolve annotators' disagreements? Is it possible to generate these data automatically (e.g., ChatGPT, rule-based generation, ...)
    - **Privacy**: What user data can we access? How do we access? How do we get user feedback on the system? Online or periodic data? Anonymity? 
    - **Logistics**: Where are the data (local or cloud)? What data structures? How big? What biases exist in such data?
3. **Data Processing**
4. **Feature Selection**

# Modeling

1. Give a baseline method (no model)
2. Give an easy model
3. Give a hard model (i.e., NNs)

For each model, specify:
- Functionality
- Assumption/Conditions (if any)
- Pros & Cons
- Objective & Optimization (i.e., loss & training)

# Production

1. Inference location:
    - local (user's phone/PC): low latency, require their memory/storage
    - server (ours): low memory/storage usage, high latency, privacy concerns
2. Feature serving:
    - **batch features** should be handled offline and served online (i.e., need daily/weekly jobs for data generation/collection)
    - **real-time features** should be handled and served at request time (i.e., need to create a feature store to look up features at serve time; caching may be necessary) $\rightarrow$ watch out for scalability and latency.
3. Performance Monitoring: latency, biases, data drift, CPU load, memory, ...
4. Retrain Frequency
5. Online A/B Testing

## Online A/B Testing
1. **Goal**: Define the objective. (e.g., improving click-through rates, increasing sign-up rates, etc.)
    - **Significancec level** ($\alpha$): threshold of whether the observed difference between the control and treatment groups is statistically significant or occurred by chance.
        - Common values: 0.05 and 0.01
        - A lower $\alpha$ makes it more challenging to detect difference
    - **Power** ($1-\beta$): probability of correctly rejecting $H_0$ when it is false, meaning the ability to detect a meaningful difference when it exists
        - Common value: 80%
        - A higher power requires larger sample sizes to achieve
    - *notes:
        - $\alpha=P$(FP), Type I error: probability of rejecting $H_0$ when it is true.
        - $\beta=P$(FN), Type II error: probability of not rejecting $H_0$ when it is false.

<t>

2. **Variation**: Generation 2/more versions of whatever element we want to test. (e.g., 2 different designs of a button; blue round button to **control group** A; green square button to **treatment group** B)

<t>

3. **Traffic**: Calculate required sample size per variation
$$n=2\times\left(\frac{Z_{\frac{\alpha}{2}}+Z_{\beta}}{\text{MDE}}\right)^2\times p(1-p)$$
    - $p$ (baseline conversion rate): the rate at which a desired action/event occurs in the control group
    - $\text{MDE}$ (Minimum Detectable Effect): smallest difference in the conversion rate that you want to be able to detect as statistically significant
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