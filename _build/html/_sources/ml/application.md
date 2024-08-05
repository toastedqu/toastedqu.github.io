---
title : "Applications"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: true
images: []
weight: 6
---
There will be no common notations as different sections involve different aspects. However, most notations will be straightforward and used in all previous chapters.

Congrats! By understanding all the previous chapters, we finally became able to understand 1% of how ML works in practice. Very nice.

Ever wonder why MLE interviews focus hugely on software engineering, system design, and experiences with data processing?

# Overview of System Design

**1. Clarify Requirements**
- Data
    - What size? small $ \rightarrow$ simpler model; large $\rightarrow $ complex model
    - What is $ y $? Regression/Classification/Clustering
    - What is $ x $?
- Hardware Constraints
    - Time limit?
    - Space limit?
- Objective
    - Accuracy or Speed? small models $ \rightarrow$ faster; large models $\rightarrow $ accurate
    - Will we retrain after eval?

**2. Determine Metrics**
- Offline Metrics:
    - i.e., training/testing metrics
    - e.g., AUC, precision/F1, R^2, MSE, etc.
- Online Metrics:
    - i.e., eval metrics
    - e.g., click rate, active hours, etc.
- Non-functional metrics:
    - Training speed
    - Scalability to large datasets
    - Extensibility to new techniques
    - Convenience for deployment

**3. Input & Data**
- Target variable $ y $
    - Explicit: a user action that directly indicates the label/value of $ y $. (e.g., "bought", "liked") (BEST)
    - Implicit: a user action that might potentially relate to $ y $. (e.g., "save for later", long review, etc.) (extra)
- Features $ x $
    - Identify different features for different systems.
    - Feature engineering
        - Train/Test split
        - Missing values/Outliers
        - Scaling
        - Balance pos/neg samples (e.g., up-sampling, under-sampling, SMOTE)
    - Feature selection
        - Use some models like trees, L1, L0, etc.
        - Unnecessary for large models.
- Extra concerns
    - Sample range: are we sampling from a large enough subset of demographics?
    - Privacy/law: anonymize? remove some features violating privacy?
    - Data accessibility:
        - tabular: SQL
        - images/videos: GCP

**4. Model**
- Order:
    - A baseline model with no ML component
        - e.g., majority vote, max/min, mean, etc.
    - Traditional ML models (typically small & fast)
    - Advanced models (typically large & slow)
- Model Explanation
    - Idea & Procedure (rough)
    - Key hyperparameters
    - Loss/Optimization objective
    - Pros & Cons

**5. Output & Serving**
- Online A/B Testing
- Where to run inference
- Monitoring performance
- Biases/Misuses of model
- Retraining frequency

# Experimental Design

## Active Learning
Problem: Given existing data $ (X,y) $, choose where to collect more labels.

Assumption: access to cheap unlabeled samples.

Procedure:
1. Active learner picks which sample(s) $ \textbf{x} $ to query by minimizing some loss (risk).
2. An oracle (ground truth) generates a label/response $ y $.
3. Update the params of the model.
4. Repeat Step 1-3.

**Uncertainty Sampling**: Select **most uncertain** samples to query
- Measures of **uncertainty**: entropy, least confident predicted label, Euclidean distance (e.g., point closest to SVM margin)
- Cons: Sensitive to noisy data where uncertainty is the highest (e.g., images)

**Information-based Loss**: Select **most informative** samples to query
- Quantification of **information gain**:
    - Maximize **KL divergence** between posterior and prior $ KL(p(\textbf{w}|X,\textbf{x}')||p(\textbf{w}|X))$ (i.e., maximize $\\# $bits gained)
    - Maximize **model entropy reduction** between posterior and prior (i.e., reduce $ \\# $bits required to describe distribution)
- Pros: Better than Uncertainty Sampling (by looking at the expected effect after adding the sample to the model)

Pros:
- Save computational cost

## Optimal Experimental Design
Problem: Given a parametric model, find which queries are maximally informative (i.e., best minimizes $ \text{Var}[\hat{\textbf{w}}]\leftarrow $ minimize the inverse covariance matrix of samples):
$$
\hat{\textbf{w}}\sim N(\textbf{w},\sigma^2(X^TX)^{-1})
$$
- Linear models: optimal design is independent of $ \textbf{w} $.
- Nonlinear models: use Taylor expansion to linear model.
- The more samples, the smaller the covariance.

Solutions:
- **A-optimal (average) design**: minimize $ \text{trace}(X^TX)^{-1} $
- **D-optimal (log&det) design**: minimize $ \log\det(X^TX)^{-1} $
- **E-optimal (extreme) design**: minimize $ \max\text{eigenvalue}(X^TX)^{-1} $
- The best one so far is A-optimal, which is equivalent to minimizing Frobenius norm $ \rightarrow $ minimizing size of matrix.
- A-optimal effectively chooses points near the border of the dataset.

## Response Surface Modeling
Problem: Find the optimum of an unknown function $ y=f(\textbf{x}) $.
- Why? In some situations we don't care about what the model is. We simply want to know where the optimum is. (e.g., find optimal conditions for growing cell cultures)

Idea: Simultaneously learn what the function looks like near the optimum and find the optimum.

Procedure: **RL**
1. Initialize: Given data $ (X,\textbf{y})$, fit $y=f(\textbf{x};\textbf{w}) $ as the **response surface**.
2. Repeat:
    - Pick the next $ \textbf{x}_i$ by doing GD on $f(\textbf{x}) $ (i.e., action).
    - Measure the corresponding $ y_i $ (i.e., reward).
    - Add $ (\textbf{x}_i,y_i)$ to the training data to update response surface $f(\textbf{x}) $.

# Explanable AI (XAI)
Why do we want to explain ML/AI? 
- Explain the model: **Debugging/Verification**
- Explain the world: **Science**
- Explain the prediction: **Decision support**

Types of Explanations:
- **Interventional vs Conditional**:
    - Interventional changes one feature with others fixed (focus on explaining the model).
    - Conditional changes other features to respect correlations.
- **Model-based vs Model-agnostic**:
    - Model-based looks at the params.
    - Model-agnostic focuses on explaining the world.
- **Local vs Global**:
    - Local looks at the feature importance for a single sample or a small group of samples.
    - Global looks at the average feature importance over all samples.

Measures of **Feature Importance**:
- **Univariate Correlation**
- **Replacement with Mean/Zero** (e.g., LIME uses Zero and explains which features in a model are most important for predicting at a particular point in the training data)
- **Permutation**
- **Remove & Retrain**
- **Partial Dependence Plot**: marginalize over all other features and look at the effect of the current feature (effectively assume features are independent):
$$
\hat{f}S(x_S)=\frac{1}{m}\sum\_{i=1}^{m}\hat{f}(x_S,\textbf{x}_C^{(i)})
$$
- **Shapley values (SHAP)**: a class of efficiently computable metrics with great axiomatic properties.
    - **Local SV**: measure the change in prediction accuracy between using the true value of the feature(s) and masking the feature(s) with baseline values (e.g., mean):
    $$
    \phi\_{ij}(\textbf{w}^T\textbf{x})=w_j(x\_{ij}-\mathbb{E}[\textbf{x}_j])
    $$
    - **Global SV**: measure the average of the absolute value of the accuracy changes over all samples:
    $$
    \phi_j(\textbf{w}^T\textbf{x})=\frac{1}{m}\sum\_{i=1}^mw_j(x\_{ij}-\mathbb{E}[\textbf{x}_j])
    $$
    - Properties of SV:
        - **Efficiency**: The feature contributions $ \phi_j $ must add up to the difference of the prediction for x and the average.
        $$
        \sum\_{j=1}^p\phi_j=\hat{f}(x)-\mathbb{E}_X[\hat{f}(X)]
        $$
        - **Additivity**: For a game with combined payoutts $ val+val^+$, the respective Shapley values are $\phi_j+\phi_j^+ $.
        - **Dummy**: A feature $ j$ that has no effect has $\phi_j=0 $.
        - **Symmetry**: For features $ i,j$ with identical effect, $\phi_i=\phi_j $.
        - **Consistency**: If a model is altered so that the marginal contribution of a feature value increases (regardless of other features), the Shapley value also increases.

**Contenability**: It generally doesn’t make sense to change one feature without others changing as well. (i.e., Cons of measures of feature importance)

# AutoML
Problem: Automate hyperparameter tuning.

AutoML automates the search for the best hyperparams in an ensemble of sklearn models / NN structures. So far, it has insane performance, is less likely to overfit, and produces varieties of ensembles for various problem types.

Yes. R.I.P. "*import sklearn*" folks.

## Auto-sklearn
Idea: Combined Algorithm Selection and Hyperparameter Optimization (CASH Opt)

Overview: 15 classifiers, 14 feature preprocessing methods, 4 data preprocessing techniques $ \rightarrow $ 110 hyperparams

Assumption: **Similar problems need similar models.**

**Warmstart/Metalearning**: start with hyperparams that worked in the past for similar datasets (38 metafeatures from 140 datasets)

**Bayesian Optimization**:
- Use a fitted random forest to find the optimal hyperparams.
    - Fit on what: Given the problem description and hyperparams, predict how accurate they will be.
- Discard bad values on the first fold of 10-fold CV to accelerate.

**Ensemble Selection**: Stagewisely generate an ensemble of the top 50 classifiers
- Start with an empty ensemble.
- Iteratively add the model that minimizes ensemble validation loss, with uniform weight but allowing repetitions.
    - Why no weight optimization? To **avoid overfitting**: We do not want perfect fitting to residuals at each step, so equal weight.
    - The more accurate we get to, the weaker model gets added to the ensemble.

**Metafeatures**: useful for assessing similarities between two problems/datasets.
- $ \\#$samples, $\\# $features
- $ \\# $missing
- Fraction of Numerical/Categorical features
- Data transformations
- Exploratory statistics
- PCs, entropy, etc.
- The modern way: Use **language embeddings** from text descriptions of problems to pick hyperparams
    1. Generate/Acquire vector embeddings of dataset title, description and keywords.
    2. Calculate similarity between past datasets and new dataset (the similarity metric here is learnt in advance).
    3. Find the most similar prior dataset and use its hyperparams.

**Nested CV**: further reduction in overfitting
1. For each of 10 folds on the selected 90% data, do 10-fold CV to find the best method.
2. Observe performance on the held-out 10%.

Pros:
- Bye sklearn!

Cons:
- (weak) High computational cost (BUT it is worth it)
- Limited to sklearn (i.e., tabular data)
- Unstable in rare cases (still need DS folks to monitor at the moment)

## AutoDL
Idea: Use RL to search for the best NN architecture.
- Search in the embedding space of PyTorch code and do GD on it. Generate tons of them and see which ones are good, like states in RL.
- Why RL instead of other search algorithms? We want only a little exploration and most exploitation.

Agent: a controller RNN which generates sequence of characters (Python code).
- Action: text specifying the NN structure.

Environment: train a NN with architecture $ A$ to get accuracy/reward $R $.

Agent $ \rightarrow$ Environment: sample architecture $A$ with probability $p $.

Environment $ \rightarrow$ Agent: compute gradient of $p$ and scale it by $R $ to update the controller's policy.

Pros & Cons: still in progress.





<!-- # What really matters in ML?
ML system design consists 4 parts:

<center>
<img src="/images/ml/system_design_flow.png"/>
</center>

## Problem
1. Reword the problem: inputs? outputs? why doing this?
2. Focus on **user experience**: user POV? user interaction? recent user experience report? company intention for users?
3. Clarify the problem scope
    - **Data constraints**: #samples? #features? feature types? (just get a rough idea at this step; smaller -> simpler; larger -> complex)
    - **Model constraints**: performance or quality? interpretability? retrainability? single general or multiple specific?
    - **Resource constraints**: time? computing resources? local or cloud?
4. Determine the evaluation metrics
    - **Offline**: MSE, P/R/F1, AUC, R^2, etc.
    - **Online**: usage time, usage frequency, click rate, etc.

## Data
1. Identify **features** ($ X$) and **target** ($Y $)
    - 3 types of features: user, content, context
    - 2 types of target: explicit (direct indicators), implicit (indirect indicators)
2. Confirm data availability
    - **Available**: What is available? How much is available? Annotated? How good are the annotations?
    - **Unavailable**: What is unavailable? How expensive is data annotation? How many annotators for each sample? How to resolve annotators' disagreements? Is it possible to generate these data automatically (e.g., ChatGPT, rule-based generation, etc.)
    - **Privacy**: What user data can we access? How do we access? How do we get user feedback on the system? Online or periodic data? Anonymity? 
    - **Logistics**: Where are the data (local or cloud)? What data structures? How big? What biases exist in such data?
3. **Data Processing**
4. **Feature Selection**

## Modeling
1. Give a baseline method (no model)
2. Give an easy model
3. Give a hard model (i.e., NNs)

For each model, specify:
- Functionality
- Assumption/Conditions (if any)
- Pros & Cons
- Objective & Optimization (i.e., loss & training)

## Production
1. Inference location:
    - local (user's phone/PC): low latency, require their memory/storage
    - server (ours): low memory/storage usage, high latency, privacy concerns
2. Feature serving:
    - **batch features** should be handled offline and served online (i.e., need daily/weekly jobs for data generation/collection)
    - **real-time features** should be handled and served at request time (i.e., need to create a feature store to look up features at serve time; caching may be necessary) $ \rightarrow $ watch out for scalability and latency.
3. Performance Monitoring: latency, biases, data drift, CPU load, memory, etc.
4. Retrain Frequency
5. Online A/B Testing

### Online A/B Testing
1. **Goal**: Define the objective. (e.g., improving click-through rates, increasing sign-up rates, etc.)
    - **Significancec level** ($ \alpha $): threshold of whether the observed difference between the control and treatment groups is statistically significant or occurred by chance.
        - Common values: 0.05 and 0.01
        - A lower $ \alpha $ makes it more challenging to detect difference
    - **Power** ($ 1-\beta$): probability of correctly rejecting $H_0 $ when it is false, meaning the ability to detect a meaningful difference when it exists
        - Common value: 80%
        - A higher power requires larger sample sizes to achieve
    - *notes:
        - $ \alpha=P$(FP), Type I error: probability of rejecting $H_0 $ when it is true.
        - $ \beta=P$(FN), Type II error: probability of not rejecting $H_0 $ when it is false.

<t>

2. **Variation**: Generation 2/more versions of whatever element we want to test. (e.g., 2 different designs of a button; blue round button to **control group** A; green square button to **treatment group** B)

<t>

3. **Traffic**: Calculate required sample size per variation
$ $n=2\times\left(\frac{Z_{\frac{\alpha}{2}}+Z_{\beta}}{\text{MDE}}\right)^2\times p(1-p)$ $
    - $ p $ (baseline conversion rate): the rate at which a desired action/event occurs in the control group
    - $ \text{MDE} $ (Minimum Detectable Effect): smallest difference in the conversion rate that you want to be able to detect as statistically significant
        - e.g., if $ p=10\\%$ and we want a minimum improvement of $2\\%$, then $\text{MDE}=20\\% $
    - Both groups should be statistically similar in terms of characteristics for a fair comparison

<t>

4. **Splitting**: Randomly assign users into control & treatment groups 
    - **User-level**: the user consistently experience the same variation throughout their entire interaction.
        - Pros: useful when the tested variations are expected to have a long-lasting impact on user experience, reducing potential biases or confusion due to inconsistency.
    - **Request-level**: the system randomly determines the variation to be shown at each request made by the user, regardless of previous assignments.
        - Pros: useful for measuring immediate or session-specific effects of the tested variations, allowing us to capture any potential context-dependent or short-term effects.

<t>

5. **Measurement & Analysis**: Track and record user interactions, events, or conversions for both the control and treatment groups.  Compare the performance of the control and treatment groups using statistical analysis. Determine if there is a statistically significant difference in the metrics you are measuring between the variations. -->
