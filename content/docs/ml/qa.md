---
title : "Q&A"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: false
images: []
weight: 7
---
# Types of ML
- **Supervised vs Unsupervised**
    - **Supervised**: labeled - learn a mapping from input data to output labels
    - **Unsupervised**: unlabeled - find patterns/structures/relationships in the data with no labeled output
    - **Weakly-supervised**: partially labeled - leverage limited label info to make predictions or discover patterns
    - **Semi-supervised**: partially labeled - use labeled data to guide learning, use unlabeled data to uncover patterns or improve model performance
    - **Active**: continuously labeled - add most informative samples for labeling in an iterative and adaptive manner
- **Parametric vs Nonparametric**
    - **Parametric**: has trainable parameters
        - Assumption: prior functional form/shape of data distribution
        - Goal: estimate trainable parameters from data 
        - Usage: when we have strong prior knowledge/evidence of data distribution
        - Pros: more interpretable, require fewer data, less computational complexity
    - **Non-Parametric**: 
        - Assumption: n/a
        - Goal: learn pattern based on data alone
        - Usage: when we have complex/unknown data distribution; EDA
        - Pros: more versatile
- **Classification vs Regression**:
    - Classification: a discrete set of labels
    - Regression: a continuous range of values
    - Cls $\rightarrow$ Reg: assign numerical values to labels, treat them as target values
    - Reg $\rightarrow$ Cls: bucket numerical values into bins, treat them as labels

&nbsp;

&nbsp;

# Ensemble
Def: combine a bunch of smaller models together for prediction
- Types:
    - **Voting**: majority vote (cls) or average vote (reg)
    - **Bagging**: train multiple models with bootstrapped subsets of the same data
    - **Boosting**: train sequential models where each new model focuses on the data points that the previous models got wrong. 
    - **Stacking**: train a meta-model which takes the predictions of multiple base models as input and makes the final prediction (like an emperor).
- Q: Why does ensembling independently trained models generally improve performance?
- A: 
    - **Variance Reduction**: average out different errors on different data subsets $\rightarrow$ more stable and accurate predictions 
    - **Generalization**: reduce bias in the same way above $\rightarrow$ better balance between bias and variance $\rightarrow$ better generalization
    - **Diversity**: if each model has its own unique strength, the result combined would be way better
    - **Robustness to Noise & Outliers**: because they are averaged out

&nbsp;

&nbsp;

# Occam's Razor
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

&nbsp;

# Production
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
