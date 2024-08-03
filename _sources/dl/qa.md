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
# Q&A
Please use the right sidebar as TOC.

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
**What**: Simpler solutions are better.
- higher interpretability, fewer computational resources, faster training and easiness to debug/retrain, etc.

**Why does it matter?**
- **Model Selection**: If a simpler model reaches comparable performance as a complex model, the simpler model wins.
- **Feature Selection**: If a smaller feature set reaches comparable performance as a larger feature set, the smaller set wins.
- **Generalization**: Overfitting is a core issue in ML. Simpler configs for hyperparameters are best start points to reduce overfitting and lead to better generalization for unseen data.
- **Ensemble methods**: Aggregating simpler models reaches better performance than complex models in many cases.

**Why deep learning if Occam's Razor holds?**
- **Big Data**: Complex structures work better with larger data.
- **Computational Resources**: Thx Nvidia.
- **Transformer**: Thx Google.
- **Transfer Learning**: We can pretrain a gigantic general-purpose model and finetune it on specialized tasks to reach significantly better performance.


## Universal Approximation Theorem
**What**: A neural network with 1 hidden layer can approximate any continuous function for inputs within a specific range.

**Why not a single layer?** There are many factors to consider when it comes to optimization.
- **Layer size & Function complexity**: Simpler functions may be approximated with small #neurons, but complex functions may require an impractically enormous #neurons.
- **Computational constraints**: Even if theoretically we can reach a small error, if we need like 1 quadrillion neurons to do it, we just can't.
- **Optimization & Objective**: There are different objectives and different optimization methods for different types of problems, which determines what the error is like.
- **Overfitting & Underfitting**: The training data is not representative of the entire population, or it's just noisy.
- **Expressiveness**: 1 layers is, after all, a linear model that is most effective in capturing linear functions. Adding more layers with nonlinear activation functions would help the NN capture many more nonlinear functions, thus higher expressiveness.

**Why is deep NN more expressive than wide NN?** There are many more complexities when we consider more complex functions that a single linear layer, no matter how wide, always fails to capture:
- **Nonlinear Transformations**: As explained above, each layer in deep NN introduces a new set of nonlinear transformations, allowing the network to model intricate relationships in the data.
- **Representational Capacity**: As explained above, deep NNs can represent a broader range of functions. As you increase the depth of the network, it gains the ability to approximate more complex functions, which can be essential for solving intricate tasks.
- **Hierarchical Feature Extraction**: Deep NNs are designed to learn hierarchical representations of data. Each layer in a deep network extracts features from the output of the previous layer. For instance, in image processing, lower layers may detect edges and simple shapes, while higher layers may recognize complex objects. This hierarchical feature extraction allows the network to capture complex patterns by building upon simpler patterns and aids in solving tasks that require understanding data at different levels of granularity.
- **Generalization**: Deep NNs tend to generalize better because they can learn more abstract and reusable features, whereas wide NNs overfit.
- **Parameter Sharing**: Hey, while they have the same #params, they don't necessarily have the same #trainable params. Deep NNs can share parameters across layers, reducing the overall number of parameters required to capture complex relationships. This parameter sharing helps regularize the model and make it more data-efficient.

But this doesn't necessarily mean wide NNs suck. If they are adequate for a problem, use them instead. Occam's Razor still holds.

## Local Minimum & Saddle Point
**What**: 
- **Local Minimum**: A point in the parameter space where the loss is minimum, locally.
- **Saddle Point**: A point in the parameter space where the loss gradient is 0 but it is not a local or global minimum. (Try visualize a saddle in coordinates. There you go.)

**Why do we hate them?** Gradient descent.
- **Local Minimum**: When loss hits a local minimum, it can't get out. If it can't get out, the model is suboptimal despite having potential to do better (with global minimum).
- **Saddle Point**: When loss hits a saddle point, it thinks it hits a minimum because where else can the gradient be 0? Then the model becomes suboptimal despite having potential to do better (with global minimum).

**Which one is worse for large NNs?** Saddle points. At least local minima are stable, but saddle points can lead to slow convergence because the gradients are small in some directions, giving the appearance of convergence, even though the algorithm is not making meaningful progress toward the global minimum. In fact, saddle point is like a mixture of minima and maxima, and we want to avoid maxima at all cost.

**What can we do about them?** 3 directions: 
- Optimization (tell GD to get outta there): SGD, Momentum, Adam, etc.
- Initialization (reduce the chance of getting 0 in gradient): random init, etc.
- Hyperparameter tuning (further reduce the chance of getting 0 in gradient): early stopping, etc.