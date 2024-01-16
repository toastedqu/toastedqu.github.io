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
# Concepts

**Universal Approximation Theorem**: A neural network with 1 hidden layer can approximate any continuous function for inputs within a specific range.
- Q: Why can't we reach a small error with a single layer then?
- A: There are many factors to consider when it comes to optimization.
    - **Layer size & Function complexity**: Simpler functions may be approximated with small #neurons, but complex functions may require an impractically enormous #neurons.
    - **Computational constraints**: Even if theoretically we can reach a small error, if we need like 1 quadrillion neurons to do it, we just can't.
    - **Optimization & Objective**: There are different objectives and different optimization methods for different types of problems, which determines what the error is like.
    - **Overfitting & Underfitting**: The training data is not representative of the entire population, or it's just noisy.
    - **Expressiveness**: 1 layers is, after all, a linear model that is most effective in capturing linear functions. Adding more layers with nonlinear activation functions would help the NN capture many more nonlinear functions, thus higher expressiveness.

&nbsp;

- Q: Ok but why is deep NN more expressive than wide NN, even if they have the same #params?
- A: There are many more complexities when we consider more complex functions that a single linear layer, no matter how wide, always fails to capture:
    - **Nonlinear Transformations**: As explained above, each layer in deep NN introduces a new set of nonlinear transformations, allowing the network to model intricate relationships in the data.
    - **Representational Capacity**: As explained above, deep NNs can represent a broader range of functions. As you increase the depth of the network, it gains the ability to approximate more complex functions, which can be essential for solving intricate tasks.
    - **Hierarchical Feature Extraction**: Deep NNs are designed to learn hierarchical representations of data. Each layer in a deep network extracts features from the output of the previous layer. For instance, in image processing, lower layers may detect edges and simple shapes, while higher layers may recognize complex objects. This hierarchical feature extraction allows the network to capture complex patterns by building upon simpler patterns and aids in solving tasks that require understanding data at different levels of granularity.
    - **Generalization**: Deep NNs tend to generalize better because they can learn more abstract and reusable features, whereas wide NNs overfit.
    - **Parameter Sharing**: Hey, while they have the same #params, they don't necessarily have the same #trainable params. Deep NNs can share parameters across layers, reducing the overall number of parameters required to capture complex relationships. This parameter sharing helps regularize the model and make it more data-efficient.

    But this doesn't necessarily mean wide NNs suck. If they are adequate for a problem, use them instead. Occam's Razor still holds!

&nbsp;

**Local Minimum**: A point in the parameter space where the loss is minimum, locally.
- Q: Why do we hate local minima?
- A: Because of **gradient descent**. When it hits a local minimum, it can't get out. If it can't get out, the model is suboptimal despite having potential to do better (with global minimum).

**Saddle Point**: A point in the parameter space where the loss gradient is 0 but it is not a local or global minimum. (Try visualize a saddle in coordinates. There you go.)
- Q: Why do we hate saddle points?
- A: Because of **gradient descent**. When it hits a saddle point, it thinks it hits a minimum because where else can the gradient be 0 right? Then the model becomes suboptimal despite having potential to do better (with global minimum).

&nbsp;

- Q: Which one is worse for large NNs, local minima or saddle points?
- A: **Saddle points**. At least local minima are stable, but saddle points can lead to slow convergence because the gradients are small in some directions, giving the appearance of convergence, even though the algorithm is not making meaningful progress toward the global minimum. In fact, saddle point is like a mixture of minima and maxima, and we want to avoid maxima at all cost.

&nbsp;

- Q: What do we do about local minima and saddle points then?
- A: 3 directions: 
    - Optimization (tell GD to get outta there): SGD, Momentum, Adam, ...
    - Initialization (reduce the chance of getting 0 in gradient): random init, ...
    - Hyperparameter tuning (further reduce the chance of getting 0 in gradient): early stopping, ...

