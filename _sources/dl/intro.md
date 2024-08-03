# Deep Learning

## What
**Intuition**: Deep learning is a subset of machine learning that uses neural networks with many layers (hence "deep") to model complex patterns in data.

**Mechanism**: It involves training neural networks by adjusting weights through backpropagation to minimize error in predictions or classifications.

**Components**:
- **Neurons**: Basic units that receive inputs, apply a function, and produce outputs.
- **Layers**: Stacked neurons where each layer processes outputs from the previous layer.
- **Activation Functions**: Functions like ReLU or sigmoid that introduce non-linearity.
- **Loss Function**: Measures the difference between the network's predictions and actual values.
- **Optimizer**: Algorithm like Adam or SGD that adjusts weights to minimize the loss.

### How
**Usage Instructions**:
1. **Data Preparation**: Collect and preprocess data (normalization, augmentation).
2. **Model Building**: Define the architecture (number of layers, neurons per layer).
3. **Training**: Use a training dataset to adjust weights via backpropagation.
4. **Validation**: Evaluate model performance on a validation set to fine-tune parameters.
5. **Testing**: Test the final model on unseen data to assess its generalization ability.

### When
**Applicable Conditions/Assumptions**:
- Large and high-quality datasets are available.
- Sufficient computational resources (GPUs/TPUs).
- Problems where patterns and structures need to be learned from data.
- Situations requiring scalable and adaptable models.

### Where
**Application Areas**:
- **Natural Language Processing (NLP)**: Chatbots, language translation, sentiment analysis.
- **Computer Vision**: Image recognition, object detection, autonomous vehicles.
- **Healthcare**: Medical imaging analysis, drug discovery.
- **Finance**: Fraud detection, algorithmic trading.
- **Entertainment**: Content recommendation, game AI.

### Pros & Cons
**Advantages**:
- **Accuracy**: High performance on complex tasks with enough data.
- **Adaptability**: Can be applied to various domains with minimal changes.
- **Feature Extraction**: Automatically learns relevant features from raw data.

**Disadvantages**:
- **Data Dependency**: Requires large amounts of labeled data.
- **Computationally Intensive**: Needs significant processing power and time.
- **Black Box**: Lack of interpretability in decision-making processes.
- **Overfitting**: Risk of poor generalization if not properly managed.