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
# Data
2 Types of data pipelines:
- **ETL**: Extract → Transform → Load
- **ELT**: Extract → Load → Transform

This guide focuses on ETL basics.

## Data Collection
Sources:
- **Internal**: Transaction logs, customer databases, sensor data, operational data.
- **External**: Public datasets, third-party data, social media feeds, open-source platforms, user reviews.
- **Synthetic**: Auto-generated data for simulation, used when real data is unavailable due to privacy/rarity.

Procedure:
1. Define requirements.
2. Establish scalable infra.
3. Ensure quality.
4. Continuously monitor.

Methods:
- **Direct**: Surveys, observations, experiments, purchases
    - Tailored for ML tasks, common in R&D.
- **Indirect**: Scraping, database access, APIs
    - Require preprocessing.
- **Crowdsourcing**: Large group data generation
    - Often used for annotations.

Challenges: Volume, variety, veracity, velocity, ethics.

## Data Cleaning
Procedure:
- **Handle missing data**:
  - **Deletion**: When the proportion is negligible.
  - **Imputation**: Applicable to randomly missing data.
- **Remove unwanted samples/features**: Duplicates, irrelevant samples/features, etc.
- **Fix structural errors**: Typos, mislabels, inconsistencies, etc.
- **Filter outliers**: Use domain-specific filters or statistical methods (Z-score, IQR, etc.).
  - Remove for non-robust models, keep for robust models.
- **Handle text data**: Lowercase, punctuation, typos, stopwords, lemmatization, etc.
- **Handle image data**:
  - **Size**: Resizing, cropping, padding, etc.
  - **Color**: Grayscale conversion, histogram equalization, color space transformation, etc.
  - **Noise**: Gaussian blur, median blur, denoising, artifact removal.
  - **File**: Ensure uniform format.

**Challenges**: Scalability, unstructured data, information loss due to over-cleaning, etc.

## Data Imputation
Procedure (like EM):
1. Estimate missing data.
2. Estimate params for imputation.
3. Repeat.

Types:
- **Simple**: Zero, majority, mean (usually best).
  - Assumes no multicollinearity.
- **Complex**:
  - **Regression**: Fit missing features on other features, assumes multicollinearity.
    - Cons: potential assumption failures.
  - **Indicator addition**: Add 0-1 indicators for missing features.
    - Cons: feature size doubles.
  - **Category addition**: Add "missing" category for missing values.
    - Pros: straightforward, better than doubling.
  - **Unsupervised Learning**: Used if many categories/features.

## Data Transformation
### Standardization
$$X_\text{new}=\frac{X-\bar{X}}{\Sigma_X}$$

Pros:
- Removes mean & scales data to unit variance (i.e., $ x_i\sim N(0,1)$).

Cons:
- Sensitive to outliers (because they affect empirical mean & std).
- Destroys sparsity (because center is shifted).

### Normalization
$$X_\text{new}=\frac{X}{\text{norm}(X)}$$

Pros:
- Scales samples to unit norms.
- Supports L1/L2/max norms.

### Min-Max Scaling
$$\begin{align*}
&x\in[0,1]: &&X_\text{new}=\frac{X-\min{(X)}}{\max{(X)}-\min{(X)}}\\
&x\in[\min,\max]: &&X_\text{new}=\frac{X-\min{(X)}}{\max{(X)}-\min{(X)}}(\text{max}-\text{min})+\text{min}
\end{align*}$$

Pros:
- Scales data to a customizable range.

Cons:
- Sensitive to outliers (because they affect empirical min & max).
- Destroys sparsity (because center is shifted).

### Max-Abs Scaling
$$X_\text{new}=\frac{X}{\max{(|X|)}}$$

Pros:
- Preserves signs.
- Preserves sparsity.
- Scales data to $[-1,1]$.

Cons:
- Sensitive to outliers.

### Robust Scaling
$$X_\text{new}=\frac{X-\text{med}(X)}{Q_{75\%}(X)-Q_{25\%}(X)}$$

Pros:
- Robust to outliers

Cons:
- Destroys sparsity (because center is shifted).

### Quantile Transform
- Original: $X_\text{new}=Q^{-1}(F(X))$
    - $Q^{-1}$: Quantile function (i.e., PPF, inverse of CDF).
    - $F$: Empirical CDF.
- Uniform: $X_\text{new}=F_U^{-1}(F(X))\in[0,1]$
- Gaussian: $X_\text{new}=F_N^{-1}(F(X))\sim N(0,1)$

Pros:
- Robust to outliers (by collapsing them).

Cons:
- Distorts linear correlations between diff features.
- Requires large #samples.


### Power Transform
- Yeo-Johnson Transform

    $$
    \mathbf{x}_i^{(\lambda)}=\begin{cases}
    \frac{(\mathbf{x}_i+1)^\lambda-1}{\lambda} & \text{if }\lambda\neq0,\mathbf{x}_i\geq0 \\
    \ln{(\mathbf{x}_i+1)}                      & \text{if }\lambda=0,\mathbf{x}_i\geq0 \\
    \frac{1-(1-\mathbf{x}_i)^{2-\lambda}}{2-\lambda} & \text{if }\lambda\neq2,\mathbf{x}_i<0 \\
    -\ln{(1-\mathbf{x}_i)}                           & \text{if }\lambda=2,\mathbf{x}_i<0
    \end{cases}
    $$

    - $\lambda$: Determined by MLE.

- Box-Cox Transform

    $$
    \mathbf{x}_i^{(\lambda)}=\begin{cases}
    \frac{\mathbf{x}_i^\lambda-1}{\lambda} & \text{if }\lambda\neq0 \\
    \ln{(\mathbf{x}_i)} & \text{if }\lambda=0
    \end{cases}
    $$

    - Requires $\mathbf{x}_i>0$.
    
Pros:
- Maps data to Gaussian distribution (stabilizes variance & minimizes skewness)
- Useful against heteroskedasticity.
- Sklearn's PowerTransformer converts data to $N(0,1)$ by default.

Cons:
- Distorts linear correlations between diff features.

### Categorical features
- **One-Hot Encoding**: Converts each category into a 0-1 feature, better for nominal data.
- **Label Encoding**: Converts each category into a numerical label, better for ordinal data.

## Data Loading
Loading data IRL is more complex than school projects.

Procedure:
1. **Choose Storage**:
    - **Databases**: SQL (relational, structured), NoSQL (unstructured).
    - **Data warehouses**: Ideal for analytical tasks.
    - **Data lakes**: Store raw big data from various sources.
    - **Cloud storage**
2. **Validate**: Check schema, data quality, integrity, etc.
3. **Format**: Ensure proper encoding, batching (for big data), raw saving.
4. **Load**:
    - **Bulk Loading**: Load large data chunks
        - Minimizes logging and transaction overhead.
        - Requires system downtime.
    - **Incremental Loading**: Load data in small increments
        - Uses timestamps/logs to track changes.
        - Minimizes disruption.
        - Ideal for real-time processing.
    - **Streaming**: Load data continuously in real-time.
5. **Optimize**: Reduce data volume for faster execution.
    - **[Indexing](#indexing)**: Use primary/secondary indexes for faster data retrieval. 
        - Best for tables with low data churn.
    - **[Partitioning](#sharding-horizontal-partitioning)**: Divide databases/tables for independent querying.
        - Best for older records.
    - **[Parallel Processing](#parallel-processing)**
6. **Handle Errors**
7. **Ensure Security**: Encryption, access control, etc.
8. **Verify**: Audit with test queries, reconcile loaded data with source data, etc.

### Indexing
What: Create quick lookup paths using B trees/B+ trees for faster data retrieval.

Types:
- **Single-column**: For frequent access/filtering of one column.
- **Composite**: For frequent filtering/sorting based on multiple columns.
- **Unique**: Ensures all index values are unique, used as primary key.
- **Full-text**: For complex queries on unstructured texts.
- **Spatial**: For geospatial operations.

Pros: Fast retrieval, automatic sorting, high time efficiency.

Cons: Low space efficiency, high maintenance complexity.

### Sharding (Horizontal Partitioning)
What: Distribute data across multiple servers/locations using a shard key for horizontal database scaling.

Types:
- **Hash-based**: Even data division using a hash function.
- **Range-based**: Numerical data division using key value ranges.
- **List-based**: Categorical data division using predefined shard key lists.

Uses: Web apps, real-time analytics, game/media services, etc.

Pros: Horizontal scalability, high availability.

Cons: Implementation & maintenance complexity.

### Parallel Processing
Parallel processing runs on a single computer (node).

Concepts:
- **Core**: Independent instruction execution units in a processor.
- **Thread**: Sub-tasks run independently on a core.
- **Memory**: Shared or distributed.

Types:
- **Data Parallelism**: Process data chunks simultaneously on different cores.
- **Task Parallelism**: Execute different tasks in parallel.

Methods:
- **Multithreading**: Use libraries like C/C++ OpenMP, Python threading, etc.
- **GPGPU**: Use CUDA/OpenCL for efficient parallel computing.

### Distributed Computing
Distributed Computing runs on multiple independent computers (nodes).

Concepts:
- **Networks**: Connect multiple computers.
- **Horizontal Scalability**: Performance improves with more machines.
- **Fault Tolerance**: Handle node/network failures without affecting the overall task.

**Methods**:
- **MapReduce**: Process big data with distributed algorithms.
    - **Map**: Filter and sort data.
    - **Reduce**: Summarize data.
- **Distributed Databases**: Store data across multiple locations but appear as a single database.
- **Load Balancing**: Evenly distribute workloads to maximize resource usage and minimize response time.