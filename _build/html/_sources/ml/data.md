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
Data pipelines in ML follow either one below:
- **ETL**: Extract $\rightarrow$ Transform $\rightarrow$ Load
- **ELT**: Extract $\rightarrow$ Load $\rightarrow$ Transform

This page follows ETL and only covers the basics because I'm not a data engineer.

## Data Collection
Sources:
- **Internal**: transaction logs, customer databases, sensor data, operational data, etc.
- **External**: public datasets, third-party purchases, social media feeds, open-source platforms, user reviews, etc.
- **Synthetic**: auto-generated data to simulate real world; used when real data are unavailable due to privacy concerns/rarity

Procedure:
1. Define requirements
2. Establish scalable infra
3. Quality assurance
4. Continuous monitoring

Methods:
- **Direct**: surveys, observations, experiments, purchases, etc.; tailored for ML tasks; commonly used in R&D
- **Indirect**: scraping, database access, APIs, etc.; require preprocessing
- **Crowdsourcing**: gather a large group of people to generate data; commonly used for annotations

Challenges: volume, variety, veracity, velocity, ethics, etc.



## Data Cleaning
Procedure (unordered):
- **Handle missing data**: either at random/not at random
    - **Deletion** (only when the proportion is negligible)
    - [**Imputation**](#imputation) (only applicable to random missing, likely non-factual)
- **Remove unwanted samples/features**: duplicates, irrelevant samples/features, etc.
- **Fix structural errors**: typos, mislabels, inconsistency, etc.
- **Filter unwanted outliers**: statistical methods (Z-score, IQR [interquartile range], etc.), domain-specific filters
    - remove if non-robust model, keep if robust model
- **Handle text data**: lowercase, punctuations, typos, stopwords, lemmatization (reduce word to root form), etc.
- **Handle image data**: 
    - **Size**: resizing, cropping, padding, etc.
    - **Color**: grayscale conversion, histogram equalization (enhance contrast), color space transformation (e.g., RGB $\rightarrow$ HSV)
    - **Noise**: Gaussian blur, median blur, denoising, artifact remmoval (e.g., text overlays, watermarks, etc.)
    - **File**: ensure uniform file format

Challenges: scalability, unstructured data, info loss due to overcleaning, etc.

### Data Imputation
Procedure (like EM): 
1. Estimate the missing data.
2. Estimate the params for imputation.
3. Repeat.

Types:
- Simple imputation: zero, majority, mean (usually best)
    - assume no multicollinearity
- Complex imputation:
    - **Regression**: fit missing feature on other features (assume multicollinearity)
        - NOT necessarily better than simple imputations because assumptions can fail
    - **Indicator addition**: add 0-1 indicators for each feature on whether this feature is missing (0: present; 1: absent)
        - treat the fact "missing" as an informative value
        - Cons: doubled feature size
    - **Category addition**: add one more category called "missing" to represent missing values
        - treat the fact "missing" as an informative value
        - Pros: straightforward & much better than doubled numerical values
    - **Unsupervised Learning**: ussed if there are lots of categories and/or features



## Data Transformation
### Standardization
$$X_\text{new}=\frac{X-\bar{X}}{\Sigma_X}$$

Pros:
- remove mean and/or scale data to unit variance (i.e., $ x_i\sim N(0,1)$)

Cons:
- highly sensitive to outliers (because they greatly impact empirical mean & std)
- destroy sparsity (because center is shifted)

### Normalization
$$X_\text{new}=\frac{X}{\text{norm}(X)}$$

Pros:
- scale individual samples to their unit norms
- can choose l1/l2/max as $\text{norm}(\cdot)$

### Min-Max Scaling
$$\begin{align*}
&x\in[0,1]: &&X_\text{new}=\frac{X-\min{(X)}}{\max{(X)}-\min{(X)}}\\
&x\in[\min,\max]: &&X_\text{new}=\frac{X-\min{(X)}}{\max{(X)}-\min{(X)}}(\text{max}-\text{min})+\text{min}
\end{align*}$$

Pros:
- can scale each $ x_i $ into a range of your choice

Cons:
- highly sensitive to outliers (because they greatly impact empirical min & max)
- destroy sparsity (because center is shifted)

### Max-Abs Scaling
$$X_\text{new}=\frac{X}{\max{(|X|)}}$$

Pros:
- preserve signs of each $ x_i $.- preserve sparsity
- scale each $x_i$  into a range of $[-1,1]$ ($[-1,0)$ for neg entries, $(0,1]$ for pos entries)

Cons:
- highly sensitive to outliers

### Robust Scaling
$$X_\text{new}=\frac{X-\text{med}(X)}{Q_{75\%}(X)-Q_{25\%}(X)}$$

Pros:
- robust to outliers

Cons:
- destroy sparsity (because center is shifted)

### Quantile Transform
- Original form: $X_\text{new}=Q^{-1}(F(X))$
    - $Q^{-1}$: quantile func (i.e., PPF [percent-point func], inverse of CDF)
    - $F$: empirical CDF
- Uniform outputs: $X_\text{new}=F_U^{-1}(F(X))\in[0,1]$
- Gaussian outputs: $X_\text{new}=F_N^{-1}(F(X))\sim N(0,1)$

Pros:
- robust to outliers (literally collapse them)

Cons:
- distort linear correlations between diff features
- only work well with sufficiently large #samples


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
    - $ \lambda $: determined by MLE

- Box-Cox Transform
    $$
    \mathbf{x}_i^{(\lambda)}=\begin{cases}
    \frac{\mathbf{x}_i^\lambda-1}{\lambda} & \text{if }\lambda\neq0 \\
    \ln{(\mathbf{x}_i)} & \text{if }\lambda=0
    \end{cases}
    $$
    - only applicable when $ \mathbf{x}_i>0 $
    
Pros:
- map any data to Gaussian distribution (i.e., stabilize variance and minimize skewness)
- useful against heteroskedasticity (i.e., non-const variance).
- Sklearn's PowerTransformer converts data to $ N(0,1) $ by default.

Cons:
- distort linear correlations between diff features

### Categorical features
- **One-Hot Encoding**: convert each category into a 0-1 feature (excluding a dummy)
    - better for nominal data (i.e., no inherent order among categories)
- **Label Encoding**: convert each category into a numerical label
    - better for ordinal data (i.e., inherent order among categories)



## Data Loading
NOTE: Loading is significantly more complex IRL compared to school projects.

Procedure:
1. Choose storage:
    - **Databases**: SQL/Relational (only structured data), NoSQL (better for unstructured data)
    - **Data Warehouses**: best for analytical tasks
    - **Data Lakes**: raw storage of big data from various sources
    - **Cloud Storage**
2. Validate: check schema, data quality, integrity, etc.
3. Format: encoding, batching (for big data), raw saving
4. Load:
    - **Bulk loading**: load data in large chunks
        - minimize logging & transaction overhead
        - require system downtime
    - **Incremental loading**: load data in small increments
        - use timestamps/logs to track changes
        - minimize system disruption
        - best for real-time data processing
    - **Streaming**: load data continuously in real time
5. Optimize (i.e., reduce data volume for faster execution)
    - [**Indexing**](#indexing): set a primary/secondary index on features with unique/repeated values to retrieve them faster
        - best for tables with low data churn (i.e., with fewer inserts/deletes/updates)
    - [**Partitioning**](#sharding-horizontal-partitioning): divide a database/table into distinct sections/tables to query them independently
        - best for storing older records
    - [**Parallel Processing**](#parallel-processing)
6. Handle errors
7. Handle security: encryption, access control, etc.
8. Verify: audit using test queries, reconcile loaded data with source data, etc.

### Indexing
Why: faster retrieval

What: create quick lookup paths to access the data

How: Each Index stores values of specific columns & the location of corresponding rows. Indices are stored as B trees/B+ trees.
- **B tree**: balanced tree where each node contains multiple keys sorted in order & pointers to child nodes
- **B+ tree**: B tree & all records are stored at leaf level with leaf nodes linked to one another

Types:
- **Single-column Indexes**: created on only one column; used when queries frequently access/filter based on that specific column
- **Composite Indexes** (Multi-column Indexes): created on two/more columns; used when queries frequently filter/sort based on these columns in combination
- **Unique Indexes**: ensure all index values are unique; used as a primary key
- **Full-text Indexes**: allow complex queries on unstructured texts (e.g., search engines, document search)
- **Spatial Indexes**: used when queries do geospatial operations

Pros:
- fast retrieval
- auto sort (useful for fetching & presenting data)
- high time efficiency (reduce #disk accesses required during queries)

Cons:
- low space efficiency (require additional space to store indices)
- high maintenance complexity

### Sharding (horizontal partitioning)
Why: scale databases horizontally

What: distribute data across multiple servers/locations, where each shard is an independent database and all shards form a single logical database

How: use a specific (set of) columns as a **shard key** that determines how the data is distributed across the shards
- **Hash-based**: use a hash function to determine which shard will store a given data row; useful for even data division
- **Range-based**: use ranges of shard key values to divide data; useful for numerical data division
- **List-based**: use predefined lists of shard keys to divide; useful for categorical data division

Where: web apps, real-time analytics, game/media services, etc.

Pros:
- horizontal scalability
- high availability

Cons:
- implementation complexity
- high maintenance complexity

### Parallel Processing
Concepts:
- **Core**: A processor has multiple cores, each of which can execute instructions independently.
- **Thread**: A core can run multiple threads, each of which can execute sub-tasks independently.
- **Memory**:
    - Shared memory architectures allow multiple cores to access the same memory space.
    - Distributed memory architectures require communication between cores.

Types:
- **Data Parallelism**: divide data into smaller chunks & processing each chunk simultaneously on different cores
- **Task Parallelism**: execute different tasks of one program in parallel

Methods:
- **Multithreading**: C/C++ OpenMP, Python threading libraries, etc.
- **GPGPU** (General-Purpose computing on Graphics Processing Units): use CUDA and OpenCL to perform highly parallelized computing more efficiently than CPUs

### Distributed Computing
NOTE: Distributed Computing $\neq$ Parallel Processing
- Parallel processing runs on a single computer.
- Distributed computing runs on multiple independent computers (i.e., nodes).

Concepts:
- **Networks**: to connect multiple computers
- **Horizontal scalability**: the more machines the better performance (unless beyond capacity)
- **Fault Tolerance**: handle failures of individual nodes / network components without affecting the overall task

Methods:
- **MapReduce**: process big data with a distributed algorithm on a cluster
    - Map: filter & sort data
    - Reduce: perform a summary operation
- **Distributed Databases**: store data across multiple physical locations BUT appear as a single database to users
- **Load Balancing**: distribute workloads evenly across all nodes to maximize resource usage & minimize response time