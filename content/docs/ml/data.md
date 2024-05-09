---
title : "Data"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: false
images: []
weight: 2
---
Data pipelines in ML follow either one below:
- **ETL**: Extract {{<math>}}$\rightarrow${{</math>}} Transform {{<math>}}$\rightarrow${{</math>}} Load
- **ELT**: Extract {{<math>}}$\rightarrow${{</math>}} Load {{<math>}}$\rightarrow${{</math>}} Transform

This page follows ETL and only covers the basics because I'm not a data engineer.

# Data Collection
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

&nbsp;

# Data Cleaning
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
    - **Color**: grayscale conversion, histogram equalization (enhance contrast), color space transformation (e.g., RGB {{<math>}}$\rightarrow${{</math>}} HSV)
    - **Noise**: Gaussian blur, median blur, denoising, artifact remmoval (e.g., text overlays, watermarks, etc.)
    - **File**: ensure uniform file format

Challenges: scalability, unstructured data, info loss due to overcleaning, etc.

## Data Imputation
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

&nbsp;

# Data Transformation
## Standardization
{{<math class=text-center >}}$$X_\text{new}=\frac{X-\bar{X}}{\Sigma_X}$${{</math>}}

Pros:
- remove mean and/or scale data to unit variance (i.e., {{<math>}}$ x_i\sim N(0,1)${{</math>}})

Cons:
- highly sensitive to outliers (because they greatly impact empirical mean & std)
- destroy sparsity (because center is shifted)

## Normalization
{{<math class=text-center >}}$$X_\text{new}=\frac{X}{\text{norm}(X)}$${{</math>}}

Pros:
- scale individual samples to their unit norms
- can choose l1/l2/max as {{<math>}}$\text{norm}(\cdot)${{</math>}}

## Min-Max Scaling
{{< math class=text-center >}}$$\begin{align*}
&x\in[0,1]: &&X_\text{new}=\frac{X-\min{(X)}}{\max{(X)}-\min{(X)}}\\
&x\in[\min,\max]: &&X_\text{new}=\frac{X-\min{(X)}}{\max{(X)}-\min{(X)}}(\text{max}-\text{min})+\text{min}
\end{align*}$${{</math>}}

Pros:
- can scale each {{<math>}}$ x_i ${{</math>}} into a range of your choice

Cons:
- highly sensitive to outliers (because they greatly impact empirical min & max)
- destroy sparsity (because center is shifted)

## Max-Abs Scaling
{{<math class=text-center >}}$$X_\text{new}=\frac{X}{\max{(|X|)}}$${{</math>}}

Pros:
- preserve signs of each {{<math>}}$ x_i ${{</math>}}.- preserve sparsity
- scale each {{<math>}}$x_i${{</math>}}  into a range of {{<math>}}$[-1,1]${{</math>}} ({{<math>}}$[-1,0)${{</math>}} for neg entries, {{<math>}}$(0,1]${{</math>}} for pos entries)

Cons:
- highly sensitive to outliers

## Robust Scaling
{{<math class=text-center >}}$$X_\text{new}=\frac{X-\text{med}(X)}{Q_{75\%}(X)-Q_{25\%}(X)}$${{</math>}}

Pros:
- robust to outliers

Cons:
- destroy sparsity (because center is shifted)

## Quantile Transform
- Original form: {{<math>}}$X_\text{new}=Q^{-1}(F(X))${{</math>}}
    - {{<math>}}$Q^{-1}${{</math>}}: quantile func (i.e., PPF [percent-point func], inverse of CDF)
    - {{<math>}}$F${{</math>}}: empirical CDF
- Uniform outputs: {{<math>}}$X_\text{new}=F_U^{-1}(F(X))\in[0,1]${{</math>}}
- Gaussian outputs: {{<math>}}$X_\text{new}=F_N^{-1}(F(X))\sim N(0,1)${{</math>}}

Pros:
- robust to outliers (literally collapse them)

Cons:
- distort linear correlations between diff features
- only work well with sufficiently large #samples


## Power Transform
- Yeo-Johnson Transform
    {{<math class=text-center >}}$$
    \mathbf{x}_i^{(\lambda)}=\begin{cases}
    \frac{(\mathbf{x}_i+1)^\lambda-1}{\lambda} & \text{if }\lambda\neq0,\mathbf{x}_i\geq0 \\
    \ln{(\mathbf{x}_i+1)}                      & \text{if }\lambda=0,\mathbf{x}_i\geq0 \\
    \frac{1-(1-\mathbf{x}_i)^{2-\lambda}}{2-\lambda} & \text{if }\lambda\neq2,\mathbf{x}_i<0 \\
    -\ln{(1-\mathbf{x}_i)}                           & \text{if }\lambda=2,\mathbf{x}_i<0
    \end{cases}
    $${{</math>}}
    - {{<math>}}$ \lambda ${{</math>}}: determined by MLE

- Box-Cox Transform
    {{<math class=text-center >}}$$
    \mathbf{x}_i^{(\lambda)}=\begin{cases}
    \frac{\mathbf{x}_i^\lambda-1}{\lambda} & \text{if }\lambda\neq0 \\
    \ln{(\mathbf{x}_i)} & \text{if }\lambda=0
    \end{cases}
    $${{</math>}}
    - only applicable when {{<math>}}$ \mathbf{x}_i>0 ${{</math>}}
    
Pros:
- map any data to Gaussian distribution (i.e., stabilize variance and minimize skewness)
- useful against heteroskedasticity (i.e., non-const variance).
- Sklearn's PowerTransformer converts data to {{<math>}}$ N(0,1) ${{</math>}} by default.

Cons:
- distort linear correlations between diff features

## Categorical features
- **One-Hot Encoding**: convert each category into a 0-1 feature (excluding a dummy)
    - better for nominal data (i.e., no inherent order among categories)
- **Label Encoding**: convert each category into a numerical label
    - better for ordinal data (i.e., inherent order among categories)

&nbsp;

# Data Loading
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

## Indexing
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

## Sharding (horizontal partitioning)
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

## Parallel Processing
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

## Distributed Computing
NOTE: Distributed Computing {{<math>}}$\neq${{</math>}} Parallel Processing
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