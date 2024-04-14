---
title : "Introduction"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: true
images: []
weight: 100
---
Just some exam notes for UPenn's [CIS 545 Big Data Analytics](https://sites.google.com/seas.upenn.edu/cis545).

**Big Data**: high-dimensional, hard to understand, requires computation & I/O costs
</br>
</br>

**Data Preprocessing**: extract key parts of data {{< math >}}$ \rightarrow$ model & annotate data $\rightarrow$ clean data $\rightarrow ${{</ math>}} link & coregister data
</br>
</br>

**Data Analytics**: data {{< math >}}$ \rightarrow ${{</ math>}} knowledge/action
- Goals:
    - Pattern detection: data {{< math >}}$ \rightarrow$ patterns $\rightarrow ${{</ math>}} partial understanding (i.e., descriptive statistics)
    - Hypothesis: experiment over sample {{< math >}}$ \rightarrow ${{</ math>}} significance (i.e., inferential statistics)
- Aspects:
    - Acquisition & Access (data may not be accessible)
    - Wrangling (data may be in the wrong form)
    - Integration & Representation (relationships may not be captured)
    - Cleaning & Filtering (data may have variable quality)
    - Hypothesizing, Querying, Analyzing, Modeling (from data to info)
    - Understanding, Iterating, Exploring (build knowledge)
    - Ethical obligations
- Reality:
    - heavily rely on **human expertise** (domain knowledge) to impose models over features
    - spent TONS of time on data understanding, cleaning, wrangling.
- Process: question {{< math >}}$ \rightarrow$ scope $\rightarrow$ data $\rightarrow$ techniques $\rightarrow$ evaluation $\rightarrow ${{</ math>}} maintenance
</br>
</br>
</br>
</br>

# Data Acquisition & Wrangling
**Structured Data**
- **Instances**: observations/samples from a population
- **Features**: measurable properties/characteristics
- Transformation: raw data {{< math >}}$ \rightarrow ${{</ math>}} single tabular representation
- Basic encodings:
    - **Tables**: **tuples** that may have different field types
        - dataframe in Pandas; relations in SQL
    - **Arrays**: cells of same type in a multi-dimensional coordinate space
        - matrices or tensors as well
    - **Nested**: compositions of **maps** and **lists**
        - JSON, XML, etc.
- Wrangling:
    - Web & Text data: pattern-based extraction
    - Database: API
- Acquisition:
    - **Structured files** (i.e., CSV)
        - In dataframe, column names may be assigned based on file header
        - When storing dataframe to SQL, we have an option of saving/dropping the dataframe index
        - **Projection**: return a subset of columns in a dataframe
    - HTML tables & info
        - **DOM** (Document Object Model): HTML/XML are **hierarchical** (tree) data, with tags/content nested within tags
        - **XPath**: a template for matching against trees based on nesting structure and data values
            - simplifies matching in DOM
            - corresponds to pathnames in Unix
            - describes a series of path steps
            - returns ordered sets of nodes (as Python lists)
    - **DBMSs** (Database Management Systems)
        - Functions: data storage (typically tabular), data updates, data queries & transformations
        - Components:
            - an active connection to remote machine
            - a request to fetch data from table/via query
        - Pros: faster than CSV/HTML, more stable because links can break
</br>
</br>
</br>
</br>

# Data Transformation

Pandas vs SQL
- Pandas: direct control, not persistent
- SQL: automatically optimized, persistent
- Apache Spark: a combination of both

## Column-wise operations

2 forms of projection: double brackets return dataframe, single bracket returns 1 column as series.

3 forms of projected column data:
- List: no schema or common value type, 0-indexed
- Series: common value type, special indices, use series funcs
- 1-column dataframe: schema + common value type, special indices, use df funcs

Tip: It's best NOT to iterate over the elements and modify them. Call apply with a function (manual/lambda) instead.
- Iterating over items in a loop forces a sequence, while "projection {{< math >}}$ \rightarrow ${{</ math>}} apply" is possible in parallel

Tip: apply()
- axis=0: per column
- axis=1: per row

Cons of dataframe: must fit in memory
- When the data is too large or stored in database, use SQL. Much easier.
</br>
</br>

## Filtering rows
Columns vs Rows: column = property; row = instance

Select items:
- Define a filter mask (bool series)
- Use it to request a subset of rows
- Project a column 

Tip: in SQL, `!=` is `<>`.

Basic Operations over single tables
- Projection: pull out a slice of the table on certain columns
- Selection: pull out rows with matching conditions
- apply(): compute/modify over columns
- Inplace updates for dataframes
</br>
</br>

## Joining tables
Join: match rows with same values

Inner vs Outer: Innerjoin is on equality (i.e., exact match) only; Outerjoin will include “partial” rows when one side (e.g., the left) doesn’t have a match on the other side (e.g., the right)
- An outerjoin will return the exact same results as an innerjoin if every tuple in the input relations has a match.
- If we join relations that have columns of the same names, Pandas will rename the input columns by appending unique suffixes.
- Pandas does Innerjoin by default (but can be modified); SQL can specify a more general condition for the join
- We can compose joins to link multiple tables
</br>
</br>
</br>
</br>

# Data Integration
**Validation**: detecting when there is bad data
- It's best to not count on humans to look at data at scale. We want pipelines that run periodically.
- We have standard validation approaches and tools, but beware of exceptions.
- We nearly always have FPs and FNs.
- Need to log and periodically inspect the data that violates rules.

**Cleaning**: transforming data if it's dirty
- Requires domain expertise and can bias the data


Validation & Cleaning is our best effort. Lots of data errors will be impossible to spot. Even knowing data is bad or incomplete doesn't always suggest a fix.

## General Techniques for Data Validation
**Validation libraries**: rely on validation rules for stuff like IP address, URL, email, etc. (e.g., Python validator library)
- Lots of domain-specific rules can be captured, but they only look at patterns and are vulnerable to invalid values that look plausible.

**Validation against a Master List**: check if value belongs to a list of all known values
- It captures all possible domain values rather than patterns
</br>
</br>

## Record Linking
**Record linking**: Given {{< math >}}$ r,s$ from tables $R,S$, join $r,s ${{</ math>}} if there is a semantic link between them.
- look at rows in different tables and figure out if they should join (i.e., join with tolerance via similarity check)

**Deduplication**: Given {{< math >}}$ t_1,t_2$ in table $T$, join $t_1,t_2 ${{</ math>}} if they represent the same instance.

How to determine that 2 tuples "represent the same instance" or "are semantically linked"?
- Similarity measures
- e.g., **string similarity**
    - **Edit distance**: #edits needed to convert one string into another (requires DP)
    - **Overlap**: measure how much is common between 2 strings
        - **q-grams**: take all substrings of length {{< math >}}$ q ${{</ math>}}
        - **Jaccard similarity**: {{< math >}}$ \frac{|qgram(x)\cap qgram(y)|}{|qgram(x)\cup qgram(y)|} ${{</ math>}}
</br>
</br>

## Automated Data Wrangling
Wrangling vs Validation & Cleaning
- Wrangling gets the data into structured form
- Validation & Cleaning fixes errors in the structured data
- Both can be automated in workflows

**ETL** (Extract-Transform-Load): a sequence of data-oriented modules that are executed in a script which perform ETL tasks (in Pandas/SQL/Perl/Call to Web/arbitrary code)
- ETL is a special case of **workflow**: some notion of scripts which get **triggered** by some event, takes **params**, executes, and produces output
</br>
</br>
</br>
</br>

# Data Analysis (simple)
**Grouping**: bins subsets of rows based on common values
- can apply aggregation functions to the columns
- can plot groupby aggregations easily

## Text Data
Modern NLP: predict words given context
- **Document vectors**: to answer a question, find docs with matching words by encoding each doc into a vector which counts the #occurrences of each word (and do the same to query as well)
    - Ways to count occurrences
        - **BoW** (Bags of Words): unordered set of all words (i.e., lexicon)
        - **n-grams**: replace individual words with {{< math >}}$ n ${{</ math>}}-word groups in a vector
        - **TF-IDF**: downweight words based on how commonly they occur in some dataset (useful for killing stopwords)
    - Docs and queries with overlapping terms will have similar vectors (measurable via cosine similarity)
    - Cons:
        - lose word senses (context)
        - lose word order (e.g., subject vs object)
        - lose word significance (e.g., stopwords are useless)
    - **Distributional Hypothesis**: words that are used and occur in the same contexts tend to purport similar meanings.
    - **Term-Frequency matrix**: count how frequently each word term (row) appears in each doc (col)
        - Similar/related words keep co-occurring in docs.
        - Pros: capture how often term appears for different topics, catch synonyms
        - Cons: topics are too coarse-grained, {{< math >}}$ O(mn)$ is huge ($n$ words times $m ${{</ math>}} docs)
    - **Term-Term matrix**: count how frequently a word (row) appears within the neighborhood context of another word (col)
        - Similar/related words co-occur with the similar sets of words in sentences.
        - Pros: can estimate semantic similarity by vector similarity
        - Cons: {{< math >}}$ O(n^2)$ is huge ($n$ words times $n ${{</ math>}} words)
- **Word2Vec**: learn an embedding for each word
    - **Embedding**: a vector that predicts the probability distribution of (embeddings of) next word given context (consisting of prior word embeddings)
    - Pros: compression
</br>
</br>

Building blocks for NLP: vector databases and chains
- Simply predicting next words can do tons of tasks over knowledge and input
- Neutral sentences don't necessarily have neutral sentiments
- SpaCy: NER is done by transformers simultaneously with POS tagging
- Entity Resolution is hard
    - Approximate string match can be ambiguous or misleading when there are abbreviations and ambiguous names (e.g., product models, city names, etc.)
- Relation Extraction
    - Template-based IE: write/learn templates to extract entities and relations between them.
    - LLM prompting
- Information Extraction is hard: language has complexities beyond pulling things from docs
    - Synonyms and orthonyms
    - Complex structure leads to ambiguity
    - LLMs aren't always robust to out-of-domain text and generally won't know orthonyms.
    - IE depends on redundant info in text to address the issue that it has low recall.
</br>
</br>
</br>
</br>

# DS Ethics
De-identification is not enough to protect user privacy
- Even if data is de-identified, entries can be correlated with entries in other datasets to make informed guesses as to identity. This is mainly caused by **data sparsity**.
- 87% of the U.S. population can be identified using a combination of their gender, birthdate and zip code.

**Differential privacy** aims to maximize the accuracy of queries from statistical databases while minimizing the chances of identifying its records, by adding noise and providing guarantees against a "privacy budget".
</br>
</br>
</br>
</br>

# Data Representation
2 (interchangeable) ways to model relationships
- logical constraints as SQL queries
- paths in a graph

**ER graphs**: codify entity sets, relationship sets, properties, instances.
- **Entity set**: represents all of the entities of a type, and their properties
- **Relationship set**: represents a link between people

**IS-A**: Subclass inherits all properties of superclass; Superclass includes all members of subclasses.

Relationial data can not only capture flat relationships but also represent graphs, which can be traversed by **join** queries.

**NoSQL vs SQL** (Hierarchy vs Relations): NoSQL databases store trees (**nested objects**).
- Well-known NoSQL systems:
    - MongoDB: stores JSON
    - Google Bigtable: stores tuples with irregular properties
    - Amazon S3: stores binary files by key
- Pros:
    - Querying is far simpler than SQL (no joins)
        - JSON may be faster to access from MongoDB than df querying because we can query for individual JSON trees.
- Cons:
    - Support limited notions of **consistency** when update
        - Sometimes we need to allow for failures and "undo". Relational DBMS typically provide atomic transactions for this, but not NoSQL DBMS.
        - **Concurrency control**: how do we handle concurrent updates?
        - **Consistency**: when do I see changes?
    - Doesn't work well with data visualization or ML
- NoSQL {{< math >}}$ \rightarrow ${{</ math>}} SQL:
    - Split hierarchical data into tables when data has multiple values per parent item (e.g., an array as value, then it is no longer 1:1).
    - Nesting becomes links (foreign key)
        - **Foreign key**: takes on a value from a key in another table
    - Use left outerjoins to reassemble hierarchy because there may be parent items with no children.

SQL Views: give the results of a query a name and make it a table
```sql
CREATE VIEW table_name AS SELECT ...
```
</br>
</br>
</br>
</br>

# Efficient Data Processing

## Computer Architecture
Takeaways:
- Different program instructions take different amount of work and time.
- Computer's memory hierarchy means accessing data is not uniform.
- Modern hardware includes special optimizations to
    - do the same operation on multiple data items simultaneously.
    - run multiple independent pieces of code at the same time.
- A typical machine can run billions instructions in 1 sec.

Memory hierarchy:
- Cache amortizes expensive memory fetches across multiple requests
- Accessing data in predictable ways is better
    - CPU predicts and "pre-fetches" the data into caches
    - Repeated requests to related data keep memory in the caches
- Smaller "memory footprints" are better
- Packing multiple data values in the same memory and applying repeated computation is

Take advantage of memory hierarchy:
- Focus on data reduction: filter rows & cols
- Put the most frequently used items together
- Find ways to pass over data once, instead of many times.
</br>
</br>

## Relational Expression Optimization
Why are simple operations slow?
- Scanning through memory
- Parsing strings
- Updating list data
- Converting to DF

Reduce data as soon as possible (push down operations)
- **Select**/Filter at the earliest point possible
- **Project** once columns aren't needed (data better fits into CPU cache)
- SQL does this automatically
- This reduces both memory footprint and #operations

**Indexing** speeds selection
- Index: a map from index key to set of values
    - allows direct find matches to the key without scanning the data ({{< math >}}$ O(1) ${{</ math>}})
    - can be in-memory or on disk
- 2 types of indices
    - **tree** (B+ trees): find all values </>/= key
    - **hash**: look up all values = key
</br>
</br>

## Join Optimization
Takeaways:
- Joins are expensive ({{< math >}}$ O(n^2) ${{</ math>}})
- Cleverly ordering our joins to **reduce intermediate result** sizes makes a huge performance difference
- Eval order matters because intermediate result size matters. Pandas needs ballpark estimate to determine order, while SQL can choose this automatically for us.
</br>
</br>

## Algorithmic Efficiency
Change how we access data:
- Maps/In-memory indices (much faster than exact-match pandas merge)
    - If use this to join 2 tables S&T, {{< math >}}$ O($#rows(S)+#rows(T)$) ${{</ math>}}
    - cardinality = #rows
- Buffering/Blocks
    - When tables are bigger than memory, this reduce overall #disk fetches.

For big data, we will need to supply our own operations
- Functions to be called via apply/applymap
- Functions that take collections of data
</br>
</br>

## Multiprocessing
How to handle large volumes of data?
- reduce #data
- use more efficient data structures/algorithms
- split work among multi processors/workers

**Bulk operations**: do the same thing over many values (i.e., on different rows in parallel)

**Data parallelism**: same instructions, multi data items

### Multicore processing (Dask)
Takeaways: 
- Dask processes data items independently on different cores by partitioning df by row.
- For some (not all) relational operations, we can conceptually get speedups up to min(#cores, #rows).
- Many DBMSs like PostgreSQL can handle parallel processing, but big data cannot be handled on a single computer (limited by I/O).
</br>
</br>

### Cluster computing
Takeaways:
- Today we build parallel computation by linking together many compute nodes, each with its own local CPU cache, memory, and disk
- Communication costs vary dramatically at different levels
    - Internal CPU state (registers, cache)
    - Memory (latency 1000 times slower)
    - SSD (latency 1000+ times slower)
- **Cloud**: data centers run by commercial providers
    - **Hybrid cloud**: cloud + local clusters administered the same way
- Data center is necessary when we don't have enough power and cooling for a machine room.
- Clusters and data centers cannot be programmed byst using Python and Pandas.
</br>
</br>

### Data processing in cluster
2 issues:
- make computation parallel
    - let data reside on multiple compute nodes
    - a coordinator tells multiple nodes on the cluster when and which relational algebra operators to apply
- minimize communication and coordination

**Horizontal partitioning**: randomly put diff rows on diff machines
- Problem: groupby doesn't work.

**Sharding**: map key range to machine ID, send all tuples of that key range to the machine

**Hashing**: take a value (a hash key) and return a large int.
- {{< math >}}$ \forall k_1,k_2: k_1=k_2\Rightarrow h(k_1)=h(k_2) ${{</ math>}}
- {{< math >}}$ \forall k_1,k_2: k_1\neq k_2\Rightarrow P(h(k_1)\neq h(k_2)) ${{</ math>}} is high
- e.g., we typically put data for key {{< math >}}$ k$ at node $h(k) \% n ${{</ math>}}
</br>
</br>

### Cluster-based processing (Spark)
Spark: a platform for sharded big data
- Spark DFs have a **typed schema**: it cannot determine on-the-fly whether input fields are strings vs integers like Python does.
- Spark supports both Pandas and SQL style operations.
- Given a cluster with {{< math >}}$ n$ workers, Spark creates a table with at least $n ${{</ math>}} partitions.
- Selection + Projection + Apply is farmed out to each worker, run simultaneously.
- Grouping typically requires the machine to exchange/repartition data.
- Failure:
    - If a large cluster runs for a long time, machines may die or software may crash
    - Spark actually handles such failures transparently
        - It periodically checkpoints or snapshots what has happened
        - If a node dies, it can restart the computation elsewhere.
</br>
</br>
</br>
</br>

# Big Data and Cloud Services
5 Vs of big data:
- **value**
- **veracity** (high quality)
- **variety** (heterogeneous)
- **volume** (many rows, dimensions, large data objects)
- **velocity** (frequent changes in data)

Taxonomy for Cloud Service Layers
- **Software as a Service** (SaaS): applications hosted on the cloud
- **Platform as a Service** (PaaS): libraries, specialized platforms (e.g., Spark)
- **Infra as a Service** (IaaS): raw machines & storage
- e.g., Colab & SageMaker are cloud-hosted Paas/SaaS hybrid.

To install Python packages across clusters, set up a shell script as a **bootstrap** action.

**Distributed Joins**: shard one df by ID and the other by the joined-on col; add exchange/repartition/shuffle operators if necessary.
- Every join/groupby needs the data to be sharded on the key. if it isn't, we need exchange/repartition (which we want to minimize). (amortize repartitions across multiple operations if possible)
- If a worker fails in execution, its work is re-executed.

**Query Optimizer**: Spark's QO seeks to estimate how big input sources are, how many results will be produced in each filter/join/groupby, compare diff orderings of operations, and find the strategy that minimizes overall cost, including repartitions and join costs.

Spark queries are lazy to maximize optimizaton
- Upon show()/save(), queries are combined, plan is generated, which minimizes cost.

## Data Storage on Cloud
Key questions:
- How complex and large is the data and its content?
- How will I query my data?
- Do I need transactions?

Amazon S3 supports **buckets**: virtual disk volumes (for videos, images, large parquet files, large CSVs)

DynamoDB/BigTable supports small object lookup
- objects in a map from keys to hierarchical values (e.g., JSON, dicts)
- queries largely limited to lookups by key

RDS: Relational DBMSs are best if we want:
- complex queries that return data subsets to Spark
- atomic updates across tables in transactions

Both DynamoDB and RDS are good for ID-based retrieval.
</br>
</br>

## Materialization of Query Results
**View materialization**: strategically store redundant info for generating analysis results.
- used for
    - commonly used subqueries
    - generated reports/hierarchical data
    - recursive computations

If we use inheritance in an E-R diagram, the tables are naturally partitioned such that instances show up in parent and child tables, but columns other than ID are split.

Takeaways:
- View materialization sacrifices storage (and update costs) for query performance
- Can be done by saving a result directly or df.persist()
- Very commonly used in big data scenarios
</br>
</br>
</br>
</br>

# Graphs and Big Data
Terminology:
- **Triangle**: 3 vertices that are pariwsie adjacent
- **Clique**: any set of vertices that are pairwise adjacent (generalize triangle)
- **Edge list**: represent graph in df

**Network Centrality**: measure importance
- Degree centrality: for a node, #nodes it is connected to
- Betweenness centrality: for a node, #shortest paths go through the node
- Eigenvector centrality: PageRank (recursive measure)

Exploration:
- BFS: requires access to a global queue, and is inherently sequential.
- Distributed BFS: In a Spark-based iterative approach to BFS, we traverse edges one hop at a time via a join with edges_df.
    - Sometimes we need to repartition one of the dataframes because of no sharding.
    - We may want to keep info about distance, path, etc.
    - We may want to prune all non-minimal path.
- Triadic closure: add edges to complete most triangles

Link Analysis: 
- Link Analysis for the web defines a node's influence in terms of influence of a node's neighbors.
- The links that matter for ranking are considered to be those between sites.

PageRank: recursive measure of importance
- Intuition:
    - Initially, each page has 1 vote.
    - Each page votes for all the pages it has an out link to.
    - Pages voting for more than 1 page must split their vote equally between them.
    - Voting proceeds in rounds. In each round, each page has #votes it received in the previous round.
- Simplified version:
    - Each page {{< math >}}$ x$ is given a rank PageRank($x ${{</ math>}})
    - Goal: assign PageRank({{< math >}}$ x ${{</ math>}}) s.t. rank of each page is governed by the ranks of the pages linked to it.
    - PageRank({{< math >}}$ x$) $=\sum_{j\in B(x)}\frac{1}{N_j}$ PageRank($j$), where $N_j=$ #links out from page $j ${{</ math>}}.
- 2 properties:
    - It converges
    - It can be computed independently of the query
- Caveat: query independence means it only looks at **structure** (no semantics).
- Implementation:
    - Initialize all ranks: {{< math >}}$ PageRank^{(0)}(x)=\frac{1}{|V|} ${{</ math>}}
    - Iterate until convergence: {{< math >}}$ PageRank^{(i)}(x)=\sum_{j\in B(x)}\frac{1}{N_j}PageRank^{(i-1)}(j) ${{</ math>}}
- Explanation:
    - Initialize all ranks
    - Propagate weights across out-edges
    - Compute weights based on in-edges
- Can use recursive join computations in Spark for this.
- Implementation (Linear Algebra ver.):
    - {{< math >}}$ PageRank^{(i)}=M\cdot PageRank^{(i-1)} ${{</ math>}}
    - {{< math >}}$ M_{ij}=\frac{1}{N_j}$ if page $i$ is pointed by page $j$, and page $j$ has $N_j ${{</ math>}} outgoing links, or 0.
    - {{< math >}}$ PageRank=[PageRank(p_1);\cdots;PageRank(p_m)] ${{</ math>}}
    - Computes principal eigenvector via power iteration
- Random Surfer Model: reduce rank hogs and dead ends (where rank eventually becomes 0)
    - Remove out-degree 0 nodes
    - Add damping/decay factor {{< math >}}$ \alpha ${{</ math>}} to deal with sinks
        - {{< math >}}$ PageRank^{(i)}=\alpha M\cdot PageRank^{(i-1)}+\beta ${{</ math>}}
        - typical values: {{< math >}}$ \alpha=0.85, \beta=1-\alpha$ is a $m\times 1 ${{</ math>}} vector.
    - Intuition: with probability {{< math >}}$ \alpha$, clicks on a random outlink; with probability $\beta ${{</ math>}}, jumps to a random page
- Personalize PageRank: label propagation starts at labeled nodes, estimates how often we end up at a destination if we randomly walked from each labeled node.