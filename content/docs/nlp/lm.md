---
title : "Language Models"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: false
images: []
weight: 22
---
For consistency, notations strictly follow the original papers.

# PLMs
This section includes basic transformer-based LM structures that are still widely used when LLMs are unavailable.

## GPT
Ref: [Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

Ideas:
- **transformer decoders** only
- **unsupervised pretraining** on a diverse unlabeled corpus
- **supervised finetuning** on each specific task
- (in practice) [EOS] token as separator

Unsupervised pretraining:
- Objective: Autoregressive MLE on tokens
    $$
    L_1(\mathcal{U})=\sum_i\log P(u_i|u_{i-k},\cdots,u_{i-1};\Theta)
    $$
    - {{< math >}}$ \mathcal{U}=\\{u_1,\cdots,u_n\\} ${{</ math>}}: unlabeled corpus
    - {{< math >}}$ k ${{</ math>}}: context window size
    - {{< math >}}$ \Theta ${{</ math>}}: param set
- Structure: Transformer decoders
    {{< math class=text-center >}}$$\begin{align*}
    &h_0=UW_e+W_p \\\\
    &h_l=\text{decoder}(h_{l-1})\ \ \forall i\in[1,n] \\\\
    &P(u)=\text{softmax}(h_nW_e^T)
    \end{align*}$${{< /math >}}
    - {{< math >}}$ W_e ${{</ math>}}: token embedding matrix
    - {{< math >}}$ W_p ${{</ math>}}: position embedding matrix
    - {{< math >}}$ U=(u_{-k},\cdots,u_{-1}) ${{</ math>}}: context vector of tokens
    - {{< math >}}$ n ${{</ math>}}: #layers

&nbsp;

Supervised finetuning:
- Objective (basic): depends on the specific task, generally MLE if discriminative
    $$
    L_2(\mathcal{C})=\sum_{(x,y)}\log P(y|x^1,\cdots,x^m)
    $$
    - {{< math >}}$ \mathcal{C} ${{</ math>}}: labeled dataset
    - {{< math >}}$ {x^1,\cdots,x^m} ${{</ math>}}: seq of input tokens
    - {{< math >}}$ y ${{</ math>}}: label
- Objective (hybrid): include LM as auxiliary objective
    $$
    L_3(\mathcal{C})=L_2(\mathcal{C})+\lambda L_1(\mathcal{C})
    $$
    - {{< math >}}$ \lambda ${{</ math>}}: weight
    - Pros:
        - improve generalization of supervised model
        - accelerate convergence

&nbsp;

### GPT-2
Ref: [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

Ideas:
- Larger (#params: 1.5B > 117M)
- More diverse data
- **Task-agnostic**: learn supervised downstream tasks without explicit supervision
- **Zero-shot** capability

&nbsp;

### GPT-3
Ref: [GPT-3](https://arxiv.org/pdf/2005.14165.pdf)

Ideas:
- Even larger (#params: 175B > 1.5B)
- Even more diverse data
- Even more task-agnostic
- **Few-shot** capability
- Prompting

&nbsp;

## BERT
Ref: [Bidirectional Encoder Representations from Transformers](https://arxiv.org/pdf/1810.04805.pdf)

Ideas:
- **transformer encoders** only
- **bidirectional context** in pretraining
- (in practice) [CLS] token first, [SEP] token as separator

Input representation:
<center>
<img src="/images/nlp/bert.png" width="600"/>
</center>

&nbsp;

Unsupervised pretraining:
- **Masked Language Modeling** (MLM): randomly mask some words (15% in original experiment) and predict them.
    - Problem: [MASK] token does not exist in downstream tasks
    - Solution: further randomness - if a token is chosen to be masked, replace with
        - [MASK] 80% of the time
        - random token 10% of the time
        - itself 10% of the time
- **Next Sentence Prediction** (NSP): predict whether sentence {{< math >}}$ B$ is the next sentence of sentence $A ${{</ math>}}, in order to understand relationships between sentences.
    - When selecting training samples,
        - 50% of the time {{< math >}}$ B$ is actually $A ${{</ math>}}'s next sentence, labeled as `IsNext`.
        - 50% of the time {{< math >}}$ B$ is not $A ${{</ math>}}'s next sentence, labeled as `NotNext`.
        - Predict on [CLS] head.

&nbsp;

### RoBERTa
Ref: [Robustly Optimized BERT Approach](https://arxiv.org/pdf/1907.11692.pdf)

Ideas:
- Enhanced BERT pretraining:
    - **Dynamic masking**: generate masking when feeding a seq to the model, instead of using same masking from data processing.
    - **Remove NSP**: remove NSP loss + use `FULL-SENTENCES` (packing seqs from multiple docs)
    - **Large mini-batches**: 


&nbsp;

&nbsp;

# LLMs
This section includes the widely used LLMs on the market.

## GPT-3.5

## GPT-4

## Google

### LaMDA

### PaLM

## Meta

### OPT

### LLaMA