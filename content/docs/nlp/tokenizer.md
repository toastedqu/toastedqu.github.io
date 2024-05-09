---
title : "Tokenizer"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: false
images: []
weight: 21
---
Tokenizers convert texts to numerical data so that models can process.

2 objectives of a good tokenizer:
- meaningful representation (i.e., effectiveness)
- smallest representation (i.e., efficiency)

Types of tokenizers:
| Type | Def | Pros | Cons |
|:-----|:----|:-----|:-----|
| **Word-based** | Split on spaces/punctuations | Easy to implement & use<br>High interpretability | High memory cost (huge vocabulary);<br>Risk of too many unknown tokens |
| **Character-based** | Split into characters | Low memory cost (small vocabulary);<br>Few unknown tokens | Low performance (for languages where characters are not meaningful);<br>Large sequence length |
| **Subword** | Keep frequent words;<br>Split rare words into meaningful subwords;<br>Add a word separator token at the end of each word | Combine pros of both methods above | N/A |

How to build a tokenizer:
1. **Normalization**: clean text
2. **Pre-tokenization**: split text into words
3. **Modeling**: convert words into a token sequence
4. **Post-processing**: add special tokens, generate attention mask & token type IDs

&nbsp;

# BPE (Byte-Pair Encoding)
Idea: 
1. Get a unique set of words & word counts from corpus.
2. Get a set of base vocabulary of all characters in the corpus (preferably all ASCII characters).
3. Repeatedly merge the pair of tokens with max counts, till desired vocabulary size.

Usage: RoBERTa, DeBERTa, BART, GPT, GPT-2

Pros:
- Guarantee no [UNK]
- Flexible vocabulary management
- Balance words and characters

Cons:
- Ignore context {{<math>}}$ \rightarrow ${{</math>}} suboptimal splits for words with different meanings in different contexts

&nbsp;

# WordPiece
Idea:
1. Get a unique set of words & word counts from corpus.
2. Get a set of base vocabulary of all characters in the corpus, but add a prefix "##" to all the characters inside each word.
3. Repeatedly merge the pair of tokens with the following score formula, till desired vocabulary size.
$$
\text{score}=\frac{\text{freq}(pair)}{\text{freq}(part_1)\times\text{freq}(part_2)}
$$

Usage: BERT, DistilBERT, MobileBERT, Funnel Transformers, MPNET

Pros:
- Prioritize rare words, where individual parts are less frequent in the vocabulary

Cons:
- Only save the final vocabulary, not the merge rules {{<math>}}$ \rightarrow ${{</math>}} label an entire word as [UNK] when any part is not in the vocabulary
- Ignore context {{<math>}}$ \rightarrow ${{</math>}} suboptimal splits for words with different meanings in different contexts

&nbsp;

# Unigram
Idea:
1. Get a large vocabulary (via most common substrings in pre-tokenized words, or BPE on initial corpus with a large vocabulary size)
2. 

Usage: T5, ALBERT, mBART, XLNet