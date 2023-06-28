---
title : "Conversational Recommender System"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: false
images: []
weight: 200
---
# Preliminaries

## Dialogue Systems

- Problem:

    - Input: conversation history (+ background knowledge)
    - Output: response utterance (+ actions with external tools)
- Goal:

    - **Naturalness**: talk like how humans talk
    - **Specialization**: perform exceptionally well in its specialized scenarios
    - **Automation**: help solve user issues with no manual effort from both sides
- Taxonomy:

    - Objective:
        - **Open-domain**: chat about any topic, focus on engagement and interactiveness
        - **Goal-oriented**: chat about a specific field of topics, focus on guidance for task completion
    - Architecture:
        - **Modular**: a pipeline of modules, each with a specific functionality
        - **End-to-End**: a single model
    - Interaction: (S: system; U: user; A: active; P: passive; E: engage)
        - **SAUP**: system asks clarifying questions; user responds directly
        - **SAUE**: system asks questions; user responds; allow chit-chat
        - **SAUA**: both sides can lead; allow chit-chat
        - **UASP**: user asks questions; system responds
    - Length:
        - Single-turn: QA, prompting
        - Multi-turn: conversation
        - Perpetual: conversation while memorizing info about the user and past conversations

## Conversational Recommender Systems

- Definition: support users in achieving recommendation-related goals through a multi-turn dialogue
    - Categories: Goal-oriented + Multi-turn
- Input: conversation history (+ user preferences + item knowledge + domain knowledge)
- Output: response utterance (+ explanation + item recommendation)

# Architecture

## Modular CRS

- **Input Processing module**:
    - Input: generic user input (in various forms: natural language, formatted texts (e.g., forms), buttons/clicks, voice, handwritings, multi-language, ...)
    - Output: user intent + dialogue state update
    - Procedure: acquire input -> decode input -> process info
    - Directions: NLP & NLU
        - **Intent Recognition**: the informative need of the user (e.g., ask for recommendation, provide preferences, provide feedbacks, ask questions about items, ...)
        - **Named Entity Recognition**: tne informative elements (e.g., names, locations, characteristics, items, ...)
        - **Sentiment Analysis**: the tone of the message

- **User Modeling module**:
    - Input: content of user input + external knowledge base (+ relevant entities + sentiment + dialogue state)
    - Output: user profile representation (+ dialogue state update)
    - Procedure: extract preferences/needs from input
        - Objective properties: item characteristics from exogenous knowledge sources (i.e., knowledge graphs) based on **Entity Linking**
        - Subjective properties: personal feelings about the item based on **Aspect Extraction** (from reviews)

- **Recommendation module**:
    - When to recommend: depends on the interaction strategy and the dialogue flexibility
        - Ask "good" questions (to guide recommendation)
        - Decide when enough prerferences are collected
    - What to recommend: aware of knowledge from previous utterances, background knowledge, item features, ...

- **Output Generation module**:
    - Template-based Generation:
        - Pros: easy to implement, very straightforward and effective
        - Cons: static, repetitive
    - Dynamic Generation:
        - Pros: dynamic
        - Cons: poorly significant answers
    - Directions:
        - Concept explanation
        - Retrieval-based answer generation

## End-to-End CRS
One single neural network handles everything. Any LM/LLM can be trained to do this.
- Pros: simple, flexible
- Cons: requires training, hard to debug

# Evaluation
What makes a CRS "good"?
- Accurate recommendations
- Fewer interaction turns
- Engaging interactions
- Realistic & Natural utterances

What is the goal for CRS evaluation?
- Research goals: high prediction accuracy, high effectiveness of algorithm, ability to mimic humans, ...
- Business goals: more sales, more engagements, more profits, less costs, ...

Current evaluation paradigms:
- Online studies: experimental research with users via A/B Testing
- Offline studies: accuracy evaluation of single components, simulation of user behavior
- Qualitative studies: mainly based on human annotations

Current evaluation dimensions:
- Effectiveness: do we get better recommendations with CRS?
    - metrics: task completion rate, hit rate, choice satisfaction, accuracy metrics, ...
- Efficiency: is CRS faster than other RSs?
    - metrics: interaction time, #interactions in each conversation, cognitive effort, perceived effort, ...
- Quality: is the conversation natural, fluent, and relevant?
    - metrics: perplexity, BLEU, ROGUE, usability, misrecognition rate, quality of dialogue, user control, ...