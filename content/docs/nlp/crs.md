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

- Input Processing module:
    - Input: generic user input (in various forms)
    - Output: user intent + dialogue state update
    - Procedure: acquire input -> decode input -> process info
    - Directions:
        - Supported interactions: natural language, formatted texts (e.g., forms), buttons/clicks, ...
        - Interaction strategies: text, voice, handwritings, multi-language, ...
        - NLP & NLU: intent recognition, named entity recognition, sentiment analysis
            - intent examples: ask for recommendation, provide preferences, provide feedbacks, ask questions about items, ...
- User Modeling module:
    - Preference/Need modeling: entities, objective features, subjective features, ...
    - 
- Recommendation module:
    - Timing: when to end preference elicitation and start recommendation
    - Info: previous utterances, background knowledge, item features, ...
- Output Generation module:
    f, ...
- Dialogue Management module:

## End-to-End CRS
