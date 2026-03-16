
---

# README for Project3

# Project3 — LLM Bias Experiment

This project tests whether an LLM responds differently to the same scientific question when the user is described with different personas.

## Goal

The experiment compares responses across personas such as:

- student
- parent
- scientist
- journalist
- policymaker
- skeptic
- anxious user
- no-context user

It uses questions about:

- GMO safety
- nuclear energy safety
- vaccines and autism
- climate change consensus

## Metrics

For each response, the project measures:

- word count
- hedge density
- deflection rate
- fact coverage
- sentiment
- average sentence length

## Requirements

Install dependencies:

```bash
pip install openai matplotlib seaborn pandas numpy scipy
