# Project2 — Nim Trustworthiness Experiment

This project tests how well an LLM follows the rules of the Nim game under two interface settings:

- **Strict mode**: structured output with schema parsing
- **Loose mode**: plain text output with regex/JSON parsing

The experiment compares the LLM against:

- a **Random** bot
- an **Optimal** bot

It measures:

- win rate
- illegal moves
- parse failures
- repair attempts
- optimal move rate

## Files

- `main.py` — runs the full Nim experiment
- `plt_loose_only.py` — creates figures using only the **loose-mode** results

## Requirements

Install dependencies:

```bash
pip install openai matplotlib pydantic
