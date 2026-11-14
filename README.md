# Risk Desk in a Box

A personal risk manager for an ordinary investor. It was built as the final project for a university Generative AI course.
The full brief is in [risk-desk-in-a-box.md](risk-desk-in-a-box.md).

> The market moves are AI estimates, not forecasts, and not investment advice.

## Setup

Requires [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 for this project by itself.

```powershell
uv sync                      # create .venv and install dependencies
copy .env.example .env       # then put your Groq key in .env
uv run python app.py         # open http://127.0.0.1:7860
uv run pytest                # run the tests
```

## Hosting

The app runs locally. It can also open a temporary Gradio share link for demos, using one of these:
`uv run python app.py --share` or `RISKDESK_SHARE=true` in `.env`.

The Hugging Face free tier no longer offers Gradio Spaces (checked 14 Nov 2026), so we don't use it.
We revisit hosting at the end only if a permanent link is needed.

## Language models

- **Groq** is the main engine (free tier).
- **Gemini** is an optional backup. The app works without it.
- No paid APIs.

## Layout

| Path | What it holds |
|---|---|
| `app.py` | Gradio entry point |
| `src/riskdesk/config.py` | Settings and API keys, loaded from `.env` |
| `src/riskdesk/data/` | Market data download and crisis library build |
| `src/riskdesk/engine/` | Risk engine: portfolio, loss, VaR/ES, reverse search (no AI) |
| `src/riskdesk/retrieval/` | Embeddings and similar-event retrieval (RAG) |
| `src/riskdesk/llm/` | Groq/Gemini client, answer cache, translator, memo writer |
| `src/riskdesk/checks/` | JSON sanity limits and the memo number checker |
| `src/riskdesk/tagger/` | Runs the trained headline tagger on CPU |
| `data/` | `market_data.csv`, `events.csv`, crisis library, index |
| `notebooks/` | Kaggle training notebook for the tagger (the only notebook) |
| `cache/dev/` | LLM answers cached during development (git-ignored) |
| `cache/demo/` | Cached answers for the demo scenarios (committed) |
| `eval/` | Evaluation scripts and result tables |
| `tests/` | pytest tests |
