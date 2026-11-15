# Risk Desk in a Box

A personal risk manager for an ordinary investor, built as the final project for a university Generative AI course.

> The market moves are AI estimates, not forecasts, and not investment advice. Every money figure is calculated by plain code.

## What it does

- **Forward mode: "what does this news do to me?"** Type a headline or a what-if sentence, such as
  *"Oil jumps to $150 as a Gulf shipping route closes"*. The app finds the most similar past market events,
  asks a language model to turn the headline into market moves grounded in those events, applies the moves to
  your portfolio and shows the loss, the result per holding and a comparison with historical Value-at-Risk.
- **Backward mode: "what would make me lose 10%?"** Code searches for market moves that produce exactly that loss;
  the language model writes each one as a short news story.
- **Risk memo.** A plain-language memo of the result. A number checker compares every number in it with the
  calculated figures and makes the model correct the memo when one does not match.

## Generative AI techniques

| Technique | Job in this project |
|---|---|
| Embeddings | 123 past market events and every new headline are embedded (`BAAI/bge-small-en-v1.5`, on CPU) to find similar events |
| RAG | The most similar past events, and what markets really did after them, are given to the model as evidence |
| Words to structured numbers | The model turns a sentence into a fixed set of market moves (JSON), checked against plausible limits |
| AI writing, checked by code | The memo and the backward-mode stories are written by the model; code verifies every number they contain |

## Setup

Requires [uv](https://docs.astral.sh/uv/), which installs Python 3.12 for this project by itself.

```powershell
uv sync                      # create .venv and install dependencies
copy .env.example .env       # then put your Groq key (and optionally a Gemini key) in .env
uv run python app.py         # open http://127.0.0.1:7860
uv run pytest                # run the tests
```

**Try it without any API key:** keep the sample portfolio and click the sample headlines, or use the backward tab
with the 10% target. Their answers are saved in `cache/demo/`, so they work offline. In the forward tab, tick
*"Demo: plant a wrong number in the memo's first draft"* to watch the number checker catch it.
The *Recalculate with these moves* button needs no AI at all.

## Language models

- **Groq** (free tier) is the main engine: `openai/gpt-oss-120b`, then `openai/gpt-oss-20b` when rate-limited.
- **Gemini** Flash-Lite (free tier) is the backup when Groq cannot answer. Optional.
- Both are called through their OpenAI-compatible endpoints with one client (`src/riskdesk/llm/client.py`).
  Every answer is cached, so the same question never costs twice.

## Data

- `data/market_data.csv`: 5,502 trading days (2005 to 2026) for 6 risk drivers: S&P 500, Brent oil, gold,
  the US dollar index, bitcoin and the US 10-year yield (Yahoo Finance and FRED, downloaded once by
  `src/riskdesk/data/download.py`; the app never calls a market API).
- `data/events.csv`: 123 past market events, each checked by hand, with its date, FRM risk type and region.
- `data/crisis_library.csv`: what each driver did after each event. Rebuild the library and the index with
  `uv run python data/build_library.py`.

## Layout

| Path | What it holds |
|---|---|
| `app.py` | Gradio app |
| `src/riskdesk/engine/` | Risk engine: holdings, stress loss, VaR / Expected Shortfall, reverse search (no AI) |
| `src/riskdesk/retrieval/` | Embeddings and similar-event search |
| `src/riskdesk/llm/` | Model client and cache, scenario translator, memo writer, backward-mode stories |
| `src/riskdesk/checks/` | Limits on the market moves and the memo number checker |
| `src/riskdesk/forward.py`, `backward.py` | The two modes, end to end |
| `src/riskdesk/demo.py`, `scripts/build_demo_cache.py` | The demo flows and the script that saves their answers |
| `data/` | Market data, event list, crisis library, embedding index |
| `eval/` | Evaluation scripts, test sets and result tables |
| `cache/demo/` | Saved answers for the sample headlines (committed) |
| `tests/` | pytest tests |

## Evaluations

```powershell
uv run python eval/run_translator.py          # 20 written scenarios, without past events
uv run python eval/run_translator.py --rag    # the same, with past events (RAG)
uv run python eval/compare.py                 # before vs after RAG
uv run python eval/run_retrieval.py           # does the right past event come up?
uv run python eval/run_levels.py              # headlines that name a level ("oil jumps to $150")
uv run python eval/run_heldout.py             # answers vs what markets really did, on hidden past events
```

Results are written to `eval/results/`.
