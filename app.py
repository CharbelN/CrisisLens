"""Risk Desk in a Box: Gradio entry point.

Run locally:              uv run python app.py
With a public demo link:  uv run python app.py --share
   (or set RISKDESK_SHARE=true in .env)
"""

from __future__ import annotations

import argparse

import gradio as gr
import pandas as pd

from riskdesk.backward import BackwardResult, run_backward
from riskdesk.checks.levels import latest_levels
from riskdesk.data.library import load_library
from riskdesk.engine.reverse import MAX_TARGET_PCT
from riskdesk.checks.moves import HORIZON_DAYS, plausible_limits
from riskdesk.config import load_settings
from riskdesk.data.market import driver_changes, load_market_data
from riskdesk.forward import ForwardResult, run_forward, run_manual
from riskdesk.demo import DEMO_TARGETS, sample_headlines
from riskdesk.llm.client import DEMO_CACHE_DIR, default_client
from riskdesk.llm.memo import write_memo
from riskdesk.retrieval.index import Retriever, StaleIndexError
from riskdesk.ui import (
    backward_markdown, holdings_table, memo_markdown, moves_table, scenarios_table, parse_moves, parse_portfolio, portfolio_table, summary_markdown,
    twins_table,
)

DISCLAIMER = (
    "**These are AI estimates, not forecasts and not investment advice.** "
    "Market moves are the model's educated guesses; all money figures are calculated by plain code."
)


def _error(message: str) -> ForwardResult:
    return ForwardResult("error", message, None, None, {})


def build_app() -> gr.Blocks:
    settings = load_settings()
    levels = load_market_data()
    changes = driver_changes(levels, HORIZON_DAYS)
    limits = plausible_limits(levels)
    start = latest_levels(levels)
    library = load_library()
    client = default_client()
    try:
        retriever = Retriever.load()
        retrieval_note = "historical twins on"
    except (FileNotFoundError, StaleIndexError) as exc:
        retriever = None
        retrieval_note = f"historical twins OFF ({exc})"
    samples = sample_headlines()
    n_demo = len(list(DEMO_CACHE_DIR.glob("*.json")))

    def outputs(result: ForwardResult, moves_df: pd.DataFrame, text: str = "", plant: bool = False):
        memo = write_memo(result, text, client, plant) if result.stress is not None else None
        return summary_markdown(result), moves_df, holdings_table(result), memo_markdown(memo), twins_table(result)

    def on_run(portfolio_df, text, plant):
        try:
            portfolio = parse_portfolio(portfolio_df)
            if not (text or "").strip():
                return outputs(_error("Type a headline or what-if sentence first."), moves_table())
            result = run_forward(portfolio, text, client, limits, changes, retriever, start)
        except ValueError as exc:  # bad portfolio or moves
            return outputs(_error(str(exc)), moves_table())
        moves = result.translation.moves if result.translation and result.translation.moves else None
        return outputs(result, moves_table(moves), text, plant)

    def on_recalc(portfolio_df, moves_df, text, plant):
        try:
            result = run_manual(parse_portfolio(portfolio_df), parse_moves(moves_df), changes)
        except ValueError as exc:
            return outputs(_error(str(exc)), moves_df)[:4]
        return outputs(result, moves_df, text, plant)[:4]  # keep the twins from the last Run on screen

    def on_find(portfolio_df, target_pct):
        try:
            portfolio = parse_portfolio(portfolio_df)
            result = run_backward(portfolio, float(target_pct or 0), changes, limits, library, client)
        except ValueError as exc:  # bad portfolio
            result = BackwardResult("error", str(exc), None, None)
        return backward_markdown(result), scenarios_table(result)

    with gr.Blocks(title="Risk Desk in a Box") as demo:
        gr.Markdown("# Risk Desk in a Box")
        gr.Markdown(DISCLAIMER)
        portfolio = gr.Dataframe(value=portfolio_table(), label="Your portfolio (used by both tabs)", interactive=True)
        with gr.Tab("Forward: what does this news do to me?"):
            headline = gr.Textbox(
                label="Headline or what-if sentence",
                placeholder="Oil jumps to $150 as a Gulf shipping route closes",
                lines=2,
            )
            gr.Markdown("Samples:")
            with gr.Row():
                sample_buttons = [gr.Button(s, size="sm") for s in samples]
            with gr.Row():
                run = gr.Button("Run", variant="primary")
                plant = gr.Checkbox(label="Demo: plant a wrong number in the memo's first draft", value=False)
            summary = gr.Markdown()
            with gr.Row():
                moves = gr.Dataframe(
                    value=moves_table(),
                    label=f"Market moves over {HORIZON_DAYS} trading days (edit and press Recalculate)",
                    interactive=True,
                )
                per_holding = gr.Dataframe(label="Result per holding", interactive=False)
            recalc = gr.Button("Recalculate with these moves")
            memo = gr.Markdown()
            twins = gr.Dataframe(
                label=f"Historical twins: most similar past events and what markets actually did (first day / {HORIZON_DAYS} trading days)",
                interactive=False,
            )

            for btn, text in zip(sample_buttons, samples):
                btn.click(lambda t=text: t, outputs=headline)
            run_out = [summary, moves, per_holding, memo, twins]
            run.click(on_run, inputs=[portfolio, headline, plant], outputs=run_out)
            headline.submit(on_run, inputs=[portfolio, headline, plant], outputs=run_out)
            recalc.click(on_recalc, inputs=[portfolio, moves, headline, plant], outputs=run_out[:4])

        with gr.Tab("Backward: what would make me lose X%?"):
            gr.Markdown("Code searches history for sets of market moves that lose exactly this much, then the AI "
                        "writes each one as a short news story. Every number in a story is checked against the moves.")
            with gr.Row():
                target = gr.Number(value=DEMO_TARGETS[0], label="Target loss (%)", minimum=0.1, maximum=MAX_TARGET_PCT)
                find = gr.Button("Find scenarios", variant="primary")
            back_summary = gr.Markdown()
            back_table = gr.Dataframe(label=f"Scenarios: market moves over {HORIZON_DAYS} trading days",
                                      interactive=False)
            find.click(on_find, inputs=[portfolio, target], outputs=[back_summary, back_table])

        gr.Markdown(
            f"Engine status: Groq key {'found' if settings.has_groq else 'missing (type moves by hand)'}, "
            f"backups: OpenRouter {'on' if settings.has_openrouter else 'off'}, "
            f"Gemini {'on' if settings.has_gemini else 'off'} (optional), {retrieval_note}, "
            f"starting levels from the latest close in the data ({start.date}), "
            f"{n_demo} saved demo answer(s) (used first, so the samples work offline with the sample portfolio)."
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Risk Desk in a Box.")
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio link.")
    args = parser.parse_args()

    share = args.share or load_settings().share_enabled
    build_app().launch(server_name="127.0.0.1", server_port=7860, share=share)


if __name__ == "__main__":
    main()
