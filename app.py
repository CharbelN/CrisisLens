"""Risk Desk in a Box: Gradio entry point.

Run locally:              uv run python app.py
With a public demo link:  uv run python app.py --share
   (or set RISKDESK_SHARE=true in .env)
"""

from __future__ import annotations

import argparse

import gradio as gr
import pandas as pd

from riskdesk.checks.moves import HORIZON_DAYS, plausible_limits
from riskdesk.config import PROJECT_ROOT, load_settings
from riskdesk.data.market import driver_changes, load_market_data
from riskdesk.forward import ForwardResult, run_forward, run_manual
from riskdesk.llm.client import GroqClient
from riskdesk.ui import holdings_table, moves_table, parse_moves, parse_portfolio, portfolio_table, summary_markdown

DISCLAIMER = (
    "**These are AI estimates, not forecasts and not investment advice.** "
    "Market moves are the model's educated guesses; all money figures are calculated by plain code."
)
SAMPLE_IDS = ["S01", "S02", "S03", "S06", "S10"]


def _error(message: str) -> ForwardResult:
    return ForwardResult("error", message, None, None, {})


def build_app() -> gr.Blocks:
    settings = load_settings()
    levels = load_market_data()
    changes = driver_changes(levels, HORIZON_DAYS)
    limits = plausible_limits(levels)
    client = GroqClient()
    scenarios = pd.read_csv(PROJECT_ROOT / "eval" / "scenarios.csv", dtype=str, keep_default_na=False)
    samples = scenarios.set_index("id").loc[SAMPLE_IDS, "scenario"].tolist()

    def outputs(result: ForwardResult, moves_df: pd.DataFrame):
        return summary_markdown(result), moves_df, holdings_table(result)

    def on_run(portfolio_df, text):
        try:
            portfolio = parse_portfolio(portfolio_df)
            if not (text or "").strip():
                return outputs(_error("Type a headline or what-if sentence first."), moves_table())
            result = run_forward(portfolio, text, client, limits, changes)
        except ValueError as exc:  # bad portfolio or moves
            return outputs(_error(str(exc)), moves_table())
        moves = result.translation.moves if result.translation and result.translation.moves else None
        return outputs(result, moves_table(moves))

    def on_recalc(portfolio_df, moves_df):
        try:
            result = run_manual(parse_portfolio(portfolio_df), parse_moves(moves_df), changes)
        except ValueError as exc:
            return outputs(_error(str(exc)), moves_df)
        return outputs(result, moves_df)

    with gr.Blocks(title="Risk Desk in a Box") as demo:
        gr.Markdown("# Risk Desk in a Box")
        gr.Markdown(DISCLAIMER)
        with gr.Tab("Forward: what does this news do to me?"):
            with gr.Row():
                with gr.Column(scale=2):
                    portfolio = gr.Dataframe(value=portfolio_table(), label="Your portfolio", interactive=True)
                with gr.Column(scale=3):
                    headline = gr.Textbox(
                        label="Headline or what-if sentence",
                        placeholder="Oil jumps to $150 as a Gulf shipping route closes",
                        lines=2,
                    )
                    gr.Markdown("Samples:")
                    sample_buttons = [gr.Button(s, size="sm") for s in samples]
                    run = gr.Button("Run", variant="primary")
            summary = gr.Markdown()
            with gr.Row():
                moves = gr.Dataframe(
                    value=moves_table(),
                    label=f"Market moves over {HORIZON_DAYS} trading days (edit and press Recalculate)",
                    interactive=True,
                )
                per_holding = gr.Dataframe(label="Result per holding", interactive=False)
            recalc = gr.Button("Recalculate with these moves")

            for btn, text in zip(sample_buttons, samples):
                btn.click(lambda t=text: t, outputs=headline)
            run.click(on_run, inputs=[portfolio, headline], outputs=[summary, moves, per_holding])
            headline.submit(on_run, inputs=[portfolio, headline], outputs=[summary, moves, per_holding])
            recalc.click(on_recalc, inputs=[portfolio, moves], outputs=[summary, moves, per_holding])

        gr.Markdown(
            f"Engine status: Groq key {'found' if settings.has_groq else 'missing (type moves by hand)'}, "
            f"Gemini backup {'found' if settings.has_gemini else 'not set (optional)'}."
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
