"""Risk Desk in a Box: Gradio entry point.

Run locally:              uv run python app.py
With a public demo link:  uv run python app.py --share
   (or set RISKDESK_SHARE=true in .env)
"""

from __future__ import annotations

import argparse

import gradio as gr

from riskdesk.config import load_settings

DISCLAIMER = (
    "**These are AI estimates, not forecasts and not investment advice.** "
    "Market moves are the model's educated guesses, grounded in past events."
)


def echo(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "Risk Desk is alive. Type a headline to test the page."
    return f"Risk Desk is alive. You typed: {text}"


def build_app() -> gr.Blocks:
    settings = load_settings()
    with gr.Blocks(title="Risk Desk in a Box") as demo:
        gr.Markdown("# Risk Desk in a Box")
        gr.Markdown(DISCLAIMER)
        headline = gr.Textbox(label="Headline or what-if sentence", placeholder="Oil jumps to $150 as a Gulf shipping route closes")
        output = gr.Textbox(label="Status", interactive=False)
        gr.Button("Check").click(echo, inputs=headline, outputs=output)
        gr.Markdown(
            f"Engine status: Groq key {'found' if settings.has_groq else 'missing'}, "
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
