"""Build or check the demo cache (cache/demo/, committed).

    uv run python scripts/build_demo_cache.py          # run the demo flows, save every answer
    uv run python scripts/build_demo_cache.py --check  # replay with Groq switched off
    uv run python scripts/build_demo_cache.py --with-gemini  # Groq out: Gemini fills the gaps

Build: runs the demo flows (src/riskdesk/demo.py) with the main model, paced under the free
tokens-per-minute limit, and saves EVERY answer it receives into cache/demo/, including answers
that failed a check, so a replay follows exactly the same steps (same retries, same result).
Answers already in cache/dev/ are reused, not paid for again.

Check: the same flows with no network. Any step that would need a live call is reported.
Run it after changing a prompt, the market data or the crisis library: those change the
questions, so the old answers no longer match.
"""

from __future__ import annotations

import argparse
import sys

from riskdesk.demo import run_demo_flows
from riskdesk.llm.client import (DEMO_CACHE_DIR, GEMINI_LITE, GROQ_MAIN, LLMClient, LLMReply, LLMUnavailableError,
                                 Messages, PacedClient, save_reply)


class Recorder:
    """Passes questions on and saves every answer into the demo cache."""

    def __init__(self, inner) -> None:
        self.inner, self.saved = inner, 0

    def chat_json(self, messages: Messages) -> LLMReply:
        reply = self.inner.chat_json(messages)
        save_reply(DEMO_CACHE_DIR, messages, reply)
        self.saved += 1
        return reply


class Offline:
    """Answers only from the caches; counts the questions that would need a live call."""

    class _NoNetwork:
        def __getattr__(self, name):
            raise LLMUnavailableError("offline check: no live calls")

    def __init__(self) -> None:
        self.inner = LLMClient(cache_dir=None, sdk=self._NoNetwork())  # demo cache only
        self.misses = 0

    def chat_json(self, messages: Messages) -> LLMReply:
        try:
            return self.inner.chat_json(messages)
        except LLMUnavailableError:
            self.misses += 1
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="replay from cache/demo/ only, no network")
    parser.add_argument("--with-gemini", action="store_true",
                        help="when Groq is out, ask Gemini for the missing answers (cached answers are reused)")
    args = parser.parse_args()

    builder = PacedClient(inner=LLMClient([GROQ_MAIN, GEMINI_LITE])) if args.with_gemini else PacedClient()
    client = Offline() if args.check else Recorder(builder)
    results = run_demo_flows(client)
    for r in results:
        print(f"{'ok  ' if r.ok else 'FAIL'} {r.name}: {r.detail}")
    if args.check:
        print(f"\n{client.misses} question(s) not in the demo cache.")
        sys.exit(1 if client.misses or not all(r.ok for r in results) else 0)
    n_files = len(list(DEMO_CACHE_DIR.glob("*.json")))
    print(f"\nSaved {client.saved} answer(s); cache/demo/ now holds {n_files} file(s). Run with --check to confirm.")


if __name__ == "__main__":
    main()
