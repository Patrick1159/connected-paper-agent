#!/usr/bin/env python3
"""Entry point: python main.py <arxiv_url> [--config config.yaml]"""
from __future__ import annotations

import argparse
import logging
import sys


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Research Literature Trace Agent — traces citation lineage from an arXiv paper."
    )
    parser.add_argument("arxiv_url", help="arXiv paper URL or ID, e.g. https://arxiv.org/abs/2305.10601")
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: auto-detect)")
    parser.add_argument("--max-rounds", type=int, default=None, help="Override max iteration rounds")
    parser.add_argument("--top-k", type=int, default=None, help="Override top-k candidates per round")
    args = parser.parse_args()

    from src.paper_agent.config.settings import Settings
    from src.paper_agent.tracing.agent import build_agent
    from src.paper_agent.tracing.state import AgentState

    settings = Settings.load(args.config)
    if args.max_rounds is not None:
        settings.agent.max_rounds = args.max_rounds
    if args.top_k is not None:
        settings.agent.top_k = args.top_k

    if not settings.llm.api_key:
        print(
            "[ERROR] No API key found. Set PAPER_AGENT_API_KEY environment variable "
            "or add api_key to config.yaml.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[*] Starting trace for: {args.arxiv_url}")
    print(f"    max_rounds={settings.agent.max_rounds}, top_k={settings.agent.top_k}")
    print(f"    model={settings.llm.model_id} ({settings.llm.protocol})")

    agent = build_agent(settings)
    initial_state = AgentState(root_url=args.arxiv_url)

    final_state = agent.invoke(initial_state)
    if isinstance(final_state, dict):
        final_state = AgentState(**final_state)

    if final_state.error:
        print(f"[ERROR] {final_state.error}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[✓] Report saved to: {final_state.report_path}")
    print(f"[✓] Lineage chain ({len(final_state.lineage_chain)} papers):")
    for i, aid in enumerate(final_state.lineage_chain):
        print(f"    {i+1}. {aid}")


if __name__ == "__main__":
    main()
