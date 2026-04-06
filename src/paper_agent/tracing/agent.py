from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from typing import Any

from langgraph.graph import END, StateGraph

from ..config.settings import Settings
from ..evaluation.evaluator import select_best_lineage
from ..graph_store.citation_graph import CitationGraph, NodeStatus, PaperNode
from ..arxiv_client import configure_arxiv_access
from ..ingestion.arxiv_fetcher import fetch_metadata, _canonicalize_arxiv_id, _normalize_arxiv_id
from ..llm.factory import build_llm_client
from ..parsing.pdf_parser import parse_pdf
from ..ranking.ranker import rank_candidates
from ..reporting.reporter import render_report, save_report
from ..retrieval.reference_resolver import extract_candidate_ids
from ..summarization.summarizer import summarize
from .state import AgentState


logger = logging.getLogger(__name__)


def build_agent(settings: Settings):
    configure_arxiv_access(settings.arxiv)
    llm = build_llm_client(settings.llm)
    graph = CitationGraph()

    def _snapshot_status(state: AgentState) -> str:
        if state.error:
            return "error"
        if state.completed_at is not None:
            return "completed"
        return "running"

    def _snapshot_path(root_id: str) -> str:
        os.makedirs(settings.paths.data_dir, exist_ok=True)
        return f"{settings.paths.data_dir}/{root_id.replace('/', '_')}_graph.json"

    def _persist_snapshot(state: AgentState, phase: str, detail: str = "") -> None:
        if not state.root_id:
            return

        if not state.graph_path:
            state.graph_path = _snapshot_path(state.root_id)

        state.token_usage = llm.get_token_usage()

        snapshot = graph.to_dict()
        snapshot["meta"] = {
            "root_id": state.root_id,
            "status": _snapshot_status(state),
            "phase": phase,
            "detail": detail,
            "current_round": state.current_round,
            "max_rounds": settings.agent.max_rounds,
            "frontier": list(state.frontier),
            "frontier_size": len(state.frontier),
            "skipped_refs": list(state.skipped_refs),
            "lineage_chain": list(state.lineage_chain),
            "lineage_rationale": state.lineage_rationale,
            "report_path": state.report_path,
            "started_at": state.started_at,
            "completed_at": state.completed_at,
            "elapsed_seconds": state.elapsed_seconds,
            "updated_at": time.time(),
            "token_usage": {
                "prompt_tokens": state.token_usage.prompt_tokens,
                "completion_tokens": state.token_usage.completion_tokens,
                "total_tokens": state.token_usage.total_tokens,
                "request_count": state.token_usage.request_count,
                "estimated_request_count": state.token_usage.estimated_request_count,
            },
        }

        tmp_path = f"{state.graph_path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(snapshot, f, indent=2)
        os.replace(tmp_path, state.graph_path)

    # ------------------------------------------------------------------ #
    # Node: ingest root paper                                              #
    # ------------------------------------------------------------------ #
    def ingest_root(state: AgentState) -> AgentState:
        logger.info("[ingest] Resolving root paper from input: %s", state.root_url)
        try:
            arxiv_id = _canonicalize_arxiv_id(_normalize_arxiv_id(state.root_url))
        except ValueError as e:
            state.error = str(e)
            logger.error("[ingest] Invalid root input: %s", e)
            return state
        state.root_id = arxiv_id

        try:
            paper = fetch_metadata(arxiv_id)
        except Exception as e:
            state.error = f"Could not fetch root paper: {e}"
            logger.error("[ingest] Failed to fetch root paper %s: %s", arxiv_id, e)
            return state

        node = PaperNode(
            arxiv_id=arxiv_id,
            title=paper.title,
            authors=paper.authors,
            abstract=paper.abstract,
            year=paper.year,
            status=NodeStatus.PENDING,
            round_added=0,
        )
        graph.add_node(node)
        state.graph_path = _snapshot_path(arxiv_id)
        state.frontier = [arxiv_id]
        _persist_snapshot(state, "ingest_completed", f"Root paper {arxiv_id} added to graph")
        logger.info("[ingest] Root paper ready: %s | title=%s", arxiv_id, paper.title)
        return state

    # ------------------------------------------------------------------ #
    # Node: process frontier (one round)                                  #
    # ------------------------------------------------------------------ #
    def process_round(state: AgentState) -> AgentState:
        round_no = state.current_round + 1
        _persist_snapshot(state, "round_started", f"Starting round {round_no}")
        logger.info(
            "[round %s/%s] Starting with %s paper(s) in frontier",
            round_no,
            settings.agent.max_rounds,
            len(state.frontier),
        )
        next_frontier: list[str] = []

        for arxiv_id in state.frontier:
            node_data = graph.get_node(arxiv_id)
            if node_data is None:
                logger.warning("[round %s] Missing graph node for %s, skipping", round_no, arxiv_id)
                continue
            if node_data["status"] == NodeStatus.ANALYZED:
                logger.info("[round %s] %s already analyzed, skipping", round_no, arxiv_id)
                continue

            logger.info("[round %s] Processing paper %s", round_no, arxiv_id)
            node_depth = node_data.get("round_added", 0)
            allow_expansion = node_depth < settings.agent.max_rounds

            # --- fetch full paper + PDF ---
            try:
                paper = fetch_metadata(arxiv_id)
            except Exception:
                graph.update_node(arxiv_id, status=NodeStatus.FAILED)
                state.skipped_refs.append(arxiv_id)
                _persist_snapshot(state, "paper_failed", f"Metadata fetch failed for {arxiv_id}")
                logger.exception("[round %s] Failed to fetch metadata for %s", round_no, arxiv_id)
                continue

            # update node with fetched metadata (title/abstract may now be available)
            graph.update_node(arxiv_id, title=paper.title, abstract=paper.abstract)
            _persist_snapshot(state, "metadata_fetched", f"Fetched metadata for {arxiv_id}")
            logger.info("[round %s] Metadata fetched for %s", round_no, arxiv_id)

            # --- download & parse PDF ---
            pdf_ids: list[str] = []
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    from ..ingestion.arxiv_fetcher import download_pdf

                    pdf_path = download_pdf(paper, dest_dir=tmpdir)
                    parsed = parse_pdf(pdf_path)
                    pdf_ids = parsed.arxiv_ids_found
                    raw_refs = parsed.references_raw
                logger.info(
                    "[round %s] PDF parsed for %s, found %s arXiv reference(s)",
                    round_no,
                    arxiv_id,
                    len(pdf_ids),
                )
            except Exception as e:
                raw_refs = []
                logger.warning(
                    "[round %s] PDF parsing failed for %s, continuing without references: %s",
                    round_no,
                    arxiv_id,
                    e,
                )

            # --- summarize ---
            try:
                logger.info("[round %s] Summarizing %s via LLM", round_no, arxiv_id)
                summary = summarize(llm, node_data["title"], node_data["abstract"])
                graph.update_node(
                    arxiv_id,
                    status=NodeStatus.ANALYZED,
                    core_idea=summary.core_idea,
                    method=summary.method,
                    problem_solved=summary.problem_solved,
                    research_field=summary.field,
                )
                _persist_snapshot(state, "paper_analyzed", f"Summarization completed for {arxiv_id}")
                logger.info("[round %s] Summarization completed for %s", round_no, arxiv_id)
            except Exception as e:
                graph.update_node(arxiv_id, status=NodeStatus.FAILED)
                state.skipped_refs.append(f"{arxiv_id} (summarization failed: {e})")
                _persist_snapshot(state, "paper_failed", f"Summarization failed for {arxiv_id}")
                logger.error("[round %s] Summarization failed for %s: %s", round_no, arxiv_id, e)
                continue

            if not allow_expansion:
                logger.info(
                    "[round %s] %s is at max tracing depth (%s), analyze-only mode enabled",
                    round_no,
                    arxiv_id,
                    settings.agent.max_rounds,
                )
                continue

            # --- resolve candidate references ---
            candidate_ids = extract_candidate_ids(
                llm,
                node_data["title"],
                pdf_ids,
                raw_refs,
                title_shortlist_size=settings.agent.title_shortlist_size,
            )
            current_canonical_id = _canonicalize_arxiv_id(arxiv_id)
            # filter current paper and duplicate versioned variants of the same paper
            new_ids = [
                c for c in candidate_ids
                if _canonicalize_arxiv_id(c) != current_canonical_id
            ]
            logger.info(
                "[round %s] %s produced %s candidate reference(s)",
                round_no,
                arxiv_id,
                len(new_ids),
            )

            # fetch metadata for new candidates
            candidates_meta: list[dict] = []
            for cid in new_ids:
                if not graph.has_node(cid):
                    try:
                        cp = fetch_metadata(cid)
                    except Exception:
                        state.skipped_refs.append(cid)
                        logger.warning(
                            "[round %s] Failed to fetch candidate metadata for %s",
                            round_no,
                            cid,
                        )
                        continue
                    child_node = PaperNode(
                        arxiv_id=cid,
                        title=cp.title,
                        authors=cp.authors,
                        abstract=cp.abstract,
                        year=cp.year,
                        status=NodeStatus.PENDING,
                        round_added=state.current_round + 1,
                    )
                    graph.add_node(child_node)
                    _persist_snapshot(state, "candidate_added", f"Added candidate node {cid}")
                    logger.info("[round %s] Added candidate node %s", round_no, cid)
                graph.add_edge(arxiv_id, cid)
                _persist_snapshot(state, "edge_added", f"Added citation edge {arxiv_id} -> {cid}")
                cd = graph.get_node(cid)
                candidates_meta.append({
                    "arxiv_id": cid,
                    "title": cd.get("title", ""),
                    "abstract": cd.get("abstract", ""),
                })

            # --- rank top-k ---
            if candidates_meta:
                current_summary = (
                    f"{node_data.get('title', '')}\n"
                    f"Method: {summary.method}\n"
                    f"Problem: {summary.problem_solved}"
                )
                try:
                    logger.info(
                        "[round %s] Ranking %s candidate(s) for %s, selecting top %s",
                        round_no,
                        len(candidates_meta),
                        arxiv_id,
                        settings.agent.top_k,
                    )
                    top_ids = rank_candidates(
                        llm,
                        current_summary,
                        candidates_meta,
                        settings.agent.top_k,
                    )
                    if not top_ids:
                        logger.warning(
                            "[round %s] Ranking returned no papers for %s, using first %s candidate(s)",
                            round_no,
                            arxiv_id,
                            settings.agent.top_k,
                        )
                        top_ids = [
                            c["arxiv_id"] for c in candidates_meta[: settings.agent.top_k]
                        ]
                    next_frontier.extend(top_ids)
                    logger.info(
                        "[round %s] Selected next papers from %s: %s",
                        round_no,
                        arxiv_id,
                        ", ".join(top_ids) if top_ids else "none",
                    )
                except Exception:
                    # fallback: take first top_k
                    logger.exception(
                        "[round %s] Ranking failed for %s, using first %s candidate(s)",
                        round_no,
                        arxiv_id,
                        settings.agent.top_k,
                    )
                    next_frontier.extend(
                        [c["arxiv_id"] for c in candidates_meta[: settings.agent.top_k]]
                    )
            else:
                logger.info("[round %s] No candidates available for %s", round_no, arxiv_id)

        # deduplicate frontier, skip already-analyzed nodes
        seen_frontier: dict[str, None] = {}
        for fid in next_frontier:
            nd = graph.get_node(fid)
            if nd and nd.get("status") != NodeStatus.ANALYZED:
                seen_frontier[fid] = None
        state.frontier = list(seen_frontier.keys())
        state.current_round += 1
        _persist_snapshot(state, "round_completed", f"Completed round {round_no}")
        logger.info(
            "[round %s] Completed. Next frontier has %s paper(s)",
            round_no,
            len(state.frontier),
        )
        return state

    # ------------------------------------------------------------------ #
    # Node: evaluate best lineage                                         #
    # ------------------------------------------------------------------ #
    def evaluate(state: AgentState) -> AgentState:
        all_nodes = graph.all_nodes()
        _persist_snapshot(state, "evaluation_started", "Selecting best lineage")
        logger.info("[evaluate] Evaluating best lineage across %s node(s)", len(all_nodes))
        try:
            result = select_best_lineage(llm, all_nodes, graph.all_edges(), state.root_id)
            state.lineage_chain = result.get("chain", [state.root_id])
            state.lineage_rationale = result.get("rationale", "")
            _persist_snapshot(state, "evaluation_completed", "Selected lineage chain")
            logger.info(
                "[evaluate] Selected lineage length: %s",
                len(state.lineage_chain),
            )
        except Exception as e:
            state.lineage_chain = [state.root_id]
            state.lineage_rationale = f"Evaluation failed: {e}"
            _persist_snapshot(state, "evaluation_failed", "Lineage evaluation failed")
            logger.error("[evaluate] Lineage evaluation failed: %s", e)
        return state

    # ------------------------------------------------------------------ #
    # Node: generate report                                               #
    # ------------------------------------------------------------------ #
    def report(state: AgentState) -> AgentState:
        state.completed_at = time.time()
        state.elapsed_seconds = state.completed_at - state.started_at
        state.token_usage = llm.get_token_usage()
        all_nodes = graph.all_nodes()
        logger.info("[report] Rendering report for %s node(s)", len(all_nodes))
        content = render_report(
            root_id=state.root_id,
            chain=state.lineage_chain,
            rationale=state.lineage_rationale,
            nodes=all_nodes,
            skipped=state.skipped_refs,
            elapsed_seconds=state.elapsed_seconds,
            token_usage=state.token_usage,
        )
        path = save_report(content, settings.paths.outputs_dir, state.root_id)
        state.report_path = path
        logger.info("[report] Markdown report saved to %s", path)

        _persist_snapshot(state, "report_completed", f"Report saved to {path}")
        logger.info("[report] Graph snapshot saved to %s", state.graph_path)
        return state

    # ------------------------------------------------------------------ #
    # Routing                                                             #
    # ------------------------------------------------------------------ #
    def should_continue(state: AgentState) -> str:
        if state.error:
            logger.warning("[route] Encountered error, switching to evaluation: %s", state.error)
            return "evaluate"
        if not state.frontier:
            logger.info("[route] Frontier empty, switching to evaluation")
            return "evaluate"
        logger.info("[route] Continuing to next processing round")
        return "process"

    # ------------------------------------------------------------------ #
    # Build graph                                                         #
    # ------------------------------------------------------------------ #
    builder = StateGraph(AgentState)
    builder.add_node("ingest", ingest_root)
    builder.add_node("process", process_round)
    builder.add_node("evaluate", evaluate)
    builder.add_node("report", report)

    builder.set_entry_point("ingest")
    builder.add_conditional_edges(
        "ingest",
        lambda s: "evaluate" if s.error else "process",
        {"process": "process", "evaluate": "evaluate"},
    )
    builder.add_conditional_edges(
        "process",
        should_continue,
        {"process": "process", "evaluate": "evaluate"},
    )
    builder.add_edge("evaluate", "report")
    builder.add_edge("report", END)

    return builder.compile()
