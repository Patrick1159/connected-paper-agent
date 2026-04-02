from __future__ import annotations

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
        state.frontier = [arxiv_id]
        logger.info("[ingest] Root paper ready: %s | title=%s", arxiv_id, paper.title)
        return state

    # ------------------------------------------------------------------ #
    # Node: process frontier (one round)                                  #
    # ------------------------------------------------------------------ #
    def process_round(state: AgentState) -> AgentState:
        round_no = state.current_round + 1
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
                logger.exception("[round %s] Failed to fetch metadata for %s", round_no, arxiv_id)
                continue

            # update node with fetched metadata (title/abstract may now be available)
            graph.update_node(arxiv_id, title=paper.title, abstract=paper.abstract)
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
                logger.info("[round %s] Summarization completed for %s", round_no, arxiv_id)
            except Exception as e:
                graph.update_node(arxiv_id, status=NodeStatus.FAILED)
                state.skipped_refs.append(f"{arxiv_id} (summarization failed: {e})")
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
            candidate_ids = extract_candidate_ids(llm, pdf_ids, raw_refs)
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
                    logger.info("[round %s] Added candidate node %s", round_no, cid)
                graph.add_edge(arxiv_id, cid)
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
        logger.info("[evaluate] Evaluating best lineage across %s node(s)", len(all_nodes))
        try:
            result = select_best_lineage(llm, all_nodes, graph.all_edges(), state.root_id)
            state.lineage_chain = result.get("chain", [state.root_id])
            state.lineage_rationale = result.get("rationale", "")
            logger.info(
                "[evaluate] Selected lineage length: %s",
                len(state.lineage_chain),
            )
        except Exception as e:
            state.lineage_chain = [state.root_id]
            state.lineage_rationale = f"Evaluation failed: {e}"
            logger.error("[evaluate] Lineage evaluation failed: %s", e)
        return state

    # ------------------------------------------------------------------ #
    # Node: generate report                                               #
    # ------------------------------------------------------------------ #
    def report(state: AgentState) -> AgentState:
        state.completed_at = time.time()
        state.elapsed_seconds = state.completed_at - state.started_at
        all_nodes = graph.all_nodes()
        logger.info("[report] Rendering report for %s node(s)", len(all_nodes))
        content = render_report(
            root_id=state.root_id,
            chain=state.lineage_chain,
            rationale=state.lineage_rationale,
            nodes=all_nodes,
            skipped=state.skipped_refs,
            elapsed_seconds=state.elapsed_seconds,
        )
        path = save_report(content, settings.paths.outputs_dir, state.root_id)
        state.report_path = path
        logger.info("[report] Markdown report saved to %s", path)

        # Also save graph snapshot
        import os as _os
        snap_dir = settings.paths.data_dir
        _os.makedirs(snap_dir, exist_ok=True)
        graph_path = f"{snap_dir}/{state.root_id.replace('/', '_')}_graph.json"
        graph.save(graph_path)
        logger.info("[report] Graph snapshot saved to %s", graph_path)
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
