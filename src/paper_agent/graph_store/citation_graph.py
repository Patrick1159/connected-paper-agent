from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx


class NodeStatus:
    PENDING = "pending"
    ANALYZED = "analyzed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PaperNode:
    arxiv_id: str
    title: str = ""
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    year: Optional[int] = None
    status: str = NodeStatus.PENDING
    round_added: int = 0
    # Analysis results
    core_idea: str = ""
    method: str = ""
    problem_solved: str = ""
    research_field: str = ""


class CitationGraph:
    """networkx DiGraph wrapper. Edges go from cited → citing (child → parent)."""

    def __init__(self):
        self._g: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def has_node(self, arxiv_id: str) -> bool:
        return self._g.has_node(arxiv_id)

    def add_node(self, node: PaperNode) -> None:
        self._g.add_node(node.arxiv_id, **asdict(node))

    def get_node(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        if not self._g.has_node(arxiv_id):
            return None
        return dict(self._g.nodes[arxiv_id])

    def update_node(self, arxiv_id: str, **kwargs) -> None:
        if not self._g.has_node(arxiv_id):
            raise KeyError(arxiv_id)
        self._g.nodes[arxiv_id].update(kwargs)

    def all_nodes(self) -> List[Dict[str, Any]]:
        return [dict(d) for _, d in self._g.nodes(data=True)]

    # ------------------------------------------------------------------
    # Edge operations  (parent cites child)
    # ------------------------------------------------------------------

    def add_edge(self, parent_id: str, child_id: str) -> None:
        """parent_id paper cites child_id paper."""
        self._g.add_edge(parent_id, child_id)

    def has_edge(self, parent_id: str, child_id: str) -> bool:
        return self._g.has_edge(parent_id, child_id)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return nx.node_link_data(self._g, edges="links")

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CitationGraph":
        with open(path) as f:
            data = json.load(f)
        g = cls()
        g._g = nx.node_link_graph(data, edges="links")
        return g

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def nodes_by_status(self, status: str) -> List[str]:
        return [
            n for n, d in self._g.nodes(data=True) if d.get("status") == status
        ]
