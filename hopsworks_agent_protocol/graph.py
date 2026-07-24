"""Extract an agent's structure graph for the chat panel's Graph tab.

LangChain/LangGraph compiled graphs expose ``.get_graph()`` returning a
drawable graph with ``.nodes`` and ``.edges``; LlamaIndex workflows and plain
dicts are accepted too. The result is a framework-neutral ``{nodes, edges}``
spec the panel renders with reactflow.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def to_graph_spec(graph: Any) -> dict[str, Any] | None:
    """Normalise a framework graph to ``{"nodes": [...], "edges": [...]}``.

    Accepts a compiled LangGraph/LangChain runnable (``.get_graph()`` is
    called), an already-drawable graph (``.nodes`` / ``.edges``), or a plain
    ``{"nodes", "edges"}`` dict. Returns None (best-effort) when the shape is
    unrecognised — the Graph capability is simply not advertised.
    """
    if graph is None:
        return None
    if isinstance(graph, dict) and "nodes" in graph and "edges" in graph:
        return graph

    drawable = graph
    get_graph = getattr(graph, "get_graph", None)
    if callable(get_graph):
        try:
            drawable = get_graph()
        except Exception:  # noqa: BLE001
            log.warning("Could not read the agent graph via get_graph()", exc_info=True)
            return None

    nodes_attr = getattr(drawable, "nodes", None)
    edges_attr = getattr(drawable, "edges", None)
    if nodes_attr is None or edges_attr is None:
        return None

    try:
        node_items = (
            nodes_attr.values() if isinstance(nodes_attr, dict) else nodes_attr
        )
        nodes = []
        for node in node_items:
            node_id = getattr(node, "id", None)
            if node_id is None:
                node_id = str(node)
            label = getattr(node, "name", None) or str(node_id)
            nodes.append({"id": str(node_id), "label": str(label)})

        edges = []
        for edge in edges_attr:
            source = getattr(edge, "source", None)
            target = getattr(edge, "target", None)
            if source is None or target is None:
                continue
            spec: dict[str, Any] = {"source": str(source), "target": str(target)}
            data = getattr(edge, "data", None)
            if data:
                spec["label"] = str(data)
            if getattr(edge, "conditional", False):
                spec["conditional"] = True
            edges.append(spec)
    except Exception:  # noqa: BLE001
        log.warning("Could not extract nodes/edges from the agent graph", exc_info=True)
        return None

    if not nodes:
        return None
    return {"nodes": nodes, "edges": edges}
