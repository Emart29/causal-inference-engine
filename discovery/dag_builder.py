"""Construction and validation of the causal graph an analysis is identified under.

A causal graph is not discovered from data — it encodes assumptions the analyst
is making about how the world works. Every estimate this engine produces is
conditional on the graph being right, so the graph is stored alongside the
results and shown to the reader rather than hidden inside the pipeline.
"""

from __future__ import annotations

import networkx as nx


class CausalDAG:
    """A directed acyclic graph over the variables in an analysis.

    Nodes are column names; an edge ``a -> b`` asserts that ``a`` is a direct
    cause of ``b``. Cycles are permitted during editing so a user can rearrange
    edges freely, but :meth:`validate` must pass before the graph is used for
    identification.
    """

    def __init__(
        self,
        nodes: list[str] | None = None,
        edges: list[tuple[str, str]] | None = None,
    ) -> None:
        self._graph = nx.DiGraph()
        if nodes:
            self._graph.add_nodes_from(nodes)
        if edges:
            for cause, effect in edges:
                self.add_edge(cause, effect)

    @property
    def graph(self) -> nx.DiGraph:
        """The underlying NetworkX graph."""
        return self._graph

    @property
    def nodes(self) -> list[str]:
        """All variable names in the graph."""
        return list(self._graph.nodes)

    @property
    def edges(self) -> list[tuple[str, str]]:
        """All causal edges as ``(cause, effect)`` pairs."""
        return list(self._graph.edges)

    def add_edge(self, cause: str, effect: str) -> None:
        """Assert that ``cause`` directly causes ``effect``.

        Raises:
            ValueError: If ``cause`` and ``effect`` are the same variable, since
                nothing can be its own direct cause.
        """
        if cause == effect:
            raise ValueError(f"'{cause}' cannot cause itself")
        self._graph.add_edge(cause, effect)

    def remove_edge(self, cause: str, effect: str) -> None:
        """Remove a causal edge if it is present."""
        if self._graph.has_edge(cause, effect):
            self._graph.remove_edge(cause, effect)

    def parents(self, node: str) -> list[str]:
        """Direct causes of ``node``."""
        return list(self._graph.predecessors(node)) if node in self._graph else []

    def children(self, node: str) -> list[str]:
        """Direct effects of ``node``."""
        return list(self._graph.successors(node)) if node in self._graph else []

    def ancestors(self, node: str) -> set[str]:
        """Every variable with a directed path into ``node``."""
        return nx.ancestors(self._graph, node) if node in self._graph else set()

    def descendants(self, node: str) -> set[str]:
        """Every variable reachable from ``node`` along directed edges."""
        return nx.descendants(self._graph, node) if node in self._graph else set()

    def causal_paths(self, treatment: str, outcome: str) -> list[list[str]]:
        """All directed paths carrying the causal effect from treatment to outcome."""
        if treatment not in self._graph or outcome not in self._graph:
            return []
        return [list(p) for p in nx.all_simple_paths(self._graph, treatment, outcome)]

    def validate(self) -> dict:
        """Check that the graph is usable for causal identification.

        Returns:
            Dictionary with ``is_dag`` (whether the graph is acyclic), any
            ``cycles`` found, ``isolated_nodes`` that connect to nothing, and a
            human-readable ``message``.
        """
        is_dag = nx.is_directed_acyclic_graph(self._graph)
        cycles: list[list[str]] = []
        if not is_dag:
            cycles = [list(c) for c in nx.simple_cycles(self._graph)]
        isolated = [n for n in self._graph.nodes if self._graph.degree(n) == 0]

        if not is_dag:
            message = (
                "The graph contains a cycle, so it cannot be a causal model: a "
                "variable would have to cause itself through a chain of effects."
            )
        elif isolated:
            message = (
                f"{len(isolated)} variable(s) are not connected to anything and "
                "will not influence the analysis."
            )
        else:
            message = "The graph is a valid causal model."

        return {
            "is_dag": is_dag,
            "cycles": cycles,
            "isolated_nodes": isolated,
            "message": message,
        }

    def to_dot(self) -> str:
        """Render the graph in DOT format, which causal libraries accept directly."""
        lines = ["digraph {"]
        for node in self._graph.nodes:
            lines.append(f'  "{node}";')
        for cause, effect in self._graph.edges:
            lines.append(f'  "{cause}" -> "{effect}";')
        lines.append("}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialise the graph for storage."""
        return {"nodes": self.nodes, "edges": [list(e) for e in self.edges]}

    @classmethod
    def from_dict(cls, data: dict) -> "CausalDAG":
        """Rebuild a graph previously serialised by :meth:`to_dict`."""
        edges = [tuple(e) for e in data.get("edges", [])]
        return cls(nodes=data.get("nodes", []), edges=edges)


def auto_dag(
    treatment: str,
    outcome: str,
    covariates: list[str],
    instrument: str | None = None,
) -> CausalDAG:
    """Build the standard adjustment graph for a set of variables.

    Every covariate is assumed to cause both the treatment and the outcome, and
    the treatment is assumed to cause the outcome. This encodes the conventional
    "adjust for all pre-treatment covariates" position.

    This is an assumption, not a discovery. Causal structure is not identifiable
    from observational data without further assumptions, so this function makes a
    defensible default explicit rather than pretending to have learned it. The
    user is expected to correct it: in particular, any covariate that is actually
    caused by the treatment is a mediator and must not be adjusted for.

    Args:
        treatment: Name of the treatment variable.
        outcome: Name of the outcome variable.
        covariates: Variables assumed to be common causes of both.
        instrument: Optional instrument, drawn as affecting treatment only.

    Returns:
        The assembled graph.
    """
    dag = CausalDAG(nodes=[treatment, outcome, *covariates])
    dag.add_edge(treatment, outcome)
    for covariate in covariates:
        dag.add_edge(covariate, treatment)
        dag.add_edge(covariate, outcome)
    if instrument:
        # An instrument affects the outcome only through the treatment. The
        # absence of an instrument -> outcome edge is the exclusion restriction.
        dag.add_edge(instrument, treatment)
    return dag
