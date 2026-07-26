"""Classification of covariates into the roles that decide how they are handled.

Not every variable should be controlled for. Adjusting for a confounder removes
bias; adjusting for a mediator removes part of the very effect being measured;
adjusting for a collider actively creates bias where none existed. This module
sorts covariates by their role in the causal graph so those mistakes are caught
before an estimate is produced rather than discovered afterwards.
"""

from __future__ import annotations

import networkx as nx

from discovery.dag_builder import CausalDAG

#: Roles a covariate can play, with the action each one implies.
ROLE_GUIDANCE: dict[str, str] = {
    "confounder": "Adjust for this. It causes both the treatment and the outcome, so leaving it out biases the estimate.",
    "mediator": "Do not adjust for this. It sits on the causal path, so controlling for it removes part of the effect you are trying to measure.",
    "collider": "Do not adjust for this. Both the treatment and the outcome cause it, so controlling for it creates a spurious association.",
    "instrument": "Do not adjust for this. It affects the treatment only, which makes it useful as an instrument rather than a control.",
    "outcome_predictor": "Optional. It affects the outcome but not the treatment, so adjusting for it reduces variance without changing the estimate.",
    "irrelevant": "Safe to ignore. It has no causal connection to either the treatment or the outcome.",
}


def identify_backdoor_set(dag: CausalDAG, treatment: str, outcome: str) -> dict:
    """Find a set of variables that closes every backdoor path.

    A backdoor path is a non-causal route between treatment and outcome that runs
    against the direction of causation, which is how confounding leaks in.
    Blocking all of them makes the causal effect identifiable by adjustment.

    The graph's non-descendant parents of the treatment are used as the candidate
    adjustment set, which is the standard sufficient set whenever it blocks all
    backdoor paths.

    Args:
        dag: The causal graph.
        treatment: Name of the treatment variable.
        outcome: Name of the outcome variable.

    Returns:
        Dictionary with the proposed ``backdoor_set``, whether it
        ``is_sufficient``, any ``open_paths`` that remain, and a ``reason``.
    """
    graph = dag.graph
    if treatment not in graph or outcome not in graph:
        return {
            "backdoor_set": [],
            "is_sufficient": False,
            "open_paths": [],
            "reason": "Treatment or outcome is not present in the graph.",
        }

    descendants_of_treatment = dag.descendants(treatment)
    # Conditioning on a descendant of the treatment blocks part of the effect, so
    # the candidate set is restricted to the treatment's non-descendant parents.
    candidates = [p for p in dag.parents(treatment) if p not in descendants_of_treatment]

    open_paths = _open_backdoor_paths(dag, treatment, outcome, set(candidates))
    is_sufficient = not open_paths

    if is_sufficient and candidates:
        reason = (
            f"Adjusting for {', '.join(sorted(candidates))} closes every backdoor "
            "path, so the effect is identifiable from this data."
        )
    elif is_sufficient:
        reason = (
            "No backdoor paths exist, so the treatment is effectively randomised "
            "with respect to the outcome and no adjustment is needed."
        )
    else:
        reason = (
            f"{len(open_paths)} backdoor path(s) remain open, so the effect is not "
            "identifiable by adjustment alone. An instrument or a different design "
            "is required."
        )

    return {
        "backdoor_set": sorted(candidates),
        "is_sufficient": is_sufficient,
        "open_paths": open_paths,
        "reason": reason,
    }


def _open_backdoor_paths(
    dag: CausalDAG, treatment: str, outcome: str, adjustment_set: set[str]
) -> list[list[str]]:
    """Return backdoor paths that remain unblocked by ``adjustment_set``."""
    undirected = dag.graph.to_undirected()
    if treatment not in undirected or outcome not in undirected:
        return []

    open_paths: list[list[str]] = []
    for path in nx.all_simple_paths(undirected, treatment, outcome):
        # A backdoor path starts with an edge pointing into the treatment.
        if not dag.graph.has_edge(path[1], treatment):
            continue
        if not _path_is_blocked(dag, path, adjustment_set):
            open_paths.append(list(path))
    return open_paths


def _path_is_blocked(dag: CausalDAG, path: list[str], adjustment_set: set[str]) -> bool:
    """Determine whether conditioning on ``adjustment_set`` blocks ``path``.

    A path is blocked when any non-collider on it is conditioned on, or when a
    collider on it is not conditioned on and neither are its descendants.
    """
    for i in range(1, len(path) - 1):
        previous, node, following = path[i - 1], path[i], path[i + 1]
        is_collider = dag.graph.has_edge(previous, node) and dag.graph.has_edge(following, node)

        if is_collider:
            # Colliders block by default, and open when conditioned on.
            conditioned = node in adjustment_set or bool(
                dag.descendants(node) & adjustment_set
            )
            if not conditioned:
                return True
        elif node in adjustment_set:
            return True
    return False


def classify_covariates(
    dag: CausalDAG,
    treatment: str,
    outcome: str,
    covariates: list[str],
) -> dict:
    """Sort covariates into causal roles and say what to do with each.

    Args:
        dag: The causal graph encoding the analyst's assumptions.
        treatment: Name of the treatment variable.
        outcome: Name of the outcome variable.
        covariates: Variables to classify.

    Returns:
        Dictionary with ``roles`` mapping each covariate to its role and
        guidance, ``by_role`` grouping names under each role, the
        ``safe_adjustment_set`` that may be controlled for, and ``warnings``
        naming any variable that would damage the analysis if adjusted for.
    """
    roles: dict[str, dict] = {}

    for covariate in covariates:
        if covariate not in dag.graph:
            role = "irrelevant"
        else:
            affects_treatment = _has_path(dag, covariate, treatment)
            # Every cause of the treatment also reaches the outcome through it, so
            # "affects the outcome" only distinguishes roles when it means a path
            # that does not run through the treatment.
            affects_outcome = _has_path_avoiding(dag, covariate, outcome, avoid=treatment)
            caused_by_treatment = covariate in dag.descendants(treatment)
            caused_by_outcome = covariate in dag.descendants(outcome)

            if caused_by_treatment and caused_by_outcome:
                role = "collider"
            elif caused_by_treatment and affects_outcome:
                role = "mediator"
            elif caused_by_treatment:
                # Downstream of treatment but not on a path to the outcome.
                role = "collider" if caused_by_outcome else "mediator"
            elif affects_treatment and affects_outcome:
                role = "confounder"
            elif affects_treatment:
                role = "instrument"
            elif affects_outcome:
                role = "outcome_predictor"
            else:
                role = "irrelevant"

        roles[covariate] = {"role": role, "guidance": ROLE_GUIDANCE[role]}

    by_role: dict[str, list[str]] = {}
    for name, info in roles.items():
        by_role.setdefault(info["role"], []).append(name)

    safe_adjustment_set = sorted(
        by_role.get("confounder", []) + by_role.get("outcome_predictor", [])
    )

    warnings: list[str] = []
    for name in by_role.get("mediator", []):
        warnings.append(
            f"'{name}' is a mediator: adjusting for it would remove part of the "
            "treatment effect and understate the result."
        )
    for name in by_role.get("collider", []):
        warnings.append(
            f"'{name}' is a collider: adjusting for it would create an association "
            "that does not exist in the data."
        )

    return {
        "roles": roles,
        "by_role": by_role,
        "safe_adjustment_set": safe_adjustment_set,
        "warnings": warnings,
    }


def _has_path(dag: CausalDAG, source: str, target: str) -> bool:
    """Return whether a directed causal path runs from ``source`` to ``target``."""
    if source not in dag.graph or target not in dag.graph:
        return False
    return nx.has_path(dag.graph, source, target)


def _has_path_avoiding(dag: CausalDAG, source: str, target: str, avoid: str) -> bool:
    """Return whether ``source`` reaches ``target`` without passing through ``avoid``.

    Used to separate a genuine direct influence on the outcome from one that only
    exists because the variable influences the treatment, which every instrument
    does by definition.
    """
    graph = dag.graph
    if source not in graph or target not in graph or source == avoid:
        return False
    reduced = graph.subgraph([n for n in graph.nodes if n != avoid])
    if source not in reduced or target not in reduced:
        return False
    return nx.has_path(reduced, source, target)
