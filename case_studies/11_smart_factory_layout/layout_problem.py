"""Discrete smart-factory layout benchmark with explicit hard constraints."""

from __future__ import annotations

from itertools import permutations
from typing import Iterable

import numpy as np
import pandas as pd


DEPARTMENTS = ("cutting", "machining", "assembly", "shipping")
CELLS = {
    0: (0, 0),
    1: (1, 0),
    2: (2, 0),
    3: (0, 1),
    4: (1, 1),
    5: (2, 1),
}
BASE_FLOWS = {
    (0, 1): 8.0,
    (1, 2): 10.0,
    (2, 3): 12.0,
    (0, 2): 3.0,
    (1, 3): 2.0,
}
FEATURE_COLUMNS = (
    "material_handling_cost",
    "critical_path_distance",
    "adjacency_breaks",
    "central_zone_count",
    "max_pair_distance",
    "flow_scale",
    "assembly_shipping_adjacent",
)


def manhattan(cell_a: int, cell_b: int) -> int:
    ax, ay = CELLS[int(cell_a)]
    bx, by = CELLS[int(cell_b)]
    return abs(ax - bx) + abs(ay - by)


def hard_feasible_assignment(assignment: Iterable[int]) -> bool:
    """Check deterministic geometry/safety rules.

    Rules intentionally stay explicit:
    - one department per cell;
    - shipping must be adjacent to the right-side dock;
    - machining and assembly cannot occupy the fire-exit cell 0;
    - cutting cannot occupy the dock-service cell 5.
    """
    assignment = tuple(int(x) for x in assignment)
    if len(assignment) != len(DEPARTMENTS):
        return False
    if any(cell not in CELLS for cell in assignment):
        return False
    if len(set(assignment)) != len(assignment):
        return False

    cutting, machining, assembly, shipping = assignment
    if shipping not in {2, 5}:
        return False
    if machining == 0 or assembly == 0:
        return False
    if cutting == 5:
        return False
    return True


def enumerate_hard_feasible_layouts() -> list[tuple[int, ...]]:
    return [
        tuple(int(x) for x in assignment)
        for assignment in permutations(CELLS, len(DEPARTMENTS))
        if hard_feasible_assignment(assignment)
    ]


def layout_features(
    assignment: Iterable[int],
    *,
    flow_scale: float = 1.0,
) -> dict[str, float]:
    assignment = tuple(int(x) for x in assignment)
    if not hard_feasible_assignment(assignment):
        raise ValueError("assignment violates explicit hard layout constraints")
    if flow_scale <= 0:
        raise ValueError("flow_scale must be positive")

    handling_cost = sum(
        flow * float(flow_scale) * manhattan(assignment[i], assignment[j])
        for (i, j), flow in BASE_FLOWS.items()
    )
    chain_distances = (
        manhattan(assignment[0], assignment[1]),
        manhattan(assignment[1], assignment[2]),
        manhattan(assignment[2], assignment[3]),
    )
    pair_distances = [
        manhattan(assignment[i], assignment[j])
        for i in range(len(assignment))
        for j in range(i + 1, len(assignment))
    ]
    return {
        "material_handling_cost": float(handling_cost),
        "critical_path_distance": float(sum(chain_distances)),
        "adjacency_breaks": float(sum(distance > 1 for distance in chain_distances)),
        "central_zone_count": float(sum(cell in {1, 4} for cell in assignment)),
        "max_pair_distance": float(max(pair_distances)),
        "flow_scale": float(flow_scale),
        "assembly_shipping_adjacent": float(
            manhattan(assignment[2], assignment[3]) == 1
        ),
    }


def hidden_operational_score(
    features: dict[str, float] | pd.Series,
) -> float:
    """Synthetic operational acceptability used only as benchmark ground truth."""
    cost = float(features["material_handling_cost"])
    breaks = float(features["adjacency_breaks"])
    central = float(features["central_zone_count"])
    scale = float(features["flow_scale"])
    adjacent = float(features["assembly_shipping_adjacent"])
    max_pair = float(features["max_pair_distance"])

    return float(
        3.8
        - 0.045 * cost
        - 0.45 * breaks
        - 0.35 * central * scale
        + 0.35 * adjacent
        - 0.15 * (max_pair - 2.0) ** 2
    )


def candidate_frame(*, flow_scale: float = 1.0) -> pd.DataFrame:
    rows = []
    for assignment in enumerate_hard_feasible_layouts():
        features = layout_features(assignment, flow_scale=flow_scale)
        score = hidden_operational_score(features)
        row = {
            **features,
            "cutting_cell": assignment[0],
            "machining_cell": assignment[1],
            "assembly_cell": assignment[2],
            "shipping_cell": assignment[3],
            "hard_layout_compliant": 1,
            "hidden_operational_score": score,
            "operational_feasible": int(score >= 0.0),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def generate_layout_data(
    n_samples: int = 4000,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate historical hard-feasible layouts under varying flow intensity."""
    if n_samples < 50:
        raise ValueError("n_samples must be at least 50")

    rng = np.random.default_rng(random_state)
    layouts = enumerate_hard_feasible_layouts()
    rows = []
    for _ in range(n_samples):
        assignment = layouts[int(rng.integers(0, len(layouts)))]
        flow_scale = float(rng.uniform(0.75, 1.35))
        features = layout_features(assignment, flow_scale=flow_scale)
        score = hidden_operational_score(features)
        rows.append(
            {
                **features,
                "cutting_cell": assignment[0],
                "machining_cell": assignment[1],
                "assembly_cell": assignment[2],
                "shipping_cell": assignment[3],
                "hard_layout_compliant": 1,
                "hidden_operational_score": score,
                "operational_feasible": int(score >= 0.0),
            }
        )
    return pd.DataFrame(rows)
