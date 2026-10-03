"""Constraint-learning wrapper for the smart-factory layout benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from industrial_constraint_learning.tabular import (
    TabularConstraintEvaluation,
    TabularConstraintLearner,
)
from layout_problem import FEATURE_COLUMNS, candidate_frame, hard_feasible_assignment


@dataclass(frozen=True)
class LayoutOptimizationResult:
    assignment: tuple[int, int, int, int]
    material_handling_cost: float
    feasibility_probability: float
    candidates_evaluated: int
    safe_candidates: int


class SmartFactoryLayoutConstraintLearner:
    """Learn operational acceptability while preserving hard layout constraints."""

    def __init__(self, data: pd.DataFrame, *, random_state: int = 42) -> None:
        self.data = data.copy()
        self.random_state = int(random_state)
        self.learner = TabularConstraintLearner(
            self.data,
            FEATURE_COLUMNS,
            "operational_feasible",
            random_state=self.random_state,
        )

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return tuple(FEATURE_COLUMNS)

    @property
    def best_params_(self):
        return self.learner.best_params_

    def fit(self, *, tune: bool = True) -> "SmartFactoryLayoutConstraintLearner":
        self.learner.fit(tune=tune)
        return self

    def evaluate(self) -> TabularConstraintEvaluation:
        return self.learner.evaluate()

    def predict_probability(self, candidates: pd.DataFrame):
        return self.learner.predict_proba(candidates)[:, 1]

    def optimize(
        self,
        *,
        flow_scale: float = 1.0,
        min_probability: float = 0.50,
    ) -> LayoutOptimizationResult:
        candidates = candidate_frame(flow_scale=flow_scale)
        optimizer = self.learner.safe_optimizer(min_probability=min_probability)
        result = optimizer.optimize(
            candidates,
            objective=lambda frame: frame["material_handling_cost"],
            hard_constraint=lambda frame: frame["hard_layout_compliant"].astype(bool),
            maximize=False,
        )
        assignment = (
            int(result.point["cutting_cell"]),
            int(result.point["machining_cell"]),
            int(result.point["assembly_cell"]),
            int(result.point["shipping_cell"]),
        )
        if not hard_feasible_assignment(assignment):
            raise RuntimeError("optimizer returned a hard-infeasible layout")
        return LayoutOptimizationResult(
            assignment=assignment,
            material_handling_cost=float(result.objective_value),
            feasibility_probability=float(result.feasibility_probability),
            candidates_evaluated=int(result.candidates_evaluated),
            safe_candidates=int(result.safe_candidates),
        )
