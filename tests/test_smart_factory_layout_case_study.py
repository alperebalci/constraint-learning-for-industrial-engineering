"""Tests for smart-factory layout optimization with learned constraints."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "case_studies" / "11_smart_factory_layout"
SRC = ROOT / "src"
for path in (CASE_DIR, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from layout_constraint_learner import SmartFactoryLayoutConstraintLearner  # noqa: E402
from layout_problem import (  # noqa: E402
    candidate_frame,
    enumerate_hard_feasible_layouts,
    generate_layout_data,
    hard_feasible_assignment,
    hidden_operational_score,
    layout_features,
)


def test_layout_data_is_reproducible() -> None:
    a = generate_layout_data(n_samples=300, random_state=31)
    b = generate_layout_data(n_samples=300, random_state=31)
    assert a.equals(b)


def test_enumerated_layouts_satisfy_explicit_hard_constraints() -> None:
    layouts = enumerate_hard_feasible_layouts()
    assert len(layouts) >= 50
    assert all(hard_feasible_assignment(layout) for layout in layouts)


def test_operational_labels_match_hidden_benchmark_rule() -> None:
    data = generate_layout_data(n_samples=500, random_state=32)
    expected = (data["hidden_operational_score"] >= 0.0).astype(int)
    assert (data["operational_feasible"] == expected).all()


def test_layout_classes_are_non_degenerate() -> None:
    data = generate_layout_data(n_samples=2000, random_state=33)
    share = data["operational_feasible"].mean()
    assert 0.20 < share < 0.80


def test_layout_constraint_model_has_useful_held_out_performance() -> None:
    data = generate_layout_data(n_samples=3500, random_state=34)
    learner = SmartFactoryLayoutConstraintLearner(data, random_state=34).fit(tune=False)
    evaluation = learner.evaluate()
    assert evaluation.balanced_accuracy >= 0.80
    assert evaluation.roc_auc >= 0.90
    assert evaluation.average_precision >= 0.85
    assert evaluation.false_feasible_rate <= 0.25


def test_hybrid_optimizer_returns_hard_feasible_true_acceptable_layout() -> None:
    data = generate_layout_data(n_samples=4500, random_state=35)
    learner = SmartFactoryLayoutConstraintLearner(data, random_state=35).fit(tune=False)
    result = learner.optimize(flow_scale=1.0, min_probability=0.60)

    assert hard_feasible_assignment(result.assignment)
    assert result.feasibility_probability >= 0.60
    features = layout_features(result.assignment, flow_scale=1.0)
    assert hidden_operational_score(features) >= 0.0

    candidates = candidate_frame(flow_scale=1.0)
    assert result.candidates_evaluated == len(candidates)
    assert result.safe_candidates > 0
