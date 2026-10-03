"""Run the smart-factory layout constraint-learning benchmark."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
CASE = Path(__file__).resolve().parent
for path in (SRC, CASE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from layout_constraint_learner import SmartFactoryLayoutConstraintLearner  # noqa: E402
from layout_problem import generate_layout_data, hidden_operational_score, layout_features  # noqa: E402


def main() -> None:
    data = generate_layout_data(n_samples=5000, random_state=42)
    learner = SmartFactoryLayoutConstraintLearner(data, random_state=42).fit(tune=True)
    evaluation = learner.evaluate()
    result = learner.optimize(flow_scale=1.0, min_probability=0.60)

    selected_features = layout_features(result.assignment, flow_scale=1.0)
    true_score = hidden_operational_score(selected_features)

    print("Smart factory layout optimization with learned constraints")
    print(f"Historical operational-feasible share: {data['operational_feasible'].mean():.3f}")
    print(f"Balanced accuracy: {evaluation.balanced_accuracy:.3f}")
    print(f"ROC AUC: {evaluation.roc_auc:.3f}")
    print(f"Average precision: {evaluation.average_precision:.3f}")
    print(f"False-feasible rate: {evaluation.false_feasible_rate:.3f}")
    print()
    print("Selected layout cells (cutting, machining, assembly, shipping):")
    print(result.assignment)
    print(f"Material-handling cost: {result.material_handling_cost:.3f}")
    print(f"Learned feasibility probability: {result.feasibility_probability:.3f}")
    print(f"Synthetic hidden operational score: {true_score:.3f}")
    print(f"Safe candidates: {result.safe_candidates}/{result.candidates_evaluated}")


if __name__ == "__main__":
    main()
