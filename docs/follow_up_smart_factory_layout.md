# Follow-up Study: Smart Factory Layout Optimization with Learned Constraints

**Status:** Proposed research extension. This document is a project specification, not an implemented benchmark.

## Research question

Can a factory-layout optimizer reduce material-handling cost while using data-driven models to capture operational feasibility patterns that are difficult to encode exactly, without replacing validated spatial and safety rules?

## Problem definition

Choose machine/workcell locations and orientations to minimize material-flow cost while respecting:

- building boundaries and fixed infrastructure;
- machine footprints and minimum geometric clearances;
- fire, egress, aisle, maintenance-access and exclusion zones;
- process-flow precedence and adjacency requirements;
- learned operational constraints such as historically problematic congestion, access, supervision or interference patterns.

The key modeling rule is to keep known safety, regulatory and engineering requirements explicit. Learned constraints should represent residual operational structure, not substitute for hard rules.

## Recommended formulation

A practical benchmark should use a hybrid OR + constraint-learning architecture:

1. formulate the base layout problem as QAP, MILP, CP-SAT or a decomposition/heuristic model;
2. generate or collect feasible/infeasible layout examples;
3. encode each candidate layout with geometric and flow features, or with a graph representation of departments/machines and pairwise relations;
4. train a calibrated classifier or graph model for empirical operational feasibility;
5. evaluate false-feasible risk on held-out layouts;
6. use the learned model as a candidate-screening layer or distill it into a solver-compatible surrogate;
7. optimize material-handling cost subject to explicit hard constraints plus the validated learned-feasibility rule.

A Physics-Informed Neural Network is not the default choice here because factory layout is primarily a discrete/geometric optimization problem rather than a PDE-constrained physical system. Physics-informed structure is appropriate only if the benchmark explicitly couples layout to a physical field model such as heat, airflow, vibration or contamination.

## Candidate decision variables

Depending on the benchmark:

- discrete bay/cell assignment;
- continuous or grid-based `(x, y)` position;
- orientation;
- department adjacency;
- aisle allocation;
- material-flow path selection.

A first version should prefer discrete cells so exact feasibility checks remain transparent.

## Objective

Primary objective:

```text
minimize total material-handling cost
= sum(flow_ij * travel_distance_ij * handling_cost_ij)
```

Optional secondary terms:

- relocation cost from an existing layout;
- congestion proxy;
- line-side replenishment distance;
- change-management penalty;
- utility-connection cost.

## Baselines

The learned method should be compared against:

- explicit-constraint MILP/CP-SAT only;
- a transparent constructive or local-search heuristic;
- OR model + unconstrained learned score;
- OR model + calibrated learned constraint;
- OR model + risk-controlled/conformal screening when sample size permits.

The benchmark should not assume that the learned model improves the optimizer.

## Data design

A reproducible synthetic benchmark can create:

- fixed floor geometry;
- machine footprints and process flows;
- explicit safety/clearance constraints;
- hidden operational rules used only for benchmark evaluation;
- nominal and out-of-distribution layout families.

If real historical layouts are available, labels should distinguish "physically valid" from "operationally accepted" so the learner does not conflate hard engineering feasibility with expert preference.

## Evaluation

Report at minimum:

- material-handling objective;
- hard-constraint violation count;
- true feasible rate where synthetic ground truth is known;
- false-feasible and false-infeasible rates;
- learned-model calibration;
- accepted candidate-set size;
- solver/runtime overhead;
- improvement relative to the best non-learning baseline;
- performance under OOD floor geometry, demand mix and flow intensity.

For redesign claims, report measured wall-clock time and number of optimizer iterations. Do not claim "weeks to hours" without a documented industrial comparison.

## Acceptance criteria

A learned layout method should be considered useful only if:

1. every explicit hard safety and geometry rule remains satisfied;
2. false-feasible risk is reported and remains within a declared operational tolerance;
3. objective quality is competitive with strong non-learning baselines;
4. the learned layer adds value on held-out or shifted layouts, not only training examples;
5. solver/runtime overhead is justified by better operational feasibility or objective quality.

## Suggested implementation sequence

### Phase 1 — exact small fixture
Small discrete layout with exhaustive or exact verification.

### Phase 2 — learned operational feasibility
Add a hidden nonlinear operational rule and train a calibrated tabular/graph learner.

### Phase 3 — hybrid optimization
Screen or embed the learned constraint inside the OR search.

### Phase 4 — robustness
Evaluate layout changes under demand/flow shifts and modified floor geometry.

### Phase 5 — optional physical coupling
Only if justified, add airflow, heat, contamination or vibration fields and test a physics-informed surrogate.

## Relationship to this repository

This follow-up extends the repository from process, warehouse, scheduling and design-space constraint learning into **spatial facility design**. It is intentionally specified separately from the existing ten executable case studies so the current benchmark remains reproducible and feature-complete.
