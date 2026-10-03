# Constraint learning: taxonomy and manufacturing use

## Is constraint learning a machine-learning method?

**Constraint learning is better viewed as a problem class and integration pattern at the intersection of machine learning, operations research, and constraint programming than as one specific ML algorithm.**

In optimization with constraint learning, one or more constraints are unknown, implicit, expensive to evaluate, simulator-defined, or difficult to express analytically. Data from historical operation, experiments, simulations, or expert-labelled feasible/infeasible examples are used to learn an approximation of that constraint.

A typical formulation is

```text
minimize    f(x)

subject to  g_j(x) <= 0                     explicit / engineered constraints
            P(feasible | x) >= tau           learned operational constraint
            x in X
```

The learned component may be a classifier, regressor, probabilistic surrogate, neural network, decision tree, or another predictive model.

There are two closely related traditions:

1. **Constraint acquisition / constraint learning in constraint programming** — recovering symbolic constraint networks from positive/negative examples or interactive queries.
2. **Optimization with constraint learning / empirical model learning** — learning data-driven components of an optimization model and embedding or querying them during prescriptive optimization.

Therefore, saying that constraint learning is "an ML method" is directionally correct when the constraint is inferred from data, but technically imprecise: **it is an ML-enabled optimization methodology, not a single learning algorithm.**

## Why it is useful in manufacturing

Manufacturing systems often contain constraints that are real but not available as clean equations. Examples include:

- process windows that depend nonlinearly on temperature, speed, pressure, feed, tool wear, or material lot,
- combinations of settings that create unacceptable scrap or defect risk,
- machine-stability and vibration regions,
- throughput/WIP regimes that become unstable only under interacting loads,
- energy-quality trade-offs,
- maintenance or degradation states that reduce effective capacity,
- scheduling combinations that are technically possible but operationally unreliable,
- simulator-defined feasibility that is too expensive to evaluate inside every optimization iteration.

Constraint learning can convert these observations into an empirical feasible region that an optimizer can use.

A practical architecture is:

```text
plant / simulator / historical data
        |
        v
feasible-infeasible labels or continuous outcomes
        |
        v
constraint learner
        |
        +--> calibration / uncertainty / conformal safety layer
        |
        v
prescriptive optimizer
(MILP, CP-SAT, nonlinear optimization, candidate search, BO, etc.)
        |
        v
hard-constraint check + original-model audit
        |
        v
deployment / experiment / next data
```

The key design principle is to keep **validated deterministic constraints** separate from learned constraints. OEM limits, physical safety limits, regulations, contractual rules, precedence constraints, and certified capacity bounds should remain explicit whenever they are known.

## Relationship to GPR, Bayesian optimization, BNNs, and GNNs

These methods are complementary to constraint learning rather than alternatives to it.

| Method | Role in a constraint-learning system | Manufacturing fit |
|---|---|---|
| Gaussian Process Regression (GPR) | Probabilistic surrogate for an unknown constraint function; provides uncertainty | Small-data process optimization, expensive experiments, simulator-based feasibility |
| Bayesian Optimization (BO) | Sequentially chooses the next experiment using objective and constraint surrogates | Expensive process tuning, design of experiments, constrained parameter search |
| Bayesian Neural Network (BNN) | High-capacity nonlinear constraint model with epistemic uncertainty | Higher-dimensional sensor/process data where uncertainty-aware decisions matter |
| Graph Neural Network (GNN) | Learns relational or topology-dependent feasibility/performance | Job shops, production networks, machine interactions, routing, line balancing, supply networks |
| SVM / tree / boosting | Direct feasible-vs-infeasible region learning | Strong baselines for tabular industrial data |
| Conformal prediction | Distribution-free calibration layer for empirical risk control under stated assumptions | Screening learned feasible regions before downstream optimization |
| Online learning / drift detection | Updates or invalidates the learned constraint as the plant changes | Tool wear, product-mix change, material drift, changing operating regimes |

### GPR + constrained Bayesian optimization

For low-dimensional, expensive-to-evaluate manufacturing experiments, a Gaussian process is often one of the most natural constraint models:

```text
objective surrogate:   y_obj = GP_obj(x)
constraint surrogate:  y_con = GP_con(x)

choose next x using an acquisition function
subject to estimated feasibility probability
```

This is especially attractive when each physical experiment or high-fidelity simulation is expensive.

### BNNs

BNNs become attractive when the constraint boundary is strongly nonlinear and the feature space is too large for standard Gaussian processes. The important distinction is that the posterior predictive uncertainty should be validated before it is interpreted as operational risk.

### GNNs

GNNs are useful when feasibility depends on relationships rather than only on a flat feature vector. Examples include:

- operation-machine compatibility graphs,
- precedence graphs in scheduling,
- production-line topology,
- material-flow networks,
- multi-stage supply/manufacturing networks.

A GNN can predict a constraint, feasibility score, or cost on the graph; the resulting prediction can then be used by a downstream optimizer.

## When constraint learning is preferable

Constraint learning is particularly appropriate when:

- an explicit analytical constraint is unavailable,
- feasibility can be observed or simulated,
- the system is too complex for first-principles modeling alone,
- evaluating the true constraint is expensive,
- the learned region can be validated out of sample,
- false-feasible errors can be measured and controlled operationally.

It is less appropriate when a trustworthy physical or regulatory constraint is already known exactly. In that case, learning an approximation usually creates unnecessary model risk.

## Relationship to adjacent ML + optimization fields

Constraint learning sits near several other areas but should not be conflated with them:

- **Decision-focused learning:** trains predictive models for downstream decision quality.
- **Inverse optimization:** infers objectives, parameters, or sometimes constraints from observed decisions.
- **Learning-augmented optimization:** uses predictions to accelerate or improve classical algorithms while retaining algorithmic structure.
- **Reinforcement learning:** learns a policy through sequential interaction; constraints may be enforced separately or learned as part of safe RL.
- **Surrogate optimization:** replaces expensive objectives or constraints with cheaper learned approximations.
- **Physics-informed ML:** incorporates known physical structure into the learner rather than learning the feasible region from data alone.

For industrial optimization, these areas are often composable rather than mutually exclusive.

## Model-selection guide

Use the simplest model that captures the operational boundary and can be validated for the downstream decision.

| Situation | Good starting point |
|---|---|
| Small tabular feasible/infeasible dataset | SVM, decision tree, gradient boosting |
| Smooth low-dimensional expensive constraint | Gaussian process |
| Expensive sequential experiments | Constrained Bayesian optimization |
| High-dimensional nonlinear process data | Neural network or BNN |
| Network/topology-dependent state | GNN |
| Need direct MILP/CP-SAT embedding | Shallow tree, linear/piecewise-linear surrogate, solver-compatible NN |
| Nonstationary plant | Online adaptation + drift monitoring |
| Safety-sensitive learned screening | Calibration + conformal/risk-control layer + hard-constraint audit |

## References

- Fajemisin, A. O., Maragno, D., & den Hertog, D. (2024). *Optimization with constraint learning: A framework and survey*. European Journal of Operational Research, 314(1), 1-14. DOI: https://doi.org/10.1016/j.ejor.2023.04.041
- Lombardi, M., Milano, M., & Bartolini, A. (2017). *Empirical decision model learning*. Artificial Intelligence, 244, 343-367. DOI: https://doi.org/10.1016/j.artint.2016.01.005
- Bessiere, C., Koriche, F., Lazaar, N., & O'Sullivan, B. (2017). *Constraint acquisition*. Artificial Intelligence, 244, 315-342. DOI: https://doi.org/10.1016/j.artint.2015.08.001
- Prestwich, S. D., Freuder, E. C., O'Sullivan, B., et al. (2021). *Classifier-based constraint acquisition*. Annals of Mathematics and Artificial Intelligence, 89, 655-674.
