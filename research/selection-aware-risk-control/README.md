# Selection-Aware Risk Control

Implemented bounded research baseline, v0.1, 2026-10-01. This is a research extension inside the existing standalone repository, not a new umbrella.

Calibrate the ENTIRE choose-lowest-cost-eligible-candidate policy rather than treating a randomly sampled candidate's score as a post-selection guarantee. A frozen finite family of thresholds is evaluated on independent calibration episodes. Each threshold receives an exact one-sided Clopper-Pearson binomial risk upper bound with Bonferroni confidence delta/K. Any calibrated policy selected from the simultaneously certified set retains the stated familywise guarantee.

## Statistical contract

With IID calibration episodes, a fixed independently trained score function and a threshold family fixed before calibration, with probability at least 1-delta over the calibration sample the selected policy's unconditional episode violation probability is at most alpha. Dependence among candidates WITHIN an episode is allowed. When no candidate is eligible or no threshold is certified, the policy abstains; this benchmark assigns abstention zero violation loss and reports coverage separately.

This is NOT risk conditional on acceptance, a physical safety guarantee, a guarantee under distribution shift, or a guarantee for an adaptively expanded threshold family. Recalibration is required when the candidate-generation policy or episode distribution changes. A zero-risk abstention fallback must be justified operationally; this study does not establish it for a physical plant.

## Run

```bash
python -m pip install -r requirements.txt
python -m unittest checks -v
python study.py --output local-results.json
```

Core API: policy_losses(scores,costs,unsafe,threshold), calibrate(...,thresholds,alpha,delta). Inputs contain one full candidate set per episode. Training of the underlying classifier is deliberately separate.

Eight local checks passed. Executed smoke experiments use 600 calibration and 1500 disjoint test episodes, 20 fixed thresholds, alpha=0.1 and delta=0.05, separately for 10/100/1000 candidates. Committed results.json is a summary; running the script emits every candidate's calibration bound. No claim of superiority over a fixed threshold is made: coverage and risk are separate criteria.

Research lineage: https://www.gsb.stanford.edu/faculty-research/publications/learn-then-test-calibrating-predictive-algorithms-achieve-risk

Independent implementation. Existing repository licensing applies. Existing classification/conformal code is unchanged; connecting real teacher outputs to this API is a separate integration step.
