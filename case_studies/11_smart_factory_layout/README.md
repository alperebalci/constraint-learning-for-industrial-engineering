# Smart Factory Layout Optimization with Learned Constraints

This executable case study combines a discrete facility-layout search with a learned operational-feasibility constraint.

Hard geometry and safety rules are always explicit. The classifier is trained only on hard-feasible historical layouts and learns a synthetic operational acceptability rule involving material flow, adjacency, central-zone loading and flow intensity.

Run:

```bash
python case_studies/11_smart_factory_layout/run_case_study.py
```

The benchmark reports held-out classification quality, false-feasible risk and the minimum material-handling-cost candidate that passes the learned probability threshold.
