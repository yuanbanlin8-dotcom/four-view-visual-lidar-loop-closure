# Public-release checks — 2026-10-30

- 25 existing unit checks passed; 3 GTSAM-dependent checks were skipped on the publication workstation.
- One additional path-override check passed (`research.multifloor.tests.test_public_paths`).
- NCD metrics were recomputed from the included candidates, poses and edges: R@1 30/31; R@5 31/31; 96/100 accepted edges spatially correct; 32/38 positive pairs covered; 0/5 opposite-heading positives covered.
- Final floor-height CSV arithmetic and the 26-loop floor distribution passed validation.
- Seven SVG files match the embedded SVG bytes in the final manuscript and supplement.
- Research/tool Python files passed syntax parsing. Bundled upstream ROS utilities include legacy Python 2 scripts and retain upstream compatibility requirements.
- A scan of tracked files found no GitHub-token, AWS access-key or private-key signatures.
- Full ROS sensor processing and GTSAM optimization were not run for this repository cleanup.

Sources: repository main commit `08db3b2bfc23d5c31a7b97b9f3e6d14a29906de1`, the latest manuscript table/figure assets, and the 24 s sequence-gap correction. This public repository starts with a clean history; earlier development files remain in private storage.
