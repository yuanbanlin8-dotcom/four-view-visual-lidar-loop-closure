# Multilevel pipeline

Run from the repository root: `python -m research.multifloor.run --help`.

`retrieval.py` scores cyclic four-camera correspondences; `verify.py` applies visual/LiDAR checks; `sequence.py` applies route support with a 24 s query gap; `backend.py` implements spatial-anchor selection, GNC and floor factors. `floors.py` and `ground_fit.py` estimate floor hypotheses and ground observations.

See [the reproduction guide](../../REPRODUCIBILITY.md) for input formats, commands and experimental conditions.
