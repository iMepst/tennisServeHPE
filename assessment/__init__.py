"""Feasibility assessment of the serve pipeline.

Mostly synthetic, recording-free analyses of the measurement chain. The error
budget has four sources:

- E1 pose estimation  -> propagation.py (landmark noise, Monte Carlo). sigma is
  not measured: the clips carry no hand-annotated landmarks and the model card
  reports no pixel-level error, so it is swept over config.sigma_sweep.
- E2 projection       -> projection.py (monocular foreshortening)
- E3 event error      -> annotation.py (manual frame check)
- E4 definitional     -> NOT simulated

E4 is the gap between the estimator's keypoint definition and the anatomical
joint centres behind the reference values. Quantifying it would need a
three-dimensional joint-centre reference, so it stays an unquantified offset.

Q3 rests on the decidability criterion (decidability.py) and on the flags the
pipeline returns on the corpus (serve_pipeline/report.py).

run_meta.json records every parameter, so a run is reproducible.
"""
