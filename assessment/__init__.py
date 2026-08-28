"""Feasibility assessment of the serve pipeline.

Mostly synthetic, recording-free analyses of the measurement chain. The error
budget has four sources:

- E1 pose estimation  -> propagation.py (landmark noise, Monte Carlo). sigma is
  not measured; it is taken from the estimator's reported accuracy and swept
  over config.sigma_sweep, so the induced spread and the decidability verdict
  read as a function of noise level.
- E2 projection       -> projection.py (monocular foreshortening)
- E3 event error      -> annotation.py (manual frame check)
- E4 definitional     -> NOT simulated

E4 is the gap between surface landmarks and the reference joint centres. Left
as a documented, unquantified offset (worst on trunk inclination): quantifying
it would need joint-centre ground truth the work lacks.

Q3 is answered by the decidability criterion (decidability.py) alone.

Every analysis takes its parameters from the shared config and returns them in
its result, so a run is reproducible from its output alone.
"""
