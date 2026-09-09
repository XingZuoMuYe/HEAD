# Vendored Pluto inference subset

This directory contains the minimal UniTraj/Pluto model, feature builder, and
trajectory evaluator needed by HEAD's closed-loop Pluto policy. Training data,
caches, compiled extensions, and unrelated UniTraj models are intentionally
excluded. The source was integrated from the local `unitraj_benchmark-pluto`
checkout; its upstream license is preserved in `LICENSE`.

Official checkpoints are runtime artifacts and are not committed here. Place
`pluto_1M_aux_cil.ckpt` under
`artifacts/weights/imitation/pluto/` or override the configured checkpoint.
