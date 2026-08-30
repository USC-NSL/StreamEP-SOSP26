# Experiments

- Raw run artifacts live under `experiments/sgl-<id>/` or `experiments/amoe-<id>/`.
- Those run directories are git-ignored.
- Shared plotting / control scripts live directly under `experiments/` and stay tracked.
- Generated plots meant for review live under `experiments/plots/<id>/`.
- For SGLang recorder runs, export `SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR` to the raw artifact directory and use the HTTP workflow: `start` -> benchmark -> `stop` -> `dump`.
