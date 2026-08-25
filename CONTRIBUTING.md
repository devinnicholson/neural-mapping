# Contributing

This repository treats experiment design, implementation, and reporting as one reviewable unit. Code changes are welcome, but a scientific claim is not complete until its provenance and failure modes are visible.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,numeric]'
pytest -q
```

GPU experiments use the pinned Modal image in `modal_app.py`. Keep local utilities importable without Torch so split, metric, and record audits remain inexpensive.

## Change requirements

- Add focused unit tests for math, split logic, data conversion, and record auditing.
- Do not alter a preregistered protocol after viewing outcomes. Corrections made before outcome inspection must be appended as dated amendments with the reason, affected run IDs, and replacement commit.
- Keep training, validation, and test frames disjoint. State the experimental unit; pixels from one image are not independent replicates.
- Preserve every completed prespecified run, including negative results and infrastructure failures. Never replace an unfavorable seed or trajectory.
- Record the code commit, split seed, training seed, dataset source, dependency versions, GPU type, run IDs, and artifact paths.
- New headline claims require an external dataset or fresh-scene replication, matched baselines, geometry outcomes where applicable, and uncertainty intervals or clearly labeled descriptive uncertainty.

## Pull requests

Explain the research question or engineering invariant, list the verification commands, and identify any claims or protocols affected. Generated model weights and datasets stay out of Git; commit compact machine-readable metrics, manifests, figures, and reconstruction instructions instead.
