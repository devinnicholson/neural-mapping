.PHONY: test smoke results-summary tree

test:
	python -m unittest discover -s tests

smoke:
	python scripts/generate_splits.py --frames examples/frames.txt --budgets 4 6 --val-count 2 --test-count 2 --scene example --seed 7 --output data/splits/example_split.json
	python scripts/compute_uncertainty_metrics.py --input examples/metric_input.json --bad-threshold 0.5 --output outputs/reports/example_metrics.json

results-summary:
	python scripts/summarize_active_metrics.py --input outputs/modal_metrics_latest.log --pairs-file configs/active_metric_pairs.json --format markdown

tree:
	find . -maxdepth 3 -type f | sort
