.PHONY: test smoke results-summary record report-assets verify-research paper tree

test:
	python -m pytest -q

smoke:
	python scripts/generate_splits.py --frames examples/frames.txt --budgets 4 6 --val-count 2 --test-count 2 --scene example --seed 7 --output data/splits/example_split.json
	python scripts/compute_uncertainty_metrics.py --input examples/metric_input.json --bad-threshold 0.5 --output outputs/reports/example_metrics.json

results-summary:
	python scripts/summarize_active_metrics.py --input outputs/modal_metrics_latest.log --pairs-file configs/active_metric_pairs.json --format markdown

record:
	python scripts/build_icl_benchmark_record.py \
		--artifact-root experiments/artifacts/icl_nuim_multitrajectory_v1 \
		--protocol experiments/protocols/icl_nuim_multitrajectory_v1.json \
		--run-manifest experiments/run_manifests/icl_nuim_multitrajectory_v1.json \
		--output experiments/records/icl_nuim_multitrajectory_v1.json

report-assets:
	python scripts/generate_icl_report_assets.py

verify-research: test record report-assets
	git diff --exit-code -- experiments/records paper/tables experiments/tables docs/icl_nuim_multitrajectory_v1.md

paper:
	cd paper && latexmk -pdf main.tex

tree:
	find . -maxdepth 3 -type f | sort
