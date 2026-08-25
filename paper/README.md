# Paper build

This directory is the workshop-paper source of record. Quantitative prose and tables must be generated from the audited JSON experiment record; do not hand-copy numbers from terminal output.

```bash
make verify-research
cd paper
latexmk -pdf main.tex
```

CI repeats the record/table audit and compiles `main.tex` in a pinned TeX Live
action. A paper build is not accepted if generated evidence drifts from the
committed record.

Before a release, verify that:

- the abstract states only conclusions supported by the frozen decision gates;
- every main-table number is traceable to a committed machine-readable record;
- the main comparison reports all four trajectories and both training seeds;
- negative results and the pre-outcome protocol amendment are visible;
- runtime, GPU, dependency, split, and artifact provenance appear in the appendix;
- the PDF builds without warnings about missing citations or references.
