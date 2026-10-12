# Paper 2 V2–V3 Validation Synthesis

This directory contains the reproducible, validation-only synthesis of the
completed Paper 2 V2 primary run and V3 fixed-frontend decomposition.

- `summary.json`: bounded analytical snapshot with data-quality gates,
  cross-dataset rankings, exact-twin deltas, scaling decomposition, and compute
  summaries.
- `artifact.json`: canonical Data Analytics technical-report input.
- `analysis.sqlite`: queryable copy of every bounded report row used for
  source-provenance inspection.
- `analysis.ipynb`: executable companion notebook and headline spot checks.
- `report.html`: self-contained portable report generated from `artifact.json`.
- `report.html.validation.json`: durable packaging and browser-QA receipt.

Regenerate the summary and canonical report input from the immutable caches:

```bash
PYTHONPATH=paper2_benchmark/src ./.venv/bin/python \
  paper2_benchmark/scripts/build_v2_v3_validation_synthesis.py
```

Execute the companion notebook:

```bash
./.venv/bin/python \
  paper2_benchmark/scripts/execute_v2_v3_validation_notebook.py
```

Package and browser-verify the report:

```bash
node paper2_benchmark/scripts/package_v2_v3_validation_report.mjs \
  --input paper2_benchmark/reports/v2-v3-validation-synthesis-2026-08-14/artifact.json \
  --output paper2_benchmark/reports/v2-v3-validation-synthesis-2026-08-14/report.html
```

The report is descriptive validation evidence. It contains no primary-test
result, p-value, multiplicity-adjusted inference, energy-efficiency claim, or
quantum-advantage claim.

The portable package passes canonical schema, provenance, payload equality,
static-chart extraction, source-dialog interaction, and browser QA at 1440 px
and 390 px. Its packaging script applies a narrow, audited presentation-only
override for the shared reader's `100vw` scrollbar overflow and mobile
negative-margin containment; it does not modify the analytical payload.
