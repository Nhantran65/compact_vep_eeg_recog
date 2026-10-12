# Paper 2 V2 primary-test report

This directory is the reviewed, reproducible synthesis of the one-time V2
seed-42 primary-test evaluation: 490 claims, 490 results, four audited recovery
records, and 490 independent metric recomputations. V3 remains validation-only
fixed-frontend context and did not access the test set.

## Headline results

- Three positive exact-control results pass the amended two-frontend plus Holm
  rule, all on MSS: LIF versus continuous (+0.475 pp), LIF prototype versus
  continuous prototype (+0.339 pp), and quantum kernel versus random features
  on the identical capped subset (+1.942 pp).
- Spike-rate RBF is worse than continuous-stat RBF on MSS (-0.608 pp) after
  Holm correction.
- Current VEP and Human Infant have no Holm-confirmed exact-control effects.
- Same-budget MSS 5k exceeds 1k in 44/50 paired cells, averaging +0.752 pp.

The bootstrap p-value rule was locked after test access but before aggregate
outcomes were read. The resulting Holm statements are amended inference, not
fully preregistered confirmation. Quantum-kernel absolute scores are excluded
from full-test rankings because evaluation is capped at 25 examples per class.

## Files

- `report.html`: portable technical report, browser-verified at desktop and
  mobile widths.
- `summary.json`: concise machine-readable findings and audit inventory.
- `artifact.json`: canonical report artifact used to build the HTML.
- `analysis.sqlite`: bounded reviewed snapshot used by the report.
- `report.html.validation.json`: package and browser-verification receipt.

## Reproduce

```bash
env PYTHONPATH=paper2_benchmark/src .venv/bin/python \
  paper2_benchmark/scripts/build_v2_primary_test_report.py

node paper2_benchmark/scripts/package_v2_v3_validation_report.mjs \
  --input paper2_benchmark/reports/v2-primary-test-2026-08-14/artifact.json \
  --output paper2_benchmark/reports/v2-primary-test-2026-08-14/report.html
```
