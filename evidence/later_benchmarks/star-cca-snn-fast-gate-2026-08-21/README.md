# STAR-CCA-SNN Fast Template-Transport Gate

**Terminal decision: `STOP_STAR_ORACLE_CEILING_TOO_LOW`.**

This is a throwaway exploratory screen on the already-spent Nakanishi2015
subjects 1--3 partition. It is not confirmatory or paper evidence.

## Blocking gates

- P0 provenance and role isolation: **pass**
- P1 synthetic instrument: **pass**
- O0 oracle ceiling: **fail**
- T0 template diagnostics: **not_evaluated**
- T1 transported STAR: **not_evaluated**

## Oracle ceiling result

Equal-weighted O0 unseen accuracy was **0.304** (gate >= 0.800).
Genuine FBCCA unseen accuracy was **0.296**.

| Subject | O0 unseen mean |
| ---: | ---: |
| 1 | 0.273 |
| 2 | 0.260 |
| 3 | 0.380 |

## Consequence and boundary

The runner obeyed the registered sequential stop. Transported arms were
opened only if O0 passed. No SNN, QNN, cost, energy, new subject, new
window, new support label, hypernetwork, or rescue tuning was run.

Exact role identities and hashes, templates, score matrices, predictions,
metrics, gate checks, dependency versions, runtime, and RSS are in `report.json`.

## Validation

- Stored prediction-to-metric recomputation: **pass**
- JSON parse/finite check: **pass**
- Report JSON SHA256: `986b2769bfdfa83130d9760d7e66f65e79ffbe911f818cb73f0bb67920092476`
- Total wall time: 19.3 s.
