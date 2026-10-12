# RESONA-VEP spike — S1-R result (Amendment 1 retry)

Date: 2026-08-19
Plan: [docs/plans/active/paper2-resona-vep-spike.md](../../../docs/plans/active/paper2-resona-vep-spike.md), "Amendment 1"
Script: [paper2_benchmark/scripts/run_resona_vep_spike_s1r.py](../../scripts/run_resona_vep_spike_s1r.py)
Report: [s1r_report.json](s1r_report.json)

## Preflight P1 — reported, not gated

```
RF_RHO_TARGET: 0.995
tau: 0.779 s
-3dB bandwidth: 0.408 Hz
class spacing: 0.5 Hz
bandwidth / spacing ratio: 0.817
```

Ratio < 1: the resonator population's bandwidth is narrower than the class
spacing, so bandwidth alone does not obviously prevent resolving adjacent
classes. This rules out "the population is too broadband to work" as an
explanation for what follows, before any training ran.

## Gate G-fit — FAIL

```
threshold: train accuracy > 0.20 (chance = 0.083), BOTH arms, EVERY subject
```

| subject | A (RF) train | B (LIF) train | both > 0.20? |
|---|---|---|---|
| 1 | 0.208 | 0.108 | No (B fails) |
| 2 | 0.125 | 0.092 | No (both fail) |
| 3 | 0.267 | 0.217 | Yes |

2 of 3 subjects fail the precondition (subject 1 on arm B alone, subject 2
on both arms). G-fit requires every subject, both arms, so the overall
verdict is **fail**.

## Gate G-mech — not evaluated, per the locked decision rule

G-fit failing means the A-vs-B comparison is not interpretable and was not
computed as a decision-relevant number, exactly as the amendment specifies.

For transparency (not as a finding): in the raw per-subject data, arm A's
*test* accuracy exceeded arm B's on all 3 subjects (0.133 vs 0.050, 0.100
vs 0.083, 0.133 vs 0.100). This is recorded in `s1r_report.json` but **must
not be read as a resonance signal** — the decision rule is explicit that no
conclusion about RF vs LIF is licensed while G-fit fails, and a 3-subject
tally under these conditions carries no more information now than it did
in S1.

## What changed vs. S1, and what it did and didn't fix

Both of the amendment's changes worked as implemented and are confirmed
directly, not assumed:

- **Spatial stage**: `nn.Linear(8 -> K, bias=False)` (implemented via an
  equivalent `einsum` for a training-speed reason unrelated to correctness,
  see the performance note below) applied identically to both arms before
  the spike dynamics. Parameter counts: arm A 1068, arm B 1020 (residual
  difference 48 = RF's `w`+`b` per neuron vs. LIF's `beta` alone).
- **Learnable beta for arm B**: confirmed moving from its 0.90 init (e.g.
  subject 1's arm B beta spread was 0.883-0.912 after only 5 diagnostic
  epochs), so arm B does carry learnable dynamics now, closing the
  matched-budget gap the amendment identified.

Both changes measurably helped relative to S1 (train accuracy rose from
"stuck at chance on every subject" to "above 0.20 on 4 of 6 subject-arm
pairs, with two of those clearing 0.21-0.27"), but not far enough, and not
consistently enough across subjects, to clear the precondition the
amendment itself set. Subject 2 in particular looks like a genuinely harder
subject by every method tried so far, including the untrained classical
baseline (FBCCA: subject 1 = 0.750, subject 2 = 0.617, subject 3 = 0.983 —
the same ranking across S1, S1-R, and the S1 linear-baseline diagnostic).

## A training-speed bug caught and fixed before the gated run

Not a correctness issue, but worth recording: the first implementation of
the spatial stage (`permute -> nn.Linear -> permute`) made training ~50x
slower (~30 s/batch instead of ~0.5 s/batch), because the RF/LIF population
loop re-slices its input 1063 times per forward call, and PyTorch's
backward for repeated indexing into the *same* grad-requiring tensor
rebuilds a full-sized gradient buffer on every slice. This never showed up
in S0 or S1 because their population input never required grad (it was raw
data, not the output of a learnable layer), so the expensive backward path
was never exercised. Fixed by unbinding the time dimension once
(`torch.unbind`) instead of re-slicing in the loop — confirmed by a direct
timing test (33.6 s -> 0.09 s for an isolated repro) before spending the
run's time budget on it. Produces bit-identical results; applied to the
shared `RFPopulation`/`LIFPopulation` code in `run_resona_vep_spike_s0.py`
as well as S1-R's own `LearnableBetaLIFPopulation`.

## Decision rule outcome

Per the amendment's locked decision rule:

```
G-fit FAIL -> The resonance question remains UNTESTED after two attempts
              (S1 and S1-R). Report the measured train accuracies and P1.
              Bring back for a human decision. Do NOT conclude anything
              about RF vs LIF, and do NOT start a third variant inside
              this plan.
```

This is where the spike stops. Two attempts (S1, S1-R) have now failed to
produce a harness that reliably fits real SSVEP EEG across all 3 subjects
at the locked training budget (100 epochs, Adam lr 1e-3, batch 32), even
after fixing the split bug, the amplitude-scale bug, and the missing
spatial-mixing stage. Per the amendment's own "still forbidden" list, no
second retry (`S1-R2`) is authorized inside this plan; a third attempt
would need a fresh, explicit, human-authorized decision, the same way this
retry did.

```
real_eeg_touched_in_S0: false
subjects_used_S1R: 3
moabb_dataset: Nakanishi2015
tsinghua_benchmark_or_beta_accessed: false
throwaway_spike: true
wall_seconds_total: 1358.1
```
