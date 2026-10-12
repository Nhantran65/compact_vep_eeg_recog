# Current VEP Manifest

This directory locks the existing Current VEP preprocessed windows without
modifying the raw EDF, EEGLAB `.set`, or `CSV-FEATURES` payloads.

Identity authority:

- `Participant_info.xlsx` assigns one global `Sub N` identifier to each of the
  32 participants and lists the two recording phases assigned to that person.
- The pinned public dataset record is version 2, DOI `10.17632/g9shp2gxhy.2`.
- `EEGLAB-SET_STEP3_EPOCHED/epoching_report.csv` is hashed and supplies the
  authoritative preprocessing success/failure state.
- Raw JSON `recordId` values identify individual recordings.

Tracked artifacts:

- `recordings.jsonl`: all 64 expected participant/phase recording identities,
  including explicit missing or failed preprocessing states;
- `trials.jsonl`: the 3,612 existing one-second windows and their immutable
  source CSV hashes and row identities;
- `audit.json`: counts, exclusions, limitations, and snapshot/manifest hashes;
- `splits/subject_disjoint_v1/`: five deterministic train/validation/test folds.

The label on each existing window is inherited from its legacy recording
folder. It is not a stimulus-marker label. All results using this manifest must
retain that limitation.

Regenerate from the repository root:

```bash
PYTHONPATH=paper2_benchmark/src ./.venv/bin/python paper2_benchmark/scripts/build_current_vep_manifest.py
```
