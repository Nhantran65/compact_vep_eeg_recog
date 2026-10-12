# Porting record

Source repository retained to preserve QNN reverse imports and recovery. No data deletion or Git reset was performed.

- Recovered 60 CSV-FEATURES files from immutable HEAD (59 epoch sources + dataset summary).
- Copied all SNN artifacts, including Git-ignored checkpoints, and original result JSON unchanged.
- Extracted only the ShallowConvNet class from QNN; class AST is identical, state dictionary keys unchanged. Replaced QNN runtime import with local backbone import and checkpoint defaults with baselines/checkpoints.
- Preserved five Shallow result snapshots from QNN without quantum models.
- Archived EEGPT extensions requiring external caches; core imports no longer require FM/QNN source trees.
- Added packaging metadata, inventory/verification scripts, documented recovered source order and reconstructed legacy folds with row metadata.
- Preserved original MATLAB preprocessing and notebook outputs as historical sources. Raw EDF/SET are not copied; no new preprocessing or training protocol is claimed.
- Kept subject-disjoint manifests and later research reports in evidence, separate from runnable historical pipeline. Their absolute old source paths remain provenance; they are not a new independent benchmark runner.
- Kept historical JSON config/checkpoint absolute paths verbatim. They are metadata only in the tested core commands.

source_manifest.json records original copy hashes. package_manifest.json records final installed hashes, excluding runs and caches. Inspect docs/validation.json and runs/reproduction/results for observed reproduction differences.
