from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snn_vep.probes import run_embedding_probe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SNN embedding + SVC probes.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "snn_embedding_probe.yaml"))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trainfold-dir", default="trainfold")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--checkpoints-dir", default=str(ROOT / "artifacts" / "checkpoints"))
    parser.add_argument("--embeddings-dir", default=str(ROOT / "artifacts" / "embeddings"))
    args = parser.parse_args()

    result = run_embedding_probe(
        args.config,
        fold=args.fold,
        checkpoint=args.checkpoint,
        trainfold_dir=args.trainfold_dir,
        results_dir=args.results_dir,
        checkpoints_dir=args.checkpoints_dir,
        embeddings_dir=args.embeddings_dir,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "name": result["name"],
                "fold": result["fold"],
                "embedding_shape": result["embedding_shape"],
                "heads": [
                    {
                        "name": head["name"],
                        "test_metrics": head["test_metrics"],
                    }
                    for head in result["heads"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
