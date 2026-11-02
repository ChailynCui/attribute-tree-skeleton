"""User-facing commands. Import GPU dependencies only for model operations."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from .config import load_config
from .io import read_json, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="tree-skeleton",
        description="Attribute-conditioned explicit 3D tree skeleton experiments",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def configured(name, help):
        p = sub.add_parser(name, help=help)
        p.add_argument("--config", required=True)
        p.add_argument("--data-root")
        return p

    p = configured(
        "verify-data", "Validate external geometry, labels, frozen splits and file hashes"
    )
    p.add_argument("--output")
    p = sub.add_parser(
        "prepare", help="Canonicalize external source pickles and fit train-only attributes"
    )
    p.add_argument("--sources", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--definition", default=str(Path(__file__).parent / "attribute_definition.json"))
    p.add_argument("--split")
    p.add_argument("--scale-fit", choices=["train", "all"], default="train")
    p.add_argument("--scale-target", type=float, default=0.99)
    p = configured("train", "Train tokenizer or generator according to an explicit configuration")
    p.add_argument("--output", required=True)
    p.add_argument("--tokenizer")
    p.add_argument("--resume")
    p.add_argument("--cpu", action="store_true")
    p = configured("requests", "Create Q4 or Marginal-7 requests from external assets")
    p.add_argument("--protocol", choices=["q4", "marginal7"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed-count", type=int, default=32)
    p = configured("controls", "Materialize matched S, J, and G control-label artifacts")
    p.add_argument("--output", required=True)
    p = configured("fit-priors", "Fit Direct attachment priors using the training split only")
    p.add_argument("--output", required=True)
    p = configured("generate", "Generate raw skeletons and optional Direct assemblies")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--requests", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--priors")
    p.add_argument("--device", default="cuda")
    p.add_argument("--temperature", type=float, default=0.8)
    p = configured(
        "evaluate", "Measure attributes, topology, and set geometry with fixed coordinates"
    )
    p.add_argument("--requests", required=True)
    p.add_argument("--generated", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--variant", choices=["raw", "direct"], default="direct")
    p.add_argument("--workers", type=int, default=4)
    p = configured("references", "Evaluate morphology retrieval and fixed-tree scaling")
    p.add_argument("--requests", required=True)
    p.add_argument("--output", required=True)
    p = configured(
        "proximity", "Scan every same-category training tree for nearest Chamfer proximity"
    )
    p.add_argument("--requests", required=True)
    p.add_argument("--generated", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from .data.prepare import prepare_sources

        result = prepare_sources(
            args.sources,
            args.output,
            read_json(args.definition),
            split_path=args.split,
            scale_fit=args.scale_fit,
            scale_target=args.scale_target,
        )
    else:
        c = load_config(args.config, data_root=args.data_root)
        if args.command == "verify-data":
            from .data.prepare import verify_assets

            result = verify_assets(c)
            if args.output:
                write_json(args.output, result)
        elif args.command == "train":
            from .training.run import run_training

            result = run_training(
                c,
                args.output,
                tokenizer_checkpoint=args.tokenizer,
                resume=args.resume,
                cpu=args.cpu,
            )
        elif args.command == "requests":
            from .experiments.requests import build_requests

            result = build_requests(
                c["data"]["attributes"],
                c["data"]["split"],
                args.output,
                protocol=args.protocol,
                seeds=args.seed_count,
                arm=c.get("control_arm", "main"),
            )
            result = {k: v for k, v in result.items() if k != "rows"}
        elif args.command == "controls":
            from .data.controls import build_controls

            result = build_controls(
                c["data"]["attributes"], c["data"]["split"], args.output, seed=c["seed"]
            )
        elif args.command == "fit-priors":
            from .geometry.priors import build_priors

            result = build_priors(
                Path(c["data"]["geometry"]), Path(c["data"]["split"]), Path(args.output)
            )
        elif args.command == "generate":
            from .generation import generate

            result = generate(
                c,
                args.checkpoint,
                args.requests,
                args.output,
                priors=args.priors,
                device=args.device,
                temperature=args.temperature,
            )
        elif args.command == "evaluate":
            from .experiments.evaluate import evaluate

            result = evaluate(
                c,
                args.requests,
                args.generated,
                args.output,
                variant=args.variant,
                workers=args.workers,
            )
        elif args.command == "references":
            from .experiments.references import evaluate_references

            result = evaluate_references(c, args.requests, args.output)
        elif args.command == "proximity":
            from .experiments.proximity import evaluate_proximity

            result = evaluate_proximity(c, args.requests, args.generated, args.output)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
