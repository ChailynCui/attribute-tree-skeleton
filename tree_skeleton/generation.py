"""Generate raw segment streams and their deterministic Direct assembly."""

from __future__ import annotations
import random
import time
from pathlib import Path
import numpy as np
import torch
from .config import SPECIES
from .io import read_json, write_json, write_geometry, fingerprint
from .nn.factory import load_generator
from .geometry.assembly import build_direct_geometry


def sample_directory(root, row):
    species = row["species"]
    condition = row.get("physical_condition_id", row["condition_id"])
    seed = int(row["generation_seed"])
    import re

    if species not in SPECIES or not re.fullmatch(r"[A-Za-z0-9_-]+", condition) or seed < 0:
        raise ValueError("Invalid species, condition ID, or seed")
    return Path(root) / "samples" / species / condition / f"seed{seed}"


def generate(config, checkpoint, requests, output, *, priors=None, device="cuda", temperature=0.8):
    if device.startswith("cpu"):
        torch.set_num_threads(4)
    model, package = load_generator(config, checkpoint, device)
    del package
    manifest = read_json(requests)
    rows = manifest["rows"]
    output = Path(output)
    if manifest.get("arm", "main") != config.get("control_arm", "main"):
        raise ValueError("Request encoding does not match this model/control configuration")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run.json").exists():
        raise FileExistsError("Use a new generation output directory")
    physical_count = len({sample_directory(output, r) for r in rows})
    write_json(
        output / "run.json",
        {
            "checkpoint_sha256": fingerprint(checkpoint),
            "requests_sha256": fingerprint(requests),
            "planned_logical_count": len(rows),
            "planned_physical_count": physical_count,
            "temperature": temperature,
            "max_tokens": model.max_seq_len,
            "ground_truth_geometry_or_length_input": False,
            "postprocessor": "Direct" if priors else None,
        },
    )
    results = []
    seen = set()
    for row in rows:
        directory = sample_directory(output, row)
        if directory in seen:
            continue
        seen.add(directory)
        directory.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        result = {**row, "status": "failed"}
        try:
            seed = int(row["generation_seed"])
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            u = np.asarray(row["attribute_percentiles"], dtype=float)
            if u.shape != (3,) or not np.isfinite(u).all() or np.any((u < 0) | (u > 1)):
                raise ValueError("Expected three finite percentiles in [0,1]")
            sid = SPECIES.index(row["species"])
            if int(row["species_id"]) != sid:
                raise ValueError("Species ID/name mismatch")
            with torch.inference_mode():
                codes = model.generate(
                    batch_size=1,
                    species_id=torch.tensor([[sid]], device=device),
                    attribute_values=torch.tensor(u[None], dtype=torch.float32, device=device),
                    temperature=temperature,
                    max_seq_len=model.max_seq_len,
                    min_seq_len=4,
                    return_codes=True,
                )
                nonpadding = codes[0][codes[0] >= 0]
                exhausted = int(nonpadding.numel()) >= model.max_seq_len
                segments, mask = model.tokenizer.codes_to_segments(codes)
                raw = segments[0, mask[0]].cpu().numpy()
            vertices = raw.reshape(-1, 3)
            lines = np.arange(len(vertices)).reshape(-1, 2)
            write_geometry(directory / "raw_tree.npz", vertices, lines)
            result.update(
                raw_edges=len(raw),
                natural_eos=not exhausted,
                termination="length_cap" if exhausted else "eos",
            )
            if exhausted:
                raise RuntimeError("Reached the token cap without natural EOS")
            if priors:
                prior = read_json(Path(priors) / f"{row['species']}_training_morphology_prior.json")
                v, e, audit = build_direct_geometry(raw, prior)
                write_geometry(directory / "direct_tree.npz", v, e)
                write_json(directory / "assembly.json", audit)
            result["status"] = "completed"
        except Exception as error:
            result["error"] = str(error)
        result["seconds"] = time.monotonic() - started
        write_json(directory / "sample.json", result)
        results.append(result)
        write_json(
            output / "generation_summary.json",
            {
                "planned_logical_count": len(rows),
                "planned_physical_count": physical_count,
                "rows": results,
                "completed": sum(x["status"] == "completed" for x in results),
                "failures": sum(x["status"] != "completed" for x in results),
            },
        )
    return {
        "planned_logical_count": len(rows),
        "planned_physical_count": physical_count,
        "completed": sum(x["status"] == "completed" for x in results),
        "output": str(output),
    }
