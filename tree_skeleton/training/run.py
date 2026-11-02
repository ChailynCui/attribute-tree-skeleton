"""One training API for the geometry tokenizer and conditional generator."""

from __future__ import annotations
import os
from pathlib import Path
import random
import numpy as np
import torch
from tree_skeleton.data.loader import SkeletonDataset, xy_dihedral_augment
from tree_skeleton.io import fingerprint, write_json
from tree_skeleton.nn.factory import make_tokenizer, make_generator


def datasets(config, *, tokenizer=False):
    d = config["data"]
    shared = dict(
        dataset_file_path=d["geometry"],
        attribute_json_path=d["attributes"],
        split_manifest_path=d["split"],
        replica=1,
        species_names=() if tokenizer else tuple(d["species"]),
        attribute_names=() if tokenizer else tuple(d["attribute_names"]),
        attribute_input_type="continuous",
        attribute_value_key=d.get("value_key", "percentile"),
        training_condition_mode=d.get("condition_mode", "correct"),
        training_condition_seed=int(config["seed"]),
    )
    train = SkeletonDataset(
        **shared, is_train=True, transform=xy_dihedral_augment, split_name="train"
    )
    validation = SkeletonDataset(**shared, is_train=False, transform=None, split_name="validation")
    return train, validation


def run_training(config, output, *, tokenizer_checkpoint=None, resume=None, cpu=False):
    torch.set_num_threads(int(config.get("cpu_threads", 4)))
    rank = int(os.environ.get("RANK", "0"))
    seed = int(config["seed"]) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not cpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    stage = config["stage"]
    is_tokenizer = stage == "tokenizer"
    if stage not in ("tokenizer", "generator"):
        raise ValueError("stage must be tokenizer or generator")
    from tree_skeleton.data.prepare import verify_assets

    coordinate_contract = verify_assets(config)
    train, val = datasets(config, tokenizer=is_tokenizer)
    runtime = config["runtime"]
    plan = config["optimization"]
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != int(runtime["processes"]):
        raise ValueError(
            f"Use {runtime['processes']} training processes for this configuration; actual {world}"
        )
    per_epoch = len(train) // (int(runtime["batch_size"]) * world)
    if per_epoch < 1:
        raise ValueError("Training split is smaller than one distributed minibatch")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not resume and (output / "resolved_config.json").exists():
        raise FileExistsError(
            "Output already contains a run; use a fresh path or explicit --resume"
        )
    schema = {
        "version": "four_species_three_attribute_condition_v1",
        "species_names": list(config["data"]["species"]),
        "attribute_names": list(config["data"]["attribute_names"]),
        "attribute_input_type": "continuous",
        "attribute_value_key": config["data"].get("value_key", "percentile"),
        "training_condition_mode": config["data"].get("condition_mode", "correct"),
        "control_arm": config.get("control_arm", "main"),
        "geometry_normalization": "one fixed global scale within species",
        "dataset_file_sha256": fingerprint(config["data"]["geometry"]),
        "attribute_file_sha256": fingerprint(config["data"]["attributes"]),
        "split_manifest_sha256": fingerprint(config["data"]["split"]),
        "coordinate_scale_fit": coordinate_contract["geometry_scale_fit"],
        "coordinate_protocol_sha256": fingerprint(config["data"]["coordinate_protocol"])
        if config["data"].get("coordinate_protocol")
        else None,
    }
    if rank == 0:
        write_json(output / "resolved_config.json", config)
        write_json(output / "input_contract.json", schema)
    tokenizer_contract = {
        "coordinate_scale_fit": schema["coordinate_scale_fit"],
        "coordinate_protocol_sha256": schema["coordinate_protocol_sha256"],
        "geometry_sha256": schema["dataset_file_sha256"],
    }
    kwargs = dict(
        dataset=train,
        val_dataset=val,
        num_train_steps=int(plan["schedule_updates"]),
        batch_size=int(runtime["batch_size"]),
        num_workers=int(runtime["workers"]),
        grad_accum_every=int(runtime["accumulation"]),
        learning_rate=float(plan["learning_rate"]),
        val_every_step=int(plan["validate_every"]),
        checkpoint_folder=str(output),
        checkpoint_file_name=resume or "auto",
        resume_training=bool(resume),
        log_every_step=int(plan.get("log_every", 10)),
        num_step_per_epoch=per_epoch,
        use_wandb_tracking=False,
        keep_last_checkpoints=1,
        accelerator_kwargs={
            "cpu": cpu,
            "mixed_precision": "no" if cpu else runtime["precision"],
            "step_scheduler_with_optimizer": False,
        },
        data_kwargs=("vertices", "segments"),
        scheduler_kwargs={
            "num_train_steps": int(plan["schedule_updates"]),
            "num_warmup_steps": int(0.1 * plan["schedule_updates"]),
        },
    )
    if is_tokenizer:
        from .tokenizer import TokenizerTrainingLoop

        engine = TokenizerTrainingLoop(
            make_tokenizer(config),
            **kwargs,
            weight_decay=float(plan["weight_decay"]),
            max_grad_norm=float(plan["gradient_clip"]),
            checkpoint_every_step=int(plan["checkpoint_every"]),
            coordinate_contract=tokenizer_contract,
        )
    else:
        from .generator import GeneratorTrainingLoop

        if not resume and not tokenizer_checkpoint:
            raise ValueError(
                "Generator training requires --tokenizer from the shared tokenizer run"
            )
        if tokenizer_checkpoint:
            checkpoint = torch.load(tokenizer_checkpoint, map_location="cpu", weights_only=False)
            saved_contract = checkpoint.get("coordinate_contract")
            if saved_contract != tokenizer_contract and (
                saved_contract or schema["coordinate_scale_fit"] == "train"
            ):
                raise ValueError(
                    "Tokenizer was trained under another coordinate protocol; train a matching tokenizer or use an explicit legacy configuration"
                )
            del checkpoint
        engine = GeneratorTrainingLoop(
            make_generator(config),
            **kwargs,
            weight_decay=float(plan["weight_decay"]),
            max_grad_norm=float(plan["gradient_clip"]),
            checkpoint_every_setp=int(plan["checkpoint_every"]),
            vqvae_ckpt_path=str(tokenizer_checkpoint or ""),
            validation_mode="full",
            early_stop_patience=0,
            train_stop_step=int(plan["stop_updates"]),
            expected_num_gpus=world,
            trial_id=config.get("name", "generator"),
            rung_id="explicit_update_budget",
            condition_schema=schema,
        )
    engine(project="attribute-tree-skeleton", run=config.get("name", stage), hps=config)
