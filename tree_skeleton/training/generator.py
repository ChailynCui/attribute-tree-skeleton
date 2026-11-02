# SPDX-License-Identifier: MIT
# Adapted research implementation; see README.md for license notices.
import os
import json
import gc
import datetime
import math
import hashlib
from pathlib import Path
from functools import partial
from contextlib import nullcontext
import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LRScheduler
from pytorch_custom_utils import get_adam_optimizer, add_wandb_tracker_contextmanager

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Optional, Tuple, Type, List

from tree_skeleton.data.loader import custom_collate
from tree_skeleton.nn.generator import AttributeTreeGenerator
from .optimization import OptimizerWithScheduler


# constants

DEFAULT_DDP_KWARGS = DistributedDataParallelKwargs(find_unused_parameters=True)

from tree_skeleton._tensor_ops import (
    exists,
    get_current_time,
    cycle,
    get_lr,
    divisible_by,
)

from tree_skeleton.representation.segments import segment_coords_to_skeleton

from tree_skeleton.representation.segments import load_model
from .logging import append_metrics
from .checkpoints import prune_milestone_checkpoints


# validation

VALIDATION_INDEX_KEY = "__validation_dataset_index__"
VALIDATION_MODES = frozenset({"full", "legacy_sampled"})
LENGTH_PROGRESS_STATE_PREFIXES = (
    "planning.predictor.",
    "planning.progress.",
)


def resolve_training_step_plan(
    *,
    epochs: int,
    num_step_per_epoch: int,
    schedule_horizon_steps: Optional[int] = None,
    train_stop_step: Optional[int] = None,
):
    """Resolve the LR-schedule horizon independently from an HPO rung stop."""

    epochs = int(epochs)
    num_step_per_epoch = int(num_step_per_epoch)
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if num_step_per_epoch <= 0:
        raise ValueError("num_step_per_epoch must be positive")

    legacy_horizon = epochs * num_step_per_epoch
    if schedule_horizon_steps is None:
        schedule_horizon_steps = legacy_horizon
    schedule_horizon_steps = int(schedule_horizon_steps)
    if schedule_horizon_steps <= 0:
        raise ValueError("schedule_horizon_steps must be positive")

    if train_stop_step is not None:
        train_stop_step = int(train_stop_step)
        if train_stop_step <= 0:
            raise ValueError("train_stop_step must be positive")
        if train_stop_step > schedule_horizon_steps:
            raise ValueError(
                "train_stop_step cannot exceed schedule_horizon_steps: "
                f"{train_stop_step} > {schedule_horizon_steps}"
            )

    return schedule_horizon_steps, train_stop_step


def build_training_contract(
    *,
    schedule_horizon_steps: int,
    train_stop_step: Optional[int],
    expected_num_gpus: Optional[int],
    actual_world_size: int,
    trial_id: str,
    rung_id: str,
    early_stop_patience: int,
    validation_mode: str,
    should_validate: bool,
):
    """Validate and serialize the fixed-budget HPO execution contract."""

    explicit_stop = train_stop_step is not None
    trial_id = str(trial_id).strip()
    rung_id = str(rung_id).strip()
    expected_num_gpus = None if expected_num_gpus is None else int(expected_num_gpus)
    actual_world_size = int(actual_world_size)
    early_stop_patience = int(early_stop_patience)

    if explicit_stop:
        if expected_num_gpus is None or expected_num_gpus <= 0:
            raise ValueError("train_stop_step requires a positive expected_num_gpus")
        if actual_world_size != expected_num_gpus:
            raise RuntimeError(
                "distributed world-size contract failed: "
                f"actual={actual_world_size}, "
                f"configured num_gpus={expected_num_gpus}"
            )
        if not trial_id or not rung_id:
            raise ValueError(
                "train_stop_step requires non-empty trial_id and rung_id "
                "for auditable HPO provenance"
            )
        if not should_validate:
            raise ValueError("train_stop_step requires a validation dataset")
        if validation_mode != "full":
            raise ValueError('train_stop_step requires validation_mode="full"')
        if early_stop_patience != 0:
            raise ValueError(
                "train_stop_step requires early_stop_patience=0 so every HPO "
                "candidate reaches the same successful optimizer-step budget"
            )

    return {
        "version": "transformer_hpo_rung_v1",
        "trial_id": trial_id or None,
        "rung_id": rung_id or None,
        "schedule_horizon_steps": int(schedule_horizon_steps),
        "train_stop_step": int(train_stop_step) if explicit_stop else None,
        "step_semantics": "successful_optimizer_updates",
        "expected_world_size": expected_num_gpus,
        "actual_world_size": actual_world_size,
        "early_stop_patience": early_stop_patience,
        "validation_mode": str(validation_mode),
    }


def audited_model_warmstart(
    model: Module,
    checkpoint_path,
    *,
    expected_condition_schema=None,
):
    """Load an old backbone while allowing only the new A1 planner weights."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"warmstart checkpoint not found: {checkpoint_path}")
    package = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(package, dict) or "model" not in package:
        raise ValueError("warmstart checkpoint has no model state dictionary")
    if (
        expected_condition_schema is not None
        and package.get("condition_schema") != expected_condition_schema
    ):
        raise ValueError(
            "warmstart checkpoint condition_schema does not match the current "
            "species/attribute/dataset contract"
        )

    from tree_skeleton.nn.checkpoint_keys import generator_state

    incompatible = model.load_state_dict(generator_state(package["model"]), strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    disallowed_missing = [
        key for key in missing if not key.startswith(LENGTH_PROGRESS_STATE_PREFIXES)
    ]
    if unexpected or disallowed_missing:
        raise RuntimeError(
            "unsafe warmstart state mismatch: "
            f"disallowed_missing={disallowed_missing}, unexpected={unexpected}"
        )
    if getattr(model, "length_progress_conditioning", False) and not missing:
        raise RuntimeError(
            "A1 warmstart expected newly initialized length/progress parameters, "
            "but the checkpoint supplied every model key"
        )

    with checkpoint_path.open("rb") as handle:
        checkpoint_sha256 = hashlib.sha256(handle.read()).hexdigest()
    return {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": int(package.get("step", 0)),
        "missing_new_keys": missing,
        "unexpected_keys": unexpected,
    }


class IndexedValidationDataset(Dataset):
    """Attach a stable local index to dict samples for full-pass auditing."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if not isinstance(sample, dict):
            return sample
        if VALIDATION_INDEX_KEY in sample:
            raise ValueError(
                f"validation sample already contains reserved key {VALIDATION_INDEX_KEY!r}"
            )

        indexed_sample = dict(sample)
        # A one-element tensor is compatible with custom_collate/pad_sequence.
        indexed_sample[VALIDATION_INDEX_KEY] = torch.tensor([int(index)], dtype=torch.long)
        return indexed_sample


def build_validation_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pad_id: int,
    validation_mode: str,
):
    if validation_mode not in VALIDATION_MODES:
        raise ValueError(
            f"validation_mode must be one of {sorted(VALIDATION_MODES)}, got {validation_mode!r}"
        )

    if validation_mode == "full":
        dataset = IndexedValidationDataset(dataset)
        shuffle = False
        drop_last = False
    else:
        shuffle = True
        drop_last = True

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=partial(custom_collate, pad_id=pad_id),
    )


def _validation_batch_to_forward_kwargs(batch, data_kwargs):
    if isinstance(batch, tuple):
        return dict(zip(data_kwargs, batch))
    if isinstance(batch, dict):
        return dict(batch)
    raise TypeError(f"unsupported validation batch type: {type(batch).__name__}")


def _move_tensors_to_device(forward_kwargs: dict, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in forward_kwargs.items()
    }


def _infer_validation_batch_size(forward_kwargs: dict) -> int:
    preferred_keys = (
        "segments",
        "codes",
        "vertices",
        "species_id",
        "attribute_ids",
        "attribute_values",
        "text_embeddings",
    )
    for key in preferred_keys:
        value = forward_kwargs.get(key)
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
        if isinstance(value, (list, tuple)):
            return len(value)
    raise RuntimeError("unable to infer validation batch size from model inputs")


def count_autoregressive_target_tokens(forward_kwargs: dict, model: Module) -> int:
    """Count the exact non-pad CE targets, including one EOS per sample."""

    if not hasattr(model, "forward_on_codes"):
        raise TypeError(
            "full token-weighted validation currently requires a "
            "AttributeTreeGenerator-compatible model with forward_on_codes"
        )

    pad_id = int(model.pad_id)
    codes = forward_kwargs.get("codes")
    if torch.is_tensor(codes):
        if codes.ndim < 2:
            raise ValueError(f"codes must include batch and token dimensions, got {codes.shape}")
        batch_size = int(codes.shape[0])
        token_count = int((codes != pad_id).sum().item()) + batch_size
    else:
        segments = forward_kwargs.get("segments")
        if not torch.is_tensor(segments) or segments.ndim != 3:
            raise ValueError(
                "full token-weighted validation requires either codes or padded "
                "[batch, segment, endpoint] segments"
            )
        if int(segments.shape[-1]) != 2:
            raise ValueError(f"segments must have two endpoints, got {segments.shape}")
        num_quantizers = int(model.tokenizer.num_quantizers)
        if num_quantizers <= 0:
            raise ValueError("model.tokenizer.num_quantizers must be positive")
        valid_segments = (segments != pad_id).all(dim=-1)
        batch_size = int(segments.shape[0])
        token_count = int(valid_segments.sum().item()) * 2 * num_quantizers + batch_size

    if token_count <= 0:
        raise RuntimeError("full validation batch contains no autoregressive targets")
    return token_count


def resolve_resumed_best_val_loss(checkpoint_package: dict, validation_mode: str):
    """Do not compare a full-pass metric with a legacy sampled checkpoint metric."""

    checkpoint_mode = str(checkpoint_package.get("validation_mode", "legacy_sampled"))
    if checkpoint_mode != validation_mode:
        return float("inf"), True, checkpoint_mode
    return float(checkpoint_package.get("best_val_loss", float("inf"))), False, checkpoint_mode


def run_full_validation_pass(
    *,
    model: Module,
    dataloader: DataLoader,
    expected_samples: int,
    device,
    data_kwargs,
    autocast_context=nullcontext,
):
    """Evaluate every validation sample once and aggregate by CE target token."""

    expected_samples = int(expected_samples)
    if expected_samples <= 0:
        raise ValueError("expected_samples must be positive")

    weighted_loss_sum = 0.0
    weighted_component_sums = {}
    target_token_count = 0
    sample_count = 0
    batch_count = 0
    seen_indices = []
    index_accounting_available = None
    was_training = model.training
    model.eval()

    try:
        # Deliberately iterate the finite loader directly: no cycle and no cap.
        for batch in dataloader:
            forward_kwargs = _validation_batch_to_forward_kwargs(batch, data_kwargs)
            validation_indices = forward_kwargs.pop(VALIDATION_INDEX_KEY, None)
            forward_kwargs = _move_tensors_to_device(forward_kwargs, device)

            inferred_batch_size = _infer_validation_batch_size(forward_kwargs)
            if validation_indices is None:
                if index_accounting_available is True:
                    raise RuntimeError("validation index metadata disappeared mid-pass")
                index_accounting_available = False
                batch_sample_count = inferred_batch_size
            else:
                if index_accounting_available is False:
                    raise RuntimeError("validation index metadata appeared mid-pass")
                index_accounting_available = True
                if torch.is_tensor(validation_indices):
                    batch_indices = [
                        int(index) for index in validation_indices.reshape(-1).tolist()
                    ]
                else:
                    batch_indices = [int(index) for index in validation_indices]
                batch_sample_count = len(batch_indices)
                if batch_sample_count != inferred_batch_size:
                    raise RuntimeError(
                        "validation index count does not match inferred batch size: "
                        f"{batch_sample_count} != {inferred_batch_size}"
                    )
                seen_indices.extend(batch_indices)

            with autocast_context(), torch.no_grad():
                val_loss, loss_dict = model(**forward_kwargs)

            if not torch.is_tensor(val_loss) or val_loss.numel() != 1:
                raise TypeError("validation model must return one scalar mean loss")
            batch_mean_loss = float(val_loss.detach().double().cpu().item())
            if not math.isfinite(batch_mean_loss):
                raise RuntimeError(f"non-finite validation loss: {batch_mean_loss}")

            batch_target_tokens = count_autoregressive_target_tokens(
                forward_kwargs,
                model,
            )
            weighted_loss_sum += batch_mean_loss * batch_target_tokens
            if loss_dict is not None:
                for key, value in loss_dict.items():
                    if torch.is_tensor(value):
                        if value.numel() != 1:
                            continue
                        value = float(value.detach().double().cpu().item())
                    elif isinstance(value, (int, float)):
                        value = float(value)
                    else:
                        continue
                    if not math.isfinite(value):
                        raise RuntimeError(f"non-finite validation loss component {key}: {value}")
                    weighted_component_sums[key] = (
                        weighted_component_sums.get(key, 0.0) + value * batch_target_tokens
                    )
            target_token_count += batch_target_tokens
            sample_count += batch_sample_count
            batch_count += 1
    finally:
        model.train(was_training)

    if batch_count == 0:
        raise RuntimeError("full validation dataloader yielded no batches")
    if sample_count != expected_samples:
        raise RuntimeError(
            "full validation sample-count audit failed: "
            f"observed {sample_count}, expected {expected_samples}"
        )

    unique_sample_count = None
    if index_accounting_available:
        unique_sample_count = len(set(seen_indices))
        expected_indices = list(range(expected_samples))
        if seen_indices != expected_indices or unique_sample_count != expected_samples:
            missing = sorted(set(expected_indices) - set(seen_indices))
            duplicates = sorted(
                index for index in set(seen_indices) if seen_indices.count(index) > 1
            )
            raise RuntimeError(
                "full validation index audit failed: "
                f"unique={unique_sample_count}/{expected_samples}, "
                f"missing={missing[:10]}, duplicates={duplicates[:10]}"
            )

    metrics = {
        "val_loss": weighted_loss_sum / target_token_count,
        "val_batches": batch_count,
        "val_samples": sample_count,
        "val_expected_samples": expected_samples,
        "val_unique_samples": unique_sample_count,
        "val_target_tokens": target_token_count,
        "val_index_accounting": bool(index_accounting_available),
        "validation_mode": "full",
        "val_loss_weighting": "autoregressive_target_token",
    }
    metrics.update(
        {f"val_{key}": value / target_token_count for key, value in weighted_component_sums.items()}
    )
    return metrics


# line transformer trainer


@add_wandb_tracker_contextmanager()
class GeneratorTrainingLoop(Module):
    @beartype
    def __init__(
        self,
        model: Module,
        dataset: Dataset,
        num_train_steps: int = 100,
        batch_size: int = 1,
        num_workers: int = 8,
        grad_accum_every: int = 1,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.0,
        max_grad_norm: Optional[float] = 0.5,
        val_dataset: Optional[Dataset] = None,
        val_every_step=1,
        val_num_batches=5,
        scheduler: Optional[Type[LRScheduler]] = None,
        scheduler_kwargs: dict = dict(),
        accelerator_kwargs: dict = dict(),
        optimizer_kwargs: dict = dict(),
        checkpoint_every_setp=1000,
        checkpoint_folder="./checkpoints_transformer",
        data_kwargs: Tuple[str, ...] = ["vertices", "lines", "line_edges"],
        log_every_step=10,
        use_wandb_tracking=False,
        resume_training=False,
        checkpoint_file_name: str = "",
        num_step_per_epoch=1000,
        from_start=False,
        vqvae_ckpt_path: str = "",
        early_stop_patience: int = 0,
        condition_schema: Optional[dict] = None,
        generation_plan_schema: Optional[dict] = None,
        warmstart_ckpt_path: str = "",
        keep_last_checkpoints: int = 0,
        validation_mode: str = "legacy_sampled",
        train_stop_step: Optional[int] = None,
        expected_num_gpus: Optional[int] = None,
        trial_id: str = "",
        rung_id: str = "",
    ):
        super().__init__()

        # experiment tracker

        self.use_wandb_tracking = use_wandb_tracking

        if use_wandb_tracking:
            accelerator_kwargs["log_with"] = "wandb"

        self.log_every_step = log_every_step

        if "kwargs_handlers" not in accelerator_kwargs:
            accelerator_kwargs["kwargs_handlers"] = [DEFAULT_DDP_KWARGS]

        self.accelerator = Accelerator(**accelerator_kwargs)
        self.accelerator.print(f"device {str(self.accelerator.device)} is used!")

        if resume_training and warmstart_ckpt_path:
            raise ValueError("resume_training and warmstart_ckpt_path are mutually exclusive")

        if not resume_training:
            model.tokenizer = load_model(
                model.tokenizer, vqvae_ckpt_path, self.accelerator.device
            )

        self.warmstart_audit = None
        if warmstart_ckpt_path:
            self.warmstart_audit = audited_model_warmstart(
                model,
                warmstart_ckpt_path,
                expected_condition_schema=condition_schema,
            )
            self.accelerator.print(
                "audited model warmstart: "
                f"{self.warmstart_audit['checkpoint_path']} "
                f"missing_new_keys={len(self.warmstart_audit['missing_new_keys'])}"
            )

        self.model = model

        if self.is_main:
            self.print_params_num()

        optimizer = get_adam_optimizer(
            model.parameters(),
            lr=learning_rate,
            wd=weight_decay,
            filter_by_requires_grad=True,  # filter ae model params
            **optimizer_kwargs,
        )

        self.optimizer = OptimizerWithScheduler(
            accelerator=self.accelerator,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs
            if len(scheduler_kwargs) > 0
            else dict(num_train_steps=num_train_steps),
            max_grad_norm=max_grad_norm,
        )

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=partial(custom_collate, pad_id=model.pad_id),
        )

        self.should_validate = exists(val_dataset)

        self.validation_mode = str(validation_mode)
        if self.validation_mode not in VALIDATION_MODES:
            raise ValueError(
                f"validation_mode must be one of {sorted(VALIDATION_MODES)}, "
                f"got {self.validation_mode!r}"
            )
        self.val_every_step = val_every_step
        if self.should_validate and self.is_main:
            assert len(val_dataset) > 0, "your validation dataset is empty"

            self.val_dataset_size = len(val_dataset)
            self.val_num_batches = val_num_batches
            self.val_dataloader = build_validation_dataloader(
                val_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pad_id=model.pad_id,
                validation_mode=self.validation_mode,
            )
            self.print(
                f"validation mode={self.validation_mode} expected_samples={self.val_dataset_size}"
            )

        if hasattr(dataset, "data_kwargs") and exists(dataset.data_kwargs):
            assert is_bearable(dataset.data_kwargs, List[str])
            self.data_kwargs = dataset.data_kwargs
        else:
            self.data_kwargs = data_kwargs

        (
            self.model,
            self.dataloader,
        ) = self.accelerator.prepare(
            self.model,
            self.dataloader,
        )

        self.grad_accum_every = grad_accum_every
        self.num_train_steps = int(num_train_steps)
        if self.num_train_steps <= 0:
            raise ValueError("num_train_steps must be positive")
        self.has_explicit_train_stop = train_stop_step is not None
        self.train_stop_step = (
            self.num_train_steps if train_stop_step is None else int(train_stop_step)
        )
        if self.train_stop_step <= 0:
            raise ValueError("train_stop_step must be positive")
        if self.train_stop_step > self.num_train_steps:
            raise ValueError(
                "train_stop_step cannot exceed the scheduler horizon "
                f"({self.num_train_steps}), got {self.train_stop_step}"
            )
        self.expected_num_gpus = None if expected_num_gpus is None else int(expected_num_gpus)
        self.actual_world_size = int(self.accelerator.num_processes)
        self.trial_id = str(trial_id).strip()
        self.rung_id = str(rung_id).strip()
        self.register_buffer("step", torch.tensor(0))

        self.checkpoint_every_step = checkpoint_every_setp
        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(exist_ok=True, parents=True)
        self.metrics_path = self.checkpoint_folder / "metrics.jsonl"
        self.best_checkpoint_path = self.checkpoint_folder / "best.pth"
        self.best_metric_path = self.checkpoint_folder / "best_metrics.json"
        self.best_val_loss = float("inf")
        self.keep_last_checkpoints = int(keep_last_checkpoints)
        if self.keep_last_checkpoints < 0:
            raise ValueError("keep_last_checkpoints must be non-negative")

        self.from_start = from_start
        self.condition_schema = condition_schema
        self.generation_plan_schema = generation_plan_schema
        self.early_stop_patience = int(early_stop_patience)
        self.training_contract = build_training_contract(
            schedule_horizon_steps=self.num_train_steps,
            train_stop_step=(self.train_stop_step if self.has_explicit_train_stop else None),
            expected_num_gpus=self.expected_num_gpus,
            actual_world_size=self.actual_world_size,
            trial_id=self.trial_id,
            rung_id=self.rung_id,
            early_stop_patience=self.early_stop_patience,
            validation_mode=self.validation_mode,
            should_validate=self.should_validate,
        )
        self.val_checks_without_improvement = 0

        if resume_training:
            print("loading checkpoint from the file: ", checkpoint_file_name)
            self.load(checkpoint_file_name, from_start=from_start)

        self.num_step_per_epoch = num_step_per_epoch // dataset.replica

    def log(self, **data_kwargs):
        self.accelerator.log(data_kwargs, step=self.step.item())

    @property
    def device(self):
        return self.unwrapped_model.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def print(self, msg):
        return self.accelerator.print(msg)

    def next_data_to_forward_kwargs(self, dl_iter) -> dict:
        data = next(dl_iter)

        if isinstance(data, tuple):
            forward_kwargs = dict(zip(self.data_kwargs, data))

        elif isinstance(data, dict):
            forward_kwargs = data

        return forward_kwargs

    def print_params_num(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model Total parameters: {total_params / 1e6} M")

        non_trainable_params = sum(
            p.numel() for p in self.model.parameters() if not p.requires_grad
        )
        print(f"Number of non-trainable parameters: {non_trainable_params / 1e6}")

    def _save_package_atomic(self, pkg: dict, destination: Path):
        tmp_path = destination.with_name(f".{destination.name}.tmp")

        try:
            torch.save(pkg, str(tmp_path))
            os.replace(tmp_path, destination)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _write_json_atomic(self, payload: dict, destination: Path):
        tmp_path = destination.with_name(f".{destination.name}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, destination)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _checkpoint_sort_key(self, path: Path):
        if path.name == "best.pth":
            return (-1, path.name)

        stem = path.stem
        if stem.startswith("model-"):
            suffix = stem.removeprefix("model-")
            if suffix.isdigit():
                return (int(suffix), path.name)

        return (-1, path.name)

    def _get_resume_candidates(self, requested_file_name: str) -> list[str]:
        normalized = (requested_file_name or "").strip()
        auto_resume = normalized.lower() in {"", "auto", "latest"}

        candidates = []
        seen = set()

        def add_candidate(path: Path):
            if not path.is_file() or path.name in seen:
                return
            candidates.append(path.name)
            seen.add(path.name)

        if not auto_resume:
            add_candidate(self.checkpoint_folder / normalized)

        for path in sorted(
            self.checkpoint_folder.glob("model-*.pt"), key=self._checkpoint_sort_key, reverse=True
        ):
            add_candidate(path)

        add_candidate(self.best_checkpoint_path)

        return candidates

    def _load_checkpoint_package(self, file_name: str):
        checkpoint_path = self.checkpoint_folder / file_name
        return torch.load(
            str(checkpoint_path), map_location=self.accelerator.device, weights_only=False
        )

    def save(self, milestone):

        pkg = dict(
            model=self.unwrapped_model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            step=self.step.item(),
            scaler=self.accelerator.scaler.state_dict()
            if exists(self.accelerator.scaler)
            else None,
            best_val_loss=self.best_val_loss,
            condition_schema=self.condition_schema,
            generation_plan_schema=self.generation_plan_schema,
            warmstart_audit=self.warmstart_audit,
            validation_mode=self.validation_mode,
            schedule_horizon_steps=self.num_train_steps,
            train_stop_step=(self.train_stop_step if self.has_explicit_train_stop else None),
            training_contract=dict(self.training_contract),
        )

        self._save_package_atomic(pkg, self.checkpoint_folder / f"model-{milestone}.pt")
        if self.keep_last_checkpoints > 0:
            removed = prune_milestone_checkpoints(
                self.checkpoint_folder,
                keep_last=self.keep_last_checkpoints,
            )
            if removed:
                self.print(
                    "checkpoint retention removed: " + ", ".join(path.name for path in removed)
                )

    def save_stop_checkpoint(self, validation_metrics: dict):
        """Save the exact HPO rung endpoint outside milestone retention."""

        step = int(self.step.item())
        if step != self.train_stop_step:
            raise RuntimeError(
                "refusing to save an HPO stop checkpoint at the wrong step: "
                f"observed={step}, configured={self.train_stop_step}"
            )
        if not isinstance(validation_metrics, dict):
            raise TypeError("HPO stop checkpoint requires terminal validation metrics")
        if validation_metrics.get("validation_mode") != "full":
            raise ValueError("HPO stop checkpoint requires terminal full-validation metrics")

        destination = self.checkpoint_folder / f"stop-step-{step:09d}.pt"
        pkg = dict(
            model=self.unwrapped_model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            step=step,
            scaler=(
                self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
            ),
            best_val_loss=self.best_val_loss,
            condition_schema=self.condition_schema,
            generation_plan_schema=self.generation_plan_schema,
            warmstart_audit=self.warmstart_audit,
            validation_mode=self.validation_mode,
            schedule_horizon_steps=self.num_train_steps,
            train_stop_step=self.train_stop_step,
            stop_reason="configured_train_stop_step",
            terminal_validation=dict(validation_metrics),
            training_contract={
                **self.training_contract,
                "terminal_successful_optimizer_step": step,
            },
        )
        self._save_package_atomic(pkg, destination)
        self._write_json_atomic(
            {
                "version": "transformer_hpo_terminal_metrics_v1",
                "step": step,
                "trial_id": self.trial_id,
                "rung_id": self.rung_id,
                "schedule_horizon_steps": self.num_train_steps,
                "train_stop_step": self.train_stop_step,
                "step_semantics": "successful_optimizer_updates",
                "terminal_validation": dict(validation_metrics),
                "checkpoint": str(destination.resolve()),
            },
            self.checkpoint_folder / "terminal_metrics.json",
        )
        return destination

    def save_best(self, val_loss, validation_metrics=None):
        pkg = dict(
            model=self.unwrapped_model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            step=self.step.item(),
            scaler=self.accelerator.scaler.state_dict()
            if exists(self.accelerator.scaler)
            else None,
            best_val_loss=val_loss,
            condition_schema=self.condition_schema,
            generation_plan_schema=self.generation_plan_schema,
            warmstart_audit=self.warmstart_audit,
            validation_mode=self.validation_mode,
            schedule_horizon_steps=self.num_train_steps,
            train_stop_step=(self.train_stop_step if self.has_explicit_train_stop else None),
            training_contract=dict(self.training_contract),
        )

        self._save_package_atomic(pkg, self.best_checkpoint_path)

        with self.best_metric_path.open("w", encoding="utf-8") as f:
            metric_payload = {
                "best_val_loss": float(val_loss),
                "step": int(self.step.item()),
                "epoch": float(self.step.item() / self.num_step_per_epoch),
                "checkpoint": str(self.best_checkpoint_path),
                "validation_mode": self.validation_mode,
            }
            if validation_metrics is not None:
                metric_payload["validation_audit"] = validation_metrics
            json.dump(
                metric_payload,
                f,
                ensure_ascii=True,
                indent=2,
            )

    def load(self, file_name, from_start=False):
        candidates = self._get_resume_candidates(file_name)

        if len(candidates) == 0:
            raise FileNotFoundError(
                f'No checkpoint found in {self.checkpoint_folder} for resume request "{file_name}".'
            )

        last_error = None
        loaded_file_name = None
        pkg = None

        for candidate in candidates:
            try:
                pkg = self._load_checkpoint_package(candidate)
                loaded_file_name = candidate
                if candidate != file_name:
                    self.print(f"resume fallback selected checkpoint: {candidate}")
                break
            except Exception as err:
                last_error = err
                self.print(f"failed to load checkpoint {self.checkpoint_folder / candidate}: {err}")

        if pkg is None or loaded_file_name is None:
            raise RuntimeError(
                f"Unable to load any checkpoint from {self.checkpoint_folder}. Last error: {last_error}"
            ) from last_error

        checkpoint_schema = pkg.get("condition_schema")
        if self.condition_schema is not None and checkpoint_schema != self.condition_schema:
            raise ValueError(
                "checkpoint condition_schema does not match the current "
                "species/attribute/dataset contract"
            )

        checkpoint_plan_schema = pkg.get("generation_plan_schema")
        if checkpoint_plan_schema != self.generation_plan_schema:
            raise ValueError(
                "checkpoint generation_plan_schema does not match the current "
                "length/progress conditioning contract"
            )

        self.unwrapped_model.load_state_dict(pkg["model"])

        if not from_start:
            self.optimizer.load_state_dict(pkg["optimizer"])
            self.step.copy_(pkg["step"])
            (
                self.best_val_loss,
                reset_best_metric,
                checkpoint_validation_mode,
            ) = resolve_resumed_best_val_loss(pkg, self.validation_mode)
            if reset_best_metric:
                self.print(
                    "resume checkpoint validation metric reset: "
                    f"checkpoint mode={checkpoint_validation_mode}, "
                    f"current mode={self.validation_mode}"
                )

        print(f"loaded checkpoint from {self.checkpoint_folder / loaded_file_name}")

    def log_loss(self, loss, loss_dict, cur_lr=None, total_norm=None, step=None):
        log_data = {"total_loss": loss}

        if loss_dict is not None:
            log_data.update(loss_dict)

        if cur_lr is not None:
            log_data["cur_lr"] = cur_lr
        if total_norm is not None:
            log_data["total_norm"] = total_norm if exists(total_norm) else 0.0

        if self.is_main:
            append_metrics(
                self.metrics_path,
                step=step,
                epoch=step / self.num_step_per_epoch,
                split="train",
                metrics=log_data,
            )

        if not self.use_wandb_tracking:
            log_str = (
                str(step)
                + " | "
                + " | ".join(
                    [
                        f"{key}: {value.item() if torch.is_tensor(value) and value.dim() == 0 else value:.3f}"
                        for key, value in log_data.items()
                    ]
                )
            )
            print(log_str)
        else:
            self.log(**log_data)

    def _run_validation_at_step(self, step: int, val_dl_iter=None):
        if self.validation_mode == "full":
            validation_metrics = run_full_validation_pass(
                model=self.unwrapped_model,
                dataloader=self.val_dataloader,
                expected_samples=self.val_dataset_size,
                device=self.device,
                data_kwargs=self.data_kwargs,
                autocast_context=self.accelerator.autocast,
            )
            total_val_loss = validation_metrics["val_loss"]
            audit_summary = (
                f" samples={validation_metrics['val_samples']}"
                f"/{validation_metrics['val_expected_samples']}"
                f" unique={validation_metrics['val_unique_samples']}"
                f" batches={validation_metrics['val_batches']}"
                f" target_tokens={validation_metrics['val_target_tokens']}"
            )
        else:
            if val_dl_iter is None:
                raise RuntimeError("legacy sampled validation requires an iterator")
            total_val_loss = 0.0
            num_val_batches = self.val_num_batches * self.grad_accum_every
            was_training = self.unwrapped_model.training
            self.unwrapped_model.eval()
            try:
                for _ in range(num_val_batches):
                    with self.accelerator.autocast(), torch.no_grad():
                        forward_kwargs = self.next_data_to_forward_kwargs(val_dl_iter)
                        forward_kwargs = _move_tensors_to_device(
                            forward_kwargs,
                            self.device,
                        )
                        val_loss, _ = self.unwrapped_model(**forward_kwargs)
                        total_val_loss += (
                            float(val_loss.detach().double().cpu().item()) / num_val_batches
                        )
            finally:
                self.unwrapped_model.train(was_training)
            validation_metrics = {
                "val_loss": total_val_loss,
                "val_batches": num_val_batches,
                "validation_mode": "legacy_sampled",
                "val_loss_weighting": "batch_mean",
            }
            audit_summary = f" sampled_batches={num_val_batches}"

        current_time = get_current_time()
        self.print(
            current_time
            + f" valid recon loss: {total_val_loss:.6f}"
            + f" mode={self.validation_mode}"
            + audit_summary
        )

        append_metrics(
            self.metrics_path,
            step=step,
            epoch=step / self.num_step_per_epoch,
            split="val",
            metrics=validation_metrics,
        )
        self.log(
            **{
                key: value
                for key, value in validation_metrics.items()
                if isinstance(value, (int, float)) and value is not None
            }
        )

        current_val_loss = float(total_val_loss)
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.val_checks_without_improvement = 0
            self.save_best(
                current_val_loss,
                validation_metrics=validation_metrics,
            )
            self.print(
                get_current_time() + f" best checkpoint updated at {self.best_checkpoint_path} "
                f"(val_loss={current_val_loss:.3f})"
            )
        else:
            self.val_checks_without_improvement += 1
            if (
                self.early_stop_patience > 0
                and self.val_checks_without_improvement >= self.early_stop_patience
            ):
                self.print(
                    get_current_time()
                    + f" early stopping after {self.val_checks_without_improvement} "
                    "validation checks without improvement"
                )
                self.accelerator.set_trigger()

        return validation_metrics

    def train(self):
        step = self.step.item()
        if self.has_explicit_train_stop and step >= self.train_stop_step:
            raise RuntimeError(
                "resume step must be below train_stop_step for an HPO rung: "
                f"{step} >= {self.train_stop_step}"
            )
        dl_iter = cycle(self.dataloader)

        is_multi_loss = False
        val_dl_iter = None

        if self.should_validate and self.is_main and self.validation_mode == "legacy_sampled":
            val_dl_iter = cycle(self.val_dataloader)

        while step < self.train_stop_step:
            for i in range(self.grad_accum_every):
                is_last = i == (self.grad_accum_every - 1)
                maybe_no_sync = (
                    partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext
                )

                forward_kwargs = self.next_data_to_forward_kwargs(dl_iter)

                with self.accelerator.autocast(), maybe_no_sync():
                    loss, loss_dict = self.model(**forward_kwargs)

                    self.accelerator.backward(loss / self.grad_accum_every)

            if self.is_main and divisible_by(step, self.log_every_step):
                cur_lr = get_lr(self.optimizer.optimizer)

                total_norm = self.optimizer.total_norm

                self.log_loss(loss, loss_dict, cur_lr, total_norm, step)

            optimizer_step_succeeded = self.optimizer.step()
            self.optimizer.zero_grad()

            if not optimizer_step_succeeded:
                if self.has_explicit_train_stop:
                    self.print(
                        get_current_time()
                        + f" optimizer update skipped at successful_step={step}; "
                        "scheduler and HPO train step were not advanced"
                    )
                    self.wait()
                    continue
                self.print(
                    get_current_time() + f" optimizer update skipped at step={step}; scheduler was "
                    "not advanced, preserving legacy train-step accounting"
                )

            step += 1
            self.step.add_(1)

            self.wait()

            validated_this_step = False
            validation_metrics_this_step = None

            if self.is_main and self.should_validate and divisible_by(step, self.val_every_step):
                validation_metrics_this_step = self._run_validation_at_step(
                    step,
                    val_dl_iter=val_dl_iter,
                )
                validated_this_step = True

            reached_configured_stop = step >= self.train_stop_step
            if (
                self.is_main
                and self.has_explicit_train_stop
                and reached_configured_stop
                and not validated_this_step
            ):
                # Explicit HPO rung stops always receive a complete validation
                # pass.  If the stop is already a regular validation boundary,
                # the branch above ran it exactly once.
                validation_metrics_this_step = self._run_validation_at_step(step)
                validated_this_step = True

            self.wait()

            if self.accelerator.check_trigger():
                break

            if self.is_main and self.has_explicit_train_stop and reached_configured_stop:
                stop_path = self.save_stop_checkpoint(validation_metrics_this_step)
                self.print(get_current_time() + f" HPO stop checkpoint saved at {stop_path}")
            elif self.is_main and (
                divisible_by(step, self.checkpoint_every_step) or step >= self.num_train_steps
            ):
                checkpoint_num = (
                    step + self.checkpoint_every_step - 1
                ) // self.checkpoint_every_step
                idx = str(checkpoint_num).zfill(2)
                self.save(milestone=idx)

                current_time = get_current_time()
                self.print(
                    current_time
                    + f" checkpoint saved at {self.checkpoint_folder / f'model-{idx}.pt'}"
                )

            self.wait()

        # Make sure that the wandb tracker finishes correctly
        self.accelerator.end_training()

        self.print("training complete")

    def forward(self, project: str, run: str | None = None, hps: dict | None = None):
        if self.is_main and self.use_wandb_tracking:
            print("using wandb tracking")

            with self.wandb_tracking(project=project, run=run, hps=hps):
                self.train()
        else:
            print("not using wandb tracking")

            self.train()


# Compatibility alias for existing training integrations.
GeneratorTrainer = GeneratorTrainingLoop
