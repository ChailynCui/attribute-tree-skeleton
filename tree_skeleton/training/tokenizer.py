# SPDX-License-Identifier: MIT
# Adapted research implementation; see README.md for license notices.
import json
import os
from pathlib import Path
from functools import partial
from contextlib import nullcontext
import numpy as np

import torch
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LRScheduler

from pytorch_custom_utils import get_adam_optimizer, add_wandb_tracker_contextmanager

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Optional, Tuple, Type, List

from ema_pytorch import EMA

from tree_skeleton.data.loader import custom_collate

from tree_skeleton.nn.tokenizer import TreeSegmentTokenizer

from .optimization import OptimizerWithScheduler

# constants

DEFAULT_DDP_KWARGS = DistributedDataParallelKwargs(find_unused_parameters=True)

# helper functions

from tree_skeleton._tensor_ops import (
    exists,
    cycle,
    get_current_time,
    get_lr,
    divisible_by,
)
from .logging import append_metrics
from .checkpoints import prune_milestone_checkpoints

# autoencoder trainer


@add_wandb_tracker_contextmanager()
class TokenizerTrainingLoop(Module):
    @beartype
    def __init__(
        self,
        model: TreeSegmentTokenizer,
        dataset: Dataset,
        num_train_steps: int = 10,
        batch_size: int = 1,
        num_workers: int = 8,
        grad_accum_every: int = 1,
        val_dataset: Optional[Dataset] = None,
        val_every_step: int = 1,
        val_num_batches: int = 5,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        max_grad_norm: Optional[float] = 10,
        ema_kwargs: dict = dict(),
        scheduler: Optional[Type[LRScheduler]] = None,
        scheduler_kwargs: dict = dict(),
        accelerator_kwargs: dict = dict(),
        optimizer_kwargs: dict = dict(),
        checkpoint_every_step=1000,
        checkpoint_folder="./checkpoints_ae",
        data_kwargs: Tuple[str, ...] = ["vertices", "lines", "line_edges"],
        log_every_step=10,
        use_wandb_tracking=False,
        resume_training=False,
        checkpoint_file_name="model.pt",
        from_start=False,
        num_step_per_epoch=1000,
        keep_last_checkpoints: int = 0,
        coordinate_contract: Optional[dict] = None,
    ):
        super().__init__()
        self.coordinate_contract = dict(coordinate_contract or {})

        self.use_wandb_tracking = use_wandb_tracking

        if use_wandb_tracking:
            accelerator_kwargs["log_with"] = "wandb"

        self.log_every_step = log_every_step

        if "kwargs_handlers" not in accelerator_kwargs:
            accelerator_kwargs["kwargs_handlers"] = [DEFAULT_DDP_KWARGS]

        # accelerator

        self.accelerator = Accelerator(**accelerator_kwargs)
        self.accelerator.print(f"device {str(self.accelerator.device)} is used!")

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

        self.val_every_step = val_every_step

        if self.should_validate and self.is_main:
            assert len(val_dataset) > 0, "your validation dataset is empty"

            self.val_every_step = val_every_step
            self.val_num_batches = val_num_batches

            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=True,
                collate_fn=partial(custom_collate, pad_id=model.pad_id),
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

        if self.is_main:
            self.ema_model = EMA(model, **ema_kwargs)
            self.ema_model.to(self.device)

        self.grad_accum_every = grad_accum_every
        self.num_train_steps = num_train_steps
        self.register_buffer("step", torch.tensor(0))

        self.checkpoint_every_step = checkpoint_every_step
        self.checkpoint_folder = Path(checkpoint_folder)
        Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.checkpoint_folder / "metrics.jsonl"
        self.best_checkpoint_path = self.checkpoint_folder / "best.pth"
        self.best_metric_path = self.checkpoint_folder / "best_metrics.json"
        self.best_val_loss = float("inf")
        self.keep_last_checkpoints = int(keep_last_checkpoints)
        if self.keep_last_checkpoints < 0:
            raise ValueError("keep_last_checkpoints must be non-negative")

        if resume_training:
            print("loading checkpoint from the file: ", checkpoint_file_name)
            self.load(checkpoint_file_name, from_start=from_start)

        self.num_step_per_epoch = num_step_per_epoch // dataset.replica

    def print_params_num(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model Total parameters: {total_params / 1e6} M")

        non_trainable_params = sum(
            p.numel() for p in self.model.parameters() if not p.requires_grad
        )
        print(f"Number of non-trainable parameters: {non_trainable_params / 1e6}")

    @property
    def ema_tokenizer(self):
        return self.ema_model.ema_model

    def tokenize(self, *args, **kwargs):
        return self.ema_tokenizer.tokenize(*args, **kwargs)

    def log(self, **data_kwargs):
        self.accelerator.log(data_kwargs, step=self.step.item())

    @property
    def device(self):
        return self.unwrapped_model.device

    @property
    def local_process_index(self):
        return self.accelerator.local_process_index

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

    def _save_package_atomic(self, package: dict, destination: Path):
        tmp_path = destination.with_name(f".{destination.name}.tmp")
        try:
            torch.save(package, str(tmp_path))
            os.replace(tmp_path, destination)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _checkpoint_sort_key(self, path: Path):
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
            self.checkpoint_folder.glob("model-*.pt"),
            key=self._checkpoint_sort_key,
            reverse=True,
        ):
            add_candidate(path)
        add_candidate(self.best_checkpoint_path)
        return candidates

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = dict(
            model=self.accelerator.get_state_dict(self.model),
            ema_model=self.ema_model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            step=self.step.item(),
            scaler=self.accelerator.scaler.state_dict()
            if exists(self.accelerator.scaler)
            else None,
            best_val_loss=self.best_val_loss,
            coordinate_contract=self.coordinate_contract,
        )

        destination = self.checkpoint_folder / f"model-{milestone}.pt"
        self._save_package_atomic(data, destination)
        if self.keep_last_checkpoints > 0:
            removed = prune_milestone_checkpoints(
                self.checkpoint_folder,
                keep_last=self.keep_last_checkpoints,
            )
            if removed:
                self.print(
                    "checkpoint retention removed: " + ", ".join(path.name for path in removed)
                )

    def save_best(self, val_loss):
        if not self.accelerator.is_local_main_process:
            return

        data = dict(
            model=self.accelerator.get_state_dict(self.model),
            ema_model=self.ema_model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            step=self.step.item(),
            scaler=self.accelerator.scaler.state_dict()
            if exists(self.accelerator.scaler)
            else None,
            best_val_loss=val_loss,
            coordinate_contract=self.coordinate_contract,
        )

        self._save_package_atomic(data, self.best_checkpoint_path)

        with self.best_metric_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_val_loss": float(val_loss),
                    "step": int(self.step.item()),
                    "epoch": float(self.step.item() / self.num_step_per_epoch),
                    "checkpoint": str(self.best_checkpoint_path),
                },
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
                pkg = torch.load(
                    str(self.checkpoint_folder / candidate),
                    map_location=self.accelerator.device,
                    weights_only=False,
                )
                loaded_file_name = candidate
                if candidate != file_name:
                    self.print(f"resume fallback selected checkpoint: {candidate}")
                break
            except Exception as err:
                last_error = err
                self.print(f"failed to load checkpoint {self.checkpoint_folder / candidate}: {err}")

        if pkg is None or loaded_file_name is None:
            raise RuntimeError(
                f"Unable to load any checkpoint from {self.checkpoint_folder}. "
                f"Last error: {last_error}"
            ) from last_error

        saved_contract = pkg.get("coordinate_contract")
        if (
            self.coordinate_contract
            and saved_contract != self.coordinate_contract
            and (saved_contract or self.coordinate_contract.get("coordinate_scale_fit") == "train")
        ):
            raise ValueError("Tokenizer checkpoint coordinate protocol differs from this run")
        self.unwrapped_model.load_state_dict(pkg["model"])

        if self.is_main:
            self.ema_model.load_state_dict(pkg["ema_model"])
            self.ema_model.to(self.device)

        if not from_start:
            self.optimizer.load_state_dict(pkg["optimizer"])
            self.step.copy_(pkg["step"])
            self.best_val_loss = float(pkg.get("best_val_loss", self.best_val_loss))

        if exists(self.accelerator.scaler) and exists(pkg["scaler"]):
            self.accelerator.scaler.load_state_dict(pkg["scaler"])

        print(f"loaded checkpoint from {self.checkpoint_folder / loaded_file_name}")

    def next_data_to_forward_kwargs(self, dl_iter) -> dict:
        data = next(dl_iter)

        if isinstance(data, tuple):
            forward_kwargs = dict(zip(self.data_kwargs, data))

        elif isinstance(data, dict):
            forward_kwargs = data

        return forward_kwargs

    def log_loss(self, loss, loss_dict, cur_lr=None, total_norm=None, step=None):
        log_data = {"total_loss": loss}

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

    def train_step(self, forward_kwargs):

        if isinstance(forward_kwargs, dict):
            loss, loss_dict = self.model(**forward_kwargs, return_loss_breakdown=True)
        elif isinstance(forward_kwargs, torch.Tensor):
            loss, loss_dict = self.model(
                forward_kwargs,
                return_loss_breakdown=True,
            )
        else:
            raise ValueError(f"unknown forward_kwargs")

        return loss, loss_dict

    def train(self):
        step = self.step.item()
        dl_iter = cycle(self.dataloader)

        if self.is_main and self.should_validate:
            val_dl_iter = cycle(self.val_dataloader)

        while step < self.num_train_steps:
            for i in range(self.grad_accum_every):
                is_last = i == (self.grad_accum_every - 1)
                maybe_no_sync = (
                    partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext
                )

                forward_kwargs = self.next_data_to_forward_kwargs(dl_iter)

                with self.accelerator.autocast(), maybe_no_sync():
                    loss, loss_dict = self.train_step(forward_kwargs)

                    self.accelerator.backward(loss / self.grad_accum_every)

            if divisible_by(step, self.log_every_step) and self.is_main:
                cur_lr = get_lr(self.optimizer.optimizer)

                total_norm = self.optimizer.total_norm

                self.log_loss(loss, loss_dict, cur_lr, total_norm, step)

            self.optimizer.step()
            self.optimizer.zero_grad()

            step += 1
            self.step.add_(1)

            self.wait()

            if self.is_main:
                self.ema_model.update()

            self.wait()

            if self.is_main and self.should_validate and divisible_by(step, self.val_every_step):
                total_val_loss = 0.0
                total_val_recon_loss = 0.0
                total_val_commit_loss = 0.0
                self.ema_model.eval()

                num_val_batches = self.val_num_batches * self.grad_accum_every

                for _ in range(num_val_batches):
                    with self.accelerator.autocast(), torch.no_grad():
                        forward_kwargs = self.next_data_to_forward_kwargs(val_dl_iter)
                        forward_kwargs = {k: v.to(self.device) for k, v in forward_kwargs.items()}

                        val_loss, val_loss_breakdown = self.ema_model(
                            **forward_kwargs, return_loss_breakdown=True
                        )

                        val_recon_loss = val_loss_breakdown["recon_loss"]
                        val_commit_loss = val_loss_breakdown["commit_loss"]

                        total_val_loss += val_loss / num_val_batches
                        total_val_recon_loss += val_recon_loss / num_val_batches
                        total_val_commit_loss += val_commit_loss / num_val_batches

                current_time = get_current_time()

                self.print(
                    current_time
                    + f" valid total loss: {total_val_loss:.3f} | recon loss: {total_val_recon_loss:.3f} | commit loss: {total_val_commit_loss:.3f}"
                )

                append_metrics(
                    self.metrics_path,
                    step=step,
                    epoch=step / self.num_step_per_epoch,
                    split="val",
                    metrics={
                        "val_loss": total_val_loss,
                        "val_recon_loss": total_val_recon_loss,
                        "val_commit_loss": total_val_commit_loss,
                    },
                )
                self.log(val_loss=total_val_loss)

                current_val_loss = float(total_val_loss.detach().cpu().item())
                if current_val_loss < self.best_val_loss:
                    self.best_val_loss = current_val_loss
                    self.save_best(current_val_loss)
                    self.print(
                        get_current_time()
                        + f" best checkpoint updated at {self.best_checkpoint_path} (val_loss={current_val_loss:.3f})"
                    )

            self.wait()

            if self.is_main and (
                divisible_by(step, self.checkpoint_every_step) or step >= self.num_train_steps
            ):
                checkpoint_num = (
                    step + self.checkpoint_every_step - 1
                ) // self.checkpoint_every_step

                milestone = str(checkpoint_num).zfill(2)

                self.save(milestone)

                self.print(
                    get_current_time()
                    + f" checkpoint saved at {self.checkpoint_folder / f'model-{milestone}.pt'}"
                )

            if divisible_by(step, self.num_step_per_epoch):
                if self.is_main:
                    print(get_current_time() + " dataset one epoch at ", step)

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
TokenizerTrainer = TokenizerTrainingLoop
