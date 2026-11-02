"""Model assembly and strict loading of historical state dictionaries."""

from __future__ import annotations
import torch
from .tokenizer import TreeSegmentTokenizer
from .generator import AttributeTreeGenerator
from .checkpoint_keys import tokenizer_state, generator_state


def make_tokenizer(config):
    return TreeSegmentTokenizer(**config["tokenizer"])


def make_generator(config):
    return AttributeTreeGenerator(make_tokenizer(config), **config["generator"])


def load_tokenizer(model, path):
    package = torch.load(path, map_location="cpu", weights_only=False)
    if "ema_model" in package:
        # Match the online tokenizer used by the historical generator training.
        state = {
            k.removeprefix("online_model."): v
            for k, v in package["ema_model"].items()
            if k.startswith("online_model.")
        }
    else:
        state = package.get("model", package)
    model.load_state_dict(tokenizer_state(state), strict=True)
    return model


def load_generator(config, checkpoint, device="cpu"):
    model = make_generator(config)
    package = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(generator_state(package.get("model", package)), strict=True)
    schema = package.get("condition_schema")
    expected_scope = config["data"].get("coordinate_scale_fit", "train")
    saved_scope = (schema or {}).get("coordinate_scale_fit", "all")
    if saved_scope != expected_scope:
        raise ValueError(
            "Checkpoint coordinate fitting scope differs from configuration; use a checkpoint trained with the same coordinate protocol."
        )
    if expected_scope == "train":
        from tree_skeleton.io import fingerprint

        protocol = config["data"].get("coordinate_protocol")
        if not protocol or (schema or {}).get("coordinate_protocol_sha256") != fingerprint(
            protocol
        ):
            raise ValueError("Checkpoint is not bound to these train-fitted coordinate constants")
    if schema and schema.get("species_names") != list(config["data"]["species"]):
        raise ValueError("Checkpoint species order differs from the configuration")
    if schema and schema.get("attribute_names") != list(config["data"]["attribute_names"]):
        raise ValueError("Checkpoint attribute order differs from the configuration")
    if schema and schema.get("control_arm", config.get("control_arm", "main")) != config.get(
        "control_arm", "main"
    ):
        raise ValueError("Checkpoint control arm differs from the configuration")
    if schema and schema.get("attribute_value_key") != config["data"].get(
        "value_key", "percentile"
    ):
        raise ValueError("Checkpoint percentile encoding differs from the configuration")
    return model.to(device).eval(), package
