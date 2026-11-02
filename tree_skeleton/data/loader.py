# SPDX-License-Identifier: MIT
# Adapted research implementation; see README.md for license notices.
import pickle
import json
import hashlib
import torch
from torch import is_tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np


def scale_and_jitter(vertices, interval=(0.9, 1.1), jitter=0.1):
    assert isinstance(interval, tuple)

    scale = torch.rand(1, 3) * (interval[1] - interval[0]) + interval[0]
    vertices = vertices * scale

    jitter = (torch.rand(1, 3) * 2 - 1) * jitter
    vertices = vertices + jitter

    vertices = torch.clamp(vertices, min=-0.99, max=0.99)

    return vertices


def xy_dihedral_augment(vertices):
    transform_id = int(torch.randint(0, 8, (1,)).item())
    xy = vertices[:, :2]

    if transform_id % 4 == 1:
        xy = torch.stack((-xy[:, 1], xy[:, 0]), dim=-1)
    elif transform_id % 4 == 2:
        xy = -xy
    elif transform_id % 4 == 3:
        xy = torch.stack((xy[:, 1], -xy[:, 0]), dim=-1)

    if transform_id >= 4:
        xy = torch.stack((-xy[:, 0], xy[:, 1]), dim=-1)

    return torch.cat((xy, vertices[:, 2:]), dim=-1)


# dataset


class SkeletonDataset(Dataset):
    def __init__(
        self,
        dataset_folder="data",
        dataset_file_path="",
        is_train=True,
        transform=scale_and_jitter,
        replica=10,
        attribute_json_path="",
        attribute_names=(),
        attribute_input_type="bucket",
        attribute_bucket_key="species_relative_bucket",
        attribute_value_key="percentile",
        attribute_value_quantization="none",
        species_name="",
        species_names=(),
        training_condition_mode="correct",
        training_condition_seed=0,
        training_condition_fixed_value=0.5,
        training_condition_fixed_id=1,
        split_manifest_path="",
        split_name="",
        condition_schema=None,
        text_embedding_path="",
        text_embedding_manifest_path="",
        text_embedding_seed=0,
    ):
        self.dataset_folder = dataset_folder
        self.data_path = dataset_file_path
        self.is_train = is_train
        self.transform = transform
        self.replica = replica
        self.attribute_json_path = attribute_json_path
        self.attribute_names = tuple(attribute_names)
        self.attribute_input_type = attribute_input_type
        self.attribute_bucket_key = attribute_bucket_key
        self.attribute_value_key = attribute_value_key
        self.attribute_value_quantization = attribute_value_quantization
        self.species_name = species_name
        self.species_names = tuple(species_names)
        self.training_condition_mode = training_condition_mode
        self.training_condition_seed = int(training_condition_seed)
        self.training_condition_fixed_value = float(training_condition_fixed_value)
        self.training_condition_fixed_id = int(training_condition_fixed_id)
        self.split_manifest_path = split_manifest_path
        self.split_name = split_name
        self.condition_schema = dict(condition_schema or {})
        self.text_embedding_path = text_embedding_path
        self.text_embedding_manifest_path = text_embedding_manifest_path
        self.text_embedding_seed = int(text_embedding_seed)

        valid_condition_modes = {
            "correct",
            "fixed",
            "shuffle_joint",
            "shuffle_joint_within_species",
            "shuffle_independent",
        }
        if self.training_condition_mode not in valid_condition_modes:
            raise ValueError(
                f"unknown training_condition_mode: {self.training_condition_mode}; "
                f"expected one of {sorted(valid_condition_modes)}"
            )
        if self.attribute_value_quantization not in {"none", "tertile_centers"}:
            raise ValueError(
                "attribute_value_quantization must be one of ['none', 'tertile_centers']"
            )
        if len(set(self.species_names)) != len(self.species_names):
            raise ValueError("species_names must contain unique names in ID order")
        if any(not isinstance(name, str) or not name for name in self.species_names):
            raise ValueError("species_names must contain non-empty strings")

        self._validate_condition_schema_binding()

        (
            self.data,
            self.species_ids,
            self.attribute_ids,
            self.attribute_values,
            self.text_embeddings,
        ) = self._load_data()

    def _validate_condition_schema_binding(self):
        """Ensure a control checkpoint schema is bound to these input files."""
        if self.condition_schema.get("version") != "four_species_control_condition_v2":
            return
        required = (
            "condition_artifact_role",
            "condition_artifact_sha256",
            "split_manifest_sha256",
            "control_manifest_sha256",
        )
        missing = [key for key in required if not self.condition_schema.get(key)]
        if missing:
            raise ValueError(
                "control condition schema is incomplete; missing " + ", ".join(missing)
            )
        if not self.attribute_json_path or not self.split_manifest_path:
            raise ValueError(
                "control condition schema requires attribute_json_path and split_manifest_path"
            )
        with open(self.attribute_json_path, "rb") as handle:
            attribute_hash = hashlib.sha256(handle.read()).hexdigest()
        with open(self.split_manifest_path, "rb") as handle:
            split_hash = hashlib.sha256(handle.read()).hexdigest()
        if attribute_hash != self.condition_schema.get("condition_artifact_sha256"):
            raise ValueError("condition artifact hash does not match the checkpoint schema")
        if split_hash != self.condition_schema.get("split_manifest_sha256"):
            raise ValueError("condition split manifest hash does not match the checkpoint schema")

    @staticmethod
    def _ordered_name_sha256(data):
        names = []
        for index, sample in enumerate(data):
            name = sample.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"dataset sample {index} has no stable string name for split validation"
                )
            names.append(name)
        return hashlib.sha256(("\n".join(names) + "\n").encode("utf-8")).hexdigest()

    def _load_split_indices(self, data):
        if not self.split_manifest_path:
            return None

        expected_len = len(data)

        if not self.split_name:
            raise ValueError("split_name is required when split_manifest_path is set")

        with open(self.split_manifest_path, "r") as f:
            payload = json.load(f)

        declared_count = payload.get("sample_count")
        if declared_count is None:
            declared_count = payload.get("dataset", {}).get("sample_count")
        if declared_count is not None and int(declared_count) != expected_len:
            raise ValueError(
                f"split manifest sample_count ({declared_count}) does not match "
                f"dataset sample count ({expected_len})"
            )

        expected_name_hash = payload.get("dataset", {}).get("ordered_name_sha256")
        if expected_name_hash:
            observed_name_hash = self._ordered_name_sha256(data)
            if observed_name_hash != expected_name_hash:
                raise ValueError(
                    "dataset ordered-name hash does not match split manifest; "
                    "the manifest belongs to a different dataset or ordering"
                )

        expected_attribute_hash = payload.get("attributes", {}).get("file_sha256")
        if expected_attribute_hash:
            if not self.attribute_json_path:
                raise ValueError("split manifest requires an attribute JSON file")
            with open(self.attribute_json_path, "rb") as f:
                observed_attribute_hash = hashlib.sha256(f.read()).hexdigest()
            if observed_attribute_hash != expected_attribute_hash:
                raise ValueError(
                    "attribute JSON hash does not match split manifest; "
                    "positional labels may be misaligned"
                )

        splits = payload.get("splits", payload)
        if self.split_name not in splits:
            raise ValueError(f"split {self.split_name!r} not found in {self.split_manifest_path}")

        indices = splits[self.split_name]
        if not isinstance(indices, list):
            raise ValueError(f"split {self.split_name!r} must be a JSON list of indices")

        normalized = []
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError(f"split index must be an integer, got {index!r}")
            if index < 0 or index >= expected_len:
                raise ValueError(
                    f"split index {index} is outside dataset range [0, {expected_len})"
                )
            normalized.append(index)

        if len(set(normalized)) != len(normalized):
            raise ValueError(f"split {self.split_name!r} contains duplicate indices")

        return normalized

    def _transform_training_conditions(
        self,
        attribute_ids,
        attribute_values,
        species_ids=None,
    ):
        """Apply deterministic negative-control labels to the training split only."""
        if not self.is_train or self.training_condition_mode == "correct":
            return attribute_ids, attribute_values

        values = attribute_values if attribute_values is not None else attribute_ids
        if values is None:
            return attribute_ids, attribute_values

        num_samples = len(values)
        if num_samples == 0:
            return attribute_ids, attribute_values

        if self.training_condition_mode == "fixed":
            if attribute_values is not None:
                attribute_values = [
                    [self.training_condition_fixed_value] * len(row) for row in attribute_values
                ]
            if attribute_ids is not None:
                attribute_ids = [
                    [self.training_condition_fixed_id] * len(row) for row in attribute_ids
                ]
            return attribute_ids, attribute_values

        rng = np.random.default_rng(self.training_condition_seed)

        if self.training_condition_mode in {
            "shuffle_joint",
            "shuffle_joint_within_species",
        }:
            if self.training_condition_mode == "shuffle_joint":
                permutation = rng.permutation(num_samples)
            else:
                if species_ids is None:
                    raise ValueError("shuffle_joint_within_species requires species IDs")
                if len(species_ids) != num_samples:
                    raise ValueError(
                        "species and attribute counts must match before joint shuffling"
                    )
                species_array = np.asarray(species_ids)
                permutation = np.arange(num_samples)
                for species_id in sorted(np.unique(species_array).tolist()):
                    group_indices = np.flatnonzero(species_array == species_id)
                    permutation[group_indices] = rng.permutation(group_indices)
            if attribute_values is not None:
                attribute_values = [attribute_values[int(index)] for index in permutation]
            if attribute_ids is not None:
                attribute_ids = [attribute_ids[int(index)] for index in permutation]
            return attribute_ids, attribute_values

        def independently_shuffle(rows):
            array = np.asarray(rows)
            shuffled = array.copy()
            for attribute_index in range(array.shape[1]):
                permutation = rng.permutation(num_samples)
                shuffled[:, attribute_index] = array[permutation, attribute_index]
            return shuffled.tolist()

        if attribute_values is not None:
            attribute_values = independently_shuffle(attribute_values)
        if attribute_ids is not None:
            attribute_ids = independently_shuffle(attribute_ids)
        return attribute_ids, attribute_values

    def _load_attributes(self, expected_len):
        if not self.attribute_json_path:
            if self.species_names:
                raise ValueError("attribute_json_path is required when species_names is configured")
            return None, None, None
        if not self.attribute_names and not self.species_names:
            return None, None, None

        if self.attribute_input_type not in {"bucket", "continuous"}:
            raise ValueError(f"unknown attribute_input_type: {self.attribute_input_type}")

        with open(self.attribute_json_path, "r") as f:
            payload = json.load(f)

        samples = payload["samples"]
        if self.species_name:
            samples = [sample for sample in samples if sample.get("species") == self.species_name]

        if len(samples) != expected_len:
            raise ValueError(
                f"attribute sample count ({len(samples)}) does not match dataset sample count ({expected_len})"
            )

        species_ids = None
        if self.species_names:
            species_to_id = {name: index for index, name in enumerate(self.species_names)}
            declared_vocabulary = payload.get("species_vocabulary")
            if declared_vocabulary is not None and declared_vocabulary != species_to_id:
                raise ValueError(
                    "attribute JSON species_vocabulary does not match configured "
                    f"species_names order: {self.species_names}"
                )
            species_ids = []
            for sample_index, sample in enumerate(samples):
                species = sample.get("species")
                if species not in species_to_id:
                    raise ValueError(
                        f"attribute sample {sample_index} has unknown species "
                        f"{species!r}; expected one of {self.species_names}"
                    )
                species_id = species_to_id[species]
                declared_species_id = sample.get("species_id")
                if declared_species_id is not None and int(declared_species_id) != species_id:
                    raise ValueError(
                        f"attribute sample {sample_index} species_id "
                        f"{declared_species_id} conflicts with species {species!r} "
                        f"configured as {species_id}"
                    )
                species_ids.append(species_id)

        attribute_ids = (
            [] if self.attribute_names and self.attribute_input_type == "bucket" else None
        )
        attribute_values = (
            [] if self.attribute_names and self.attribute_input_type == "continuous" else None
        )
        if self.attribute_names:
            for sample in samples:
                attrs = sample["attributes"]
                if self.attribute_input_type == "bucket":
                    attribute_ids.append(
                        [
                            int(attrs[name][self.attribute_bucket_key])
                            for name in self.attribute_names
                        ]
                    )
                else:
                    attribute_values.append(
                        [
                            self._quantize_attribute_value(
                                float(attrs[name][self.attribute_value_key])
                            )
                            for name in self.attribute_names
                        ]
                    )

        return species_ids, attribute_ids, attribute_values

    def _load_text_embeddings(self, data):
        if self.text_embedding_path or self.text_embedding_manifest_path:
            raise ValueError("This paper release uses structured conditions, not text caches")
        return None

    def _quantize_attribute_value(self, value):
        if self.attribute_value_quantization == "none":
            return value
        if value < (1.0 / 3.0):
            return 0.25
        if value < (2.0 / 3.0):
            return 0.5
        return 0.75

    def _load_data(self):
        data = []

        with open(self.data_path, "rb") as f:
            data = pickle.load(f)

        print(f"load {len(data)} data")
        species_ids, attribute_ids, attribute_values = self._load_attributes(len(data))
        text_embeddings = self._load_text_embeddings(data)

        split_indices = self._load_split_indices(data)
        if split_indices is not None:
            data = [data[index] for index in split_indices]
            if species_ids is not None:
                species_ids = [species_ids[index] for index in split_indices]
            if attribute_ids is not None:
                attribute_ids = [attribute_ids[index] for index in split_indices]
            if attribute_values is not None:
                attribute_values = [attribute_values[index] for index in split_indices]
            if text_embeddings is not None:
                text_embeddings = text_embeddings[split_indices]
            attribute_ids, attribute_values = self._transform_training_conditions(
                attribute_ids,
                attribute_values,
                species_ids,
            )
            return data, species_ids, attribute_ids, attribute_values, text_embeddings

        split_at = int(len(data) * 0.9)
        if self.is_train:
            if species_ids is not None:
                species_ids = species_ids[:split_at]
            if attribute_ids is not None:
                attribute_ids = attribute_ids[:split_at]
            if attribute_values is not None:
                attribute_values = attribute_values[:split_at]
            if text_embeddings is not None:
                text_embeddings = text_embeddings[:split_at]
            data = data[:split_at]
            attribute_ids, attribute_values = self._transform_training_conditions(
                attribute_ids,
                attribute_values,
                species_ids,
            )
        else:
            if species_ids is not None:
                species_ids = species_ids[split_at:]
            if attribute_ids is not None:
                attribute_ids = attribute_ids[split_at:]
            if attribute_values is not None:
                attribute_values = attribute_values[split_at:]
            if text_embeddings is not None:
                text_embeddings = text_embeddings[split_at:]
            data = data[split_at:]

        return data, species_ids, attribute_ids, attribute_values, text_embeddings

    def __len__(self):
        if self.is_train != True:
            return len(self.data)
        else:
            return len(self.data) * self.replica

    def __getitem__(self, idx):
        expanded_idx = int(idx)
        idx = expanded_idx % len(self.data)

        sample = self.data[idx]

        vertices = sample["vertices"]

        segments = sample["lines"].astype(np.int32)

        vertices = torch.from_numpy(vertices).float()
        segments = torch.from_numpy(segments).long()

        # transform

        if self.transform:
            vertices = self.transform(vertices)

        data = dict()

        data["vertices"] = vertices
        data["segments"] = segments

        if self.species_ids is not None:
            # Keep a one-element sequence so custom_collate produces [B, 1].
            data["species_id"] = torch.tensor([self.species_ids[idx]], dtype=torch.long)
        if self.attribute_ids is not None:
            data["attribute_ids"] = torch.tensor(self.attribute_ids[idx], dtype=torch.long)
        if self.attribute_values is not None:
            data["attribute_values"] = torch.tensor(self.attribute_values[idx], dtype=torch.float)
        if self.text_embeddings is not None:
            variant_index = 0
            if self.is_train:
                replica_index = expanded_idx // len(self.data)
                variant_index = (
                    idx + replica_index + self.text_embedding_seed
                ) % self.text_embeddings.shape[1]
            embedding = np.array(
                self.text_embeddings[idx, variant_index],
                dtype=np.float32,
                copy=True,
            )
            data["text_embeddings"] = torch.from_numpy(embedding)

        return data


# custom collater


def first(it):
    return it[0]


def custom_collate(data, pad_id=-1):
    is_dict = isinstance(first(data), dict)

    if is_dict:
        keys = first(data).keys()
        data = [d.values() for d in data]

    output = []

    for datum in zip(*data):
        if is_tensor(first(datum)):
            datum = pad_sequence(datum, batch_first=True, padding_value=pad_id)
        else:
            datum = list(datum)

        output.append(datum)

    output = tuple(output)

    if is_dict:
        output = dict(zip(keys, output))

    return output
