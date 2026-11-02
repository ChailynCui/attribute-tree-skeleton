from __future__ import annotations
import hashlib
from typing import Iterable

ARTIFACT_VERSION = "four_species_single_seed_control_artifacts_v2"
ContractError = ValueError


def deterministic_derangement(indices: Iterable[int], *, species: str, seed: int) -> dict[int, int]:
    group = sorted(int(index) for index in indices)
    if len(group) < 2:
        raise ContractError(f"species {species} needs at least two train rows")

    def priority(index: int) -> bytes:
        return hashlib.sha256(
            f"{ARTIFACT_VERSION}\0{seed}\0{species}\0{index}".encode("utf-8")
        ).digest()

    order = sorted(group, key=lambda index: (priority(index), index))
    shift_digest = hashlib.sha256(
        f"{ARTIFACT_VERSION}\0{seed}\0{species}\0shift".encode("utf-8")
    ).digest()
    shift = 1 + int.from_bytes(shift_digest[:8], "big") % (len(order) - 1)
    mapping = {
        recipient: order[(position + shift) % len(order)]
        for position, recipient in enumerate(order)
    }
    if sorted(mapping) != group or sorted(mapping.values()) != group:
        raise AssertionError("derangement is not a bijection")
    if any(recipient == donor for recipient, donor in mapping.items()):
        raise AssertionError("derangement contains a fixed point")
    return mapping
