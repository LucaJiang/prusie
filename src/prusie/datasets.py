"""Packaged toy examples for offline use and installation checks."""

from importlib.resources import files
import hashlib
import io
import json
import numpy as np


def load_example():
    """Load an independent 500-variant synthetic toy example from package resources.

    Returns
    -------
    dict
        New mapping with ``inputs`` (z of shape (500,), signed R of shape
        (500, 500), n=1000 and ordered variant IDs), ``parameters`` (explicit
        fitting options), and ``metadata`` (generation method, seed, synthetic
        coordinates and true effects). Arrays are owned by the caller.

    Raises
    ------
    ValueError
        A packaged input or metadata checksum differs from its frozen manifest.

    Notes
    -----
    No working-directory, network or R dependency is required. The synthetic
    response/predictor units are recorded in the metadata.
    """
    directory = files("prusie").joinpath("data/teaching")
    hashes = json.loads(directory.joinpath("checksums.json").read_text())
    blobs = {
        name: directory.joinpath(name).read_bytes()
        for name in ("inputs.npz", "metadata.json")
    }
    for name, blob in blobs.items():
        if hashlib.sha256(blob).hexdigest() != hashes[name]:
            raise ValueError(f"Packaged example checksum mismatch: {name}")
    metadata = json.loads(blobs["metadata.json"])
    with np.load(io.BytesIO(blobs["inputs.npz"]), allow_pickle=False) as data:
        inputs = dict(
            z=data["z"].copy(),
            R=data["R"].copy(),
            n=metadata["n"],
            variant_ids=np.asarray(metadata["variant_ids"], dtype=str),
        )
    return dict(
        inputs=inputs, parameters=metadata["parameters"].copy(), metadata=metadata
    )
