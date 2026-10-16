"""Packaged toy examples for offline use and installation checks."""

import hashlib
import io
import json
from importlib.resources import files

import numpy as np


def load_example():
    """Load the packaged 500-variant synthetic toy example.

    Returns
    -------
    dict
        A new mapping with ``inputs`` (z of shape (500,), signed R of shape
        (500, 500), n = 1000 and variant IDs, suitable for :func:`susie_rss`),
        ``parameters`` (the fitting options of the frozen reference) and
        ``metadata`` (generation method, seed, synthetic coordinates and true
        effects). Arrays are copies owned by the caller.

    Raises
    ------
    ValueError
        A packaged input or metadata file does not match its recorded SHA-256
        checksum.

    Notes
    -----
    The data are simulated; no human study or genome build is implied. The
    loader reads installed package resources and needs neither network access
    nor R.

    Examples
    --------
    >>> example = load_example()
    >>> example["inputs"]["R"].shape
    (500, 500)
    """
    directory = files("prusie").joinpath("data/teaching")
    hashes = json.loads(directory.joinpath("checksums.json").read_text())
    blobs = {
        name: directory.joinpath(name).read_bytes() for name in ("inputs.npz", "metadata.json")
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
    return dict(inputs=inputs, parameters=metadata["parameters"].copy(), metadata=metadata)


def load_convergence_example():
    """Load the 200-variant sufficient statistics of the susieR convergence example.

    Returns
    -------
    dict
        ``inputs`` (XtX, Xty, yty and n for :func:`susie_suff_stat`),
        ``parameters``, ``metadata`` (source and license) and ``reference``
        (frozen arrays of the corresponding susieR fits).

    Raises
    ------
    ValueError
        A packaged file does not match its recorded SHA-256 checksum.

    Notes
    -----
    The statistics derive from the ``FinemappingConvergence`` data of susieR
    (BSD-3-Clause) and contain no individual genotypes. Variant IDs are
    positional; physical coordinates and alleles are not available. The
    reference arrays are unmodified susieR fits. prusie refinement differs
    from that of susieR in two respects: excluded credible-set variants
    receive exactly zero prior probability, and candidates are compared by
    the objective recomputed after final trimming.
    """
    directory = files("prusie").joinpath("data/convergence")
    manifest = json.loads(directory.joinpath("checksums.json").read_text())
    blobs = {name: directory.joinpath(name).read_bytes() for name in manifest}
    for name, blob in blobs.items():
        if hashlib.sha256(blob).hexdigest() != manifest[name]:
            raise ValueError("Convergence toy checksum mismatch: " + name)
    with np.load(io.BytesIO(blobs["inputs.npz"]), allow_pickle=False) as data:
        inputs = {k: data[k].copy() for k in ("XtX", "Xty", "yty", "n")}
        inputs["yty"] = float(inputs["yty"])
        inputs["n"] = float(inputs["n"])
    with np.load(io.BytesIO(blobs["reference.npz"]), allow_pickle=False) as data:
        reference = {k: data[k].copy() for k in data.files}
    metadata = json.loads(blobs["metadata.json"])
    return dict(
        inputs=inputs, parameters=metadata["parameters"], metadata=metadata, reference=reference
    )
