# Rename to prusie

The repository, distribution and Python import are now all **prusie**. The
renamed distribution keeps version **0.2.3rc4**; its name and artifact hashes
distinguish it from the earlier `pyrsusie 0.2.3rc4` distribution. Install the
provided local wheel or source archive using its actual filename, or run
`python -m pip install .` from this repository. No PyPI publication is implied.

```python
import prusie
from importlib.metadata import version

assert prusie.__version__ == version("prusie")
from prusie import _native
print(_native.backend_version())
```

Change old `import pyrsusie` statements to `import prusie`, and imports from
its submodules likewise. There are no compatibility alias packages. Use a
fresh environment to avoid accidentally retaining the old distribution.
The native module is `prusie._native`, the Cargo package is `prusie-native`,
and new fit backend strings start with `prusie-rust/`.

Rename environment settings to `PRUSIE_NUM_THREADS` and `PRUSIE_SER_MATH`.
Their behavior is unchanged; old environment variable names are not aliases.
Public functions, arguments, numerical arrays and result fields are unchanged.
Use the accompanying **pycoloc 0.2.2rc3** for the updated optional fitting example.

## Saved results and prior evidence

`SusieResult` is now defined in `prusie.result`. Python pickles store module
paths, so old class pickles referring to `pyrsusie.result` will not load in a
clean prusie-only environment. The package does not remap old pickle imports.
For existing trusted results, use the original environment to export the
documented arrays and metadata, then consume those portable data in the new
environment. Avoid unpickling data from untrusted sources. New prusie result
pickles use the new module path; portable arrays plus JSON remain preferable.

Frozen R outputs, teaching inputs, native goldens, source hashes and historical
performance records retain their original bytes and labels. The example checker
reads the old `runtime.pyrsusie` key solely to identify the preserved native
snapshot; all newly generated runtime metadata uses `prusie`. This metadata
adapter neither imports the old package nor changes numerical comparisons.

The R-agreement report and its downloadable records describe the pre-rename
binaries. The 118/200 timing majority remains that historical measurement;
the renamed binary has not undergone another timing campaign. B01 remains
excluded. This naming change introduces no optimization, statistical change,
approximation or tolerance change.

## Rename verification scope

Release checks cover isolated wheel and source-distribution installations,
native loading, absent alias imports, distribution identity, the existing full
test suites, offline 500-SNP examples and optional native pair, and both static
documentation sites. Source comparison permits only the declared name symbols
and pycoloc version update. PIP acceptance stays atol=1e-5, rtol=0; pycoloc
retains its frozen comparison tolerances. The prior independent statistical
and engineering reviews remain prior-version evidence; this mechanical rename
review is not a new independent statistical sign-off.

## License status

The user accepted the existing **GPL-3.0-or-later** package licensing after
initially requesting MIT. Existing notices and upstream license/version terms
remain intact; this rename does not relicense third-party code or data.
The concrete retained provenance includes the native R 4.4 `Brent_fmin`
adaptation (GPL-2-or-later), pycoloc's adaptation from GPL coloc R files, and
the GPL coloc teaching fixture. Mathematical-method agreement alone is not
the reason for those notices. The earlier MIT preference is resolved by the
explicit GPL decision; no licensing rewrite is part of this release.
