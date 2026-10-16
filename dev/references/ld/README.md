# R reference values for LD diagnostics

These files retain the recorded values from susieR 0.16.6. Each JSON wrapper
identifies the source record and its SHA-256; `results` contains the reference
used by the tests. `ld_oracles.json` records conditional z statistics and
mixture likelihoods. `kriging_certified_oracle.json` also checks optimality of
the mixture weights. The tests distinguish a failed default mixture optimum
from the certified solution; see `tests/test_ld_diagnostics.py` for tolerances.

Extract `results` to a new local directory before running the two optional tests:

```sh
mkdir -p reference-check
python - <<'PYTHON'
import json
from pathlib import Path
for name in ("ld_oracles.json", "kriging_certified_oracle.json"):
    source = Path("dev/references/ld") / name
    (Path("reference-check") / name).write_text(
        json.dumps(json.loads(source.read_text())["results"]) + "\n"
    )
PYTHON
PRUSIE_LD_ORACLE="$PWD/reference-check/ld_oracles.json" \
PRUSIE_LD_CERTIFIED_ORACLE="$PWD/reference-check/kriging_certified_oracle.json" \
python -m pytest tests/test_ld_diagnostics.py
```

This reads the saved R values; it does not require an R installation.
