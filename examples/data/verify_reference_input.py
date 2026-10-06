"""Check pinned original upstream teaching data before optional R regeneration."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('rda',type=Path);a=p.parse_args()
lock=json.loads(Path(__file__).with_name('reference.lock.json').read_text())
actual=hashlib.sha256(a.rda.read_bytes()).hexdigest();expected=lock['fixture']['sha256']
if actual!=expected:raise SystemExit(f'FAIL original fixture SHA256: expected {expected}, got {actual}')
print(f'PASS original fixture SHA256 {actual}')
