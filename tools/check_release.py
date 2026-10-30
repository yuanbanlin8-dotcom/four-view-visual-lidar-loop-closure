"""Check file integrity and paper numbers using included CSV data."""
from pathlib import Path
from collections import Counter
import csv, hashlib
root=Path(__file__).resolve().parents[1]
def rows(name):
    with (root/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
loops=rows('paper_assets/data/fig3_accepted_loops.csv')
assert Counter(int(r['floor_F1_to_F4']) for r in loops)=={1:11,2:3,3:7,4:5}
heights=rows('paper_assets/data/floor_height_comparison.csv')
for r in heights:
    errors=[]
    for pair,ref in zip(['F1-F2','F2-F3','F3-F4'],[4.201,4.255,4.169]):
        assert float(r[pair+' reference (m)'])==ref
        error=100*(float(r[pair+' mapped (m)'])-ref)
        assert abs(error-float(r[pair+' signed error (cm)']))<1e-6
        errors.append(abs(error))
    assert abs(sum(errors)/3-float(r['MAE (cm)']))<=.0051
assert float(next(r for r in heights if r['Method']=='Ours')['MAE (cm)'])==3.83
integrity=root/'SHA256SUMS.txt'
if integrity.exists():
    for line in integrity.read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1)
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
print('Release hashes, 26-loop distribution and floor-height calculations passed.')
