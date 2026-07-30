"""Why do 85 of 86 recordings yield no segments?

Dumps how events are actually stored across several files, not just the first.
Also checks the PSQI sheet for the global-score column.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

R = Path(sys.argv[1])
files = sorted((R / "EEG_downsample").glob("*.mat"))
print(f"{len(files)} EEG files\n")

for f in [files[0], files[1], files[2], files[10], files[40], files[-1]]:
    print("=" * 62)
    print(f.name)
    try:
        m = loadmat(f, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print("  loadmat FAILED:", e)
        continue
    print("  top-level keys:", [k for k in m if not k.startswith("__")])
    if "EEG" not in m:
        continue
    E = m["EEG"]
    print(f"  srate={getattr(E,'srate','?')}  "
          f"data={getattr(getattr(E,'data',None),'shape','?')}")

    for field in ("urevent", "event"):
        v = getattr(E, field, None)
        print(f"  {field}: {type(v).__name__} shape={getattr(v,'shape','-')}")
        if v is None:
            continue
        try:
            arr = np.atleast_1d(v)
        except Exception as e:
            print(f"     cannot atleast_1d: {e}")
            continue
        if len(arr) == 0:
            print("     EMPTY")
            continue
        e0 = arr[0]
        flds = [a for a in dir(e0) if not a.startswith("_")]
        print(f"     n={len(arr)} fields={flds}")
        for e in arr[:12]:
            row = {}
            for a in flds:
                try:
                    val = getattr(e, a)
                    row[a] = (val, type(val).__name__)
                except Exception:
                    row[a] = "ERR"
            print("      ", row)

print("=" * 62)
print("PSQI columns (looking for a global/total score)")
try:
    d = pd.read_excel(R / "scale.xlsx", sheet_name="PSQI")
    for c in d.columns:
        num = pd.api.types.is_numeric_dtype(d[c])
        print(f"  [{'num' if num else 'txt'}] {c}")
except Exception as e:
    print("  !", e)

print("=" * 62)
print("SDS columns")
try:
    d = pd.read_excel(R / "scale.xlsx", sheet_name="SDS")
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]):
            print(f"  [num] {c}")
except Exception as e:
    print("  !", e)