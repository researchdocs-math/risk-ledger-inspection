"""Check rerun metrics and saved traces against immutable reference outputs."""
from pathlib import Path
import json, numpy as np, pandas as pd
from pandas.api.types import is_numeric_dtype
ROOT = Path(__file__).resolve().parent

def main():
    checks = []
    for name in ["main.csv", "replays.csv"]:
        expected = pd.read_csv(ROOT / "reference_results" / name)
        actual = pd.read_csv(ROOT / "results" / name)
        assert list(expected.columns) == list(actual.columns), name + " columns"
        assert expected.shape == actual.shape, name + " shape"
        for col in expected:
            a, b = actual[col], expected[col]
            if is_numeric_dtype(b):
                np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-10, equal_nan=True, err_msg=name + ":" + col)
            else:
                assert a.fillna("").equals(b.fillna("")), name + ":" + col
        checks.append({"file": name, "rows": len(actual), "columns": len(actual.columns), "match": True})
    count = 0
    for reference in sorted((ROOT / "reference_results").glob("trace_*.npz")):
        with np.load(reference) as b, np.load(ROOT / "results" / reference.name) as a:
            assert set(a.files) == set(b.files)
            for key in b.files:
                np.testing.assert_allclose(a[key], b[key], rtol=1e-9, atol=1e-10, err_msg=reference.name + ":" + key)
        count += 1
    report = {"metric_checks": checks, "matching_trace_files": count, "status": "passed"}
    (ROOT / "results" / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
if __name__ == "__main__": main()
