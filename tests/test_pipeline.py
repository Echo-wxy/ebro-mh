import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_demo_runs_end_to_end(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_demo.py")],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (ROOT / "outputs" / "demo_metrics.csv").exists()
