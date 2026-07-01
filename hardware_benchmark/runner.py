import json
import os
import subprocess
from pathlib import Path

from .preflight import run_preflight


def _record(root: Path, run_name: str) -> dict:
    preparation = json.loads(
        (root / "hls_projects" / "preparation.json").read_text(encoding="utf-8")
    )
    for record in preparation["records"]:
        if record["run_name"] == run_name:
            return record
    raise KeyError(f"No prepared record for {run_name}")


def synthesize_one(
    root: Path,
    run_name: str,
    confirm: bool,
    allow_unverified_license: bool,
) -> dict:
    if not confirm:
        raise RuntimeError("Synthesis requires --confirm-synthesis")
    record = _record(root, run_name)
    if record["status"] != "project_generated":
        raise RuntimeError(
            f"{run_name} is not synthesis-ready: {record['status']}"
        )
    validation = record.get("validation")
    if not validation or not validation.get("finite"):
        raise RuntimeError(f"{run_name} has no passing numerical validation")

    preflight = run_preflight(root)
    if not preflight["vitis_hls"]["available"]:
        raise RuntimeError("Vitis HLS is unavailable")
    if not preflight["license"]["available"] and not allow_unverified_license:
        raise RuntimeError(
            "Xilinx license availability is unverified; pass "
            "--allow-unverified-license only after checking the site license."
        )

    project_dir = root / "hls_projects" / run_name / "native"
    if (project_dir / "run_hls.tcl").exists():
        script = "run_hls.tcl"
    elif (project_dir / "build_prj.tcl").exists():
        script = "build_prj.tcl"
    else:
        raise FileNotFoundError(f"No HLS Tcl script under {project_dir}")
    environment = os.environ.copy()
    environment["PATH"] = (
        str(Path(preflight["vitis_hls"]["path"]).parent)
        + os.pathsep
        + environment.get("PATH", "")
    )
    log_path = project_dir / "synthesis.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [preflight["vitis_hls"]["path"], "-f", script],
            cwd=project_dir,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    status = {
        "run_name": run_name,
        "returncode": result.returncode,
        "log": str(log_path.relative_to(root)),
        "synthesis_run": True,
    }
    (project_dir / "synthesis_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    if result.returncode:
        raise RuntimeError(f"Synthesis failed; see {log_path}")
    return status
