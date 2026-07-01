import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_transfer_manifest(root: Path) -> dict:
    path = root / "results" / "hardware_transfer_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    missing = []
    size_mismatch = []
    checksum_mismatch = []
    for relative, expected in manifest["files"].items():
        artifact = root / relative
        if not artifact.exists():
            missing.append(relative)
            continue
        if artifact.stat().st_size != expected["bytes"]:
            size_mismatch.append(relative)
            continue
        if sha256(artifact) != expected["sha256"]:
            checksum_mismatch.append(relative)
    return {
        "manifest": str(path.relative_to(root)),
        "checked": len(manifest["files"]),
        "missing": missing,
        "size_mismatch": size_mismatch,
        "checksum_mismatch": checksum_mismatch,
        "valid": not (missing or size_mismatch or checksum_mismatch),
    }


def refresh_transfer_manifest(root: Path) -> dict:
    path = root / "results" / "hardware_transfer_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    existing = {root / relative for relative in manifest["files"]}
    roots = [
        root / "hardware_benchmark",
        root / "hls_projects",
        root / "tests",
        root / "results" / "synthesis",
    ]
    additions = {
        file
        for directory in roots
        if directory.exists()
        for file in directory.rglob("*")
        if file.is_file() and "__pycache__" not in file.parts
    }
    additions.update(
        {
            root / "README.md",
            root / "run_hardware_benchmark.py",
            root / "configs" / "hardware_benchmark.json",
            root / "results" / "hardware_preflight.json",
            root / "results" / "synthesis_plan_v2.json",
            root / "results" / "hardware_implementation_status.md",
        }
    )
    files = sorted(file for file in existing | additions if file.exists())
    manifest["implementation"] = {
        "driver": "run_hardware_benchmark.py",
        "prepared_projects": "hls_projects/preparation.json",
        "synthesis_run": False,
    }
    manifest["files"] = {
        str(file.relative_to(root)): {
            "bytes": file.stat().st_size,
            "sha256": sha256(file),
        }
        for file in files
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
