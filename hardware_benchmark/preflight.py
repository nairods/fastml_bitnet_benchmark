import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .artifacts import readiness_rows
from .manifest import verify_transfer_manifest


def _command_version(command, arguments, environment_root=None):
    executable = shutil.which(command)
    if not executable and environment_root:
        candidate = Path(environment_root) / "bin" / command
        if candidate.exists():
            executable = str(candidate)
    if not executable:
        return {"available": False, "path": None, "version": ""}
    result = subprocess.run(
        [executable, *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "available": result.returncode == 0,
        "path": executable,
        "version": result.stdout.strip().splitlines()[0] if result.stdout else "",
    }


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _license_status():
    variables = {
        key: os.environ.get(key)
        for key in ("XILINXD_LICENSE_FILE", "LM_LICENSE_FILE")
        if os.environ.get(key)
    }
    lmutil = shutil.which("lmutil")
    checked = []
    if lmutil:
        for variable, servers in variables.items():
            result = subprocess.run(
                [lmutil, "lmstat", "-a", "-c", servers],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=30,
            )
            checked.append(
                {
                    "variable": variable,
                    "returncode": result.returncode,
                    "xilinx_features_seen": any(
                        token in result.stdout.lower()
                        for token in ("vivado", "vitis", "synthesis")
                    ),
                }
            )
    return {
        "environment": variables,
        "lmutil": lmutil,
        "checks": checked,
        "available": any(item["xilinx_features_seen"] for item in checked),
        "conclusive": bool(checked),
    }


def _route_status(row):
    route = row["conversion_route"]
    if route == "ONNX/PyTorch":
        return {"state": "ready", "implementation": "direct_pytorch"}
    if "BitNet" in route or "binary" in route or "ternary" in route:
        return {
            "state": "ready",
            "implementation": "custom_bitnet_dynamic_quantizer",
            "warning": "The transferred hardware ONNX is approximate; use the quantized state with the custom route.",
        }
    if route == "native QKeras":
        return {
            "state": "ready_in_hlsenv310",
            "implementation": "qkeras_hls4ml",
            "warning": "Use hlsenv310; the default hlsenv has an incompatible Keras import.",
        }
    if route == "native HGQ":
        return {
            "state": "ready_in_hlsenv310",
            "implementation": "hgq_hls4ml",
            "warning": "Restore Keras variables explicitly, calibrate min/max, then use HGQ to_proxy_model.",
        }
    return {
        "state": "lowering_required",
        "implementation": "custom_operator_package",
        "reason": "Stock hls4ml fails at the learned feature tokenizer.",
    }


def run_preflight(root: Path) -> dict:
    routes = {
        row["representative_run"]: _route_status(row)
        for row in readiness_rows(root)
    }
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": {
            name: _package_version(name)
            for name in ("hls4ml", "torch", "tensorflow", "qkeras", "HGQ", "onnx", "onnxruntime")
        },
        "vivado": _command_version(
            "vivado", ["-version"], os.environ.get("XILINX_VIVADO")
        ),
        "vitis_hls": _command_version(
            "vitis_hls", ["-version"], os.environ.get("XILINX_VITIS")
        ),
        "license": _license_status(),
        "manifest": verify_transfer_manifest(root),
        "routes": routes,
    }


def write_preflight(root: Path, output: Path) -> dict:
    report = run_preflight(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
