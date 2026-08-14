import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path


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


def run_preflight() -> dict:
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": {
            name: _package_version(name)
            for name in ("hls4ml", "conifer", "torch", "tensorflow", "qkeras", "HGQ")
        },
        "vivado": _command_version(
            "vivado", ["-version"], os.environ.get("XILINX_VIVADO")
        ),
        "vitis_hls": _command_version(
            "vitis_hls", ["-version"], os.environ.get("XILINX_VITIS")
        ),
        "license": _license_status(),
    }
