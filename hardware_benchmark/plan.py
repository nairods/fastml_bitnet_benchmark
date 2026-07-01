import json
from pathlib import Path

from .artifacts import readiness_rows


def build_plan(root: Path) -> list[dict]:
    protocol = json.loads(
        (root / "configs" / "hardware_benchmark.json").read_text(encoding="utf-8")
    )
    target = protocol["target"]
    widths = protocol["protocol"]["controlled_widths"]
    jobs = []
    for row in readiness_rows(root):
        common = {
            "base_run_name": row["base_run_name"],
            "representative_run": row["representative_run"],
            "conversion_route": row["conversion_route"],
            "part": target["part"],
            "clock_period_ns": target["clock_period_ns"],
            "io_type": target["io_type"],
            "reuse_factor": 1,
            "seed": 42,
            "input_data": "data/synthesis/x_test.npy",
            "labels": "data/synthesis/y_test.npy",
            "reference_predictions": row["reference_predictions"],
        }
        jobs.append(
            {
                **common,
                "experiment_id": f"{row['base_run_name']}__native",
                "precision_policy": "native",
                "tier": "native",
                "primary_comparison": True,
            }
        )
        for width in widths:
            jobs.append(
                {
                    **common,
                    "experiment_id": f"{row['base_run_name']}__controlled_{width}",
                    "precision_policy": f"controlled_{width}",
                    "tier": "controlled",
                    "primary_comparison": True,
                }
            )
        if "BitNet" in row["conversion_route"] or "binary" in row["conversion_route"] or "ternary" in row["conversion_route"]:
            for variant in ("faithful_zero_dsp", "faithful_low_dsp", "faithful_balanced"):
                jobs.append(
                    {
                        **common,
                        "experiment_id": f"{row['base_run_name']}__{variant}",
                        "precision_policy": "native",
                        "tier": "expert_custom",
                        "custom_variant": variant,
                        "primary_comparison": False,
                    }
                )
            jobs.append(
                {
                    **common,
                    "experiment_id": f"{row['base_run_name']}__simplified_v26_style",
                    "precision_policy": "native",
                    "tier": "expert_custom",
                    "custom_variant": "simplified_v26_style",
                    "primary_comparison": False,
                    "approximate_model": True,
                    "notes": "Omits per-layer dynamic activation quantization; never mix with faithful native results.",
                }
            )
    return jobs


def write_plan(root: Path, output: Path) -> list[dict]:
    jobs = build_plan(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
    return jobs
