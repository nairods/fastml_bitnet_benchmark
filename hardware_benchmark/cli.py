import argparse
import json
from pathlib import Path

import numpy as np

from .bitnet import (
    export_hls_project,
    export_hls_project_v26_style,
    load_quantized_layers,
    predict_exact,
    predict_v26_style,
)
from .lowering import export_pytorch_lowering_package
from .manifest import refresh_transfer_manifest, verify_transfer_manifest
from .metrics import compare_predictions
from .native import export_pytorch_hls
from .plan import write_plan
from .preflight import write_preflight
from .prepare import prepare_all
from .reports import parse_csynth_xml
from .runner import synthesize_one


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")


def command_validate_bitnet(args):
    state_path = ROOT / "onnx" / "hardware" / f"{args.run_name}_quantized.pt"
    layers = load_quantized_layers(state_path)
    inputs = np.load(ROOT / "data" / "synthesis" / "x_test.npy", mmap_mode="r")
    labels = np.load(ROOT / "data" / "synthesis" / "y_test.npy", mmap_mode="r")
    reference = np.load(
        ROOT / "data" / "synthesis" / "reference_predictions" / f"{args.run_name}.npy",
        mmap_mode="r",
    )
    limit = min(args.samples, len(inputs)) if args.samples else len(inputs)
    predictor = predict_v26_style if args.v26_style else predict_exact
    logits = predictor(layers, inputs[:limit], batch_size=args.batch_size)
    result = compare_predictions(logits, reference[:limit], labels[:limit])
    result.update(
        {
            "run_name": args.run_name,
            "implementation": (
                "cumulative_alpha_v26_style"
                if args.v26_style
                else "faithful_dynamic_activation_quantization"
            ),
            "state": str(state_path.relative_to(ROOT)),
        }
    )
    if args.output:
        _write_json(Path(args.output), result)
    print(json.dumps(result, indent=2))
    if not result["finite"]:
        raise SystemExit(2)


def command_export_bitnet(args):
    state_path = ROOT / "onnx" / "hardware" / f"{args.run_name}_quantized.pt"
    layers = load_quantized_layers(state_path)
    if args.v26_style:
        metadata = export_hls_project_v26_style(
            layers,
            Path(args.output),
            args.project_name or args.run_name.replace("-", "_") + "_v26",
            args.part,
            args.clock,
        )
    else:
        metadata = export_hls_project(
            layers,
            Path(args.output),
            args.project_name or args.run_name.replace("-", "_"),
            args.part,
            args.clock,
            dynamic_activation=not args.simplified,
        )
    print(json.dumps(metadata, indent=2))


def command_export_lowering(args):
    checkpoint = ROOT / "models" / f"{args.run_name}.pt"
    graph = export_pytorch_lowering_package(checkpoint, Path(args.output))
    print(json.dumps(graph, indent=2))


def command_export_dense(args):
    checkpoint = ROOT / "models" / f"{args.run_name}.pt"
    result = export_pytorch_hls(
        checkpoint, Path(args.output), args.part, args.clock
    )
    print(json.dumps(result, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Prepare and validate fair hls4ml/Vivado benchmark projects."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify-manifest")
    verify.set_defaults(
        function=lambda _args: print(json.dumps(verify_transfer_manifest(ROOT), indent=2))
    )

    refresh = commands.add_parser("refresh-manifest")
    refresh.set_defaults(
        function=lambda _args: print(
            f"Checksummed {len(refresh_transfer_manifest(ROOT)['files'])} files"
        )
    )

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--output", default="results/hardware_preflight.json")
    preflight.set_defaults(
        function=lambda args: print(
            json.dumps(write_preflight(ROOT, ROOT / args.output), indent=2)
        )
    )

    plan = commands.add_parser("build-plan")
    plan.add_argument("--output", default="results/synthesis_plan_v2.json")
    plan.set_defaults(
        function=lambda args: print(
            f"Wrote {len(write_plan(ROOT, ROOT / args.output))} experiments"
        )
    )

    validate = commands.add_parser("validate-bitnet")
    validate.add_argument("--run-name", required=True)
    validate.add_argument("--samples", type=int, default=16384)
    validate.add_argument("--batch-size", type=int, default=4096)
    validate.add_argument("--output")
    validate.add_argument("--v26-style", action="store_true")
    validate.set_defaults(function=command_validate_bitnet)

    bitnet = commands.add_parser("export-bitnet")
    bitnet.add_argument("--run-name", required=True)
    bitnet.add_argument("--output", required=True)
    bitnet.add_argument("--project-name")
    bitnet.add_argument("--part", default="xcvu13p-flga2577-2-e")
    bitnet.add_argument("--clock", type=float, default=5.0)
    bitnet.add_argument("--simplified", action="store_true")
    bitnet.add_argument("--v26-style", action="store_true")
    bitnet.set_defaults(function=command_export_bitnet)

    lowering = commands.add_parser("export-lowering")
    lowering.add_argument("--run-name", required=True)
    lowering.add_argument("--output", required=True)
    lowering.set_defaults(function=command_export_lowering)

    dense = commands.add_parser("export-dense")
    dense.add_argument("--run-name", required=True)
    dense.add_argument("--output", required=True)
    dense.add_argument("--part", default="xcvu13p-flga2577-2-e")
    dense.add_argument("--clock", type=float, default=5.0)
    dense.set_defaults(function=command_export_dense)

    prepare = commands.add_parser("prepare-all")
    prepare.add_argument("--output", default="hls_projects")
    prepare.add_argument("--validation-samples", type=int, default=16384)
    prepare.set_defaults(
        function=lambda args: print(
            json.dumps(
                prepare_all(ROOT, ROOT / args.output, args.validation_samples),
                indent=2,
            )
        )
    )

    synthesize = commands.add_parser("synthesize")
    synthesize.add_argument("--run-name", required=True)
    synthesize.add_argument("--confirm-synthesis", action="store_true")
    synthesize.add_argument("--allow-unverified-license", action="store_true")
    synthesize.set_defaults(
        function=lambda args: print(
            json.dumps(
                synthesize_one(
                    ROOT,
                    args.run_name,
                    args.confirm_synthesis,
                    args.allow_unverified_license,
                ),
                indent=2,
            )
        )
    )

    report = commands.add_parser("parse-report")
    report.add_argument("xml")
    report.set_defaults(
        function=lambda args: print(
            json.dumps(parse_csynth_xml(Path(args.xml)), indent=2)
        )
    )
    return parser


def main():
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
