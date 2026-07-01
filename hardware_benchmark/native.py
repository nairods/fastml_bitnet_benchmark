from pathlib import Path


def export_pytorch_hls(checkpoint_path: Path, output_dir: Path, part: str, clock: float):
    import sys

    import hls4ml
    import torch

    root = checkpoint_path.parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from model_registry import build_registered_model

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    output_dim = int(
        checkpoint["metadata"].get(
            "output_dim", len(checkpoint["metadata"]["class_names"])
        )
    )
    model = build_registered_model(checkpoint["config"], 16, output_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    if (
        int(checkpoint["config"].get("model", {}).get("output_dim", output_dim)) == 1
        and checkpoint["config"].get("model", {}).get("output_mode") == "binary_sigmoid"
    ):
        model = torch.nn.Sequential(model, torch.nn.Sigmoid())
        model.eval()
    config = hls4ml.utils.config_from_pytorch_model(
        model,
        input_shape=(None, 16),
        granularity="name",
        backend="Vivado",
        default_precision="ap_fixed<16,5,AP_RND,AP_SAT>",
        default_reuse_factor=1,
    )
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model,
        output_dir=str(output_dir),
        project_name="jet_classifier",
        backend="Vivado",
        hls_config=config,
        part=part,
        clock_period=clock,
        io_type="io_parallel",
    )
    hls_model.write()
    return {
        "route": "direct_pytorch",
        "layers": len(hls_model.get_layers()),
        "output_dir": str(output_dir),
        "compiled": False,
        "synthesized": False,
    }
