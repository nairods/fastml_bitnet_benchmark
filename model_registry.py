import sys
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: Callable
    description: str
    backend: str = "pytorch"
    available: bool = True
    unavailable_reason: str = ""


MODEL_REGISTRY: Dict[str, ModelSpec] = {}


def register(name, description, backend="pytorch"):
    def decorator(builder):
        MODEL_REGISTRY[name] = ModelSpec(name, builder, description, backend)
        return builder

    return decorator


def register_unavailable(name, description, backend, reason):
    MODEL_REGISTRY[name] = ModelSpec(
        name=name,
        builder=lambda *_args, **_kwargs: None,
        description=description,
        backend=backend,
        available=False,
        unavailable_reason=reason,
    )


def _bitlinear_classes():
    candidates = [
        (ROOT / "../hep_bitnet").resolve(),
        (ROOT / "../../hep_bitnet").resolve(),
    ]
    package_root = next((path for path in candidates if path.exists()), None)
    if package_root is None:
        locations = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"hep_bitnet not found in any of: {locations}")
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from bitnet.bitlinear import BitLinear, BitLinear158b

    return BitLinear, BitLinear158b


class DenseMLP(nn.Module if nn is not None else object):
    def __init__(self, input_dim, output_dim, hidden_dims, layer_factory):
        super().__init__()
        dimensions = [input_dim, *hidden_dims, output_dim]
        self.layers = nn.ModuleList(
            layer_factory(left, right)
            for left, right in zip(dimensions[:-1], dimensions[1:])
        )

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)


def _dense_builder(hidden_dims):
    return lambda input_dim, output_dim, _config: DenseMLP(
        input_dim, output_dim, hidden_dims, nn.Linear
    )


def _bit_builder(hidden_dims, ternary=False):
    def builder(input_dim, output_dim, config):
        bitlinear, bitlinear158 = _bitlinear_classes()
        layer_class = bitlinear158 if ternary else bitlinear
        options = config["model"].get("bitnet", {})

        def factory(left, right):
            return layer_class(
                left,
                right,
                b=options.get("activation_bits", 8),
                frac_bits=options.get("frac_bits"),
                beta_quant=options.get("beta_quant"),
                beta_shift_min=options.get("beta_shift_min"),
                beta_shift_max=options.get("beta_shift_max"),
                quant_export=False,
            )

        return DenseMLP(input_dim, output_dim, hidden_dims, factory)

    return builder


@register("mlp_baseline", "Dense MLP: 16-64-32-32-5")
def build_mlp_baseline(input_dim, output_dim, config):
    hidden_dims = config["model"].get("hidden_dims", [64, 32, 32])
    return DenseMLP(input_dim, output_dim, hidden_dims, nn.Linear)


@register("mlp_topo", "TorchTOPO dense MLP: 16-128-32-5")
def build_mlp_topo(input_dim, output_dim, _config):
    return DenseMLP(input_dim, output_dim, [128, 32], nn.Linear)


@register("bitnet_mlp", "Configurable binary BitLinear MLP")
def build_bitnet_mlp(input_dim, output_dim, config):
    hidden_dims = config["model"].get("hidden_dims", [64, 32, 32])
    return _bit_builder(hidden_dims)(input_dim, output_dim, config)


@register("bitnet_topo", "BitTOPO binary BitLinear MLP: 16-128-32-5")
def build_bitnet_topo(input_dim, output_dim, config):
    return _bit_builder([128, 32])(input_dim, output_dim, config)


@register("bit158_mlp", "Configurable ternary BitLinear158b MLP")
def build_bit158_mlp(input_dim, output_dim, config):
    hidden_dims = config["model"].get("hidden_dims", [64, 32, 32])
    return _bit_builder(hidden_dims, ternary=True)(input_dim, output_dim, config)


@register("bit158_topo", "BitLinear158b TOPO MLP: 16-128-32-5")
def build_bit158_topo(input_dim, output_dim, config):
    return _bit_builder([128, 32], ternary=True)(input_dim, output_dim, config)


@register("binary_large", "Binary BitLinear MLP: 16-448-224-224-5")
def build_binary_large(input_dim, output_dim, config):
    return _bit_builder([448, 224, 224])(input_dim, output_dim, config)


@register("ternary_large", "Ternary BitLinear158b MLP: 16-128-64-64-64-5")
def build_ternary_large(input_dim, output_dim, config):
    return _bit_builder([128, 64, 64, 64], ternary=True)(
        input_dim, output_dim, config
    )


class FeatureTokenizer(nn.Module if nn is not None else object):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, embedding_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class DeepSetsHLF(nn.Module if nn is not None else object):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.tokenizer = FeatureTokenizer(input_dim, 64)
        self.phi = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, output_dim))

    def forward(self, x):
        return self.rho(self.phi(self.tokenizer(x)).mean(dim=1))


@register(
    "deepsets_hlf",
    "Tabular adaptation of Deep Sets using 16 learned feature tokens",
)
def build_deepsets(input_dim, output_dim, _config):
    return DeepSetsHLF(input_dim, output_dim)


class MixerBlock(nn.Module if nn is not None else object):
    def __init__(self, token_count, embedding_dim, token_hidden, channel_hidden):
        super().__init__()
        self.token_norm = nn.LayerNorm(embedding_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(token_count, token_hidden),
            nn.GELU(),
            nn.Linear(token_hidden, token_count),
        )
        self.channel_norm = nn.LayerNorm(embedding_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(embedding_dim, channel_hidden),
            nn.GELU(),
            nn.Linear(channel_hidden, embedding_dim),
        )

    def forward(self, x):
        x = x + self.token_mlp(self.token_norm(x).transpose(1, 2)).transpose(1, 2)
        return x + self.channel_mlp(self.channel_norm(x))


class MLPMixerHLF(nn.Module if nn is not None else object):
    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        options = config["model"].get("mixer", {})
        embedding_dim = options.get("embedding_dim", 32)
        blocks = options.get("blocks", 2)
        self.tokenizer = FeatureTokenizer(input_dim, embedding_dim)
        self.blocks = nn.Sequential(
            *[
                MixerBlock(
                    input_dim,
                    embedding_dim,
                    options.get("token_hidden", 32),
                    options.get("channel_hidden", 64),
                )
                for _ in range(blocks)
            ]
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        return self.head(self.norm(self.blocks(self.tokenizer(x))).mean(dim=1))


@register(
    "mlp_mixer_hlf",
    "Tabular 16-token MLP-Mixer adaptation; not the particle-sequence paper model",
)
def build_mixer(input_dim, output_dim, config):
    return MLPMixerHLF(input_dim, output_dim, config)


class TransformerHLF(nn.Module if nn is not None else object):
    def __init__(self, input_dim, output_dim, config, variant):
        super().__init__()
        options = config["model"].get("transformer", {})
        embedding_dim = options.get("embedding_dim", 32)
        heads = options.get("heads", 4)
        layers = options.get("layers", 2)
        feedforward_dim = options.get("feedforward_dim", 64)
        dropout = options.get("dropout", 0.0)
        self.variant = variant
        self.tokenizer = FeatureTokenizer(input_dim, embedding_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        if variant == "linformer":
            projection_size = options.get("projection_size", 8)
            self.projection = nn.Parameter(
                torch.empty(input_dim + 1, projection_size)
            )
            nn.init.xavier_uniform_(self.projection)
            token_count = projection_size
        else:
            self.projection = None
            token_count = input_dim + 1
        self.position = nn.Parameter(torch.zeros(1, token_count, embedding_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        tokens = self.tokenizer(x)
        cls = self.cls.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        if self.projection is not None:
            tokens = torch.einsum("btd,tp->bpd", tokens, self.projection)
        encoded = self.encoder(tokens + self.position)
        pooled = encoded[:, 0] if self.projection is None else encoded.mean(dim=1)
        return self.head(self.norm(pooled))


@register(
    "multihead_attention_hlf",
    "Tabular 16-token multi-head self-attention adaptation",
)
def build_mha(input_dim, output_dim, config):
    return TransformerHLF(input_dim, output_dim, config, "mha")


@register(
    "linformer_hlf",
    "Tabular low-rank token-projection transformer adaptation",
)
def build_linformer(input_dim, output_dim, config):
    return TransformerHLF(input_dim, output_dim, config, "linformer")


@register(
    "jetformer_hlf",
    "Compact encoder-only JetFormer-style HLF adaptation",
)
def build_jetformer(input_dim, output_dim, config):
    return TransformerHLF(input_dim, output_dim, config, "jetformer")


MODEL_REGISTRY["qkeras_mlp"] = ModelSpec(
    name="qkeras_mlp",
    builder=lambda *_args, **_kwargs: None,
    description="QKeras quantized dense MLP: 16-64-32-32-5",
    backend="qkeras",
)
if importlib.util.find_spec("xgboost") is None:
    register_unavailable(
        "xgboost_bdt",
        "XGBoost boosted decision tree baseline on the same 16 HLF inputs",
        "xgboost",
        "The xgboost package is not installed in the current environment.",
    )
else:
    MODEL_REGISTRY["xgboost_bdt"] = ModelSpec(
        name="xgboost_bdt",
        builder=lambda *_args, **_kwargs: None,
        description="XGBoost boosted decision tree baseline on the same 16 HLF inputs",
        backend="xgboost",
    )
if importlib.util.find_spec("HGQ") is None:
    register_unavailable(
        "hgq_mlp",
        "HGQ high-granularity quantized dense reference model",
        "hgq",
        "The HGQ package is not installed in the current environment.",
    )
else:
    MODEL_REGISTRY["hgq_mlp"] = ModelSpec(
        name="hgq_mlp",
        builder=lambda *_args, **_kwargs: None,
        description="HGQ high-granularity quantized MLP: 16-64-32-32-5",
        backend="hgq",
    )


LEGACY_MODEL_NAMES = {
    "baseline": "mlp_baseline",
    "bitnet": "bitnet_mlp",
}


def resolve_model_name(config):
    model = config["model"]
    return model.get("name") or LEGACY_MODEL_NAMES.get(model.get("type"), model.get("type"))


def build_registered_model(config, input_dim, output_dim):
    name = resolve_model_name(config)
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {name!r}. Available: {', '.join(sorted(MODEL_REGISTRY))}"
        )
    spec = MODEL_REGISTRY[name]
    if not spec.available:
        raise RuntimeError(f"Model {name!r} is unavailable: {spec.unavailable_reason}")
    if spec.backend != "pytorch":
        raise RuntimeError(f"Model {name!r} requires backend {spec.backend!r}")
    return spec.builder(input_dim, output_dim, config)


def registry_rows():
    return [
        {
            "name": spec.name,
            "backend": spec.backend,
            "available": spec.available,
            "description": spec.description,
            "unavailable_reason": spec.unavailable_reason,
        }
        for spec in sorted(MODEL_REGISTRY.values(), key=lambda item: item.name)
    ]
