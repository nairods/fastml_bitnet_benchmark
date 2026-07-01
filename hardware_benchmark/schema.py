from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    base_run_name: str
    representative_run: str
    conversion_route: str
    precision_policy: str
    part: str
    clock_period_ns: float
    io_type: str
    reuse_factor: int
    input_data: str
    labels: str
    reference_predictions: str
    tier: str = "controlled"
    seed: int = 42

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "Experiment":
        policy = values["precision_policy"]
        tier = values.get("tier") or ("native" if policy == "native" else "controlled")
        selected = {
            key: values[key]
            for key in cls.__dataclass_fields__
            if key in values and key != "tier"
        }
        return cls(
            **selected,
            tier=tier,
        )

    def resolve(self, root: Path, field_name: str) -> Path:
        return root / getattr(self, field_name)


@dataclass
class NumericalResult:
    samples: int
    finite: bool
    maximum_absolute_error: float
    mean_absolute_error: float
    class_agreement: float
    accuracy: float
    reference_accuracy: float
    accuracy_delta: float
    macro_auc: float
    reference_macro_auc: float
    macro_auc_delta: float


@dataclass
class HardwareResult:
    experiment_id: str
    status: str
    route: str
    tier: str
    tool: str = ""
    tool_version: str = ""
    part: str = ""
    clock_target_ns: Optional[float] = None
    clock_achieved_ns: Optional[float] = None
    latency_cycles_min: Optional[int] = None
    latency_cycles_max: Optional[int] = None
    initiation_interval_cycles: Optional[int] = None
    lut: Optional[int] = None
    ff: Optional[int] = None
    dsp: Optional[int] = None
    bram: Optional[float] = None
    uram: Optional[int] = None
    numerical: Optional[NumericalResult] = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
