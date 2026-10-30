from __future__ import annotations
 
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import get_type_hints

from .decay_sampling import DecayConfig
from .datasets import DatasetConfig
from .training import TrainingConfig, RegularizationConfig
from .recording import RecordingConfig
from .srnn import SRNNConfig

@dataclass
class Config:
    run_name: str
    training: TrainingConfig = field(default_factory=TrainingConfig)
    srnn: SRNNConfig = field(default_factory=SRNNConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
 
    # ---- (de)serialisation, only needed to pass a config between processes ----
    def to_dict(self) -> dict:
        return asdict(self)
 
    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        return _build(cls, data)
 
    def save_json(self, dir: Path) -> None:
        with open(dir / "config.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)
 
    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        with open(path) as f:
            return cls.from_dict(json.load(f))

def _build(cls, data: dict):
    """Recursively turn a (nested) dict back into nested dataclasses.
 
    Unknown keys raise an error instead of being silently ignored.
    """
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown key(s) {sorted(unknown)} for {cls.__name__}; "
                         f"allowed: {sorted(known)}")
    kwargs = {}
    for key, value in data.items():
        hint = hints[key]
        if is_dataclass(hint) and isinstance(value, dict):
            kwargs[key] = _build(hint, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)