from dataclasses import dataclass
from typing import Any, Dict
import yaml

@dataclass
class Config:
    raw: Dict[str, Any]
    def __getitem__(self, k): return self.raw[k]

def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(raw=data)
