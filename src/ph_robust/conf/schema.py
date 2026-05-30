# src/ph_robust/conf/schema.py
from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class DataConfig:
    name: str = MISSING
    hf_id: str = MISSING
    num_classes: int = MISSING
    image_size: int = 64
    path: str = "./data"
    num_versions: int = 10


@dataclass
class ModelConfig:
    kind: str = MISSING
    hidden_size: int = 256
    freeze_weights: bool = False
    tbs: str = "normal"
    ft_attn: bool = False
    fg: bool = False


@dataclass
class OptimConfig:
    name: str = "adam"
    lr: float = 3e-4
    lr_b: float = 3e-5
    weight_decay: float = 1e-4
    scheduler: str = "cosine"


@dataclass
class TopoConfig:
    vectorization: str = "pi"
    dim: int = 1
    concat: bool = False
    tbs: str = "normal"
    resolution: int = 64
    bandwidth: float = 5.0
    max_norm: bool = False


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 64
    num_workers: int = 2
    val_size: float = 0.2
    compile: bool = True
    save_model: bool = False


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    topo: TopoConfig = field(default_factory=TopoConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    run_name: str = ""
    seed: int = 119
