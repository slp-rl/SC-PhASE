from dataclasses import dataclass


@dataclass
class DemucsConfig:
    chin: int = 1
    chout: int = 1
    hidden: int = 48
    depth: int = 5
    kernel_size: int = 8
    stride: int = 4
    causal: bool = True
    resample: int = 4
    growth: int = 2
    max_hidden: int = 10000
    normalize: bool = True
    glu: bool = True
    rescale: float = 0.1
    floor: float = 1e-3
    sample_rate: int = 16000
