from dataclasses import dataclass


@dataclass
class FeaturesConfig:
    include_ft: bool = False
    feature_model: str = 'hubert'
    state_dict_path: str = '/cs/labs/adiyoss/shared/pretrained_weights/hubert/hubert_base_ls960.pt'
    features_factor: float = 0.01
    features_dim: int = 768  # 768 for hubert and 512 for ASR
    get_ft_after_lstm: bool = True
    use_as_conditioning: bool = False
    merge_method: str = 'inter'
    features_dim_for_conditioning: int = 128
    use_as_supervision: bool = False
    supervision_factor: float = 0.01
    device: str = 'cpu'
    layer: int = 9
    layers: list = None
    learnable: bool = False
