import logging
from models.representation_models.ASR_wrapper import AsrFeatExtractor
from models.representation_models.HuBERT_wrapper import huBERT

logger = logging.getLogger(__name__)


def load_lexical_model(model_name, lexical_path, device="cuda", sr=16000, layer=6):
    if model_name.lower() == 'hubert':
        ret = huBERT(lexical_path, layer)
        ret.model.to(device)
        return ret
    elif model_name.lower() == "asr":
        return AsrFeatExtractor(device, sr=sr)
    else:
        logger.error("Unknown model.")