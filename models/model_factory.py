from typing import Dict, Any
from models.speech_enhancement.phonetic_aware_demucs import PhoneticAwareDemucs


class ModelFactory:

    supported_models: Dict[str, Any] = {
        "phonetic_aware_demucs": PhoneticAwareDemucs,
    }

    @staticmethod
    def get_model(args):
        if args.model not in ModelFactory.supported_models.keys():
            raise ValueError(f"Model: {args.model} is not supported by ModelFactory.\nPlease make sure implementation is valid.")
        return ModelFactory.supported_models[args.model](args)