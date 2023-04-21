from typing import TYPE_CHECKING

from transformers.file_utils import (
    _LazyModule,
    is_tokenizers_available,
    is_torch_available,
)

_import_structure = {
    "configuration_bros": ["BROS_PRETRAINED_CONFIG_ARCHIVE_MAP", "BrosConfig"],
    "tokenization_bros": ["BrosTokenizer"],
}

import os
import sys
MODEL_FOLDER, _ = os.path.split(os.path.realpath(__file__))
sys.path.insert(0, MODEL_FOLDER)

if is_tokenizers_available():
    _import_structure["tokenization_bros_fast"] = ["BrosTokenizerFast"]

if is_torch_available():
    _import_structure["modeling_bros"] = [
        "BROS_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BrosForMaskedLM",
        "BrosForPreTraining",
        "BrosForSequenceClassification",
        "BrosForTokenClassification",
        "BrosModel",
        "BrosLMHeadModel",
        "BrosPreTrainedModel",
    ]
    _import_structure["modeling_bros_convnext"] = [
        "GeoLayoutLMModel",
        "PairGeometricHead",
        "MultiPairsGeometricHead",
    ]

if TYPE_CHECKING:
    from .configuration_bros import BROS_PRETRAINED_CONFIG_ARCHIVE_MAP, BrosConfig
    from .tokenization_bros import BrosTokenizer

    if is_tokenizers_available():
        from .tokenization_bros_fast import BrosTokenizerFast

    if is_torch_available():
        from .modeling_bros import (
            BROS_PRETRAINED_MODEL_ARCHIVE_LIST,
            BrosForMaskedLM,
            BrosForPreTraining,
            BrosForSequenceClassification,
            BrosForTokenClassification,
            BrosLMHeadModel,
            BrosModel,
            BrosPreTrainedModel,
        )

        from .modeling_bros_convnext import(
            GeoLayoutLMModel,
            PairGeometricHead,
            MultiPairsGeometricHead,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure
    )
