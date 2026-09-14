"""Classificação de imagens com o modelo treinado (InceptionV3, .keras).

O modelo já embute a camada de pré-processamento InceptionV3
(`preprocess_input`), conforme o notebook de treinamento fornecido. Portanto
o worker apenas decodifica a imagem, converte para RGB, redimensiona para o
tamanho de entrada do modelo (299x299) e envia os pixels em escala 0-255,
sem normalização adicional — replicar uma normalização aqui duplicaria o
pré-processamento e degradaria a predição.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

INPUT_SIZE = (299, 299)  # (largura, altura), conforme IMG_SIZE no notebook de treinamento


class InvalidImageError(Exception):
    """Levantado quando os bytes recebidos não podem ser decodificados como
    imagem válida (arquivo corrompido ou formato não suportado)."""


@dataclass(frozen=True)
class ClassificationResult:
    class_index: int
    confidence: float
    all_confidences: list[float]


class BirdClassifier:
    """Encapsula o modelo Keras carregado uma única vez por processo."""

    def __init__(self, model_path: Path, num_classes: int):
        self._model_path = model_path
        self._num_classes = num_classes
        self._model = self._load_model(model_path)
        self._validate_output_shape()

    @classmethod
    def from_loaded_model(cls, model, num_classes: int) -> "BirdClassifier":
        """Constrói o classificador a partir de um modelo já carregado (ou de
        um stub/fake), sem tocar o disco nem importar TensorFlow. Usado nos
        testes para isolar a lógica de decodificação/pré-processamento do
        carregamento real do modelo."""
        instance = cls.__new__(cls)
        instance._model_path = None
        instance._num_classes = num_classes
        instance._model = model
        instance._validate_output_shape()
        return instance

    def _load_model(self, model_path: Path):
        # Import tardio: TensorFlow é pesado para importar e não deve ser
        # carregado só para inspecionar configuração (ex.: comandos de bootstrap/seed).
        import tensorflow as tf

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Modelo não encontrado em {model_path}. Configure MODEL_PATH corretamente."
            )
        logger.info("Carregando modelo Keras de %s", model_path)
        return tf.keras.models.load_model(model_path)

    def _validate_output_shape(self) -> None:
        output_shape = self._model.output_shape
        num_outputs = output_shape[-1]
        if num_outputs != self._num_classes:
            raise ValueError(
                f"O modelo produz {num_outputs} classes, mas o catálogo define "
                f"{self._num_classes}. Verifique MODEL_PATH e o catálogo de espécies."
            )

    def classify(self, image_bytes: bytes) -> ClassificationResult:
        array = self._decode_and_preprocess(image_bytes)
        predictions = self._model.predict(array, verbose=0)[0]
        class_index = int(np.argmax(predictions))
        confidence = float(predictions[class_index])
        return ClassificationResult(
            class_index=class_index,
            confidence=confidence,
            all_confidences=[float(p) for p in predictions],
        )

    @staticmethod
    def _decode_and_preprocess(image_bytes: bytes) -> np.ndarray:
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                img = img.convert("RGB").resize(INPUT_SIZE)
                array = np.asarray(img, dtype=np.float32)
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError(f"Não foi possível decodificar a imagem: {exc}") from exc

        return np.expand_dims(array, axis=0)
