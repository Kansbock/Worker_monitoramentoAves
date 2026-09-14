import io

import numpy as np
import pytest
from PIL import Image

from feeder_worker.classifier import BirdClassifier, InvalidImageError


class _StubModel:
    """Simula um modelo Keras: `.output_shape` e `.predict()` determinísticos."""

    def __init__(self, num_classes: int, predictions: list[float]):
        self.output_shape = (None, num_classes)
        self._predictions = np.array([predictions], dtype=np.float32)
        self.last_input = None

    def predict(self, array, verbose=0):
        self.last_input = array
        return self._predictions


def _jpeg_bytes(size=(50, 50), color=(10, 20, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_classify_returns_argmax_class_and_confidence():
    model = _StubModel(num_classes=3, predictions=[0.05, 0.9, 0.05])
    classifier = BirdClassifier.from_loaded_model(model, num_classes=3)

    result = classifier.classify(_jpeg_bytes())

    assert result.class_index == 1
    assert result.confidence == pytest.approx(0.9)
    assert result.all_confidences == pytest.approx([0.05, 0.9, 0.05])


def test_classify_resizes_and_does_not_normalize_pixels():
    model = _StubModel(num_classes=3, predictions=[1.0, 0.0, 0.0])
    classifier = BirdClassifier.from_loaded_model(model, num_classes=3)

    classifier.classify(_jpeg_bytes(size=(600, 400), color=(255, 0, 0)))

    fed_array = model.last_input
    assert fed_array.shape == (1, 299, 299, 3)
    assert fed_array.dtype == np.float32
    # Pixels devem permanecer em escala 0-255 (sem normalização adicional):
    # o preprocess_input do InceptionV3 já está embutido no modelo.
    assert fed_array.max() > 1.0


def test_classify_raises_invalid_image_error_for_corrupted_bytes():
    model = _StubModel(num_classes=3, predictions=[1.0, 0.0, 0.0])
    classifier = BirdClassifier.from_loaded_model(model, num_classes=3)

    with pytest.raises(InvalidImageError):
        classifier.classify(b"not a real image")


def test_constructor_validates_output_shape_matches_catalog():
    model = _StubModel(num_classes=2, predictions=[0.5, 0.5])

    with pytest.raises(ValueError, match="produz 2 classes"):
        BirdClassifier.from_loaded_model(model, num_classes=3)


def test_load_model_raises_file_not_found_for_missing_path(tmp_path):
    missing_path = tmp_path / "does-not-exist.keras"

    with pytest.raises(FileNotFoundError):
        BirdClassifier(model_path=missing_path, num_classes=3)
