"""Crop image classifier served from a TFLite model.

Train the model with scripts/train_crop_classifier.py on the
src/Components/Agricultural-crops photo set. Until a model file exists the
endpoint answers 503 rather than guessing.
"""
import io
import json
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.config import get_settings


class ModelNotReady(RuntimeError):
    pass


class Classifier:
    def __init__(self, model_path: str, labels_path: str):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:  # pragma: no cover
            from tflite_runtime.interpreter import Interpreter  # type: ignore
        meta = json.loads(Path(labels_path).read_text())
        self.labels: list[str] = meta["labels"]
        self.version: str = meta.get("version", "unknown")
        self.size: int = int(meta.get("image_size", 224))
        self._interp = Interpreter(model_path=model_path)
        self._interp.allocate_tensors()
        self._in = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]
        self._lock = threading.Lock()  # TFLite interpreters are not thread-safe

    def predict(self, image_bytes: bytes, top_k: int = 3) -> list[tuple[str, float]]:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((self.size, self.size))
        x = np.asarray(img, dtype=np.float32)[None, ...]  # model includes its own preprocessing layer
        with self._lock:
            self._interp.set_tensor(self._in["index"], x)
            self._interp.invoke()
            probs = self._interp.get_tensor(self._out["index"])[0]
        order = np.argsort(probs)[::-1][:top_k]
        return [(self.labels[i], float(probs[i])) for i in order]


_classifier: Classifier | None = None
_lock = threading.Lock()


def get_classifier() -> Classifier:
    global _classifier
    with _lock:
        if _classifier is None:
            s = get_settings()
            if not (Path(s.vision_model_path).exists() and Path(s.vision_labels_path).exists()):
                raise ModelNotReady("The crop classifier has not been trained yet.")
            _classifier = Classifier(s.vision_model_path, s.vision_labels_path)
        return _classifier
