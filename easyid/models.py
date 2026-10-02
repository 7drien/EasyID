"""
Gestionnaire et téléchargeur automatique des modèles de Deep Learning MediaPipe.
"""

import os
import urllib.request
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

MODELS_CACHE_DIR = Path.home() / ".cache" / "easyid" / "models"

MODEL_URLS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "selfie_segmenter.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
}


def get_model_path(model_name: str) -> str:
    """
    Récupère le chemin d'accès local au modèle, en le téléchargeant si nécessaire.
    """
    if model_name not in MODEL_URLS:
        raise ValueError(f"Modèle inconnu : {model_name}. Disponibles: {list(MODEL_URLS.keys())}")

    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    destination = MODELS_CACHE_DIR / model_name

    if not destination.exists() or destination.stat().st_size == 0:
        url = MODEL_URLS[model_name]
        logger.info(f"Téléchargement du modèle {model_name} depuis {url}...")
        temp_dest = destination.with_suffix(".tmp")
        try:
            urllib.request.urlretrieve(url, temp_dest)
            temp_dest.replace(destination)
            logger.info(f"Modèle {model_name} enregistré dans {destination}")
        except Exception as e:
            if temp_dest.exists():
                temp_dest.unlink()
            raise RuntimeError(f"Échec du téléchargement du modèle {model_name}: {e}")

    return str(destination)
