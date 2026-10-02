"""
Module de segmentation et de remplacement d'arrière-plan avec MediaPipe ImageSegmenter.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from easyid.config import DEFAULT_CONFIG, IDPhotoConfig
from easyid.models import get_model_path


@dataclass
class SegmentationResult:
    mask: np.ndarray  # Masque alpha flottant (H, W, 1) compris entre 0.0 et 1.0
    segmented_image: np.ndarray  # Image avec arrière-plan remplacé


class BackgroundSegmenter:
    """Segmenteur de premier plan et remplacement d'arrière-plan réglementaire."""

    def __init__(self, config: IDPhotoConfig = DEFAULT_CONFIG):
        self.config = config
        model_path = get_model_path("selfie_segmenter.tflite")
        options = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            output_confidence_masks=True,
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options)

    def extract_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Extrait le masque alpha du premier plan (personne = 1.0, fond = 0.0).
        Retourne un tableau numpy (H, W, 1) float32 dans [0.0, 1.0].
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.segmenter.segment(mp_image)

        if not result.confidence_masks or len(result.confidence_masks) == 0:
            # Fallback : masque uniforme à 1.0
            return np.ones((image_bgr.shape[0], image_bgr.shape[1], 1), dtype=np.float32)

        raw_mask = result.confidence_masks[0].numpy_view()

        # Nettoyage et lissage léger des bords
        mask = np.clip(raw_mask, 0.0, 1.0).astype(np.float32)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)

        # Flou gaussien subtil sur le masque pour adoucir les transitions
        mask_2d = cv2.GaussianBlur(mask[:, :, 0], (3, 3), 0)
        return np.expand_dims(mask_2d, axis=-1).astype(np.float32)

    def replace_background(
        self,
        image_bgr: np.ndarray,
        mask: Optional[np.ndarray] = None,
        bg_color_bgr: Optional[Tuple[int, int, int]] = None,
    ) -> SegmentationResult:
        """
        Remplace l'arrière-plan de l'image par une couleur unie neutre.
        """
        if mask is None:
            mask = self.extract_mask(image_bgr)

        if bg_color_bgr is None:
            bg_color_bgr = self.config.DEFAULT_BG_COLOR_BGR

        bg = np.full_like(image_bgr, bg_color_bgr, dtype=np.uint8)

        # Alpha blending
        alpha = mask
        fg_float = image_bgr.astype(np.float32)
        bg_float = bg.astype(np.float32)

        blended = fg_float * alpha + bg_float * (1.0 - alpha)
        blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)

        return SegmentationResult(mask=mask, segmented_image=blended_uint8)
