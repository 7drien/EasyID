"""
Module de recadrage intelligent et transformation affine haute résolution.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

from easyid.config import DEFAULT_CONFIG, IDPhotoConfig
from easyid.core.detector import FaceDetection
from easyid.core.geometry import AffineTransformResult, compute_id_affine_transform


@dataclass
class CropResult:
    cropped_image: np.ndarray  # Image recadrée aux dimensions cibles (H, W, 3)
    affine_result: AffineTransformResult
    config: IDPhotoConfig
    dpi: int
    target_width_px: int
    target_height_px: int
    face_ratio: float  # Ratio de la hauteur du visage par rapport à la hauteur totale (ex: 0.755)
    transformed_keypoints: Optional[np.ndarray] = None


def apply_affine_crop(
    image_bgr: np.ndarray,
    matrix: np.ndarray,
    target_width: int,
    target_height: int,
    border_mode: int = cv2.BORDER_REFLECT_101,
) -> np.ndarray:
    """
    Applique la transformation affine sur l'image avec interpolation Lanczos haute qualité.
    """
    cropped = cv2.warpAffine(
        image_bgr,
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=border_mode,
    )
    return cropped


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Transforme un ensemble de points 2D [x, y] avec la matrice affine 2x3.
    """
    pts = np.atleast_2d(points)[:, :2]
    homog = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
    transformed = (matrix @ homog.T).T
    return transformed


class FaceCropper:
    """Recadreur de photo d'identité conforme aux spécifications géométriques."""

    def __init__(self, config: IDPhotoConfig = DEFAULT_CONFIG):
        self.config = config

    def crop(
        self,
        image_bgr: np.ndarray,
        detection: FaceDetection,
        dpi: int = 300,
        mask: Optional[np.ndarray] = None,
    ) -> CropResult:
        """
        Calcule et applique le recadrage optimal conforme à partir d'une détection faciale.
        """
        if not detection.detected or detection.keypoints is None:
            raise ValueError("Aucun visage détecté pour effectuer le recadrage.")

        target_w = self.config.target_width_px(dpi)
        target_h = self.config.target_height_px(dpi)
        target_face_h = self.config.target_face_height_px(dpi)

        affine_res = compute_id_affine_transform(
            keypoints=detection.keypoints,
            target_width=target_w,
            target_height=target_h,
            target_face_height=target_face_h,
            mask=mask,
        )

        cropped_img = apply_affine_crop(
            image_bgr=image_bgr,
            matrix=affine_res.matrix,
            target_width=target_w,
            target_height=target_h,
        )

        # Calcul des repères transformés
        transformed_pts = transform_points(
            detection.keypoints.all_landmarks, affine_res.matrix
        )

        face_ratio = target_face_h / target_h

        return CropResult(
            cropped_image=cropped_img,
            affine_result=affine_res,
            config=self.config,
            dpi=dpi,
            target_width_px=target_w,
            target_height_px=target_h,
            face_ratio=face_ratio,
            transformed_keypoints=transformed_pts,
        )
