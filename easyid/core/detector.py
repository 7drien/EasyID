"""
Module de détection faciale et extraction de repères (Face Landmarks) via MediaPipe.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from easyid.models import get_model_path


@dataclass
class FaceKeypoints:
    chin: np.ndarray  # [x, y, z] en pixels (pt 152)
    left_iris: np.ndarray  # [x, y, z] en pixels (pt 473 - œil gauche de la personne)
    right_iris: np.ndarray  # [x, y, z] en pixels (pt 468 - œil droit de la personne)
    glabella: np.ndarray  # [x, y, z] en pixels (pt 168 - entre les sourcils)
    nose_tip: np.ndarray  # [x, y, z] en pixels (pt 1)
    forehead: np.ndarray  # [x, y, z] en pixels (pt 10 - haut du front)
    left_eye_outer: np.ndarray  # pt 263
    left_eye_inner: np.ndarray  # pt 362
    right_eye_inner: np.ndarray  # pt 133
    right_eye_outer: np.ndarray  # pt 33
    lip_top: np.ndarray  # pt 13
    lip_bottom: np.ndarray  # pt 14
    all_landmarks: np.ndarray  # shape (N, 3) en pixels


@dataclass
class FaceDetection:
    detected: bool
    count: int = 0
    keypoints: Optional[FaceKeypoints] = None
    pitch: float = 0.0  # Inclinaison haut/bas en degrés
    yaw: float = 0.0  # Rotation gauche/droite en degrés
    roll: float = 0.0  # Roulis / inclinaison latérale en degrés
    roll_2d_deg: float = 0.0  # Roulis géométrique calculé sur les pupilles
    blendshapes: Dict[str, float] = field(default_factory=dict)
    raw_matrix: Optional[np.ndarray] = None


def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """Convertit une matrice de rotation 3x3 en angles d'Euler (pitch, yaw, roll) en degrés."""
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = 0.0
    return float(np.degrees(pitch)), float(np.degrees(yaw)), float(np.degrees(roll))


class FaceDetector:
    """Détecteur de visage et de repères 3D denses basé sur MediaPipe FaceLandmarker."""

    def __init__(self, max_faces: int = 1):
        self.max_faces = max_faces
        model_path = get_model_path("face_landmarker.task")
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=max_faces,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, image_bgr: np.ndarray) -> FaceDetection:
        """
        Détecte le visage et extrait l'ensemble des points clés et angles.
        Image en entrée : image OpenCV BGR (H, W, 3).
        """
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        result = self.landmarker.detect(mp_image)
        faces_count = len(result.face_landmarks)

        if faces_count == 0:
            return FaceDetection(detected=False, count=0)

        # Prendre le premier visage
        raw_landmarks = result.face_landmarks[0]
        pts = np.zeros((len(raw_landmarks), 3), dtype=np.float32)
        for i, lm in enumerate(raw_landmarks):
            pts[i] = [lm.x * w, lm.y * h, lm.z * w]

        # Points clés
        chin = pts[152]
        glabella = pts[168]
        nose_tip = pts[1]
        forehead = pts[10]

        # Iris (repères 468 & 473 disponibles si 478 points, sinon fallback sur centre des yeux)
        if len(pts) >= 478:
            right_iris = pts[468]  # Œil droit du sujet (gauche de l'image)
            left_iris = pts[473]  # Œil gauche du sujet (droite de l'image)
        else:
            right_iris = (pts[33] + pts[133]) / 2.0
            left_iris = (pts[362] + pts[263]) / 2.0

        keypoints = FaceKeypoints(
            chin=chin,
            left_iris=left_iris,
            right_iris=right_iris,
            glabella=glabella,
            nose_tip=nose_tip,
            forehead=forehead,
            left_eye_outer=pts[263],
            left_eye_inner=pts[362],
            right_eye_inner=pts[133],
            right_eye_outer=pts[33],
            lip_top=pts[13],
            lip_bottom=pts[14],
            all_landmarks=pts,
        )

        # Calcul de l'angle 2D des yeux (roulis à corriger)
        # Note : right_iris est à gauche dans l'image (x inférieur), left_iris est à droite (x supérieur)
        dx = left_iris[0] - right_iris[0]
        dy = left_iris[1] - right_iris[1]
        roll_2d_deg = float(np.degrees(np.arctan2(dy, dx)))

        # Extraction des blendshapes
        blendshapes_dict: Dict[str, float] = {}
        if result.face_blendshapes and len(result.face_blendshapes) > 0:
            for category in result.face_blendshapes[0]:
                blendshapes_dict[category.category_name] = category.score

        # Extraction des angles 3D via la matrice de transformation
        pitch = 0.0
        yaw = 0.0
        roll = roll_2d_deg
        raw_mat = None

        if result.facial_transformation_matrixes and len(result.facial_transformation_matrixes) > 0:
            raw_mat = np.array(result.facial_transformation_matrixes[0])
            if raw_mat.shape == (4, 4):
                R = raw_mat[:3, :3]
                p, y, r = rotation_matrix_to_euler(R)
                pitch, yaw, roll = p, y, r

        return FaceDetection(
            detected=True,
            count=faces_count,
            keypoints=keypoints,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            roll_2d_deg=roll_2d_deg,
            blendshapes=blendshapes_dict,
            raw_matrix=raw_mat,
        )
