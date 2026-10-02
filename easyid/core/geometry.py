"""
Module de calculs géométriques, estimation anatomique du crâne et transformations affines.
"""

from dataclasses import dataclass
from typing import Tuple
import cv2
import numpy as np

from easyid.core.detector import FaceKeypoints


@dataclass
class AffineTransformResult:
    matrix: np.ndarray  # Matrice 2x3 de transformation
    scale: float  # Facteur d'échelle appliqué
    rotation_deg: float  # Angle de rotation appliqué en degrés
    skull_crown_src: np.ndarray  # Point sommet du crâne estimé dans l'image source [x, y]
    chin_src: np.ndarray  # Point menton dans l'image source [x, y]
    face_height_src_px: float  # Hauteur du visage en pixels dans l'image source
    face_height_dst_px: float  # Hauteur du visage en pixels dans l'image destination


def estimate_skull_crown(keypoints: FaceKeypoints) -> np.ndarray:
    """
    Estime la position 2D du sommet anatomique du crâne (vertex), hors chevelure.
    
    Conforme aux proportions anthropométriques crânio-faciales (norme ISO/IEC 19794-5 / Farkas) :
    La ligne des yeux se situe à environ 53-55% de la hauteur totale menton-sommet du crâne.
    Le repère 10 de MediaPipe correspond au haut du front (trichion / limite basse de la calotte).
    """
    chin = keypoints.chin[:2]
    eyes_center = (keypoints.left_iris[:2] + keypoints.right_iris[:2]) / 2.0
    forehead = keypoints.forehead[:2]

    # Vecteur directeur du visage du menton vers les yeux
    vec_chin_to_eyes = eyes_center - chin
    dist_chin_to_eyes = np.linalg.norm(vec_chin_to_eyes)

    if dist_chin_to_eyes < 1.0:
        # Fallback de secours
        return forehead

    unit_up = vec_chin_to_eyes / dist_chin_to_eyes

    # Estimation basée sur le ratio oculaire anthropométrique
    # Head height = dist_chin_to_eyes / 0.54
    # Distance eyes to crown = dist_chin_to_eyes * (1 - 0.54) / 0.54 ~= 0.852 * dist_chin_to_eyes
    crown_from_eyes = chin + unit_up * (dist_chin_to_eyes * 1.85)

    # Estimation basée sur le haut du front (repère 10) + épaisseur de la voûte crânienne osseuse
    dist_chin_to_forehead = np.dot(forehead - chin, unit_up)
    # La calotte crânienne au-dessus du front représente environ 15-20% de la distance menton-front
    cranial_vault_thickness = 0.18 * dist_chin_to_forehead
    crown_from_forehead = forehead + unit_up * cranial_vault_thickness

    # Moyenne pondérée robuste entre proportion oculaire et repère frontal
    estimated_crown = 0.6 * crown_from_eyes + 0.4 * crown_from_forehead
    return estimated_crown


def compute_id_affine_transform(
    keypoints: FaceKeypoints,
    target_width: int,
    target_height: int,
    target_face_height: int,
    target_top_margin: int,
) -> AffineTransformResult:
    """
    Calcule la matrice de transformation affine (similitude : échelle + rotation + translation)
    qui garantit :
      1. Redressement de la tête (horizontalité parfaite de la ligne des yeux).
      2. Mise à l'échelle pour que la hauteur menton-sommet du crâne = target_face_height.
      3. Centrage horizontal parfait de l'axe médian du visage (X = target_width / 2).
      4. Positionnement du sommet du crâne à target_top_margin du haut de l'image.
    """
    # 1. Sommet du crâne estimé et menton
    crown_src = estimate_skull_crown(keypoints)
    chin_src = keypoints.chin[:2]

    # 2. Angle de la ligne des yeux (en degrés)
    dx = keypoints.left_iris[0] - keypoints.right_iris[0]
    dy = keypoints.left_iris[1] - keypoints.right_iris[1]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = float(np.degrees(angle_rad))

    # 3. Projection de la hauteur menton-crâne le long de l'axe anatomique
    vec_chin_crown = crown_src - chin_src
    up_dir = np.array([np.sin(angle_rad), -np.cos(angle_rad)], dtype=np.float32)
    face_height_src = abs(float(np.dot(vec_chin_crown, up_dir)))

    if face_height_src < 5.0:
        face_height_src = float(np.linalg.norm(vec_chin_crown))

    # 4. Facteur d'échelle
    scale = float(target_face_height / face_height_src)

    # 5. Centre de rotation : milieu des yeux
    eyes_center_src = (keypoints.left_iris[:2] + keypoints.right_iris[:2]) / 2.0

    # 6. Matrice de rotation OpenCV autour du centre des yeux
    # Pour annuler l'inclinaison angle_deg dans le repère image (Y vers le bas),
    # cv2.getRotationMatrix2D avec +angle_deg ramène la ligne interoculaire à l'horizontale
    M = cv2.getRotationMatrix2D(
        (float(eyes_center_src[0]), float(eyes_center_src[1])),
        angle_deg,
        scale,
    )

    # 7. Coordonnées transformées actuelles du sommet du crâne et du centre
    crown_homog = np.array([crown_src[0], crown_src[1], 1.0], dtype=np.float32)
    eyes_homog = np.array([eyes_center_src[0], eyes_center_src[1], 1.0], dtype=np.float32)

    crown_transformed = M @ crown_homog
    eyes_transformed = M @ eyes_homog

    # 8. Ajustement de translation pour atteindre les coordonnées cibles exactes
    target_center_x = target_width / 2.0
    tx = target_center_x - eyes_transformed[0]
    ty = target_top_margin - crown_transformed[1]

    M[0, 2] += tx
    M[1, 2] += ty

    return AffineTransformResult(
        matrix=M,
        scale=scale,
        rotation_deg=angle_deg,
        skull_crown_src=crown_src,
        chin_src=chin_src,
        face_height_src_px=face_height_src,
        face_height_dst_px=float(target_face_height),
    )
