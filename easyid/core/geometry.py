"""
Module de calculs géométriques, estimation anatomique du crâne et transformations affines.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
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
    Estime la position 2D du sommet anatomique du crâne osseux (vertex), hors chevelure.
    
    Conforme aux proportions anthropométriques crânio-faciales (norme ISO/IEC 19794-5 / Farkas) :
    La ligne des yeux se situe à environ 53-55% de la hauteur totale menton-sommet du crâne.
    Le repère 10 de MediaPipe correspond au haut du front (trichion / limite de la calotte).
    """
    chin = keypoints.chin[:2]
    eyes_center = (keypoints.left_iris[:2] + keypoints.right_iris[:2]) / 2.0
    forehead = keypoints.forehead[:2]

    # Vecteur directeur du visage du menton vers les yeux
    vec_chin_to_eyes = eyes_center - chin
    dist_chin_to_eyes = np.linalg.norm(vec_chin_to_eyes)

    if dist_chin_to_eyes < 1.0:
        return forehead

    unit_up = vec_chin_to_eyes / dist_chin_to_eyes

    # Estimation basée sur le ratio oculaire anthropométrique
    # Head height = dist_chin_to_eyes / 0.54 ~= dist_chin_to_eyes * 1.85
    crown_from_eyes = chin + unit_up * (dist_chin_to_eyes * 1.80)

    # Estimation basée sur le haut du front (repère 10) + voûte crânienne osseuse
    dist_chin_to_forehead = np.dot(forehead - chin, unit_up)
    cranial_vault_thickness = 0.16 * dist_chin_to_forehead
    crown_from_forehead = forehead + unit_up * cranial_vault_thickness

    # Moyenne pondérée robuste
    estimated_crown = 0.5 * crown_from_eyes + 0.5 * crown_from_forehead
    return estimated_crown


def compute_id_affine_transform(
    keypoints: FaceKeypoints,
    target_width: int,
    target_height: int,
    target_face_height: int,
    target_top_margin: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
) -> AffineTransformResult:
    """
    Calcule la matrice de transformation affine (similitude : échelle + rotation + translation)
    qui garantit :
      1. Redressement de la tête (horizontalité parfaite de la ligne des yeux).
      2. Mise à l'échelle pour que la hauteur menton-sommet du crâne = target_face_height (conforme 32-36 mm).
      3. Centrage horizontal strict de l'axe médian du visage (X = target_width / 2).
      4. Positionnement vertical intelligent :
         - Si un masque est fourni, le sommet réel visible de la tête (cheveux compris) est
           positionné sous le bord supérieur avec une marge de sécurité pour que la tête ne dépasse JAMAIS.
         - Si aucun masque n'est fourni, alignement direct avec target_top_margin.
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
    M = cv2.getRotationMatrix2D(
        (float(eyes_center_src[0]), float(eyes_center_src[1])),
        angle_deg,
        scale,
    )

    # 7. Coordonnées transformées actuelles
    crown_homog = np.array([crown_src[0], crown_src[1], 1.0], dtype=np.float32)
    eyes_homog = np.array([eyes_center_src[0], eyes_center_src[1], 1.0], dtype=np.float32)

    crown_transformed = M @ crown_homog
    eyes_transformed = M @ eyes_homog

    # 8. Ajustement horizontal : centrage parfait
    target_center_x = target_width / 2.0
    tx = target_center_x - eyes_transformed[0]
    M[0, 2] += tx

    # 9. Ajustement vertical : prise en compte des cheveux pour éviter tout débordement
    if mask is not None:
        # Rotation du masque pour mesurer la hauteur réelle de la chevelure au-dessus du crâne
        h_m, w_m = mask.shape[:2]
        M_rot_only = cv2.getRotationMatrix2D(
            (float(eyes_center_src[0]), float(eyes_center_src[1])),
            angle_deg,
            1.0,
        )
        rot_mask = cv2.warpAffine(mask, M_rot_only, (w_m, h_m), flags=cv2.INTER_LINEAR)
        rot_eyes = M_rot_only @ eyes_homog
        rot_crown = M_rot_only @ crown_homog

        eye_dist = abs(keypoints.left_iris[0] - keypoints.right_iris[0])
        x_min = max(0, int(rot_eyes[0] - eye_dist * 0.85))
        x_max = min(w_m, int(rot_eyes[0] + eye_dist * 0.85))

        strip = rot_mask[:, x_min:x_max]
        if strip.ndim == 3:
            strip = strip[:, :, 0]
        rows = np.where(strip > 0.35)[0]

        if len(rows) > 0 and rows.min() < rot_crown[1]:
            rot_head_top_y = float(rows.min())
        else:
            # Estimation de chevelure normale (~8% de la hauteur du visage)
            rot_head_top_y = rot_crown[1] - face_height_src * 0.08

        # Épaisseur des cheveux au-dessus du crâne dans l'image finale
        hair_thickness_dst = max(0.0, (rot_crown[1] - rot_head_top_y) * scale)

        # Marges cibles en pixels
        ideal_top_margin_px = round(target_height * 0.075)  # ~3.4 mm au-dessus des cheveux
        min_top_margin_px = round(target_height * 0.045)  # ~2.0 mm marge minimale
        min_bottom_margin_px = round(target_height * 0.080)  # ~3.6 mm sous le menton

        desired_crown_y = ideal_top_margin_px + hair_thickness_dst
        desired_chin_y = desired_crown_y + target_face_height
        max_chin_y = target_height - min_bottom_margin_px

        if desired_chin_y > max_chin_y:
            excess = desired_chin_y - max_chin_y
            shift = min(excess, ideal_top_margin_px - min_top_margin_px)
            desired_crown_y -= shift
            remaining = excess - shift
            if remaining > 0:
                desired_crown_y -= min(remaining, min_top_margin_px - 8)

        ty = desired_crown_y - crown_transformed[1]
    else:
        effective_top = target_top_margin if target_top_margin is not None else round(target_height * 0.08)
        ty = effective_top - crown_transformed[1]

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
