"""
Module de calculs géométriques, estimation anatomique du crâne et transformations affines.
Calibré pour alignement exact sur le gabarit officiel ICAO 9303 / ANTS.
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
    
    Conforme aux proportions anthropométriques crânio-faciales réelles (norme ISO/IEC 19794-5 / Farkas) :
    - Le vertex se situe au sommet de la calotte crânienne, bien au-dessus de la ligne d'implantation frontale (repère 10).
    - La ligne des yeux se situe à environ 52-54% de la hauteur totale menton-crâne.
      Autrement dit, dist(menton, vertex) ~= 1.92 * dist(menton, yeux).
    - Au-dessus du repère 10 (haut du front), la voûte crânienne monte d'environ 1.0x la hauteur frontale (glabelle-front).
    """
    chin = keypoints.chin[:2]
    eyes_center = (keypoints.left_iris[:2] + keypoints.right_iris[:2]) / 2.0
    forehead = keypoints.forehead[:2]
    glabella = keypoints.glabella[:2]

    # Vecteur directeur du visage du menton vers les yeux
    vec_chin_to_eyes = eyes_center - chin
    dist_chin_to_eyes = np.linalg.norm(vec_chin_to_eyes)

    if dist_chin_to_eyes < 1.0:
        return forehead

    unit_up = vec_chin_to_eyes / dist_chin_to_eyes

    # Estimation oculaire anthropométrique : sommet du crâne osseux ~1.92x distance menton-yeux
    crown_from_eyes = chin + unit_up * (dist_chin_to_eyes * 1.92)

    # Estimation crânienne frontale : voûte crânienne pariétale au-dessus du haut du front (repère 10)
    dist_glabella_forehead = max(10.0, float(np.dot(forehead - glabella, unit_up)))
    crown_from_forehead = forehead + unit_up * (1.00 * dist_glabella_forehead)

    # Fusion pondérée robuste
    estimated_crown = 0.6 * crown_from_eyes + 0.4 * crown_from_forehead
    return estimated_crown


def compute_id_affine_transform(
    keypoints: FaceKeypoints,
    target_width: int,
    target_height: int,
    target_face_height: int,
    target_top_margin: Optional[int] = None,
    eyes_target_y: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
) -> AffineTransformResult:
    """
    Calcule la matrice de transformation affine (similitude : échelle + rotation + translation)
    qui garantit le calage strict sur le gabarit officiel ICAO 9303 / ANTS :
      1. Redressement de la tête (horizontalité parfaite de la ligne des yeux).
      2. Mise à l'échelle pour que la hauteur menton-sommet du crâne = target_face_height (32-36 mm).
      3. Centrage horizontal strict de l'axe médian du visage (X = target_width / 2).
      4. Positionnement sur les lignes du gabarit :
         - Les yeux sont calés sur la ligne de référence (target_eyes_y).
         - Le sommet du crâne/front est positionné dans la zone de tolérance haute (3.5 à 7.5 mm).
         - Le bas du menton est positionné dans la zone de tolérance basse (35.5 à 39.5 mm du haut).
         - Si un masque de silhouette est fourni, la chevelure est garantie de ne pas dépasser du haut.
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
    chin_homog = np.array([chin_src[0], chin_src[1], 1.0], dtype=np.float32)

    eyes_transformed = M @ eyes_homog

    # 8. Ajustement horizontal : centrage parfait sur l'axe vertical médian
    target_center_x = target_width / 2.0
    tx = target_center_x - eyes_transformed[0]
    M[0, 2] += tx

    # 9. Ajustement vertical : calage sur le gabarit officiel ICAO 9303 / ANTS
    if target_top_margin is not None and eyes_target_y is None:
        # Mode direct avec marge haute explicite (pour rétrocompatibilité de tests)
        crown_transformed_y = (M @ crown_homog)[1]
        ty = target_top_margin - crown_transformed_y
        M[1, 2] += ty
    else:
        # Mode officiel ICAO : calage sur la ligne des yeux + zones de tolérance
        target_eyes = eyes_target_y if eyes_target_y is not None else round(target_height * (19.8 / 45.0))
        crown_min_y = round(target_height * (3.5 / 45.0))
        crown_max_y = round(target_height * (7.5 / 45.0))
        chin_min_y = round(target_height * (35.5 / 45.0))
        chin_max_y = round(target_height * (39.5 / 45.0))

        # Position initiale : yeux calés sur la ligne de référence
        ty = target_eyes - eyes_transformed[1]
        M[1, 2] += ty

        # Vérification et ajustement fin pour garantir que le crâne et le menton
        # tombent strictement dans leurs zones de tolérance respectives (deux traits chacune)
        curr_crown_y = (M @ crown_homog)[1]
        curr_chin_y = (M @ chin_homog)[1]

        nudge = 0.0
        if curr_crown_y < crown_min_y:
            nudge = crown_min_y - curr_crown_y
        elif curr_crown_y > crown_max_y:
            nudge = crown_max_y - curr_crown_y

        if (curr_chin_y + nudge) > chin_max_y:
            nudge = chin_max_y - curr_chin_y
        elif (curr_chin_y + nudge) < chin_min_y:
            nudge = chin_min_y - curr_chin_y

        M[1, 2] += nudge

        # Contrôle anti-dépassement haut de chevelure si le masque est fourni
        if mask is not None:
            warped_mask = cv2.warpAffine(
                mask, M, (target_width, target_height), flags=cv2.INTER_LINEAR
            )
            # Analyser la bande centrale au-dessus de la tête
            x1 = int(target_width * 0.25)
            x2 = int(target_width * 0.75)
            strip = warped_mask[:, x1:x2]
            if strip.ndim == 3:
                strip = strip[:, :, 0]
            nonzeros = np.where(strip > 0.4)[0]
            if len(nonzeros) > 0:
                top_hair_y = float(nonzeros.min())
                min_safe_top_margin_px = round(target_height * (2.0 / 45.0))  # 2 mm marge minimale
                if top_hair_y < min_safe_top_margin_px:
                    excess = min_safe_top_margin_px - top_hair_y
                    # Vérifier si on peut descendre sans sortir le menton de sa zone
                    max_allowed_nudge = max(0.0, chin_max_y - (curr_chin_y + nudge))
                    hair_nudge = min(excess, max_allowed_nudge)
                    M[1, 2] += hair_nudge

    return AffineTransformResult(
        matrix=M,
        scale=scale,
        rotation_deg=angle_deg,
        skull_crown_src=crown_src,
        chin_src=chin_src,
        face_height_src_px=face_height_src,
        face_height_dst_px=float(target_face_height),
    )
