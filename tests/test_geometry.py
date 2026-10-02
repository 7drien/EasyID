"""
Tests unitaires pour les calculs géométriques et transformations affines.
"""

import numpy as np
import pytest

from easyid.config import mm_to_px, px_to_mm, DEFAULT_CONFIG
from easyid.core.detector import FaceKeypoints
from easyid.core.geometry import compute_id_affine_transform, estimate_skull_crown


def test_metric_conversions():
    # 25.4 mm = 1 inch -> at 300 DPI, exactly 300 px
    assert mm_to_px(25.4, dpi=300) == 300
    assert abs(px_to_mm(300, dpi=300) - 25.4) < 1e-4

    # 35 mm at 300 DPI: round(35 / 25.4 * 300) = 413
    assert mm_to_px(35.0, dpi=300) == 413

    # 45 mm at 300 DPI: round(45 / 25.4 * 300) = 531
    assert mm_to_px(45.0, dpi=300) == 531

    # Ratio 35/45 = 7/9
    assert abs((35.0 / 45.0) - (413 / 531)) < 0.005


def test_affine_transform_exactness():
    """Vérifie que la matrice affine place le sommet du crâne et le menton aux positions cibles."""
    # Création d'un visage synthétique vertical
    # Yeux à Y=200, Menton à Y=300, Front à Y=130
    chin = np.array([250.0, 320.0, 0.0], dtype=np.float32)
    left_iris = np.array([290.0, 200.0, 0.0], dtype=np.float32)  # X supérieur
    right_iris = np.array([210.0, 200.0, 0.0], dtype=np.float32)  # X inférieur
    glabella = np.array([250.0, 190.0, 0.0], dtype=np.float32)
    nose = np.array([250.0, 240.0, 0.0], dtype=np.float32)
    forehead = np.array([250.0, 120.0, 0.0], dtype=np.float32)

    all_pts = np.zeros((478, 3), dtype=np.float32)
    all_pts[152] = chin
    all_pts[473] = left_iris
    all_pts[468] = right_iris
    all_pts[168] = glabella
    all_pts[1] = nose
    all_pts[10] = forehead

    keypoints = FaceKeypoints(
        chin=chin,
        left_iris=left_iris,
        right_iris=right_iris,
        glabella=glabella,
        nose_tip=nose,
        forehead=forehead,
        left_eye_outer=left_iris,
        left_eye_inner=left_iris,
        right_eye_inner=right_iris,
        right_eye_outer=right_iris,
        lip_top=chin - 30,
        lip_bottom=chin - 10,
        all_landmarks=all_pts,
    )

    target_w = 413
    target_h = 531
    target_face_h = 402  # ~34 mm à 300 DPI
    target_top_margin = 47  # ~4 mm à 300 DPI

    affine_res = compute_id_affine_transform(
        keypoints=keypoints,
        target_width=target_w,
        target_height=target_h,
        target_face_height=target_face_h,
        target_top_margin=target_top_margin,
    )

    M = affine_res.matrix
    crown_src = affine_res.skull_crown_src
    chin_src = affine_res.chin_src

    crown_dst = M @ np.array([crown_src[0], crown_src[1], 1.0])
    chin_dst = M @ np.array([chin_src[0], chin_src[1], 1.0])

    # Le crâne doit être centré horizontalement à target_w / 2
    assert abs(crown_dst[0] - (target_w / 2.0)) < 1.0
    # Le crâne doit être à target_top_margin du haut
    assert abs(crown_dst[1] - target_top_margin) < 1.0

    # Le menton doit être aligné verticalement sous le crâne
    assert abs(chin_dst[0] - (target_w / 2.0)) < 1.0
    # La distance menton-crâne doit être exactement target_face_h
    measured_face_h = chin_dst[1] - crown_dst[1]
    assert abs(measured_face_h - target_face_h) < 1.0


def test_tilted_face_roll_correction():
    """Vérifie que la transformation corrige une tête penchée (roll != 0)."""
    # Tête penchée de ~15 degrés
    angle_rad = np.radians(15.0)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    center = np.array([300.0, 300.0])
    # Points relatifs
    r_eye_rel = np.array([-40.0, 0.0])
    l_eye_rel = np.array([+40.0, 0.0])
    chin_rel = np.array([0.0, +120.0])
    forehead_rel = np.array([0.0, -80.0])

    # Rotation des points
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    r_eye = center + rot @ r_eye_rel
    l_eye = center + rot @ l_eye_rel
    chin = center + rot @ chin_rel
    forehead = center + rot @ forehead_rel

    all_pts = np.zeros((478, 3), dtype=np.float32)
    keypoints = FaceKeypoints(
        chin=np.append(chin, 0.0),
        left_iris=np.append(l_eye, 0.0),
        right_iris=np.append(r_eye, 0.0),
        glabella=np.append(center, 0.0),
        nose_tip=np.append(center, 0.0),
        forehead=np.append(forehead, 0.0),
        left_eye_outer=np.append(l_eye, 0.0),
        left_eye_inner=np.append(l_eye, 0.0),
        right_eye_inner=np.append(r_eye, 0.0),
        right_eye_outer=np.append(r_eye, 0.0),
        lip_top=np.append(chin, 0.0),
        lip_bottom=np.append(chin, 0.0),
        all_landmarks=all_pts,
    )

    affine_res = compute_id_affine_transform(
        keypoints=keypoints,
        target_width=413,
        target_height=531,
        target_face_height=402,
        target_top_margin=47,
    )

    # Vérification que l'angle détecté est proche de 15°
    assert abs(affine_res.rotation_deg - 15.0) < 0.5

    # Après transformation, les deux yeux doivent être strictement au même niveau Y
    M = affine_res.matrix
    l_eye_dst = M @ np.array([l_eye[0], l_eye[1], 1.0])
    r_eye_dst = M @ np.array([r_eye[0], r_eye[1], 1.0])

    assert abs(l_eye_dst[1] - r_eye_dst[1]) < 0.1, "La ligne des yeux doit être parfaitement horizontale"


def test_icao_template_zones_alignment():
    """Vérifie que les yeux sont sur la ligne, et que le crâne et le menton sont dans leurs zones à 2 traits."""
    chin = np.array([250.0, 340.0, 0.0], dtype=np.float32)
    left_iris = np.array([290.0, 200.0, 0.0], dtype=np.float32)
    right_iris = np.array([210.0, 200.0, 0.0], dtype=np.float32)
    glabella = np.array([250.0, 190.0, 0.0], dtype=np.float32)
    nose = np.array([250.0, 240.0, 0.0], dtype=np.float32)
    forehead = np.array([250.0, 130.0, 0.0], dtype=np.float32)

    all_pts = np.zeros((478, 3), dtype=np.float32)
    keypoints = FaceKeypoints(
        chin=chin,
        left_iris=left_iris,
        right_iris=right_iris,
        glabella=glabella,
        nose_tip=nose,
        forehead=forehead,
        left_eye_outer=left_iris,
        left_eye_inner=left_iris,
        right_eye_inner=right_iris,
        right_eye_outer=right_iris,
        lip_top=chin - 30,
        lip_bottom=chin - 10,
        all_landmarks=all_pts,
    )

    dpi = 300
    target_w = 413
    target_h = 531
    target_face_h = round((DEFAULT_CONFIG.FACE_HEIGHT_TARGET_MM / 25.4) * dpi)  # ~396 px (33.5 mm)

    res = compute_id_affine_transform(
        keypoints=keypoints,
        target_width=target_w,
        target_height=target_h,
        target_face_height=target_face_h,
        eyes_target_y=DEFAULT_CONFIG.eyes_target_px(dpi),
    )

    M = res.matrix
    eyes_dst = M @ np.array([250.0, 200.0, 1.0])
    crown_dst = M @ np.array([res.skull_crown_src[0], res.skull_crown_src[1], 1.0])
    chin_dst = M @ np.array([chin[0], chin[1], 1.0])

    # 1. Les yeux doivent être sur la ligne cible (19.8 mm du haut +/- 1 mm)
    expected_eyes_y = DEFAULT_CONFIG.eyes_target_px(dpi)  # ~234 px
    assert abs(eyes_dst[1] - expected_eyes_y) <= 12.0  # Tolérance < 1 mm

    # 2. Le sommet du crâne doit être dans la zone [3.5 mm, 7.5 mm] du haut
    crown_min_y = round((DEFAULT_CONFIG.CROWN_ZONE_MIN_MM / 25.4) * dpi)  # ~41 px
    crown_max_y = round((DEFAULT_CONFIG.CROWN_ZONE_MAX_MM / 25.4) * dpi)  # ~89 px
    assert crown_min_y <= crown_dst[1] <= crown_max_y, f"Crâne {crown_dst[1]} hors zone [{crown_min_y}, {crown_max_y}]"

    # 3. Le menton doit être dans la zone [35.5 mm, 39.5 mm] du haut (5.5 à 9.5 mm du bas)
    chin_min_y = round((DEFAULT_CONFIG.CHIN_ZONE_MIN_MM / 25.4) * dpi)  # ~419 px
    chin_max_y = round((DEFAULT_CONFIG.CHIN_ZONE_MAX_MM / 25.4) * dpi)  # ~467 px
    assert chin_min_y <= chin_dst[1] <= chin_max_y, f"Menton {chin_dst[1]} hors zone [{chin_min_y}, {chin_max_y}]"

    # 4. Taille du visage conforme (32-36 mm)
    face_h_mm = ((chin_dst[1] - crown_dst[1]) / dpi) * 25.4
    assert 32.0 <= face_h_mm <= 36.0
