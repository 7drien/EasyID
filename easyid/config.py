"""
Configuration des normes réglementaires et constantes pour EasyID.
Norme ANTS / ISO/IEC 19794-5 / ICAO (format 35 x 45 mm).
"""

from dataclasses import dataclass
from typing import Tuple


def mm_to_px(mm: float, dpi: int = 300) -> int:
    """Convertit des millimètres en pixels selon la résolution DPI."""
    return round((mm / 25.4) * dpi)


def px_to_mm(px: int, dpi: int = 300) -> float:
    """Convertit des pixels en millimètres selon la résolution DPI."""
    return (px / dpi) * 25.4


@dataclass
class IDPhotoConfig:
    """Configuration géométrique et réglementaire pour les photos d'identité."""

    # Dimensions officielles en millimètres
    WIDTH_MM: float = 35.0
    HEIGHT_MM: float = 45.0

    # Taille du visage réglementaire (menton au sommet du crâne, hors cheveux) : 32 à 36 mm (70 à 80%)
    FACE_HEIGHT_MIN_MM: float = 32.0  # 71.1%
    FACE_HEIGHT_MAX_MM: float = 36.0  # 80.0%
    FACE_HEIGHT_TARGET_MM: float = 33.4  # ~74.2% (valeur médiane idéale du gabarit officiel)

    # Positionnement officiel ICAO 9303 / ANTS des zones de tolérance (confidence)
    # 1. Axe oculaire de référence (25.2 mm du bas = 19.8 mm du haut, norme 24 à 28 mm)
    EYES_Y_TARGET_MM: float = 19.8

    # 2. Zone de tolérance du sommet du crâne / front (deux traits)
    CROWN_ZONE_MIN_MM: float = 3.5  # Trait supérieur du crâne (41.5 mm du bas)
    CROWN_ZONE_MAX_MM: float = 7.5  # Trait inférieur du crâne (37.5 mm du bas)

    # 3. Zone de tolérance du bas du menton (deux traits)
    CHIN_ZONE_MIN_MM: float = 35.5  # Trait supérieur du menton (9.5 mm du bas)
    CHIN_ZONE_MAX_MM: float = 39.5  # Trait inférieur du menton (5.5 mm du bas)

    # Résolution par défaut
    DEFAULT_DPI: int = 300

    # Arrière-plan réglementaire (Gris clair neutre, RGB: 235, 238, 240)
    # L'ANTS interdit le fond blanc pur (255, 255, 255)
    DEFAULT_BG_COLOR_RGB: Tuple[int, int, int] = (235, 238, 240)
    DEFAULT_BG_COLOR_BGR: Tuple[int, int, int] = (240, 238, 235)

    # Seuils de contrôle de conformité ICAO
    MAX_ROLL_DEG: float = 5.0  # Tolérance avant redressement
    MAX_PITCH_DEG: float = 8.0  # Inclinaison tête haut/bas max
    MAX_YAW_DEG: float = 8.0  # Rotation tête gauche/droite max
    MAX_EYE_BLINK_SCORE: float = 0.45  # Seuil œil fermé
    MAX_MOUTH_OPEN_SCORE: float = 0.25  # Seuil bouche ouverte
    MIN_SHARPNESS_LAPLACIAN: float = 60.0  # Seuil minimal de netteté

    # Format de planche d'impression standard (10x15 cm / 4x6 pouces)
    PRINT_SHEET_WIDTH_MM: float = 150.0
    PRINT_SHEET_HEIGHT_MM: float = 100.0

    def target_width_px(self, dpi: int = 300) -> int:
        return mm_to_px(self.WIDTH_MM, dpi)

    def target_height_px(self, dpi: int = 300) -> int:
        return mm_to_px(self.HEIGHT_MM, dpi)

    def target_face_height_px(self, dpi: int = 300) -> int:
        return mm_to_px(self.FACE_HEIGHT_TARGET_MM, dpi)

    def eyes_target_px(self, dpi: int = 300) -> int:
        return mm_to_px(self.EYES_Y_TARGET_MM, dpi)


# Instance globale réutilisable
DEFAULT_CONFIG = IDPhotoConfig()
