"""
Module de génération de planches d'impression 10x15 cm et export avec métadonnées DPI réelles.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image

from easyid.config import DEFAULT_CONFIG, IDPhotoConfig, mm_to_px


def generate_printable_sheet(
    id_photo_bgr: np.ndarray,
    dpi: int = 300,
    rows: int = 2,
    cols: int = 3,
    config: IDPhotoConfig = DEFAULT_CONFIG,
    draw_cut_marks: bool = True,
) -> np.ndarray:
    """
    Crée une planche d'impression standard 10x15 cm (150 mm x 100 mm)
    contenant `rows * cols` photos d'identité (par défaut 6 photos : 2 lignes de 3)
    avec des repères de découpe discrets.
    """
    sheet_w_px = mm_to_px(config.PRINT_SHEET_WIDTH_MM, dpi)  # 150 mm (1772 px à 300 DPI)
    sheet_h_px = mm_to_px(config.PRINT_SHEET_HEIGHT_MM, dpi)  # 100 mm (1181 px à 300 DPI)

    photo_h_px, photo_w_px = id_photo_bgr.shape[:2]

    # Créer une feuille blanche
    sheet = np.full((sheet_h_px, sheet_w_px, 3), 255, dtype=np.uint8)

    # Calcul de l'espacement pour centrer la grille
    total_grid_w = cols * photo_w_px
    total_grid_h = rows * photo_h_px

    # Marges et espacements
    spacing_x = max(mm_to_px(5.0, dpi), (sheet_w_px - total_grid_w) // (cols + 1))
    spacing_y = max(mm_to_px(3.0, dpi), (sheet_h_px - total_grid_h) // (rows + 1))

    # Recalcul des offsets de départ pour centrage parfait
    grid_total_w_with_spacing = cols * photo_w_px + (cols - 1) * spacing_x
    grid_total_h_with_spacing = rows * photo_h_px + (rows - 1) * spacing_y

    start_x = (sheet_w_px - grid_total_w_with_spacing) // 2
    start_y = (sheet_h_px - grid_total_h_with_spacing) // 2

    mark_len = mm_to_px(2.0, dpi)  # Traits de repère de 2 mm
    mark_color = (180, 180, 180)  # Gris discret

    for r in range(rows):
        for c in range(cols):
            x = start_x + c * (photo_w_px + spacing_x)
            y = start_y + r * (photo_h_px + spacing_y)

            # Insertion de la photo
            sheet[y : y + photo_h_px, x : x + photo_w_px] = id_photo_bgr

            # Tracé des repères de coupe aux 4 coins de chaque photo
            if draw_cut_marks:
                # Coin haut gauche
                cv2.line(sheet, (x - mark_len, y), (x, y), mark_color, 1)
                cv2.line(sheet, (x, y - mark_len), (x, y), mark_color, 1)
                # Coin haut droit
                cv2.line(sheet, (x + photo_w_px, y), (x + photo_w_px + mark_len, y), mark_color, 1)
                cv2.line(sheet, (x + photo_w_px, y - mark_len), (x + photo_w_px, y), mark_color, 1)
                # Coin bas gauche
                cv2.line(sheet, (x - mark_len, y + photo_h_px), (x, y + photo_h_px), mark_color, 1)
                cv2.line(sheet, (x, y + photo_h_px), (x, y + photo_h_px + mark_len), mark_color, 1)
                # Coin bas droit
                cv2.line(sheet, (x + photo_w_px, y + photo_h_px), (x + photo_w_px + mark_len, y + photo_h_px), mark_color, 1)
                cv2.line(sheet, (x + photo_w_px, y + photo_h_px), (x + photo_w_px, y + photo_h_px + mark_len), mark_color, 1)

    return sheet


def save_image_with_dpi(image_bgr: np.ndarray, output_path: str, dpi: int = 300) -> None:
    """
    Sauvegarde l'image avec les métadonnées DPI réelles intégrées (JPEG/PNG).
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    pil_img.save(output_path, dpi=(dpi, dpi), quality=98)
