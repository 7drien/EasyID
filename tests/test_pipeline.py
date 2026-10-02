"""
Tests d'intégration du pipeline complet EasyID.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from easyid.config import DEFAULT_CONFIG
from easyid.pipeline import EasyIDPipeline


@pytest.fixture
def sample_image_path():
    path = Path("tests/sample_portrait.jpg")
    assert path.exists(), "L'image de test tests/sample_portrait.jpg doit être présente."
    return str(path)


def test_pipeline_execution_300dpi(sample_image_path):
    pipeline = EasyIDPipeline()
    result = pipeline.process(
        image_input=sample_image_path,
        dpi=300,
        replace_background=True,
        generate_sheet=True,
    )

    # 1. Succès
    assert result.success is True, f"Pipeline échoué : {result.error_message}"
    assert result.id_photo is not None

    # 2. Dimensions exactes (format 35x45 mm à 300 DPI -> 413x531 px)
    h, w, c = result.id_photo.shape
    assert w == 413, f"Largeur attendue 413 px, obtenu {w}"
    assert h == 531, f"Hauteur attendue 531 px, obtenu {h}"
    assert c == 3

    # 3. Ratio du visage conforme (entre 70% et 80%)
    assert 70.0 <= result.face_ratio_percent <= 80.0, (
        f"Ratio du visage non conforme : {result.face_ratio_percent}%"
    )

    # 4. Taille du visage en mm conforme (entre 32 et 36 mm)
    assert 32.0 <= result.face_height_mm <= 36.0, (
        f"Hauteur du visage en mm non conforme : {result.face_height_mm} mm"
    )

    # 5. Planche 10x15 cm à 300 DPI (1772 x 1181 px)
    assert result.printable_sheet is not None
    sh_h, sh_w, _ = result.printable_sheet.shape
    assert sh_w == 1772, f"Largeur planche attendue 1772 px, obtenu {sh_w}"
    assert sh_h == 1181, f"Hauteur planche attendue 1181 px, obtenu {sh_h}"

    # 6. Rapport de conformité généré
    assert result.compliance_report is not None
    assert "face_count" in result.compliance_report.checks
    assert "yaw" in result.compliance_report.checks
    assert "pitch" in result.compliance_report.checks
    assert "face_size_ratio" in result.compliance_report.checks


def test_pipeline_execution_600dpi(sample_image_path):
    pipeline = EasyIDPipeline()
    result = pipeline.process(
        image_input=sample_image_path,
        dpi=600,
        replace_background=False,
        generate_sheet=False,
    )

    assert result.success is True
    assert result.id_photo is not None

    # Dimensions à 600 DPI : 827 x 1063 px
    h, w, _ = result.id_photo.shape
    assert w == 827, f"Largeur 600 DPI attendue 827 px, obtenu {w}"
    assert h == 1063, f"Hauteur 600 DPI attendue 1063 px, obtenu {h}"


def test_pipeline_template_zones_compliance(sample_image_path):
    pipeline = EasyIDPipeline()
    result = pipeline.process(
        image_input=sample_image_path,
        dpi=300,
        replace_background=False,
        generate_sheet=False,
    )
    assert result.success is True
    aff = result.crop_result.affine_result
    M = aff.matrix
    crown_dst = M @ np.array([aff.skull_crown_src[0], aff.skull_crown_src[1], 1.0])
    chin_dst = M @ np.array([aff.chin_src[0], aff.chin_src[1], 1.0])

    cfg = pipeline.config
    crown_min_y = round((cfg.CROWN_ZONE_MIN_MM / 25.4) * 300)
    crown_max_y = round((cfg.CROWN_ZONE_MAX_MM / 25.4) * 300)
    chin_min_y = round((cfg.CHIN_ZONE_MIN_MM / 25.4) * 300)
    chin_max_y = round((cfg.CHIN_ZONE_MAX_MM / 25.4) * 300)

    # Le crâne doit être dans sa zone de tolérance
    assert crown_min_y <= crown_dst[1] <= crown_max_y, f"Crâne {crown_dst[1]} hors zone [{crown_min_y}, {crown_max_y}]"
    # Le menton doit être dans sa zone de tolérance
    assert chin_min_y <= chin_dst[1] <= chin_max_y, f"Menton {chin_dst[1]} hors zone [{chin_min_y}, {chin_max_y}]"


def test_invalid_input():
    pipeline = EasyIDPipeline()
    # Image sans visage (bruit noir complet)
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    result = pipeline.process(blank)
    assert result.success is False
    assert "Aucun visage" in result.error_message
