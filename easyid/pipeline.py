"""
Pipeline d'orchestration principal pour EasyID.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np

from easyid.config import DEFAULT_CONFIG, IDPhotoConfig, px_to_mm
from easyid.core.cropper import CropResult, FaceCropper, apply_affine_crop
from easyid.core.detector import FaceDetection, FaceDetector
from easyid.core.geometry import AffineTransformResult
from easyid.core.segmenter import BackgroundSegmenter
from easyid.core.validator import ComplianceReport, ComplianceValidator
from easyid.printable import generate_printable_sheet, save_image_with_dpi


@dataclass
class PipelineResult:
    success: bool
    id_photo: Optional[np.ndarray] = None  # Photo recadrée 35x45 mm (H, W, 3)
    printable_sheet: Optional[np.ndarray] = None  # Planche 10x15 cm (H, W, 3)
    compliance_report: Optional[ComplianceReport] = None
    detection: Optional[FaceDetection] = None
    crop_result: Optional[CropResult] = None
    error_message: Optional[str] = None
    face_height_mm: Optional[float] = None
    face_ratio_percent: Optional[float] = None

    def save(
        self,
        output_photo_path: Union[str, Path],
        output_sheet_path: Optional[Union[str, Path]] = None,
        dpi: int = 300,
    ) -> None:
        """Sauvegarde la photo d'identité et optionnellement la planche avec métadonnées DPI."""
        if not self.success or self.id_photo is None:
            raise RuntimeError("Impossible de sauvegarder un résultat de pipeline en échec.")

        save_image_with_dpi(self.id_photo, str(output_photo_path), dpi=dpi)

        if output_sheet_path and self.printable_sheet is not None:
            save_image_with_dpi(self.printable_sheet, str(output_sheet_path), dpi=dpi)


class EasyIDPipeline:
    """Orchestrateur complet de traitement et validation de photos d'identité."""

    def __init__(self, config: IDPhotoConfig = DEFAULT_CONFIG):
        self.config = config
        self.detector = FaceDetector()
        self.cropper = FaceCropper(config=config)
        self.segmenter = BackgroundSegmenter(config=config)
        self.validator = ComplianceValidator(config=config)

    def process(
        self,
        image_input: Union[str, Path, np.ndarray],
        dpi: int = 300,
        replace_background: bool = True,
        bg_color_bgr: Optional[Tuple[int, int, int]] = None,
        generate_sheet: bool = True,
        sheet_rows: int = 2,
        sheet_cols: int = 3,
    ) -> PipelineResult:
        """
        Traite une photo brute :
          1. Chargement & validation de l'image
          2. Détection faciale et repères 3D denses
          3. Redressement horizontal et recadrage au gabarit 35x45 mm (visage 70-80%)
          4. Segmentation et remplacement de l'arrière-plan par un fond uni réglementaire
          5. Contrôle qualité et conformité réglementaire ICAO / ANTS
          6. Génération de la planche d'impression 10x15 cm
        """
        # 1. Chargement de l'image
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            img_bgr = cv2.imread(path_str)
            if img_bgr is None:
                return PipelineResult(
                    success=False,
                    error_message=f"Impossible de lire le fichier image : {path_str}",
                )
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input.copy()
        else:
            return PipelineResult(
                success=False,
                error_message=f"Type d'entrée non supporté : {type(image_input)}",
            )

        # 2. Détection faciale
        detection = self.detector.detect(img_bgr)
        if not detection.detected or detection.keypoints is None:
            report = self.validator.validate(img_bgr, detection)
            return PipelineResult(
                success=False,
                detection=detection,
                compliance_report=report,
                error_message="Aucun visage détecté sur la photo fournie.",
            )

        # 3. Recadrage et alignement géométrique
        try:
            crop_res = self.cropper.crop(img_bgr, detection, dpi=dpi)
        except Exception as e:
            return PipelineResult(
                success=False,
                detection=detection,
                error_message=f"Erreur lors du calcul géométrique : {str(e)}",
            )

        final_photo = crop_res.cropped_image

        # 4. Remplacement d'arrière-plan
        if replace_background:
            if bg_color_bgr is None:
                bg_color_bgr = self.config.DEFAULT_BG_COLOR_BGR

            # Extraction du masque sur l'image source puis application de la même transformation affine
            source_mask = self.segmenter.extract_mask(img_bgr)
            target_w = crop_res.target_width_px
            target_h = crop_res.target_height_px

            # Transformation affine du masque
            warped_mask = cv2.warpAffine(
                source_mask,
                crop_res.affine_result.matrix,
                (target_w, target_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            if warped_mask.ndim == 2:
                warped_mask = np.expand_dims(warped_mask, axis=-1)

            # Composition finale avec la couleur de fond
            bg = np.full_like(final_photo, bg_color_bgr, dtype=np.uint8)
            fg_f = final_photo.astype(np.float32)
            bg_f = bg.astype(np.float32)
            blended = fg_f * warped_mask + bg_f * (1.0 - warped_mask)
            final_photo = np.clip(blended, 0, 255).astype(np.uint8)

        # 5. Validation de la conformité
        compliance_report = self.validator.validate(
            image_bgr=final_photo,
            detection=detection,
            face_ratio=crop_res.face_ratio,
        )

        # 6. Planche d'impression 10x15 cm
        sheet = None
        if generate_sheet:
            sheet = generate_printable_sheet(
                id_photo_bgr=final_photo,
                dpi=dpi,
                rows=sheet_rows,
                cols=sheet_cols,
                config=self.config,
            )

        face_h_mm = px_to_mm(round(crop_res.affine_result.face_height_dst_px), dpi)
        face_ratio_pct = crop_res.face_ratio * 100.0

        return PipelineResult(
            success=True,
            id_photo=final_photo,
            printable_sheet=sheet,
            compliance_report=compliance_report,
            detection=detection,
            crop_result=crop_res,
            face_height_mm=round(face_h_mm, 2),
            face_ratio_percent=round(face_ratio_pct, 1),
        )
