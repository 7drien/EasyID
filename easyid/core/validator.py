"""
Module de validation et contrôle de conformité ICAO / ANTS.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import cv2
import numpy as np

from easyid.config import DEFAULT_CONFIG, IDPhotoConfig
from easyid.core.detector import FaceDetection


@dataclass
class CheckItem:
    name: str
    passed: bool
    value: float
    threshold: str
    message: str


@dataclass
class ComplianceReport:
    passed: bool
    score: float  # Score global de conformité de 0 à 100%
    checks: Dict[str, CheckItem] = field(default_factory=dict)
    summary_messages: List[str] = field(default_factory=list)


class ComplianceValidator:
    """Vérificateur automatique de critères de conformité (norme ICAO / ANTS)."""

    def __init__(self, config: IDPhotoConfig = DEFAULT_CONFIG):
        self.config = config

    def validate(
        self,
        image_bgr: np.ndarray,
        detection: FaceDetection,
        face_ratio: Optional[float] = None,
    ) -> ComplianceReport:
        """
        Vérifie l'ensemble des critères de conformité réglementaires.
        """
        checks: Dict[str, CheckItem] = {}

        # 1. Présence d'un visage unique
        single_face = detection.detected and detection.count == 1
        checks["face_count"] = CheckItem(
            name="Visage unique",
            passed=single_face,
            value=float(detection.count),
            threshold="1",
            message="Un et un seul visage doit être présent sur la photo."
            if not single_face
            else "Visage unique bien détecté.",
        )

        if not detection.detected or detection.keypoints is None:
            return ComplianceReport(
                passed=False,
                score=0.0,
                checks=checks,
                summary_messages=["Aucun visage détecté sur l'image."],
            )

        # 2. Angle de rotation Yaw (gauche/droite)
        abs_yaw = abs(detection.yaw)
        yaw_ok = abs_yaw <= self.config.MAX_YAW_DEG
        checks["yaw"] = CheckItem(
            name="Tête de face (rotation horizontale)",
            passed=yaw_ok,
            value=round(abs_yaw, 1),
            threshold=f"<= {self.config.MAX_YAW_DEG}°",
            message=f"Rotation horizontale de {abs_yaw:.1f}° "
            + ("conforme." if yaw_ok else "excessive (tournez la tête vers l'objectif)."),
        )

        # 3. Angle de plongée/contre-plongée Pitch (haut/bas)
        abs_pitch = abs(detection.pitch)
        pitch_ok = abs_pitch <= self.config.MAX_PITCH_DEG
        checks["pitch"] = CheckItem(
            name="Tête droite (inclinaison verticale)",
            passed=pitch_ok,
            value=round(abs_pitch, 1),
            threshold=f"<= {self.config.MAX_PITCH_DEG}°",
            message=f"Inclinaison verticale de {abs_pitch:.1f}° "
            + ("conforme." if pitch_ok else "excessive (regardez droit devant)."),
        )

        # 4. Yeux ouverts
        blink_left = detection.blendshapes.get("eyeBlinkLeft", 0.0)
        blink_right = detection.blendshapes.get("eyeBlinkRight", 0.0)
        max_blink = max(blink_left, blink_right)
        eyes_open = max_blink <= self.config.MAX_EYE_BLINK_SCORE
        checks["eyes_open"] = CheckItem(
            name="Yeux ouverts",
            passed=eyes_open,
            value=round(max_blink, 2),
            threshold=f"<= {self.config.MAX_EYE_BLINK_SCORE}",
            message="Les deux yeux sont bien ouverts."
            if eyes_open
            else "Un œil ou les deux yeux semblent fermés.",
        )

        # 5. Bouche fermée et expression neutre
        jaw_open = detection.blendshapes.get("jawOpen", 0.0)
        mouth_closed = jaw_open <= self.config.MAX_MOUTH_OPEN_SCORE
        checks["mouth_closed"] = CheckItem(
            name="Bouche fermée",
            passed=mouth_closed,
            value=round(jaw_open, 2),
            threshold=f"<= {self.config.MAX_MOUTH_OPEN_SCORE}",
            message="Bouche bien fermée."
            if mouth_closed
            else "La bouche semble entrouverte ou ouverte.",
        )

        # Sourire excessif
        smile_l = detection.blendshapes.get("mouthSmileLeft", 0.0)
        smile_r = detection.blendshapes.get("mouthSmileRight", 0.0)
        max_smile = max(smile_l, smile_r)
        expression_neutral = max_smile <= 0.40
        checks["neutral_expression"] = CheckItem(
            name="Expression neutre",
            passed=expression_neutral,
            value=round(max_smile, 2),
            threshold="<= 0.40",
            message="Expression faciale neutre conforme."
            if expression_neutral
            else "Sourire détecté (l'expression doit rester neutre sans sourire prononcé).",
        )

        # 6. Netteté (Variance du Laplacien)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness_ok = lap_var >= self.config.MIN_SHARPNESS_LAPLACIAN
        checks["sharpness"] = CheckItem(
            name="Netteté de l'image",
            passed=sharpness_ok,
            value=round(lap_var, 1),
            threshold=f">= {self.config.MIN_SHARPNESS_LAPLACIAN}",
            message=f"Indice de netteté ({lap_var:.1f}) suffisant."
            if sharpness_ok
            else f"Photo floue ou manquant de piqué ({lap_var:.1f}).",
        )

        # 7. Ratio de la taille du visage (si calculé après recadrage)
        if face_ratio is not None:
            min_ratio = self.config.FACE_HEIGHT_MIN_MM / self.config.HEIGHT_MM  # ~0.711
            max_ratio = self.config.FACE_HEIGHT_MAX_MM / self.config.HEIGHT_MM  # ~0.800
            ratio_ok = (min_ratio - 0.02) <= face_ratio <= (max_ratio + 0.02)
            checks["face_size_ratio"] = CheckItem(
                name="Taille du visage (70-80%)",
                passed=ratio_ok,
                value=round(face_ratio * 100, 1),
                threshold="70% à 80%",
                message=f"Le visage occupe {face_ratio * 100:.1f}% de la hauteur (conforme 32-36 mm)."
                if ratio_ok
                else f"Taille du visage non réglementaire ({face_ratio * 100:.1f}%).",
            )

        # Synthèse
        total_checks = len(checks)
        passed_checks = sum(1 for c in checks.values() if c.passed)
        overall_passed = passed_checks == total_checks
        score = (passed_checks / total_checks) * 100.0 if total_checks > 0 else 0.0

        summary = [c.message for c in checks.values() if not c.passed]
        if overall_passed:
            summary = ["Tous les critères de conformité réglementaires sont respectés."]

        return ComplianceReport(
            passed=overall_passed,
            score=score,
            checks=checks,
            summary_messages=summary,
        )
