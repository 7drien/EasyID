"""
EasyID: Générateur et vérificateur automatique de photos d'identité conformes aux normes ICAO / ANTS.
"""

__version__ = "0.1.0"

from easyid.config import IDPhotoConfig, DEFAULT_CONFIG
from easyid.pipeline import EasyIDPipeline, PipelineResult
from easyid.core.detector import FaceDetector, FaceDetection
from easyid.core.cropper import FaceCropper, CropResult
from easyid.core.segmenter import BackgroundSegmenter
from easyid.core.validator import ComplianceValidator, ComplianceReport

__all__ = [
    "EasyIDPipeline",
    "PipelineResult",
    "IDPhotoConfig",
    "DEFAULT_CONFIG",
    "FaceDetector",
    "FaceDetection",
    "FaceCropper",
    "CropResult",
    "BackgroundSegmenter",
    "ComplianceValidator",
    "ComplianceReport",
]
