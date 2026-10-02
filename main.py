#!/usr/bin/env python3
"""
EasyID — Point d'entrée principal (CLI & Démo).
Générateur et vérificateur automatique de photos d'identité conformes (ANTS / ICAO).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from easyid.config import DEFAULT_CONFIG
from easyid.pipeline import EasyIDPipeline


def launch_ui():
    """Lance l'interface utilisateur native Desktop Tkinter."""
    from easyid.ui.app import start_app
    print("🚀 Lancement de l'interface graphique Tkinter...")
    start_app()


def main():
    parser = argparse.ArgumentParser(
        description="EasyID: Alignement et recadrage automatique de photos d'identité conformes (ANTS / ICAO)."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=None,
        help="Chemin vers la photo brute à traiter (JPG, PNG, WEBP).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="photo_identite_35x45.jpg",
        help="Chemin de sauvegarde de la photo d'identité (défaut: photo_identite_35x45.jpg).",
    )
    parser.add_argument(
        "-s", "--sheet",
        type=str,
        default="planche_impression_10x15.jpg",
        help="Chemin de sauvegarde de la planche 10x15 cm (défaut: planche_impression_10x15.jpg).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Résolution d'impression (défaut: 300 DPI).",
    )
    parser.add_argument(
        "--no-bg-replace",
        action="store_true",
        help="Désactiver le remplacement d'arrière-plan par le fond gris neutre officiel.",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Démarrer l'interface graphique native Desktop (Tkinter).",
    )

    args = parser.parse_args()

    if args.ui:
        launch_ui()
        return

    # Si aucun fichier n'est fourni, on utilise l'image d'exemple intégrée
    if args.input is None:
        sample_path = Path(__file__).parent / "tests" / "sample_portrait.jpg"
        if not sample_path.exists():
            print("Usage: python main.py --input <chemin_photo> ou python main.py --ui")
            sys.exit(1)
        print(f"ℹ️ Aucun fichier spécifié. Exécution en mode démo sur : {sample_path}")
        input_path = str(sample_path)
    else:
        input_path = args.input

    if not os.path.exists(input_path):
        print(f"❌ Erreur : Le fichier spécifié n'existe pas : {input_path}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("📸 EasyID — Traitement de la Photo d'Identité")
    print(f"Format cible : {DEFAULT_CONFIG.WIDTH_MM} × {DEFAULT_CONFIG.HEIGHT_MM} mm | Résolution : {args.dpi} DPI")
    print(f"Taille du visage cible : {DEFAULT_CONFIG.FACE_HEIGHT_MIN_MM} à {DEFAULT_CONFIG.FACE_HEIGHT_MAX_MM} mm (70-80%)")
    print("=" * 60 + "\n")

    pipeline = EasyIDPipeline()
    result = pipeline.process(
        image_input=input_path,
        dpi=args.dpi,
        replace_background=not args.no_bg_replace,
        generate_sheet=True,
    )

    if not result.success:
        print(f"❌ Échec : {result.error_message}")
        sys.exit(1)

    # Sauvegarde des résultats
    result.save(output_photo_path=args.output, output_sheet_path=args.sheet, dpi=args.dpi)

    print(f"✅ Photo d'identité générée avec succès !")
    print(f"  - Fichier unitaire (35x45 mm) : {args.output}")
    print(f"  - Dimensions en pixels         : {result.id_photo.shape[1]} × {result.id_photo.shape[0]} px ({args.dpi} DPI)")
    print(f"  - Taille du visage mesurée     : {result.face_height_mm:.2f} mm ({result.face_ratio_percent:.1f}% de la photo)")
    print(f"  - Planche d'impression 10x15cm : {args.sheet} (6 photos avec repères)")

    # Rapport de conformité
    report = result.compliance_report
    if report:
        print("\n" + "-" * 60)
        print(f"📋 RAPPORT DE CONFORMITÉ ANTS / ICAO (Score: {report.score:.0f}%)")
        print("-" * 60)
        for name, chk in report.checks.items():
            status = " [VALIDE] " if chk.passed else "[ATTENTION]"
            print(f" {status} {chk.name:<35} : {chk.message}")
        print("-" * 60)

    print("\n💡 Pour lancer l'application graphique de bureau (Tkinter) :")
    print("   python main.py --ui\n")


if __name__ == "__main__":
    main()
