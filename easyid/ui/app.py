"""
Application Web Interactive EasyID (Streamlit).
Permet de tester en temps réel avec upload d'image ou webcam.
"""

import io
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import streamlit as st

from easyid.config import IDPhotoConfig, DEFAULT_CONFIG
from easyid.pipeline import EasyIDPipeline


def draw_official_overlay(image_bgr: np.ndarray, config: IDPhotoConfig, dpi: int = 300) -> np.ndarray:
    """Dessine le gabarit réglementaire officiel en surimpression pour contrôle visuel."""
    overlay = image_bgr.copy()
    h, w = image_bgr.shape[:2]

    # Ligne médiane verticale (axe sagittal)
    cv2.line(overlay, (w // 2, 0), (w // 2, h), (0, 255, 255), 1)

    # Ligne sommet du crâne cible
    top_margin_px = config.target_top_margin_px(dpi)
    cv2.line(overlay, (0, top_margin_px), (w, top_margin_px), (0, 255, 0), 1)
    cv2.putText(overlay, "Sommet crane (3.5-4.5mm)", (10, top_margin_px - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1)

    # Zone menton réglementaire (entre 32 et 36 mm du sommet)
    chin_min_px = top_margin_px + config.target_face_height_px(dpi) - round((2.0 / 25.4) * dpi)
    chin_max_px = top_margin_px + config.target_face_height_px(dpi) + round((2.0 / 25.4) * dpi)

    cv2.line(overlay, (0, chin_min_px), (w, chin_min_px), (255, 100, 0), 1)
    cv2.line(overlay, (0, chin_max_px), (w, chin_max_px), (255, 100, 0), 1)
    cv2.putText(overlay, "Zone menton reglementaire (32-36mm)", (10, chin_max_px + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 0), 1)

    # Fusion semi-transparente
    return cv2.addWeighted(overlay, 0.85, image_bgr, 0.15, 0)


def bgr_to_pil(bgr_img: np.ndarray, dpi: int = 300) -> Image.Image:
    """Convertit une image BGR OpenCV en image PIL RGB avec DPI."""
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil.info["dpi"] = (dpi, dpi)
    return pil


def pil_to_bytes(pil_img: Image.Image, format: str = "JPEG", dpi: int = 300) -> bytes:
    """Sérialise une image PIL en bytes pour le téléchargement avec DPI."""
    buf = io.BytesIO()
    pil_img.save(buf, format=format, dpi=(dpi, dpi), quality=98)
    return buf.getvalue()


def main():
    st.set_page_config(
        page_title="EasyID — Photo d'Identité IA Conforme ICAO / ANTS",
        page_icon="📸",
        layout="wide",
    )

    st.title("📸 EasyID — Photos d'Identité Conformes")
    st.markdown(
        "Générez et vérifiez automatiquement vos photos d'identité aux **normes officielles françaises (ANTS / ICAO)** : "
        "format **35 × 45 mm**, taille du visage **32 à 36 mm (70 à 80%)**, alignement du regard et fond uni clair."
    )

    # Sidebar des paramètres
    st.sidebar.header("⚙️ Configuration")
    dpi = st.sidebar.selectbox("Résolution d'impression (DPI)", [300, 600], index=0)
    replace_bg = st.sidebar.checkbox("Remplacer le fond par un fond neutre", value=True)
    bg_color_hex = st.sidebar.color_picker(
        "Couleur de fond réglementaire",
        value="#EBEFF0",
        help="L'ANTS impose un fond uni clair (gris ou bleu pâle). Le fond blanc pur (#FFFFFF) est interdit.",
    )
    show_overlay = st.sidebar.checkbox("Afficher le gabarit de contrôle officiel", value=True)

    # Conversion HEX en BGR
    hex_clean = bg_color_hex.lstrip("#")
    r, g, b = tuple(int(hex_clean[i : i + 2], 16) for i in (0, 2, 4))
    bg_bgr = (b, g, r)

    # Sélection de la source d'image
    source_type = st.radio(
        "Choisir la source de l'image :",
        ["Charger un fichier image", "Prendre une photo via la webcam", "Image de test de démonstration"],
        horizontal=True,
    )

    input_bgr = None

    if source_type == "Charger un fichier image":
        uploaded_file = st.file_uploader(
            "Téléversez votre portrait (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
        )
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            input_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    elif source_type == "Prendre une photo via la webcam":
        camera_photo = st.camera_input("Prenez votre photo bien en face avec expression neutre")
        if camera_photo is not None:
            file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
            input_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    else:
        sample_path = Path("tests/sample_portrait.jpg")
        if sample_path.exists():
            input_bgr = cv2.imread(str(sample_path))
            st.info("Utilisation de l'image de démonstration intégrée.")
        else:
            st.warning("Image de test non trouvée. Veuillez uploader une photo.")

    if input_bgr is None:
        st.info("👉 Veuillez fournir une photo pour lancer l'alignement et le recadrage automatique.")
        return

    # Pipeline de traitement
    with st.spinner("Analyse du visage par Deep Learning, alignement et recadrage..."):
        pipeline = EasyIDPipeline()
        res = pipeline.process(
            input_bgr,
            dpi=dpi,
            replace_background=replace_bg,
            bg_color_bgr=bg_bgr,
            generate_sheet=True,
        )

    if not res.success or res.id_photo is None:
        st.error(f"❌ Échec du traitement : {res.error_message}")
        return

    # Affichage des résultats
    col1, col2, col3 = st.columns([1, 1, 1.2])

    with col1:
        st.subheader("1. Photo Source")
        st.image(cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        if res.detection and res.detection.detected:
            st.caption(
                f"Angles bruts détectés : Roulis {res.detection.roll_2d_deg:+.1f}° | "
                f"Yaw {res.detection.yaw:+.1f}° | Pitch {res.detection.pitch:+.1f}°"
            )

    with col2:
        st.subheader("2. Photo d'Identité (35 × 45 mm)")
        display_photo = res.id_photo
        if show_overlay:
            display_photo = draw_official_overlay(display_photo, DEFAULT_CONFIG, dpi=dpi)

        st.image(cv2.cvtColor(display_photo, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption(
            f"Dimensions : {res.id_photo.shape[1]} × {res.id_photo.shape[0]} px ({dpi} DPI)\n"
            f"Taille visage : {res.face_height_mm:.1f} mm ({res.face_ratio_percent:.1f}% de la hauteur)"
        )

        # Bouton téléchargement photo unitaire
        pil_photo = bgr_to_pil(res.id_photo, dpi=dpi)
        photo_bytes = pil_to_bytes(pil_photo, "JPEG", dpi=dpi)
        st.download_button(
            label="💾 Télécharger la photo d'identité (35x45 mm)",
            data=photo_bytes,
            file_name="photo_identite_35x45.jpg",
            mime="image/jpeg",
        )

    with col3:
        st.subheader("3. Planche 10 × 15 cm (6 photos)")
        if res.printable_sheet is not None:
            st.image(cv2.cvtColor(res.printable_sheet, cv2.COLOR_BGR2RGB), use_container_width=True)
            pil_sheet = bgr_to_pil(res.printable_sheet, dpi=dpi)
            sheet_bytes = pil_to_bytes(pil_sheet, "JPEG", dpi=dpi)
            st.download_button(
                label="🖨️ Télécharger la planche prête à imprimer (10x15 cm)",
                data=sheet_bytes,
                file_name="planche_impression_10x15.jpg",
                mime="image/jpeg",
            )

    # Section Contrôle Qualité / Conformité ANTS
    st.divider()
    st.subheader("📋 Rapport de Conformité Réglementaire (ANTS / ICAO)")

    report = res.compliance_report
    if report:
        if report.passed:
            st.success(f"✅ Photo Conforme ! Score global : {report.score:.0f}%")
        else:
            st.warning(f"⚠️ Non-conformités détectées. Score global : {report.score:.0f}%")

        crit_cols = st.columns(len(report.checks))
        for col, (key, chk) in zip(crit_cols, report.checks.items()):
            with col:
                status_icon = "✅" if chk.passed else "❌"
                st.metric(
                    label=chk.name,
                    value=f"{status_icon} {chk.value}",
                    help=f"Seuil réglementaire : {chk.threshold}\n{chk.message}",
                )

        with st.expander("Détails des contrôles de conformité :"):
            for key, chk in report.checks.items():
                icon = "🟢" if chk.passed else "🔴"
                st.write(f"{icon} **{chk.name}** : {chk.message} *(valeur: {chk.value}, seuil: {chk.threshold})*")


if __name__ == "__main__":
    main()
