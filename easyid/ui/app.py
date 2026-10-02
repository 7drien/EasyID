"""
Interface Utilisateur Moderne Desktop pour EasyID.
Basée sur Tkinter et ttkbootstrap pour un design contemporain, épuré et réactif.
Intègre le sélecteur de fichiers natif moderne (Zenity sous Linux avec aperçus/vignettes)
et une capture webcam guidée avec gabarit ovale en temps réel.
"""

from pathlib import Path
import shutil
import subprocess
from typing import Optional
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

import ttkbootstrap as tb
from ttkbootstrap.constants import *

from easyid.config import DEFAULT_CONFIG, IDPhotoConfig
from easyid.pipeline import EasyIDPipeline, PipelineResult


def ask_open_image_file(parent=None, title="Sélectionner une photo de portrait") -> Optional[str]:
    """
    Ouvre le sélecteur de fichiers natif moderne du bureau (Zenity / GTK sur Linux avec
    vignettes d'images, dossiers récents et favoris), avec fallback automatique sur Tkinter.
    """
    zenity_bin = shutil.which("zenity")
    if zenity_bin:
        try:
            cmd = [
                zenity_bin,
                "--file-selection",
                f"--title={title}",
                "--file-filter=Images (*.jpg, *.png, *.webp) | *.jpg *.jpeg *.png *.webp *.JPG *.JPEG *.PNG *.WEBP *.bmp *.BMP",
                "--file-filter=Tous les fichiers | *",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                chosen = result.stdout.strip()
                if chosen and Path(chosen).is_file():
                    return chosen
            elif result.returncode == 1:
                # L'utilisateur a cliqué sur 'Annuler'
                return None
        except Exception:
            pass

    # Fallback standard
    return filedialog.askopenfilename(
        parent=parent,
        title=title,
        filetypes=[
            ("Images", "*.jpg *.jpeg *.png *.webp *.bmp"),
            ("Tous les fichiers", "*.*"),
        ],
    )


def ask_save_image_file(
    parent=None,
    title="Enregistrer l'image",
    initial_file="photo.jpg",
    file_type="jpg",
) -> Optional[str]:
    """
    Ouvre la boîte de dialogue d'enregistrement native moderne du système (Zenity)
    avec confirmation d'écrasement, ou fallback sur Tkinter.
    """
    zenity_bin = shutil.which("zenity")
    if zenity_bin:
        try:
            filter_arg = (
                "--file-filter=Images JPEG (*.jpg) | *.jpg *.jpeg"
                if file_type == "jpg"
                else "--file-filter=Images PNG (*.png) | *.png"
            )
            cmd = [
                zenity_bin,
                "--file-selection",
                "--save",
                "--confirm-overwrite",
                f"--title={title}",
                f"--filename={initial_file}",
                filter_arg,
                "--file-filter=Tous les fichiers | *",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                chosen = result.stdout.strip()
                if chosen:
                    if not chosen.lower().endswith(f".{file_type}"):
                        chosen = f"{chosen}.{file_type}"
                    return chosen
            elif result.returncode == 1:
                return None
        except Exception:
            pass

    return filedialog.asksaveasfilename(
        parent=parent,
        title=title,
        defaultextension=f".{file_type}",
        initialfile=initial_file,
        filetypes=[("Image JPEG", "*.jpg"), ("Image PNG", "*.png")],
    )


def draw_official_overlay(image_bgr: np.ndarray, config: IDPhotoConfig, dpi: int = 300) -> np.ndarray:
    """Dessine le gabarit réglementaire officiel en surimpression pour contrôle visuel."""
    overlay = image_bgr.copy()
    h, w = image_bgr.shape[:2]

    # 1. Ligne médiane verticale (axe sagittal de symétrie)
    cv2.line(overlay, (w // 2, 0), (w // 2, h), (0, 230, 255), 1)

    # 2. Zone supérieure recommandée pour la tête / cheveux (2 à 5 mm)
    top_min_px = round((2.0 / 25.4) * dpi)
    top_ideal_px = round((3.5 / 25.4) * dpi)
    cv2.line(overlay, (0, top_min_px), (w, top_min_px), (0, 255, 100), 1)
    cv2.putText(overlay, "Limite haute cheveux (>=2mm)", (8, top_min_px - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 220, 100), 1)

    # 3. Ligne des yeux recommandée (environ 24 à 30 mm du bas)
    eyes_ref_px = h - round((26.0 / 25.4) * dpi)
    cv2.line(overlay, (w // 4, eyes_ref_px), (3 * w // 4, eyes_ref_px), (255, 200, 0), 1)
    cv2.putText(overlay, "Axe des yeux", (w // 4 + 4, eyes_ref_px - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 200, 0), 1)

    # 4. Zone basse menton réglementaire (environ 5 à 9 mm du bas)
    chin_min_px = h - round((5.0 / 25.4) * dpi)
    chin_max_px = h - round((9.0 / 25.4) * dpi)
    cv2.line(overlay, (0, chin_min_px), (w, chin_min_px), (0, 140, 255), 1)
    cv2.line(overlay, (0, chin_max_px), (w, chin_max_px), (0, 140, 255), 1)
    cv2.putText(overlay, "Zone menton", (8, chin_min_px + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 140, 255), 1)

    return cv2.addWeighted(overlay, 0.85, image_bgr, 0.15, 0)


def bgr_to_imagetk(bgr_img: np.ndarray, max_w: int, max_h: int) -> ImageTk.PhotoImage:
    """Redimensionne pour aperçu et convertit BGR OpenCV en ImageTk.PhotoImage."""
    h, w = bgr_img.shape[:2]
    scale = min(max_w / w, max_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(bgr_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(pil_img)


class WebcamCaptureDialog:
    """Fenêtre modale de capture Webcam en direct avec guide ovale intégré."""

    def __init__(self, parent, on_capture_callback):
        self.parent = parent
        self.on_capture_callback = on_capture_callback
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            messagebox.showerror(
                "Erreur Webcam",
                "Impossible d'accéder à la caméra. Vérifiez qu'elle est bien connectée.",
                parent=parent,
            )
            return

        self.window = tb.Toplevel(parent)
        self.window.title("📷 Capture Webcam — EasyID")
        self.window.transient(parent)
        self.window.grab_set()

        header = tb.Frame(self.window, padding=10)
        header.pack(fill=X)
        tb.Label(
            header,
            text="Placez votre visage dans l'ovale guide, bien de face avec une expression neutre",
            font=("Helvetica", 10, "italic"),
            bootstyle="secondary",
        ).pack(anchor=CENTER)

        self.lbl_video = tb.Label(self.window)
        self.lbl_video.pack(padx=15, pady=5)

        btn_box = tb.Frame(self.window, padding=12)
        btn_box.pack(fill=X)

        tb.Button(
            btn_box,
            text="📸 Prendre la photo",
            bootstyle="success",
            command=self.capture,
        ).pack(side=LEFT, expand=True, padx=8)

        tb.Button(
            btn_box,
            text="Annuler",
            bootstyle="secondary-outline",
            command=self.close,
        ).pack(side=LEFT, expand=True, padx=8)

        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.running = True
        self.current_raw_frame = None
        self.update_video()

    def update_video(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if ret:
            # Effet miroir naturel pour selfie
            frame = cv2.flip(frame, 1)
            self.current_raw_frame = frame.copy()

            # Guide visuel ovale semi-transparent
            h_f, w_f = frame.shape[:2]
            center_ov = (w_f // 2, int(h_f * 0.48))
            axes_ov = (int(w_f * 0.22), int(h_f * 0.35))
            overlay_cam = frame.copy()
            cv2.ellipse(overlay_cam, center_ov, axes_ov, 0, 0, 360, (0, 230, 200), 2)
            cv2.circle(overlay_cam, (center_ov[0], center_ov[1] - axes_ov[1]), 5, (0, 255, 100), -1)
            cv2.circle(overlay_cam, (center_ov[0], center_ov[1] + axes_ov[1]), 5, (0, 140, 255), -1)
            display_frame = cv2.addWeighted(overlay_cam, 0.75, frame, 0.25, 0)

            img_tk = bgr_to_imagetk(display_frame, 520, 390)
            self.lbl_video.configure(image=img_tk)
            self.lbl_video.image = img_tk
        self.window.after(30, self.update_video)

    def capture(self):
        if self.current_raw_frame is not None:
            clean_photo = self.current_raw_frame.copy()
            self.close()
            self.on_capture_callback(clean_photo)

    def close(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.window.destroy()


class EasyIDModernApp:
    """Application Desktop moderne pour EasyID."""

    AVAILABLE_THEMES = [
        "cosmo",      # Thème clair professionnel (défaut)
        "flatly",     # Thème clair moderne épuré
        "litera",     # Thème clair minimaliste
        "darkly",     # Thème sombre moderne
        "superhero",  # Thème sombre bleuté
        "cyborg",     # Thème sombre high-tech
    ]

    def __init__(self, root: tb.Window):
        self.root = root
        self.root.title("EasyID — Photos d'Identité Conformes (ANTS / ICAO)")
        self.root.geometry("1160x860")
        self.root.minsize(980, 720)

        self.pipeline = EasyIDPipeline()
        self.current_input_bgr: Optional[np.ndarray] = None
        self.current_result: Optional[PipelineResult] = None

        # Variables d'état
        self.var_replace_bg = tb.BooleanVar(value=True)
        self.var_show_overlay = tb.BooleanVar(value=True)
        self.var_dpi = tb.IntVar(value=300)
        self.var_theme = tb.StringVar(value="cosmo")

        self._build_ui()

        # Charger la démo d'exemple au démarrage si disponible
        sample = Path(__file__).resolve().parent.parent.parent / "tests" / "sample_portrait.jpg"
        if sample.exists():
            img = cv2.imread(str(sample))
            if img is not None:
                self.load_image_array(img)

    def _build_ui(self):
        # 1. En-tête moderne avec dégradé / barre supérieure
        header = tb.Frame(self.root, padding=(15, 12), bootstyle="secondary")
        header.pack(fill=X)

        title_box = tb.Frame(header, bootstyle="secondary")
        title_box.pack(side=LEFT)

        tb.Label(
            title_box,
            text="📸 EasyID",
            font=("Helvetica", 17, "bold"),
            bootstyle="inverse-secondary",
        ).pack(side=LEFT, padx=(0, 10))

        tb.Label(
            title_box,
            text="Norme ANTS / ISO 19794-5",
            font=("Helvetica", 9, "bold"),
            bootstyle="success-inverse",
            padding=(6, 2),
        ).pack(side=LEFT, padx=(0, 12))

        tb.Label(
            title_box,
            text="Format 35×45 mm • Visage 32–36 mm (70–80%) • 100% Hors-ligne",
            font=("Helvetica", 10),
            bootstyle="inverse-secondary",
        ).pack(side=LEFT)

        # Sélecteur de thème dans l'en-tête
        theme_box = tb.Frame(header, bootstyle="secondary")
        theme_box.pack(side=RIGHT)

        tb.Label(
            theme_box,
            text="🎨 Thème :",
            font=("Helvetica", 9),
            bootstyle="inverse-secondary",
        ).pack(side=LEFT, padx=4)

        theme_combo = tb.Combobox(
            theme_box,
            values=self.AVAILABLE_THEMES,
            textvariable=self.var_theme,
            width=10,
            state="readonly",
        )
        theme_combo.pack(side=LEFT)
        theme_combo.bind("<<ComboboxSelected>>", self.on_theme_changed)

        # 2. Barre d'actions & options modernes
        toolbar_card = tb.Labelframe(self.root, text="Contrôles & Options", padding=10)
        toolbar_card.pack(fill=X, padx=15, pady=(8, 4))

        # Boutons sources
        btn_src_box = tb.Frame(toolbar_card)
        btn_src_box.pack(side=LEFT)

        tb.Button(
            btn_src_box,
            text="📁 Ouvrir photo...",
            bootstyle="primary",
            command=self.on_open_file,
        ).pack(side=LEFT, padx=4)

        tb.Button(
            btn_src_box,
            text="📷 Webcam",
            bootstyle="info-outline",
            command=self.on_open_webcam,
        ).pack(side=LEFT, padx=4)

        tb.Button(
            btn_src_box,
            text="🖼️ Démo",
            bootstyle="secondary-outline",
            command=self.on_load_demo,
        ).pack(side=LEFT, padx=4)

        tb.Separator(toolbar_card, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=12)

        # Toggles modernes
        toggles_box = tb.Frame(toolbar_card)
        toggles_box.pack(side=LEFT)

        tb.Checkbutton(
            toggles_box,
            text="Fond neutre officiel",
            variable=self.var_replace_bg,
            bootstyle="success-round-toggle",
            command=self.reprocess_current,
        ).pack(side=LEFT, padx=8)

        tb.Checkbutton(
            toggles_box,
            text="Gabarit officiel (32-36mm)",
            variable=self.var_show_overlay,
            bootstyle="warning-round-toggle",
            command=self.update_photo_display,
        ).pack(side=LEFT, padx=8)

        tb.Separator(toolbar_card, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=12)

        # DPI
        dpi_box = tb.Frame(toolbar_card)
        dpi_box.pack(side=LEFT)
        tb.Label(dpi_box, text="DPI :").pack(side=LEFT, padx=2)
        dpi_combo = tb.Combobox(
            dpi_box,
            values=[300, 600],
            textvariable=self.var_dpi,
            width=5,
            state="readonly",
        )
        dpi_combo.pack(side=LEFT, padx=2)
        dpi_combo.bind("<<ComboboxSelected>>", lambda e: self.reprocess_current())

        # Boutons de sauvegarde
        btn_save_box = tb.Frame(toolbar_card)
        btn_save_box.pack(side=RIGHT)

        self.btn_save_photo = tb.Button(
            btn_save_box,
            text="💾 Enregistrer Photo 35×45",
            bootstyle="success",
            command=self.on_save_photo,
            state=DISABLED,
        )
        self.btn_save_photo.pack(side=LEFT, padx=4)

        self.btn_save_sheet = tb.Button(
            btn_save_box,
            text="🖨️ Enregistrer Planche 10×15",
            bootstyle="primary-outline",
            command=self.on_save_sheet,
            state=DISABLED,
        )
        self.btn_save_sheet.pack(side=LEFT, padx=4)

        # 3. Zone d'aperçu à 3 panneaux (Cartes modernes)
        preview_container = tb.Frame(self.root, padding=(15, 4))
        preview_container.pack(fill=BOTH, expand=True)

        preview_grid = tb.Frame(preview_container)
        preview_grid.pack(fill=BOTH, expand=True)

        # Carte 1 : Source
        card_src = tb.Labelframe(preview_grid, text="1. Photo d'Origine", padding=8, bootstyle="default")
        card_src.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 6))

        self.lbl_src_img = tb.Label(card_src, text="Aucune photo chargée\n\nCliquez sur 'Ouvrir photo' ou 'Webcam'", anchor=CENTER)
        self.lbl_src_img.pack(fill=BOTH, expand=True)

        self.lbl_src_info = tb.Label(card_src, text="", font=("Helvetica", 9), bootstyle="secondary")
        self.lbl_src_info.pack(anchor=W, pady=(4, 0))

        # Carte 2 : Photo d'identité
        card_id = tb.Labelframe(preview_grid, text="2. Photo d'Identité (35 × 45 mm)", padding=8, bootstyle="primary")
        card_id.pack(side=LEFT, fill=BOTH, expand=True, padx=6)

        self.lbl_id_img = tb.Label(card_id, text="En attente de traitement", anchor=CENTER)
        self.lbl_id_img.pack(fill=BOTH, expand=True)

        self.lbl_id_info = tb.Label(card_id, text="", font=("Helvetica", 9, "bold"), bootstyle="primary")
        self.lbl_id_info.pack(anchor=W, pady=(4, 0))

        # Carte 3 : Planche d'impression
        card_sheet = tb.Labelframe(preview_grid, text="3. Planche 10 × 15 cm (6 photos)", padding=8, bootstyle="info")
        card_sheet.pack(side=LEFT, fill=BOTH, expand=True, padx=(6, 0))

        self.lbl_sheet_img = tb.Label(card_sheet, text="En attente", anchor=CENTER)
        self.lbl_sheet_img.pack(fill=BOTH, expand=True)

        self.lbl_sheet_info = tb.Label(card_sheet, text="6 photos prêtes avec repères", font=("Helvetica", 9), bootstyle="secondary")
        self.lbl_sheet_info.pack(anchor=W, pady=(4, 0))

        # 4. Panneau de conformité ANTS / ICAO
        qa_frame = tb.Labelframe(self.root, text="📋 Contrôle Qualité Réglementaire (ANTS / ICAO)", padding=10)
        qa_frame.pack(fill=X, padx=15, pady=(4, 8))

        # Bannière de statut global
        self.banner_score = tb.Label(
            qa_frame,
            text="Statut : En attente d'analyse",
            font=("Helvetica", 11, "bold"),
            padding=(10, 4),
            bootstyle="secondary",
        )
        self.banner_score.pack(fill=X, pady=(0, 6))

        # Table des critères
        self.tree_report = tb.Treeview(
            qa_frame,
            columns=("status", "name", "value", "threshold", "comment"),
            show="headings",
            height=6,
            bootstyle="primary",
        )
        self.tree_report.heading("status", text="Statut")
        self.tree_report.heading("name", text="Critère officiel")
        self.tree_report.heading("value", text="Valeur mesurée")
        self.tree_report.heading("threshold", text="Norme")
        self.tree_report.heading("comment", text="Diagnostic")

        self.tree_report.column("status", width=80, anchor=CENTER)
        self.tree_report.column("name", width=230, anchor=W)
        self.tree_report.column("value", width=120, anchor=CENTER)
        self.tree_report.column("threshold", width=100, anchor=CENTER)
        self.tree_report.column("comment", width=480, anchor=W)

        self.tree_report.pack(fill=X, expand=True)

        # 5. Barre de statut inférieure
        self.statusbar = tb.Label(
            self.root,
            text="Prêt. Ouvrez une photo ou utilisez la webcam pour commencer.",
            font=("Helvetica", 9),
            padding=(10, 4),
            bootstyle="inverse-secondary",
        )
        self.statusbar.pack(fill=X, side=BOTTOM)

    # --- Événements & Traitement ---

    def on_theme_changed(self, event=None):
        new_theme = self.var_theme.get()
        tb.Style().theme_use(new_theme)

    def on_open_file(self):
        file_path = ask_open_image_file(
            parent=self.root,
            title="Sélectionner une photo de portrait",
        )
        if file_path:
            img = cv2.imread(file_path)
            if img is None:
                messagebox.showerror("Erreur", f"Impossible d'ouvrir l'image : {file_path}")
                return
            self.load_image_array(img)

    def on_open_webcam(self):
        WebcamCaptureDialog(self.root, on_capture_callback=self.load_image_array)

    def on_load_demo(self):
        sample = Path(__file__).resolve().parent.parent.parent / "tests" / "sample_portrait.jpg"
        if not sample.exists():
            messagebox.showwarning("Démo", "L'image d'exemple tests/sample_portrait.jpg est introuvable.")
            return
        img = cv2.imread(str(sample))
        if img is not None:
            self.load_image_array(img)

    def load_image_array(self, image_bgr: np.ndarray):
        self.current_input_bgr = image_bgr
        img_tk = bgr_to_imagetk(image_bgr, 340, 360)
        self.lbl_src_img.configure(image=img_tk, text="")
        self.lbl_src_img.image = img_tk
        h, w = image_bgr.shape[:2]
        self.lbl_src_info.configure(text=f"Résolution originale : {w} × {h} px")

        self.reprocess_current()

    def reprocess_current(self):
        if self.current_input_bgr is None:
            return

        self.statusbar.configure(text="⏳ Traitement par Deep Learning (détection, alignement, crop)...")
        self.root.update_idletasks()

        dpi = self.var_dpi.get()
        replace_bg = self.var_replace_bg.get()

        res = self.pipeline.process(
            self.current_input_bgr,
            dpi=dpi,
            replace_background=replace_bg,
            generate_sheet=True,
        )
        self.current_result = res

        if not res.success:
            self.statusbar.configure(text=f"❌ Erreur : {res.error_message}")
            messagebox.showwarning("Non conforme", f"Analyse impossible : {res.error_message}")
            self.btn_save_photo.configure(state=DISABLED)
            self.btn_save_sheet.configure(state=DISABLED)
            return

        self.btn_save_photo.configure(state=NORMAL)
        self.btn_save_sheet.configure(state=NORMAL)

        # Mise à jour affichage photo ID
        self.update_photo_display()

        # Mise à jour affichage planche
        if res.printable_sheet is not None:
            sheet_tk = bgr_to_imagetk(res.printable_sheet, 340, 240)
            self.lbl_sheet_img.configure(image=sheet_tk, text="")
            self.lbl_sheet_img.image = sheet_tk
            sh_h, sh_w = res.printable_sheet.shape[:2]
            self.lbl_sheet_info.configure(text=f"Planche 10×15 cm : {sh_w} × {sh_h} px ({dpi} DPI)")

        # Mise à jour tableau de conformité
        self._populate_report(res)
        self.statusbar.configure(text="✅ Traitement terminé avec succès. Photo prête à sauvegarder.")

    def update_photo_display(self):
        if not self.current_result or not self.current_result.success or self.current_result.id_photo is None:
            return

        photo = self.current_result.id_photo
        dpi = self.var_dpi.get()

        if self.var_show_overlay.get():
            photo = draw_official_overlay(photo, DEFAULT_CONFIG, dpi=dpi)

        id_tk = bgr_to_imagetk(photo, 280, 360)
        self.lbl_id_img.configure(image=id_tk, text="")
        self.lbl_id_img.image = id_tk

        h, w = self.current_result.id_photo.shape[:2]
        self.lbl_id_info.configure(
            text=f"Format : 35×45 mm ({w}×{h} px à {dpi} DPI)\n"
            f"Taille visage : {self.current_result.face_height_mm:.1f} mm ({self.current_result.face_ratio_percent:.1f}% de la hauteur)"
        )

    def _populate_report(self, res: PipelineResult):
        for item in self.tree_report.get_children():
            self.tree_report.delete(item)

        report = res.compliance_report
        if not report:
            return

        if report.passed:
            self.banner_score.configure(
                text=f"✅ CONFORME AUX NORMES OFFICIELLES (Score : {report.score:.0f}%)",
                bootstyle="success",
            )
        else:
            self.banner_score.configure(
                text=f"⚠️ ATTENTION : CRITÈRES NON CONFORMES DÉTECTÉS (Score : {report.score:.0f}%)",
                bootstyle="warning",
            )

        for key, chk in report.checks.items():
            status_text = "✅ Conforme" if chk.passed else "❌ Non conforme"
            self.tree_report.insert(
                "",
                tk.END,
                values=(status_text, chk.name, str(chk.value), chk.threshold, chk.message),
            )

    def on_save_photo(self):
        if not self.current_result or self.current_result.id_photo is None:
            return

        dest = ask_save_image_file(
            parent=self.root,
            title="Enregistrer la photo d'identité (35×45 mm)",
            initial_file="photo_identite_35x45.jpg",
            file_type="jpg",
        )
        if dest:
            self.current_result.save(output_photo_path=dest, dpi=self.var_dpi.get())
            messagebox.showinfo("Succès", f"Photo enregistrée avec métadonnées DPI dans :\n{dest}")

    def on_save_sheet(self):
        if not self.current_result or self.current_result.printable_sheet is None:
            return

        dest = ask_save_image_file(
            parent=self.root,
            title="Enregistrer la planche d'impression (10×15 cm)",
            initial_file="planche_impression_10x15.jpg",
            file_type="jpg",
        )
        if dest:
            self.current_result.save(
                output_photo_path="/tmp/easyid_tmp_photo.jpg",
                output_sheet_path=dest,
                dpi=self.var_dpi.get(),
            )
            messagebox.showinfo("Succès", f"Planche d'impression 10×15 cm enregistrée dans :\n{dest}")


def start_app():
    root = tb.Window(themename="cosmo")
    app = EasyIDModernApp(root)
    root.mainloop()


if __name__ == "__main__":
    start_app()
