"""
Interface Utilisateur Native Desktop avec Tkinter pour EasyID.
100% local, rapide, sans navigateur et sans aucune dépendance web.
"""

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from PIL import Image, ImageTk

from easyid.config import DEFAULT_CONFIG, IDPhotoConfig
from easyid.pipeline import EasyIDPipeline, PipelineResult


def draw_official_overlay(image_bgr: np.ndarray, config: IDPhotoConfig, dpi: int = 300) -> np.ndarray:
    """Dessine le gabarit réglementaire officiel en surimpression pour contrôle visuel."""
    overlay = image_bgr.copy()
    h, w = image_bgr.shape[:2]

    # Ligne médiane verticale (axe sagittal)
    cv2.line(overlay, (w // 2, 0), (w // 2, h), (0, 255, 255), 1)

    # Ligne sommet du crâne cible
    top_margin_px = config.target_top_margin_px(dpi)
    cv2.line(overlay, (0, top_margin_px), (w, top_margin_px), (0, 255, 0), 1)

    # Zone menton réglementaire (entre 32 et 36 mm du sommet)
    chin_min_px = top_margin_px + config.target_face_height_px(dpi) - round((2.0 / 25.4) * dpi)
    chin_max_px = top_margin_px + config.target_face_height_px(dpi) + round((2.0 / 25.4) * dpi)

    cv2.line(overlay, (0, chin_min_px), (w, chin_min_px), (0, 165, 255), 1)
    cv2.line(overlay, (0, chin_max_px), (w, chin_max_px), (0, 165, 255), 1)

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
    """Fenêtre modale de capture Webcam en direct."""

    def __init__(self, parent, on_capture_callback):
        self.parent = parent
        self.on_capture_callback = on_capture_callback
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            messagebox.showerror(
                "Erreur Webcam",
                "Impossible d'accéder à la webcam. Vérifiez qu'elle est bien connectée.",
                parent=parent,
            )
            return

        self.window = tk.Toplevel(parent)
        self.window.title("📷 Capture Webcam — EasyID")
        self.window.transient(parent)
        self.window.grab_set()

        self.label = ttk.Label(self.window)
        self.label.pack(padx=10, pady=10)

        info = ttk.Label(
            self.window,
            text="Positionnez votre visage bien en face, droit, avec une expression neutre.",
            font=("Arial", 10, "italic"),
        )
        info.pack(pady=4)

        btn_box = ttk.Frame(self.window)
        btn_box.pack(pady=10)

        ttk.Button(btn_box, text="📸 Capturer", command=self.capture).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_box, text="Annuler", command=self.close).pack(side=tk.LEFT, padx=10)

        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.running = True
        self.current_frame = None
        self.update_video()

    def update_video(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if ret:
            # Miroir horizontal naturel pour selfie
            frame = cv2.flip(frame, 1)
            self.current_frame = frame
            # Aperçu
            img_tk = bgr_to_imagetk(frame, 500, 380)
            self.label.configure(image=img_tk)
            self.label.image = img_tk
        self.window.after(30, self.update_video)

    def capture(self):
        if self.current_frame is not None:
            frame_to_process = self.current_frame.copy()
            self.close()
            self.on_capture_callback(frame_to_process)

    def close(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.window.destroy()


class EasyIDTkinterApp:
    """Application principale Tkinter pour EasyID."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("EasyID — Photos d'Identité Conformes (ANTS / ICAO)")
        self.root.geometry("1100x820")
        self.root.minsize(950, 720)

        self.pipeline = EasyIDPipeline()
        self.current_input_bgr: Optional[np.ndarray] = None
        self.current_result: Optional[PipelineResult] = None

        # Variables de contrôle
        self.var_replace_bg = tk.BooleanVar(value=True)
        self.var_show_overlay = tk.BooleanVar(value=True)
        self.var_dpi = tk.IntVar(value=300)

        # Style
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self._build_ui()

        # Charger la démo par défaut au démarrage si présente
        sample = Path(__file__).resolve().parent.parent.parent / "tests" / "sample_portrait.jpg"
        if sample.exists():
            img = cv2.imread(str(sample))
            if img is not None:
                self.load_image_array(img)

    def _build_ui(self):
        # 1. En-tête
        header_frame = ttk.Frame(self.root, padding=10)
        header_frame.pack(fill=tk.X)

        title = ttk.Label(
            header_frame,
            text="📸 EasyID — Photos d'Identité Conformes",
            font=("Arial", 16, "bold"),
        )
        title.pack(anchor=tk.W)

        subtitle = ttk.Label(
            header_frame,
            text="Format officiel 35 × 45 mm • Taille visage 32 à 36 mm (70 à 80%) • 100% Local sur votre machine",
            font=("Arial", 10),
            foreground="#555555",
        )
        subtitle.pack(anchor=tk.W)

        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # 2. Barre d'outils / Contrôles
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(fill=tk.X)

        ttk.Button(toolbar, text="📁 Ouvrir une photo...", command=self.on_open_file).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="📷 Webcam", command=self.on_open_webcam).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="🖼️ Démo", command=self.on_load_demo).pack(side=tk.LEFT, padx=4)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Checkbutton(
            toolbar,
            text="Fond neutre (gris officiel)",
            variable=self.var_replace_bg,
            command=self.reprocess_current,
        ).pack(side=tk.LEFT, padx=6)

        ttk.Checkbutton(
            toolbar,
            text="Gabarit officiel (32-36mm)",
            variable=self.var_show_overlay,
            command=self.update_photo_display,
        ).pack(side=tk.LEFT, padx=6)

        ttk.Label(toolbar, text="Résolution :").pack(side=tk.LEFT, padx=(10, 2))
        dpi_combo = ttk.Combobox(
            toolbar,
            values=[300, 600],
            textvariable=self.var_dpi,
            width=5,
            state="readonly",
        )
        dpi_combo.pack(side=tk.LEFT, padx=2)
        dpi_combo.bind("<<ComboboxSelected>>", lambda e: self.reprocess_current())
        ttk.Label(toolbar, text="DPI").pack(side=tk.LEFT, padx=(1, 10))

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self.btn_save_photo = ttk.Button(
            toolbar, text="💾 Sauvegarder Photo", command=self.on_save_photo, state=tk.DISABLED
        )
        self.btn_save_photo.pack(side=tk.LEFT, padx=4)

        self.btn_save_sheet = ttk.Button(
            toolbar, text="🖨️ Sauvegarder Planche 10x15", command=self.on_save_sheet, state=tk.DISABLED
        )
        self.btn_save_sheet.pack(side=tk.LEFT, padx=4)

        # 3. Zone principale : Aperçus
        main_content = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_content.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Panneau gauche : Photo source
        frame_src = ttk.LabelFrame(main_content, text="1. Photo Source", padding=8)
        main_content.add(frame_src, weight=1)

        self.lbl_src_img = ttk.Label(frame_src, text="Aucune photo chargée", anchor=tk.CENTER)
        self.lbl_src_img.pack(fill=tk.BOTH, expand=True)

        self.lbl_src_info = ttk.Label(frame_src, text="", font=("Arial", 9))
        self.lbl_src_info.pack(anchor=tk.W, pady=2)

        # Panneau centre : Photo d'identité (35x45mm)
        frame_id = ttk.LabelFrame(main_content, text="2. Photo d'Identité (35 × 45 mm)", padding=8)
        main_content.add(frame_id, weight=1)

        self.lbl_id_img = ttk.Label(frame_id, text="En attente de traitement", anchor=tk.CENTER)
        self.lbl_id_img.pack(fill=tk.BOTH, expand=True)

        self.lbl_id_info = ttk.Label(frame_id, text="", font=("Arial", 9, "bold"))
        self.lbl_id_info.pack(anchor=tk.W, pady=2)

        # Panneau droite : Planche 10x15 cm
        frame_sheet = ttk.LabelFrame(main_content, text="3. Planche 10 × 15 cm (6 photos)", padding=8)
        main_content.add(frame_sheet, weight=1)

        self.lbl_sheet_img = ttk.Label(frame_sheet, text="En attente", anchor=tk.CENTER)
        self.lbl_sheet_img.pack(fill=tk.BOTH, expand=True)

        self.lbl_sheet_info = ttk.Label(
            frame_sheet,
            text="Gabarit 10x15cm avec repères de découpe",
            font=("Arial", 9),
        )
        self.lbl_sheet_info.pack(anchor=tk.W, pady=2)

        # 4. Panneau inférieur : Rapport de conformité réglementaire ANTS / ICAO
        frame_bottom = ttk.LabelFrame(self.root, text="📋 Contrôle de Conformité Réglementaire (ANTS / ICAO)", padding=8)
        frame_bottom.pack(fill=tk.X, padx=10, pady=8)

        self.lbl_overall_status = ttk.Label(
            frame_bottom,
            text="Statut : En attente d'analyse",
            font=("Arial", 11, "bold"),
        )
        self.lbl_overall_status.pack(anchor=tk.W, pady=2)

        self.tree_report = ttk.Treeview(
            frame_bottom,
            columns=("status", "name", "value", "threshold", "comment"),
            show="headings",
            height=6,
        )
        self.tree_report.heading("status", text="Statut")
        self.tree_report.heading("name", text="Critère officiel")
        self.tree_report.heading("value", text="Valeur mesurée")
        self.tree_report.heading("threshold", text="Norme")
        self.tree_report.heading("comment", text="Diagnostic")

        self.tree_report.column("status", width=70, anchor=tk.CENTER)
        self.tree_report.column("name", width=220, anchor=tk.W)
        self.tree_report.column("value", width=110, anchor=tk.CENTER)
        self.tree_report.column("threshold", width=100, anchor=tk.CENTER)
        self.tree_report.column("comment", width=460, anchor=tk.W)

        self.tree_report.pack(fill=tk.X, expand=True, pady=4)

        # 5. Barre de statut
        self.statusbar = ttk.Label(self.root, text="Prêt.", relief=tk.SUNKEN, anchor=tk.W, padding=4)
        self.statusbar.pack(fill=tk.X, side=tk.BOTTOM)

    # --- Actions ---

    def on_open_file(self):
        file_path = filedialog.askopenfilename(
            parent=self.root,
            title="Sélectionner une photo de portrait",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.webp *.bmp"),
                ("Tous les fichiers", "*.*"),
            ],
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
        # Affichage aperçu source
        img_tk = bgr_to_imagetk(image_bgr, 320, 360)
        self.lbl_src_img.configure(image=img_tk, text="")
        self.lbl_src_img.image = img_tk
        h, w = image_bgr.shape[:2]
        self.lbl_src_info.configure(text=f"Résolution d'origine : {w} × {h} px")

        self.reprocess_current()

    def reprocess_current(self):
        if self.current_input_bgr is None:
            return

        self.statusbar.configure(text="Traitement en cours (détection faciale, alignement, recadrage)...")
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
            self.statusbar.configure(text=f"Erreur : {res.error_message}")
            messagebox.showwarning("Non conforme", f"Analyse impossible : {res.error_message}")
            self.btn_save_photo.configure(state=tk.DISABLED)
            self.btn_save_sheet.configure(state=tk.DISABLED)
            return

        self.btn_save_photo.configure(state=tk.NORMAL)
        self.btn_save_sheet.configure(state=tk.NORMAL)

        # Mise à jour des affichages
        self.update_photo_display()

        # Affichage planche
        if res.printable_sheet is not None:
            sheet_tk = bgr_to_imagetk(res.printable_sheet, 320, 220)
            self.lbl_sheet_img.configure(image=sheet_tk, text="")
            self.lbl_sheet_img.image = sheet_tk
            sh_h, sh_w = res.printable_sheet.shape[:2]
            self.lbl_sheet_info.configure(text=f"Planche 10×15 cm : {sh_w} × {sh_h} px ({dpi} DPI)")

        # Mise à jour rapport
        self._populate_report(res)
        self.statusbar.configure(text="Traitement terminé avec succès.")

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
            text=f"Format : 35×45 mm ({w}×{h} px)\n"
            f"Visage : {self.current_result.face_height_mm:.1f} mm ({self.current_result.face_ratio_percent:.1f}%)"
        )

    def _populate_report(self, res: PipelineResult):
        # Nettoyer l'ancien rapport
        for item in self.tree_report.get_children():
            self.tree_report.delete(item)

        report = res.compliance_report
        if not report:
            return

        if report.passed:
            self.lbl_overall_status.configure(
                text=f"✅ CONFORME — Score de conformité : {report.score:.0f}%",
                foreground="#008800",
            )
        else:
            self.lbl_overall_status.configure(
                text=f"⚠️ ATTENTION : Non-conformités détectées — Score : {report.score:.0f}%",
                foreground="#CC6600",
            )

        for key, chk in report.checks.items():
            status_icon = "✅ Conforme" if chk.passed else "❌ Rejet"
            self.tree_report.insert(
                "",
                tk.END,
                values=(status_icon, chk.name, str(chk.value), chk.threshold, chk.message),
            )

    def on_save_photo(self):
        if not self.current_result or self.current_result.id_photo is None:
            return

        dest = filedialog.asksaveasfilename(
            parent=self.root,
            title="Enregistrer la photo d'identité (35×45 mm)",
            defaultextension=".jpg",
            initialfile="photo_identite_35x45.jpg",
            filetypes=[("Image JPEG", "*.jpg"), ("Image PNG", "*.png")],
        )
        if dest:
            self.current_result.save(output_photo_path=dest, dpi=self.var_dpi.get())
            messagebox.showinfo("Succès", f"Photo enregistrée avec métadonnées DPI dans :\n{dest}")

    def on_save_sheet(self):
        if not self.current_result or self.current_result.printable_sheet is None:
            return

        dest = filedialog.asksaveasfilename(
            parent=self.root,
            title="Enregistrer la planche d'impression (10×15 cm)",
            defaultextension=".jpg",
            initialfile="planche_impression_10x15.jpg",
            filetypes=[("Image JPEG", "*.jpg"), ("Image PNG", "*.png")],
        )
        if dest:
            self.current_result.save(
                output_photo_path="/tmp/easyid_tmp_photo.jpg",
                output_sheet_path=dest,
                dpi=self.var_dpi.get(),
            )
            messagebox.showinfo("Succès", f"Planche d'impression 10x15 cm enregistrée dans :\n{dest}")


def start_app():
    root = tk.Tk()
    app = EasyIDTkinterApp(root)
    root.mainloop()


if __name__ == "__main__":
    start_app()
