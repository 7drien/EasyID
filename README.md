# 📸 EasyID — Photos d'Identité Conformes par Deep Learning

Générateur et vérificateur automatique de photos d'identité aux **normes officielles françaises (ANTS) et internationales (ISO/IEC 19794-5 / ICAO)**.

---

## 🎯 Fonctionnalités Clés

- 📐 **Format Officiel Strict :**
  - Dimensions : **3,5 cm × 4,5 cm** (413 × 531 px à 300 DPI / 827 × 1063 px à 600 DPI).
  - Taille du visage : **3,2 cm à 3,6 cm** (**70 % à 80 %** de la hauteur totale).
  - Sommet du crâne anatomique (hors chevelure) au bas du menton.
- 🧠 **Vision par Ordinateur & Deep Learning :**
  - **Repères faciaux denses (MediaPipe Face Mesh 478 points)** avec détection des pupilles/iris.
  - **Alignement & Redressement automatique :** correction de l'angle d'inclinaison de la tête (*roll angle*).
  - **Détourage & Fond neutre :** remplacement automatique de l'arrière-plan par un gris clair neutre conforme ANTS (le blanc pur étant interdit).
- 📋 **Contrôle Qualité & Conformité ICAO :**
  - Vérification de la pose de face (angles yaw et pitch).
  - Détection des yeux ouverts et regard fixé.
  - Vérification de la bouche fermée et expression neutre.
  - Indice de netteté (variance du Laplacien).
- 🖨️ **Planche d'Impression 10 × 15 cm :**
  - Planche standard 4×6 pouces prête à imprimer contenant 6 photos avec repères de découpe discrets.
  - Métadonnées EXIF réelles de 300 ou 600 DPI.

---

## 🚀 Installation

```bash
# Activer l'environnement virtuel
source .venv/bin/activate

# Installer les dépendances
pip install -e .
```

---

## 💻 Utilisation

### 1. Interface Graphique Native Desktop (Tkinter)
Lancez l'application de bureau avec prévisualisation en direct, webcam et contrôle de conformité :

```bash
python main.py --ui
```

### 2. En Ligne de Commande (CLI)

```bash
# Traiter une photo personnelle
python main.py --input mon_portrait.jpg --output photo_id.jpg --sheet planche_10x15.jpg

# Choisir une résolution de 600 DPI
python main.py --input mon_portrait.jpg --dpi 600

# Exécuter la démonstration sur l'image d'exemple
python main.py
```

### 3. En Python (API)

```python
from easyid import EasyIDPipeline

pipeline = EasyIDPipeline()
result = pipeline.process("portrait.jpg", dpi=300)

if result.success:
    print(f"Visage : {result.face_height_mm} mm ({result.face_ratio_percent}%)")
    print(f"Conformité : {result.compliance_report.passed} ({result.compliance_report.score}%)")
    
    # Sauvegarde avec métadonnées DPI 300
    result.save("photo_identite.jpg", "planche_10x15.jpg", dpi=300)
```

---

## 🧪 Tests Automatisés

Le projet dispose d'une suite de tests unitaires et d'intégration validant les conversions métriques, le redressement affine et la conformité :

```bash
pytest -v
```

---

## 📁 Structure du Projet

```
EasyID/
├── Agents.md                  # Spécifications détaillées & guide pour agents IA
├── README.md                  # Documentation du projet
├── pyproject.toml             # Configuration & dépendances
├── main.py                    # Point d'entrée CLI et lanceur UI
├── easyid/
│   ├── config.py              # Normes ANTS/ICAO et conversions mm/px
│   ├── models.py              # Téléchargement & cache des modèles Deep Learning
│   ├── pipeline.py            # Orchestrateur complet (entrée -> sortie)
│   ├── printable.py           # Génération de la planche 10x15 cm (6 photos)
│   ├── core/
│   │   ├── detector.py        # Détecteur FaceLandmarker (repères 3D denses)
│   │   ├── geometry.py        # Calculs angulaires, estimation crânienne & matrice affine
│   │   ├── cropper.py         # Recadrage Lanczos haute résolution
│   │   ├── segmenter.py       # Détourage & fond neutre (SelfieSegmenter)
│   │   └── validator.py       # Contrôle de conformité réglementaire ICAO/ANTS
│   └── ui/
│       └── app.py             # Interface graphique native Desktop Tkinter (webcam + upload)
└── tests/
    ├── test_geometry.py       # Tests géométrie et redressement
    └── test_pipeline.py       # Tests d'intégration bout en bout
```
