# Projet Ball Tracking Tennis/Padel - Résumé

## 🎯 Objectif du projet
Tracker une balle de tennis/padel en vidéo et appliquer de l'IA pour l'analyse (rebonds, trajectoire, etc.)

---

## 📊 État de l'art - Ball Tracking

### Les défis principaux identifiés
1. **Ball tracking** - La balle est petite, rapide (200+ km/h), souvent floue (motion blur)
2. **Détection des rebonds** - Changement brusque de trajectoire, nécessite un tracking propre

### Évolution de TrackNet

| Version | Année | Innovation clé | Performance |
|---------|-------|----------------|-------------|
| TrackNetV1 | 2019 | Premier deep learning pour ball tracking, VGG-16 + DeconvNet | Baseline |
| TrackNetV2 | 2020 | Multi-input/output (3in-3out), skip connections, weighted BCE loss | ~156 FPS, F1: 97.1% |
| TrackNetV3 | 2023 | Background estimation, trajectory rectification | ~15 FPS, F1: 97.5% |
| TrackNetV4 | 2024 | **Motion Attention Maps** (seulement 2 params supplémentaires) | ~155 FPS, F1: 97.5% |

### TrackNetV4 - Le state of the art actuel

**Paper**: "TrackNetV4: Enhancing Fast Sports Object Tracking with Motion Attention Maps" (Raj et al., ICASSP 2025)

**Innovation principale**: Motion Prompt Layer qui génère des attention maps à partir des différences entre frames consécutives.

**Architecture**:
- Input: 3 frames consécutives (9 channels, 288×512)
- Motion Prompt Layer: Extrait l'attention sur le mouvement
- Encoder-Decoder: Style VGG avec skip connections
- Output: Heatmaps de probabilité (3 channels, 288×512)

**Résultats** (depuis le paper):

| Dataset | Modèle | Accuracy | F1 | FPS |
|---------|--------|----------|-----|-----|
| Tennis | TrackNetV2 | 94.6% | 97.1% | 156.9 |
| Tennis | TrackNetV2 + Motion (V4) | 95.2% | 97.5% | 155.7 |
| Shuttlecock | TrackNetV2 | 85.6% | 90.6% | 163.3 |
| Shuttlecock | TrackNetV2 + Motion (V4) | 86.6% | 91.4% | 161.1 |
| Shuttlecock | TrackNetV3 + Motion | 96.4% | 97.9% | 15.1 |

**Verdict**: V4 apporte des gains marginaux (+0.4-0.8% F1) mais est **plug-and-play** et réduit significativement les **false negatives** (balles manquées).

---

## 🔧 Ressources disponibles

### Repos GitHub

#### 1. TrackNetV4 (Repo officiel)
- **URL**: https://github.com/AnInsomniacy/tracknet-series-pytorch
- **Stars**: 10
- **Commits**: 121
- **Features**: 
  - Support Multi-GPU DDP
  - Config YAML centralisé
  - Stream video prediction
  - TrackNetV2 et V4 implémentés
- **Release**: v1.0.0 (20 déc 2025)

```bash
git clone https://github.com/AnInsomniacy/tracknet-series-pytorch.git
cd tracknet-series-pytorch
pip install -r requirements.txt
```

#### 2. Site officiel TrackNetV4
- **URL**: https://time.griffith.edu.au/paper-sites/tracknet-v4/
- **Note**: Code marqué "coming soon", pas de weights officiels disponibles

#### 3. Weights pré-entraînés TrackNetV2
- **URL**: https://drive.google.com/file/d/1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl/view
- **Source**: Repo yastrebksv/TrackNet
- **Utilisation**: Charger ces weights V2, ajouter le Motion Prompt Layer, fine-tuner

### Paper
- **ArXiv**: https://arxiv.org/abs/2409.14543
- **PDF**: Téléchargé et analysé

---

## 📁 Datasets disponibles

### 1. PadelTracker100 ⭐⭐⭐⭐⭐ (RECOMMANDÉ)

**URL**: https://zenodo.org/records/14653706

| Critère | Détail |
|---------|--------|
| Taille | ~100,000 frames |
| Source | World Padel Tour 2022 Finals |
| Annotations | Ball positions (x,y), player poses, shot events |
| Format | Frames + CSV annotations |
| Licence | CC-BY 4.0 |
| Téléchargement | 7.1 GB |

**Pourquoi c'est parfait**:
- ✅ Ball tracking déjà annoté
- ✅ 100k frames (dataset tennis = 17k)
- ✅ Single camera angle (moins d'occlusions)
- ✅ Shot events inclus (pour les rebonds!)
- ✅ Jupyter notebook pour explorer les données

### 2. Dataset Tennis original (TrackNet)
- ~17,000 frames
- 10 matchs
- Disponible via CoachAI

---

## 📋 Format des données pour TrackNet

### Structure source (PadelTracker100 - COCO format)
```
dataset/
├── 2022_BCN_FinalF_1.mp4          # Vidéo match
├── 2022_BCN_FinalM_1.mp4
└── labels/
    ├── 2022_BCN_FinalF_1_ball.json  # Annotations COCO (bbox balle)
    ├── 2022_BCN_FinalF_1_shots.csv  # Types de coups
    └── 2022_BCN_FinalM_1_ball.json
```

### Structure preprocessed (TrackNet format)
```
dataset/preprocessed/train/
├── match1/
│   ├── inputs/
│   │   └── frame0/
│   │       ├── 0.jpg, 1.jpg, 2.jpg...  # RGB 512×288
│   └── heatmaps/
│       └── frame0/
│           ├── 0.jpg, 1.jpg, 2.jpg...  # Grayscale gaussian 512×288
└── match2/
    └── ...
```

### Format des tenseurs
- **Input**: `[9, 288, 512]` - 3 frames RGB concaténées, normalisé [0,1]
- **Heatmap**: `[3, 288, 512]` - 3 heatmaps gaussiennes, normalisé [0,1]

### Pourquoi des vidéos et pas des photos?
TrackNet utilise **3 frames consécutives** en input (9 channels). Le modèle exploite:
- L'information temporelle pour prédire la trajectoire
- Le motion blur (indique la direction)
- Les différences entre frames (Motion Prompt Layer V4)

---

## 🚀 Plan d'action

### Étape 1: Setup ✅
```bash
# Cloner le repo TrackNetV4
git clone https://github.com/AnInsomniacy/tracknet-series-pytorch.git
cd tracknet-series-pytorch
uv sync  # ou pip install -r requirements.txt
```

### Étape 2: Télécharger le dataset ✅
```bash
# PadelTracker100 depuis Zenodo
# 7.1 GB - contient vidéos MP4 + annotations COCO JSON
# Télécharger manuellement depuis https://zenodo.org/records/14653706
# Extraire dans dataset/
```

### Étape 3: Explorer les données ✅
```bash
# Structure obtenue:
# dataset/
# ├── 2022_BCN_FinalF_1.mp4 (45934 frames, 30 FPS)
# ├── 2022_BCN_FinalM_1.mp4 (53953 frames, 30 FPS)
# └── labels/
#     ├── *_ball.json (positions COCO)
#     └── *_shots.csv (types de coups)
```

### Étape 4: Convertir les annotations ✅
Script créé: `preprocessing/convert_padeltracker.py`

```bash
# Test sur un échantillon (50 frames)
uv run python preprocessing/convert_padeltracker.py \
  --source dataset \
  --output dataset/preprocessed/test \
  --max-frames 50 \
  --force

# Conversion complète (~100k frames, ~20-30 min)
uv run python preprocessing/convert_padeltracker.py \
  --source dataset \
  --output dataset/preprocessed/train \
  --force
```

**Options disponibles**:
- `--sigma 3.0` : Taille du gaussian pour les heatmaps
- `--frames-per-group 100` : Frames par dossier
- `--max-frames N` : Limiter le nombre de frames (pour tests)

### Étape 5: Entraînement
```bash
# Configurer config.yaml puis:
uv run python train.py --config config.yaml

# TensorBoard pour monitoring:
tensorboard --logdir outputs/
```

### Étape 6: Inférence
```bash
# Sur une vidéo
uv run python predict/streem_video_predict.py \
  --model outputs/exp_*/checkpoints/best_model.pth \
  --input ma_video.mp4 \
  --output predictions/
```

---

## 🎾 Spécificités Padel vs Tennis

| Aspect | Tennis | Padel |
|--------|--------|-------|
| Vitesse balle | Jusqu'à 250 km/h | Jusqu'à 180 km/h |
| Rebonds | Sol uniquement | Sol + vitres + grillage |
| Occlusions | Rares | Fréquentes (vitres, joueurs) |
| Taille terrain | Plus grand | Plus petit, caméra plus proche |

**Implications pour le tracking**:
- Plus de rebonds à détecter en padel
- Occlusions plus fréquentes → V4 avec motion attention utile
- Peut nécessiter du fine-tuning spécifique padel

---

## 📚 Références

### Papers
1. TrackNetV4: Raj et al., "Enhancing Fast Sports Object Tracking with Motion Attention Maps", ICASSP 2025
2. TrackNetV3: Chen & Wang, "Enhancing Shuttlecock Tracking with Augmentations and Trajectory Rectification", MMAsia 2023
3. TrackNetV2: Sun et al., "Efficient Shuttlecock Tracking Network", ICPAI 2020
4. TrackNetV1: Huang et al., "A Deep Learning Network for Tracking High-speed and Tiny Objects", KDD 2019

### Liens utiles
- Paper V4: https://arxiv.org/abs/2409.14543
- Repo V4: https://github.com/AnInsomniacy/tracknet-series-pytorch
- Dataset Padel: https://zenodo.org/records/14653706
- Site officiel V4: https://time.griffith.edu.au/paper-sites/tracknet-v4/

---

## ⚠️ Points d'attention

1. **Pas de weights V4 officiels** - Il faudra soit:
   - Entraîner from scratch sur PadelTracker100
   - Utiliser weights V2 + fine-tuner avec le Motion Prompt Layer

2. ~~**Format annotations**~~ ✅ Résolu - Script `preprocessing/convert_padeltracker.py` créé

3. **Détection des rebonds** - Non inclus dans TrackNet de base, mais possible via:
   - Analyse de trajectoire (changement de direction)
   - Classification temporelle (LSTM/Transformer sur positions)
   - Exploitation des motion attention maps

4. **GPU recommandé** - L'entraînement sur 100k frames nécessite un bon GPU (au moins 8GB VRAM)

---

*Résumé généré le 20 décembre 2025*
