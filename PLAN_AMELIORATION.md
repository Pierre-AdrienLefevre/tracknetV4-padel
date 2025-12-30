# Plan d'amélioration TrackNetV4 - Padel

## Résultats actuels (exp_20251225_223042)

| Métrique | Valeur actuelle | Objectif (paper) |
|----------|-----------------|------------------|
| F1 Score | 92.53% | 97.5% |
| Accuracy | 87.81% | 95%+ |
| Precision | 94.72% | 97%+ |
| Recall | 90.44% | 97%+ |

**Test vidéo**: Fonctionne bien sur `2022_BCN_FinalF_1_sample_predict.mp4`

---

## Phase 1: Data Augmentation

### Objectif
Améliorer la généralisation du modèle en augmentant artificiellement la diversité des données.

### Augmentations à implémenter

| Augmentation | Paramètres | Impact attendu |
|--------------|------------|----------------|
| Horizontal Flip | p=0.5 | +1-2% F1 |
| Brightness | ±20% | Robustesse éclairage |
| Contrast | ±20% | Robustesse conditions |
| Gaussian Noise | σ=0.01-0.02 | Robustesse bruit |
| Color Jitter | Saturation ±10% | Robustesse couleurs |

### Points d'attention
- **Ne PAS faire de rotation** (la gravité compte pour la trajectoire)
- **Appliquer les mêmes transformations** aux 3 frames ET aux 3 heatmaps
- **Flip horizontal**: inverser aussi les coordonnées x des heatmaps

### Fichiers à modifier
- `data/dataset.py` - Ajouter les transformations dans le DataLoader
- `config.yaml` - Ajouter section `augmentation:`

---

## Phase 2: Entraînement prolongé

### Configuration recommandée

```yaml
train:
  epochs: 100          # Actuellement 30
  batch: 16            # OK avec 16GB VRAM
  optimizer: Adam      # Tester vs Adadelta
  lr: 0.0001           # Learning rate plus stable
  scheduler: CosineAnnealingLR  # Meilleur que ReduceLROnPlateau pour long training
```

### Stratégie
1. Partir du `best_model.pth` actuel (transfer learning)
2. Réduire le LR initial (fine-tuning)
3. Ajouter early stopping (patience=10-15 epochs)

---

## Phase 3: Weighted Loss (optionnel)

### Problème
La balle occupe ~0.01% de l'image → déséquilibre de classes massif

### Solution
Augmenter le poids des pixels positifs dans la BCE Loss:

```python
# Poids suggéré: ratio négatifs/positifs
pos_weight = (512 * 288) / (pi * 3^2)  # ~5000
# En pratique, utiliser pos_weight entre 10 et 100
```

---

## Phase 4: Données supplémentaires (si nécessaire)

### Options
1. **Annoter plus de vidéos padel** (chronophage mais efficace)
2. **Dataset Tennis original** (~17k frames, même format)
3. **Synthetic data** (générer des trajectoires artificielles)

### Priorité
Attendre les résultats des phases 1-2 avant d'investir dans plus de données.

---

## Métriques de suivi

### Checkpoints de validation

| Phase | F1 cible | Action si non atteint |
|-------|----------|----------------------|
| Post-Aug | 94%+ | Ajuster augmentations |
| 50 epochs | 95%+ | Ajuster LR/optimizer |
| 100 epochs | 96%+ | Weighted loss |
| Final | 97%+ | Plus de données |

### Commandes utiles

```bash
# Reprendre l'entraînement depuis le best model
uv run python train.py --config config.yaml

# Tester sur une vidéo
uv run python predict/streem_video_predict.py \
  --model outputs/exp_20251225_223042/checkpoints/best_model.pth \
  --input dataset/2022_BCN_FinalF_1_sample.mp4 \
  --output predictions/

# Monitoring
# https://wandb.ai/pierreadrienlefevre-ecole-de-technologie-superieure/tracknet-padel
```

---

## Prochaine étape immédiate

**Implémenter les augmentations de données** dans `data/dataset.py`:
1. Ajouter les imports (torchvision.transforms, albumentations)
2. Créer une classe `TrackNetAugmentation`
3. Appliquer aux inputs ET heatmaps de manière synchronisée
4. Tester sur quelques samples avant de relancer l'entraînement

---

*Créé le 26 décembre 2025*