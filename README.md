# Deep Learning J3 — CNN et classification d'images (cats vs dogs)

Trois itérations sur le même problème de classification binaire d'images,
dans un seul notebook : [`tp_jour3_cnn_cats_vs_dogs.ipynb`](tp_jour3_cnn_cats_vs_dogs.ipynb).

1. **TP1 — CNN from scratch** : 3 blocs Conv2D/MaxPooling, overfitting assumé
2. **TP2 — Augmentation + Dropout** : RandomFlip/Rotation/Zoom + Dropout(0.4)
3. **TP3 — Transfer learning MobileNetV2** : base ImageNet gelée puis fine-tuning
   des 20% dernières couches

## Résultats

| Itération | val_acc | Params | Temps | Taille |
|---|---|---|---|---|
| CNN scratch | 73.2% | 2 452 801 | 1537s | 28.1 Mo |
| CNN augmenté + Dropout | 76.7% | 2 452 801 | 400s | 28.1 Mo |
| MobileNetV2 fine-tuning | **98.8%** | 2 422 081 | 649s | 23.4 Mo |

La cible du cours (95% avec transfer learning) est dépassée. Le modèle le plus
léger est le plus performant : la différence n'est pas dans le code, elle est
dans les représentations (ImageNet, 14M d'images vues avant d'arriver).

**Export TFLite** : 23.4 Mo (.keras) → 9.1 Mo (FP32, 2.6x) → **2.8 Mo (INT8)**
avec une accuracy INT8 de 0.988 = identique au FP32 : la quantization est
gratuite sur ce modèle.

## Dataset et adaptation locale

Le TP prévoit Colab + API Kaggle. Travail fait en local (CPU uniquement), donc :

- zip **officiel Microsoft** du dataset Cats vs Dogs (787 Mo, 25 000 images,
  téléchargement direct sans compte Kaggle) ;
- filtrage des JPEG corrompus (défaut connu de ce dataset : ~1 800 fichiers
  sans en-tête JFIF valide) ;
- sous-échantillonnage à 1 500 images/classe (critère « ≥ 1 000 images par
  classe » respecté), split 80/20 seed=42 → 1 200 train / 300 val par classe ;
- `IMG_SIZE=(96, 96)` en TP1/TP2 (l'énoncé autorise 64-160), `(160, 160)` pour
  MobileNetV2.

## Contenu du repo

- `tp_jour3_cnn_cats_vs_dogs.ipynb` — le notebook exécuté (sorties réelles)
- `curves_*.png`, `comparison_*.png`, `augmentation_grid.png` — les courbes
- `model_tl.keras` — le meilleur modèle (MobileNetV2 fine-tuné, 98.8%)
- `cat_vs_dog_mobilenet.tflite` / `_int8.tflite` — les exports mobiles
- `logs/` — runs TensorBoard des 3 entraînements (`tensorboard --logdir logs`)

## Observations honnêtes

- L'EarlyStopping (patience=5) a coupé TP1 : la val_loss remontait dès
  l'epoch ~5 pendant que la train_accuracy filait vers 91% — l'overfitting
  que la phase 1.4 voulait faire observer.
- TP2 : le gap train/val passe de ~18 points à ~3 points. L'augmentation
  ralentit la convergence (chaque epoch voit des images différentes) mais
  généralise mieux — il aurait fallu plus de 20 epochs pour saturer.
- Sur une image hors distribution (ni chat ni chien), le modèle binaire force
  quand même une prédiction confiante : aucun mécanisme « je ne sais pas »,
  limite architecturale à connaître avant toute mise en prod.
- Référence J2 au passage : le PMC plafonnait à 98.06% sur digits (images 8x8
  simples) ; ici un CNN transfer-learné atteint 98.8% sur de vraies photos
  400x300 — c'est toute la différence convolutions + pré-entraînement.
