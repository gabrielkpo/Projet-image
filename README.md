# Green Video-Resolution

> Analyse et mise en place de différentes méthodes d'optimisation pour le streaming mobile.

**Projet de Recherche IATI — Intelligence Artificielle & Traitement de l'Image**  
Étudiants : Gabriel KPODOH & Thomas Delacretaz-Martinon  
Superviseur : Thomas Oberlin | Année : 2026-2027

---

## Contexte & Motivation

Le streaming vidéo représente une part croissante de la consommation énergétique mondiale. La demande électrique des data centers devrait **doubler entre 2022 et 2026** (460 TWh → 1 000 TWh, source AIE).

Ce projet explore un **changement de paradigme** :

| Modèle actuel ("Red AI") | Modèle cible ("Green AI") |
|---|---|
| Envoyer 100 % des pixels via le réseau | Envoyer 25 % des pixels, reconstruire le reste |
| Coût ∝ débit (Bitrate) | Coût ∝ complexité du modèle (investissement unique) |
| Surcharge Data Centers & 5G (Scope 2) | Sollicite le NPU du terminal (Scope 3 aval) |

**L'équation fondamentale du projet :**

```
E_transport(HD) > E_transport(SD) + E_inférence(Mobile)
```

Le projet est viable si et seulement si l'énergie économisée sur les infrastructures réseau est **supérieure** au surplus consommé par le NPU/GPU pour l'upscaling.

---

## Architecture du projet

```
green_sr/
├── data/
│   ├── raw/                  # vidéos sources originales
│   ├── frames_hr/            # frames extraites haute résolution
│   └── frames_lr/            # frames dégradées (générées, non versionnées)
│
├── pipeline/
│   ├── extract_frames.py     # ffmpeg wrapper : vidéo → frames PNG
│   ├── degrade.py            # HR → LR (bicubique + bruit gaussien)
│   └── metrics.py            # PSNR, SSIM, Tinf, Einf (CodeCarbon)
│
├── methods/
│   ├── level1_interpolation.py   # bicubique, bilinéaire, Lanczos
│   ├── level2_frequency.py       # DWT, FFT zero-padding
│   ├── level3_denoising.py       # bilatéral, NLM, BM3D
│   ├── level4_ml_classic.py      # sparse coding, random forest, PCA
│   ├── level5_srcnn.py           # SRCNN (architecture + inférence)
│   └── level6_cnn_optimized.py   # ESPCN, FSRCNN
│
├── train/
│   ├── train_srcnn.py
│   ├── train_espcn.py
│   └── train_fsrcnn.py
│
├── benchmark/
│   ├── run_all.py            # lance tous les algos sur le même jeu de frames
│   └── results/              # CSV + figures (générés automatiquement)
│
├── config.py                 # facteur d'upscaling, chemins, hyperparamètres
└── requirements.txt
```

---

## Les 6 niveaux d'algorithmes

Le projet construit une **progression rigoureuse** du plus simple au plus complexe, permettant de quantifier le gain de chaque famille de méthodes en qualité (PSNR, SSIM) et en coût énergétique (E_inf, T_inf).

### Niveau 1 — Baseline : Interpolation classique
Méthodes de référence sans apprentissage. Reconstruction polynomiale des pixels manquants.
- **Bilinéaire** (ordre 1)
- **Bicubique** (ordre 3) — étalon de comparaison
- **Lanczos** (noyau sinusoïdal fenêtré)

### Niveau 2 — Domaine fréquentiel : Ondelettes & FFT
- **DWT** — décomposition en sous-bandes (LL, LH, HL, HH) + reconstruction IDWT
- **FFT zero-padding** — extension du spectre aux hautes fréquences

### Niveau 3 — Débruitage & Restauration
Utilisés comme post-traitement après upscaling pour atténuer les artefacts.
- **Filtre bilatéral** — lissage non-linéaire préservant les contours
- **NLM** (Non-Local Means) — moyenne pondérée par similarité de patches
- **BM3D** — état de l'art non-ML, transformée de Wiener sur blocs 3D

### Niveau 4 — ML classique (sans réseaux profonds)
Apprentissage supervisé sur paires (LR, HR).
- **Sparse coding** — dictionnaire de patches LR→HR
- **Random Forest** — régression par descripteurs locaux
- **PCA + régression linéaire** — espace latent compact

### Niveau 5 — SRCNN (Dong et al., 2014)
Premier CNN appliqué à la super-résolution. Architecture 3 couches (~57k paramètres).
```
F1(I) = ReLU(W1 * I + B1)   → extraction de patches (9×9)
F2(I) = ReLU(W2 * F1 + B2)  → mapping non-linéaire (1×1)
F3(I) = W3 * F2 + B3        → reconstruction (5×5)
```

### Niveau 6 — CNN optimisés : ESPCN & FSRCNN
Objectif final du projet — optimisés pour l'inférence temps réel (T_inf < 33 ms).

**ESPCN** (Shi et al., 2016) : toutes les convolutions dans l'espace LR, upscaling par *pixel shuffle* en toute fin de réseau → complexité réduite d'un facteur r².

**FSRCNN** (Dong et al., 2016) : structure en sablier (extraction → réduction → mapping → expansion → déconvolution), sans interpolation bicubique en entrée.

---

## Stratégie d'optimisation "Green"

Au-delà de l'architecture, deux techniques de compression sont appliquées :

1. **Quantification Post-Entraînement (PTQ)** : conversion FP32 → INT8, divise par 4 la taille du modèle, accélère l'inférence sur ARM.
2. **Pruning (Élagage)** : suppression des neurones proches de zéro, allège le calcul matriciel.

**Piste innovante — Upscaling Sélectif :**  
Une carte de saillance détecte les zones texturées (cheveux, herbe). L'IA ne s'active que sur ces zones (Heavy Compute) et laisse les zones plates à une interpolation classique (Low Compute).

---

## Métriques d'évaluation

| Métrique | Description | Contrainte |
|---|---|---|
| **PSNR** (dB) | Peak Signal-to-Noise Ratio | > bicubique + 1 dB |
| **SSIM** | Structural Similarity Index | Perceptibilité humaine |
| **T_inf** (ms/frame) | Temps d'inférence | < 33 ms (30 FPS) |
| **E_inf** (J/frame) | Énergie consommée (CodeCarbon) | < E_transport évité |

---

## Installation & Utilisation

```bash
pip install -r requirements.txt
```

### 1. Préparer les données
```bash
# Placer les vidéos dans data/raw/, puis :
python pipeline/extract_frames.py   # extraction des frames HR
python pipeline/degrade.py          # génération des frames LR
```

### 2. Lancer le benchmark (méthodes classiques)
```bash
python benchmark/run_all.py
# → benchmark/results/benchmark.csv
# → benchmark/results/benchmark.png
```

### 3. Entraîner les réseaux
```bash
python train/train_srcnn.py
python train/train_espcn.py
python train/train_fsrcnn.py
```

### 4. Configurer
Tous les hyperparamètres sont centralisés dans [config.py](config.py) :
```python
SCALE_FACTOR = 4      # facteur d'upscaling
NOISE_STD    = 10     # bruit gaussien ajouté à la dégradation
EPOCHS       = 50
DEVICE       = "cpu"  # "cuda" si GPU disponible
```

---

## Dataset

[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) — référence académique pour la super-résolution (800 images d'entraînement, résolution 2K).  
Dégradation : bicubique ×4 + bruit gaussien optionnel.

**Data augmentation frugale** : rotations (90°, 180°) + retournements horizontaux pour maximiser l'apprentissage sur un volume restreint (sobriété carbone).

---

## Références

- Dong et al., *Learning a Deep Convolutional Network for Image Super-Resolution* (SRCNN, 2014)
- Shi et al., *Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel CNN* (ESPCN, 2016)
- Dong et al., *Accelerating the Super-Resolution CNN* (FSRCNN, 2016)
- AI FOR GREEN & GREEN AI — Institut G9+, Numeum, Cigref (2024)
