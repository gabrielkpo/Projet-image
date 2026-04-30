from pathlib import Path

# Chemins
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FRAMES_HR_DIR = DATA_DIR / "frames_hr"
FRAMES_LR_DIR = DATA_DIR / "frames_lr"
RESULTS_DIR = ROOT / "benchmark" / "results"

# Upscaling
SCALE_FACTOR = 4          # facteur d'upscaling (2, 3 ou 4)
PATCH_SIZE = 33           # taille des patches pour l'entraînement
STRIDE = 14

# Dégradation
BLUR_KERNEL_SIZE = 3
NOISE_STD = 10            # écart-type du bruit gaussien (0 = pas de bruit)

# Entraînement
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 50
DEVICE = "cpu"            # "cpu" ou "cuda"

# CodeCarbon
COUNTRY_ISO = "FRA"
