from pathlib import Path

# Chemins
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FRAMES_HR_DIR = DATA_DIR / "frames_hr"
FRAMES_LR_DIR        = DATA_DIR / "frames_lr"
FRAMES_LR_PHASE1_DIR = FRAMES_LR_DIR / "phase1_bicubic"   # dégradation pure, sans bruit
FRAMES_LR_PHASE2_DIR = FRAMES_LR_DIR / "phase2_noisy"     # bicubique + AWGN σ=25 + flou
RESULTS_DIR = ROOT / "benchmark" / "results"

# Upscaling
SCALE_FACTOR = 4          # facteur par défaut
VIDEO_SCALE = {
    "DGbwtVtthu8": 8,     # source 4K → LR ×8
    "33mqqm4QlJ8": 4,     # source 1080p → LR ×4
}
PATCH_SIZE = 33           # taille des patches pour l'entraînement
STRIDE = 14

# Dégradation
BLUR_KERNEL_SIZE = 3
NOISE_STD_PHASE2 = 25     # AWGN σ=25 (phase 2 uniquement)

# Entraînement
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 50
DEVICE = "cpu"            # "cpu" ou "cuda"

# CodeCarbon
COUNTRY_ISO = "FRA"

# Vidéo source
VIDEO_URL1   = "https://www.youtube.com/watch?v=DGbwtVtthu8"
VIDEO_URL2  = "https://www.youtube.com/watch?v=33mqqm4QlJ8"
EXTRACT_FPS = 1   # 1 frame/seconde (0 = FPS natif de la vidéo)

