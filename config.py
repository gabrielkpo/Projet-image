from pathlib import Path

ROOT     = Path(__file__).parent

# ── Données brutes (vidéos source) ───────────────────────────────────────────
DATA_DIR  = ROOT / "data"
RAW_DIR   = DATA_DIR / "raw"
DIV2K_DIR = DATA_DIR / "div2k"      # → placer DIV2K_train_HR/ ici

# ── Benchmark sparse (1 fps, toutes durées) ── utilisé par benchmark/run_all.py
FRAMES_HR_DIR        = DATA_DIR / "frames_hr"
FRAMES_LR_DIR        = DATA_DIR / "frames_lr"
FRAMES_LR_PHASE1_DIR = FRAMES_LR_DIR / "phase1_bicubic"
FRAMES_LR_PHASE2_DIR = FRAMES_LR_DIR / "phase2_noisy"

# ── Runs (clips 5s à FPS natif) ── structure par vidéo
#    runs/<video_id>/frames_hr/
#    runs/<video_id>/frames_lr/
#    runs/<video_id>/frames_sr/<method>/
#    runs/<video_id>/videos/
RUNS_DIR = ROOT / "runs"

# ── Résultats (benchmark + visuels) ─────────────────────────────────────────
RESULTS_DIR         = ROOT / "results"
RESULTS_VISUALS_DIR = RESULTS_DIR / "visuals"

# ── Facteurs d'upscaling ─────────────────────────────────────────────────────
SCALE_FACTOR = 4
VIDEO_SCALE  = {
    "DGbwtVtthu8": 8,   # source 4K → LR ×8
    "33mqqm4QlJ8": 4,   # source 1080p → LR ×4
}

# ── Patches pour l'entraînement ──────────────────────────────────────────────
PATCH_SIZE = 33
STRIDE     = 14

# ── Dégradation ──────────────────────────────────────────────────────────────
BLUR_KERNEL_SIZE = 3
NOISE_STD_PHASE2 = 25      # AWGN σ=25 (phase 2 uniquement)

# ── Entraînement CNN ─────────────────────────────────────────────────────────
BATCH_SIZE    = 64
LEARNING_RATE = 1e-4
EPOCHS        = 50
DEVICE        = "cpu"       # "cpu" ou "cuda"

# ── CodeCarbon ───────────────────────────────────────────────────────────────
COUNTRY_ISO = "FRA"

# ── Vidéos source ────────────────────────────────────────────────────────────
VIDEO_URL1   = "https://www.youtube.com/watch?v=DGbwtVtthu8"
VIDEO_URL2   = "https://www.youtube.com/watch?v=33mqqm4QlJ8"
EXTRACT_FPS  = 1            # 1 frame/seconde pour le benchmark sparse
CLIP_DURATION = 5           # secondes extraites à FPS natif pour les runs
