"""Extraction de frames depuis une vidéo via ffmpeg."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR, FRAMES_HR_DIR


def extract_frames(video_path: Path, output_dir: Path, fps: int = 1) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "frame_%05d.png"
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "1",
        str(pattern),
        "-y",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error:\n{result.stderr}")
    frames = list(output_dir.glob("frame_*.png"))
    print(f"[extract_frames] {len(frames)} frames extraites dans {output_dir}")
    return len(frames)


if __name__ == "__main__":
    for video in RAW_DIR.glob("*.mp4"):
        out = FRAMES_HR_DIR / video.stem
        extract_frames(video, out, fps=1)
