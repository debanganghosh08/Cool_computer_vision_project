"""Reading frames in, writing frames out, and putting the sound back."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Source:
    """A video file or a webcam, behind one small interface."""

    cap: cv2.VideoCapture
    fps: float
    width: int
    height: int
    frames: int  # 0 when unknown (webcam)
    is_camera: bool
    path: Path | None

    @classmethod
    def open(cls, spec: str) -> "Source":
        is_camera = spec.isdigit()
        if is_camera:
            index = int(spec)
            # CAP_DSHOW starts far faster than the default backend on Windows.
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            cap = cv2.VideoCapture(index, backend)
            path = None
        else:
            path = Path(spec)
            if not path.exists():
                raise FileNotFoundError(f"Input video not found: {path}")
            cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open {spec!r}. "
                + ("Is another app using the camera?" if is_camera else "Is it a video file?")
            )
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 1.0 or fps > 240.0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = 0 if is_camera else max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        return cls(cap, fps, width, height, frames, is_camera, path)

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self) -> None:
        self.cap.release()


def make_writer(path: Path, fps: float, size: tuple[int, int], gray: bool = False) -> cv2.VideoWriter:
    """size is (width, height). mp4v is the codec every OpenCV build can write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size, isColor=not gray)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open {path} for writing.")
    return writer


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def mux_audio(video: Path, audio_from: Path, out: Path) -> bool:
    """Copy the sound of the original clip onto the rendered one.

    OpenCV writes picture only, so without this step the export is silent.
    Returns False (and leaves the silent file in place) when ffmpeg is missing
    or the source has no audio track.
    """
    if not has_ffmpeg():
        return False
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", str(video),
        "-i", str(audio_from),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(out),
    ]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0 and out.exists() and out.stat().st_size > 0
