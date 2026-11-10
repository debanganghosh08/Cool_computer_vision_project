"""Hand tracking on top of the MediaPipe Tasks HandLandmarker.

One job: give the rest of the program two stable points per frame, the centre of
the left palm and the centre of the right palm, in pixels.

Why the palm centre and not the wrist: the wrist landmark sits at the very edge
of the hand and swings a long way whenever the forearm rolls, so a portal pinned
to it slides around. The mean of the wrist and the four finger knuckles sits in
the middle of the palm and barely moves when the hand rotates.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_NAME = "hand_landmarker.task"

# Wrist + the knuckle of every finger. Their mean is the palm centre.
PALM_POINTS = (0, 5, 9, 13, 17)


def default_model_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "models"


def ensure_model(path: str | os.PathLike[str] | None = None, quiet: bool = False) -> Path:
    """Return a local path to hand_landmarker.task, downloading it once if needed."""
    target = Path(path) if path else default_model_dir() / MODEL_NAME
    if target.is_dir():
        target = target / MODEL_NAME
    if target.exists() and target.stat().st_size > 1_000_000:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(f"[handportal] downloading hand landmark model -> {target}", file=sys.stderr)
    tmp = target.with_suffix(".part")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(target)
    except Exception as exc:  # noqa: BLE001 - the message matters more than the type
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the hand landmark model from {MODEL_URL}\n"
            f"Reason: {exc}\n"
            f"Download it in a browser and save it as: {target}"
        ) from exc
    if not quiet:
        print(f"[handportal] model ready ({target.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)
    return target


@dataclass(frozen=True)
class Hand:
    """One detected hand in pixel coordinates."""

    points: np.ndarray  # (21, 2) float32, pixels
    palm: tuple[float, float]
    score: float

    @property
    def span(self) -> float:
        """Rough hand size in pixels - wrist to middle fingertip."""
        return float(np.linalg.norm(self.points[12] - self.points[0]))


@dataclass(frozen=True)
class Frame:
    """What the tracker saw in one frame."""

    left: Hand | None  # the hand further left on screen
    right: Hand | None  # the hand further right on screen

    @property
    def both(self) -> bool:
        return self.left is not None and self.right is not None

    @property
    def hands(self) -> list[Hand]:
        return [h for h in (self.left, self.right) if h is not None]


class HandTracker:
    """Thin, boring wrapper. Everything MediaPipe-shaped stays inside this class."""

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        *,
        video_mode: bool = True,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        quiet: bool = False,
    ) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision
        except ImportError as exc:  # pragma: no cover - import guard
            raise SystemExit(
                "MediaPipe is not installed in this interpreter.\n"
                "  pip install -r requirements.txt\n"
                "See docs/INSTALL.md if that fails."
            ) from exc

        self._mp = mp
        self._model_path = ensure_model(model_path, quiet=quiet)
        self._running_mode = vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self._model_path)),
            running_mode=self._running_mode,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._closed = False

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._landmarker.close()
            self._closed = True

    # -- the only method that matters -------------------------------------
    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> Frame:
        """Run one BGR frame (as OpenCV gives it) through the landmarker."""
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self._running_mode.name == "VIDEO":
            result = self._landmarker.detect_for_video(image, int(timestamp_ms))
        else:
            result = self._landmarker.detect(image)

        h, w = frame_bgr.shape[:2]
        found: list[Hand] = []
        for i, landmarks in enumerate(result.hand_landmarks or []):
            pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
            palm = pts[list(PALM_POINTS)].mean(axis=0)
            score = 1.0
            if result.handedness and i < len(result.handedness) and result.handedness[i]:
                score = float(result.handedness[i][0].score)
            found.append(Hand(points=pts, palm=(float(palm[0]), float(palm[1])), score=score))

        if not found:
            return Frame(left=None, right=None)
        if len(found) == 1:
            # One hand only. Call it left or right by which half of the frame it is in,
            # so the caller can keep holding the other side's last known position.
            only = found[0]
            return Frame(left=only, right=None) if only.palm[0] < w / 2 else Frame(left=None, right=only)

        # Sort by screen x, not by MediaPipe's handedness label. Handedness flips
        # whenever the camera image is mirrored, and a portal that swaps ends
        # mid-shot looks broken. Screen position never lies.
        found.sort(key=lambda hand: hand.palm[0])
        return Frame(left=found[0], right=found[-1])
