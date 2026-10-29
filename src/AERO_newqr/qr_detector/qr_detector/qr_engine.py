"""Rotation-invariant QR engine (section 5 of the original script, algorithm unchanged), ROS-free.

detect             largest AprilTag (tag36h11, pupil_apriltags); its id is returned as the text
detect_all         every AprilTag in view (tag36h11), ids as text
extract_qr_features warp the QR quad to 210x210, Otsu, sample a 21x21 module grid
compute_similarity  fraction of agreeing modules, best of the 4 rotations
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np

from pupil_apriltags import Detector as AprilTagDetector

from bc_common.geometry import order_corners

try:
    from pyzbar.pyzbar import decode as _pyzbar_decode
except Exception:  # noqa: BLE001 - pyzbar (and libzbar) are optional
    _pyzbar_decode = None


class QrFinder:
    def __init__(self):
        self.detector = cv2.QRCodeDetector()
        self.apriltag = AprilTagDetector(families="tag36h11", quad_decimate=1.5, refine_edges=True,
                                         decode_sharpening=0.25)
        self.pyzbar_decode = _pyzbar_decode

    def detect(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], np.ndarray, str]:
        """Returns (found, quad points (1, 4, 2) float32, gray image, tag id as text or '').

        AprilTag (tag36h11) detection; with several tags in view the largest one is returned."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        tags = self.apriltag.detect(gray)
        if not tags:
            return False, None, gray, ""
        tag = max(tags, key=lambda t: cv2.contourArea(np.asarray(t.corners, dtype=np.float32)))
        pts = np.asarray(tag.corners, dtype=np.float32).reshape(1, 4, 2)
        return True, pts, gray, str(tag.tag_id)

    def detect_all(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple[np.ndarray, str]]]:
        """Every AprilTag in view as ((4, 2) corners, tag id as text), each id paired with its own corners."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        found = [(np.asarray(t.corners, dtype=np.float32).reshape(-1, 2)[:4], str(t.tag_id))
                 for t in self.apriltag.detect(gray)]
        return gray, found


def extract_qr_features(gray: np.ndarray, quad_pts, grid_size: int = 21) -> Tuple[np.ndarray, np.ndarray]:
    quad = order_corners(quad_pts)
    side = grid_size * 10
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype=np.float32)
    warped = cv2.warpPerspective(gray, cv2.getPerspectiveTransform(quad, dst), (side, side))
    _, binary = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    for r in range(grid_size):
        for c in range(grid_size):
            grid[r, c] = 1 if np.mean(binary[r * 10 + 2:r * 10 + 8, c * 10 + 2:c * 10 + 8]) < 128 else 0
    return grid, warped


def compute_similarity(grid_ref: Optional[np.ndarray], grid_cand: Optional[np.ndarray]) -> float:
    if grid_ref is None or grid_cand is None:
        return 0.0
    best = 0.0
    cand = grid_cand.copy()
    for _ in range(4):
        best = max(best, float(np.sum(grid_ref == cand)) / float(grid_ref.size))
        cand = np.rot90(cand)
    return best


def match(ref_grid, ref_text, cand_grid, cand_text, threshold, reject_text_mismatch=True) -> Tuple[float, bool]:
    """(score, matched).

    Exact decoded-text match: score 1.0, matched. Otherwise the 4-rotation grid similarity is compared
    with `threshold`, as in the original script. With reject_text_mismatch (default), two codes that
    both decode to DIFFERENT texts never match: QR codes of the same version share their finder and
    timing patterns, so unrelated codes can score > 0.9 on the grid alone."""
    if cand_text and ref_text and cand_text == ref_text:
        return 1.0, True
    score = compute_similarity(ref_grid, cand_grid)
    if reject_text_mismatch and cand_text and ref_text:
        return score, False
    return score, score >= threshold
