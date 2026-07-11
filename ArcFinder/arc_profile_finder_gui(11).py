#!/usr/bin/env python3
"""
Arc Profile Finder GUI

Purpose:
    A 2D tool for extracting a mirrored top-view profile from a photo and fitting
    circular arcs to one side of that profile.

This version is tuned for bike seatpost top-view openings:
    - It looks for the dark inner opening, not the full background.
    - It tries multiple strict dark thresholds.
    - It rejects components that are too close to image borders.
    - It prefers dark, compact, centered, elongated components.
    - It uses the detected dark opening to find the symmetry axis.
    - It rotates the opening vertical and assumes the wider end should be up.
    - It can auto-extract the one-side profile and mirror it.
    - It explicitly closes the mirrored profile at the top and bottom axis points.
    - It uses endpoint-constrained arc fitting so adjacent arcs meet cleanly.
    - It uses denser point spacing near the top and bottom tips.
    - It can target either the inner dark opening or the outer outside edge.
    - In outer-edge mode, it estimates independent outer top/bottom closure points.
    - The user can lock/manual-edit the top and bottom closure points.
    - The user can tune outer-edge sensitivity, search distance, and smoothing per image.
    - Controls are split into tabs for smaller screens.
    - Quick Instructions can be collapsed to free screen space.
    - Tuning sliders are stacked vertically so they do not run off-screen.
    - Manual Editing controls are split across two rows.

Important limitation:
    The app cannot know real-world scale from a photo alone unless the image includes
    a known reference dimension, ruler, or you manually set scale. Without scale,
    results are in pixels. After setting scale, click Fit Arcs again.

Install:
    python -m pip install numpy matplotlib pillow

Run:
    python arc_profile_finder_gui.py

Recommended workflow:
    1. Load Image.
       If "Auto process on load" is checked, the app will attempt the whole process.
    2. If the image is upside-down, click Flip Up Direction.
    3. If needed, set scale.
    4. Review/edit profile points and breakpoints.
    5. Fit Arcs / Export CSV.

Controls:
    Load Image:
        Opens an image. With auto-process enabled, runs the full automatic pipeline.

    Auto Process All:
        Runs orientation, axis detection, profile extraction, breakpoint detection, and arc fitting.
        If Lock manual closures is enabled, the current axis endpoints are treated as final.

    Auto Orient Image + Axis:
        Finds the dark inner opening, rotates it vertical, and sets the symmetry axis.

    Auto Extract Profile:
        Uses the selected Profile target to create one mirrored side profile.
        Inner dark opening traces the black opening.
        Outer outside edge traces the outside chrome/body profile.

    Auto Break + Fit:
        Automatically picks breakpoints and fits circular arcs.

    Flip Up Direction:
        Rotates the current image 180 degrees and updates the annotations.

    Set Scale:
        Click two points with a known distance, then enter the real distance.

Notes:
    - This detector is meant for the INNER dark opening.
    - If you need the OUTER silver rim profile instead, that is a different detector.
    - The manual tools are still available because glare, shadows, and perspective can confuse
      automatic detection.
"""

import csv
import math
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from PIL import Image

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


@dataclass
class ArcResult:
    segment_number: int
    start_index: int
    end_index: int
    point_count: int
    center_x: float
    center_y: float
    mirrored_center_x: float
    mirrored_center_y: float
    radius: float
    theta_start_rad: float
    theta_end_rad: float
    included_angle_degrees: float
    direction: str
    arc_length: float
    chord_length: float
    rms_error: float
    max_error: float
    start_x: float
    start_y: float
    end_x: float
    end_y: float


@dataclass
class AutoOrientationReport:
    rotation_degrees: float = 0.0
    used_180_flip: bool = False
    detected_area_px: int = 0
    top_width_px: float = 0.0
    bottom_width_px: float = 0.0
    axis_top_px: Optional[np.ndarray] = None
    axis_bottom_px: Optional[np.ndarray] = None
    threshold_value: float = 0.0
    component_score: float = 0.0
    component_centroid_px: Optional[np.ndarray] = None
    component_bbox_px: Optional[Tuple[float, float, float, float]] = None
    detector_note: str = ""

    def text(self) -> str:
        lines = [
            "Auto Image Orientation",
            "=" * 32,
            f"Rotation applied: {self.rotation_degrees:.3f}°",
            f"Used 180° up flip: {self.used_180_flip}",
            f"Detected dark area: {self.detected_area_px} px",
            f"Top width estimate: {self.top_width_px:.2f} px",
            f"Bottom width estimate: {self.bottom_width_px:.2f} px",
            f"Threshold value: {self.threshold_value:.2f}",
            f"Component score: {self.component_score:.3f}",
        ]

        if self.component_centroid_px is not None:
            lines.append(f"Component centroid px: {format_vector_2d(self.component_centroid_px)}")

        if self.component_bbox_px is not None:
            x0, y0, x1, y1 = self.component_bbox_px
            lines.append(f"Component bbox px: [{x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}]")

        if self.axis_top_px is not None and self.axis_bottom_px is not None:
            lines.append(f"Axis top px: {format_vector_2d(self.axis_top_px)}")
            lines.append(f"Axis bottom px: {format_vector_2d(self.axis_bottom_px)}")

        if self.detector_note:
            lines.append(f"Detector note: {self.detector_note}")

        return "\n".join(lines)


@dataclass
class AutoProfileReport:
    target: str = "Inner dark opening"
    point_count: int = 0
    locked_manual_closures: bool = False
    edge_sensitivity: float = 0.28
    outer_search_scale: float = 1.0
    smooth_window: int = 5
    y_min: float = 0.0
    y_max: float = 0.0
    max_half_width: float = 0.0
    extraction_bins: int = 0

    def text(self) -> str:
        return "\n".join(
            [
                "Auto Profile Extraction",
                "=" * 32,
                f"Target: {self.target}",
                f"Profile points: {self.point_count}",
                f"Locked manual closures: {self.locked_manual_closures}",
                f"Outer edge sensitivity: {self.edge_sensitivity:.3f}",
                f"Outer search scale: {self.outer_search_scale:.3f}",
                f"Smooth window: {self.smooth_window}",
                f"Y min: {self.y_min:.3f}",
                f"Y max: {self.y_max:.3f}",
                f"Max half-width: {self.max_half_width:.3f}",
                f"Bins requested: {self.extraction_bins}",
                "Ends: closed to symmetry axis if enabled",
                "Spacing: cosine-dense near top/bottom tips",
            ]
        )


@dataclass
class AutoBreakpointReport:
    breakpoint_indices: List[int]
    tolerance: float
    max_arcs: int
    min_points_per_arc: int

    def text(self) -> str:
        return "\n".join(
            [
                "Auto Breakpoint Detection",
                "=" * 32,
                f"Breakpoints: {self.breakpoint_indices}",
                f"Tolerance: {self.tolerance:.3f}",
                f"Max arcs: {self.max_arcs}",
                f"Min points per arc: {self.min_points_per_arc}",
            ]
        )


def format_vector_2d(v) -> str:
    if v is None:
        return "n/a"

    return f"[{float(v[0]):.3f}, {float(v[1]):.3f}]"


def normalize(vector: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)

    if length < 1e-12:
        raise ValueError("Cannot normalize a near-zero vector.")

    return vector / length


def pil_to_display_array(image_pil: Image.Image) -> np.ndarray:
    return np.asarray(image_pil.convert("RGB"))


def image_to_gray_array(image_pil: Image.Image) -> np.ndarray:
    image_rgb = image_pil.convert("RGB")
    arr = np.asarray(image_rgb).astype(np.float64)

    gray = (
        0.2126 * arr[:, :, 0]
        + 0.7152 * arr[:, :, 1]
        + 0.0722 * arr[:, :, 2]
    )

    return gray


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values.copy()

    if window % 2 == 0:
        window += 1

    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)

    return np.convolve(padded, kernel, mode="valid")


def fit_circle_least_squares(points_xy: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    if len(points_xy) < 3:
        raise ValueError("At least 3 points are required to fit an arc.")

    x = points_xy[:, 0]
    y = points_xy[:, 1]

    matrix = np.column_stack([x, y, np.ones_like(x)])
    rhs = -(x * x + y * y)

    solution, _, _, _ = np.linalg.lstsq(matrix, rhs, rcond=None)

    d, e, f = solution
    center = np.array([-d / 2.0, -e / 2.0])

    radius_squared = center[0] * center[0] + center[1] * center[1] - f

    if radius_squared <= 0:
        raise ValueError("Circle fit failed because computed radius was invalid.")

    radius = math.sqrt(radius_squared)

    distances = np.linalg.norm(points_xy - center, axis=1)
    errors = distances - radius

    return center, radius, errors


def fit_circle_endpoint_constrained(points_xy: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Fit a circular arc that is constrained to pass exactly through the first
    and last points.

    A plain least-squares circle fit can float away from the endpoints. That
    makes adjacent arc segments look disconnected. This function forces the
    arc through the segment endpoints, so the displayed arc starts and ends
    exactly at the profile/breakpoint locations.

    Method:
        Put the chord midpoint at the origin, with x along the chord and y
        perpendicular to it. The center of any circle through the two endpoints
        must lie on the perpendicular bisector of the chord:

            center = midpoint + h * normal

        The scalar h is solved with an algebraic least-squares equation.
    """
    if len(points_xy) < 3:
        raise ValueError("At least 3 points are required to fit an arc.")

    p0 = np.asarray(points_xy[0], dtype=float)
    p1 = np.asarray(points_xy[-1], dtype=float)

    chord = p1 - p0
    chord_length = float(np.linalg.norm(chord))

    if chord_length < 1e-9:
        raise ValueError("Arc endpoints are too close together.")

    u = chord / chord_length
    n = np.array([-u[1], u[0]], dtype=float)

    midpoint = (p0 + p1) / 2.0
    relative = points_xy - midpoint

    x = relative @ u
    y = relative @ n

    half_chord = chord_length / 2.0

    denominator = 2.0 * float(np.sum(y * y))

    if abs(denominator) < 1e-12:
        # Points are nearly collinear. Use a very large-radius circle whose
        # center is far along the normal. This behaves visually like a line
        # while still preserving the existing arc drawing path.
        h = 1e9
    else:
        numerator = float(np.sum(y * (x * x + y * y - half_chord * half_chord)))
        h = numerator / denominator

    center = midpoint + h * n
    radius = float(math.sqrt(half_chord * half_chord + h * h))

    distances = np.linalg.norm(points_xy - center, axis=1)
    errors = distances - radius

    return center, radius, errors


def cosine_spaced_values(start: float, end: float, count: int) -> np.ndarray:
    """
    Return values from start to end with denser spacing near both ends.

    This helps top-view profiles because the top and bottom tips often change
    direction quickly. Uniform spacing undersamples those regions.
    """
    if count <= 1:
        return np.array([start], dtype=float)

    t = np.linspace(0.0, math.pi, count)
    return start + (end - start) * (1.0 - np.cos(t)) / 2.0


def unwrap_segment_angles(points_xy: np.ndarray, center: np.ndarray) -> np.ndarray:
    angles = np.arctan2(points_xy[:, 1] - center[1], points_xy[:, 0] - center[0])
    return np.unwrap(angles)


def bilinear_sample_gray(gray: np.ndarray, x: float, y: float) -> float:
    """
    Bilinear sample a grayscale image at floating-point image coordinates.
    Returns NaN if the point is outside the image.
    """
    height, width = gray.shape

    if x < 0.0 or y < 0.0 or x >= width - 1.0 or y >= height - 1.0:
        return float("nan")

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    dx = x - x0
    dy = y - y0

    top = (1.0 - dx) * gray[y0, x0] + dx * gray[y0, x1]
    bottom = (1.0 - dx) * gray[y1, x0] + dx * gray[y1, x1]

    return float((1.0 - dy) * top + dy * bottom)


def smooth_1d(values: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Smooth a 1D signal while preserving length.
    """
    if window <= 1 or len(values) < window:
        return values.copy()

    if window % 2 == 0:
        window += 1

    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)

    return np.convolve(padded, kernel, mode="valid")


def robust_outer_edge_from_scan(
    x_samples: np.ndarray,
    intensity: np.ndarray,
    inner_half_width: float,
    min_rim_px: float,
    max_rim_px: float,
    sensitivity: float = 0.28,
) -> Optional[float]:
    """
    Estimate the outer outside edge from a 1D outward scan.

    The scan starts near the symmetry axis and moves outward.
    For outer-edge mode, the dark inner opening usually creates the strongest
    edge, but that is the WRONG edge. We skip past the inner opening and look
    for a later transition caused by the outside chrome/body edge.

    The method is intentionally heuristic because photos vary:
        - chrome can be close in brightness to background
        - glare can create false edges
        - background texture can add noise
    """
    valid = np.isfinite(intensity)

    if np.count_nonzero(valid) < 20:
        return None

    x = x_samples[valid]
    i = intensity[valid]

    if len(x) < 20:
        return None

    i_smooth = smooth_1d(i, window=7)
    grad = np.abs(np.gradient(i_smooth, x))

    # Start looking beyond the inner opening. This avoids selecting the
    # black-opening-to-chrome edge.
    search_min = max(inner_half_width + min_rim_px, inner_half_width * 1.04)
    search_max = inner_half_width + max_rim_px

    search_mask = (x >= search_min) & (x <= search_max)

    if np.count_nonzero(search_mask) < 5:
        search_mask = x >= max(inner_half_width * 1.02, inner_half_width + 2.0)

    if np.count_nonzero(search_mask) < 5:
        return None

    x_search = x[search_mask]
    grad_search = grad[search_mask]

    # Choose a meaningful gradient threshold but do not make it too strict,
    # because chrome-to-background contrast may be weak.
    sensitivity = max(0.05, min(float(sensitivity), 0.95))

    threshold = max(
        float(np.percentile(grad_search, 75.0)),
        float(np.max(grad_search)) * sensitivity,
        0.4,
    )

    candidate_indices = np.where(grad_search >= threshold)[0]

    if len(candidate_indices) == 0:
        # Fallback: strongest edge after the inner opening.
        return float(x_search[int(np.argmax(grad_search))])

    # Prefer the later strong edge, because earlier strong edges tend to be
    # inner wall/glare/rim transitions.
    candidate_x = x_search[candidate_indices]

    # Avoid choosing the final sample, which can be an out-of-image or crop edge.
    safe_candidates = candidate_x[candidate_x < x_search[-1] - 2.0]

    if len(safe_candidates) == 0:
        safe_candidates = candidate_x

    return float(np.max(safe_candidates))


def pca_major_axis(points_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points_xy) < 3:
        raise ValueError("Need at least 3 points for PCA.")

    center = points_xy.mean(axis=0)
    centered = points_xy - center

    covariance = centered.T @ centered / max(len(points_xy) - 1, 1)

    values, vectors = np.linalg.eigh(covariance)

    major_axis = vectors[:, np.argmax(values)]
    major_axis = normalize(major_axis)

    # In image coordinates, positive Y points downward.
    # Make the major axis point generally top -> bottom.
    if major_axis[1] < 0:
        major_axis = -major_axis

    minor_axis = np.array([major_axis[1], -major_axis[0]])
    minor_axis = normalize(minor_axis)

    return center, major_axis, minor_axis


def rotate_image_pil(
    image_pil: Image.Image,
    angle_degrees: float,
    expand: bool = True,
    fill_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    image_rgb = image_pil.convert("RGB")

    return image_rgb.rotate(
        angle_degrees,
        resample=Image.Resampling.BICUBIC,
        expand=expand,
        fillcolor=fill_color,
    )


def connected_components_4(mask: np.ndarray, min_area: int = 100) -> List[Dict]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)

    components = []

    for y_start in range(height):
        for x_start in range(width):
            if not mask[y_start, x_start] or visited[y_start, x_start]:
                continue

            stack = [(x_start, y_start)]
            visited[y_start, x_start] = True

            xs = []
            ys = []
            touches_border = False

            while stack:
                x, y = stack.pop()

                xs.append(x)
                ys.append(y)

                if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                    touches_border = True

                neighbors = (
                    (x + 1, y),
                    (x - 1, y),
                    (x, y + 1),
                    (x, y - 1),
                )

                for nx, ny in neighbors:
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue

                    if visited[ny, nx] or not mask[ny, nx]:
                        continue

                    visited[ny, nx] = True
                    stack.append((nx, ny))

            area = len(xs)

            if area < min_area:
                continue

            components.append(
                {
                    "area": area,
                    "xs": np.array(xs, dtype=np.float64),
                    "ys": np.array(ys, dtype=np.float64),
                    "touches_border": touches_border,
                }
            )

    return components


def choose_strict_dark_thresholds(gray: np.ndarray) -> List[float]:
    """
    Generate several candidate thresholds for detecting a black/dark opening.

    The previous version used a loose percentile like p35. That can grab paper
    texture and shadows. This version stays much darker.
    """
    percentiles = [2, 4, 6, 8, 10, 12, 15, 18, 22]
    values = [float(np.percentile(gray, p)) for p in percentiles]

    # Add absolute cutoffs for black/dark interiors.
    values.extend([35.0, 45.0, 55.0, 65.0, 80.0, 95.0, 110.0])

    # Use only thresholds that are not absurdly bright. Anything above ~120
    # tends to include paper towel texture, shadows, etc.
    values = [v for v in values if 5.0 <= v <= 120.0]

    # Unique sorted.
    unique = sorted(set(round(v, 3) for v in values))

    return unique


def score_dark_component(component: Dict, width: int, height: int, threshold: float) -> Tuple[float, Dict]:
    xs = component["xs"]
    ys = component["ys"]

    area = float(component["area"])

    x0 = float(np.min(xs))
    x1 = float(np.max(xs))
    y0 = float(np.min(ys))
    y1 = float(np.max(ys))

    bbox_width = max(x1 - x0 + 1.0, 1.0)
    bbox_height = max(y1 - y0 + 1.0, 1.0)
    bbox_area = bbox_width * bbox_height

    fill_ratio = area / bbox_area

    centroid = np.array([float(xs.mean()), float(ys.mean())])
    image_center = np.array([width / 2.0, height / 2.0])
    diagonal = math.sqrt(width * width + height * height)
    center_distance = float(np.linalg.norm(centroid - image_center)) / max(diagonal, 1.0)

    border_margin = min(x0, y0, width - 1.0 - x1, height - 1.0 - y1)
    border_margin_fraction = border_margin / max(min(width, height), 1.0)

    pts = np.column_stack([xs, ys])
    elongation = 1.0

    if len(pts) >= 3:
        centered = pts - pts.mean(axis=0)
        covariance = centered.T @ centered / max(len(pts) - 1, 1)
        values, _ = np.linalg.eigh(covariance)
        values = np.sort(values)

        if values[0] > 1e-9:
            elongation = math.sqrt(float(values[-1] / values[0]))

    image_area = float(width * height)
    area_fraction = area / max(image_area, 1.0)

    # Reject likely background or tiny noise before scoring.
    if component["touches_border"]:
        return -1.0, {}

    if border_margin_fraction < 0.015:
        return -1.0, {}

    if area_fraction < 0.002:
        return -1.0, {}

    # A top-view opening can be large, but if it is most of the image, it is probably
    # background/shadow rather than the opening.
    if area_fraction > 0.55:
        return -1.0, {}

    if bbox_width < 20 or bbox_height < 20:
        return -1.0, {}

    # Useful profile components are usually fairly filled, centered, and elongated/oval.
    # The score intentionally penalizes loose thresholds because they grow into the background.
    threshold_penalty = 1.0 / (1.0 + max(threshold - 70.0, 0.0) / 30.0)

    score = area
    score *= max(fill_ratio, 0.05) ** 0.8
    score *= 1.0 + min(elongation, 5.0) * 0.12
    score *= 1.0 / (1.0 + center_distance * 2.5)
    score *= threshold_penalty

    metadata = {
        "bbox": (x0, y0, x1, y1),
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area": bbox_area,
        "fill_ratio": fill_ratio,
        "centroid": centroid,
        "center_distance": center_distance,
        "border_margin_fraction": border_margin_fraction,
        "elongation": elongation,
        "area_fraction": area_fraction,
        "threshold": threshold,
    }

    return score, metadata


def find_main_dark_component(image_pil: Image.Image, max_dimension: int = 750) -> Dict:
    """
    Find the dark inner opening/profile component.

    This is stricter than the older detector:
        - tries multiple low thresholds
        - scores components at each threshold
        - rejects border/background regions
        - prefers dark centered filled elongated components
    """
    image_rgb = image_pil.convert("RGB")
    full_width, full_height = image_rgb.size

    scale_down = max(full_width, full_height) / float(max_dimension)

    if scale_down > 1.0:
        small_width = max(1, int(round(full_width / scale_down)))
        small_height = max(1, int(round(full_height / scale_down)))

        image_small = image_rgb.resize(
            (small_width, small_height),
            Image.Resampling.BILINEAR
        )
    else:
        image_small = image_rgb
        scale_down = 1.0

    gray = image_to_gray_array(image_small)
    thresholds = choose_strict_dark_thresholds(gray)

    height, width = gray.shape

    best_component = None
    best_metadata = None
    best_score = -1.0
    best_threshold = None

    for threshold in thresholds:
        mask = gray < threshold
        components = connected_components_4(mask, min_area=80)

        for component in components:
            score, metadata = score_dark_component(component, width, height, threshold)

            if score > best_score:
                best_score = score
                best_component = component
                best_metadata = metadata
                best_threshold = threshold

    if best_component is None or best_score <= 0:
        raise ValueError(
            "Could not find a suitable dark opening. Try cropping the image closer "
            "to the seatpost opening or improve contrast/lighting."
        )

    xs_full = best_component["xs"] * scale_down
    ys_full = best_component["ys"] * scale_down
    points_full = np.column_stack([xs_full, ys_full])

    bbox_small = best_metadata["bbox"]
    bbox_full = tuple(float(v * scale_down) for v in bbox_small)

    centroid_full = best_metadata["centroid"] * scale_down

    note = (
        f"strict dark opening detector; fill={best_metadata['fill_ratio']:.3f}, "
        f"area_fraction={best_metadata['area_fraction']:.3f}, "
        f"elongation={best_metadata['elongation']:.3f}"
    )

    return {
        "points_px": points_full,
        "area_px": int(best_component["area"] * scale_down * scale_down),
        "threshold": float(best_threshold),
        "scale_down": scale_down,
        "score": float(best_score),
        "touches_border": bool(best_component["touches_border"]),
        "bbox_px": bbox_full,
        "centroid_px": centroid_full,
        "detector_note": note,
    }


def estimate_axis_and_end_widths(points_px: np.ndarray) -> Dict:
    center, major_axis, minor_axis = pca_major_axis(points_px)

    t = (points_px - center) @ major_axis
    q = (points_px - center) @ minor_axis

    top_mask = t <= np.percentile(t, 15.0)
    bottom_mask = t >= np.percentile(t, 85.0)

    top_width = float(np.percentile(q[top_mask], 95.0) - np.percentile(q[top_mask], 5.0))
    bottom_width = float(np.percentile(q[bottom_mask], 95.0) - np.percentile(q[bottom_mask], 5.0))

    t_min = float(np.min(t))
    t_max = float(np.max(t))

    # Use the true detected component endpoints along the major axis.
    #
    # Older versions slightly shrank these endpoints inward, which made the
    # auto-extracted mirrored profile stop short of the real top/bottom closure
    # points. For a closed airfoil/teardrop-like post opening, the profile should
    # connect back to the symmetry axis at both ends.
    axis_top = center + major_axis * t_min
    axis_bottom = center + major_axis * t_max

    return {
        "center": center,
        "major_axis": major_axis,
        "minor_axis": minor_axis,
        "top_width": top_width,
        "bottom_width": bottom_width,
        "axis_top": axis_top,
        "axis_bottom": axis_bottom,
        "t_min": t_min,
        "t_max": t_max,
    }


def find_best_circle_split(
    points_xy: np.ndarray,
    start: int,
    end: int,
    tolerance: float,
    min_points: int,
) -> Optional[int]:
    segment = points_xy[start:end + 1]

    if len(segment) < max(3, min_points):
        return None

    try:
        _, _, errors = fit_circle_endpoint_constrained(segment)
    except Exception:
        midpoint = (start + end) // 2

        if midpoint <= start + 2 or midpoint >= end - 2:
            return None

        return midpoint

    abs_errors = np.abs(errors)
    max_error = float(np.max(abs_errors))

    if max_error <= tolerance:
        return None

    local_index = int(np.argmax(abs_errors))
    split_index = start + local_index

    if split_index <= start + max(2, min_points // 3):
        split_index = start + len(segment) // 2

    if split_index >= end - max(2, min_points // 3):
        split_index = start + len(segment) // 2

    if split_index <= start + 2 or split_index >= end - 2:
        return None

    return split_index


def auto_breakpoints_recursive(
    points_xy: np.ndarray,
    tolerance: float = 2.5,
    max_arcs: int = 6,
    min_points_per_arc: int = 12,
) -> List[int]:
    if len(points_xy) < 3:
        return []

    segments = [(0, len(points_xy) - 1)]

    while len(segments) < max_arcs:
        best_candidate = None

        for segment_index, (start, end) in enumerate(segments):
            if (end - start + 1) < min_points_per_arc * 2:
                continue

            split_index = find_best_circle_split(
                points_xy,
                start,
                end,
                tolerance=tolerance,
                min_points=min_points_per_arc,
            )

            if split_index is None:
                continue

            candidate_length = end - start

            if best_candidate is None or candidate_length > best_candidate["length"]:
                best_candidate = {
                    "segment_index": segment_index,
                    "start": start,
                    "end": end,
                    "split": split_index,
                    "length": candidate_length,
                }

        if best_candidate is None:
            break

        segment_index = best_candidate["segment_index"]
        start = best_candidate["start"]
        end = best_candidate["end"]
        split = best_candidate["split"]

        segments.pop(segment_index)
        segments.append((start, split))
        segments.append((split, end))
        segments = sorted(segments)

    breakpoints = sorted({start for start, _ in segments if start != 0})

    return breakpoints


class ArcProfileFinderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Arc Profile Finder")
        self.root.geometry("1320x820")
        self.root.minsize(1000, 700)

        self.image_path: Optional[str] = None
        self.image_pil: Optional[Image.Image] = None
        self.image_array = None

        self.axis_points_px: List[np.ndarray] = []
        self.scale_points_px: List[np.ndarray] = []
        self.profile_points_px: List[np.ndarray] = []
        self.break_indices = set()

        self.scale_units = "px"
        self.scale_per_px = 1.0
        self.has_real_scale = False

        self.image_rotation_degrees = 0.0
        self.auto_orientation_report: Optional[AutoOrientationReport] = None
        self.auto_profile_report: Optional[AutoProfileReport] = None
        self.auto_breakpoint_report: Optional[AutoBreakpointReport] = None

        self.arc_results: List[ArcResult] = []

        self.mode_var = tk.StringVar(value="axis")
        self.status_var = tk.StringVar(value="Load an image to begin.")

        self.show_mirror_var = tk.BooleanVar(value=True)
        self.show_fitted_arcs_var = tk.BooleanVar(value=True)
        self.show_point_numbers_var = tk.BooleanVar(value=False)
        self.auto_process_on_load_var = tk.BooleanVar(value=True)
        self.close_ends_var = tk.BooleanVar(value=True)

        self.profile_target_var = tk.StringVar(value="Outer outside edge")
        self.lock_manual_closures_var = tk.BooleanVar(value=False)

        self.auto_point_count_var = tk.StringVar(value="90")
        self.auto_arc_tolerance_var = tk.StringVar(value="2.5")
        self.auto_max_arcs_var = tk.StringVar(value="6")

        self.outer_edge_sensitivity_var = tk.DoubleVar(value=0.28)
        self.outer_search_scale_var = tk.DoubleVar(value=1.0)
        self.smooth_window_var = tk.DoubleVar(value=5.0)

        self._build_ui()
        self._bind_shortcuts()

    def _build_ui(self):
        self.root.columnconfigure(0, weight=7)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(1, weight=1)

        title = ttk.Label(
            self.root,
            text="Arc Profile Finder",
            font=("Segoe UI", 15, "bold"),
            anchor="w"
        )
        title.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 4))

        left_frame = ttk.Frame(self.root)
        left_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=5)
        left_frame.rowconfigure(3, weight=1)
        left_frame.columnconfigure(0, weight=1)

        right_frame = ttk.Frame(self.root)
        right_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=5)
        right_frame.rowconfigure(2, weight=1)
        right_frame.columnconfigure(0, weight=1)

        self._build_button_bar(left_frame)
        self._build_auto_options_bar(left_frame)
        self._build_mode_bar(left_frame)
        self._build_plot_area(left_frame)
        self._build_status_bar(left_frame)

        self._build_results_panel(right_frame)

    def _build_button_bar(self, parent):
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        ttk.Button(frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(frame, text="Auto Process All", command=self.auto_process_all).pack(side=tk.LEFT, padx=3)
        ttk.Button(frame, text="Auto Orient Image + Axis", command=self.auto_orient_image_and_axis).pack(side=tk.LEFT, padx=3)
        ttk.Button(frame, text="Auto Extract Profile", command=self.auto_extract_profile).pack(side=tk.LEFT, padx=3)
        ttk.Button(frame, text="Auto Break + Fit", command=self.auto_break_and_fit).pack(side=tk.LEFT, padx=3)
        ttk.Button(frame, text="Rebuild Current", command=self.rebuild_current_profile).pack(side=tk.LEFT, padx=3)
        ttk.Button(frame, text="Flip Up Direction", command=self.flip_up_direction).pack(side=tk.LEFT, padx=3)

    def _build_auto_options_bar(self, parent):
        """
        Build auto controls in tabs.

        This is more reliable on smaller screens than trying to keep every
        option visible in one horizontal row. The controls are grouped by task:

            Core:
                target, profile points, arc tolerance, max arcs, closure options

            Tuning:
                edge sensitivity, outer search distance, smoothing

            Actions:
                clear/undo/fit/export buttons
        """
        frame = ttk.LabelFrame(parent, text="Auto Options")
        frame.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        frame.columnconfigure(0, weight=1)

        notebook = ttk.Notebook(frame)
        notebook.grid(row=0, column=0, sticky="ew", padx=4, pady=4)

        core_tab = ttk.Frame(notebook)
        tuning_tab = ttk.Frame(notebook)
        actions_tab = ttk.Frame(notebook)

        notebook.add(core_tab, text="Core")
        notebook.add(tuning_tab, text="Tuning")
        notebook.add(actions_tab, text="Actions")

        self._build_core_options_tab(core_tab)
        self._build_tuning_options_tab(tuning_tab)
        self._build_actions_options_tab(actions_tab)

    def _build_core_options_tab(self, parent):
        parent.columnconfigure(0, weight=1)

        row1 = ttk.Frame(parent)
        row1.grid(row=0, column=0, sticky="ew", padx=4, pady=(5, 2))

        row2 = ttk.Frame(parent)
        row2.grid(row=1, column=0, sticky="ew", padx=4, pady=(2, 5))

        ttk.Checkbutton(
            row1,
            text="Auto process on load",
            variable=self.auto_process_on_load_var
        ).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Checkbutton(
            row1,
            text="Close ends to axis",
            variable=self.close_ends_var
        ).pack(side=tk.LEFT, padx=12)

        ttk.Checkbutton(
            row1,
            text="Lock manual closures",
            variable=self.lock_manual_closures_var
        ).pack(side=tk.LEFT, padx=12)

        ttk.Label(row2, text="Target:").pack(side=tk.LEFT, padx=(4, 2))
        target_combo = ttk.Combobox(
            row2,
            textvariable=self.profile_target_var,
            values=["Outer outside edge", "Inner dark opening"],
            width=18,
            state="readonly"
        )
        target_combo.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(row2, text="Profile pts:").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Entry(row2, textvariable=self.auto_point_count_var, width=7).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(row2, text="Arc tolerance:").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Entry(row2, textvariable=self.auto_arc_tolerance_var, width=7).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(row2, text="Max arcs:").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Entry(row2, textvariable=self.auto_max_arcs_var, width=7).pack(side=tk.LEFT, padx=(0, 12))

    def _build_tuning_options_tab(self, parent):
        """
        Build tuning sliders in vertical rows.

        This avoids the previous problem where three sliders in one horizontal
        row still ran off-screen on smaller displays.
        """
        parent.columnconfigure(0, weight=1)

        def add_slider_row(
            parent_frame,
            row_index,
            label_text,
            variable,
            from_value,
            to_value,
            entry_width,
            help_text,
        ):
            row = ttk.Frame(parent_frame)
            row.grid(row=row_index, column=0, sticky="ew", padx=6, pady=(5, 2))
            row.columnconfigure(2, weight=1)

            ttk.Label(row, text=label_text, width=18, anchor="w").grid(
                row=0,
                column=0,
                sticky="w",
                padx=(0, 4),
            )

            ttk.Entry(row, textvariable=variable, width=entry_width).grid(
                row=0,
                column=1,
                sticky="w",
                padx=(0, 8),
            )

            slider = ttk.Scale(
                row,
                variable=variable,
                from_=from_value,
                to=to_value,
                orient=tk.HORIZONTAL,
            )
            slider.grid(row=0, column=2, sticky="ew", padx=(0, 8))

            ttk.Label(
                parent_frame,
                text=help_text,
                foreground="gray35",
                wraplength=900,
                justify=tk.LEFT,
            ).grid(row=row_index + 1, column=0, sticky="ew", padx=8, pady=(0, 4))

            return slider

        add_slider_row(
            parent,
            0,
            "Edge sensitivity:",
            self.outer_edge_sensitivity_var,
            0.05,
            0.95,
            6,
            "Lower accepts weaker edges; higher requires stronger contrast.",
        )

        add_slider_row(
            parent,
            2,
            "Search distance:",
            self.outer_search_scale_var,
            0.25,
            3.0,
            6,
            "Higher searches farther outward from the inner opening.",
        )

        add_slider_row(
            parent,
            4,
            "Smooth:",
            self.smooth_window_var,
            1.0,
            25.0,
            5,
            "Higher smooths the extracted edge more, but can erase corners.",
        )

    def _build_actions_options_tab(self, parent):
        row = ttk.Frame(parent)
        row.grid(row=0, column=0, sticky="w", padx=4, pady=8)

        ttk.Button(row, text="Clear Axis", command=self.clear_axis).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Clear Scale", command=self.clear_scale).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Undo Point", command=self.undo_profile_point).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Clear Profile", command=self.clear_profile).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Fit Arcs", command=self.fit_arcs).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Export CSV", command=self.export_csv).pack(side=tk.LEFT, padx=4)

    def _build_mode_bar(self, parent):
        """
        Manual editing controls split into two rows.

        Row 1: editing modes
        Row 2: display toggles

        This keeps the row from running off the right side on smaller screens.
        """
        frame = ttk.LabelFrame(parent, text="Manual Editing Mode")
        frame.grid(row=2, column=0, sticky="ew", pady=(0, 5))
        frame.columnconfigure(0, weight=1)

        mode_row = ttk.Frame(frame)
        mode_row.grid(row=0, column=0, sticky="ew", padx=4, pady=(3, 1))

        display_row = ttk.Frame(frame)
        display_row.grid(row=1, column=0, sticky="ew", padx=4, pady=(1, 4))

        modes = [
            ("Set Axis / Closures", "axis"),
            ("Set Scale", "scale"),
            ("Add Profile Points", "profile"),
            ("Add Breakpoints", "breakpoints"),
        ]

        for text, value in modes:
            ttk.Radiobutton(
                mode_row,
                text=text,
                value=value,
                variable=self.mode_var,
                command=self.update_status_for_mode
            ).pack(side=tk.LEFT, padx=8, pady=2)

        ttk.Checkbutton(
            display_row,
            text="Show mirror",
            variable=self.show_mirror_var,
            command=self.redraw
        ).pack(side=tk.LEFT, padx=8)

        ttk.Checkbutton(
            display_row,
            text="Show fitted arcs",
            variable=self.show_fitted_arcs_var,
            command=self.redraw
        ).pack(side=tk.LEFT, padx=8)

        ttk.Checkbutton(
            display_row,
            text="Point numbers",
            variable=self.show_point_numbers_var,
            command=self.redraw
        ).pack(side=tk.LEFT, padx=8)

    def _build_plot_area(self, parent):
        self.figure = Figure(figsize=(10, 7), dpi=100)
        self.ax_image = self.figure.add_subplot(121)
        self.ax_profile = self.figure.add_subplot(122)

        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().grid(row=3, column=0, sticky="nsew")

        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.grid(row=4, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        self.canvas.mpl_connect("button_press_event", self.on_mouse_click)

        self._draw_empty()

    def _build_status_bar(self, parent):
        label = ttk.Label(parent, textvariable=self.status_var, anchor="w")
        label.grid(row=5, column=0, sticky="ew", pady=(5, 0))

    def _build_results_panel(self, parent):
        parent.rowconfigure(2, weight=1)
        parent.columnconfigure(0, weight=1)

        instructions_outer = ttk.LabelFrame(parent, text="Quick Instructions")
        instructions_outer.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        instructions_outer.columnconfigure(0, weight=1)

        self.instructions_visible_var = tk.BooleanVar(value=True)

        header_row = ttk.Frame(instructions_outer)
        header_row.grid(row=0, column=0, sticky="ew", padx=4, pady=(3, 2))
        header_row.columnconfigure(0, weight=1)

        ttk.Label(
            header_row,
            text="Workflow help",
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        self.instructions_toggle_button = ttk.Button(
            header_row,
            text="Hide",
            width=8,
            command=self.toggle_quick_instructions,
        )
        self.instructions_toggle_button.grid(row=0, column=1, sticky="e")

        self.instructions_content_frame = ttk.Frame(instructions_outer)
        self.instructions_content_frame.grid(row=1, column=0, sticky="ew", padx=4, pady=(0, 4))

        instructions = (
            "1. Load Image.\n"
            "2. Auto Process All gives a first pass.\n"
            "3. If top/bottom are wrong: Set Axis / Closures, click top then bottom.\n"
            "4. Check Lock manual closures, then Rebuild Current.\n"
            "5. Use the Tuning tab if the edge is weak or over/under-shoots.\n"
            "6. Set Scale if you need mm instead of pixels.\n"
            "7. Export CSV."
        )

        ttk.Label(
            self.instructions_content_frame,
            text=instructions,
            justify=tk.LEFT,
            wraplength=430,
        ).pack(anchor="w", padx=4, pady=4)

        info_frame = ttk.LabelFrame(parent, text="Scale / Axis / Orientation")
        info_frame.grid(row=1, column=0, sticky="ew", pady=(0, 5))

        self.axis_info_var = tk.StringVar(value="Axis: not set")
        self.scale_info_var = tk.StringVar(value="Scale: not set; using pixels")
        self.orientation_info_var = tk.StringVar(value="Image orientation: not applied")
        self.profile_info_var = tk.StringVar(value="Auto profile: not extracted")

        ttk.Label(info_frame, textvariable=self.axis_info_var, anchor="w", wraplength=430).pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(info_frame, textvariable=self.scale_info_var, anchor="w", wraplength=430).pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(info_frame, textvariable=self.orientation_info_var, anchor="w", wraplength=430).pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(info_frame, textvariable=self.profile_info_var, anchor="w", wraplength=430).pack(fill=tk.X, padx=6, pady=2)

        results_frame = ttk.LabelFrame(parent, text="Arc Results")
        results_frame.grid(row=2, column=0, sticky="nsew")
        results_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)

        self.results_text = tk.Text(
            results_frame,
            wrap="none",
            width=52,
            height=24
        )
        self.results_text.grid(row=0, column=0, sticky="nsew")

        scroll_y = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        scroll_y.grid(row=0, column=1, sticky="ns")

        scroll_x = ttk.Scrollbar(results_frame, orient="horizontal", command=self.results_text.xview)
        scroll_x.grid(row=1, column=0, sticky="ew")

        self.results_text.configure(
            yscrollcommand=scroll_y.set,
            xscrollcommand=scroll_x.set
        )

        ttk.Button(
            parent,
            text="Copy Results",
            command=self.copy_results
        ).grid(row=3, column=0, sticky="ew", pady=(8, 0))

        self.update_results_text()

    def toggle_quick_instructions(self):
        if self.instructions_visible_var.get():
            self.instructions_content_frame.grid_remove()
            self.instructions_visible_var.set(False)
            self.instructions_toggle_button.configure(text="Show")
        else:
            self.instructions_content_frame.grid()
            self.instructions_visible_var.set(True)
            self.instructions_toggle_button.configure(text="Hide")

    def _bind_shortcuts(self):
        self.root.bind("<Control-z>", lambda event: self.undo_profile_point())
        self.root.bind("<Control-Z>", lambda event: self.undo_profile_point())
        self.root.bind("f", lambda event: self.fit_arcs())
        self.root.bind("F", lambda event: self.fit_arcs())
        self.root.bind("a", lambda event: self.set_mode("axis"))
        self.root.bind("s", lambda event: self.set_mode("scale"))
        self.root.bind("p", lambda event: self.set_mode("profile"))
        self.root.bind("b", lambda event: self.set_mode("breakpoints"))

    def safe_int_from_var(self, var, default, minimum=None, maximum=None) -> int:
        try:
            value = int(float(var.get()))
        except Exception:
            value = default

        if minimum is not None:
            value = max(value, minimum)

        if maximum is not None:
            value = min(value, maximum)

        try:
            var.set(value)
        except Exception:
            var.set(str(value))

        return value

    def safe_float_from_var(self, var, default, minimum=None, maximum=None) -> float:
        try:
            value = float(var.get())
        except Exception:
            value = default

        if minimum is not None:
            value = max(value, minimum)

        if maximum is not None:
            value = min(value, maximum)

        try:
            var.set(value)
        except Exception:
            var.set(str(value))

        return value

    def set_mode(self, mode):
        self.mode_var.set(mode)
        self.update_status_for_mode()

    def set_image(self, image_pil: Image.Image):
        self.image_pil = image_pil.convert("RGB")
        self.image_array = pil_to_display_array(self.image_pil)

    def clear_annotations_for_new_image(self):
        self.axis_points_px.clear()
        self.scale_points_px.clear()
        self.profile_points_px.clear()
        self.break_indices.clear()
        self.arc_results.clear()

        self.scale_units = "px"
        self.scale_per_px = 1.0
        self.has_real_scale = False

        self.image_rotation_degrees = 0.0
        self.auto_orientation_report = None
        self.auto_profile_report = None
        self.auto_breakpoint_report = None

    def clear_annotations_for_new_orientation(self):
        self.axis_points_px.clear()
        self.scale_points_px.clear()
        self.profile_points_px.clear()
        self.break_indices.clear()
        self.arc_results.clear()

        self.scale_units = "px"
        self.scale_per_px = 1.0
        self.has_real_scale = False

        self.auto_profile_report = None
        self.auto_breakpoint_report = None

    def update_status_for_mode(self):
        mode = self.mode_var.get()

        if mode == "axis":
            self.status_var.set("Set Axis / Closures mode: click TOP closure point, then BOTTOM closure point.")
        elif mode == "scale":
            self.status_var.set("Set Scale mode: click two points with a known real-world distance.")
        elif mode == "profile":
            self.status_var.set("Add Profile Points mode: click one side of the outline in order.")
        elif mode == "breakpoints":
            self.status_var.set("Add Breakpoints mode: click near a profile point to toggle an arc boundary.")

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*"),
            ]
        )

        if not path:
            return

        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Image load failed", str(exc))
            return

        self.image_path = path
        self.set_image(image)

        self.clear_annotations_for_new_image()

        self.mode_var.set("axis")
        self.status_var.set("Image loaded.")

        self.update_axis_scale_info()
        self.update_results_text()
        self.redraw()

        if self.auto_process_on_load_var.get():
            self.root.after(100, self.auto_process_all)
        else:
            self.status_var.set(
                "Image loaded. Click Auto Process All, or manually set the axis."
            )

    def get_outer_edge_tuning(self) -> Tuple[float, float, int]:
        """
        Read user-facing outer-edge tuning controls.

        Edge sens:
            Lower = easier to accept weak edges.
            Higher = requires stronger contrast.
            Typical range: 0.15 to 0.45

        Search:
            Multiplier for how far outward to look past the inner opening.
            Higher = looser/wider search.
            Typical range: 0.7 to 1.8

        Smooth:
            Moving-average window for the extracted half-width curve.
            Higher = smoother but can erase corners.
            Typical range: 3 to 9
        """
        sensitivity = self.safe_float_from_var(
            self.outer_edge_sensitivity_var,
            default=0.28,
            minimum=0.05,
            maximum=0.95,
        )

        search_scale = self.safe_float_from_var(
            self.outer_search_scale_var,
            default=1.0,
            minimum=0.25,
            maximum=3.0,
        )

        smooth_window = self.safe_int_from_var(
            self.smooth_window_var,
            default=5,
            minimum=1,
            maximum=25,
        )

        if smooth_window % 2 == 0:
            smooth_window += 1
            self.smooth_window_var.set(str(smooth_window))

        return sensitivity, search_scale, smooth_window

    def rebuild_current_profile(self):
        """
        Re-extract profile and re-fit arcs using current image, current axis,
        and current tuning settings.
        """
        if self.image_pil is None:
            messagebox.showwarning("No image", "Load an image first.")
            return

        if len(self.axis_points_px) != 2:
            messagebox.showwarning("Axis required", "Set the axis/closure points first.")
            return

        try:
            self.auto_extract_profile(show_errors=False)
            self.auto_break_and_fit(show_errors=False)
        except Exception as exc:
            messagebox.showerror("Rebuild failed", str(exc))
            self.status_var.set("Rebuild failed.")
            return

        self.update_axis_scale_info()
        self.update_results_text()
        self.redraw()
        self.status_var.set(
            f"Rebuilt profile using current axis/settings. "
            f"{len(self.profile_points_px)} points, {len(self.arc_results)} arc(s)."
        )

    def auto_process_all(self):
        if self.image_pil is None:
            messagebox.showwarning("No image", "Load an image first.")
            return

        try:
            self.status_var.set("Auto-processing: finding dark opening and orienting...")
            self.root.update_idletasks()
            self.auto_orient_image_and_axis(show_errors=False)

            self.status_var.set("Auto-processing: extracting profile...")
            self.root.update_idletasks()
            self.auto_extract_profile(show_errors=False)

            self.status_var.set("Auto-processing: choosing breakpoints and fitting arcs...")
            self.root.update_idletasks()
            self.auto_break_and_fit(show_errors=False)

        except Exception as exc:
            messagebox.showerror("Auto process failed", str(exc))
            self.status_var.set("Auto process failed.")
            return

        self.mode_var.set("profile")
        self.update_axis_scale_info()
        self.update_results_text()
        self.redraw()
        self.status_var.set(
            f"Auto process complete: {len(self.profile_points_px)} profile points, "
            f"{len(self.arc_results)} fitted arc(s)."
        )

    def auto_orient_image_and_axis(self, show_errors=True):
        if self.image_pil is None:
            if show_errors:
                messagebox.showwarning("No image", "Load an image first.")
            raise ValueError("No image loaded.")

        try:
            component = find_main_dark_component(self.image_pil)
            axis_info = estimate_axis_and_end_widths(component["points_px"])

            major_axis = axis_info["major_axis"]

            theta_degrees = math.degrees(math.atan2(major_axis[1], major_axis[0]))
            rotation_degrees = theta_degrees - 90.0

            rotated_image = rotate_image_pil(self.image_pil, rotation_degrees, expand=True)

            rotated_component = find_main_dark_component(rotated_image)
            rotated_axis_info = estimate_axis_and_end_widths(rotated_component["points_px"])

            used_180_flip = False

            # "Up" heuristic:
            # The wider end of a top-view teardrop/airfoil profile is assumed to go at the top.
            if rotated_axis_info["bottom_width"] > rotated_axis_info["top_width"]:
                rotated_image = rotate_image_pil(rotated_image, 180.0, expand=False)
                rotation_degrees += 180.0
                used_180_flip = True

                rotated_component = find_main_dark_component(rotated_image)
                rotated_axis_info = estimate_axis_and_end_widths(rotated_component["points_px"])

            self.set_image(rotated_image)
            self.clear_annotations_for_new_orientation()

            self.axis_points_px = [
                np.array(rotated_axis_info["axis_top"], dtype=float),
                np.array(rotated_axis_info["axis_bottom"], dtype=float),
            ]

            self.image_rotation_degrees += rotation_degrees

            self.auto_orientation_report = AutoOrientationReport(
                rotation_degrees=rotation_degrees,
                used_180_flip=used_180_flip,
                detected_area_px=rotated_component["area_px"],
                top_width_px=rotated_axis_info["top_width"],
                bottom_width_px=rotated_axis_info["bottom_width"],
                axis_top_px=np.array(rotated_axis_info["axis_top"], dtype=float),
                axis_bottom_px=np.array(rotated_axis_info["axis_bottom"], dtype=float),
                threshold_value=rotated_component["threshold"],
                component_score=rotated_component["score"],
                component_centroid_px=np.array(rotated_component["centroid_px"], dtype=float),
                component_bbox_px=rotated_component["bbox_px"],
                detector_note=rotated_component["detector_note"],
            )

        except Exception as exc:
            if show_errors:
                messagebox.showerror("Auto orientation failed", str(exc))
            raise

        self.mode_var.set("profile")
        self.update_axis_scale_info()
        self.update_results_text()
        self.redraw()

        if show_errors:
            flip_text = " A 180° up-flip was applied." if used_180_flip else ""
            self.status_var.set(
                f"Auto-oriented image and set axis. Rotation applied: "
                f"{rotation_degrees:.3f}°.{flip_text}"
            )

    def auto_extract_profile(self, show_errors=True):
        if self.image_pil is None:
            if show_errors:
                messagebox.showwarning("No image", "Load an image first.")
            raise ValueError("No image loaded.")

        if len(self.axis_points_px) != 2:
            if show_errors:
                messagebox.showwarning("Axis required", "Auto-orient or manually set the axis first.")
            raise ValueError("Axis not set.")

        requested_points = self.safe_int_from_var(
            self.auto_point_count_var,
            default=90,
            minimum=25,
            maximum=250,
        )

        target = self.profile_target_var.get()

        try:
            component = find_main_dark_component(self.image_pil)
            component_points = component["points_px"]

            if target == "Outer outside edge":
                profile_xy = self.extract_outer_profile_xy_from_image(
                    component_points,
                    requested_points=requested_points,
                )
            else:
                profile_xy = self.extract_symmetric_profile_xy_from_component(
                    component_points,
                    requested_points=requested_points,
                )

            if len(profile_xy) < 8:
                raise ValueError("Profile extraction produced too few points.")

            self.profile_points_px = [
                self.profile_xy_to_image_point(float(x), float(y))
                for x, y in profile_xy
            ]

            self.break_indices.clear()
            self.arc_results.clear()

            sensitivity, search_scale, smooth_window = self.get_outer_edge_tuning()

            self.auto_profile_report = AutoProfileReport(
                target=target,
                point_count=len(profile_xy),
                locked_manual_closures=bool(self.lock_manual_closures_var.get()),
                edge_sensitivity=sensitivity,
                outer_search_scale=search_scale,
                smooth_window=smooth_window,
                y_min=float(np.min(profile_xy[:, 1])),
                y_max=float(np.max(profile_xy[:, 1])),
                max_half_width=float(np.max(profile_xy[:, 0])),
                extraction_bins=requested_points,
            )

            self.auto_breakpoint_report = None

        except Exception as exc:
            if show_errors:
                messagebox.showerror("Auto profile extraction failed", str(exc))
            raise

        self.mode_var.set("profile")
        self.update_axis_scale_info()
        self.update_results_text()
        self.redraw()

        if show_errors:
            self.status_var.set(
                f"Auto-extracted {len(self.profile_points_px)} profile points for {target}."
            )

    def current_axis_basis_values(self):
        """
        Capture the current axis basis.

        This is useful for outer-edge mode because the initial axis comes from
        the dark inner opening, but the final outer profile may need a longer
        axis. We can keep using the old basis for scanning, then replace the
        axis endpoints after we find the true outer top/bottom.
        """
        top, bottom, axis_dir, axis_perp, axis_length_px = self.axis_basis()
        side_sign = self.profile_side_sign()

        return top.copy(), bottom.copy(), axis_dir.copy(), axis_perp.copy(), axis_length_px, side_sign

    def profile_xy_to_image_point_with_basis(
        self,
        x: float,
        y: float,
        top: np.ndarray,
        axis_dir: np.ndarray,
        axis_perp: np.ndarray,
        axis_length_px: float,
        side_sign: float,
    ) -> np.ndarray:
        """
        Convert profile coordinates to image pixels using a supplied axis basis.

        This allows outer-edge extraction to scan beyond the original inner-axis
        endpoints before replacing the axis with the larger outer-axis endpoints.
        """
        s_px = axis_length_px - (y / self.scale_per_px)
        radial_px = (x / self.scale_per_px) * side_sign

        return top + axis_dir * s_px + axis_perp * radial_px

    def sample_outer_edge_at_old_y(
        self,
        gray: np.ndarray,
        old_y: float,
        inner_h: float,
        old_basis,
        max_inner: float,
        require_strong_edge: bool,
        sensitivity: float,
        search_scale: float,
    ) -> Optional[float]:
        """
        Sample the right-side outer edge at a Y coordinate expressed in the
        original inner-axis coordinate system.

        The scan starts at the symmetry axis and moves outward. The algorithm
        deliberately skips the inner opening edge and searches for a later edge
        that should correspond to the outside body/chrome boundary.
        """
        old_top, _, old_axis_dir, old_axis_perp, old_axis_length, old_side_sign = old_basis

        min_rim_px = max(4.0 * self.scale_per_px, max_inner * 0.015)
        max_rim_px = max(120.0 * self.scale_per_px, max_inner * 0.75) * search_scale

        scan_start = 0.0
        scan_end = max(inner_h + max_rim_px, inner_h * (1.0 + 0.65 * search_scale) + 35.0 * self.scale_per_px)
        scan_end = max(scan_end, inner_h + 30.0 * self.scale_per_px)

        sample_count = 420
        x_samples = np.linspace(scan_start, scan_end, sample_count)

        intensities = []

        for x in x_samples:
            px = self.profile_xy_to_image_point_with_basis(
                float(x),
                float(old_y),
                old_top,
                old_axis_dir,
                old_axis_perp,
                old_axis_length,
                old_side_sign,
            )
            intensities.append(bilinear_sample_gray(gray, float(px[0]), float(px[1])))

        intensities = np.array(intensities, dtype=float)

        edge_x = robust_outer_edge_from_scan(
            x_samples,
            intensities,
            inner_half_width=float(inner_h),
            min_rim_px=float(min_rim_px),
            max_rim_px=float(max_rim_px),
            sensitivity=float(sensitivity),
        )

        if edge_x is not None:
            return float(edge_x)

        if require_strong_edge:
            return None

        # Fallback used only while drawing the final profile, not while deciding
        # whether the outer body exists at a given Y.
        return float(inner_h + max(20.0 * self.scale_per_px, max_inner * 0.12))

    def choose_contiguous_outer_y_span(
        self,
        y_scan: np.ndarray,
        widths: np.ndarray,
        inner_axis_length: float,
    ) -> Tuple[float, float]:
        """
        Choose the Y-span for the outer profile.

        The scan may pick up stray background texture. We keep the valid
        contiguous run that contains the inner opening center, or the nearest
        large run if the center sample failed.
        """
        finite = np.isfinite(widths)

        if np.count_nonzero(finite) < 8:
            raise ValueError("Could not find enough outer-edge samples to estimate top/bottom.")

        finite_widths = widths[finite]
        max_width = float(np.max(finite_widths))

        if max_width <= 1e-9:
            raise ValueError("Outer-edge samples had invalid width.")

        # A real outer profile should have meaningful half-width. This removes
        # isolated weak detections far above/below the part.
        width_threshold = max(max_width * 0.08, 8.0 * self.scale_per_px)
        valid = finite & (widths >= width_threshold)

        if np.count_nonzero(valid) < 8:
            valid = finite

        valid_indices = np.where(valid)[0]

        groups = []
        start = int(valid_indices[0])
        prev = int(valid_indices[0])

        for index in valid_indices[1:]:
            index = int(index)

            if index == prev + 1:
                prev = index
            else:
                groups.append((start, prev))
                start = index
                prev = index

        groups.append((start, prev))

        center_y = inner_axis_length / 2.0

        best_group = None
        best_score = -1e99

        for start, end in groups:
            group_y0 = float(y_scan[start])
            group_y1 = float(y_scan[end])
            group_center = (group_y0 + group_y1) / 2.0
            group_length = max(group_y1 - group_y0, 1e-9)

            contains_center = group_y0 <= center_y <= group_y1
            center_distance = abs(group_center - center_y)

            score = group_length - center_distance * 0.25

            if contains_center:
                score += inner_axis_length * 2.0

            if score > best_score:
                best_score = score
                best_group = (start, end)

        if best_group is None:
            raise ValueError("Could not choose a valid outer Y span.")

        start, end = best_group

        step = float(np.median(np.diff(y_scan))) if len(y_scan) > 1 else 1.0

        # Expand a little to account for the exact closure point being between
        # scan samples.
        y_min = float(y_scan[start] - step * 0.5)
        y_max = float(y_scan[end] + step * 0.5)

        # Make sure the span at least contains the original inner opening.
        y_min = min(y_min, 0.0)
        y_max = max(y_max, inner_axis_length)

        return y_min, y_max

    def estimate_inner_half_width_curve(
        self,
        component_points_px: np.ndarray,
        sample_count: int,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Estimate the inner dark opening half-width as a function of Y.

        This is used by outer-edge mode so the scan can deliberately skip past
        the inner edge and look for the outside chrome/body edge instead.
        """
        _, _, _, _, axis_length_px = self.axis_basis()
        axis_length = axis_length_px * self.scale_per_px

        profile_coords = np.array([
            self.image_point_to_profile_xy(point)
            for point in component_points_px
        ])

        x_values = profile_coords[:, 0]
        y_values = profile_coords[:, 1]

        valid = (y_values >= 0.0) & (y_values <= axis_length)

        if np.count_nonzero(valid) < 20:
            raise ValueError("Detected component has too few usable points inside the axis span.")

        x_values = x_values[valid]
        y_values = y_values[valid]

        raw_y_low = float(np.min(y_values))
        raw_y_high = float(np.max(y_values))

        extraction_bins = max(sample_count * 3, 75)
        bins = np.linspace(raw_y_low, raw_y_high, extraction_bins + 1)

        y_measured = []
        half_measured = []

        for i in range(extraction_bins):
            y0 = bins[i]
            y1 = bins[i + 1]
            mask = (y_values >= y0) & (y_values < y1)

            if int(np.count_nonzero(mask)) < 5:
                continue

            xs = x_values[mask]
            left = float(np.percentile(xs, 1.5))
            right = float(np.percentile(xs, 98.5))
            half_width = max(0.0, (right - left) / 2.0)

            y_measured.append((y0 + y1) / 2.0)
            half_measured.append(half_width)

        if len(y_measured) < 8:
            raise ValueError("Could not estimate inner half-width curve.")

        y_arr = np.array(y_measured, dtype=float)
        half_arr = np.array(half_measured, dtype=float)

        order = np.argsort(y_arr)
        y_arr = y_arr[order]
        half_arr = moving_average(half_arr[order], window=5)

        return y_arr, half_arr, axis_length

    def extract_outer_profile_xy_from_image(
        self,
        component_points_px: np.ndarray,
        requested_points: int,
    ) -> np.ndarray:
        """
        Trace the OUTER outside edge.

        The dark inner opening is used as a guide/orientation reference.

        If Lock manual closures is OFF:
            The app scans above and below the dark opening and estimates the
            larger outer top/bottom closure points.

        If Lock manual closures is ON:
            The current axis top/bottom points are treated as authoritative.
            The app does not move them. This is the recommended workflow when
            the automatic endpoint guess is close but not reliable.
        """
        sensitivity, search_scale, smooth_window = self.get_outer_edge_tuning()

        old_basis = self.current_axis_basis_values()
        old_top, _, old_axis_dir, old_axis_perp, old_axis_length, old_side_sign = old_basis

        y_inner, inner_half, inner_axis_length = self.estimate_inner_half_width_curve(
            component_points_px,
            sample_count=requested_points,
        )

        if inner_axis_length <= 1e-9:
            raise ValueError("Axis length is too small.")

        gray = image_to_gray_array(self.image_pil)

        max_inner = float(np.max(inner_half))

        if max_inner <= 1e-9:
            raise ValueError("Could not estimate inner opening width.")

        lock_manual = bool(self.lock_manual_closures_var.get())

        if lock_manual:
            # Use the current axis exactly. This lets the user manually define
            # true outer top/bottom closure points.
            y_min_old = 0.0
            y_max_old = old_axis_length * self.scale_per_px
        else:
            # Search above and below the inner opening to estimate outer closure.
            y_extension = max(max_inner * 0.90, inner_axis_length * 0.18, 80.0 * self.scale_per_px) * search_scale
            y_scan_count = max(requested_points * 3, 160)

            y_scan = np.linspace(
                -y_extension,
                inner_axis_length + y_extension,
                y_scan_count,
            )

            inner_at_scan = np.interp(
                y_scan,
                y_inner,
                inner_half,
                left=0.0,
                right=0.0,
            )

            outer_width_scan = np.full_like(y_scan, np.nan, dtype=float)

            for index, (old_y, inner_h) in enumerate(zip(y_scan, inner_at_scan)):
                edge_x = self.sample_outer_edge_at_old_y(
                    gray=gray,
                    old_y=float(old_y),
                    inner_h=float(inner_h),
                    old_basis=old_basis,
                    max_inner=max_inner,
                    require_strong_edge=True,
                    sensitivity=sensitivity,
                    search_scale=search_scale,
                )

                if edge_x is not None:
                    outer_width_scan[index] = edge_x

            y_min_old, y_max_old = self.choose_contiguous_outer_y_span(
                y_scan,
                outer_width_scan,
                inner_axis_length,
            )

            outer_axis_length = y_max_old - y_min_old

            if outer_axis_length <= 1e-9:
                raise ValueError("Outer profile axis length is invalid.")

            # Replace axis endpoints with the estimated outer profile endpoints.
            new_bottom_px = self.profile_xy_to_image_point_with_basis(
                0.0,
                y_min_old,
                old_top,
                old_axis_dir,
                old_axis_perp,
                old_axis_length,
                old_side_sign,
            )

            new_top_px = self.profile_xy_to_image_point_with_basis(
                0.0,
                y_max_old,
                old_top,
                old_axis_dir,
                old_axis_perp,
                old_axis_length,
                old_side_sign,
            )

            self.axis_points_px = [
                np.array(new_top_px, dtype=float),
                np.array(new_bottom_px, dtype=float),
            ]

        # After any automatic endpoint replacement, use the current axis as the
        # final output coordinate system.
        final_basis = self.current_axis_basis_values()
        _, _, _, _, final_axis_length, _ = final_basis
        outer_axis_length = final_axis_length * self.scale_per_px

        close_ends = bool(self.close_ends_var.get())

        if close_ends:
            target_y_new = cosine_spaced_values(0.0, outer_axis_length, requested_points)
        else:
            target_y_new = np.linspace(0.0, outer_axis_length, requested_points)

        outer_half = []

        if lock_manual:
            final_to_old_offset = 0.0
        else:
            final_to_old_offset = y_min_old

        for new_y in target_y_new:
            old_y = final_to_old_offset + float(new_y)

            if close_ends and (new_y <= 1e-9 or new_y >= outer_axis_length - 1e-9):
                outer_half.append(0.0)
                continue

            inner_h = float(np.interp(
                old_y,
                y_inner,
                inner_half,
                left=0.0,
                right=0.0,
            ))

            edge_x = self.sample_outer_edge_at_old_y(
                gray=gray,
                old_y=float(old_y),
                inner_h=inner_h,
                old_basis=old_basis,
                max_inner=max_inner,
                require_strong_edge=False,
                sensitivity=sensitivity,
                search_scale=search_scale,
            )

            if edge_x is None:
                edge_x = float(inner_h + max(20.0 * self.scale_per_px, max_inner * 0.12))

            outer_half.append(float(edge_x))

        outer_half = np.array(outer_half, dtype=float)

        # Smooth interior but preserve exact closure.
        if len(outer_half) >= 7 and smooth_window > 1:
            smoothed = smooth_1d(outer_half, window=smooth_window)

            if close_ends:
                smoothed[0] = 0.0
                smoothed[-1] = 0.0

            outer_half = smoothed

        profile_xy = np.column_stack([outer_half, target_y_new])

        return profile_xy

    def extract_symmetric_profile_xy_from_component(
        self,
        component_points_px: np.ndarray,
        requested_points: int,
    ) -> np.ndarray:
        """
        Convert the detected dark opening into one side of a mirrored profile.

        The output side is intentionally closed at both ends:

            bottom closure:  X = 0, Y = 0
            top closure:     X = 0, Y = axis length

        The interior profile is found by slicing the detected dark component
        perpendicular to the axis and measuring half-width.

        The final sample points are cosine-spaced, not uniformly spaced, so
        more points land near the top and bottom tips where the shape changes
        direction quickly. This greatly improves low point-count profiles.
        """
        _, _, _, _, axis_length_px = self.axis_basis()
        axis_length = axis_length_px * self.scale_per_px

        if axis_length <= 1e-9:
            raise ValueError("Axis length is too small.")

        profile_coords = np.array([
            self.image_point_to_profile_xy(point)
            for point in component_points_px
        ])

        x_values = profile_coords[:, 0]
        y_values = profile_coords[:, 1]

        valid = (y_values >= 0.0) & (y_values <= axis_length)

        if np.count_nonzero(valid) < 20:
            raise ValueError("Detected component has too few usable points inside the axis span.")

        x_values = x_values[valid]
        y_values = y_values[valid]

        raw_y_low = float(np.min(y_values))
        raw_y_high = float(np.max(y_values))

        if raw_y_high <= raw_y_low:
            raise ValueError("Invalid component extent along the axis.")

        # Use more extraction bins than final output points so the width curve
        # is measured with enough local detail before resampling.
        extraction_bins = max(requested_points * 3, 75)
        bins = np.linspace(raw_y_low, raw_y_high, extraction_bins + 1)

        y_measured = []
        half_width_measured = []

        for i in range(extraction_bins):
            y0 = bins[i]
            y1 = bins[i + 1]
            mask = (y_values >= y0) & (y_values < y1)

            if int(np.count_nonzero(mask)) < 5:
                continue

            xs = x_values[mask]

            # Robust edge estimate: avoids single noisy pixels while still
            # keeping the real outline.
            left = float(np.percentile(xs, 1.5))
            right = float(np.percentile(xs, 98.5))
            half_width = max(0.0, (right - left) / 2.0)

            y_mid = (y0 + y1) / 2.0

            y_measured.append(y_mid)
            half_width_measured.append(half_width)

        if len(y_measured) < 8:
            raise ValueError("Not enough profile bins found.")

        y_arr = np.array(y_measured, dtype=float)
        half_arr = np.array(half_width_measured, dtype=float)

        order = np.argsort(y_arr)
        y_arr = y_arr[order]
        half_arr = half_arr[order]

        # Smooth the measured half-width before final sampling.
        half_arr = moving_average(half_arr, window=5)

        max_half_width = float(np.max(half_arr))

        if max_half_width <= 1e-9:
            raise ValueError("Could not estimate profile width.")

        close_ends = bool(self.close_ends_var.get())

        if close_ends:
            # Add exact closure samples to the interpolation source.
            source_y = np.concatenate([
                np.array([0.0]),
                y_arr,
                np.array([axis_length]),
            ])
            source_half = np.concatenate([
                np.array([0.0]),
                half_arr,
                np.array([0.0]),
            ])

            source_order = np.argsort(source_y)
            source_y = source_y[source_order]
            source_half = source_half[source_order]

            # Remove duplicate Y values to keep np.interp stable.
            unique_y, unique_indices = np.unique(source_y, return_index=True)
            source_y = unique_y
            source_half = source_half[unique_indices]

            # Cosine spacing gives extra resolution near top and bottom closures.
            target_y = cosine_spaced_values(0.0, axis_length, requested_points)
            target_half = np.interp(target_y, source_y, source_half)

            # Force exact closure.
            target_half[0] = 0.0
            target_half[-1] = 0.0

            profile_xy = np.column_stack([target_half, target_y])
        else:
            target_y = np.linspace(float(y_arr[0]), float(y_arr[-1]), requested_points)
            target_half = np.interp(target_y, y_arr, half_arr)
            profile_xy = np.column_stack([target_half, target_y])

        return profile_xy

    def auto_break_and_fit(self, show_errors=True):
        if len(self.profile_points_px) < 3:
            if show_errors:
                messagebox.showwarning("Need profile points", "Extract or add profile points first.")
            raise ValueError("No profile points.")

        tolerance = self.safe_float_from_var(
            self.auto_arc_tolerance_var,
            default=2.5,
            minimum=0.05,
            maximum=1000.0,
        )

        max_arcs = self.safe_int_from_var(
            self.auto_max_arcs_var,
            default=6,
            minimum=1,
            maximum=20,
        )

        # Allow smaller segments so low point-count profiles, such as 25 points,
        # can still split near sharp top/bottom changes.
        min_points = max(4, int(len(self.profile_points_px) / max(max_arcs * 2, 1)))

        try:
            profile_xy = self.get_profile_xy_array()

            breakpoint_indices = auto_breakpoints_recursive(
                profile_xy,
                tolerance=tolerance,
                max_arcs=max_arcs,
                min_points_per_arc=min_points,
            )

            self.break_indices = set(breakpoint_indices)

            self.auto_breakpoint_report = AutoBreakpointReport(
                breakpoint_indices=breakpoint_indices,
                tolerance=tolerance,
                max_arcs=max_arcs,
                min_points_per_arc=min_points,
            )

            self.fit_arcs(show_warnings=False)

        except Exception as exc:
            if show_errors:
                messagebox.showerror("Auto break/fit failed", str(exc))
            raise

        self.update_axis_scale_info()
        self.update_results_text()
        self.redraw()

        if show_errors:
            self.status_var.set(
                f"Auto breakpoints set at {sorted(self.break_indices)}. "
                f"Fit {len(self.arc_results)} arc(s)."
            )

    def transform_points_180(self, points: List[np.ndarray], width: int, height: int) -> List[np.ndarray]:
        transformed = []

        for point in points:
            transformed.append(
                np.array(
                    [
                        (width - 1.0) - point[0],
                        (height - 1.0) - point[1],
                    ],
                    dtype=float
                )
            )

        return transformed

    def flip_up_direction(self):
        if self.image_pil is None:
            messagebox.showwarning("No image", "Load an image first.")
            return

        width, height = self.image_pil.size

        self.set_image(rotate_image_pil(self.image_pil, 180.0, expand=False))

        transformed_axis = self.transform_points_180(self.axis_points_px, width, height)

        if len(transformed_axis) == 2:
            transformed_axis = [transformed_axis[1], transformed_axis[0]]

        self.axis_points_px = transformed_axis
        self.scale_points_px = self.transform_points_180(self.scale_points_px, width, height)
        self.profile_points_px = self.transform_points_180(self.profile_points_px, width, height)

        self.image_rotation_degrees += 180.0

        self.arc_results.clear()

        if self.auto_orientation_report is not None:
            self.auto_orientation_report.used_180_flip = not self.auto_orientation_report.used_180_flip
            self.auto_orientation_report.rotation_degrees += 180.0

            if len(self.axis_points_px) == 2:
                self.auto_orientation_report.axis_top_px = self.axis_points_px[0]
                self.auto_orientation_report.axis_bottom_px = self.axis_points_px[1]

        self.update_axis_scale_info()
        self.update_results_text()
        self.redraw()
        self.status_var.set("Flipped up direction by rotating image 180°. Click Fit Arcs if needed.")

    def clear_axis(self):
        self.axis_points_px.clear()
        self.arc_results.clear()
        self.auto_orientation_report = None
        self.update_axis_scale_info()
        self.update_results_text()
        self.redraw()
        self.status_var.set("Axis cleared. Click TOP axis point, then BOTTOM axis point.")

    def clear_scale(self):
        self.scale_points_px.clear()
        self.scale_units = "px"
        self.scale_per_px = 1.0
        self.has_real_scale = False
        self.arc_results.clear()

        self.update_axis_scale_info()
        self.update_results_text()
        self.redraw()
        self.status_var.set("Scale cleared. Results will use pixels until a scale is set.")

    def clear_profile(self):
        self.profile_points_px.clear()
        self.break_indices.clear()
        self.arc_results.clear()
        self.auto_profile_report = None
        self.auto_breakpoint_report = None
        self.update_results_text()
        self.redraw()
        self.status_var.set("Profile cleared.")

    def undo_profile_point(self):
        if not self.profile_points_px:
            self.status_var.set("No profile points to undo.")
            return

        removed_index = len(self.profile_points_px) - 1
        self.profile_points_px.pop()

        self.break_indices = {index for index in self.break_indices if index != removed_index}
        self.break_indices = {index for index in self.break_indices if index < len(self.profile_points_px)}

        self.arc_results.clear()
        self.auto_profile_report = None
        self.auto_breakpoint_report = None

        self.update_results_text()
        self.redraw()
        self.status_var.set("Removed last profile point.")

    def on_mouse_click(self, event):
        if event.inaxes != self.ax_image:
            return

        if event.xdata is None or event.ydata is None:
            return

        point = np.array([float(event.xdata), float(event.ydata)])
        mode = self.mode_var.get()

        if mode == "axis":
            self.handle_axis_click(point)
        elif mode == "scale":
            self.handle_scale_click(point)
        elif mode == "profile":
            self.handle_profile_click(point)
        elif mode == "breakpoints":
            self.handle_breakpoint_click(point)

    def handle_axis_click(self, point):
        if self.image_array is None:
            messagebox.showwarning("No image", "Load an image first.")
            return

        if len(self.axis_points_px) >= 2:
            self.axis_points_px.clear()

        self.axis_points_px.append(point)
        self.arc_results.clear()
        self.auto_orientation_report = None
        self.auto_profile_report = None
        self.auto_breakpoint_report = None

        if len(self.axis_points_px) == 1:
            self.status_var.set("Top closure point set. Now click the BOTTOM closure point.")
        else:
            self.status_var.set("Axis set. Optional: set scale, then add/edit profile points.")
            self.mode_var.set("profile")

        self.update_axis_scale_info()
        self.update_results_text()
        self.redraw()

    def handle_scale_click(self, point):
        if self.image_array is None:
            messagebox.showwarning("No image", "Load an image first.")
            return

        if len(self.scale_points_px) >= 2:
            self.scale_points_px.clear()

        self.scale_points_px.append(point)
        self.arc_results.clear()

        if len(self.scale_points_px) == 1:
            self.status_var.set("Scale first point set. Now click the second known-distance point.")
        else:
            distance_px = float(np.linalg.norm(self.scale_points_px[1] - self.scale_points_px[0]))

            if distance_px < 1e-9:
                messagebox.showerror("Invalid scale", "The two scale points are too close together.")
                self.scale_points_px.clear()
                self.redraw()
                return

            real_distance = simpledialog.askfloat(
                "Set Scale",
                "Enter the real distance between those two points.\nExample: 43.04",
                minvalue=1e-9,
            )

            if real_distance is None:
                self.scale_points_px.clear()
                self.status_var.set("Scale entry cancelled.")
                self.redraw()
                return

            units = simpledialog.askstring(
                "Units",
                "Enter units label, usually mm:",
                initialvalue="mm"
            )

            if not units:
                units = "units"

            self.scale_units = units
            self.scale_per_px = real_distance / distance_px
            self.has_real_scale = True

            self.status_var.set(
                f"Scale set: {self.scale_per_px:.6f} {self.scale_units}/px. "
                "Click Fit Arcs again to update dimensions."
            )
            self.mode_var.set("profile")

        self.update_axis_scale_info()
        self.update_results_text()
        self.redraw()

    def handle_profile_click(self, point):
        if len(self.axis_points_px) != 2:
            messagebox.showwarning("Axis required", "Set the symmetry axis first.")
            self.mode_var.set("axis")
            return

        self.profile_points_px.append(point)
        self.arc_results.clear()
        self.auto_profile_report = None
        self.auto_breakpoint_report = None

        self.status_var.set(
            f"Added profile point {len(self.profile_points_px) - 1}. "
            "Continue along one side of the outline."
        )

        self.update_results_text()
        self.redraw()

    def handle_breakpoint_click(self, point):
        if len(self.profile_points_px) < 3:
            messagebox.showwarning("Need profile points", "Add at least 3 profile points first.")
            self.mode_var.set("profile")
            return

        points = np.array(self.profile_points_px)
        distances = np.linalg.norm(points - point, axis=1)
        nearest_index = int(np.argmin(distances))

        if nearest_index in self.break_indices:
            self.break_indices.remove(nearest_index)
            action = "Removed"
        else:
            self.break_indices.add(nearest_index)
            action = "Added"

        self.arc_results.clear()
        self.auto_breakpoint_report = None

        self.update_results_text()
        self.redraw()
        self.status_var.set(f"{action} breakpoint at profile point {nearest_index}.")

    def axis_top_bottom(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.axis_points_px) != 2:
            raise ValueError("Axis is not set.")

        return self.axis_points_px[0], self.axis_points_px[1]

    def axis_basis(self):
        top, bottom = self.axis_top_bottom()

        axis_vector = bottom - top
        axis_length_px = float(np.linalg.norm(axis_vector))

        if axis_length_px < 1e-9:
            raise ValueError("Axis points are too close together.")

        axis_dir = axis_vector / axis_length_px

        axis_perp = np.array([axis_dir[1], -axis_dir[0]])

        return top, bottom, axis_dir, axis_perp, axis_length_px

    def profile_side_sign(self) -> float:
        if len(self.axis_points_px) != 2 or not self.profile_points_px:
            return 1.0

        top, _, _, axis_perp, _ = self.axis_basis()

        signed_values = [
            float(np.dot(point - top, axis_perp))
            for point in self.profile_points_px
        ]

        average = float(np.mean(signed_values))

        if average < 0:
            return -1.0

        return 1.0

    def image_point_to_profile_xy(self, point_px: np.ndarray) -> Tuple[float, float]:
        top, _, axis_dir, axis_perp, axis_length_px = self.axis_basis()
        side_sign = self.profile_side_sign()

        relative = point_px - top

        s_px = float(np.dot(relative, axis_dir))
        signed_radial_px = float(np.dot(relative, axis_perp))

        x = side_sign * signed_radial_px * self.scale_per_px
        y = (axis_length_px - s_px) * self.scale_per_px

        return x, y

    def profile_xy_to_image_point(self, x: float, y: float) -> np.ndarray:
        top, _, axis_dir, axis_perp, axis_length_px = self.axis_basis()
        side_sign = self.profile_side_sign()

        s_px = axis_length_px - (y / self.scale_per_px)
        radial_px = (x / self.scale_per_px) * side_sign

        return top + axis_dir * s_px + axis_perp * radial_px

    def get_profile_xy_array(self) -> np.ndarray:
        if len(self.axis_points_px) != 2:
            raise ValueError("Axis is not set.")

        coords = [
            self.image_point_to_profile_xy(point)
            for point in self.profile_points_px
        ]

        return np.array(coords, dtype=float)

    def get_segment_ranges(self) -> List[Tuple[int, int]]:
        point_count = len(self.profile_points_px)

        if point_count < 2:
            return []

        valid_breaks = sorted(
            index for index in self.break_indices
            if 0 <= index < point_count
        )

        boundaries = [0]

        for index in valid_breaks:
            if index not in boundaries:
                boundaries.append(index)

        if point_count - 1 not in boundaries:
            boundaries.append(point_count - 1)

        boundaries = sorted(set(boundaries))

        ranges = []

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end > start:
                ranges.append((start, end))

        return ranges

    def fit_arcs(self, show_warnings=True):
        if len(self.axis_points_px) != 2:
            if show_warnings:
                messagebox.showwarning("Axis required", "Set the symmetry axis first.")
            raise ValueError("Axis required.")

        if len(self.profile_points_px) < 3:
            if show_warnings:
                messagebox.showwarning("Need more points", "Add at least 3 profile points.")
            raise ValueError("Need at least 3 profile points.")

        try:
            profile_xy = self.get_profile_xy_array()
        except Exception as exc:
            if show_warnings:
                messagebox.showerror("Coordinate error", str(exc))
            raise

        ranges = self.get_segment_ranges()
        results = []

        for segment_number, (start_index, end_index) in enumerate(ranges, start=1):
            segment_points = profile_xy[start_index:end_index + 1]

            if len(segment_points) < 3:
                continue

            try:
                center, radius, errors = fit_circle_endpoint_constrained(segment_points)
            except Exception as exc:
                if show_warnings:
                    messagebox.showwarning(
                        "Arc fit skipped",
                        f"Segment {segment_number} could not be fit:\n{exc}"
                    )
                continue

            angles = unwrap_segment_angles(segment_points, center)
            theta_start = float(angles[0])
            theta_end = float(angles[-1])

            included_angle_rad = theta_end - theta_start
            included_angle_degrees = math.degrees(included_angle_rad)

            direction = "CCW" if included_angle_degrees >= 0 else "CW"
            arc_length = abs(radius * included_angle_rad)

            start_point = segment_points[0]
            end_point = segment_points[-1]
            chord_length = float(np.linalg.norm(end_point - start_point))

            rms_error = float(math.sqrt(np.mean(errors * errors)))
            max_error = float(np.max(np.abs(errors)))

            result = ArcResult(
                segment_number=segment_number,
                start_index=start_index,
                end_index=end_index,
                point_count=len(segment_points),
                center_x=float(center[0]),
                center_y=float(center[1]),
                mirrored_center_x=float(-center[0]),
                mirrored_center_y=float(center[1]),
                radius=float(radius),
                theta_start_rad=theta_start,
                theta_end_rad=theta_end,
                included_angle_degrees=float(included_angle_degrees),
                direction=direction,
                arc_length=float(arc_length),
                chord_length=chord_length,
                rms_error=rms_error,
                max_error=max_error,
                start_x=float(start_point[0]),
                start_y=float(start_point[1]),
                end_x=float(end_point[0]),
                end_y=float(end_point[1]),
            )

            results.append(result)

        self.arc_results = results
        self.update_results_text()
        self.redraw()

        if show_warnings:
            if results:
                self.status_var.set(f"Fit {len(results)} arc(s).")
            else:
                self.status_var.set("No arcs fit. Add more points or adjust breakpoints.")

    def result_arc_points(self, result: ArcResult, sample_count=100) -> np.ndarray:
        theta_values = np.linspace(result.theta_start_rad, result.theta_end_rad, sample_count)

        x = result.center_x + result.radius * np.cos(theta_values)
        y = result.center_y + result.radius * np.sin(theta_values)

        return np.column_stack([x, y])

    def redraw(self):
        self.ax_image.clear()
        self.ax_profile.clear()

        self.draw_image_view()
        self.draw_profile_view()

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _draw_empty(self):
        self.ax_image.clear()
        self.ax_profile.clear()

        self.ax_image.set_title("Image")
        self.ax_image.text(0.5, 0.5, "Load an image", ha="center", va="center", transform=self.ax_image.transAxes)
        self.ax_image.set_axis_off()

        self.ax_profile.set_title("Mirrored Profile")
        self.ax_profile.set_xlabel("X")
        self.ax_profile.set_ylabel("Y")
        self.ax_profile.grid(True)

        self.canvas.draw_idle()

    def draw_image_view(self):
        self.ax_image.set_title("Image Trace View")

        if self.image_array is None:
            self.ax_image.text(0.5, 0.5, "Load an image", ha="center", va="center", transform=self.ax_image.transAxes)
            self.ax_image.set_axis_off()
            return

        self.ax_image.imshow(self.image_array)
        self.ax_image.set_axis_on()

        if self.auto_orientation_report is not None and self.auto_orientation_report.component_bbox_px is not None:
            x0, y0, x1, y1 = self.auto_orientation_report.component_bbox_px
            self.ax_image.plot(
                [x0, x1, x1, x0, x0],
                [y0, y0, y1, y1, y0],
                linestyle=":",
                linewidth=1.5
            )

        if len(self.axis_points_px) >= 1:
            pts = np.array(self.axis_points_px)
            self.ax_image.scatter(pts[:, 0], pts[:, 1], s=45, marker="x")

        if len(self.axis_points_px) == 2:
            top, bottom = self.axis_top_bottom()
            self.ax_image.plot([top[0], bottom[0]], [top[1], bottom[1]], linewidth=2)
            self.ax_image.text(top[0], top[1], " axis top", fontsize=9)
            self.ax_image.text(bottom[0], bottom[1], " axis bottom", fontsize=9)

        if len(self.scale_points_px) >= 1:
            pts = np.array(self.scale_points_px)
            self.ax_image.scatter(pts[:, 0], pts[:, 1], s=35, marker="s")

            if len(self.scale_points_px) == 2:
                a, b = self.scale_points_px
                self.ax_image.plot([a[0], b[0]], [a[1], b[1]], linestyle="--", linewidth=2)
                self.ax_image.text(
                    (a[0] + b[0]) / 2.0,
                    (a[1] + b[1]) / 2.0,
                    " scale",
                    fontsize=9
                )

        if self.profile_points_px:
            pts = np.array(self.profile_points_px)
            self.ax_image.plot(pts[:, 0], pts[:, 1], marker="o", markersize=3, linewidth=1.5)

            if self.show_point_numbers_var.get():
                for index, point in enumerate(pts):
                    self.ax_image.text(point[0], point[1], str(index), fontsize=8)

            if self.break_indices:
                break_pts = np.array([
                    self.profile_points_px[index]
                    for index in sorted(self.break_indices)
                    if 0 <= index < len(self.profile_points_px)
                ])

                if len(break_pts):
                    self.ax_image.scatter(break_pts[:, 0], break_pts[:, 1], s=65, marker="s")

        if self.show_fitted_arcs_var.get() and len(self.axis_points_px) == 2:
            for result in self.arc_results:
                arc_xy = self.result_arc_points(result, sample_count=100)
                arc_px = np.array([
                    self.profile_xy_to_image_point(x, y)
                    for x, y in arc_xy
                ])

                self.ax_image.plot(arc_px[:, 0], arc_px[:, 1], linewidth=2.5)

                if self.show_mirror_var.get():
                    mirrored_arc_xy = arc_xy.copy()
                    mirrored_arc_xy[:, 0] *= -1.0

                    mirrored_arc_px = np.array([
                        self.profile_xy_to_image_point(x, y)
                        for x, y in mirrored_arc_xy
                    ])

                    self.ax_image.plot(mirrored_arc_px[:, 0], mirrored_arc_px[:, 1], linewidth=1.5, linestyle="--")

        self.ax_image.set_xlim(0, self.image_array.shape[1])
        self.ax_image.set_ylim(self.image_array.shape[0], 0)

    def draw_profile_view(self):
        unit = self.scale_units
        self.ax_profile.set_title("Profile Coordinates / Mirrored Shape")
        self.ax_profile.set_xlabel(f"X ({unit})")
        self.ax_profile.set_ylabel(f"Y ({unit})")
        self.ax_profile.grid(True)

        if len(self.axis_points_px) != 2:
            self.ax_profile.text(
                0.5,
                0.5,
                "Set the symmetry axis first",
                ha="center",
                va="center",
                transform=self.ax_profile.transAxes
            )
            return

        if not self.profile_points_px:
            self.ax_profile.text(
                0.5,
                0.5,
                "Auto-extract or click points along one side of the outline",
                ha="center",
                va="center",
                transform=self.ax_profile.transAxes
            )
            return

        profile_xy = self.get_profile_xy_array()

        self.ax_profile.plot(
            profile_xy[:, 0],
            profile_xy[:, 1],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label="profile side"
        )

        if self.show_mirror_var.get():
            self.ax_profile.plot(
                -profile_xy[:, 0],
                profile_xy[:, 1],
                marker="o",
                markersize=2,
                linewidth=1.0,
                linestyle="--",
                label="mirror"
            )

        if self.break_indices:
            valid = [
                index for index in sorted(self.break_indices)
                if 0 <= index < len(profile_xy)
            ]

            if valid:
                break_xy = profile_xy[valid]
                self.ax_profile.scatter(break_xy[:, 0], break_xy[:, 1], s=55, marker="s")

        if self.show_point_numbers_var.get():
            for index, point in enumerate(profile_xy):
                self.ax_profile.text(point[0], point[1], str(index), fontsize=8)

        if self.show_fitted_arcs_var.get():
            for result in self.arc_results:
                arc_xy = self.result_arc_points(result, sample_count=120)

                self.ax_profile.plot(
                    arc_xy[:, 0],
                    arc_xy[:, 1],
                    linewidth=2.5,
                    label=f"arc {result.segment_number}"
                )

                if self.show_mirror_var.get():
                    mirrored_arc_xy = arc_xy.copy()
                    mirrored_arc_xy[:, 0] *= -1.0

                    self.ax_profile.plot(
                        mirrored_arc_xy[:, 0],
                        mirrored_arc_xy[:, 1],
                        linewidth=1.5,
                        linestyle="--"
                    )

                self.ax_profile.scatter([result.center_x], [result.center_y], marker="+", s=60)

                if self.show_mirror_var.get():
                    self.ax_profile.scatter([result.mirrored_center_x], [result.mirrored_center_y], marker="+", s=45)

        self.ax_profile.axvline(0.0, linestyle="-", linewidth=1.0)

        all_xy = [profile_xy]

        if self.show_mirror_var.get():
            mirrored = profile_xy.copy()
            mirrored[:, 0] *= -1.0
            all_xy.append(mirrored)

        combined = np.vstack(all_xy)
        mins = combined.min(axis=0)
        maxs = combined.max(axis=0)

        width = max(maxs[0] - mins[0], 1.0)
        height = max(maxs[1] - mins[1], 1.0)

        pad_x = width * 0.15
        pad_y = height * 0.10

        self.ax_profile.set_xlim(mins[0] - pad_x, maxs[0] + pad_x)
        self.ax_profile.set_ylim(mins[1] - pad_y, maxs[1] + pad_y)
        self.ax_profile.set_aspect("equal", adjustable="box")

        try:
            self.ax_profile.legend(loc="best", fontsize=8)
        except Exception:
            pass

    def update_axis_scale_info(self):
        if len(self.axis_points_px) == 2:
            top, bottom = self.axis_top_bottom()
            axis_length_px = float(np.linalg.norm(bottom - top))

            if self.has_real_scale:
                axis_length_real = axis_length_px * self.scale_per_px
                self.axis_info_var.set(
                    f"Axis: {axis_length_px:.2f} px / {axis_length_real:.4f} {self.scale_units}"
                )
            else:
                self.axis_info_var.set(f"Axis: {axis_length_px:.2f} px")
        else:
            self.axis_info_var.set("Axis: not set")

        if self.has_real_scale:
            self.scale_info_var.set(
                f"Scale: {self.scale_per_px:.8f} {self.scale_units}/px"
            )
        else:
            self.scale_info_var.set("Scale: not set; using pixels")

        if self.auto_orientation_report is not None:
            flip_text = ", includes up flip" if self.auto_orientation_report.used_180_flip else ""
            self.orientation_info_var.set(
                f"Image orientation: {self.auto_orientation_report.rotation_degrees:.3f}°{flip_text}"
            )
        elif abs(self.image_rotation_degrees) > 1e-9:
            self.orientation_info_var.set(f"Image orientation: {self.image_rotation_degrees:.3f}°")
        else:
            self.orientation_info_var.set("Image orientation: not applied")

        if self.auto_profile_report is not None:
            self.profile_info_var.set(
                f"Auto profile: {self.auto_profile_report.target}, {self.auto_profile_report.point_count} pts, "
                f"max half-width {self.auto_profile_report.max_half_width:.2f} {self.scale_units}"
            )
        elif self.profile_points_px:
            self.profile_info_var.set(f"Profile: {len(self.profile_points_px)} manual/edited pts")
        else:
            self.profile_info_var.set("Auto profile: not extracted")

    def update_results_text(self):
        text = self.results_summary_text()

        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert("1.0", text)
        self.results_text.configure(state="disabled")

    def results_summary_text(self) -> str:
        unit = self.scale_units

        lines = []
        lines.append("Arc Profile Finder Results")
        lines.append("=" * 32)
        lines.append("")

        if self.image_path:
            lines.append(f"Image: {os.path.basename(self.image_path)}")
        else:
            lines.append("Image: not loaded")

        lines.append(f"Units: {unit}")

        if self.has_real_scale:
            lines.append(f"Scale: {self.scale_per_px:.8f} {unit}/px")
        else:
            lines.append("Scale: not set; values are pixels")

        lines.append(f"Profile points: {len(self.profile_points_px)}")
        lines.append(f"Breakpoints: {sorted(self.break_indices)}")
        lines.append("")

        if self.auto_orientation_report is not None:
            lines.append(self.auto_orientation_report.text())
            lines.append("")

        if self.auto_profile_report is not None:
            lines.append(self.auto_profile_report.text())
            lines.append("")

        if self.auto_breakpoint_report is not None:
            lines.append(self.auto_breakpoint_report.text())
            lines.append("")

        if not self.arc_results:
            lines.append("No arc results yet.")
            lines.append("")
            lines.append("Load an image with auto-process enabled, or click Auto Process All.")
            return "\n".join(lines)

        for result in self.arc_results:
            lines.append(f"Arc {result.segment_number}")
            lines.append("-" * 32)
            lines.append(f"Point range: {result.start_index} -> {result.end_index}")
            lines.append(f"Point count: {result.point_count}")
            lines.append(f"Radius: {result.radius:.6f} {unit}")
            lines.append(f"Center: ({result.center_x:.6f}, {result.center_y:.6f}) {unit}")
            lines.append(
                f"Mirrored center: ({result.mirrored_center_x:.6f}, "
                f"{result.mirrored_center_y:.6f}) {unit}"
            )
            lines.append(f"Start point: ({result.start_x:.6f}, {result.start_y:.6f}) {unit}")
            lines.append(f"End point: ({result.end_x:.6f}, {result.end_y:.6f}) {unit}")
            lines.append(f"Included angle: {result.included_angle_degrees:.6f} deg")
            lines.append(f"Direction: {result.direction}")
            lines.append(f"Arc length: {result.arc_length:.6f} {unit}")
            lines.append(f"Chord length: {result.chord_length:.6f} {unit}")
            lines.append(f"RMS fit error: {result.rms_error:.6f} {unit}")
            lines.append(f"Max fit error: {result.max_error:.6f} {unit}")
            lines.append("")

        return "\n".join(lines)

    def copy_results(self):
        text = self.results_summary_text()
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_var.set("Copied results to clipboard.")

    def export_csv(self):
        if not self.arc_results:
            messagebox.showwarning("No results", "Fit arcs before exporting.")
            return

        if self.image_path:
            initial_dir = os.path.dirname(self.image_path)
            base = os.path.splitext(os.path.basename(self.image_path))[0]
            initial_file = f"{base}_arc_results.csv"
        else:
            initial_dir = os.getcwd()
            initial_file = "arc_results.csv"

        path = filedialog.asksaveasfilename(
            title="Export arc results",
            initialdir=initial_dir,
            initialfile=initial_file,
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ]
        )

        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)

                writer.writerow(["image", self.image_path or ""])
                writer.writerow(["units", self.scale_units])
                writer.writerow(["scale_per_px", self.scale_per_px])
                writer.writerow(["has_real_scale", self.has_real_scale])

                if self.auto_orientation_report is not None:
                    writer.writerow(["auto_rotation_degrees", self.auto_orientation_report.rotation_degrees])
                    writer.writerow(["auto_used_180_flip", self.auto_orientation_report.used_180_flip])
                    writer.writerow(["auto_detected_area_px", self.auto_orientation_report.detected_area_px])
                    writer.writerow(["auto_top_width_px", self.auto_orientation_report.top_width_px])
                    writer.writerow(["auto_bottom_width_px", self.auto_orientation_report.bottom_width_px])
                    writer.writerow(["auto_threshold_value", self.auto_orientation_report.threshold_value])
                    writer.writerow(["auto_component_score", self.auto_orientation_report.component_score])
                    writer.writerow(["auto_detector_note", self.auto_orientation_report.detector_note])

                if self.auto_profile_report is not None:
                    writer.writerow(["auto_profile_target", self.auto_profile_report.target])
                    writer.writerow(["auto_profile_locked_manual_closures", self.auto_profile_report.locked_manual_closures])
                    writer.writerow(["auto_profile_edge_sensitivity", self.auto_profile_report.edge_sensitivity])
                    writer.writerow(["auto_profile_outer_search_scale", self.auto_profile_report.outer_search_scale])
                    writer.writerow(["auto_profile_smooth_window", self.auto_profile_report.smooth_window])
                    writer.writerow(["auto_profile_point_count", self.auto_profile_report.point_count])
                    writer.writerow(["auto_profile_max_half_width", self.auto_profile_report.max_half_width])

                if self.auto_breakpoint_report is not None:
                    writer.writerow(["auto_breakpoints", ",".join(str(i) for i in self.auto_breakpoint_report.breakpoint_indices)])
                    writer.writerow(["auto_arc_tolerance", self.auto_breakpoint_report.tolerance])
                    writer.writerow(["auto_max_arcs", self.auto_breakpoint_report.max_arcs])

                writer.writerow([])

                writer.writerow([
                    "segment_number",
                    "start_index",
                    "end_index",
                    "point_count",
                    "radius",
                    "center_x",
                    "center_y",
                    "mirrored_center_x",
                    "mirrored_center_y",
                    "start_x",
                    "start_y",
                    "end_x",
                    "end_y",
                    "included_angle_degrees",
                    "direction",
                    "arc_length",
                    "chord_length",
                    "rms_error",
                    "max_error",
                ])

                for result in self.arc_results:
                    writer.writerow([
                        result.segment_number,
                        result.start_index,
                        result.end_index,
                        result.point_count,
                        result.radius,
                        result.center_x,
                        result.center_y,
                        result.mirrored_center_x,
                        result.mirrored_center_y,
                        result.start_x,
                        result.start_y,
                        result.end_x,
                        result.end_y,
                        result.included_angle_degrees,
                        result.direction,
                        result.arc_length,
                        result.chord_length,
                        result.rms_error,
                        result.max_error,
                    ])

                if self.profile_points_px and len(self.axis_points_px) == 2:
                    writer.writerow([])
                    writer.writerow(["profile_points"])
                    writer.writerow(["index", "x", "y"])

                    profile_xy = self.get_profile_xy_array()

                    for index, point in enumerate(profile_xy):
                        writer.writerow([index, point[0], point[1]])

        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return

        self.status_var.set(f"Exported CSV: {path}")
        messagebox.showinfo("Export complete", f"Exported:\n{path}")


def main():
    root = tk.Tk()
    app = ArcProfileFinderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
