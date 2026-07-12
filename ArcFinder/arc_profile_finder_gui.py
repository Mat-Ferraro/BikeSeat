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
    - The user can tune outer-edge sensitivity, search distance, and smoothing per image.
    - Arc fitting always produces 8 arcs.
    - The widest point is always forced to be the start of one arc segment.
    - Arc distribution can be chosen from four rules.
    - Optional part length/width inputs scale the output coordinates to mm or inches.
    - Controls are split into tabs for smaller screens.
    - Quick Instructions and the whole side panel can be collapsed to free screen space.
    - The entire right-side info/results column can be hidden to give the plot more horizontal space.
    - The side panel uses tabs for Instructions, Full Results, and an 8-Arc Summary.
    - Arc/point numbering is top-to-bottom: P0 is the top point, P8 is the bottom point.
    - Manual point placement is disabled; use the matplotlib toolbar to pan/zoom.
    - Hover over profile points to see coordinates, or over arcs to see radius.
    - Point hover hit boxes are sized close to the visible point marker.
    - Arc boundary points P0..P8 are drawn as square markers, including the true top and bottom points.
    - Only P0..P8 show X/Y on hover; intermediate profile points show the arc radius.
    - Edge offset tuning can expand/contract the detected edge by up to 50 px for better accuracy.
    - Left-side, right-side, and averaged 8-arc summaries can be compared directly.
    - The plot area has two tabs: Left/Right and Average. Each tab includes the image trace plus the coordinate graph.
    - Mouse wheel zoom is enabled directly on the plots.
    - Left-click and drag pans the current plot.
    - Images no longer auto-process immediately after loading; press Auto Process All when ready.
    - Tuning sliders are stacked vertically so they do not run off-screen.
    - Manual Editing controls are split across two rows.

Important limitation:
    The app cannot know real-world scale from a photo alone unless the image includes
    a known reference dimension entered as Part Length / Part Width. Without scale,
    results are in pixels.

Install:
    python -m pip install numpy matplotlib pillow

Run:
    python arc_profile_finder_gui.py

Recommended workflow:
    1. Load Image.
       If "Auto process manually" is checked, the app will attempt the whole process.
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
        Automatically picks breakpoints and fits exactly 8 circular arcs.
        The widest point is always a segment start.

    Flip Up Direction:
        Rotates the current image 180 degrees and updates the annotations.

Notes:
    - This detector is meant for the INNER dark opening.
    - If you need the OUTER silver rim profile instead, that is a different detector.
    - If detection is imperfect, tune the extraction settings and rebuild.
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


ARC_COUNT = 8
ARC_DISTRIBUTION_OPTIONS = [
    "Half above midpoint",
    "Half above widest point",
    "Unrestricted",
]
DIMENSION_UNIT_OPTIONS = ["mm", "in"]
ARC_SUMMARY_UNIT_OPTIONS = ["px", "mm", "in"]


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
    edge_offset_px: float = 0.0
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
                f"Edge offset: {self.edge_offset_px:.3f} px",
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
    distribution_rule: str = "Unrestricted"
    widest_index: int = -1

    def text(self) -> str:
        return "\n".join(
            [
                "Auto Breakpoint Detection",
                "=" * 32,
                f"Breakpoints: {self.breakpoint_indices}",
                f"Distribution rule: {self.distribution_rule}",
                f"Widest point index: {self.widest_index}",
                f"Tolerance: {self.tolerance:.3f}",
                f"Arc count: {self.max_arcs}",
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


def nearest_index_by_y(points_xy: np.ndarray, target_y: float) -> int:
    if len(points_xy) == 0:
        return 0
    distances = np.abs(points_xy[:, 1] - target_y)
    return int(np.argmin(distances))


def widest_profile_index(points_xy: np.ndarray) -> int:
    """
    Return the index of the widest one-side profile point.

    The extracted profile is stored as a single side, where +X is the traced
    half-width from the symmetry axis. The widest full profile occurs at the
    largest +X half-width.
    """
    if len(points_xy) == 0:
        return 0

    return int(np.argmax(points_xy[:, 0]))


def normalize_arc_boundaries(
    boundaries: List[int],
    point_count: int,
    required_count: int,
    forced_indices: Optional[List[int]] = None,
) -> List[int]:
    """
    Normalize a list of boundary indices so it has exactly required_count items.

    Boundary 0 and point_count - 1 are always preserved. Forced interior
    boundaries, such as the widest point, are preserved whenever possible.
    """
    if point_count < 2:
        return [0]

    forced = set(forced_indices or [])
    forced = {int(i) for i in forced if 0 <= int(i) < point_count}

    boundary_set = {0, point_count - 1}
    boundary_set.update(int(i) for i in boundaries if 0 <= int(i) < point_count)
    boundary_set.update(forced)

    boundaries = sorted(boundary_set)

    while len(boundaries) > required_count:
        removable = [b for b in boundaries[1:-1] if b not in forced]

        if not removable:
            break

        best_remove = None
        best_cost = None

        for b in removable:
            idx = boundaries.index(b)
            left_gap = b - boundaries[idx - 1]
            right_gap = boundaries[idx + 1] - b
            cost = min(left_gap, right_gap)

            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_remove = b

        if best_remove is None:
            break

        boundaries.remove(best_remove)

    while len(boundaries) < required_count:
        best_gap = None
        best_pair = None

        for a, b in zip(boundaries[:-1], boundaries[1:]):
            gap = b - a
            if gap <= 1:
                continue
            if best_gap is None or gap > best_gap:
                best_gap = gap
                best_pair = (a, b)

        if best_pair is None:
            break

        a, b = best_pair
        new_boundary = (a + b) // 2

        if new_boundary in boundaries:
            candidates = [i for i in range(a + 1, b) if i not in boundaries]
            if not candidates:
                break
            new_boundary = candidates[len(candidates) // 2]

        boundaries.append(new_boundary)
        boundaries = sorted(set(boundaries))

    return sorted(boundaries)


def boundaries_from_y_targets(points_xy: np.ndarray, y_targets: np.ndarray) -> List[int]:
    indices = [nearest_index_by_y(points_xy, float(y)) for y in y_targets]
    return sorted(set(indices))


def choose_eight_arc_boundaries(
    points_xy: np.ndarray,
    rule: str,
    tolerance: float,
    min_points_per_arc: int,
) -> Tuple[List[int], int]:
    """
    Choose exactly 9 boundary indices for 8 arc segments.

    The widest point is always forced into the boundary list, meaning one arc
    segment starts at the widest point.
    """
    point_count = len(points_xy)

    if point_count < ARC_COUNT + 1:
        raise ValueError(f"Need at least {ARC_COUNT + 1} profile points for {ARC_COUNT} arcs.")

    widest_index = widest_profile_index(points_xy)
    y_min = float(points_xy[0, 1])
    y_max = float(points_xy[-1, 1])

    if y_max < y_min:
        y_min = float(np.min(points_xy[:, 1]))
        y_max = float(np.max(points_xy[:, 1]))

    if abs(y_max - y_min) < 1e-9:
        raise ValueError("Profile Y span is too small for arc distribution.")

    rule = rule if rule in ARC_DISTRIBUTION_OPTIONS else "Unrestricted"

    if rule == "Half above midpoint":
        split_y = (y_min + y_max) / 2.0
        lower = np.linspace(y_min, split_y, ARC_COUNT // 2 + 1)
        upper = np.linspace(split_y, y_max, ARC_COUNT // 2 + 1)[1:]
        boundaries = boundaries_from_y_targets(points_xy, np.concatenate([lower, upper]))

    elif rule == "Half above widest point":
        split_y = float(points_xy[widest_index, 1])
        span = y_max - y_min
        split_y = max(y_min + span * 0.05, min(y_max - span * 0.05, split_y))
        lower = np.linspace(y_min, split_y, ARC_COUNT // 2 + 1)
        upper = np.linspace(split_y, y_max, ARC_COUNT // 2 + 1)[1:]
        boundaries = boundaries_from_y_targets(points_xy, np.concatenate([lower, upper]))

    elif rule == "Evenly spaced":
        boundaries = boundaries_from_y_targets(points_xy, np.linspace(y_min, y_max, ARC_COUNT + 1))

    else:
        breakpoints = auto_breakpoints_recursive(
            points_xy,
            tolerance=tolerance,
            max_arcs=ARC_COUNT,
            min_points_per_arc=min_points_per_arc,
        )
        boundaries = [0] + breakpoints + [point_count - 1]

    boundaries = normalize_arc_boundaries(
        boundaries,
        point_count=point_count,
        required_count=ARC_COUNT + 1,
        forced_indices=[widest_index],
    )

    if widest_index not in boundaries and 0 < widest_index < point_count - 1:
        boundaries.append(widest_index)
        boundaries = normalize_arc_boundaries(
            boundaries,
            point_count=point_count,
            required_count=ARC_COUNT + 1,
            forced_indices=[widest_index],
        )

    return boundaries, widest_index


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
        self.right_profile_base_xy: Optional[np.ndarray] = None
        self.left_profile_base_xy: Optional[np.ndarray] = None
        self.average_profile_base_xy: Optional[np.ndarray] = None
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
        self.show_profile_points_var = tk.BooleanVar(value=True)
        self.show_fitted_arcs_var = tk.BooleanVar(value=True)
        self.show_point_numbers_var = tk.BooleanVar(value=False)
        self.auto_process_on_load_var = tk.BooleanVar(value=False)
        self.close_ends_var = tk.BooleanVar(value=True)
        self.force_convex_arcs_var = tk.BooleanVar(value=True)

        self.profile_target_var = tk.StringVar(value="Outer outside edge")
        self.lock_manual_closures_var = tk.BooleanVar(value=False)

        self.auto_point_count_var = tk.StringVar(value="90")
        self.auto_arc_tolerance_var = tk.StringVar(value="2.5")
        self.arc_distribution_var = tk.StringVar(value="Unrestricted")

        self.part_length_var = tk.StringVar(value="")
        self.part_width_var = tk.StringVar(value="")
        self.dimension_units_var = tk.StringVar(value="mm")
        self.arc_summary_units_var = tk.StringVar(value="px")

        self.right_panel_visible = True
        self.right_panel = None
        self.toggle_side_panel_button = None

        self.hover_annotation = None
        self.average_hover_annotation = None
        self.profile_hover_points = []
        self.profile_hover_arcs = []
        self.average_hover_points = []
        self.average_hover_arcs = []

        self.point_adjust_mode_var = tk.BooleanVar(value=False)
        self.nudge_step_px_var = tk.StringVar(value="1")
        self.selected_boundary_index: Optional[int] = None
        self.selected_boundary_side: Optional[str] = None
        self.selected_boundary_label_var = tk.StringVar(value="Selected: none")

        self.is_panning = False
        self.pan_axes = None
        self.pan_start_xy = None
        self.pan_start_event_xy = None
        self.pan_start_xlim = None
        self.pan_start_ylim = None
        self.pan_data_per_pixel_x = None
        self.pan_data_per_pixel_y = None

        self.outer_edge_sensitivity_var = tk.DoubleVar(value=0.28)
        self.outer_search_scale_var = tk.DoubleVar(value=1.0)
        self.smooth_window_var = tk.DoubleVar(value=5.0)
        self.edge_offset_px_var = tk.DoubleVar(value=0.0)

        self._build_ui()
        self._bind_shortcuts()

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
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

        self.left_frame = left_frame
        self.right_panel = right_frame

        self.root.columnconfigure(0, weight=7)
        self.root.columnconfigure(1, weight=2)

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
        ttk.Button(frame, text="Flip Up Direction", command=self.flip_up_direction).pack(side=tk.LEFT, padx=3)
        ttk.Button(frame, text="Export 8-Arc CSV", command=self.export_arc_summary_csv).pack(side=tk.LEFT, padx=3)

        self.toggle_side_panel_button = ttk.Button(
            frame,
            text="Hide Side Panel",
            command=self.toggle_side_panel,
        )
        self.toggle_side_panel_button.pack(side=tk.LEFT, padx=(14, 3))

    def toggle_side_panel(self):
        """
        Hide/show the entire right-side panel.

        This is different from collapsing only Quick Instructions. When hidden,
        the right column is removed from the grid and the plot/workspace gets
        the full horizontal space.
        """
        if self.right_panel is None:
            return

        if self.right_panel_visible:
            self.right_panel.grid_remove()
            self.root.columnconfigure(1, weight=0, minsize=0)
            self.right_panel_visible = False

            if self.toggle_side_panel_button is not None:
                self.toggle_side_panel_button.configure(text="Show Side Panel")
        else:
            self.right_panel.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=5)
            self.root.columnconfigure(1, weight=2)
            self.right_panel_visible = True

            if self.toggle_side_panel_button is not None:
                self.toggle_side_panel_button.configure(text="Hide Side Panel")

        # Give matplotlib a moment to resize/redraw into the new available space.
        self.root.update_idletasks()
        self.redraw()

    def _build_auto_options_bar(self, parent):
        """
        Build auto controls in tabs.

        This is more reliable on smaller screens than trying to keep every
        option visible in one horizontal row. The controls are grouped by task:

            Core:
                target, profile points, arc tolerance, max arcs, closure options

            Tuning:
                edge sensitivity, outer search distance, smoothing

                    """
        frame = ttk.LabelFrame(parent, text="Auto Options")
        frame.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        frame.columnconfigure(0, weight=1)

        notebook = ttk.Notebook(frame)
        notebook.grid(row=0, column=0, sticky="ew", padx=4, pady=4)

        core_tab = ttk.Frame(notebook)
        tuning_tab = ttk.Frame(notebook)

        notebook.add(core_tab, text="Core")
        notebook.add(tuning_tab, text="Tuning")

        self._build_core_options_tab(core_tab)
        self._build_tuning_options_tab(tuning_tab)

    def _build_core_options_tab(self, parent):
        parent.columnconfigure(0, weight=1)

        row1 = ttk.Frame(parent)
        row1.grid(row=0, column=0, sticky="ew", padx=4, pady=(5, 2))

        row2 = ttk.Frame(parent)
        row2.grid(row=1, column=0, sticky="ew", padx=4, pady=(2, 2))

        row3 = ttk.Frame(parent)
        row3.grid(row=2, column=0, sticky="ew", padx=4, pady=(2, 2))

        row4 = ttk.Frame(parent)
        row4.grid(row=3, column=0, sticky="ew", padx=4, pady=(2, 5))

        ttk.Checkbutton(
            row1,
            text="Close ends to axis",
            variable=self.close_ends_var
        ).pack(side=tk.LEFT, padx=12)

        ttk.Checkbutton(
            row1,
            text="Force convex arcs",
            variable=self.force_convex_arcs_var,
            command=self.on_force_convex_arcs_changed
        ).pack(side=tk.LEFT, padx=12)

        # Manual closure locking is intentionally hidden in this build.
        # The app is auto-detection + tuning only; image clicking is disabled.

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

        ttk.Label(row2, text="Arc distribution:").pack(side=tk.LEFT, padx=(8, 2))
        distribution_combo = ttk.Combobox(
            row2,
            textvariable=self.arc_distribution_var,
            values=ARC_DISTRIBUTION_OPTIONS,
            width=23,
            state="readonly",
        )
        distribution_combo.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(row2, text=f"Arc count: {ARC_COUNT}").pack(side=tk.LEFT, padx=(8, 2))

        ttk.Label(row3, text="Part length:").pack(side=tk.LEFT, padx=(4, 2))
        ttk.Entry(row3, textvariable=self.part_length_var, width=10).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(row3, text="Part width:").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Entry(row3, textvariable=self.part_width_var, width=10).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(row3, text="Units:").pack(side=tk.LEFT, padx=(8, 2))
        units_combo = ttk.Combobox(
            row3,
            textvariable=self.dimension_units_var,
            values=DIMENSION_UNIT_OPTIONS,
            width=6,
            state="readonly",
        )
        units_combo.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(row4, text="Summary units:").pack(side=tk.LEFT, padx=(4, 2))
        summary_units_combo = ttk.Combobox(
            row4,
            textvariable=self.arc_summary_units_var,
            values=ARC_SUMMARY_UNIT_OPTIONS,
            width=6,
            state="readonly",
        )
        summary_units_combo.pack(side=tk.LEFT, padx=(0, 12))
        summary_units_combo.bind("<<ComboboxSelected>>", lambda event: self.update_results_text())

        ttk.Label(
            row4,
            text="8-Arc Summary can be shown in pixels, mm, or inches. Force convex arcs is normally best left enabled; press Auto Process All after changing it.",
            foreground="gray35",
        ).pack(side=tk.LEFT, padx=(8, 2))

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

        add_slider_row(
            parent,
            6,
            "Edge offset px:",
            self.edge_offset_px_var,
            -50.0,
            50.0,
            6,
            "Positive moves the detected edge outward. Negative moves it inward. Larger offsets up to +/-50 px are available.",
        )

    def _build_actions_options_tab(self, parent):
        """
        Actions that do not require manual point placement.
        """
        row = ttk.Frame(parent)
        row.grid(row=0, column=0, sticky="w", padx=4, pady=8)

        ttk.Button(row, text="Auto Orient", command=self.auto_orient_image_and_axis).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Auto Extract Profile", command=self.auto_extract_profile).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Auto Break + Fit", command=self.auto_break_and_fit).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Rebuild Current", command=self.rebuild_current_profile).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Fit Arcs", command=self.fit_arcs).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Export CSV", command=self.export_csv).pack(side=tk.LEFT, padx=4)

        ttk.Label(
            parent,
            text="Manual clicking on the image is disabled. Use Auto Process / Rebuild plus the tuning controls.",
            foreground="gray35",
            wraplength=850,
            justify=tk.LEFT,
        ).grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))

    def _build_mode_bar(self, parent):
        """
        Display and quick point-adjust controls.
        """
        frame = ttk.LabelFrame(parent, text="View Options")
        frame.grid(row=2, column=0, sticky="ew", pady=(0, 5))
        frame.columnconfigure(0, weight=1)

        display_row = ttk.Frame(frame)
        display_row.grid(row=0, column=0, sticky="ew", padx=4, pady=(3, 4))

        ttk.Checkbutton(
            display_row,
            text="Show profile points",
            variable=self.show_profile_points_var,
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
            text="Point adjust mode",
            variable=self.point_adjust_mode_var,
            command=self.on_point_adjust_mode_changed
        ).pack(side=tk.LEFT, padx=8)

        ttk.Label(display_row, text="Step px:").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Entry(display_row, textvariable=self.nudge_step_px_var, width=6).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Button(display_row, text="←", width=3, command=lambda: self.nudge_selected_boundary_point(-1.0, 0.0)).pack(side=tk.LEFT, padx=1)
        ttk.Button(display_row, text="→", width=3, command=lambda: self.nudge_selected_boundary_point(1.0, 0.0)).pack(side=tk.LEFT, padx=1)
        ttk.Button(display_row, text="↑", width=3, command=lambda: self.nudge_selected_boundary_point(0.0, -1.0)).pack(side=tk.LEFT, padx=1)
        ttk.Button(display_row, text="↓", width=3, command=lambda: self.nudge_selected_boundary_point(0.0, 1.0)).pack(side=tk.LEFT, padx=1)

        ttk.Label(
            display_row,
            textvariable=self.selected_boundary_label_var,
            foreground="gray35",
        ).pack(side=tk.LEFT, padx=(12, 0))

        ttk.Label(
            display_row,
            text="Click a P0-P8 square point, then nudge. Normal left-click drag pan works when adjust mode is off.",
            foreground="gray35",
        ).pack(side=tk.LEFT, padx=16)

    def _build_plot_area(self, parent):
        """
        Build the plot area as tabs.

        The Image Trace-only tab was removed. Each graph now draws directly on
        top of the rotated source image so the user can judge how well the
        extracted points/arcs fit the real object.

        Tabs:
            Left / Right
            Average
        """
        self.plot_notebook = ttk.Notebook(parent)
        self.plot_notebook.grid(row=3, column=0, sticky="nsew")

        profile_tab = ttk.Frame(self.plot_notebook)
        average_tab = ttk.Frame(self.plot_notebook)

        self.plot_notebook.add(profile_tab, text="Left / Right")
        self.plot_notebook.add(average_tab, text="Average")

        self.fig_profile, self.ax_profile, self.canvas_profile, self.toolbar_profile = self._create_single_plot_tab(
            profile_tab,
            title="Left / Right Overlay",
        )
        self.fig_average, self.ax_average, self.canvas_average, self.toolbar_average = self._create_single_plot_tab(
            average_tab,
            title="Average Overlay",
        )

        # Backward-compatible aliases for code paths that still expect the
        # primary figure/canvas/toolbar names.
        self.figure = self.fig_profile
        self.canvas = self.canvas_profile
        self.toolbar = self.toolbar_profile

        self.figures = [self.fig_profile, self.fig_average]
        self.canvases = [self.canvas_profile, self.canvas_average]
        self.toolbars = [self.toolbar_profile, self.toolbar_average]
        self.canvas_to_toolbar = {
            self.canvas_profile: self.toolbar_profile,
            self.canvas_average: self.toolbar_average,
        }

        for canvas in self.canvases:
            canvas.mpl_connect("button_press_event", self.on_mouse_press)
            canvas.mpl_connect("button_release_event", self.on_mouse_release)
            canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
            canvas.mpl_connect("scroll_event", self.on_mouse_scroll)
            canvas.mpl_connect("figure_leave_event", self.hide_hover_annotation)

        self._draw_empty()

    def _create_single_plot_tab(self, parent, title: str):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        figure = Figure(figsize=(10, 7), dpi=100)
        axes = figure.add_subplot(111)
        axes.set_title(title)

        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.grid(row=1, column=0, sticky="ew")

        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        return figure, axes, canvas, toolbar

    def _build_status_bar(self, parent):
        label = ttk.Label(parent, textvariable=self.status_var, anchor="w")
        label.grid(row=5, column=0, sticky="ew", pady=(5, 0))

    def _build_results_panel(self, parent):
        """
        Build the right-side panel.

        The right side now uses tabs instead of stacking Quick Instructions
        above the full results. This keeps the side panel useful on smaller
        screens and lets the user cycle between:

            Instructions
            Full Results
            8-Arc Summary
        """
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        info_frame = ttk.LabelFrame(parent, text="Scale / Axis / Orientation")
        info_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        self.axis_info_var = tk.StringVar(value="Axis: not set")
        self.scale_info_var = tk.StringVar(value="Scale: not set; using pixels")
        self.orientation_info_var = tk.StringVar(value="Image orientation: not applied")
        self.profile_info_var = tk.StringVar(value="Auto profile: not extracted")

        ttk.Label(info_frame, textvariable=self.axis_info_var, anchor="w", wraplength=430).pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(info_frame, textvariable=self.scale_info_var, anchor="w", wraplength=430).pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(info_frame, textvariable=self.orientation_info_var, anchor="w", wraplength=430).pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(info_frame, textvariable=self.profile_info_var, anchor="w", wraplength=430).pack(fill=tk.X, padx=6, pady=2)

        side_notebook = ttk.Notebook(parent)
        side_notebook.grid(row=1, column=0, sticky="nsew")

        instructions_tab = ttk.Frame(side_notebook)
        full_results_tab = ttk.Frame(side_notebook)
        arc_summary_tab = ttk.Frame(side_notebook)

        side_notebook.add(instructions_tab, text="Instructions")
        side_notebook.add(full_results_tab, text="Full Results")
        side_notebook.add(arc_summary_tab, text="8-Arc Summary")

        self._build_instructions_tab(instructions_tab)
        self._build_full_results_tab(full_results_tab)
        self._build_arc_summary_tab(arc_summary_tab)

        button_row = ttk.Frame(parent)
        button_row.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)
        button_row.columnconfigure(2, weight=1)

        ttk.Button(
            button_row,
            text="Copy Full Results",
            command=self.copy_results
        ).grid(row=0, column=0, sticky="ew", padx=(0, 3))

        ttk.Button(
            button_row,
            text="Copy 8-Arc Summary",
            command=self.copy_arc_summary
        ).grid(row=0, column=1, sticky="ew", padx=3)

        ttk.Button(
            button_row,
            text="Export 8-Arc CSV",
            command=self.export_arc_summary_csv
        ).grid(row=0, column=2, sticky="ew", padx=(3, 0))

        self.update_results_text()

    def _build_instructions_tab(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        instructions_text = tk.Text(
            parent,
            wrap="word",
            width=52,
            height=24
        )
        instructions_text.grid(row=0, column=0, sticky="nsew")

        scroll_y = ttk.Scrollbar(parent, orient="vertical", command=instructions_text.yview)
        scroll_y.grid(row=0, column=1, sticky="ns")

        instructions_text.configure(yscrollcommand=scroll_y.set)

        instructions = (
            "Quick Instructions\n"
            "==================\n\n"
            "1. Load Image.\n"
            "2. Press Auto Process All when ready. Images do not auto-process on load.\n"
            "3. Use the Tuning tab if the edge is weak, over/under-shoots, or sits a few pixels inside/outside the true edge.\n"
            "4. Enter Part Length / Part Width if you need mm or inches.\n"
            "5. Arc fitting always creates 8 arcs. Force convex arcs keeps centers inward so arcs bulge outward. After changing it, press Auto Process All.\n"
            "6. Mouse wheel zoom is enabled directly on the plots.\n"
            "7. Left-click and drag pans the active plot.\n"
            "8. Hover over P0..P8 for coordinates; hover between them for arc radius.\n"
            "8. P0 is the top point and P8 is the bottom point.\n"
            "9. One arc start point is always forced to the widest point.\n"
            "10. Use the 8-Arc Summary tab for the simplified P0, Arc1, P1, Arc2... list.\n"
            "11. Choose Summary units: px, mm, or in. mm/in require Part Length/Width.\n"
            "12. Use Export 8-Arc CSV for the compact P0-P8 / A1-A8 output.\n\n"
            "Arc distribution rules\n"
            "======================\n\n"
            "Half above midpoint:\n"
            "  Attempts to place 4 arcs above the vertical midpoint and 4 below it.\n\n"
            "Half above widest point:\n"
            "  Attempts to place 4 arcs above the widest point and 4 below it.\n\n"
            "Unrestricted:\n"
            "  Uses the automatic error-based distribution behavior.\n\n"
            "Tuning controls\n"
            "===============\n\n"
            "Edge sensitivity:\n"
            "  Lower accepts weaker edges. Higher requires stronger contrast.\n\n"
            "Search distance:\n"
            "  Higher searches farther outward from the inner opening.\n\n"
            "Smooth:\n"
            "  Higher smooths the extracted edge more, but can erase corners.\n\n"
            "Edge offset px:\n"
            "  Positive moves the detected edge outward. Negative moves it inward.\n"
            "  Range is -50 to +50 px. Start small, but larger offsets can help with thick rims or soft edges.\n"
        )

        instructions_text.insert("1.0", instructions)
        instructions_text.configure(state="disabled")

    def _build_full_results_tab(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        self.results_text = tk.Text(
            parent,
            wrap="none",
            width=52,
            height=24
        )
        self.results_text.grid(row=0, column=0, sticky="nsew")

        scroll_y = ttk.Scrollbar(parent, orient="vertical", command=self.results_text.yview)
        scroll_y.grid(row=0, column=1, sticky="ns")

        scroll_x = ttk.Scrollbar(parent, orient="horizontal", command=self.results_text.xview)
        scroll_x.grid(row=1, column=0, sticky="ew")

        self.results_text.configure(
            yscrollcommand=scroll_y.set,
            xscrollcommand=scroll_x.set
        )

    def _build_arc_summary_tab(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        self.arc_summary_text = tk.Text(
            parent,
            wrap="none",
            width=52,
            height=24
        )
        self.arc_summary_text.grid(row=0, column=0, sticky="nsew")

        scroll_y = ttk.Scrollbar(parent, orient="vertical", command=self.arc_summary_text.yview)
        scroll_y.grid(row=0, column=1, sticky="ns")

        scroll_x = ttk.Scrollbar(parent, orient="horizontal", command=self.arc_summary_text.xview)
        scroll_x.grid(row=1, column=0, sticky="ew")

        self.arc_summary_text.configure(
            yscrollcommand=scroll_y.set,
            xscrollcommand=scroll_x.set
        )

    def toggle_quick_instructions(self):
        """
        Kept for backward compatibility with older builds.

        Quick Instructions now live in a tab, so there is no separate
        instruction-frame collapse behavior anymore.
        """
        return

    def _bind_shortcuts(self):
        self.root.bind("f", lambda event: self.fit_arcs())
        self.root.bind("F", lambda event: self.fit_arcs())

        self.root.bind("<Left>", lambda event: self.nudge_selected_boundary_point(-1.0, 0.0))
        self.root.bind("<Right>", lambda event: self.nudge_selected_boundary_point(1.0, 0.0))
        self.root.bind("<Up>", lambda event: self.nudge_selected_boundary_point(0.0, -1.0))
        self.root.bind("<Down>", lambda event: self.nudge_selected_boundary_point(0.0, 1.0))

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
        # Manual editing modes are disabled in this build.
        self.status_var.set("Manual point placement is disabled. Use wheel zoom, click-drag pan, auto processing, and hover tooltips.")

    def set_image(self, image_pil: Image.Image):
        self.image_pil = image_pil.convert("RGB")
        self.image_array = pil_to_display_array(self.image_pil)

    def clear_annotations_for_new_image(self):
        self.axis_points_px.clear()
        self.scale_points_px.clear()
        self.profile_points_px.clear()
        self.right_profile_base_xy = None
        self.left_profile_base_xy = None
        self.average_profile_base_xy = None
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
        self.right_profile_base_xy = None
        self.left_profile_base_xy = None
        self.average_profile_base_xy = None
        self.break_indices.clear()
        self.arc_results.clear()

        self.scale_units = "px"
        self.scale_per_px = 1.0
        self.has_real_scale = False

        self.auto_profile_report = None
        self.auto_breakpoint_report = None

    def update_status_for_mode(self):
        self.status_var.set("Manual point placement is disabled. Use wheel zoom, click-drag pan, and hover points/arcs for values.")

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

        # Do not auto-process immediately after loading.
        # It is easy and safer to press Auto Process All once the desired target,
        # units, and tuning options are set.
        self.status_var.set(
            "Image loaded. Press Auto Process All when ready."
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

    def get_edge_offset_px(self) -> float:
        """
        Read the Edge offset slider in pixels.

        This is a final geometric nudge after edge detection:
            positive = move profile edge outward
            negative = move profile edge inward

        It is useful when the detected points are consistently inside or
        outside the visual edge. Range is intentionally wide for thick rims,
        soft edges, and photos where the edge finder lands far inside the body.
        """
        return self.safe_float_from_var(
            self.edge_offset_px_var,
            default=0.0,
            minimum=-50.0,
            maximum=50.0,
        )

    def apply_edge_offset_to_half_widths(
        self,
        half_widths: np.ndarray,
        close_ends: bool,
    ) -> np.ndarray:
        """
        Apply the pixel edge offset to an array of positive-side half widths.

        The profile coordinate units here are base extraction units, where
        self.scale_per_px converts image pixels to profile units. In the normal
        workflow self.scale_per_px is 1, so the offset is literally pixels.

        Closure points stay exactly on the axis so the profile remains closed.
        """
        offset = self.get_edge_offset_px() * self.scale_per_px

        adjusted = np.array(half_widths, dtype=float).copy()

        if abs(offset) < 1e-12:
            return adjusted

        if len(adjusted) == 0:
            return adjusted

        if close_ends and len(adjusted) >= 2:
            adjusted[1:-1] = np.maximum(0.0, adjusted[1:-1] + offset)
            adjusted[0] = 0.0
            adjusted[-1] = 0.0
        else:
            adjusted = np.maximum(0.0, adjusted + offset)

        return adjusted

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
                right_profile_xy, left_profile_xy, average_profile_xy = self.extract_outer_profile_triplet_from_image(
                    component_points,
                    requested_points=requested_points,
                )
            else:
                right_profile_xy, left_profile_xy, average_profile_xy = self.extract_symmetric_profile_triplet_from_component(
                    component_points,
                    requested_points=requested_points,
                )

            profile_xy = right_profile_xy

            if len(profile_xy) < 8:
                raise ValueError("Profile extraction produced too few points.")

            self.right_profile_base_xy = np.array(right_profile_xy, dtype=float)
            self.left_profile_base_xy = np.array(left_profile_xy, dtype=float)
            self.average_profile_base_xy = np.array(average_profile_xy, dtype=float)

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
                edge_offset_px=self.get_edge_offset_px(),
                y_min=float(np.min(profile_xy[:, 1])),
                y_max=float(np.max(profile_xy[:, 1])),
                max_half_width=float(np.max(np.abs(average_profile_xy[:, 0]))),
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
        scan_side_sign: Optional[float] = None,
    ) -> Optional[float]:
        """
        Sample the right-side outer edge at a Y coordinate expressed in the
        original inner-axis coordinate system.

        The scan starts at the symmetry axis and moves outward. The algorithm
        deliberately skips the inner opening edge and searches for a later edge
        that should correspond to the outside body/chrome boundary.
        """
        old_top, _, old_axis_dir, old_axis_perp, old_axis_length, old_side_sign = old_basis

        if scan_side_sign is not None:
            old_side_sign = float(scan_side_sign)

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

    def extract_outer_profile_triplet_from_image(
        self,
        component_points_px: np.ndarray,
        requested_points: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Trace independent right and left OUTER edges, then build an averaged
        profile from the two sides.
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
            y_min_old = 0.0
            y_max_old = old_axis_length * self.scale_per_px
        else:
            y_extension = max(max_inner * 0.90, inner_axis_length * 0.18, 80.0 * self.scale_per_px) * search_scale
            y_scan_count = max(requested_points * 3, 160)
            y_scan = np.linspace(-y_extension, inner_axis_length + y_extension, y_scan_count)
            inner_at_scan = np.interp(y_scan, y_inner, inner_half, left=0.0, right=0.0)

            outer_width_scan_right = np.full_like(y_scan, np.nan, dtype=float)
            outer_width_scan_left = np.full_like(y_scan, np.nan, dtype=float)

            for index, (old_y, inner_h) in enumerate(zip(y_scan, inner_at_scan)):
                edge_right = self.sample_outer_edge_at_old_y(
                    gray=gray,
                    old_y=float(old_y),
                    inner_h=float(inner_h),
                    old_basis=old_basis,
                    max_inner=max_inner,
                    require_strong_edge=True,
                    sensitivity=sensitivity,
                    search_scale=search_scale,
                    scan_side_sign=old_side_sign,
                )
                edge_left = self.sample_outer_edge_at_old_y(
                    gray=gray,
                    old_y=float(old_y),
                    inner_h=float(inner_h),
                    old_basis=old_basis,
                    max_inner=max_inner,
                    require_strong_edge=True,
                    sensitivity=sensitivity,
                    search_scale=search_scale,
                    scan_side_sign=-old_side_sign,
                )

                if edge_right is not None:
                    outer_width_scan_right[index] = edge_right
                if edge_left is not None:
                    outer_width_scan_left[index] = edge_left

            scan_stack = np.column_stack([outer_width_scan_right, outer_width_scan_left])
            combined_scan = np.full_like(y_scan, np.nan, dtype=float)

            finite_rows = np.any(np.isfinite(scan_stack), axis=1)
            if np.any(finite_rows):
                combined_scan[finite_rows] = np.nanmax(scan_stack[finite_rows], axis=1)

            y_min_old, y_max_old = self.choose_contiguous_outer_y_span(
                y_scan,
                combined_scan,
                inner_axis_length,
            )

            outer_axis_length = y_max_old - y_min_old
            if outer_axis_length <= 1e-9:
                raise ValueError("Outer profile axis length is invalid.")

            new_bottom_px = self.profile_xy_to_image_point_with_basis(
                0.0, y_min_old, old_top, old_axis_dir, old_axis_perp, old_axis_length, old_side_sign,
            )
            new_top_px = self.profile_xy_to_image_point_with_basis(
                0.0, y_max_old, old_top, old_axis_dir, old_axis_perp, old_axis_length, old_side_sign,
            )
            self.axis_points_px = [np.array(new_top_px, dtype=float), np.array(new_bottom_px, dtype=float)]

        final_basis = self.current_axis_basis_values()
        _, _, _, _, final_axis_length, _ = final_basis
        outer_axis_length = final_axis_length * self.scale_per_px
        close_ends = bool(self.close_ends_var.get())

        if close_ends:
            target_y_new = cosine_spaced_values(0.0, outer_axis_length, requested_points)
        else:
            target_y_new = np.linspace(0.0, outer_axis_length, requested_points)

        if lock_manual:
            final_to_old_offset = 0.0
        else:
            final_to_old_offset = y_min_old

        right_half = []
        left_half = []

        for new_y in target_y_new:
            old_y = final_to_old_offset + float(new_y)

            if close_ends and (new_y <= 1e-9 or new_y >= outer_axis_length - 1e-9):
                right_half.append(0.0)
                left_half.append(0.0)
                continue

            inner_h = float(np.interp(old_y, y_inner, inner_half, left=0.0, right=0.0))

            edge_right = self.sample_outer_edge_at_old_y(
                gray=gray,
                old_y=float(old_y),
                inner_h=inner_h,
                old_basis=old_basis,
                max_inner=max_inner,
                require_strong_edge=False,
                sensitivity=sensitivity,
                search_scale=search_scale,
                scan_side_sign=old_side_sign,
            )
            edge_left = self.sample_outer_edge_at_old_y(
                gray=gray,
                old_y=float(old_y),
                inner_h=inner_h,
                old_basis=old_basis,
                max_inner=max_inner,
                require_strong_edge=False,
                sensitivity=sensitivity,
                search_scale=search_scale,
                scan_side_sign=-old_side_sign,
            )

            fallback = float(inner_h + max(20.0 * self.scale_per_px, max_inner * 0.12))
            right_half.append(float(edge_right) if edge_right is not None else fallback)
            left_half.append(float(edge_left) if edge_left is not None else fallback)

        right_half = np.array(right_half, dtype=float)
        left_half = np.array(left_half, dtype=float)

        if len(right_half) >= 7 and smooth_window > 1:
            smoothed = smooth_1d(right_half, window=smooth_window)
            if close_ends:
                smoothed[0] = 0.0
                smoothed[-1] = 0.0
            right_half = smoothed

        if len(left_half) >= 7 and smooth_window > 1:
            smoothed = smooth_1d(left_half, window=smooth_window)
            if close_ends:
                smoothed[0] = 0.0
                smoothed[-1] = 0.0
            left_half = smoothed

        right_half = self.apply_edge_offset_to_half_widths(right_half, close_ends=close_ends)
        left_half = self.apply_edge_offset_to_half_widths(left_half, close_ends=close_ends)

        return self.build_profile_triplet_from_half_widths(right_half, target_y_new, left_half)

    def extract_symmetric_profile_triplet_from_component(
        self,
        component_points_px: np.ndarray,
        requested_points: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract independent left and right half-width profiles from the detected
        dark opening, then also build an averaged one-side profile.
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

        extraction_bins = max(requested_points * 3, 75)
        bins = np.linspace(raw_y_low, raw_y_high, extraction_bins + 1)

        y_measured = []
        right_half_measured = []
        left_half_measured = []

        for i in range(extraction_bins):
            y0 = bins[i]
            y1 = bins[i + 1]
            mask = (y_values >= y0) & (y_values < y1)

            if int(np.count_nonzero(mask)) < 5:
                continue

            xs = x_values[mask]
            left = max(0.0, -float(np.percentile(xs, 1.5)))
            right = max(0.0, float(np.percentile(xs, 98.5)))
            y_mid = (y0 + y1) / 2.0

            y_measured.append(y_mid)
            right_half_measured.append(right)
            left_half_measured.append(left)

        if len(y_measured) < 8:
            raise ValueError("Not enough profile bins found.")

        y_arr = np.array(y_measured, dtype=float)
        right_arr = np.array(right_half_measured, dtype=float)
        left_arr = np.array(left_half_measured, dtype=float)

        order = np.argsort(y_arr)
        y_arr = y_arr[order]
        right_arr = moving_average(right_arr[order], window=5)
        left_arr = moving_average(left_arr[order], window=5)

        if max(float(np.max(right_arr)), float(np.max(left_arr))) <= 1e-9:
            raise ValueError("Could not estimate profile width.")

        close_ends = bool(self.close_ends_var.get())

        if close_ends:
            source_y = np.concatenate([np.array([0.0]), y_arr, np.array([axis_length])])
            right_source = np.concatenate([np.array([0.0]), right_arr, np.array([0.0])])
            left_source = np.concatenate([np.array([0.0]), left_arr, np.array([0.0])])

            source_order = np.argsort(source_y)
            source_y = source_y[source_order]
            right_source = right_source[source_order]
            left_source = left_source[source_order]

            unique_y, unique_indices = np.unique(source_y, return_index=True)
            source_y = unique_y
            right_source = right_source[unique_indices]
            left_source = left_source[unique_indices]

            target_y = cosine_spaced_values(0.0, axis_length, requested_points)
            right_half = np.interp(target_y, source_y, right_source)
            left_half = np.interp(target_y, source_y, left_source)
            right_half[0] = 0.0
            right_half[-1] = 0.0
            left_half[0] = 0.0
            left_half[-1] = 0.0
            right_half = self.apply_edge_offset_to_half_widths(right_half, close_ends=True)
            left_half = self.apply_edge_offset_to_half_widths(left_half, close_ends=True)
        else:
            target_y = np.linspace(float(y_arr[0]), float(y_arr[-1]), requested_points)
            right_half = np.interp(target_y, y_arr, right_arr)
            left_half = np.interp(target_y, y_arr, left_arr)
            right_half = self.apply_edge_offset_to_half_widths(right_half, close_ends=False)
            left_half = self.apply_edge_offset_to_half_widths(left_half, close_ends=False)

        return self.build_profile_triplet_from_half_widths(right_half, target_y, left_half)

    def auto_break_and_fit(self, show_errors=True):
        if len(self.profile_points_px) < ARC_COUNT + 1:
            if show_errors:
                messagebox.showwarning(
                    "Need more profile points",
                    f"Add or extract at least {ARC_COUNT + 1} profile points for {ARC_COUNT} arcs."
                )
            raise ValueError(f"Need at least {ARC_COUNT + 1} profile points.")

        tolerance = self.safe_float_from_var(
            self.auto_arc_tolerance_var,
            default=2.5,
            minimum=0.05,
            maximum=1000.0,
        )

        distribution_rule = self.arc_distribution_var.get()
        if distribution_rule not in ARC_DISTRIBUTION_OPTIONS:
            distribution_rule = "Unrestricted"
            self.arc_distribution_var.set(distribution_rule)

        min_points = max(3, int(len(self.profile_points_px) / max(ARC_COUNT * 2, 1)))

        try:
            profile_xy = self.get_cad_profile_xy_array()

            boundaries, widest_index = choose_eight_arc_boundaries(
                profile_xy,
                rule=distribution_rule,
                tolerance=tolerance,
                min_points_per_arc=min_points,
            )

            self.break_indices = set(boundaries[1:-1])

            self.auto_breakpoint_report = AutoBreakpointReport(
                breakpoint_indices=sorted(self.break_indices),
                tolerance=tolerance,
                max_arcs=ARC_COUNT,
                min_points_per_arc=min_points,
                distribution_rule=distribution_rule,
                widest_index=widest_index,
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
                f"Fit {len(self.arc_results)} of {ARC_COUNT} arcs. "
                f"Widest point is a segment start."
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
        self.right_profile_base_xy = None
        self.left_profile_base_xy = None
        self.average_profile_base_xy = None

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
        self.right_profile_base_xy = None
        self.left_profile_base_xy = None
        self.average_profile_base_xy = None
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
        self.right_profile_base_xy = None
        self.left_profile_base_xy = None
        self.average_profile_base_xy = None

        self.break_indices = {index for index in self.break_indices if index != removed_index}
        self.break_indices = {index for index in self.break_indices if index < len(self.profile_points_px)}

        self.arc_results.clear()
        self.auto_profile_report = None
        self.auto_breakpoint_report = None

        self.update_results_text()
        self.redraw()
        self.status_var.set("Removed last profile point.")

    def on_mouse_click(self, event):
        """
        Manual image/profile editing is disabled.

        This method is left as a no-op so older code paths do not fail.
        """
        return

    def handle_axis_click(self, point):
        self.status_var.set("Manual closure placement is disabled. Use auto orientation/extraction and tuning controls.")

    def handle_scale_click(self, point):
        self.status_var.set("Manual scale clicking is disabled. Enter Part Length / Part Width instead.")

    def handle_profile_click(self, point):
        self.status_var.set("Manual profile point placement is disabled. Use Auto Extract Profile.")

    def handle_breakpoint_click(self, point):
        self.status_var.set("Manual breakpoint placement is disabled. Use the arc distribution options.")

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

    def parse_positive_float_var(self, var) -> Optional[float]:
        try:
            raw = str(var.get()).strip()
            if not raw:
                return None
            value = float(raw)
            if value <= 0.0:
                return None
            return value
        except Exception:
            return None

    def get_dimension_inputs(self) -> Tuple[Optional[float], Optional[float], str]:
        length = self.parse_positive_float_var(self.part_length_var)
        width = self.parse_positive_float_var(self.part_width_var)
        units = self.dimension_units_var.get() if self.dimension_units_var.get() in DIMENSION_UNIT_OPTIONS else "mm"
        return length, width, units

    def get_right_base_profile_xy_array(self) -> np.ndarray:
        if self.right_profile_base_xy is not None and len(self.right_profile_base_xy):
            return np.array(self.right_profile_base_xy, dtype=float)
        return self.get_base_profile_xy_array()

    def get_left_base_profile_xy_array(self) -> np.ndarray:
        if self.left_profile_base_xy is not None and len(self.left_profile_base_xy):
            return np.array(self.left_profile_base_xy, dtype=float)
        right = self.get_right_base_profile_xy_array()
        mirrored = np.array(right, dtype=float)
        mirrored[:, 0] *= -1.0
        return mirrored

    def get_average_base_profile_xy_array(self) -> np.ndarray:
        if self.average_profile_base_xy is not None and len(self.average_profile_base_xy):
            return np.array(self.average_profile_base_xy, dtype=float)
        return self.get_right_base_profile_xy_array()

    def get_scale_reference_base_profile_xy_array(self) -> np.ndarray:
        if self.average_profile_base_xy is not None and len(self.average_profile_base_xy):
            return np.array(self.average_profile_base_xy, dtype=float)
        if self.right_profile_base_xy is not None and len(self.right_profile_base_xy):
            return np.array(self.right_profile_base_xy, dtype=float)
        return self.get_base_profile_xy_array()

    def build_profile_triplet_from_half_widths(
        self,
        right_half: np.ndarray,
        target_y: np.ndarray,
        left_half: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        right_half = np.array(right_half, dtype=float)
        target_y = np.array(target_y, dtype=float)

        if left_half is None:
            left_half = right_half.copy()
        else:
            left_half = np.array(left_half, dtype=float)

        avg_half = 0.5 * (np.abs(right_half) + np.abs(left_half))

        right_xy = np.column_stack([np.abs(right_half), target_y])
        left_xy = np.column_stack([-np.abs(left_half), target_y])
        average_xy = np.column_stack([avg_half, target_y])

        return right_xy, left_xy, average_xy

    def convert_base_profile_xy_to_requested_units(
        self,
        base_xy: np.ndarray,
        requested_units: str,
    ) -> Tuple[np.ndarray, str, str]:
        requested_units = requested_units if requested_units in ARC_SUMMARY_UNIT_OPTIONS else "px"

        if requested_units == "px":
            scale = max(float(self.scale_per_px), 1e-12)
            return np.array(base_xy, dtype=float) / scale, "px", ""

        current_units = self.get_cad_units_label()
        cad_xy = self.base_xy_to_cad_xy_array(
            np.array(base_xy, dtype=float),
            scale_reference_xy=self.get_scale_reference_base_profile_xy_array(),
        )

        if current_units == requested_units:
            return cad_xy, requested_units, ""

        if current_units == "mm" and requested_units == "in":
            return cad_xy / 25.4, "in", ""

        if current_units == "in" and requested_units == "mm":
            return cad_xy * 25.4, "mm", ""

        note = (
            f"Requested {requested_units}, but no real-world scale is available. "
            "Showing pixels. Enter Part Length/Width to enable mm/in."
        )
        scale = max(float(self.scale_per_px), 1e-12)
        return np.array(base_xy, dtype=float) / scale, "px", note

    def get_base_profile_xy_array(self) -> np.ndarray:
        if self.right_profile_base_xy is not None and len(self.right_profile_base_xy):
            return np.array(self.right_profile_base_xy, dtype=float)

        if len(self.axis_points_px) != 2:
            raise ValueError("Axis is not set.")

        coords = [
            self.image_point_to_profile_xy(point)
            for point in self.profile_points_px
        ]

        return np.array(coords, dtype=float)

    def get_cad_units_label(self) -> str:
        length, width, units = self.get_dimension_inputs()

        if length is not None or width is not None:
            return units

        return self.scale_units

    def get_cad_scale_factors(self, base_xy: Optional[np.ndarray] = None) -> Tuple[float, float, float, str]:
        """
        Return (x_scale, y_scale, y_min, units) for CAD/result coordinates.

        If part length/width are supplied, the one-side profile is scaled so:
            full width = entered Part width
            full length = entered Part length

        If only one dimension is supplied, it is used as a uniform scale.
        If neither is supplied, coordinates remain in current profile units.
        """
        if base_xy is None:
            base_xy = self.get_scale_reference_base_profile_xy_array()

        length, width, units = self.get_dimension_inputs()

        if len(base_xy) == 0 or (length is None and width is None):
            return 1.0, 1.0, 0.0, self.scale_units

        y_min = float(np.min(base_xy[:, 1]))
        y_max = float(np.max(base_xy[:, 1]))
        base_length = max(y_max - y_min, 1e-9)

        max_half_width = max(float(np.max(np.abs(base_xy[:, 0]))), 1e-9)
        base_full_width = max_half_width * 2.0

        if length is not None and width is not None:
            x_scale = width / base_full_width
            y_scale = length / base_length
        elif length is not None:
            x_scale = length / base_length
            y_scale = x_scale
        else:
            x_scale = width / base_full_width
            y_scale = x_scale

        return float(x_scale), float(y_scale), y_min, units

    def base_xy_to_cad_xy_array(
        self,
        base_xy: np.ndarray,
        scale_reference_xy: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if len(base_xy) == 0:
            return base_xy.copy()

        if scale_reference_xy is None:
            scale_reference_xy = base_xy

        x_scale, y_scale, y_min, _ = self.get_cad_scale_factors(scale_reference_xy)

        cad = base_xy.copy().astype(float)
        cad[:, 0] = cad[:, 0] * x_scale
        cad[:, 1] = (cad[:, 1] - y_min) * y_scale

        return cad

    def cad_xy_to_base_xy_array(self, cad_xy: np.ndarray) -> np.ndarray:
        base_reference = self.get_base_profile_xy_array()
        x_scale, y_scale, y_min, _ = self.get_cad_scale_factors(base_reference)

        base = cad_xy.copy().astype(float)
        base[:, 0] = base[:, 0] / max(x_scale, 1e-12)
        base[:, 1] = (base[:, 1] / max(y_scale, 1e-12)) + y_min

        return base

    def get_cad_profile_xy_array(self) -> np.ndarray:
        base_xy = self.get_base_profile_xy_array()
        return self.base_xy_to_cad_xy_array(
            base_xy,
            scale_reference_xy=self.get_scale_reference_base_profile_xy_array(),
        )

    def get_profile_xy_array(self) -> np.ndarray:
        """Backward-compatible alias for the unscaled/base profile coordinates."""
        return self.get_base_profile_xy_array()

    def get_segment_ranges(self) -> List[Tuple[int, int]]:
        """
        Return arc segment index ranges in visual/top-to-bottom order.

        Internally, the extracted profile points are usually stored bottom-to-top:
            index 0      = bottom closure
            last index   = top closure

        For CAD output and the legend, the desired convention is:
            P0 / Arc 1 starts at the top
            P8 ends at the bottom

        Therefore this function returns ranges in descending profile-index order,
        for example:
            [(top_idx, next_idx), ..., (next_idx, bottom_idx)]
        """
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

        top_to_bottom_boundaries = list(reversed(boundaries))

        ranges = []

        for start, end in zip(top_to_bottom_boundaries[:-1], top_to_bottom_boundaries[1:]):
            if start != end:
                ranges.append((start, end))

        return ranges

    def segment_points_from_range(
        self,
        profile_xy: np.ndarray,
        start_index: int,
        end_index: int,
        include_neighbors: bool = False,
    ) -> np.ndarray:
        """
        Return points for a segment while preserving the requested start->end direction.

        If start_index > end_index, the slice is reversed so the first point in
        the returned array corresponds to the visual/CAD start of the arc.
        """
        low = min(start_index, end_index)
        high = max(start_index, end_index)

        if include_neighbors:
            low = max(0, low - 1)
            high = min(len(profile_xy) - 1, high + 1)

        segment = profile_xy[low:high + 1]

        if start_index > end_index:
            segment = segment[::-1]

        return segment

    def on_force_convex_arcs_changed(self):
        """
        Lightweight handler for the Force convex arcs checkbox.

        Do not auto-rebuild here. Rebuilding/fitting can be slow and made the
        app appear frozen when the checkbox was clicked.
        """
        self.arc_results.clear()
        self.update_results_text()
        self.redraw_preserving_overlay_view()
        self.status_var.set(
            "Force convex arcs changed. Press Auto Process All to refit arcs."
        )

    def fit_circle_with_convex_preference(self, segment_points: np.ndarray):
        """
        Fit an endpoint-constrained circular arc.

        When Force convex arcs is enabled, the circle center is forced to the
        inward side of the profile whenever the mathematically equivalent
        alternate center gives the desired outward bulge.

        Why this is needed:
            For any two arc endpoints, a circle center can lie on either side of
            the chord. One side makes the edge bulge outward/convex; the other
            can make it visibly cave inward/concave. A least-squares fit can
            occasionally choose the wrong side, especially around sharp
            direction changes or noisy photo edges.
        """
        center, radius, errors = fit_circle_endpoint_constrained(segment_points)

        if not bool(self.force_convex_arcs_var.get()):
            return center, radius, errors

        if len(segment_points) < 2:
            return center, radius, errors

        # Determine which side of the axis this segment is on.
        #   right/average side: mean x is positive; inward is smaller x
        #   left side:          mean x is negative; inward is larger x
        mean_x = float(np.mean(segment_points[:, 0]))

        if abs(mean_x) < 1e-9:
            return center, radius, errors

        side_sign = 1.0 if mean_x >= 0.0 else -1.0

        p0 = np.asarray(segment_points[0], dtype=float)
        p1 = np.asarray(segment_points[-1], dtype=float)
        midpoint = (p0 + p1) / 2.0

        # The alternate circle through the same endpoints is the center on the
        # opposite side of the chord midpoint.
        alternate_center = 2.0 * midpoint - center

        current_inward_score = side_sign * float(center[0])
        alternate_inward_score = side_sign * float(alternate_center[0])

        # Lower score means "more toward the axis / inward".
        # If the alternate center is more inward, use it.
        if alternate_inward_score < current_inward_score:
            center = alternate_center
            radius = float(np.linalg.norm(p0 - center))
            distances = np.linalg.norm(segment_points - center, axis=1)
            errors = distances - radius

        return center, radius, errors

    def fit_arcs(self, show_warnings=True):
        if len(self.axis_points_px) != 2:
            if show_warnings:
                messagebox.showwarning("Axis required", "Set the symmetry axis first.")
            raise ValueError("Axis required.")

        if len(self.profile_points_px) < ARC_COUNT + 1:
            if show_warnings:
                messagebox.showwarning(
                    "Need more points",
                    f"Add at least {ARC_COUNT + 1} profile points for {ARC_COUNT} arcs."
                )
            raise ValueError(f"Need at least {ARC_COUNT + 1} profile points.")

        try:
            profile_xy = self.get_cad_profile_xy_array()
        except Exception as exc:
            if show_warnings:
                messagebox.showerror("Coordinate error", str(exc))
            raise

        tolerance = self.safe_float_from_var(
            self.auto_arc_tolerance_var,
            default=2.5,
            minimum=0.05,
            maximum=1000.0,
        )
        distribution_rule = self.arc_distribution_var.get()
        if distribution_rule not in ARC_DISTRIBUTION_OPTIONS:
            distribution_rule = "Unrestricted"
            self.arc_distribution_var.set(distribution_rule)

        min_points = max(3, int(len(self.profile_points_px) / max(ARC_COUNT * 2, 1)))
        boundaries, widest_index = choose_eight_arc_boundaries(
            profile_xy,
            rule=distribution_rule,
            tolerance=tolerance,
            min_points_per_arc=min_points,
        )
        self.break_indices = set(boundaries[1:-1])
        self.auto_breakpoint_report = AutoBreakpointReport(
            breakpoint_indices=sorted(self.break_indices),
            tolerance=tolerance,
            max_arcs=ARC_COUNT,
            min_points_per_arc=min_points,
            distribution_rule=distribution_rule,
            widest_index=widest_index,
        )

        ranges = self.get_segment_ranges()
        results = []

        for segment_number, (start_index, end_index) in enumerate(ranges, start=1):
            segment_points = self.segment_points_from_range(
                profile_xy,
                start_index,
                end_index,
                include_neighbors=False,
            )

            if len(segment_points) < 3:
                segment_points = self.segment_points_from_range(
                    profile_xy,
                    start_index,
                    end_index,
                    include_neighbors=True,
                )

            if len(segment_points) < 3:
                continue

            try:
                center, radius, errors = self.fit_circle_with_convex_preference(segment_points)
            except Exception as exc:
                if show_warnings:
                    messagebox.showwarning(
                        "Arc fit skipped",
                        f"Segment {segment_number} could not be fit:\n{exc}"
                    )
                continue

            # Use the true arc boundary points for drawing start/end, even if
            # a neighboring point was temporarily added to stabilize a tiny fit.
            arc_angle_points = self.segment_points_from_range(
                profile_xy,
                start_index,
                end_index,
                include_neighbors=False,
            )
            if len(arc_angle_points) < 2:
                arc_angle_points = segment_points

            angles = unwrap_segment_angles(arc_angle_points, center)
            theta_start = float(angles[0])
            theta_end = float(angles[-1])

            included_angle_rad = theta_end - theta_start
            included_angle_degrees = math.degrees(included_angle_rad)

            direction = "CCW" if included_angle_degrees >= 0 else "CW"
            arc_length = abs(radius * included_angle_rad)

            start_point = profile_xy[start_index]
            end_point = profile_xy[end_index]
            chord_length = float(np.linalg.norm(end_point - start_point))

            arc_profile_points = self.segment_points_from_range(
                profile_xy,
                start_index,
                end_index,
                include_neighbors=False,
            )
            distances = np.linalg.norm(arc_profile_points - center, axis=1)
            errors = distances - radius
            rms_error = float(math.sqrt(np.mean(errors * errors))) if len(errors) else 0.0
            max_error = float(np.max(np.abs(errors))) if len(errors) else 0.0

            result = ArcResult(
                segment_number=segment_number,
                start_index=start_index,
                end_index=end_index,
                point_count=abs(end_index - start_index) + 1,
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
                self.status_var.set(f"Fit {len(results)} of {ARC_COUNT} arcs.")
            else:
                self.status_var.set("No arcs fit. Add more points or adjust breakpoints.")

    def result_arc_points(self, result: ArcResult, sample_count=100) -> np.ndarray:
        theta_values = np.linspace(result.theta_start_rad, result.theta_end_rad, sample_count)

        x = result.center_x + result.radius * np.cos(theta_values)
        y = result.center_y + result.radius * np.sin(theta_values)

        return np.column_stack([x, y])

    def on_point_adjust_mode_changed(self):
        if self.point_adjust_mode_var.get():
            self.status_var.set("Point adjust mode on: click a P0-P8 square point, then use nudge buttons or arrow keys.")
        else:
            self.selected_boundary_index = None
            self.selected_boundary_side = None
            self.selected_boundary_label_var.set("Selected: none")
            self.status_var.set("Point adjust mode off. Left-click drag pans the view.")

        self.redraw_preserving_overlay_view()

    def get_nudge_step_px(self) -> float:
        return self.safe_float_from_var(
            self.nudge_step_px_var,
            default=1.0,
            minimum=0.01,
            maximum=100.0,
        )

    def get_boundary_indices_in_order(self) -> List[int]:
        labels = self.build_boundary_point_labels()
        ordered = []

        for point_number in range(ARC_COUNT + 1):
            wanted = f"P{point_number}"

            for index, label in labels.items():
                if label == wanted:
                    ordered.append(index)
                    break

        return ordered

    def get_boundary_point_number(self, profile_index: int) -> Optional[int]:
        for point_number, index in enumerate(self.get_boundary_indices_in_order()):
            if int(index) == int(profile_index):
                return point_number

        return None

    def get_profile_base_xy_for_selection_side(self, side: str) -> np.ndarray:
        if side == "left":
            return self.get_left_base_profile_xy_array()
        if side in ("average", "average_mirror"):
            return self.get_average_base_profile_xy_array()
        return self.get_right_base_profile_xy_array()

    def set_profile_base_xy_for_selection_side(self, side: str, adjusted_xy: np.ndarray):
        adjusted_xy = np.array(adjusted_xy, dtype=float)

        right = self.get_right_base_profile_xy_array()
        left = self.get_left_base_profile_xy_array()
        average = self.get_average_base_profile_xy_array()

        if side == "right":
            right = adjusted_xy
            average[:, 0] = 0.5 * (np.abs(right[:, 0]) + np.abs(left[:, 0]))
            average[:, 1] = 0.5 * (right[:, 1] + left[:, 1])
        elif side == "left":
            left = adjusted_xy
            average[:, 0] = 0.5 * (np.abs(right[:, 0]) + np.abs(left[:, 0]))
            average[:, 1] = 0.5 * (right[:, 1] + left[:, 1])
        elif side == "average":
            average = adjusted_xy
        elif side == "average_mirror":
            average = adjusted_xy.copy()
            average[:, 0] = np.abs(average[:, 0])
        else:
            return

        self.right_profile_base_xy = np.array(right, dtype=float)
        self.left_profile_base_xy = np.array(left, dtype=float)
        self.average_profile_base_xy = np.array(average, dtype=float)

        self.profile_points_px = [
            self.profile_xy_to_image_point(float(x), float(y))
            for x, y in self.right_profile_base_xy
        ]

    def select_boundary_point_near_event(self, event) -> bool:
        if event.inaxes not in (self.ax_profile, self.ax_average):
            return False

        if event.x is None or event.y is None:
            return False

        if len(self.axis_points_px) != 2 or not self.profile_points_px:
            return False

        boundary_indices = self.get_boundary_indices_in_order()
        candidates = []

        def add_candidates(side_name: str, base_xy: np.ndarray):
            image_points = self.base_profile_xy_to_image_points(base_xy)

            for point_number, profile_index in enumerate(boundary_indices):
                if 0 <= profile_index < len(image_points):
                    candidates.append((side_name, point_number, profile_index, image_points[profile_index]))

        if event.inaxes == self.ax_profile:
            add_candidates("right", self.get_right_base_profile_xy_array())
            add_candidates("left", self.get_left_base_profile_xy_array())
        else:
            average = self.get_average_base_profile_xy_array()
            add_candidates("average", average)

            average_mirror = np.array(average, dtype=float).copy()
            average_mirror[:, 0] *= -1.0
            add_candidates("average_mirror", average_mirror)

        best = None
        best_distance = float("inf")
        threshold_px = 18.0

        for side_name, point_number, profile_index, image_point in candidates:
            display_xy = event.inaxes.transData.transform(image_point)
            distance = float(np.linalg.norm(display_xy - np.array([event.x, event.y])))

            if distance < best_distance and distance <= threshold_px:
                best_distance = distance
                best = (side_name, point_number, profile_index)

        if best is None:
            self.status_var.set("No P0-P8 point selected. Click closer to a square boundary marker.")
            return False

        side_name, point_number, profile_index = best
        self.selected_boundary_side = side_name
        self.selected_boundary_index = int(profile_index)
        self.selected_boundary_label_var.set(f"Selected: {side_name} P{point_number}")
        self.status_var.set(f"Selected {side_name} P{point_number}. Use nudge buttons or arrow keys.")
        self.redraw_preserving_overlay_view()
        return True

    def nudge_selected_boundary_point(self, direction_x: float, direction_y: float):
        if not self.point_adjust_mode_var.get():
            return

        if self.selected_boundary_index is None or self.selected_boundary_side is None:
            self.status_var.set("No P0-P8 point selected. Click a square point first.")
            return

        if len(self.axis_points_px) != 2 or not self.profile_points_px:
            self.status_var.set("No extracted profile to adjust.")
            return

        side = self.selected_boundary_side
        index = int(self.selected_boundary_index)

        base_xy = self.get_profile_base_xy_for_selection_side(side)

        if index < 0 or index >= len(base_xy):
            self.status_var.set("Selected point is no longer valid.")
            return

        step = self.get_nudge_step_px()
        delta_px = np.array([float(direction_x) * step, float(direction_y) * step], dtype=float)

        _, _, axis_dir, axis_perp, _, side_sign = self.current_axis_basis_values()

        delta_y_base = float(np.dot(delta_px, axis_dir)) * self.scale_per_px
        delta_x_base = float(np.dot(delta_px, axis_perp * side_sign)) * self.scale_per_px

        adjusted = np.array(base_xy, dtype=float).copy()
        point_number = self.get_boundary_point_number(index)

        is_closure = point_number in (0, ARC_COUNT)

        if is_closure:
            adjusted[index, 1] += delta_y_base
            adjusted[index, 0] = 0.0
        else:
            adjusted[index, 0] += delta_x_base
            adjusted[index, 1] += delta_y_base

            if side in ("right", "average"):
                adjusted[index, 0] = max(0.0, adjusted[index, 0])
            elif side in ("left", "average_mirror"):
                adjusted[index, 0] = min(0.0, adjusted[index, 0])

        self.set_profile_base_xy_for_selection_side(side, adjusted)

        try:
            self.fit_arcs(show_warnings=False)
        except Exception as exc:
            self.arc_results.clear()
            self.update_results_text()
            self.redraw_preserving_overlay_view()
            self.status_var.set(f"Point moved, but arc refit failed: {exc}")
            return

        label = f"P{point_number}" if point_number is not None else f"index {index}"
        self.status_var.set(f"Nudged {side} {label} by {step:g} px.")
        self.update_results_text()
        self.redraw_preserving_overlay_view()

    def toolbar_navigation_active(self, event=None) -> bool:
        """
        Return True when the active plot's built-in matplotlib toolbar pan/zoom
        mode is active.

        Each plot tab has its own canvas and toolbar. If the user clicks a
        toolbar's pan/zoom tool, custom drag-pan should not also run.
        """
        toolbar = None

        if event is not None and hasattr(event, "canvas"):
            toolbar = getattr(self, "canvas_to_toolbar", {}).get(event.canvas)

        if toolbar is None:
            toolbar = getattr(self, "toolbar", None)

        try:
            mode = str(getattr(toolbar, "mode", "")).lower()
        except Exception:
            return False

        return ("pan" in mode) or ("zoom" in mode)

    def on_mouse_press(self, event):
        """
        Start a custom pan operation with left-click drag on either plot.

        This uses screen-pixel deltas rather than changing data coordinates.
        Using data coordinates during a drag can jitter because xdata/ydata
        are recalculated every time the axes limits change.
        """
        if self.toolbar_navigation_active(event):
            # Let the built-in matplotlib toolbar handle pan/zoom if selected.
            return

        if event.button != 1:
            return

        if self.point_adjust_mode_var.get():
            if self.select_boundary_point_near_event(event):
                return

        if event.inaxes not in (self.ax_profile, self.ax_average):
            return

        if event.x is None or event.y is None:
            return

        ax = event.inaxes
        bbox = ax.bbox

        if bbox.width <= 0 or bbox.height <= 0:
            return

        self.is_panning = True
        self.pan_axes = ax
        self.pan_start_xy = (float(event.xdata), float(event.ydata)) if event.xdata is not None and event.ydata is not None else None
        self.pan_start_event_xy = (float(event.x), float(event.y))
        self.pan_start_xlim = tuple(ax.get_xlim())
        self.pan_start_ylim = tuple(ax.get_ylim())
        self.pan_data_per_pixel_x = (self.pan_start_xlim[1] - self.pan_start_xlim[0]) / float(bbox.width)
        self.pan_data_per_pixel_y = (self.pan_start_ylim[1] - self.pan_start_ylim[0]) / float(bbox.height)

        self.hide_hover_annotation()
        self.status_var.set("Panning... drag to move view.")

    def on_mouse_release(self, event):
        """
        End a custom pan operation.
        """
        if self.is_panning:
            self.is_panning = False
            self.pan_axes = None
            self.pan_start_xy = None
            self.pan_start_event_xy = None
            self.pan_start_xlim = None
            self.pan_start_ylim = None
            self.pan_data_per_pixel_x = None
            self.pan_data_per_pixel_y = None
            self.status_var.set("Pan complete. Use mouse wheel to zoom or hover for values.")

    def on_mouse_scroll(self, event):
        """
        Zoom around the mouse cursor using the mouse wheel.
        """
        if event.inaxes not in (self.ax_profile, self.ax_average):
            return

        if event.xdata is None or event.ydata is None:
            return

        ax = event.inaxes

        # Smaller factor = zoom in, larger = zoom out
        if event.button == "up":
            zoom_factor = 1.0 / 1.15
        elif event.button == "down":
            zoom_factor = 1.15
        else:
            return

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        new_x0 = event.xdata - (event.xdata - x0) * zoom_factor
        new_x1 = event.xdata + (x1 - event.xdata) * zoom_factor
        new_y0 = event.ydata - (event.ydata - y0) * zoom_factor
        new_y1 = event.ydata + (y1 - event.ydata) * zoom_factor

        ax.set_xlim(new_x0, new_x1)
        ax.set_ylim(new_y0, new_y1)

        event.canvas.draw_idle()
        self.status_var.set("Zoomed view. Scroll to continue zooming.")

    def create_hover_annotation_for_axes(self, axes):
        """
        Create a hover tooltip annotation for the given axes.
        """
        annotation = axes.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", alpha=0.88),
            arrowprops=dict(arrowstyle="->"),
            fontsize=8,
            zorder=1000,
        )
        annotation.set_visible(False)
        return annotation

    def hide_hover_annotation(self, event=None):
        changed = False
        for annotation in (self.hover_annotation, self.average_hover_annotation):
            if annotation is not None and annotation.get_visible():
                annotation.set_visible(False)
                changed = True
        if changed:
            self.draw_all_canvases_idle()

    def format_hover_value(self, value: float, unit: str) -> str:
        return f"{value:.4f} {unit}"

    def get_hover_collections_for_axes(self, axes):
        if axes == self.ax_profile:
            return self.profile_hover_points, self.profile_hover_arcs, self.hover_annotation
        if axes == self.ax_average:
            return self.average_hover_points, self.average_hover_arcs, self.average_hover_annotation
        return None, None, None

    def on_mouse_move(self, event):
        """
        Show a tooltip when hovering over profile points or fitted arcs.

        Hover targets are only on the profile coordinate plots. The image plot is
        intentionally non-editable; it supports wheel zoom and click-drag pan.
        """
        if self.is_panning:
            if (
                self.pan_axes is not None
                and self.pan_start_event_xy is not None
                and self.pan_start_xlim is not None
                and self.pan_start_ylim is not None
                and self.pan_data_per_pixel_x is not None
                and self.pan_data_per_pixel_y is not None
                and event.x is not None
                and event.y is not None
            ):
                dx_pixels = float(event.x) - float(self.pan_start_event_xy[0])
                dy_pixels = float(event.y) - float(self.pan_start_event_xy[1])

                dx_data = dx_pixels * float(self.pan_data_per_pixel_x)
                dy_data = dy_pixels * float(self.pan_data_per_pixel_y)

                self.pan_axes.set_xlim(
                    self.pan_start_xlim[0] - dx_data,
                    self.pan_start_xlim[1] - dx_data,
                )
                self.pan_axes.set_ylim(
                    self.pan_start_ylim[0] - dy_data,
                    self.pan_start_ylim[1] - dy_data,
                )
                event.canvas.draw_idle()
            return

        if self.toolbar_navigation_active(event):
            self.hide_hover_annotation()
            return

        if event.inaxes not in (self.ax_profile, self.ax_average):
            self.hide_hover_annotation()
            return

        if event.x is None or event.y is None or event.xdata is None or event.ydata is None:
            self.hide_hover_annotation()
            return

        hover_points, hover_arcs, annotation = self.get_hover_collections_for_axes(event.inaxes)
        if hover_points is None or hover_arcs is None or annotation is None:
            self.hide_hover_annotation()
            return

        other_annotation = self.average_hover_annotation if annotation is self.hover_annotation else self.hover_annotation
        if other_annotation is not None and other_annotation.get_visible():
            other_annotation.set_visible(False)

        hover_text = None
        hover_xy = None
        best_distance = float("inf")
        point_threshold_px = 11.0
        arc_threshold_px = 8.0

        for item in hover_points:
            point_xy = item["xy"]
            display_xy = event.inaxes.transData.transform(point_xy)
            distance = float(np.linalg.norm(display_xy - np.array([event.x, event.y])))
            if distance < best_distance and distance <= point_threshold_px:
                best_distance = distance
                hover_xy = point_xy
                hover_text = item["text"]

        if hover_text is None:
            for item in hover_arcs:
                arc_xy = item["xy"]
                if arc_xy is None or len(arc_xy) == 0:
                    continue
                display_xy = event.inaxes.transData.transform(arc_xy)
                distances = np.linalg.norm(display_xy - np.array([event.x, event.y]), axis=1)
                min_index = int(np.argmin(distances))
                distance = float(distances[min_index])
                if distance < best_distance and distance <= arc_threshold_px:
                    best_distance = distance
                    hover_xy = arc_xy[min_index]
                    hover_text = item["text"]

        if hover_text is None or hover_xy is None:
            if annotation.get_visible():
                annotation.set_visible(False)
                event.canvas.draw_idle()
            return

        annotation.xy = hover_xy
        annotation.set_text(hover_text)
        annotation.set_visible(True)
        event.canvas.draw_idle()

    def build_boundary_point_labels(self) -> Dict[int, str]:
        """
        Map profile point indices to P0..P8 boundary labels.

        The segment ranges are already top-to-bottom, so the first segment start
        is P0 and each segment end is the next boundary point.
        """
        labels = {}

        ranges = self.get_segment_ranges()

        if not ranges:
            return labels

        labels[ranges[0][0]] = "P0"

        for point_number, (_, end_index) in enumerate(ranges[:ARC_COUNT], start=1):
            labels[end_index] = f"P{point_number}"

        return labels

    def build_profile_index_to_arc_result(self) -> Dict[int, ArcResult]:
        """
        Map every non-boundary profile point index to the arc that contains it.

        Boundary points P0..P8 keep coordinate hover.
        Intermediate profile sample points use this map to show the arc radius
        instead of their own X/Y coordinate.
        """
        index_to_arc = {}

        ranges = self.get_segment_ranges()

        if not ranges or not self.arc_results:
            return index_to_arc

        results_by_number = {
            result.segment_number: result
            for result in self.arc_results
        }

        for arc_number, (start_index, end_index) in enumerate(ranges[:ARC_COUNT], start=1):
            result = results_by_number.get(arc_number)

            if result is None:
                continue

            low = min(start_index, end_index)
            high = max(start_index, end_index)

            for point_index in range(low, high + 1):
                index_to_arc[point_index] = result

        return index_to_arc

    def capture_overlay_view_state(self) -> dict:
        """
        Capture the current zoom/pan limits for overlay plots.

        Used before small redraws such as P0-P8 nudge adjustments so the camera
        does not jump back to the full-image view.
        """
        state = {}

        for name in ("profile", "average"):
            axis = getattr(self, f"ax_{name}", None)

            if axis is None:
                continue

            try:
                state[name] = {
                    "xlim": tuple(axis.get_xlim()),
                    "ylim": tuple(axis.get_ylim()),
                }
            except Exception:
                pass

        return state

    def restore_overlay_view_state(self, state: dict):
        """
        Restore previously captured zoom/pan limits for overlay plots.
        """
        if not state:
            return

        for name, limits in state.items():
            axis = getattr(self, f"ax_{name}", None)

            if axis is None:
                continue

            try:
                axis.set_xlim(limits["xlim"])
                axis.set_ylim(limits["ylim"])
            except Exception:
                pass

    def redraw_preserving_overlay_view(self):
        """
        Redraw without losing the user's current zoom/pan camera.
        """
        view_state = self.capture_overlay_view_state()
        self.redraw()
        self.restore_overlay_view_state(view_state)
        self.draw_all_canvases_idle()

    def draw_all_canvases_idle(self):
        for canvas in getattr(self, "canvases", []):
            canvas.draw_idle()

    def tight_layout_all_figures(self):
        """
        Keep each tab's image overlay large.

        We intentionally do not call tight_layout here. When the overlay legend
        was placed outside the axes, tight_layout shrank the graph down to a tiny
        view to make room for the legend. Manual subplot padding keeps the image
        large and predictable.
        """
        for figure in getattr(self, "figures", []):
            try:
                figure.subplots_adjust(
                    left=0.06,
                    right=0.98,
                    top=0.92,
                    bottom=0.08,
                )
            except Exception:
                pass

    def redraw(self):
        self.ax_profile.clear()
        self.ax_average.clear()

        self.draw_profile_view()
        self.draw_average_view()

        self.tight_layout_all_figures()
        self.draw_all_canvases_idle()

    def _draw_empty(self):
        self.ax_profile.clear()
        self.ax_average.clear()

        self.ax_profile.set_title("Left / Right Overlay")
        self.ax_profile.text(
            0.5,
            0.5,
            "Load an image",
            ha="center",
            va="center",
            transform=self.ax_profile.transAxes,
        )
        self.ax_profile.set_axis_off()

        self.ax_average.set_title("Average Overlay")
        self.ax_average.text(
            0.5,
            0.5,
            "Load an image",
            ha="center",
            va="center",
            transform=self.ax_average.transAxes,
        )
        self.ax_average.set_axis_off()

        self.tight_layout_all_figures()
        self.draw_all_canvases_idle()

    def draw_image_view(self):
        """
        Backward-compatible wrapper. The active UI draws image traces on both
        graph tabs via draw_image_trace_on_axes().
        """
        self.draw_image_trace_on_axes(self.ax_image_profile)

    def draw_image_trace_on_axes(self, axes):
        axes.set_title("Image Trace View")

        if self.image_array is None:
            axes.text(0.5, 0.5, "Load an image", ha="center", va="center", transform=axes.transAxes)
            axes.set_axis_off()
            return

        axes.imshow(self.image_array)
        axes.set_axis_on()

        if self.auto_orientation_report is not None and self.auto_orientation_report.component_bbox_px is not None:
            x0, y0, x1, y1 = self.auto_orientation_report.component_bbox_px
            axes.plot(
                [x0, x1, x1, x0, x0],
                [y0, y0, y1, y1, y0],
                linestyle=":",
                linewidth=1.5
            )

        if len(self.axis_points_px) >= 1:
            pts = np.array(self.axis_points_px)
            axes.scatter(pts[:, 0], pts[:, 1], s=45, marker="x")

        if len(self.axis_points_px) == 2:
            top, bottom = self.axis_top_bottom()
            axes.plot([top[0], bottom[0]], [top[1], bottom[1]], linewidth=2)
            axes.text(top[0], top[1], " axis top", fontsize=9)
            axes.text(bottom[0], bottom[1], " axis bottom", fontsize=9)

        if len(self.scale_points_px) >= 1:
            pts = np.array(self.scale_points_px)
            axes.scatter(pts[:, 0], pts[:, 1], s=35, marker="s")

            if len(self.scale_points_px) == 2:
                a, b = self.scale_points_px
                axes.plot([a[0], b[0]], [a[1], b[1]], linestyle="--", linewidth=2)
                axes.text((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, " scale", fontsize=9)

        if self.profile_points_px:
            pts = np.array(self.profile_points_px)
            axes.plot(pts[:, 0], pts[:, 1], marker="o", markersize=3, linewidth=1.5)

            if self.show_point_numbers_var.get():
                for index, point in enumerate(pts):
                    axes.text(point[0], point[1], str(index), fontsize=8)

            if self.break_indices:
                break_pts = np.array([
                    self.profile_points_px[index]
                    for index in sorted(self.break_indices)
                    if 0 <= index < len(self.profile_points_px)
                ])

                if len(break_pts):
                    axes.scatter(break_pts[:, 0], break_pts[:, 1], s=65, marker="s")

        if self.show_fitted_arcs_var.get() and len(self.axis_points_px) == 2:
            for result in self.arc_results:
                arc_xy = self.result_arc_points(result, sample_count=100)
                base_arc_xy = self.cad_xy_to_base_xy_array(arc_xy)
                arc_px = np.array([self.profile_xy_to_image_point(x, y) for x, y in base_arc_xy])
                axes.plot(arc_px[:, 0], arc_px[:, 1], linewidth=2.5)

                if self.show_mirror_var.get():
                    mirrored_arc_xy = arc_xy.copy()
                    mirrored_arc_xy[:, 0] *= -1.0
                    mirrored_base_arc_xy = self.cad_xy_to_base_xy_array(mirrored_arc_xy)
                    mirrored_arc_px = np.array([self.profile_xy_to_image_point(x, y) for x, y in mirrored_base_arc_xy])
                    axes.plot(mirrored_arc_px[:, 0], mirrored_arc_px[:, 1], linewidth=1.5, linestyle="--")

        axes.set_xlim(0, self.image_array.shape[1])
        axes.set_ylim(self.image_array.shape[0], 0)

    def draw_rotated_image_background(self, axes, title: str):
        """
        Draw the rotated image in the same pixel coordinate system as the image
        trace view.

        The plotted profile points/arcs are converted back from profile
        coordinates into image pixels, then drawn over this image. This avoids
        visually stretching the overlay based on Part Length / Part Width.
        """
        axes.set_title(title)

        if self.image_array is None:
            axes.text(
                0.5,
                0.5,
                "Load an image",
                ha="center",
                va="center",
                transform=axes.transAxes,
            )
            axes.set_axis_off()
            return False

        axes.imshow(self.image_array)
        axes.set_axis_on()
        axes.set_xlim(0, self.image_array.shape[1])
        axes.set_ylim(self.image_array.shape[0], 0)
        axes.set_aspect("equal", adjustable="box")
        axes.set_xlabel("Image X (px)")
        axes.set_ylabel("Image Y (px)")
        return True

    def base_profile_xy_to_image_points(self, base_xy: np.ndarray) -> np.ndarray:
        """
        Convert base profile coordinates into image pixel coordinates.
        """
        if base_xy is None or len(base_xy) == 0:
            return np.empty((0, 2), dtype=float)

        return np.array([
            self.profile_xy_to_image_point(float(x), float(y))
            for x, y in np.array(base_xy, dtype=float)
        ])

    def result_arc_points_as_base_xy(self, result: "ArcResult", sample_count: int = 120) -> np.ndarray:
        """
        Get fitted arc points in base profile coordinates.

        ArcResult values are stored in current CAD/output coordinates, so convert
        back before drawing over the image.
        """
        arc_xy = self.result_arc_points(result, sample_count=sample_count)
        return self.cad_xy_to_base_xy_array(arc_xy)

    def plot_image_axis_and_detection_box(self, axes):
        """
        Draw the detected axis and component bbox over the rotated image.
        """
        if self.auto_orientation_report is not None and self.auto_orientation_report.component_bbox_px is not None:
            x0, y0, x1, y1 = self.auto_orientation_report.component_bbox_px
            axes.plot(
                [x0, x1, x1, x0, x0],
                [y0, y0, y1, y1, y0],
                linestyle=":",
                linewidth=1.5,
            )

        if len(self.axis_points_px) == 2:
            top, bottom = self.axis_top_bottom()
            axes.plot([top[0], bottom[0]], [top[1], bottom[1]], linewidth=2)
            axes.text(top[0], top[1], " axis top", fontsize=9)
            axes.text(bottom[0], bottom[1], " axis bottom", fontsize=9)

    def draw_selected_boundary_marker(self, axes, image_points, side_name: str):
        if self.selected_boundary_index is None or self.selected_boundary_side is None:
            return

        if side_name != self.selected_boundary_side:
            return

        index = int(self.selected_boundary_index)

        if index < 0 or index >= len(image_points):
            return

        point = image_points[index]

        axes.scatter(
            [point[0]],
            [point[1]],
            s=160,
            marker="s",
            facecolors="none",
            edgecolors="black",
            linewidths=2.0,
            zorder=20,
        )

    def draw_profile_view(self):
        unit = self.get_cad_units_label()
        self.profile_hover_points = []
        self.profile_hover_arcs = []
        self.hover_annotation = None

        if not self.draw_rotated_image_background(self.ax_profile, "Left / Right Overlay"):
            return

        if len(self.axis_points_px) != 2:
            self.ax_profile.text(
                0.5,
                0.5,
                "Auto-process to detect the axis",
                ha="center",
                va="center",
                transform=self.ax_profile.transAxes,
            )
            return

        self.plot_image_axis_and_detection_box(self.ax_profile)

        if not self.profile_points_px:
            self.ax_profile.text(
                0.5,
                0.5,
                "Auto-process to extract the profile",
                ha="center",
                va="center",
                transform=self.ax_profile.transAxes,
            )
            return

        right_base_xy = self.get_right_base_profile_xy_array()
        left_base_xy = self.get_left_base_profile_xy_array()

        right_px = self.base_profile_xy_to_image_points(right_base_xy)
        left_px = self.base_profile_xy_to_image_points(left_base_xy)

        boundary_labels = self.build_boundary_point_labels()
        index_to_arc_result = self.build_profile_index_to_arc_result()

        try:
            left_cad_xy = self.base_xy_to_cad_xy_array(
                left_base_xy,
                scale_reference_xy=self.get_scale_reference_base_profile_xy_array(),
            )
            left_arc_results = self.build_arc_results_from_profile_xy(left_cad_xy)
        except Exception:
            left_arc_results = []

        left_index_to_arc = {}
        left_ranges = self.get_segment_ranges()
        for (start_index, end_index), result in zip(left_ranges, left_arc_results):
            low = min(start_index, end_index)
            high = max(start_index, end_index)
            for point_index in range(low, high + 1):
                left_index_to_arc[point_index] = result

        # Hover values are still shown in user-selected units even though the
        # visual overlay is plotted in image pixels.
        scale_reference = self.get_scale_reference_base_profile_xy_array()
        right_cad_xy = self.base_xy_to_cad_xy_array(right_base_xy, scale_reference_xy=scale_reference)
        left_cad_xy = self.base_xy_to_cad_xy_array(left_base_xy, scale_reference_xy=scale_reference)

        for side_name, image_points, cad_points, arc_lookup in (
            ("Right", right_px, right_cad_xy, index_to_arc_result),
            ("Left", left_px, left_cad_xy, left_index_to_arc),
        ):
            for index, image_point in enumerate(image_points):
                boundary_label = boundary_labels.get(index)

                if boundary_label is not None:
                    if boundary_label == "P0":
                        display_label = "P0 / top point"
                    elif boundary_label == f"P{ARC_COUNT}":
                        display_label = f"P{ARC_COUNT} / bottom point"
                    else:
                        display_label = boundary_label

                    cad_point = cad_points[index]
                    hover_text = (
                        f"{side_name} {display_label}\n"
                        f"X = {self.format_hover_value(float(cad_point[0]), unit)}\n"
                        f"Y = {self.format_hover_value(float(cad_point[1]), unit)}"
                    )
                else:
                    if not self.show_profile_points_var.get():
                        continue

                    containing_arc = arc_lookup.get(index)

                    if containing_arc is None:
                        continue

                    hover_text = (
                        f"{side_name} Arc {containing_arc.segment_number}\n"
                        f"R = {self.format_hover_value(float(containing_arc.radius), unit)}"
                    )

                self.profile_hover_points.append({
                    "xy": np.array([float(image_point[0]), float(image_point[1])], dtype=float),
                    "text": hover_text,
                })

        if self.show_profile_points_var.get():
            self.ax_profile.plot(
                right_px[:, 0],
                right_px[:, 1],
                marker="o",
                markersize=3.2,
                linewidth=1.7,
            )
            self.ax_profile.plot(
                left_px[:, 0],
                left_px[:, 1],
                marker="o",
                markersize=3.2,
                linewidth=1.4,
                linestyle="--",
            )

        boundary_indices = [index for index in boundary_labels.keys() if 0 <= index < len(right_px)]

        if boundary_indices:
            right_boundary_px = right_px[boundary_indices]
            left_boundary_px = left_px[boundary_indices]
            self.ax_profile.scatter(
                right_boundary_px[:, 0],
                right_boundary_px[:, 1],
                s=60,
                marker="s",
                zorder=8,
            )
            self.ax_profile.scatter(
                left_boundary_px[:, 0],
                left_boundary_px[:, 1],
                s=60,
                marker="s",
                zorder=8,
            )

            self.draw_selected_boundary_marker(self.ax_profile, right_px, "right")
            self.draw_selected_boundary_marker(self.ax_profile, left_px, "left")

        if self.show_point_numbers_var.get():
            for index, point in enumerate(right_px):
                self.ax_profile.text(point[0], point[1], str(index), fontsize=8)
            for index, point in enumerate(left_px):
                self.ax_profile.text(point[0], point[1], str(index), fontsize=8)

        if self.show_fitted_arcs_var.get():
            for result in self.arc_results:
                arc_base_xy = self.result_arc_points_as_base_xy(result, sample_count=120)
                arc_px = self.base_profile_xy_to_image_points(arc_base_xy)
                self.ax_profile.plot(
                    arc_px[:, 0],
                    arc_px[:, 1],
                    linewidth=2.5,
                )
                self.profile_hover_arcs.append({
                    "xy": arc_px,
                    "text": f"Right Arc {result.segment_number}\nR = {self.format_hover_value(float(result.radius), unit)}",
                })

            for result in left_arc_results:
                arc_cad_xy = self.result_arc_points(result, sample_count=120)
                arc_base_xy = self.cad_xy_to_base_xy_array(arc_cad_xy)
                arc_px = self.base_profile_xy_to_image_points(arc_base_xy)
                self.ax_profile.plot(
                    arc_px[:, 0],
                    arc_px[:, 1],
                    linewidth=1.5,
                    linestyle="--",
                )
                self.profile_hover_arcs.append({
                    "xy": arc_px,
                    "text": f"Left Arc {result.segment_number}\nR = {self.format_hover_value(float(result.radius), unit)}",
                })

        self.hover_annotation = self.create_hover_annotation_for_axes(self.ax_profile)

    def draw_average_view(self):
        unit = self.get_cad_units_label()
        self.average_hover_points = []
        self.average_hover_arcs = []
        self.average_hover_annotation = None

        if not self.draw_rotated_image_background(self.ax_average, "Average Overlay"):
            return

        if len(self.axis_points_px) != 2:
            self.ax_average.text(
                0.5,
                0.5,
                "Auto-process to detect the axis",
                ha="center",
                va="center",
                transform=self.ax_average.transAxes,
            )
            return

        self.plot_image_axis_and_detection_box(self.ax_average)

        if not self.profile_points_px:
            self.ax_average.text(
                0.5,
                0.5,
                "Auto-process to extract the profile",
                ha="center",
                va="center",
                transform=self.ax_average.transAxes,
            )
            return

        average_base_xy = self.get_average_base_profile_xy_array()
        average_mirror_base_xy = np.array(average_base_xy, dtype=float).copy()
        average_mirror_base_xy[:, 0] *= -1.0

        average_px = self.base_profile_xy_to_image_points(average_base_xy)
        average_mirror_px = self.base_profile_xy_to_image_points(average_mirror_base_xy)

        boundary_labels = self.build_boundary_point_labels()

        scale_reference = self.get_scale_reference_base_profile_xy_array()
        average_cad_xy = self.base_xy_to_cad_xy_array(average_base_xy, scale_reference_xy=scale_reference)
        average_mirror_cad_xy = self.base_xy_to_cad_xy_array(average_mirror_base_xy, scale_reference_xy=scale_reference)

        try:
            average_arc_results = self.build_arc_results_from_profile_xy(average_cad_xy)
        except Exception:
            average_arc_results = []

        avg_index_to_arc = {}
        avg_ranges = self.get_segment_ranges()

        for (start_index, end_index), result in zip(avg_ranges, average_arc_results):
            low = min(start_index, end_index)
            high = max(start_index, end_index)

            for point_index in range(low, high + 1):
                avg_index_to_arc[point_index] = result

        for side_name, image_points, cad_points in (
            ("Average", average_px, average_cad_xy),
            ("Average mirrored", average_mirror_px, average_mirror_cad_xy),
        ):
            if side_name == "Average mirrored" and not self.show_mirror_var.get():
                continue

            for index, image_point in enumerate(image_points):
                boundary_label = boundary_labels.get(index)

                if boundary_label is not None:
                    if boundary_label == "P0":
                        display_label = "P0 / top point"
                    elif boundary_label == f"P{ARC_COUNT}":
                        display_label = f"P{ARC_COUNT} / bottom point"
                    else:
                        display_label = boundary_label

                    cad_point = cad_points[index]
                    hover_text = (
                        f"{side_name} {display_label}\n"
                        f"X = {self.format_hover_value(float(cad_point[0]), unit)}\n"
                        f"Y = {self.format_hover_value(float(cad_point[1]), unit)}"
                    )
                else:
                    if not self.show_profile_points_var.get():
                        continue

                    containing_arc = avg_index_to_arc.get(index)

                    if containing_arc is None:
                        continue

                    hover_text = (
                        f"Average Arc {containing_arc.segment_number}\n"
                        f"R = {self.format_hover_value(float(containing_arc.radius), unit)}"
                    )

                self.average_hover_points.append({
                    "xy": np.array([float(image_point[0]), float(image_point[1])], dtype=float),
                    "text": hover_text,
                })

        if self.show_profile_points_var.get():
            self.ax_average.plot(
                average_px[:, 0],
                average_px[:, 1],
                marker="o",
                markersize=3.2,
                linewidth=1.8,
            )

            if self.show_mirror_var.get():
                self.ax_average.plot(
                    average_mirror_px[:, 0],
                    average_mirror_px[:, 1],
                    marker="o",
                    markersize=3.2,
                    linewidth=1.3,
                    linestyle="--",
                )

        boundary_indices = [index for index in boundary_labels.keys() if 0 <= index < len(average_px)]

        if boundary_indices:
            avg_boundary_px = average_px[boundary_indices]
            self.ax_average.scatter(
                avg_boundary_px[:, 0],
                avg_boundary_px[:, 1],
                s=60,
                marker="s",
                zorder=8,
            )

            if self.show_mirror_var.get():
                avg_boundary_mirror_px = average_mirror_px[boundary_indices]
                self.ax_average.scatter(
                    avg_boundary_mirror_px[:, 0],
                    avg_boundary_mirror_px[:, 1],
                    s=60,
                    marker="s",
                    zorder=8,
                )

            self.draw_selected_boundary_marker(self.ax_average, average_px, "average")
            self.draw_selected_boundary_marker(self.ax_average, average_mirror_px, "average_mirror")

        if self.show_point_numbers_var.get():
            for index, point in enumerate(average_px):
                self.ax_average.text(point[0], point[1], str(index), fontsize=8)

            if self.show_mirror_var.get():
                for index, point in enumerate(average_mirror_px):
                    self.ax_average.text(point[0], point[1], str(index), fontsize=8)

        if self.show_fitted_arcs_var.get():
            for result in average_arc_results:
                arc_cad_xy = self.result_arc_points(result, sample_count=120)
                arc_base_xy = self.cad_xy_to_base_xy_array(arc_cad_xy)
                arc_px = self.base_profile_xy_to_image_points(arc_base_xy)

                self.ax_average.plot(
                    arc_px[:, 0],
                    arc_px[:, 1],
                    linewidth=2.4,
                )
                self.average_hover_arcs.append({
                    "xy": arc_px,
                    "text": f"Average Arc {result.segment_number}\nR = {self.format_hover_value(float(result.radius), unit)}",
                })

                if self.show_mirror_var.get():
                    mirrored_arc_base_xy = np.array(arc_base_xy, dtype=float).copy()
                    mirrored_arc_base_xy[:, 0] *= -1.0
                    mirrored_arc_px = self.base_profile_xy_to_image_points(mirrored_arc_base_xy)

                    self.ax_average.plot(
                        mirrored_arc_px[:, 0],
                        mirrored_arc_px[:, 1],
                        linewidth=1.4,
                        linestyle="--",
                    )
                    self.average_hover_arcs.append({
                        "xy": mirrored_arc_px,
                        "text": f"Average Arc {result.segment_number}\nR = {self.format_hover_value(float(result.radius), unit)}",
                    })

        self.average_hover_annotation = self.create_hover_annotation_for_axes(self.ax_average)

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

        length, width, dim_units = self.get_dimension_inputs()
        if length is not None or width is not None:
            length_text = f"L={length:g}" if length is not None else "L=auto"
            width_text = f"W={width:g}" if width is not None else "W=auto"
            self.scale_info_var.set(
                f"Dimension scaling: {length_text}, {width_text} {dim_units}"
            )
        elif self.has_real_scale:
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
                f"max half-width {self.auto_profile_report.max_half_width:.2f} {self.get_cad_units_label()}"
            )
        elif self.profile_points_px:
            self.profile_info_var.set(f"Profile: {len(self.profile_points_px)} pts")
        else:
            self.profile_info_var.set("Auto profile: not extracted")

    def update_results_text(self):
        full_text = self.results_summary_text()
        arc_summary = self.arc_summary_table_text()

        if hasattr(self, "results_text") and self.results_text is not None:
            self.results_text.configure(state="normal")
            self.results_text.delete("1.0", tk.END)
            self.results_text.insert("1.0", full_text)
            self.results_text.configure(state="disabled")

        if hasattr(self, "arc_summary_text") and self.arc_summary_text is not None:
            self.arc_summary_text.configure(state="normal")
            self.arc_summary_text.delete("1.0", tk.END)
            self.arc_summary_text.insert("1.0", arc_summary)
            self.arc_summary_text.configure(state="disabled")

    def get_raw_pixel_profile_xy_array(self) -> np.ndarray:
        """
        Return the current one-side profile in raw image-pixel-derived units.

        get_base_profile_xy_array() uses self.scale_per_px, so divide it back
        out to get the pixel coordinate system.
        """
        base_xy = self.get_base_profile_xy_array()
        scale = max(float(self.scale_per_px), 1e-12)

        return base_xy / scale

    def get_arc_summary_profiles_and_units(self):
        requested_units = self.arc_summary_units_var.get()
        if requested_units not in ARC_SUMMARY_UNIT_OPTIONS:
            requested_units = "px"
            self.arc_summary_units_var.set(requested_units)

        right_xy, unit, note = self.convert_base_profile_xy_to_requested_units(
            self.get_right_base_profile_xy_array(),
            requested_units,
        )
        left_xy, _, _ = self.convert_base_profile_xy_to_requested_units(
            self.get_left_base_profile_xy_array(),
            requested_units,
        )
        avg_xy, _, _ = self.convert_base_profile_xy_to_requested_units(
            self.get_average_base_profile_xy_array(),
            requested_units,
        )

        return {
            "right": right_xy,
            "left": left_xy,
            "average": avg_xy,
        }, unit, note

    def build_arc_results_from_profile_xy(self, profile_xy: np.ndarray) -> List[ArcResult]:
        """
        Rebuild arc results for a supplied coordinate system using the current
        boundary indices.

        This lets the 8-Arc Summary display the same arc boundaries in pixels,
        mm, or inches without changing the main plotted result.
        """
        ranges = self.get_segment_ranges()
        results = []

        for segment_number, (start_index, end_index) in enumerate(ranges[:ARC_COUNT], start=1):
            segment_points = self.segment_points_from_range(
                profile_xy,
                start_index,
                end_index,
                include_neighbors=False,
            )

            if len(segment_points) < 3:
                segment_points = self.segment_points_from_range(
                    profile_xy,
                    start_index,
                    end_index,
                    include_neighbors=True,
                )

            if len(segment_points) < 3:
                continue

            center, radius, errors = self.fit_circle_with_convex_preference(segment_points)

            arc_angle_points = self.segment_points_from_range(
                profile_xy,
                start_index,
                end_index,
                include_neighbors=False,
            )
            if len(arc_angle_points) < 2:
                arc_angle_points = segment_points

            angles = unwrap_segment_angles(arc_angle_points, center)
            theta_start = float(angles[0])
            theta_end = float(angles[-1])
            included_angle_rad = theta_end - theta_start
            included_angle_degrees = math.degrees(included_angle_rad)

            direction = "CCW" if included_angle_degrees >= 0 else "CW"
            arc_length = abs(float(radius) * included_angle_rad)

            start_point = profile_xy[start_index]
            end_point = profile_xy[end_index]
            chord_length = float(np.linalg.norm(end_point - start_point))

            arc_profile_points = self.segment_points_from_range(
                profile_xy,
                start_index,
                end_index,
                include_neighbors=False,
            )
            distances = np.linalg.norm(arc_profile_points - center, axis=1)
            error_values = distances - radius
            rms_error = float(math.sqrt(np.mean(error_values * error_values))) if len(error_values) else 0.0
            max_error = float(np.max(np.abs(error_values))) if len(error_values) else 0.0

            result = ArcResult(
                segment_number=segment_number,
                start_index=start_index,
                end_index=end_index,
                point_count=abs(end_index - start_index) + 1,
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

        return results

    def get_arc_summary_data(self):
        """
        Build the compact data needed for the 8-Arc Summary.

        Returns:
            unit, unit_note, left_points, right_points, avg_points,
            left_radii, right_radii, avg_radii
        """
        if not self.arc_results:
            raise ValueError("Fit arcs before generating the 8-arc summary.")

        summary_profiles, unit, unit_note = self.get_arc_summary_profiles_and_units()

        right_results = self.build_arc_results_from_profile_xy(summary_profiles["right"])
        left_results = self.build_arc_results_from_profile_xy(summary_profiles["left"])
        average_results = self.build_arc_results_from_profile_xy(summary_profiles["average"])

        right_map = {result.segment_number: result for result in right_results}
        left_map = {result.segment_number: result for result in left_results}
        avg_map = {result.segment_number: result for result in average_results}

        def get_point_chain(results_map):
            first = results_map.get(1)
            points = []

            if first is None:
                points.append(None)
            else:
                points.append((first.start_x, first.start_y))

            for arc_number in range(1, ARC_COUNT + 1):
                result = results_map.get(arc_number)
                points.append(None if result is None else (result.end_x, result.end_y))

            return points

        def get_radii(results_map):
            radii = []

            for arc_number in range(1, ARC_COUNT + 1):
                result = results_map.get(arc_number)
                radii.append(None if result is None else result.radius)

            return radii

        return (
            unit,
            unit_note,
            get_point_chain(left_map),
            get_point_chain(right_map),
            get_point_chain(avg_map),
            get_radii(left_map),
            get_radii(right_map),
            get_radii(avg_map),
        )

    def arc_summary_table_text(self) -> str:
        lines = []
        lines.append("8-Arc Summary")
        lines.append("=" * 32)
        lines.append("")

        if self.image_path:
            lines.append(f"Image: {os.path.basename(self.image_path)}")
        else:
            lines.append("Image: not loaded")

        lines.append(f"Requested summary units: {self.arc_summary_units_var.get()}")
        lines.append("")

        if not self.arc_results:
            lines.append("No arc results yet.")
            lines.append("")
            lines.append("Run Auto Process All before copying/exporting the summary.")
            return "\n".join(lines)

        try:
            (
                unit,
                unit_note,
                left_points,
                right_points,
                avg_points,
                left_radii,
                right_radii,
                avg_radii,
            ) = self.get_arc_summary_data()
        except Exception as exc:
            lines.append(f"Could not build summary: {exc}")
            return "\n".join(lines)

        lines.append(f"Units: {unit}")
        if unit_note:
            lines.append(f"Note: {unit_note}")

        lines.append("")
        lines.append("P0-P8 coordinates")
        lines.append("-" * 72)
        lines.append("Point | Left X | Left Y | Right X | Right Y | Avg X | Avg Y")
        lines.append("-" * 72)

        def fmt_value(value):
            return "" if value is None else f"{value:.6f}"

        for point_number in range(ARC_COUNT + 1):
            left_point = left_points[point_number]
            right_point = right_points[point_number]
            avg_point = avg_points[point_number]

            lx = None if left_point is None else left_point[0]
            ly = None if left_point is None else left_point[1]
            rx = None if right_point is None else right_point[0]
            ry = None if right_point is None else right_point[1]
            ax = None if avg_point is None else avg_point[0]
            ay = None if avg_point is None else avg_point[1]

            lines.append(
                f"P{point_number} | "
                f"{fmt_value(lx)} | {fmt_value(ly)} | "
                f"{fmt_value(rx)} | {fmt_value(ry)} | "
                f"{fmt_value(ax)} | {fmt_value(ay)}"
            )

        lines.append("")
        lines.append("A1-A8 radii")
        lines.append("-" * 44)
        lines.append("Arc | Left R | Right R | Avg R")
        lines.append("-" * 44)

        for arc_number in range(1, ARC_COUNT + 1):
            lines.append(
                f"A{arc_number} | "
                f"{fmt_value(left_radii[arc_number - 1])} | "
                f"{fmt_value(right_radii[arc_number - 1])} | "
                f"{fmt_value(avg_radii[arc_number - 1])}"
            )

        return "\n".join(lines)

    def copy_arc_summary(self):
        text = self.arc_summary_table_text()
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_var.set("Copied 8-arc summary to clipboard.")

    def export_arc_summary_csv(self):
        """
        Export only the compact 8-Arc Summary.

        The CSV contains:
            - P0-P8 coordinates for left/right/average
            - A1-A8 radii for left/right/average
        """
        if not self.arc_results:
            messagebox.showwarning("No results", "Run Auto Process All before exporting the 8-arc summary.")
            return

        if self.image_path:
            initial_dir = os.path.dirname(self.image_path)
            base = os.path.splitext(os.path.basename(self.image_path))[0]
            initial_file = f"{base}_8_arc_summary.csv"
        else:
            initial_dir = os.getcwd()
            initial_file = "8_arc_summary.csv"

        path = filedialog.asksaveasfilename(
            title="Export 8-Arc Summary",
            initialdir=initial_dir,
            initialfile=initial_file,
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
        )

        if not path:
            return

        try:
            (
                unit,
                unit_note,
                left_points,
                right_points,
                avg_points,
                left_radii,
                right_radii,
                avg_radii,
            ) = self.get_arc_summary_data()

            with open(path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)

                writer.writerow(["image", self.image_path or ""])
                writer.writerow(["units", unit])
                if unit_note:
                    writer.writerow(["note", unit_note])
                writer.writerow([])

                writer.writerow(["points"])
                writer.writerow([
                    "point",
                    "left_x",
                    "left_y",
                    "right_x",
                    "right_y",
                    "avg_x",
                    "avg_y",
                ])

                for point_number in range(ARC_COUNT + 1):
                    left_point = left_points[point_number]
                    right_point = right_points[point_number]
                    avg_point = avg_points[point_number]

                    writer.writerow([
                        f"P{point_number}",
                        "" if left_point is None else f"{left_point[0]:.9f}",
                        "" if left_point is None else f"{left_point[1]:.9f}",
                        "" if right_point is None else f"{right_point[0]:.9f}",
                        "" if right_point is None else f"{right_point[1]:.9f}",
                        "" if avg_point is None else f"{avg_point[0]:.9f}",
                        "" if avg_point is None else f"{avg_point[1]:.9f}",
                    ])

                writer.writerow([])
                writer.writerow(["radii"])
                writer.writerow([
                    "arc",
                    "left_radius",
                    "right_radius",
                    "avg_radius",
                ])

                for arc_number in range(1, ARC_COUNT + 1):
                    writer.writerow([
                        f"A{arc_number}",
                        "" if left_radii[arc_number - 1] is None else f"{left_radii[arc_number - 1]:.9f}",
                        "" if right_radii[arc_number - 1] is None else f"{right_radii[arc_number - 1]:.9f}",
                        "" if avg_radii[arc_number - 1] is None else f"{avg_radii[arc_number - 1]:.9f}",
                    ])

            self.status_var.set(f"Exported 8-Arc Summary: {path}")

        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    def results_summary_text(self) -> str:
        unit = self.get_cad_units_label()

        lines = []
        lines.append("Arc Profile Finder Results")
        lines.append("=" * 32)
        lines.append("")

        if self.image_path:
            lines.append(f"Image: {os.path.basename(self.image_path)}")
        else:
            lines.append("Image: not loaded")

        lines.append(f"Units: {unit}")

        length, width, dim_units = self.get_dimension_inputs()
        if length is not None or width is not None:
            length_text = f"{length:g}" if length is not None else "auto/uniform"
            width_text = f"{width:g}" if width is not None else "auto/uniform"
            lines.append(f"Part length input: {length_text} {dim_units}")
            lines.append(f"Part width input: {width_text} {dim_units}")
        elif self.has_real_scale:
            lines.append(f"Scale: {self.scale_per_px:.8f} {unit}/px")
        else:
            lines.append("Scale: not set; values are pixels")

        lines.append(f"Arc count required: {ARC_COUNT}")
        lines.append(f"Arc distribution: {self.arc_distribution_var.get()}")
        lines.append("Widest point is forced to be the start of one arc.")
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

        lines.append("Arc start/end/radius summary")
        lines.append("-" * 32)
        for result in self.arc_results:
            lines.append(
                f"Arc {result.segment_number}: "
                f"start=({result.start_x:.6f}, {result.start_y:.6f}) {unit}, "
                f"end=({result.end_x:.6f}, {result.end_y:.6f}) {unit}, "
                f"R={result.radius:.6f} {unit}"
            )
        lines.append("")

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
                writer.writerow(["units", self.get_cad_units_label()])
                writer.writerow(["scale_per_px", self.scale_per_px])
                writer.writerow(["has_real_scale", self.has_real_scale])
                length, width, dim_units = self.get_dimension_inputs()
                writer.writerow(["part_length", "" if length is None else length])
                writer.writerow(["part_width", "" if width is None else width])
                writer.writerow(["dimension_units", dim_units])
                writer.writerow(["arc_count_required", ARC_COUNT])
                writer.writerow(["arc_distribution", self.arc_distribution_var.get()])

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
                    writer.writerow(["auto_profile_edge_offset_px", self.auto_profile_report.edge_offset_px])
                    writer.writerow(["auto_profile_point_count", self.auto_profile_report.point_count])
                    writer.writerow(["auto_profile_max_half_width", self.auto_profile_report.max_half_width])

                if self.auto_breakpoint_report is not None:
                    writer.writerow(["auto_breakpoints", ",".join(str(i) for i in self.auto_breakpoint_report.breakpoint_indices)])
                    writer.writerow(["auto_arc_tolerance", self.auto_breakpoint_report.tolerance])
                    writer.writerow(["auto_arc_count", self.auto_breakpoint_report.max_arcs])
                    writer.writerow(["auto_arc_distribution", self.auto_breakpoint_report.distribution_rule])
                    writer.writerow(["auto_widest_point_index", self.auto_breakpoint_report.widest_index])

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

                    profile_xy = self.get_cad_profile_xy_array()

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
