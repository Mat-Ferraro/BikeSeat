#!/usr/bin/env python3
"""
Seatpost Orientation Tool - Manual Model Rotation + Undo + Masking + Cross Sections

This version uses PyVista/VTK for a hardware-accelerated surface viewer,
allows an interactive 3D box to limit which scan vertices are used for
auto-orientation, provides explicit X/Y/Z controls for rotating that box,
adds camera snap views for looking directly along each world axis, and can
extract a horizontal 2D cross section from the oriented seatpost.

Install:
    python -m pip install trimesh numpy pyvista pyvistaqt qtpy PySide6

Run:
    python seatpost_orientation_manual_model_rotation.py

GUI workflow:
    1. Load Scan
    2. Optional: manually rotate the current model around world X/Y/Z
    3. Optional: open Orientation Mask and place the box around only the post
    4. Auto Orient Vertical
    5. Manually fine-tune the model rotation and/or Flip Top/Bottom
    6. Open the Cross Section tab
    7. Select a Z height and click Generate / Update Slice
    8. Save the oriented mesh and/or export the 2D cross section

Cross-section export formats:
    - PNG: the same blue 2D slice outline shown in the app, on white
    - JPG/JPEG: the same blue 2D slice outline shown in the app, on white

Notes:
    - The viewer defaults to a shaded surface. Fast Points remains available
      for unusually large scans.
    - The orientation box can be created around the current mesh at any time,
      including after an earlier auto-orientation attempt. Creating the box
      does not undo or reset the current rotation.
    - The mask tab includes explicit +/- X, Y, and Z rotation controls so the
      box can be spun without relying on difficult freehand 3D dragging.
    - The Manual Model Rotation controls rotate the current mesh around its
      own bounding-box center in adjustable world-X, world-Y, or world-Z steps.
      They work both before and after auto-orientation.
    - An optional checkbox recenters X/Y and returns the mesh bottom to Z=0
      after each manual rotation. Pure center rotation is used when unchecked.
    - If an orientation box is active, manual model rotation carries the box
      and its selected region along with the mesh instead of invalidating it.
    - Ctrl+Z undoes the last geometry, box, camera, slice-height, or slice
      generation command, including each manual model rotation.
    - Orientation-box handles are forced back to a constant size after every
      interaction and are colored red for visibility.
    - Camera buttons snap the view to +/- X, +/- Y, +/- Z, isometric, or fit
      the current view while keeping the current viewing direction.
    - When "Keep only geometry inside the box" is enabled, pressing
      "Auto Orient Using Current Selection" crops away faces outside the box
      before applying the new orientation. The originally loaded mesh remains
      available through "Restore Original Loaded Mesh."
    - Slice coordinates are written in the mesh's native units, probably
      millimeters for your scans.
    - The horizontal slice is taken perpendicular to world Z after orientation.
    - Cross-section extraction does not require the optional NetworkX package.
    - Saved cross-section images match the 2D preview framing and blue
      outline, but omit the preview axes/grid, measurement text, labels,
      and all 3D scene elements.
    - This file intentionally uses qtpy + QT_API=pyside6 to avoid Qt binding
      mismatch errors.
"""

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh


try:
    # Force pyvistaqt/qtpy to use the same Qt binding everywhere.
    os.environ.setdefault("QT_API", "pyside6")

    import pyvista as pv
    from qtpy import QtCore, QtGui, QtWidgets
    from pyvistaqt import QtInteractor
    from vtkmodules.vtkCommonMath import vtkMatrix4x4
    from vtkmodules.vtkCommonTransforms import vtkTransform
except ImportError as exc:
    print()
    print("Missing GUI dependency.")
    print("Install the required packages with:")
    print("    python -m pip install trimesh numpy pyvista pyvistaqt qtpy PySide6")
    print()
    print(f"Original import error: {exc}")
    sys.exit(1)


Z_AXIS = np.array([0.0, 0.0, 1.0])
SUPPORTED_EXTENSIONS = {".obj", ".stl", ".ply", ".off", ".glb", ".gltf"}
UNDO_LIMIT = 12
ORIENTATION_BOX_HANDLE_SIZE = 0.006
ORIENTATION_BOX_HANDLE_COLOR = (1.0, 0.0, 0.0)
ORIENTATION_BOX_SELECTED_HANDLE_COLOR = (0.8, 0.0, 0.0)



@dataclass
class TransformReport:
    input_path: str = ""
    detected_axis: Optional[np.ndarray] = None
    fit_center: Optional[np.ndarray] = None
    rotation_angle_degrees: float = 0.0
    rotation_axis: Optional[np.ndarray] = None
    auto_pre_level_translation: Optional[np.ndarray] = None
    auto_level_translation: Optional[np.ndarray] = None
    flip_count: int = 0
    last_flip_level_translation: Optional[np.ndarray] = None
    total_transform: Optional[np.ndarray] = None
    original_bounds: Optional[np.ndarray] = None
    current_bounds: Optional[np.ndarray] = None
    orientation_source: str = "Full mesh"
    orientation_vertex_count: int = 0
    total_vertex_count: int = 0
    leveling_source: str = "Full mesh"
    level_reference_bounds: Optional[np.ndarray] = None
    orientation_pass_count: int = 0
    prior_transform_preserved: bool = False
    cropped_to_orientation_box: bool = False
    cropped_vertex_count: int = 0
    cropped_face_count: int = 0
    manual_rotation_count: int = 0
    manual_rotation_matrix: Optional[np.ndarray] = None
    last_manual_axis: str = ""
    last_manual_angle_degrees: float = 0.0
    last_manual_releveled: bool = False

    def summary_text(self) -> str:
        lines = []

        if self.input_path:
            lines.append(f"File: {os.path.basename(self.input_path)}")
            lines.append("")

        if self.original_bounds is not None:
            lines.append("Original bounds:")
            lines.append(format_bounds(self.original_bounds))
            lines.append("")

        if self.manual_rotation_count:
            lines.append("Manual model adjustment:")
            lines.append(f"  Manual rotations applied: {self.manual_rotation_count}")
            lines.append(
                f"  Last rotation: {self.last_manual_angle_degrees:+.3f}° "
                f"around world {self.last_manual_axis or 'n/a'}"
            )
            lines.append(
                "  Recentered X/Y and leveled Z after last rotation: "
                + ("yes" if self.last_manual_releveled else "no")
            )
            if self.manual_rotation_matrix is not None:
                lines.append("  Cumulative manual-adjustment matrix:")
                lines.extend(format_matrix(self.manual_rotation_matrix))
            lines.append("")

        if self.detected_axis is None:
            lines.append("No auto-orientation has been applied yet.")
            lines.append("")
            lines.append(
                "You may continue rotating manually or click Auto Orient Vertical."
            )

            if self.total_transform is not None:
                lines.append("")
                lines.append("Current total transform matrix:")
                lines.extend(format_matrix(self.total_transform))

            if self.current_bounds is not None:
                lines.append("")
                lines.append("Current bounds:")
                lines.append(format_bounds(self.current_bounds))

            lines.append("")
            lines.append("Note: move values are in the mesh's native units.")
            return "\n".join(lines)

        lines.append("Auto-orientation:")
        lines.append(f"  Rotation angle to vertical: {self.rotation_angle_degrees:.3f}°")
        lines.append(f"  Detected post axis: {format_vector(self.detected_axis)}")
        lines.append(f"  Rotation axis: {format_vector(self.rotation_axis)}")
        lines.append(f"  Fit center used: {format_vector(self.fit_center)}")
        lines.append(f"  Orientation source: {self.orientation_source}")
        if self.total_vertex_count:
            lines.append(
                f"  Fit vertices: {self.orientation_vertex_count:,} / "
                f"{self.total_vertex_count:,}"
            )
        lines.append(f"  Center/level source: {self.leveling_source}")
        if self.orientation_pass_count:
            lines.append(f"  Orientation pass: {self.orientation_pass_count}")
        if self.prior_transform_preserved:
            lines.append("  Previous rotation/translation was preserved")
        lines.append("")

        lines.append("Geometry retained:")
        if self.cropped_to_orientation_box:
            lines.append("  Cropped to the orientation box: yes")
            lines.append(f"  Retained mesh vertices: {self.cropped_vertex_count:,}")
            lines.append(f"  Retained mesh faces: {self.cropped_face_count:,}")
        else:
            lines.append("  Cropped to the orientation box: no")
        lines.append("")

        lines.append("Move / translation:")
        lines.append(f"  Rotation-center translation: {format_vector(self.auto_pre_level_translation)}")
        lines.append(f"  Level + XY-center move: {format_vector(self.auto_level_translation)}")

        if self.flip_count:
            lines.append(f"  Flip count: {self.flip_count}")
            lines.append("  Additional flip rotation: 180.000° about X per flip")
            lines.append(f"  Last post-flip level move: {format_vector(self.last_flip_level_translation)}")
        else:
            lines.append("  Flip count: 0")

        if self.total_transform is not None:
            lines.append("")
            lines.append("Total transform matrix:")
            lines.extend(format_matrix(self.total_transform))
            lines.append("")
            lines.append(
                "Total transform translation column: "
                f"{format_vector(self.total_transform[:3, 3])}"
            )

        if self.current_bounds is not None:
            lines.append("")
            lines.append("Current bounds:")
            lines.append(format_bounds(self.current_bounds))

        lines.append("")
        lines.append("Note: move values are in the mesh's native units.")
        return "\n".join(lines)


@dataclass
class CrossSectionData:
    z_height: float
    polylines: list[np.ndarray]
    bounds_2d: np.ndarray
    total_length: float
    closed_loop_count: int
    point_count: int

    @property
    def width_x(self) -> float:
        return float(self.bounds_2d[1, 0] - self.bounds_2d[0, 0])

    @property
    def depth_y(self) -> float:
        return float(self.bounds_2d[1, 1] - self.bounds_2d[0, 1])

    def summary_text(self) -> str:
        return "\n".join([
            f"Slice Z height: {self.z_height:.4f}",
            "",
            "2D bounds:",
            f"  X min/max: {self.bounds_2d[0, 0]:.4f} / {self.bounds_2d[1, 0]:.4f}",
            f"  Y min/max: {self.bounds_2d[0, 1]:.4f} / {self.bounds_2d[1, 1]:.4f}",
            f"  Width in X: {self.width_x:.4f}",
            f"  Depth in Y: {self.depth_y:.4f}",
            "",
            f"Polyline count: {len(self.polylines)}",
            f"Closed loops: {self.closed_loop_count}",
            f"Total points: {self.point_count}",
            f"Approx. total contour length: {self.total_length:.4f}",
            "",
            "Coordinates and dimensions use the mesh's native units.",
            "PNG/JPG exports match the blue 2D preview outline on white,",
            "without axes, grid lines, text, or 3D scene elements.",
        ])


def clone_transform_report(report):
    if report is None:
        return None

    values = {}
    for field_name in report.__dataclass_fields__:
        value = getattr(report, field_name)
        if isinstance(value, np.ndarray):
            value = np.array(value, copy=True)
        values[field_name] = value

    return TransformReport(**values)


def clone_cross_section_data(data):
    if data is None:
        return None

    return CrossSectionData(
        z_height=float(data.z_height),
        polylines=[np.array(polyline, copy=True) for polyline in data.polylines],
        bounds_2d=np.array(data.bounds_2d, copy=True),
        total_length=float(data.total_length),
        closed_loop_count=int(data.closed_loop_count),
        point_count=int(data.point_count),
    )


def format_vector(v) -> str:
    if v is None:
        return "n/a"

    return f"[{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}]"


def format_matrix(m) -> list[str]:
    if m is None:
        return ["  n/a"]

    return [
        "  [" + "  ".join(f"{value: .6f}" for value in row) + "]"
        for row in m
    ]


def format_bounds(bounds) -> str:
    if bounds is None:
        return "  n/a"

    return (
        f"  min {format_vector(bounds[0])}\n"
        f"  max {format_vector(bounds[1])}\n"
        f"  size {format_vector(bounds[1] - bounds[0])}"
    )


def bounds_corners(bounds):
    """Return the eight corner points of a 3D min/max bounds array."""
    bounds = np.asarray(bounds, dtype=float)

    if bounds.shape != (2, 3):
        raise ValueError("bounds must have shape (2, 3).")

    return np.array([
        [x, y, z]
        for x in (bounds[0, 0], bounds[1, 0])
        for y in (bounds[0, 1], bounds[1, 1])
        for z in (bounds[0, 2], bounds[1, 2])
    ], dtype=float)


def normalize(v):
    n = np.linalg.norm(v)

    if n < 1e-12:
        raise ValueError("Cannot normalize a near-zero vector.")

    return v / n


def translation_matrix(vector):
    transform = np.eye(4)
    transform[:3, 3] = vector
    return transform


def vtk_transform_to_numpy(transform):
    """Convert a vtkTransform into a NumPy 4x4 matrix."""
    vtk_matrix = transform.GetMatrix()
    return np.array([
        [vtk_matrix.GetElement(row, column) for column in range(4)]
        for row in range(4)
    ], dtype=float)


def numpy_to_vtk_transform(matrix):
    """Convert a NumPy 4x4 matrix into a vtkTransform."""
    matrix = np.asarray(matrix, dtype=float)

    if matrix.shape != (4, 4):
        raise ValueError("Transform matrix must have shape (4, 4).")

    vtk_matrix = vtkMatrix4x4()

    for row in range(4):
        for column in range(4):
            vtk_matrix.SetElement(row, column, float(matrix[row, column]))

    transform = vtkTransform()
    transform.SetMatrix(vtk_matrix)
    return transform


def rotation_matrix_axis_angle(axis, angle):
    axis = normalize(axis)

    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c

    return np.array([
        [c + x * x * C,      x * y * C - z * s,  x * z * C + y * s],
        [y * x * C + z * s,  c + y * y * C,      y * z * C - x * s],
        [z * x * C - y * s,  z * y * C + x * s,  c + z * z * C],
    ])


def rotation_matrix_from_vectors(source, target):
    source = normalize(source)
    target = normalize(target)

    cross = np.cross(source, target)
    dot = np.dot(source, target)

    if dot > 1.0 - 1e-10:
        return np.eye(3), np.array([0.0, 0.0, 0.0]), 0.0

    if dot < -1.0 + 1e-10:
        helper = np.array([1.0, 0.0, 0.0])

        if abs(np.dot(source, helper)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])

        axis = normalize(np.cross(source, helper))
        return rotation_matrix_axis_angle(axis, np.pi), axis, np.pi

    axis = normalize(cross)
    angle = math.acos(np.clip(dot, -1.0, 1.0))

    s = np.linalg.norm(cross)

    skew = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])

    matrix = np.eye(3) + skew + skew @ skew * ((1.0 - dot) / (s * s))
    return matrix, axis, angle


def pca_axis(points):
    center = points.mean(axis=0)
    centered = points - center

    covariance = centered.T @ centered / max(len(points) - 1, 1)

    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, np.argmax(values)]

    return normalize(axis), center


def radial_spread(points, axis, center, mask):
    pts = points[mask]

    if len(pts) < 10:
        return 0.0

    t = (pts - center) @ axis
    closest = center + np.outer(t, axis)
    radii = np.linalg.norm(pts - closest, axis=1)

    return float(np.percentile(radii, 95))


def choose_bulky_end_up(points, axis, center):
    """
    PCA finds an axis, but it does not know which direction is "up."

    This assumes the saddle-clamp/head end is bulkier than the inserted/lower end.
    If the low side is bulkier, the axis is flipped so the bulky end becomes +Z.
    """
    axis = normalize(axis)
    t = (points - center) @ axis

    low_mask = t <= np.percentile(t, 15)
    high_mask = t >= np.percentile(t, 85)

    low_spread = radial_spread(points, axis, center, low_mask)
    high_spread = radial_spread(points, axis, center, high_mask)

    if low_spread > high_spread:
        axis = -axis

    return normalize(axis)


def estimate_seatpost_axis(vertices):
    """
    Two-pass axis estimate:
        1. Find the rough long axis from all vertices.
        2. Refit using the central 90% of the length.
        3. Flip the direction so the bulkier end points upward.
    """
    rough_axis, rough_center = pca_axis(vertices)
    rough_t = (vertices - rough_center) @ rough_axis

    lo = np.percentile(rough_t, 5)
    hi = np.percentile(rough_t, 95)
    central = vertices[(rough_t >= lo) & (rough_t <= hi)]

    if len(central) > 100:
        axis, center = pca_axis(central)

        if np.dot(axis, rough_axis) < 0:
            axis = -axis
    else:
        axis, center = rough_axis, rough_center

    axis = choose_bulky_end_up(vertices, axis, center)
    return axis, center


def load_mesh(path):
    mesh = trimesh.load(path, force="mesh", process=False)

    if isinstance(mesh, trimesh.Scene):
        geometries = tuple(mesh.geometry.values())

        if not geometries:
            raise ValueError(f"No geometry found in {path}")

        mesh = trimesh.util.concatenate(geometries)

    if mesh.vertices is None or len(mesh.vertices) < 3:
        raise ValueError(f"Mesh has too few vertices: {path}")

    return mesh


def crop_mesh_to_vertex_mask(mesh, vertex_mask):
    """
    Keep only triangle faces whose vertices are all inside a selection mask.

    The scans used by this tool are dense, so retaining complete triangles is
    a stable way to remove the saddle, tire, floor, and detached fragments
    while leaving a clean mesh that can still be exported and sliced.
    """
    mesh = mesh.copy()
    vertex_mask = np.asarray(vertex_mask, dtype=bool)

    if len(vertex_mask) != len(mesh.vertices):
        raise ValueError(
            "The orientation mask no longer matches the current mesh. "
            "Create a new orientation box and try again."
        )

    if int(np.count_nonzero(vertex_mask)) < 3:
        raise ValueError("The orientation box contains too few mesh vertices.")

    faces = np.asarray(mesh.faces, dtype=np.int64)

    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError(
            "Cropping currently requires a triangular surface mesh. "
            "Disable 'Keep only geometry inside the box' to orient without "
            "cropping."
        )

    keep_faces = np.all(vertex_mask[faces], axis=1)
    kept_face_count = int(np.count_nonzero(keep_faces))

    if kept_face_count == 0:
        raise ValueError(
            "No complete triangle faces are fully inside the orientation box. "
            "Enlarge the box slightly and try again."
        )

    cropped = mesh.submesh([keep_faces], append=True, repair=False)

    if isinstance(cropped, list):
        if not cropped:
            raise ValueError("The orientation box did not retain any mesh.")
        cropped = trimesh.util.concatenate(cropped)

    cropped.remove_unreferenced_vertices()

    if len(cropped.vertices) < 3 or len(cropped.faces) == 0:
        raise ValueError("The cropped orientation region is empty.")

    return cropped


def center_xy_and_level_z(mesh, reference_points=None):
    """
    Center the mesh around X/Y = 0 and place the reference bottom at Z = 0.

    When reference_points is omitted, the full mesh bounds are used. When a
    user-created orientation mask is supplied, the full mesh is still moved,
    but the selected seatpost region determines the centering and Z level.
    """
    if reference_points is None:
        bounds = np.asarray(mesh.bounds, dtype=float)
    else:
        reference_points = np.asarray(reference_points, dtype=float)

        if reference_points.ndim != 2 or reference_points.shape[1] != 3:
            raise ValueError("reference_points must have shape (N, 3).")

        reference_points = reference_points[
            np.all(np.isfinite(reference_points), axis=1)
        ]

        if len(reference_points) < 3:
            raise ValueError("Too few valid reference points for centering.")

        bounds = np.vstack((
            reference_points.min(axis=0),
            reference_points.max(axis=0),
        ))

    xy_center = (bounds[0, :2] + bounds[1, :2]) / 2.0
    z_min = bounds[0, 2]

    move = np.array([-xy_center[0], -xy_center[1], -z_min])
    transform = translation_matrix(move)

    mesh.apply_transform(transform)
    return move, transform


def auto_orient_mesh(
    mesh,
    input_path="",
    orientation_vertices=None,
    orientation_source="Full mesh",
    use_orientation_vertices_for_leveling=False,
):
    """
    Rotate the seatpost's detected main axis to world Z.

    orientation_vertices may contain only the vertices inside the interactive
    orientation box. The calculated transform is still applied to the entire
    mesh, so unwanted scan geometry is ignored without being deleted.
    """
    original_mesh = mesh
    mesh = mesh.copy()

    all_vertices = np.asarray(mesh.vertices, dtype=float)
    all_vertices = all_vertices[np.all(np.isfinite(all_vertices), axis=1)]

    if orientation_vertices is None:
        fit_vertices = all_vertices
        orientation_source = "Full mesh"
    else:
        fit_vertices = np.asarray(orientation_vertices, dtype=float)
        fit_vertices = fit_vertices[np.all(np.isfinite(fit_vertices), axis=1)]

    if len(fit_vertices) < 20:
        raise ValueError(
            "The orientation selection contains too few vertices. "
            "Resize the box so it contains more of the seatpost."
        )

    original_bounds = np.array(original_mesh.bounds, dtype=float)

    axis, center = estimate_seatpost_axis(fit_vertices)
    rotation, rotation_axis, rotation_angle = rotation_matrix_from_vectors(
        axis,
        Z_AXIS,
    )

    rotate_about_fit_center = np.eye(4)
    rotate_about_fit_center[:3, :3] = rotation
    rotate_about_fit_center[:3, 3] = -rotation @ center

    mesh.apply_transform(rotate_about_fit_center)

    leveling_reference = None
    leveling_source = "Full mesh"

    if use_orientation_vertices_for_leveling:
        fit_homogeneous = np.column_stack((
            fit_vertices,
            np.ones(len(fit_vertices), dtype=float),
        ))
        leveling_reference = (
            rotate_about_fit_center @ fit_homogeneous.T
        ).T[:, :3]
        leveling_source = orientation_source

    level_move, level_transform = center_xy_and_level_z(
        mesh,
        reference_points=leveling_reference,
    )
    total_transform = level_transform @ rotate_about_fit_center

    if leveling_reference is None:
        level_reference_bounds = np.array(mesh.bounds, dtype=float)
    else:
        leveled_reference = leveling_reference + level_move
        level_reference_bounds = np.vstack((
            leveled_reference.min(axis=0),
            leveled_reference.max(axis=0),
        ))

    report = TransformReport(
        input_path=input_path,
        detected_axis=axis,
        fit_center=center,
        rotation_angle_degrees=math.degrees(rotation_angle),
        rotation_axis=rotation_axis,
        auto_pre_level_translation=rotate_about_fit_center[:3, 3],
        auto_level_translation=level_move,
        flip_count=0,
        last_flip_level_translation=None,
        total_transform=total_transform,
        original_bounds=original_bounds,
        current_bounds=np.array(mesh.bounds, dtype=float),
        orientation_source=orientation_source,
        orientation_vertex_count=len(fit_vertices),
        total_vertex_count=len(all_vertices),
        leveling_source=leveling_source,
        level_reference_bounds=level_reference_bounds,
        orientation_pass_count=1,
        prior_transform_preserved=False,
        cropped_to_orientation_box=False,
        cropped_vertex_count=len(mesh.vertices),
        cropped_face_count=len(mesh.faces),
    )

    return mesh, report

def flip_top_bottom(mesh, report):
    """
    Flip the vertical mesh 180 degrees so top and bottom swap.
    """
    mesh = mesh.copy()

    flip_rotation = np.eye(4)
    flip_rotation[:3, :3] = rotation_matrix_axis_angle(
        np.array([1.0, 0.0, 0.0]),
        np.pi,
    )

    mesh.apply_transform(flip_rotation)

    flipped_reference = None

    if report.level_reference_bounds is not None:
        reference_corners = bounds_corners(report.level_reference_bounds)
        reference_homogeneous = np.column_stack((
            reference_corners,
            np.ones(len(reference_corners), dtype=float),
        ))
        flipped_reference = (
            flip_rotation @ reference_homogeneous.T
        ).T[:, :3]

    level_move, level_transform = center_xy_and_level_z(
        mesh,
        reference_points=flipped_reference,
    )
    flip_total = level_transform @ flip_rotation

    if flipped_reference is None:
        level_reference_bounds = np.array(mesh.bounds, dtype=float)
    else:
        leveled_reference = flipped_reference + level_move
        level_reference_bounds = np.vstack((
            leveled_reference.min(axis=0),
            leveled_reference.max(axis=0),
        ))

    if report.total_transform is None:
        total_transform = flip_total
    else:
        total_transform = flip_total @ report.total_transform

    new_report = TransformReport(
        input_path=report.input_path,
        detected_axis=report.detected_axis,
        fit_center=report.fit_center,
        rotation_angle_degrees=report.rotation_angle_degrees,
        rotation_axis=report.rotation_axis,
        auto_pre_level_translation=report.auto_pre_level_translation,
        auto_level_translation=report.auto_level_translation,
        flip_count=report.flip_count + 1,
        last_flip_level_translation=level_move,
        total_transform=total_transform,
        original_bounds=report.original_bounds,
        current_bounds=np.array(mesh.bounds, dtype=float),
        orientation_source=report.orientation_source,
        orientation_vertex_count=report.orientation_vertex_count,
        total_vertex_count=report.total_vertex_count,
        leveling_source=report.leveling_source,
        level_reference_bounds=level_reference_bounds,
        orientation_pass_count=report.orientation_pass_count,
        prior_transform_preserved=report.prior_transform_preserved,
        cropped_to_orientation_box=report.cropped_to_orientation_box,
        cropped_vertex_count=report.cropped_vertex_count,
        cropped_face_count=report.cropped_face_count,
        manual_rotation_count=report.manual_rotation_count,
        manual_rotation_matrix=(
            None
            if report.manual_rotation_matrix is None
            else np.array(report.manual_rotation_matrix, copy=True)
        ),
        last_manual_axis=report.last_manual_axis,
        last_manual_angle_degrees=report.last_manual_angle_degrees,
        last_manual_releveled=report.last_manual_releveled,
    )

    return mesh, new_report


def manual_rotate_mesh(
    mesh,
    axis_name,
    angle_degrees,
    relevel_after_rotation=False,
    level_reference_bounds=None,
):
    """
    Rotate the current mesh around its own bounding-box center.

    The axis is a world X/Y/Z axis.  When relevel_after_rotation is enabled,
    the rotated mesh is centered in X/Y and its lowest Z is moved to zero.
    Returns the rotated mesh, the incremental 4x4 transform, and transformed
    level-reference bounds for later flip/reset bookkeeping.
    """
    axis_name = str(axis_name).lower().strip()
    axis_vectors = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }

    if axis_name not in axis_vectors:
        raise ValueError("axis_name must be 'x', 'y', or 'z'.")

    angle_degrees = float(angle_degrees)
    if not np.isfinite(angle_degrees):
        raise ValueError("Manual rotation angle must be finite.")

    rotated = mesh.copy()
    bounds = np.asarray(rotated.bounds, dtype=float)
    center = (bounds[0] + bounds[1]) / 2.0

    rotation = np.eye(4)
    rotation[:3, :3] = rotation_matrix_axis_angle(
        axis_vectors[axis_name],
        math.radians(angle_degrees),
    )
    rotate_about_center = (
        translation_matrix(center)
        @ rotation
        @ translation_matrix(-center)
    )
    rotated.apply_transform(rotate_about_center)
    incremental_transform = rotate_about_center

    if relevel_after_rotation:
        _move, level_transform = center_xy_and_level_z(rotated)
        incremental_transform = level_transform @ incremental_transform

    transformed_reference_bounds = None
    if level_reference_bounds is not None:
        corners = bounds_corners(level_reference_bounds)
        homogeneous = np.column_stack((
            corners,
            np.ones(len(corners), dtype=float),
        ))
        transformed_corners = (
            incremental_transform @ homogeneous.T
        ).T[:, :3]
        transformed_reference_bounds = np.vstack((
            transformed_corners.min(axis=0),
            transformed_corners.max(axis=0),
        ))

    return rotated, incremental_transform, transformed_reference_bounds


def transform_pyvista_surface(surface, matrix):
    """Return a deep-copied PyVista surface transformed by a 4x4 matrix."""
    if surface is None:
        return None

    transformed = surface.copy(deep=True)
    points = np.asarray(transformed.points, dtype=float)
    if len(points):
        homogeneous = np.column_stack((
            points,
            np.ones(len(points), dtype=float),
        ))
        transformed.points = (matrix @ homogeneous.T).T[:, :3]
    return transformed


def save_mesh(mesh, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mesh.export(path)


def default_output_path(input_path, flipped=False):
    folder = os.path.dirname(os.path.abspath(input_path))
    base, ext = os.path.splitext(os.path.basename(input_path))

    suffix = "_vertical_flipped" if flipped else "_vertical"
    return os.path.join(folder, f"{base}{suffix}{ext}")


def _remove_consecutive_duplicate_points(points, tolerance=1e-10):
    if len(points) < 2:
        return points

    delta = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = delta > tolerance
    return points[keep]


def _extract_section_entity_curves(section):
    """
    Read the individual curves from a trimesh Path3D without using
    ``section.discrete``.

    Some trimesh versions build ``Path3D.discrete`` through NetworkX.  That
    makes a basic mesh slice fail when NetworkX is not installed, even though
    every path entity already knows how to discretize itself.  Iterating the
    entities directly avoids that optional dependency.
    """
    vertices = np.asarray(section.vertices, dtype=float)
    curves = []

    for entity in section.entities:
        curve = None

        try:
            curve = entity.discrete(vertices)
        except TypeError:
            # Compatibility with trimesh versions whose entity.discrete method
            # also expects a scale argument.
            scale = max(float(np.ptp(vertices, axis=0).max()), 1.0)
            curve = entity.discrete(vertices, scale=scale)
        except Exception:
            # Mesh-plane intersections normally create Line entities.  This
            # fallback still handles older Line implementations by reading the
            # referenced vertex indices directly.
            point_indices = getattr(entity, "points", None)
            if point_indices is not None:
                curve = vertices[np.asarray(point_indices, dtype=np.int64)]

        if curve is None:
            continue

        curve = np.asarray(curve, dtype=float)

        if curve.ndim == 2:
            curves.append(curve)
        elif curve.ndim == 3:
            curves.extend(np.asarray(part, dtype=float) for part in curve)

    return curves


def extract_horizontal_cross_section(mesh, z_height):
    """
    Intersect a triangular mesh with the horizontal plane Z = z_height.

    Returns connected XY polylines. The resulting coordinates remain in the
    oriented mesh's coordinate system.
    """
    if mesh is None:
        raise ValueError("No mesh is available.")

    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(
            "The loaded object has no triangle faces. "
            "A surface mesh is required to calculate a cross section."
        )

    bounds = np.asarray(mesh.bounds, dtype=float)
    z_min = float(bounds[0, 2])
    z_max = float(bounds[1, 2])

    tolerance = max((z_max - z_min) * 1e-8, 1e-9)

    if z_height < z_min - tolerance or z_height > z_max + tolerance:
        raise ValueError(
            f"Slice height {z_height:.4f} is outside the mesh range "
            f"{z_min:.4f} to {z_max:.4f}."
        )

    section = mesh.section(
        plane_origin=np.array([0.0, 0.0, float(z_height)]),
        plane_normal=Z_AXIS,
    )

    if section is None:
        raise ValueError(
            "The selected plane did not intersect the mesh. "
            "Try moving the slice slightly away from the very top or bottom."
        )

    polylines = []

    # Do not call section.discrete here.  In some trimesh versions that
    # property imports NetworkX.  The individual entities can be read directly
    # and produce the same slice geometry without requiring NetworkX.
    for curve in _extract_section_entity_curves(section):
        curve = np.asarray(curve, dtype=float)

        if curve.ndim != 2 or curve.shape[0] < 2 or curve.shape[1] < 2:
            continue

        curve_2d = curve[:, :2]
        curve_2d = curve_2d[np.all(np.isfinite(curve_2d), axis=1)]
        curve_2d = _remove_consecutive_duplicate_points(curve_2d)

        if len(curve_2d) >= 2:
            polylines.append(curve_2d)

    if not polylines:
        raise ValueError(
            "The slice intersection contained no usable contour lines. "
            "Try a nearby height."
        )

    all_points = np.vstack(polylines)
    bounds_2d = np.array([
        np.min(all_points, axis=0),
        np.max(all_points, axis=0),
    ])

    total_length = 0.0
    closed_loop_count = 0
    point_count = 0

    scale = max(
        float(bounds_2d[1, 0] - bounds_2d[0, 0]),
        float(bounds_2d[1, 1] - bounds_2d[0, 1]),
        1.0,
    )
    closed_tolerance = scale * 1e-6

    for polyline in polylines:
        point_count += len(polyline)
        total_length += float(np.linalg.norm(np.diff(polyline, axis=0), axis=1).sum())

        if len(polyline) >= 3 and np.linalg.norm(polyline[0] - polyline[-1]) <= closed_tolerance:
            closed_loop_count += 1

    return CrossSectionData(
        z_height=float(z_height),
        polylines=polylines,
        bounds_2d=bounds_2d,
        total_length=total_length,
        closed_loop_count=closed_loop_count,
        point_count=point_count,
    )


def cross_section_to_pyvista(data):
    """
    Convert 2D cross-section polylines into a PyVista line dataset at slice Z.
    """
    if data is None or not data.polylines:
        return None

    points = []
    line_cells = []
    start_index = 0

    for polyline in data.polylines:
        points_3d = np.column_stack([
            polyline[:, 0],
            polyline[:, 1],
            np.full(len(polyline), data.z_height),
        ])
        points.append(points_3d)

        indices = np.arange(start_index, start_index + len(polyline), dtype=np.int64)
        line_cells.append(np.concatenate(([len(indices)], indices)))
        start_index += len(polyline)

    polydata = pv.PolyData(np.vstack(points))
    polydata.lines = np.concatenate(line_cells)
    return polydata


def _iter_cross_section_segments(data):
    for polyline_index, polyline in enumerate(data.polylines):
        for point_index in range(len(polyline) - 1):
            yield (
                polyline_index,
                point_index,
                polyline[point_index],
                polyline[point_index + 1],
            )


def export_cross_section_csv(data, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "polyline",
            "point_index",
            "x",
            "y",
            "source_slice_z",
        ])

        for polyline_index, polyline in enumerate(data.polylines):
            for point_index, point in enumerate(polyline):
                writer.writerow([
                    polyline_index,
                    point_index,
                    f"{point[0]:.9f}",
                    f"{point[1]:.9f}",
                    f"{data.z_height:.9f}",
                ])


def export_cross_section_dxf(data, path):
    """
    Write a simple ASCII DXF using LINE entities.

    Units are intentionally left unitless because the source mesh units may be
    millimeters, inches, or another native unit.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    lines = [
        "0", "SECTION",
        "2", "HEADER",
        "0", "ENDSEC",
        "0", "SECTION",
        "2", "ENTITIES",
    ]

    for _, _, start, end in _iter_cross_section_segments(data):
        lines.extend([
            "0", "LINE",
            "8", "CROSS_SECTION",
            "10", f"{start[0]:.9f}",
            "20", f"{start[1]:.9f}",
            "30", "0.0",
            "11", f"{end[0]:.9f}",
            "21", f"{end[1]:.9f}",
            "31", "0.0",
        ])

    lines.extend([
        "0", "ENDSEC",
        "0", "EOF",
    ])

    with open(path, "w", encoding="ascii", newline="\n") as file:
        file.write("\n".join(lines))
        file.write("\n")


def export_cross_section_svg(data, path):
    """
    Write an SVG that preserves the mesh's XY coordinates in its viewBox.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    min_x, min_y = data.bounds_2d[0]
    max_x, max_y = data.bounds_2d[1]

    width = max(float(max_x - min_x), 1e-9)
    height = max(float(max_y - min_y), 1e-9)
    stroke_width = max(width, height) * 0.0025

    svg_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="{min_x:.9f} {-max_y:.9f} {width:.9f} {height:.9f}" '
            f'width="{width:.9f}" height="{height:.9f}">'
        ),
        (
            f'  <g fill="none" stroke="black" '
            f'stroke-width="{stroke_width:.9f}" vector-effect="non-scaling-stroke" '
            f'transform="scale(1,-1)">'
        ),
    ]

    for polyline in data.polylines:
        points_text = " ".join(
            f"{point[0]:.9f},{point[1]:.9f}"
            for point in polyline
        )
        svg_lines.append(f'    <polyline points="{points_text}" />')

    svg_lines.extend([
        "  </g>",
        "</svg>",
    ])

    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(svg_lines))
        file.write("\n")


def export_cross_section_image(
    data,
    path,
    image_size=2048,
    line_width=4.0,
    padding_percent=5.0,
    aspect_ratio=1.0,
):
    """
    Save a clean raster image of the 2D slice preview.

    The saved image uses the same blue contour styling as the in-app 2D
    preview. It intentionally omits the preview axes/grid, measurement text,
    labels, slice height, and all 3D viewer content.

    ``image_size`` is the output width. ``aspect_ratio`` is height / width.
    """
    extension = os.path.splitext(path)[1].lower()

    if extension not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("Cross-section images must use .png, .jpg, or .jpeg.")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    image_width = int(np.clip(int(image_size), 256, 8192))
    aspect_ratio = float(np.clip(float(aspect_ratio), 0.20, 5.0))
    image_height = int(np.clip(round(image_width * aspect_ratio), 256, 8192))
    line_width = float(np.clip(float(line_width), 1.0, 50.0))
    padding_fraction = float(np.clip(float(padding_percent), 0.0, 40.0)) / 100.0

    min_x, min_y = np.asarray(data.bounds_2d[0], dtype=float)
    max_x, max_y = np.asarray(data.bounds_2d[1], dtype=float)
    span_x = max(float(max_x - min_x), 1e-12)
    span_y = max(float(max_y - min_y), 1e-12)

    margin_x = image_width * padding_fraction
    margin_y = image_height * padding_fraction
    usable_width = max(image_width - 2.0 * margin_x, 1.0)
    usable_height = max(image_height - 2.0 * margin_y, 1.0)
    scale = min(usable_width / span_x, usable_height / span_y)

    drawn_width = span_x * scale
    drawn_height = span_y * scale
    left = (image_width - drawn_width) / 2.0
    top = (image_height - drawn_height) / 2.0

    image = QtGui.QImage(
        image_width,
        image_height,
        QtGui.QImage.Format_ARGB32,
    )
    image.fill(QtGui.QColor(255, 255, 255))

    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)

    contour_pen = QtGui.QPen(QtGui.QColor(20, 70, 150))
    contour_pen.setWidthF(line_width)
    contour_pen.setCapStyle(QtCore.Qt.RoundCap)
    contour_pen.setJoinStyle(QtCore.Qt.RoundJoin)
    painter.setPen(contour_pen)
    painter.setBrush(QtCore.Qt.NoBrush)

    def map_point(point):
        x = left + (float(point[0]) - min_x) * scale
        y = top + (max_y - float(point[1])) * scale
        return QtCore.QPointF(x, y)

    close_tolerance = max(span_x, span_y, 1.0) * 1e-6

    for polyline in data.polylines:
        polyline = np.asarray(polyline, dtype=float)
        if len(polyline) < 2:
            continue

        path_item = QtGui.QPainterPath()
        path_item.moveTo(map_point(polyline[0]))
        for point in polyline[1:]:
            path_item.lineTo(map_point(point))

        if (
            len(polyline) >= 3
            and np.linalg.norm(polyline[0] - polyline[-1]) <= close_tolerance
        ):
            path_item.closeSubpath()

        painter.drawPath(path_item)

    painter.end()

    # QImage.save produced an overload/value error on some PySide6 builds.
    # QImageWriter infers the encoder from the filename extension and is more
    # reliable across PySide6 versions.
    writer = QtGui.QImageWriter(path)
    if extension in {".jpg", ".jpeg"}:
        writer.setQuality(95)

    if not writer.write(image):
        details = writer.errorString() or "unknown Qt image-writer error"
        raise IOError(f"Qt could not save the cross-section image: {path}\n{details}")

def export_cross_section(
    data,
    path,
    image_size=2048,
    line_width=4.0,
    padding_percent=5.0,
    aspect_ratio=1.0,
):
    extension = os.path.splitext(path)[1].lower()

    if extension in {".png", ".jpg", ".jpeg"}:
        export_cross_section_image(
            data,
            path,
            image_size=image_size,
            line_width=line_width,
            padding_percent=padding_percent,
            aspect_ratio=aspect_ratio,
        )
    elif extension == ".dxf":
        # Kept for compatibility with older command-line workflows, but the
        # GUI now offers image export instead.
        export_cross_section_dxf(data, path)
    elif extension == ".svg":
        export_cross_section_svg(data, path)
    elif extension == ".csv":
        export_cross_section_csv(data, path)
    else:
        raise ValueError(
            "Unsupported cross-section format. Use .png, .jpg, or .jpeg."
        )


def default_cross_section_path(input_path, z_height):
    folder = os.path.dirname(os.path.abspath(input_path))
    base = os.path.splitext(os.path.basename(input_path))[0]
    z_text = f"{z_height:.3f}".replace("-", "m").replace(".", "p")
    return os.path.join(folder, f"{base}_cross_section_Z{z_text}.png")


def points_inside_closed_surface(points, surface):
    """
    Return a Boolean mask selecting points inside a closed PyVista surface.

    Newer PyVista versions use select_interior_points. Older versions fall
    back to select_enclosed_points. The interactive box is a simple closed
    surface, so either route is appropriate.
    """
    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")

    query = pv.PolyData(points)
    closed_surface = surface.extract_surface().triangulate().clean()

    if hasattr(query, "select_interior_points"):
        try:
            selected = query.select_interior_points(
                closed_surface,
                method="cell_locator",
                check_surface=False,
            )

            for key in ("selected_points", "SelectedPoints"):
                if key in selected.point_data:
                    return np.asarray(selected.point_data[key], dtype=bool)
        except Exception:
            pass

    selected = query.select_enclosed_points(
        closed_surface,
        tolerance=0.0,
        check_surface=False,
    )

    for key in ("SelectedPoints", "selected_points"):
        if key in selected.point_data:
            return np.asarray(selected.point_data[key], dtype=bool)

    raise RuntimeError("PyVista did not return an inside/outside point mask.")


def trimesh_to_pyvista_surface(mesh):
    """
    Convert a trimesh mesh into a PyVista PolyData surface.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)

    if mesh.faces is None or len(mesh.faces) == 0:
        return pv.PolyData(vertices)

    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_sizes = np.full((len(faces), 1), 3, dtype=np.int64)
    pv_faces = np.hstack((face_sizes, faces)).ravel()

    return pv.PolyData(vertices, pv_faces)


def trimesh_to_pyvista_points(mesh, max_points=25000):
    """
    Convert a mesh into a lightweight point cloud preview.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)

    if len(vertices) > max_points:
        rng = np.random.default_rng(123)
        vertices = vertices[rng.choice(len(vertices), max_points, replace=False)]

    return pv.PolyData(vertices)


class CrossSectionPreviewWidget(QtWidgets.QWidget):
    """
    Lightweight 2D preview that avoids creating a second VTK renderer.

    The on-screen preview shows faint X/Y zero axes and a small measurement
    label. Image export uses the same contour geometry, blue line styling,
    aspect ratio, and framing while deliberately suppressing those overlays.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.polylines = []
        self.bounds_2d = None
        self.setMinimumHeight(290)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )

    def clear_preview(self):
        self.polylines = []
        self.bounds_2d = None
        self.update()

    def set_cross_section(self, data):
        self.polylines = [np.array(line, copy=True) for line in data.polylines]
        self.bounds_2d = np.array(data.bounds_2d, copy=True)
        self.update()

    def _draw_cross_section(
        self,
        painter,
        target_rect,
        *,
        show_axes,
        show_measurement,
        line_width,
        padding_fraction=None,
        padding_pixels=None,
    ):
        target_rect = QtCore.QRectF(target_rect)
        painter.fillRect(target_rect, QtGui.QColor("white"))

        if not self.polylines or self.bounds_2d is None:
            if show_measurement:
                painter.setPen(QtGui.QColor(90, 90, 90))
                painter.drawText(
                    target_rect,
                    QtCore.Qt.AlignCenter,
                    "Generate a slice to see the 2D cross section.",
                )
            return

        min_x, min_y = self.bounds_2d[0]
        max_x, max_y = self.bounds_2d[1]

        span_x = max(float(max_x - min_x), 1e-12)
        span_y = max(float(max_y - min_y), 1e-12)

        if padding_pixels is not None:
            margin_x = float(padding_pixels)
            margin_y = float(padding_pixels)
        else:
            fraction = float(np.clip(
                0.05 if padding_fraction is None else padding_fraction,
                0.0,
                0.40,
            ))
            margin_x = target_rect.width() * fraction
            margin_y = target_rect.height() * fraction

        available_width = max(target_rect.width() - 2.0 * margin_x, 1.0)
        available_height = max(target_rect.height() - 2.0 * margin_y, 1.0)
        scale = min(available_width / span_x, available_height / span_y)

        drawn_width = span_x * scale
        drawn_height = span_y * scale
        left = target_rect.left() + (target_rect.width() - drawn_width) / 2.0
        top = target_rect.top() + (target_rect.height() - drawn_height) / 2.0

        def map_point(point):
            px = left + (float(point[0]) - min_x) * scale
            py = top + (max_y - float(point[1])) * scale
            return QtCore.QPointF(px, py)

        if show_axes:
            axis_pen = QtGui.QPen(QtGui.QColor(215, 215, 215))
            axis_pen.setWidthF(1.0)
            painter.setPen(axis_pen)

            if min_x <= 0.0 <= max_x:
                p1 = map_point(np.array([0.0, min_y]))
                p2 = map_point(np.array([0.0, max_y]))
                painter.drawLine(p1, p2)

            if min_y <= 0.0 <= max_y:
                p1 = map_point(np.array([min_x, 0.0]))
                p2 = map_point(np.array([max_x, 0.0]))
                painter.drawLine(p1, p2)

        contour_pen = QtGui.QPen(QtGui.QColor(20, 70, 150))
        contour_pen.setWidthF(float(line_width))
        contour_pen.setJoinStyle(QtCore.Qt.RoundJoin)
        contour_pen.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(contour_pen)
        painter.setBrush(QtCore.Qt.NoBrush)

        for polyline in self.polylines:
            polyline = np.asarray(polyline, dtype=float)
            if len(polyline) < 2:
                continue

            path = QtGui.QPainterPath()
            path.moveTo(map_point(polyline[0]))

            for point in polyline[1:]:
                path.lineTo(map_point(point))

            painter.drawPath(path)

        if show_measurement:
            painter.setPen(QtGui.QColor(60, 60, 60))
            painter.drawText(
                int(target_rect.left()) + 10,
                int(target_rect.top()) + 18,
                f"X width: {span_x:.4f}    Y depth: {span_y:.4f}",
            )

    def paintEvent(self, event):
        del event

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self._draw_cross_section(
            painter,
            self.rect(),
            show_axes=True,
            show_measurement=True,
            line_width=2.0,
            padding_pixels=28.0,
        )
        painter.end()

    def create_clean_image(
        self,
        image_width=2048,
        line_width=4.0,
        padding_percent=5.0,
    ):
        """
        Render the current preview to a clean image.

        The output keeps the current preview widget's width/height ratio and
        blue contour appearance, but excludes axes/grid and measurement text.
        """
        if not self.polylines or self.bounds_2d is None:
            raise ValueError("Generate a cross section before saving it.")

        image_width = int(np.clip(int(image_width), 256, 8192))
        widget_width = max(int(self.width()), 1)
        widget_height = max(int(self.height()), 1)
        aspect_ratio = widget_height / widget_width
        image_height = int(np.clip(round(image_width * aspect_ratio), 256, 8192))

        image = QtGui.QImage(
            image_width,
            image_height,
            QtGui.QImage.Format_ARGB32,
        )
        image.fill(QtGui.QColor("white"))

        painter = QtGui.QPainter(image)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        self._draw_cross_section(
            painter,
            image.rect(),
            show_axes=False,
            show_measurement=False,
            line_width=float(np.clip(float(line_width), 1.0, 50.0)),
            padding_fraction=(
                float(np.clip(float(padding_percent), 0.0, 40.0)) / 100.0
            ),
        )
        painter.end()
        return image

    def save_clean_image(
        self,
        path,
        image_width=2048,
        line_width=4.0,
        padding_percent=5.0,
    ):
        extension = os.path.splitext(path)[1].lower()
        if extension not in {".png", ".jpg", ".jpeg"}:
            raise ValueError("Cross-section images must use .png, .jpg, or .jpeg.")

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        image = self.create_clean_image(
            image_width=image_width,
            line_width=line_width,
            padding_percent=padding_percent,
        )

        # Use QImageWriter rather than QImage.save. Some PySide6 releases
        # reject otherwise-valid QImage.save format arguments.
        writer = QtGui.QImageWriter(path)
        if extension in {".jpg", ".jpeg"}:
            writer.setQuality(95)

        if not writer.write(image):
            details = writer.errorString() or "unknown Qt image-writer error"
            raise IOError(
                f"Qt could not save the cross-section image: {path}\n{details}"
            )


class SeatpostOrientationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "Seatpost Orientation Tool - Manual Rotation + Undo + Masking + Cross Sections"
        )
        self.resize(1450, 900)

        self.input_path = None
        self.original_mesh = None
        self.current_mesh = None
        self.auto_oriented_mesh = None
        self.auto_oriented_report = None
        self.report = TransformReport()
        self.is_flipped = False

        self.orientation_box_surface = None
        self.orientation_vertex_mask = None
        self.orientation_reference_mesh = None
        self.orientation_box_enabled = False
        self.orientation_box_widget = None
        self.orientation_box_base_bounds = None
        self._last_committed_box_state = None
        self._suppress_box_undo = False

        self.cross_section_data = None
        self.slice_controls_ready = False
        self._slice_slider_start_value = None
        self._slice_spin_start_value = None

        self.undo_stack = []
        self._restoring_undo = False

        self.preview_quality = "Surface"
        self.show_original = False
        self.show_current = True

        self.original_actor = None
        self.current_actor = None

        self._build_ui()
        self._setup_undo_shortcut()
        self._update_undo_controls()
        self._update_transform_text()
        self._clear_cross_section(
            "Load and orient a mesh, then choose a slice height."
        )

    def _setup_undo_shortcut(self):
        self.undo_action = QtGui.QAction("Undo", self)
        self.undo_action.setShortcut(QtGui.QKeySequence.Undo)
        try:
            self.undo_action.setShortcutContext(QtCore.Qt.ApplicationShortcut)
        except Exception:
            pass
        self.undo_action.triggered.connect(self.undo_last_action)
        self.addAction(self.undo_action)

    def _update_undo_controls(self):
        enabled = bool(self.undo_stack)
        if hasattr(self, "undo_button"):
            self.undo_button.setEnabled(enabled)
            if enabled:
                label = self.undo_stack[-1]["label"]
                self.undo_button.setToolTip(
                    f"Undo: {label} (Ctrl+Z)"
                )
            else:
                self.undo_button.setToolTip(
                    "Nothing to undo (Ctrl+Z)"
                )

        if hasattr(self, "undo_action"):
            self.undo_action.setEnabled(enabled)

    def _append_undo_record(self, kind, label, state):
        if self._restoring_undo:
            return

        self.undo_stack.append({
            "kind": str(kind),
            "label": str(label),
            "state": state,
        })

        if len(self.undo_stack) > UNDO_LIMIT:
            self.undo_stack = self.undo_stack[-UNDO_LIMIT:]

        self._update_undo_controls()

    def _clear_undo_history(self):
        self.undo_stack.clear()
        self._update_undo_controls()

    def _capture_camera_state(self):
        try:
            position = self.plotter.camera_position
            return [
                tuple(float(value) for value in position[0]),
                tuple(float(value) for value in position[1]),
                tuple(float(value) for value in position[2]),
            ]
        except Exception:
            return None

    def _restore_camera_state(self, camera_state):
        if camera_state is None:
            return

        try:
            self.plotter.camera_position = camera_state
            self.plotter.reset_camera_clipping_range()
            self.plotter.render()
        except Exception:
            pass

    def _capture_box_state(self):
        surface = None
        transform_matrix = None

        if self.orientation_box_widget is not None:
            try:
                surface = self._current_orientation_box_surface()
            except Exception:
                surface = None

            try:
                vtk_transform = vtkTransform()
                self.orientation_box_widget.GetTransform(vtk_transform)
                transform_matrix = vtk_transform_to_numpy(vtk_transform)
            except Exception:
                transform_matrix = None

        if surface is None and self.orientation_box_surface is not None:
            try:
                surface = self.orientation_box_surface.copy(deep=True)
            except Exception:
                surface = self.orientation_box_surface.copy()

        base_bounds = None
        if self.orientation_box_base_bounds is not None:
            base_bounds = tuple(
                float(value) for value in self.orientation_box_base_bounds
            )

        return {
            "active": bool(
                self.orientation_box_enabled
                and self.orientation_box_widget is not None
            ),
            "surface": surface,
            "mask": (
                None
                if self.orientation_vertex_mask is None
                else np.array(self.orientation_vertex_mask, copy=True)
            ),
            "base_bounds": base_bounds,
            "transform_matrix": (
                None
                if transform_matrix is None
                else np.array(transform_matrix, copy=True)
            ),
            "use_mask": bool(self.use_mask_checkbox.isChecked()),
            "level_from_mask": bool(self.mask_leveling_checkbox.isChecked()),
            "crop_to_mask": bool(self.crop_to_mask_checkbox.isChecked()),
            "show_mask_points": bool(
                self.show_mask_points_checkbox.isChecked()
            ),
            "status_text": self.mask_status_label.text(),
        }

    def _box_states_differ(self, first, second):
        if first is None or second is None:
            return first is not second

        if bool(first.get("active")) != bool(second.get("active")):
            return True

        first_transform = first.get("transform_matrix")
        second_transform = second.get("transform_matrix")

        if first_transform is None or second_transform is None:
            if first_transform is not second_transform:
                return True
        elif not np.allclose(
            first_transform,
            second_transform,
            rtol=0.0,
            atol=1e-9,
        ):
            return True

        first_surface = first.get("surface")
        second_surface = second.get("surface")

        if first_surface is None or second_surface is None:
            return first_surface is not second_surface

        try:
            first_points = np.asarray(first_surface.points, dtype=float)
            second_points = np.asarray(second_surface.points, dtype=float)
        except Exception:
            return True

        if first_points.shape != second_points.shape:
            return True

        return not np.allclose(
            first_points,
            second_points,
            rtol=0.0,
            atol=1e-9,
        )

    def _capture_cross_section_state(self):
        return {
            "data": clone_cross_section_data(self.cross_section_data),
            "state_text": self.slice_state_label.text(),
            "summary_text": self.cross_section_text.toPlainText(),
        }

    def _restore_cross_section_state(self, state):
        self.cross_section_data = clone_cross_section_data(
            state.get("data")
        )

        if self.cross_section_data is None:
            self.cross_section_preview.clear_preview()
        else:
            self.cross_section_preview.set_cross_section(
                self.cross_section_data
            )

        self.slice_state_label.setText(
            state.get("state_text", "No cross section has been generated.")
        )
        self.cross_section_text.setPlainText(
            state.get("summary_text", "")
        )

    def _capture_mesh_state(self):
        return {
            "current_mesh": (
                None if self.current_mesh is None else self.current_mesh.copy()
            ),
            "auto_oriented_mesh": (
                None
                if self.auto_oriented_mesh is None
                else self.auto_oriented_mesh.copy()
            ),
            "report": clone_transform_report(self.report),
            "auto_oriented_report": clone_transform_report(
                self.auto_oriented_report
            ),
            "is_flipped": bool(self.is_flipped),
            "box_state": self._capture_box_state(),
            "cross_section_state": self._capture_cross_section_state(),
            "slice_height": (
                float(self.slice_height_spin.value())
                if self.current_mesh is not None
                else None
            ),
            "camera_state": self._capture_camera_state(),
            "status_text": self.status_label.text(),
        }

    def _restore_mesh_state(self, state):
        self._clear_box_widgets()

        self.current_mesh = (
            None
            if state.get("current_mesh") is None
            else state["current_mesh"].copy()
        )
        self.auto_oriented_mesh = (
            None
            if state.get("auto_oriented_mesh") is None
            else state["auto_oriented_mesh"].copy()
        )
        self.report = clone_transform_report(state.get("report"))
        if self.report is None:
            self.report = TransformReport()

        self.auto_oriented_report = clone_transform_report(
            state.get("auto_oriented_report")
        )
        self.is_flipped = bool(state.get("is_flipped", False))

        self._restore_box_state(
            state.get("box_state", {}),
            redraw=False,
        )

        self._configure_slice_controls()
        slice_height = state.get("slice_height")
        if slice_height is not None and self.current_mesh is not None:
            self._set_slice_height_value(slice_height)

        self._restore_cross_section_state(
            state.get("cross_section_state", {})
        )

        self._redraw_scene(keep_camera=False)
        self._restore_camera_state(state.get("camera_state"))
        self._update_transform_text()
        self._set_status(state.get("status_text", "Previous state restored."))

    def undo_last_action(self):
        if not self.undo_stack:
            self._set_status("Nothing to undo.")
            return

        record = self.undo_stack.pop()
        self._update_undo_controls()
        self._restoring_undo = True

        try:
            kind = record["kind"]
            state = record["state"]

            if kind == "mesh":
                self._restore_mesh_state(state)
            elif kind == "box":
                camera_state = self._capture_camera_state()
                self._restore_box_state(state, redraw=True)
                self._restore_camera_state(camera_state)
            elif kind == "camera":
                self._restore_camera_state(state)
            elif kind == "slice_height":
                self._set_slice_height_value(float(state["z_height"]))
                self._mark_slice_stale()
                self._redraw_scene(keep_camera=True)
            elif kind == "cross_section":
                self._restore_cross_section_state(state)
                self._redraw_scene(keep_camera=True)
            else:
                raise ValueError(f"Unknown undo record type: {kind}")

            self._set_status(f"Undid: {record['label']}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Undo failed",
                str(exc),
            )
        finally:
            self._restoring_undo = False
            self._update_undo_controls()

    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.KeyPress:
            try:
                is_undo = event.matches(QtGui.QKeySequence.Undo)
            except Exception:
                is_undo = bool(
                    event.key() == QtCore.Qt.Key_Z
                    and event.modifiers() & QtCore.Qt.ControlModifier
                )

            if is_undo and not event.isAutoRepeat():
                focus_widget = QtWidgets.QApplication.focusWidget()

                if focus_widget is self.slice_height_spin:
                    old_value = self._slice_spin_start_value
                    current_value = float(self.slice_height_spin.value())
                    if (
                        old_value is not None
                        and abs(current_value - old_value) > 1e-12
                    ):
                        self._set_slice_height_value(old_value)
                        self._slice_spin_start_value = None
                        self._mark_slice_stale()
                        self._redraw_scene(keep_camera=True)
                        self._set_status("Undid: slice-height input")
                        return True

                if focus_widget is self.slice_height_slider:
                    old_value = self._slice_slider_start_value
                    current_value = float(self.slice_height_spin.value())
                    if (
                        old_value is not None
                        and abs(current_value - old_value) > 1e-12
                    ):
                        self._set_slice_height_value(old_value)
                        self._slice_slider_start_value = None
                        self._mark_slice_stale()
                        self._redraw_scene(keep_camera=True)
                        self._set_status("Undid: slice-height input")
                        return True

                self.undo_last_action()
                return True

        if event.type() == QtCore.QEvent.FocusIn:
            if (
                hasattr(self, "slice_height_spin")
                and watched is self.slice_height_spin
            ):
                self._slice_spin_start_value = float(
                    self.slice_height_spin.value()
                )
            elif (
                hasattr(self, "slice_height_slider")
                and watched is self.slice_height_slider
            ):
                self._slice_slider_start_value = float(
                    self.slice_height_spin.value()
                )

        if (
            event.type() == QtCore.QEvent.FocusOut
            and hasattr(self, "slice_height_slider")
            and watched is self.slice_height_slider
        ):
            self._commit_slice_slider_edit()

        return super().eventFilter(watched, event)

    def _begin_slice_slider_edit(self):
        self._slice_slider_start_value = float(
            self.slice_height_spin.value()
        )

    def _commit_slice_slider_edit(self):
        old_value = self._slice_slider_start_value
        self._slice_slider_start_value = None

        if old_value is None:
            return

        new_value = float(self.slice_height_spin.value())
        if abs(new_value - old_value) <= 1e-12:
            return

        self._append_undo_record(
            "slice_height",
            "slice-height change",
            {"z_height": old_value},
        )

    def _commit_slice_spin_edit(self):
        old_value = self._slice_spin_start_value
        self._slice_spin_start_value = None

        if old_value is None:
            return

        new_value = float(self.slice_height_spin.value())
        if abs(new_value - old_value) <= 1e-12:
            return

        self._append_undo_record(
            "slice_height",
            "slice-height change",
            {"z_height": old_value},
        )

    def _set_slice_height_value(self, z_height):
        if self.current_mesh is None:
            return

        z_height = float(z_height)
        fraction = self._slice_fraction_from_height(z_height)

        self.slice_controls_ready = False
        self.slice_height_spin.blockSignals(True)
        self.slice_height_slider.blockSignals(True)

        self.slice_height_spin.setValue(z_height)
        self.slice_height_slider.setValue(
            int(round(fraction * 1000.0))
        )
        self.slice_percent_label.setText(
            f"{fraction * 100.0:.1f}% of mesh height"
        )

        self.slice_height_slider.blockSignals(False)
        self.slice_height_spin.blockSignals(False)
        self.slice_controls_ready = True

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QHBoxLayout(central)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)

        button_layout = QtWidgets.QHBoxLayout()

        self.load_button = QtWidgets.QPushButton("Load Scan")
        self.load_button.clicked.connect(self.load_scan)
        button_layout.addWidget(self.load_button)

        self.orient_button = QtWidgets.QPushButton("Auto Orient Vertical")
        self.orient_button.clicked.connect(self.auto_orient)
        button_layout.addWidget(self.orient_button)

        self.flip_button = QtWidgets.QPushButton("Flip Top/Bottom")
        self.flip_button.clicked.connect(self.flip_orientation)
        button_layout.addWidget(self.flip_button)

        self.save_button = QtWidgets.QPushButton("Save Current Mesh")
        self.save_button.clicked.connect(self.save_current)
        button_layout.addWidget(self.save_button)

        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_orientation)
        button_layout.addWidget(self.reset_button)

        self.undo_button = QtWidgets.QPushButton("Undo")
        self.undo_button.setToolTip("Undo the previous command (Ctrl+Z)")
        self.undo_button.clicked.connect(self.undo_last_action)
        button_layout.addWidget(self.undo_button)

        left_layout.addLayout(button_layout)

        option_layout = QtWidgets.QHBoxLayout()

        self.show_original_checkbox = QtWidgets.QCheckBox("Show original")
        self.show_original_checkbox.setChecked(False)
        self.show_original_checkbox.stateChanged.connect(
            self.on_display_options_changed
        )
        option_layout.addWidget(self.show_original_checkbox)

        self.show_current_checkbox = QtWidgets.QCheckBox("Show corrected/current")
        self.show_current_checkbox.setChecked(True)
        self.show_current_checkbox.stateChanged.connect(
            self.on_display_options_changed
        )
        option_layout.addWidget(self.show_current_checkbox)

        option_layout.addWidget(QtWidgets.QLabel("Preview quality:"))

        self.quality_combo = QtWidgets.QComboBox()
        self.quality_combo.addItems(["Surface", "Surface + Edges", "Fast Points"])
        self.quality_combo.setCurrentText("Surface")
        self.quality_combo.currentTextChanged.connect(self.on_quality_changed)
        option_layout.addWidget(self.quality_combo)

        option_layout.addStretch()
        left_layout.addLayout(option_layout)

        manual_rotation_group = QtWidgets.QGroupBox("Manual Model Rotation")
        manual_rotation_layout = QtWidgets.QGridLayout(manual_rotation_group)
        manual_rotation_layout.setContentsMargins(8, 4, 8, 4)
        manual_rotation_layout.setHorizontalSpacing(5)
        manual_rotation_layout.setVerticalSpacing(4)

        manual_rotation_layout.addWidget(
            QtWidgets.QLabel("World-axis step:"),
            0,
            0,
        )
        self.model_rotation_step_spin = QtWidgets.QDoubleSpinBox()
        self.model_rotation_step_spin.setDecimals(2)
        self.model_rotation_step_spin.setRange(0.01, 180.0)
        self.model_rotation_step_spin.setSingleStep(0.5)
        self.model_rotation_step_spin.setValue(1.0)
        self.model_rotation_step_spin.setSuffix("°")
        self.model_rotation_step_spin.setToolTip(
            "Angle applied each time a manual model-rotation button is pressed"
        )
        manual_rotation_layout.addWidget(
            self.model_rotation_step_spin,
            0,
            1,
        )

        self.manual_relevel_checkbox = QtWidgets.QCheckBox(
            "Recenter X/Y and set bottom to Z=0 after each rotation"
        )
        self.manual_relevel_checkbox.setChecked(False)
        self.manual_relevel_checkbox.setToolTip(
            "Unchecked performs a pure rotation around the current mesh center. "
            "Checked adds a translation after the rotation."
        )
        manual_rotation_layout.addWidget(
            self.manual_relevel_checkbox,
            0,
            2,
            1,
            5,
        )

        manual_rotation_layout.addWidget(
            QtWidgets.QLabel("Rotate current mesh:"),
            1,
            0,
        )
        manual_model_specs = [
            ("−X", "x", -1.0),
            ("+X", "x", 1.0),
            ("−Y", "y", -1.0),
            ("+Y", "y", 1.0),
            ("−Z", "z", -1.0),
            ("+Z", "z", 1.0),
        ]
        for column, (label, axis_name, direction) in enumerate(
            manual_model_specs,
            start=1,
        ):
            button = QtWidgets.QPushButton(label)
            button.setMinimumWidth(48)
            button.setToolTip(
                f"Rotate the current mesh around world {axis_name.upper()}"
            )
            button.clicked.connect(
                lambda checked=False, axis=axis_name, sign=direction:
                    self.rotate_current_mesh(axis, sign)
            )
            manual_rotation_layout.addWidget(button, 1, column)

        manual_rotation_note = QtWidgets.QLabel(
            "Works before or after Auto Orient. Each press can be undone with Ctrl+Z."
        )
        manual_rotation_note.setWordWrap(True)
        manual_rotation_layout.addWidget(
            manual_rotation_note,
            2,
            0,
            1,
            7,
        )
        left_layout.addWidget(manual_rotation_group)

        camera_group = QtWidgets.QGroupBox("Camera Snap Views")
        camera_layout = QtWidgets.QHBoxLayout(camera_group)
        camera_layout.setContentsMargins(8, 4, 8, 4)
        camera_layout.setSpacing(5)

        camera_layout.addWidget(QtWidgets.QLabel("Look from:"))

        camera_specs = [
            ("+X", "x", 1, "Camera on +X, looking toward the mesh"),
            ("-X", "x", -1, "Camera on -X, looking toward the mesh"),
            ("+Y", "y", 1, "Camera on +Y, looking toward the mesh"),
            ("-Y", "y", -1, "Camera on -Y, looking toward the mesh"),
            ("+Z", "z", 1, "Camera above the mesh, looking downward"),
            ("-Z", "z", -1, "Camera below the mesh, looking upward"),
        ]

        for label, axis_name, direction, tooltip in camera_specs:
            button = QtWidgets.QPushButton(label)
            button.setMinimumWidth(48)
            button.setToolTip(tooltip)
            button.clicked.connect(
                lambda checked=False, axis=axis_name, sign=direction:
                    self.snap_camera_view(axis, sign)
            )
            camera_layout.addWidget(button)

        self.camera_isometric_button = QtWidgets.QPushButton("Isometric")
        self.camera_isometric_button.setToolTip(
            "Return to the standard isometric view and fit the mesh"
        )
        self.camera_isometric_button.clicked.connect(self.reset_camera)
        camera_layout.addWidget(self.camera_isometric_button)

        self.camera_fit_button = QtWidgets.QPushButton("Fit")
        self.camera_fit_button.setToolTip(
            "Fit the mesh without changing the current viewing direction"
        )
        self.camera_fit_button.clicked.connect(self.fit_camera_to_mesh)
        camera_layout.addWidget(self.camera_fit_button)

        camera_layout.addStretch()
        left_layout.addWidget(camera_group)

        self.plotter = QtInteractor(left_panel)
        left_layout.addWidget(self.plotter.interactor)

        self.status_label = QtWidgets.QLabel("Load a scan to begin.")
        left_layout.addWidget(self.status_label)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)

        self.right_tabs = QtWidgets.QTabWidget()
        right_layout.addWidget(self.right_tabs)

        self._build_transform_tab()
        self._build_orientation_mask_tab()
        self._build_cross_section_tab()

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([980, 470])

        main_layout.addWidget(splitter)
        self._setup_plotter()

    def _build_transform_tab(self):
        transform_tab = QtWidgets.QWidget()
        transform_layout = QtWidgets.QVBoxLayout(transform_tab)

        title = QtWidgets.QLabel("Transform Readout")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        transform_layout.addWidget(title)

        self.transform_text = QtWidgets.QPlainTextEdit()
        self.transform_text.setReadOnly(True)
        self.transform_text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        transform_layout.addWidget(self.transform_text)

        self.copy_button = QtWidgets.QPushButton("Copy Transform Readout")
        self.copy_button.clicked.connect(self.copy_transform_readout)
        transform_layout.addWidget(self.copy_button)

        self.right_tabs.addTab(transform_tab, "Transform")

    def _build_orientation_mask_tab(self):
        mask_tab = QtWidgets.QWidget()
        mask_layout = QtWidgets.QVBoxLayout(mask_tab)

        title = QtWidgets.QLabel("Orientation Mask / Region of Interest")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        mask_layout.addWidget(title)

        instructions = QtWidgets.QLabel(
            "Use the interactive 3D box to surround only the portion of the "
            "current mesh that should remain and determine the post direction. "
            "You may create this box before or after an earlier orientation "
            "attempt; enabling the box no longer resets the current rotation. "
            "The handles stay red and are forced to a fixed size after use."
        )
        instructions.setWordWrap(True)
        mask_layout.addWidget(instructions)

        button_row = QtWidgets.QHBoxLayout()

        self.enable_mask_button = QtWidgets.QPushButton(
            "Enable / Reset Box on Current Mesh"
        )
        self.enable_mask_button.clicked.connect(self.enable_orientation_box)
        button_row.addWidget(self.enable_mask_button)

        self.clear_mask_button = QtWidgets.QPushButton("Clear Box / Use Current Mesh")
        self.clear_mask_button.clicked.connect(self.clear_orientation_mask)
        button_row.addWidget(self.clear_mask_button)

        mask_layout.addLayout(button_row)

        self.use_mask_checkbox = QtWidgets.QCheckBox(
            "Use current box for Auto Orient Vertical"
        )
        self.use_mask_checkbox.setChecked(True)
        mask_layout.addWidget(self.use_mask_checkbox)

        self.mask_leveling_checkbox = QtWidgets.QCheckBox(
            "Center X/Y and set Z=0 from the boxed region"
        )
        self.mask_leveling_checkbox.setChecked(True)
        mask_layout.addWidget(self.mask_leveling_checkbox)

        self.crop_to_mask_checkbox = QtWidgets.QCheckBox(
            "Keep only geometry inside the box after orientation"
        )
        self.crop_to_mask_checkbox.setChecked(True)
        mask_layout.addWidget(self.crop_to_mask_checkbox)

        self.show_mask_points_checkbox = QtWidgets.QCheckBox(
            "Highlight vertices inside the box"
        )
        self.show_mask_points_checkbox.setChecked(True)
        self.show_mask_points_checkbox.stateChanged.connect(
            self.on_show_mask_points_changed
        )
        mask_layout.addWidget(self.show_mask_points_checkbox)

        rotation_group = QtWidgets.QGroupBox("Box Rotation Controls")
        rotation_layout = QtWidgets.QGridLayout(rotation_group)

        rotation_layout.addWidget(QtWidgets.QLabel("Rotation step:"), 0, 0)
        self.box_rotation_step_spin = QtWidgets.QDoubleSpinBox()
        self.box_rotation_step_spin.setDecimals(1)
        self.box_rotation_step_spin.setRange(0.1, 90.0)
        self.box_rotation_step_spin.setSingleStep(1.0)
        self.box_rotation_step_spin.setValue(5.0)
        self.box_rotation_step_spin.setSuffix("°")
        self.box_rotation_step_spin.setToolTip(
            "Angle applied each time a box rotation button is pressed"
        )
        rotation_layout.addWidget(self.box_rotation_step_spin, 0, 1, 1, 2)

        rotation_layout.addWidget(
            QtWidgets.QLabel("Rotate around world axis:"),
            1,
            0,
            1,
            3,
        )

        for row, axis_name in enumerate(("X", "Y", "Z"), start=2):
            axis_label = QtWidgets.QLabel(axis_name)
            axis_label.setAlignment(QtCore.Qt.AlignCenter)
            rotation_layout.addWidget(axis_label, row, 0)

            minus_button = QtWidgets.QPushButton(f"−{axis_name}")
            minus_button.setMinimumHeight(32)
            minus_button.setToolTip(
                f"Rotate negatively around world {axis_name}"
            )
            minus_button.clicked.connect(
                lambda checked=False, axis=axis_name.lower():
                    self.rotate_orientation_box(axis, -1.0)
            )
            rotation_layout.addWidget(minus_button, row, 1)

            plus_button = QtWidgets.QPushButton(f"+{axis_name}")
            plus_button.setMinimumHeight(32)
            plus_button.setToolTip(
                f"Rotate positively around world {axis_name}"
            )
            plus_button.clicked.connect(
                lambda checked=False, axis=axis_name.lower():
                    self.rotate_orientation_box(axis, 1.0)
            )
            rotation_layout.addWidget(plus_button, row, 2)

        rotation_note = QtWidgets.QLabel(
            "The buttons rotate the active box about its own center and are "
            "usually easier than freehand 3D rotation."
        )
        rotation_note.setWordWrap(True)
        rotation_layout.addWidget(rotation_note, 5, 0, 1, 3)

        mask_layout.addWidget(rotation_group)

        self.mask_status_label = QtWidgets.QLabel(
            "No orientation box is active. Auto orientation will use the current mesh."
        )
        self.mask_status_label.setWordWrap(True)
        mask_layout.addWidget(self.mask_status_label)

        orient_now_button = QtWidgets.QPushButton(
            "Auto Orient Using Current Selection"
        )
        orient_now_button.clicked.connect(self.auto_orient)
        mask_layout.addWidget(orient_now_button)

        restore_button = QtWidgets.QPushButton("Restore Original Loaded Mesh")
        restore_button.clicked.connect(self.restore_loaded_mesh)
        mask_layout.addWidget(restore_button)

        tips = QtWidgets.QPlainTextEdit()
        tips.setReadOnly(True)
        tips.setMaximumHeight(175)
        tips.setPlainText(
            """Box controls:
  • Drag a face to resize one side.
  • Drag the center/outline to move the box.
  • Use the explicit +/- X/Y/Z controls for reliable rotation.
  • Freehand face rotation remains available when convenient.
  • Red handles are kept at a fixed size after every interaction.
  • Press Ctrl+Z to undo the last box move, resize, or rotation.

Recommended selection:
  • Include a long section of the post shaft.
  • Exclude the saddle whenever possible.
  • Exclude tires, floor fragments, and detached scan pieces.
  • At least 20 mesh vertices must remain selected.

Behavior:
  • The box is placed on the current mesh without undoing prior rotation.
  • With cropping enabled, geometry outside the box is removed from the
    current/exported mesh when orientation is applied.
  • Restore Original Loaded Mesh returns to the untouched scan."""
        )
        mask_layout.addWidget(tips)
        mask_layout.addStretch()

        self.right_tabs.addTab(mask_tab, "Orientation Mask")

    def _build_cross_section_tab(self):
        slice_tab = QtWidgets.QWidget()
        slice_layout = QtWidgets.QVBoxLayout(slice_tab)

        title = QtWidgets.QLabel("Horizontal 2D Cross Section")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        slice_layout.addWidget(title)

        instructions = QtWidgets.QLabel(
            "The plane is perpendicular to world Z and is applied to the "
            "current oriented mesh. Adjust the height, then generate the slice. "
            "The saved PNG/JPG contains only the outline on white."
        )
        instructions.setWordWrap(True)
        slice_layout.addWidget(instructions)

        height_group = QtWidgets.QGroupBox("Slice Height")
        height_layout = QtWidgets.QGridLayout(height_group)

        height_layout.addWidget(QtWidgets.QLabel("Z height:"), 0, 0)

        self.slice_height_spin = QtWidgets.QDoubleSpinBox()
        self.slice_height_spin.setDecimals(4)
        self.slice_height_spin.setRange(0.0, 1.0)
        self.slice_height_spin.setSingleStep(0.1)
        self.slice_height_spin.valueChanged.connect(
            self.on_slice_height_spin_changed
        )
        self.slice_height_spin.editingFinished.connect(
            self.on_slice_height_editing_finished
        )
        self.slice_height_spin.installEventFilter(self)
        height_layout.addWidget(self.slice_height_spin, 0, 1)

        self.slice_percent_label = QtWidgets.QLabel("50.0% of mesh height")
        height_layout.addWidget(self.slice_percent_label, 0, 2)

        self.slice_height_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slice_height_slider.setRange(0, 1000)
        self.slice_height_slider.setValue(500)
        self.slice_height_slider.installEventFilter(self)
        self.slice_height_slider.valueChanged.connect(
            self.on_slice_height_slider_changed
        )
        self.slice_height_slider.sliderPressed.connect(
            self._begin_slice_slider_edit
        )
        self.slice_height_slider.sliderReleased.connect(
            self.on_slice_slider_released
        )
        height_layout.addWidget(self.slice_height_slider, 1, 0, 1, 3)

        slice_layout.addWidget(height_group)

        preview_options = QtWidgets.QHBoxLayout()

        self.show_slice_plane_checkbox = QtWidgets.QCheckBox("Show slice plane")
        self.show_slice_plane_checkbox.setChecked(True)
        self.show_slice_plane_checkbox.stateChanged.connect(
            self.on_slice_display_changed
        )
        preview_options.addWidget(self.show_slice_plane_checkbox)

        self.show_slice_line_checkbox = QtWidgets.QCheckBox("Show slice contour")
        self.show_slice_line_checkbox.setChecked(True)
        self.show_slice_line_checkbox.stateChanged.connect(
            self.on_slice_display_changed
        )
        preview_options.addWidget(self.show_slice_line_checkbox)

        preview_options.addStretch()
        slice_layout.addLayout(preview_options)

        export_group = QtWidgets.QGroupBox("Outline Image Export")
        export_layout = QtWidgets.QGridLayout(export_group)

        export_layout.addWidget(QtWidgets.QLabel("Image width:"), 0, 0)
        self.slice_image_size_spin = QtWidgets.QSpinBox()
        self.slice_image_size_spin.setRange(256, 8192)
        self.slice_image_size_spin.setSingleStep(256)
        self.slice_image_size_spin.setValue(2048)
        self.slice_image_size_spin.setSuffix(" px")
        self.slice_image_size_spin.setToolTip(
            "Output width. Height automatically matches the current 2D preview window."
        )
        export_layout.addWidget(self.slice_image_size_spin, 0, 1)

        export_layout.addWidget(QtWidgets.QLabel("Outline width:"), 1, 0)
        self.slice_line_width_spin = QtWidgets.QDoubleSpinBox()
        self.slice_line_width_spin.setRange(1.0, 50.0)
        self.slice_line_width_spin.setDecimals(1)
        self.slice_line_width_spin.setSingleStep(1.0)
        self.slice_line_width_spin.setValue(4.0)
        self.slice_line_width_spin.setSuffix(" px")
        export_layout.addWidget(self.slice_line_width_spin, 1, 1)

        export_layout.addWidget(QtWidgets.QLabel("White margin:"), 2, 0)
        self.slice_padding_spin = QtWidgets.QDoubleSpinBox()
        self.slice_padding_spin.setRange(0.0, 40.0)
        self.slice_padding_spin.setDecimals(1)
        self.slice_padding_spin.setSingleStep(1.0)
        self.slice_padding_spin.setValue(5.0)
        self.slice_padding_spin.setSuffix("%")
        export_layout.addWidget(self.slice_padding_spin, 2, 1)

        export_note = QtWidgets.QLabel(
            "Saved images use the same blue outline and aspect ratio as the 2D "
            "preview above. The axes/grid, measurement text, labels, and 3D "
            "slice plane are omitted."
        )
        export_note.setWordWrap(True)
        export_layout.addWidget(export_note, 3, 0, 1, 2)

        slice_layout.addWidget(export_group)

        action_row = QtWidgets.QHBoxLayout()

        self.generate_slice_button = QtWidgets.QPushButton(
            "Generate / Update Slice"
        )
        self.generate_slice_button.clicked.connect(self.generate_cross_section)
        action_row.addWidget(self.generate_slice_button)

        self.top_view_button = QtWidgets.QPushButton("View Slice Top-Down")
        self.top_view_button.clicked.connect(self.view_slice_top_down)
        action_row.addWidget(self.top_view_button)

        self.save_slice_button = QtWidgets.QPushButton("Save Slice Preview PNG/JPG")
        self.save_slice_button.clicked.connect(self.save_cross_section)
        action_row.addWidget(self.save_slice_button)

        slice_layout.addLayout(action_row)

        self.slice_state_label = QtWidgets.QLabel(
            "No cross section has been generated."
        )
        self.slice_state_label.setWordWrap(True)
        slice_layout.addWidget(self.slice_state_label)

        self.cross_section_preview = CrossSectionPreviewWidget()
        slice_layout.addWidget(self.cross_section_preview, stretch=2)

        self.cross_section_text = QtWidgets.QPlainTextEdit()
        self.cross_section_text.setReadOnly(True)
        self.cross_section_text.setMaximumHeight(220)
        self.cross_section_text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        slice_layout.addWidget(self.cross_section_text)

        self.right_tabs.addTab(slice_tab, "Cross Section")

    def _setup_plotter(self):
        self.plotter.set_background("white")
        self.plotter.add_axes()
        self.plotter.show_grid()
        self.plotter.enable_parallel_projection()
        self.plotter.view_isometric()

    def _set_status(self, text):
        self.status_label.setText(text)

    def _mesh_to_preview(self, mesh):
        if mesh is None:
            return None

        if self.preview_quality == "Fast Points":
            return trimesh_to_pyvista_points(mesh, max_points=30000)

        return trimesh_to_pyvista_surface(mesh)

    def _add_mesh_actor(self, pv_mesh, label, is_original):
        if pv_mesh is None:
            return None

        if self.preview_quality == "Fast Points":
            actor = self.plotter.add_mesh(
                pv_mesh,
                render_points_as_spheres=True,
                point_size=3.0,
                opacity=0.45 if is_original else 1.0,
                label=label,
            )
        else:
            show_edges = self.preview_quality == "Surface + Edges"

            actor = self.plotter.add_mesh(
                pv_mesh,
                show_edges=show_edges,
                opacity=0.30 if is_original else 1.0,
                label=label,
            )

        return actor

    def _clear_box_widgets(self):
        try:
            self.plotter.clear_box_widgets()
        except Exception:
            pass

        self.orientation_box_widget = None
        self.orientation_box_enabled = False

    def _style_orientation_box_widget(self):
        """Keep orientation-box handles red and at one stable size."""
        widget = self.orientation_box_widget
        if widget is None:
            return

        try:
            widget.SetHandleSize(ORIENTATION_BOX_HANDLE_SIZE)
        except Exception:
            pass

        property_specs = [
            ("GetHandleProperty", ORIENTATION_BOX_HANDLE_COLOR),
            (
                "GetSelectedHandleProperty",
                ORIENTATION_BOX_SELECTED_HANDLE_COLOR,
            ),
        ]

        for getter_name, color in property_specs:
            try:
                prop = getattr(widget, getter_name)()
                prop.SetColor(*color)
                prop.SetOpacity(1.0)
            except Exception:
                pass

        # A red outline makes the box easier to find, while the face remains
        # nearly transparent so it does not hide the scan.
        for getter_name, color in (
            ("GetOutlineProperty", (0.85, 0.0, 0.0)),
            ("GetSelectedOutlineProperty", (1.0, 0.0, 0.0)),
        ):
            try:
                prop = getattr(widget, getter_name)()
                prop.SetColor(*color)
                prop.SetLineWidth(2.0)
            except Exception:
                pass

        try:
            face_property = widget.GetFaceProperty()
            face_property.SetColor(1.0, 0.2, 0.2)
            face_property.SetOpacity(0.04)
        except Exception:
            pass

        try:
            selected_face_property = widget.GetSelectedFaceProperty()
            selected_face_property.SetColor(1.0, 0.0, 0.0)
            selected_face_property.SetOpacity(0.08)
        except Exception:
            pass

        try:
            widget.Modified()
        except Exception:
            pass

    def _schedule_orientation_box_style(self):
        # VTK can recalculate the visual handle scale at the end of an
        # interaction. Reapplying the size immediately and once on the next Qt
        # event-loop turn keeps it from growing after the first use.
        self._style_orientation_box_widget()
        try:
            QtCore.QTimer.singleShot(0, self._style_orientation_box_widget)
        except Exception:
            pass

    def _create_orientation_box_widget(
        self,
        bounds,
        transform_matrix=None,
    ):
        bounds = tuple(float(value) for value in bounds)
        previous_suppress = self._suppress_box_undo
        self._suppress_box_undo = True

        try:
            try:
                box_widget = self.plotter.add_box_widget(
                    callback=self._orientation_box_callback,
                    bounds=bounds,
                    factor=1.0,
                    rotation_enabled=True,
                    outline_translation=True,
                    interaction_event="end",
                )
            except TypeError:
                box_widget = self.plotter.add_box_widget(
                    callback=self._orientation_box_callback,
                    bounds=bounds,
                    factor=1.0,
                    rotation_enabled=True,
                    outline_translation=True,
                )

            self.orientation_box_widget = box_widget
            self.orientation_box_enabled = True
            self.orientation_box_base_bounds = bounds

            if transform_matrix is not None:
                self.orientation_box_widget.SetTransform(
                    numpy_to_vtk_transform(transform_matrix)
                )
                self.orientation_box_widget.Modified()

            self._schedule_orientation_box_style()

            try:
                current_surface = self._current_orientation_box_surface()
                self._orientation_box_callback(current_surface)
            except Exception:
                pass

            try:
                self.plotter.remove_actor(
                    "orientation_box_static",
                    reset_camera=False,
                    render=False,
                )
            except Exception:
                pass

            self._last_committed_box_state = self._capture_box_state()
            self._schedule_orientation_box_style()
            return box_widget
        finally:
            self._suppress_box_undo = previous_suppress

    def _restore_box_state(self, state, redraw=True):
        state = state or {}
        self._clear_box_widgets()

        surface = state.get("surface")
        self.orientation_box_surface = (
            None if surface is None else surface.copy(deep=True)
        )
        mask = state.get("mask")
        self.orientation_vertex_mask = (
            None if mask is None else np.array(mask, copy=True)
        )

        self.orientation_box_base_bounds = state.get("base_bounds")
        has_box = (
            self.orientation_box_surface is not None
            or self.orientation_box_base_bounds is not None
            or bool(state.get("active"))
        )
        self.orientation_reference_mesh = (
            self.current_mesh.copy()
            if has_box and self.current_mesh is not None
            else None
        )

        checkbox_values = (
            (self.use_mask_checkbox, bool(state.get("use_mask", False))),
            (
                self.mask_leveling_checkbox,
                bool(state.get("level_from_mask", True)),
            ),
            (
                self.crop_to_mask_checkbox,
                bool(state.get("crop_to_mask", True)),
            ),
            (
                self.show_mask_points_checkbox,
                bool(state.get("show_mask_points", True)),
            ),
        )

        for checkbox, value in checkbox_values:
            checkbox.blockSignals(True)
            checkbox.setChecked(value)
            checkbox.blockSignals(False)

        self.mask_status_label.setText(
            state.get(
                "status_text",
                "No orientation box is active. Auto orientation will use the current mesh.",
            )
        )

        should_reactivate = bool(
            state.get("active")
            and self.current_mesh is not None
            and self.orientation_box_base_bounds is not None
        )

        if redraw:
            self._redraw_scene(keep_camera=True)

        if should_reactivate:
            self._create_orientation_box_widget(
                self.orientation_box_base_bounds,
                transform_matrix=state.get("transform_matrix"),
            )
            self.plotter.render()
        else:
            self._last_committed_box_state = self._capture_box_state()

    def _add_orientation_mask_preview(self):
        if self.orientation_reference_mesh is None:
            return

        if (
            self.orientation_box_surface is not None
            and not self.orientation_box_enabled
        ):
            try:
                self.plotter.add_mesh(
                    self.orientation_box_surface,
                    style="wireframe",
                    color="red",
                    line_width=3.0,
                    opacity=0.9,
                    name="orientation_box_static",
                    label="Orientation box",
                )
            except Exception:
                pass

        if (
            self.orientation_vertex_mask is None
            or not self.show_mask_points_checkbox.isChecked()
        ):
            return

        vertices = np.asarray(self.orientation_reference_mesh.vertices, dtype=float)
        selected_points = vertices[self.orientation_vertex_mask]

        if len(selected_points) == 0:
            return

        if len(selected_points) > 30000:
            rng = np.random.default_rng(321)
            selected_points = selected_points[
                rng.choice(len(selected_points), 30000, replace=False)
            ]

        self.plotter.add_mesh(
            pv.PolyData(selected_points),
            render_points_as_spheres=True,
            point_size=5.0,
            opacity=0.85,
            name="orientation_mask_points",
            label="Orientation vertices",
        )

    def _update_orientation_mask_preview(self):
        try:
            self.plotter.remove_actor(
                "orientation_mask_points",
                reset_camera=False,
                render=False,
            )
        except Exception:
            pass

        if (
            self.orientation_reference_mesh is None
            or self.orientation_vertex_mask is None
            or not self.show_mask_points_checkbox.isChecked()
        ):
            self.plotter.render()
            return

        vertices = np.asarray(self.orientation_reference_mesh.vertices, dtype=float)
        selected_points = vertices[self.orientation_vertex_mask]

        if len(selected_points) > 30000:
            rng = np.random.default_rng(321)
            selected_points = selected_points[
                rng.choice(len(selected_points), 30000, replace=False)
            ]

        if len(selected_points):
            self.plotter.add_mesh(
                pv.PolyData(selected_points),
                render_points_as_spheres=True,
                point_size=5.0,
                opacity=0.85,
                name="orientation_mask_points",
            )

        self.plotter.render()

    def _add_slice_preview(self):
        if self.current_mesh is None:
            return

        bounds = np.asarray(self.current_mesh.bounds, dtype=float)
        z_height = float(self.slice_height_spin.value())

        if self.show_slice_plane_checkbox.isChecked():
            size_x = max(float(bounds[1, 0] - bounds[0, 0]) * 1.25, 1.0)
            size_y = max(float(bounds[1, 1] - bounds[0, 1]) * 1.25, 1.0)

            plane = pv.Plane(
                center=(0.0, 0.0, z_height),
                direction=(0.0, 0.0, 1.0),
                i_size=size_x,
                j_size=size_y,
            )

            self.plotter.add_mesh(
                plane,
                opacity=0.15,
                show_edges=True,
                line_width=1.0,
                label="Slice plane",
            )

        if (
            self.show_slice_line_checkbox.isChecked()
            and self.cross_section_data is not None
        ):
            slice_lines = cross_section_to_pyvista(self.cross_section_data)

            if slice_lines is not None:
                self.plotter.add_mesh(
                    slice_lines,
                    line_width=6.0,
                    render_lines_as_tubes=True,
                    label="Cross section",
                )

    def _redraw_scene(self, keep_camera=True):
        camera_position = None
        active_box_state = None

        if self.orientation_box_enabled:
            active_box_state = self._capture_box_state()
            self._clear_box_widgets()

        if keep_camera:
            try:
                camera_position = self.plotter.camera_position
            except Exception:
                camera_position = None

        self.plotter.clear()
        self.plotter.add_axes()
        self.plotter.show_grid()

        if self.original_mesh is None and self.current_mesh is None:
            self.plotter.add_text(
                "Load a mesh to begin",
                position="upper_left",
                font_size=12,
            )
            self.plotter.render()
            return

        if self.show_original and self.original_mesh is not None:
            original_preview = self._mesh_to_preview(self.original_mesh)
            self.original_actor = self._add_mesh_actor(
                original_preview,
                "Original",
                True,
            )

        if self.show_current and self.current_mesh is not None:
            current_preview = self._mesh_to_preview(self.current_mesh)
            self.current_actor = self._add_mesh_actor(
                current_preview,
                "Current",
                False,
            )

        self._add_orientation_mask_preview()
        self._add_slice_preview()

        try:
            self.plotter.add_legend(size=(0.20, 0.15))
        except Exception:
            pass

        if camera_position is not None:
            try:
                self.plotter.camera_position = camera_position
            except Exception:
                self.plotter.reset_camera()
        else:
            self.plotter.reset_camera()

        self.plotter.render()

        if (
            active_box_state is not None
            and active_box_state.get("active")
            and active_box_state.get("base_bounds") is not None
            and self.current_mesh is not None
        ):
            try:
                self._create_orientation_box_widget(
                    active_box_state["base_bounds"],
                    transform_matrix=active_box_state.get("transform_matrix"),
                )
                self.plotter.render()
            except Exception:
                # The saved static outline remains visible if VTK cannot
                # recreate the interactive widget during this redraw.
                pass

    def _update_transform_text(self):
        self.transform_text.setPlainText(self.report.summary_text())

    def _clear_cross_section(self, message):
        self.cross_section_data = None
        self.cross_section_preview.clear_preview()
        self.cross_section_text.setPlainText(message)
        self.slice_state_label.setText(message)

    def _configure_slice_controls(self, preserve_fraction=False):
        if self.current_mesh is None:
            self.slice_controls_ready = False
            return

        bounds = np.asarray(self.current_mesh.bounds, dtype=float)
        z_min = float(bounds[0, 2])
        z_max = float(bounds[1, 2])
        z_span = max(z_max - z_min, 0.0)

        if preserve_fraction and self.slice_controls_ready:
            fraction = self.slice_height_slider.value() / 1000.0
        else:
            fraction = 0.5

        target_z = z_min + fraction * z_span

        self.slice_controls_ready = False
        self.slice_height_spin.blockSignals(True)
        self.slice_height_slider.blockSignals(True)

        self.slice_height_spin.setRange(z_min, z_max)
        self.slice_height_spin.setSingleStep(max(z_span / 100.0, 0.0001))
        self.slice_height_spin.setValue(target_z)
        self.slice_height_slider.setValue(int(round(fraction * 1000.0)))
        self.slice_percent_label.setText(f"{fraction * 100.0:.1f}% of mesh height")

        self.slice_height_slider.blockSignals(False)
        self.slice_height_spin.blockSignals(False)
        self.slice_controls_ready = True

    def _slice_fraction_from_height(self, z_height):
        if self.current_mesh is None:
            return 0.5

        bounds = np.asarray(self.current_mesh.bounds, dtype=float)
        z_min = float(bounds[0, 2])
        z_max = float(bounds[1, 2])
        z_span = z_max - z_min

        if z_span <= 1e-12:
            return 0.0

        return float(np.clip((z_height - z_min) / z_span, 0.0, 1.0))

    def _mark_slice_stale(self):
        if self.cross_section_data is None:
            self.slice_state_label.setText(
                "Choose a height and click Generate / Update Slice."
            )
            return

        current_z = float(self.slice_height_spin.value())
        tolerance = max(abs(current_z) * 1e-9, 1e-9)

        if abs(current_z - self.cross_section_data.z_height) > tolerance:
            self.slice_state_label.setText(
                "Slice height changed. Click Generate / Update Slice "
                "to recalculate the contour."
            )
        else:
            self.slice_state_label.setText(
                f"Cross section generated at Z = {current_z:.4f}."
            )

    def on_quality_changed(self, text):
        self.preview_quality = text
        self._set_status(
            f"Preview quality changed to {text}. "
            "Saved mesh is still full resolution."
        )
        self._redraw_scene(keep_camera=True)

    def on_display_options_changed(self):
        self.show_original = self.show_original_checkbox.isChecked()
        self.show_current = self.show_current_checkbox.isChecked()
        self._redraw_scene(keep_camera=True)

    def on_slice_display_changed(self):
        self._redraw_scene(keep_camera=True)

    def on_slice_height_slider_changed(self, value):
        if not self.slice_controls_ready or self.current_mesh is None:
            return

        bounds = np.asarray(self.current_mesh.bounds, dtype=float)
        z_min = float(bounds[0, 2])
        z_max = float(bounds[1, 2])
        fraction = value / 1000.0
        z_height = z_min + fraction * (z_max - z_min)

        self.slice_height_spin.blockSignals(True)
        self.slice_height_spin.setValue(z_height)
        self.slice_height_spin.blockSignals(False)

        self.slice_percent_label.setText(f"{fraction * 100.0:.1f}% of mesh height")
        self._mark_slice_stale()

    def on_slice_height_spin_changed(self, value):
        if not self.slice_controls_ready or self.current_mesh is None:
            return

        fraction = self._slice_fraction_from_height(value)

        self.slice_height_slider.blockSignals(True)
        self.slice_height_slider.setValue(int(round(fraction * 1000.0)))
        self.slice_height_slider.blockSignals(False)

        self.slice_percent_label.setText(f"{fraction * 100.0:.1f}% of mesh height")
        self._mark_slice_stale()

    def on_slice_slider_released(self):
        self._commit_slice_slider_edit()
        self._redraw_scene(keep_camera=True)

    def on_slice_height_editing_finished(self):
        self._commit_slice_spin_edit()
        self._redraw_scene(keep_camera=True)

    def on_show_mask_points_changed(self):
        self._update_orientation_mask_preview()

    def _current_orientation_box_surface(self):
        """Return the current interactive box as a cleaned surface."""
        if self.orientation_box_widget is None:
            if self.orientation_box_surface is None:
                raise ValueError("No orientation box is active.")
            return self.orientation_box_surface.copy(deep=True)

        box_polydata = pv.PolyData()
        self.orientation_box_widget.GetPolyData(box_polydata)
        return box_polydata.extract_surface().triangulate().clean()

    def rotate_orientation_box(self, axis_name, direction):
        """Rotate the active box around its center using a world X/Y/Z axis."""
        if (
            not self.orientation_box_enabled
            or self.orientation_box_widget is None
            or self.orientation_reference_mesh is None
        ):
            QtWidgets.QMessageBox.warning(
                self,
                "No active orientation box",
                "Enable the orientation box before using the rotation controls.",
            )
            return

        axis_vectors = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }
        axis_name = str(axis_name).lower()

        if axis_name not in axis_vectors:
            raise ValueError(f"Unknown rotation axis: {axis_name}")

        angle_degrees = (
            float(direction) * float(self.box_rotation_step_spin.value())
        )
        prior_state = self._capture_box_state()

        try:
            current_surface = self._current_orientation_box_surface()
            center = np.asarray(current_surface.center, dtype=float)

            current_vtk_transform = vtkTransform()
            self.orientation_box_widget.GetTransform(current_vtk_transform)
            current_matrix = vtk_transform_to_numpy(current_vtk_transform)

            delta_matrix = np.eye(4)
            delta_matrix[:3, :3] = rotation_matrix_axis_angle(
                axis_vectors[axis_name],
                math.radians(angle_degrees),
            )

            rotate_about_center = (
                translation_matrix(center)
                @ delta_matrix
                @ translation_matrix(-center)
            )
            updated_matrix = rotate_about_center @ current_matrix

            previous_suppress = self._suppress_box_undo
            self._suppress_box_undo = True
            try:
                self.orientation_box_widget.SetTransform(
                    numpy_to_vtk_transform(updated_matrix)
                )
                self.orientation_box_widget.Modified()

                updated_surface = self._current_orientation_box_surface()
                self._orientation_box_callback(updated_surface)
            finally:
                self._suppress_box_undo = previous_suppress

            self._schedule_orientation_box_style()
            self._last_committed_box_state = self._capture_box_state()
            self._append_undo_record(
                "box",
                f"box rotation {angle_degrees:+.1f}° around {axis_name.upper()}",
                prior_state,
            )

            self._set_status(
                f"Rotated orientation box {angle_degrees:+.1f}° "
                f"around world {axis_name.upper()}."
            )
            self.plotter.render()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Box rotation failed",
                str(exc),
            )

    def _orientation_box_callback(self, box):
        if self.orientation_reference_mesh is None:
            return

        try:
            box_surface = box.extract_surface().triangulate().clean()
            vertices = np.asarray(
                self.orientation_reference_mesh.vertices,
                dtype=float,
            )
            mask = points_inside_closed_surface(vertices, box_surface)
        except Exception as exc:
            self.mask_status_label.setText(
                f"Could not evaluate the orientation box: {exc}"
            )
            self._schedule_orientation_box_style()
            return

        previous_state = self._last_committed_box_state

        self.orientation_box_surface = box_surface.copy(deep=True)
        self.orientation_vertex_mask = np.asarray(mask, dtype=bool)

        selected_count = int(np.count_nonzero(self.orientation_vertex_mask))
        total_count = len(vertices)
        percentage = 100.0 * selected_count / max(total_count, 1)

        if selected_count < 20:
            self.mask_status_label.setText(
                f"Only {selected_count:,} vertices are inside the box. "
                "Enlarge it before auto-orienting."
            )
        else:
            self.mask_status_label.setText(
                f"Orientation box selects {selected_count:,} of "
                f"{total_count:,} vertices ({percentage:.2f}%). "
                "Auto Orient Vertical will use only these vertices; cropping can also remove everything outside the box."
            )

        self._schedule_orientation_box_style()
        current_state = self._capture_box_state()

        if (
            not self._suppress_box_undo
            and previous_state is not None
            and self._box_states_differ(previous_state, current_state)
        ):
            self._append_undo_record(
                "box",
                "interactive box move/resize/rotation",
                previous_state,
            )

        self._last_committed_box_state = current_state
        self._update_orientation_mask_preview()
        self._schedule_orientation_box_style()

    def enable_orientation_box(self):
        if self.current_mesh is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No mesh loaded",
                "Load a scan before creating an orientation box.",
            )
            return

        prior_state = self._capture_box_state()

        # Snapshot the current mesh coordinates. This deliberately preserves
        # every orientation/flip already applied before the box was enabled.
        self._clear_box_widgets()
        self.orientation_box_surface = None
        self.orientation_vertex_mask = None
        self.orientation_reference_mesh = self.current_mesh.copy()
        self.orientation_box_base_bounds = None
        self._last_committed_box_state = None

        for actor_name in ("orientation_mask_points", "orientation_box_static"):
            try:
                self.plotter.remove_actor(
                    actor_name,
                    reset_camera=False,
                    render=False,
                )
            except Exception:
                pass

        bounds = tuple(
            np.asarray(
                self.orientation_reference_mesh.bounds,
                dtype=float,
            ).T.ravel()
        )

        try:
            self._create_orientation_box_widget(bounds)
        except Exception as exc:
            self.orientation_reference_mesh = None
            self.orientation_box_base_bounds = None
            QtWidgets.QMessageBox.critical(
                self,
                "Orientation box failed",
                str(exc),
            )
            return

        self.use_mask_checkbox.setChecked(True)
        self.right_tabs.setCurrentIndex(1)
        self._schedule_orientation_box_style()
        self._append_undo_record(
            "box",
            "enable/reset orientation box",
            prior_state,
        )

        self._set_status(
            "Orientation box enabled on the current mesh without resetting it. "
            "Use Ctrl+Z to undo box edits."
        )
        self.plotter.render()

    def clear_orientation_mask(self, silent=False, redraw=True):
        had_box = bool(
            self.orientation_box_widget is not None
            or self.orientation_box_surface is not None
            or self.orientation_reference_mesh is not None
        )
        prior_state = self._capture_box_state() if had_box else None

        self._clear_box_widgets()
        self.orientation_box_surface = None
        self.orientation_vertex_mask = None
        self.orientation_reference_mesh = None
        self.orientation_box_base_bounds = None
        self._last_committed_box_state = None
        self.use_mask_checkbox.setChecked(False)

        try:
            self.plotter.remove_actor(
                "orientation_mask_points",
                reset_camera=False,
                render=False,
            )
            self.plotter.remove_actor(
                "orientation_box_static",
                reset_camera=False,
                render=False,
            )
        except Exception:
            pass

        self.mask_status_label.setText(
            "No orientation box is active. Auto orientation will use the current mesh."
        )

        if not silent:
            if prior_state is not None:
                self._append_undo_record(
                    "box",
                    "clear orientation box",
                    prior_state,
                )
            self._set_status("Orientation mask cleared; using the current mesh.")

        if redraw:
            self.plotter.render()

    def _selected_orientation_vertices(self):
        if self.current_mesh is None:
            raise ValueError("No current mesh is available.")

        if (
            self.use_mask_checkbox.isChecked()
            and self.orientation_vertex_mask is not None
        ):
            if self.orientation_reference_mesh is None:
                raise ValueError(
                    "The orientation box is stale. Enable a new box and try again."
                )

            total_vertices = np.asarray(
                self.orientation_reference_mesh.vertices,
                dtype=float,
            )

            if len(self.orientation_vertex_mask) != len(total_vertices):
                raise ValueError(
                    "The orientation box no longer matches the current mesh. "
                    "Enable a new box and try again."
                )

            selected = total_vertices[self.orientation_vertex_mask]

            if len(selected) < 20:
                raise ValueError(
                    "The orientation box contains fewer than 20 vertices. "
                    "Enlarge the box or clear it."
                )

            return selected.copy(), "Interactive box on current mesh"

        return None, "Current mesh"

    def _camera_reference_bounds(self):
        if self.current_mesh is not None:
            return np.asarray(self.current_mesh.bounds, dtype=float)

        if self.original_mesh is not None:
            return np.asarray(self.original_mesh.bounds, dtype=float)

        return None

    def fit_camera_to_mesh(self):
        """Fit the mesh without changing the viewing direction."""
        if self._camera_reference_bounds() is None:
            return

        prior_camera = self._capture_camera_state()
        self.plotter.reset_camera()

        try:
            self.plotter.reset_camera_clipping_range()
        except Exception:
            pass

        self.plotter.render()
        self._append_undo_record(
            "camera",
            "camera fit",
            prior_camera,
        )
        self._set_status("Camera fitted to the current mesh.")

    def snap_camera_view(self, axis_name, direction):
        """Look directly toward the mesh from a world-axis direction."""
        bounds = self._camera_reference_bounds()

        if bounds is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No mesh loaded",
                "Load a scan before changing the camera view.",
            )
            return

        prior_camera = self._capture_camera_state()

        axis_vectors = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }
        axis_name = str(axis_name).lower()

        if axis_name not in axis_vectors:
            raise ValueError(f"Unknown camera axis: {axis_name}")

        direction = 1.0 if float(direction) >= 0.0 else -1.0
        view_direction = axis_vectors[axis_name] * direction

        center = (bounds[0] + bounds[1]) / 2.0
        mesh_size = bounds[1] - bounds[0]
        distance = max(float(np.max(mesh_size)) * 3.0, 1.0)
        camera_position = center + view_direction * distance

        if axis_name == "z":
            view_up = np.array([0.0, 1.0, 0.0])
        else:
            view_up = np.array([0.0, 0.0, 1.0])

        self.plotter.enable_parallel_projection()
        self.plotter.camera_position = [
            camera_position.tolist(),
            center.tolist(),
            view_up.tolist(),
        ]
        self.plotter.reset_camera()

        try:
            self.plotter.reset_camera_clipping_range()
        except Exception:
            pass

        self.plotter.render()
        self._append_undo_record(
            "camera",
            f"camera snap to {'+' if direction > 0 else '-'}{axis_name.upper()}",
            prior_camera,
        )
        sign_text = "+" if direction > 0 else "-"
        self._set_status(
            f"Camera snapped to view from {sign_text}{axis_name.upper()}."
        )

    def reset_camera(self):
        if self._camera_reference_bounds() is None:
            return

        prior_camera = self._capture_camera_state()
        self.plotter.enable_parallel_projection()
        self.plotter.view_isometric()
        self.plotter.reset_camera()

        try:
            self.plotter.reset_camera_clipping_range()
        except Exception:
            pass

        self.plotter.render()
        self._append_undo_record(
            "camera",
            "camera reset to isometric",
            prior_camera,
        )
        self._set_status("Camera reset to the isometric view.")

    def view_slice_top_down(self):
        if self.current_mesh is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No mesh loaded",
                "Load and orient a scan first.",
            )
            return

        prior_camera = self._capture_camera_state()
        self.plotter.enable_parallel_projection()
        self.plotter.view_xy()
        self.plotter.reset_camera()
        self.plotter.render()
        self._append_undo_record(
            "camera",
            "camera view slice top-down",
            prior_camera,
        )

    def load_scan(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select seatpost scan",
            "",
            (
                "Mesh files (*.obj *.stl *.ply *.off *.glb *.gltf);;"
                "OBJ files (*.obj);;"
                "STL files (*.stl);;"
                "PLY files (*.ply);;"
                "All files (*.*)"
            ),
        )

        if not path:
            return

        try:
            mesh = load_mesh(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))
            return

        self._clear_undo_history()
        self.input_path = path
        self.original_mesh = mesh
        self.current_mesh = mesh.copy()
        self.auto_oriented_mesh = None
        self.auto_oriented_report = None
        self.is_flipped = False
        self._clear_box_widgets()
        self.orientation_box_surface = None
        self.orientation_vertex_mask = None
        self.orientation_reference_mesh = None
        self.orientation_box_base_bounds = None
        self._last_committed_box_state = None
        self.use_mask_checkbox.setChecked(False)
        self.mask_status_label.setText(
            "No orientation box is active. Auto orientation will use the current mesh."
        )

        self.report = TransformReport(
            input_path=path,
            original_bounds=np.array(mesh.bounds, dtype=float),
            current_bounds=np.array(mesh.bounds, dtype=float),
        )

        self._configure_slice_controls()
        self._clear_cross_section(
            "Mesh loaded. Auto Orient Vertical before generating a slice."
        )

        self._set_status(f"Loaded: {path}")
        self._redraw_scene(keep_camera=False)
        self._update_transform_text()

    def auto_orient(self):
        if self.current_mesh is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No scan loaded",
                "Load a scan first.",
            )
            return

        try:
            orientation_vertices, orientation_source = (
                self._selected_orientation_vertices()
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid orientation selection",
                str(exc),
            )
            return

        mask_active = (
            orientation_vertices is not None
            and self.orientation_vertex_mask is not None
            and self.orientation_reference_mesh is not None
            and self.use_mask_checkbox.isChecked()
        )

        if mask_active:
            base_mesh = self.orientation_reference_mesh.copy()
            source_total_vertex_count = len(base_mesh.vertices)
        else:
            base_mesh = self.current_mesh.copy()
            source_total_vertex_count = len(base_mesh.vertices)

        crop_applied = bool(
            mask_active and self.crop_to_mask_checkbox.isChecked()
        )

        try:
            if crop_applied:
                working_mesh = crop_mesh_to_vertex_mask(
                    base_mesh,
                    self.orientation_vertex_mask,
                )
            else:
                working_mesh = base_mesh
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Box crop failed",
                str(exc),
            )
            return

        undo_state = self._capture_mesh_state()

        selected_count = (
            len(orientation_vertices)
            if orientation_vertices is not None
            else len(working_mesh.vertices)
        )

        action_text = "cropping and auto-orienting" if crop_applied else "auto-orienting"
        self._set_status(
            f"{action_text.capitalize()} from {selected_count:,} vertices..."
        )
        QtWidgets.QApplication.processEvents()

        use_mask_for_leveling = (
            orientation_vertices is not None
            and self.mask_leveling_checkbox.isChecked()
        )

        previous_total_transform = None
        previous_pass_count = 0

        if self.report.total_transform is not None:
            previous_total_transform = np.array(
                self.report.total_transform,
                dtype=float,
            )

        if self.report.orientation_pass_count:
            previous_pass_count = self.report.orientation_pass_count
        elif self.report.detected_axis is not None:
            previous_pass_count = 1

        try:
            mesh, report = auto_orient_mesh(
                working_mesh,
                self.input_path or "",
                orientation_vertices=orientation_vertices,
                orientation_source=orientation_source,
                use_orientation_vertices_for_leveling=use_mask_for_leveling,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Auto orientation failed",
                str(exc),
            )
            self._set_status("Auto orientation failed.")
            return

        incremental_transform = np.array(report.total_transform, dtype=float)

        if previous_total_transform is not None:
            report.total_transform = incremental_transform @ previous_total_transform
            report.prior_transform_preserved = True
        else:
            report.total_transform = incremental_transform
            report.prior_transform_preserved = False

        report.original_bounds = np.array(self.original_mesh.bounds, dtype=float)
        report.current_bounds = np.array(mesh.bounds, dtype=float)
        report.orientation_vertex_count = selected_count
        report.total_vertex_count = source_total_vertex_count
        report.orientation_pass_count = previous_pass_count + 1
        report.cropped_to_orientation_box = crop_applied
        report.cropped_vertex_count = len(mesh.vertices)
        report.cropped_face_count = len(mesh.faces)
        report.manual_rotation_count = self.report.manual_rotation_count
        report.manual_rotation_matrix = (
            None
            if self.report.manual_rotation_matrix is None
            else np.array(self.report.manual_rotation_matrix, copy=True)
        )
        report.last_manual_axis = self.report.last_manual_axis
        report.last_manual_angle_degrees = (
            self.report.last_manual_angle_degrees
        )
        report.last_manual_releveled = self.report.last_manual_releveled

        self._clear_box_widgets()
        self.orientation_box_surface = None
        self.orientation_vertex_mask = None
        self.orientation_reference_mesh = None
        self.orientation_box_base_bounds = None
        self._last_committed_box_state = None
        self.use_mask_checkbox.setChecked(False)

        self.auto_oriented_mesh = mesh.copy()
        self.auto_oriented_report = TransformReport(**{
            field_name: getattr(report, field_name)
            for field_name in report.__dataclass_fields__
        })
        self.current_mesh = mesh.copy()
        self.report = report
        self.is_flipped = False

        self._configure_slice_controls()
        self._clear_cross_section(
            "Mesh oriented. Choose a Z height and generate the cross section."
        )

        crop_message = " Geometry outside the box was removed." if crop_applied else ""
        self.mask_status_label.setText(
            f"Last orientation used {report.orientation_vertex_count:,} of "
            f"{report.total_vertex_count:,} vertices from "
            f"{report.orientation_source}.{crop_message}"
        )

        self._set_status(
            f"Auto-oriented pass {report.orientation_pass_count} from "
            f"{report.orientation_vertex_count:,} vertices. "
            f"Rotated {report.rotation_angle_degrees:.3f}°."
            + crop_message
        )

        self._redraw_scene(keep_camera=False)
        self._update_transform_text()
        self._append_undo_record(
            "mesh",
            "auto orientation",
            undo_state,
        )

    def rotate_current_mesh(self, axis_name, direction):
        """Apply one undoable manual world-axis rotation to the current mesh."""
        if self.current_mesh is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No mesh loaded",
                "Load a scan before rotating the model.",
            )
            return

        axis_name = str(axis_name).lower().strip()
        step = float(self.model_rotation_step_spin.value())
        angle_degrees = step * float(direction)
        relevel = bool(self.manual_relevel_checkbox.isChecked())
        undo_state = self._capture_mesh_state()

        box_state = None
        has_box = bool(
            self.orientation_box_widget is not None
            or self.orientation_box_surface is not None
            or self.orientation_reference_mesh is not None
        )
        if has_box:
            box_state = self._capture_box_state()
            self._clear_box_widgets()

        try:
            rotated_mesh, incremental_transform, level_reference_bounds = (
                manual_rotate_mesh(
                    self.current_mesh,
                    axis_name,
                    angle_degrees,
                    relevel_after_rotation=relevel,
                    level_reference_bounds=self.report.level_reference_bounds,
                )
            )
        except Exception as exc:
            if box_state is not None:
                self._restore_box_state(box_state, redraw=False)
            QtWidgets.QMessageBox.critical(
                self,
                "Manual rotation failed",
                str(exc),
            )
            return

        self.current_mesh = rotated_mesh
        self.report.current_bounds = np.array(
            self.current_mesh.bounds,
            dtype=float,
        )
        self.report.level_reference_bounds = level_reference_bounds

        previous_total = self.report.total_transform
        if previous_total is None:
            self.report.total_transform = np.array(
                incremental_transform,
                copy=True,
            )
        else:
            self.report.total_transform = (
                incremental_transform
                @ np.asarray(previous_total, dtype=float)
            )

        previous_manual = self.report.manual_rotation_matrix
        if previous_manual is None:
            self.report.manual_rotation_matrix = np.array(
                incremental_transform,
                copy=True,
            )
        else:
            self.report.manual_rotation_matrix = (
                incremental_transform
                @ np.asarray(previous_manual, dtype=float)
            )

        self.report.manual_rotation_count += 1
        self.report.last_manual_axis = axis_name.upper()
        self.report.last_manual_angle_degrees = angle_degrees
        self.report.last_manual_releveled = relevel

        if box_state is not None:
            old_transform = box_state.get("transform_matrix")
            if old_transform is not None:
                box_state["transform_matrix"] = (
                    incremental_transform
                    @ np.asarray(old_transform, dtype=float)
                )

            old_surface = box_state.get("surface")
            if old_surface is not None:
                box_state["surface"] = transform_pyvista_surface(
                    old_surface,
                    incremental_transform,
                )

            self._restore_box_state(box_state, redraw=False)

        self._configure_slice_controls(preserve_fraction=True)
        self._clear_cross_section(
            "Model rotation changed. Generate a new cross section."
        )
        self._redraw_scene(keep_camera=True)
        self._update_transform_text()
        self._append_undo_record(
            "mesh",
            (
                f"manual model rotation {angle_degrees:+.2f}° "
                f"around {axis_name.upper()}"
            ),
            undo_state,
        )

        level_text = (
            " The mesh was recentered and leveled."
            if relevel
            else ""
        )
        box_text = (
            " The active orientation box moved with the mesh."
            if box_state is not None
            else ""
        )
        self._set_status(
            f"Rotated current mesh {angle_degrees:+.2f}° around world "
            f"{axis_name.upper()}.{level_text}{box_text}"
        )

    def restore_loaded_mesh(self):
        if self.original_mesh is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No scan loaded",
                "Load a scan first.",
            )
            return

        undo_state = self._capture_mesh_state()
        self.clear_orientation_mask(silent=True, redraw=False)
        self.current_mesh = self.original_mesh.copy()
        self.auto_oriented_mesh = None
        self.auto_oriented_report = None
        self.is_flipped = False
        self.report = TransformReport(
            input_path=self.input_path or "",
            original_bounds=np.array(self.original_mesh.bounds, dtype=float),
            current_bounds=np.array(self.original_mesh.bounds, dtype=float),
        )

        self._configure_slice_controls()
        self._clear_cross_section(
            "Original mesh restored. Auto Orient Vertical before slicing."
        )
        self._redraw_scene(keep_camera=False)
        self._update_transform_text()
        self._append_undo_record(
            "mesh",
            "restore original loaded mesh",
            undo_state,
        )
        self._set_status("Restored the untouched originally loaded mesh.")

    def flip_orientation(self):
        if self.original_mesh is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No mesh loaded",
                "Load a scan first.",
            )
            return

        if (
            self.auto_oriented_mesh is None
            or self.current_mesh is None
            or self.report.detected_axis is None
        ):
            self.auto_orient()

            if self.auto_oriented_mesh is None:
                return

        undo_state = self._capture_mesh_state()

        if self.orientation_reference_mesh is not None:
            self.clear_orientation_mask(silent=True, redraw=False)

        try:
            mesh, report = flip_top_bottom(self.current_mesh, self.report)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Flip failed", str(exc))
            return

        previous_fraction = self.slice_height_slider.value() / 1000.0

        self.current_mesh = mesh
        self.report = report
        self.is_flipped = bool(report.flip_count % 2)

        self.slice_height_slider.setValue(
            int(round((1.0 - previous_fraction) * 1000.0))
        )
        self._configure_slice_controls(preserve_fraction=True)
        self._clear_cross_section(
            "Orientation changed. Generate a new cross section."
        )

        state = "flipped" if self.is_flipped else "unflipped"
        self._set_status(f"Orientation {state}. Save when it looks correct.")

        self._redraw_scene(keep_camera=True)
        self._update_transform_text()
        self._append_undo_record(
            "mesh",
            "flip top/bottom",
            undo_state,
        )

    def reset_orientation(self):
        if self.current_mesh is None and self.original_mesh is None:
            self._set_status("Nothing to reset.")
            return

        undo_state = self._capture_mesh_state()

        if self.orientation_reference_mesh is not None:
            self.clear_orientation_mask(silent=True, redraw=False)

        if self.auto_oriented_mesh is not None:
            self.current_mesh = self.auto_oriented_mesh.copy()

            if self.auto_oriented_report is not None:
                self.report = TransformReport(**{
                    field_name: getattr(self.auto_oriented_report, field_name)
                    for field_name in self.auto_oriented_report.__dataclass_fields__
                })

            self.is_flipped = False
            self._set_status("Reset to auto-oriented vertical mesh.")
            slice_message = (
                "Reset to the auto-oriented mesh. Generate a new cross section."
            )

        elif self.original_mesh is not None:
            self._clear_box_widgets()
            self.orientation_box_surface = None
            self.orientation_vertex_mask = None
            self.orientation_reference_mesh = None
            self.current_mesh = self.original_mesh.copy()

            self.report = TransformReport(
                input_path=self.input_path or "",
                original_bounds=np.array(self.original_mesh.bounds, dtype=float),
                current_bounds=np.array(self.current_mesh.bounds, dtype=float),
            )

            self.is_flipped = False
            self._set_status("Reset to original loaded mesh.")
            slice_message = (
                "Reset to the original mesh. Auto Orient Vertical before slicing."
            )
        else:
            self._set_status("Nothing to reset.")
            return

        self._configure_slice_controls()
        self._clear_cross_section(slice_message)

        self._redraw_scene(keep_camera=False)
        self._update_transform_text()
        self._append_undo_record(
            "mesh",
            "reset orientation",
            undo_state,
        )

    def generate_cross_section(self):
        if self.original_mesh is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No mesh loaded",
                "Load a scan first.",
            )
            return

        # The requested cross section should come from an oriented mesh.
        if self.report.detected_axis is None or self.current_mesh is None:
            self.auto_orient()

            if self.report.detected_axis is None or self.current_mesh is None:
                return

        z_height = float(self.slice_height_spin.value())
        undo_state = self._capture_cross_section_state()
        self._set_status(f"Calculating horizontal slice at Z = {z_height:.4f}...")
        QtWidgets.QApplication.processEvents()

        try:
            data = extract_horizontal_cross_section(
                self.current_mesh,
                z_height,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Cross section failed",
                str(exc),
            )
            self._set_status("Cross-section calculation failed.")
            return

        self.cross_section_data = data
        self.cross_section_preview.set_cross_section(data)
        self.cross_section_text.setPlainText(data.summary_text())
        self.slice_state_label.setText(
            f"Cross section generated at Z = {z_height:.4f}."
        )

        self._set_status(
            f"Generated {len(data.polylines)} contour(s) at "
            f"Z = {z_height:.4f}."
        )
        self._redraw_scene(keep_camera=True)
        self._append_undo_record(
            "cross_section",
            "generate/update cross section",
            undo_state,
        )

    def save_current(self):
        if self.current_mesh is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No mesh to save",
                "Load and orient a scan first.",
            )
            return

        if self.input_path:
            suggested_path = default_output_path(
                self.input_path,
                self.is_flipped,
            )
        else:
            suggested_path = os.path.join(
                os.getcwd(),
                "seatpost_vertical.obj",
            )

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save oriented mesh",
            suggested_path,
            (
                "OBJ files (*.obj);;"
                "STL files (*.stl);;"
                "PLY files (*.ply);;"
                "All files (*.*)"
            ),
        )

        if not path:
            return

        try:
            save_mesh(self.current_mesh, path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))
            return

        self._set_status(f"Saved: {path}")
        QtWidgets.QMessageBox.information(
            self,
            "Saved",
            f"Saved oriented mesh:\n{path}",
        )

    def save_cross_section(self):
        if self.cross_section_data is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No cross section",
                "Generate a cross section first.",
            )
            return

        current_z = float(self.slice_height_spin.value())
        tolerance = max(abs(current_z) * 1e-9, 1e-9)

        if abs(current_z - self.cross_section_data.z_height) > tolerance:
            QtWidgets.QMessageBox.warning(
                self,
                "Slice is out of date",
                (
                    "The selected height has changed since the contour was "
                    "generated. Click Generate / Update Slice before saving."
                ),
            )
            return

        if self.input_path:
            suggested_path = default_cross_section_path(
                self.input_path,
                self.cross_section_data.z_height,
            )
        else:
            suggested_path = os.path.join(
                os.getcwd(),
                "seatpost_cross_section.png",
            )

        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save clean cross-section outline image",
            suggested_path,
            (
                "PNG images (*.png);;"
                "JPEG images (*.jpg *.jpeg)"
            ),
        )

        if not path:
            return

        extension = os.path.splitext(path)[1].lower()

        if not extension:
            if selected_filter.startswith("JPEG"):
                path += ".jpg"
            else:
                path += ".png"
        elif extension not in {".png", ".jpg", ".jpeg"}:
            QtWidgets.QMessageBox.warning(
                self,
                "Unsupported image type",
                "Choose a PNG, JPG, or JPEG filename.",
            )
            return

        try:
            self.cross_section_preview.save_clean_image(
                path,
                image_width=self.slice_image_size_spin.value(),
                line_width=self.slice_line_width_spin.value(),
                padding_percent=self.slice_padding_spin.value(),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Cross-section save failed",
                str(exc),
            )
            return

        self._set_status(f"Saved 2D slice preview image: {path}")
        QtWidgets.QMessageBox.information(
            self,
            "Cross section image saved",
            (
                "Saved the blue 2D slice preview on a white background, "
                "without axes/grid or measurement text:\n"
                f"{path}"
            ),
        )

    def copy_transform_readout(self):
        text = self.report.summary_text()
        QtWidgets.QApplication.clipboard().setText(text)
        self._set_status("Copied transform readout to clipboard.")


def run_gui():
    app = QtWidgets.QApplication(sys.argv)
    window = SeatpostOrientationWindow()
    app.installEventFilter(window)
    window.show()

    sys.exit(app.exec())


def run_cli(
    input_path,
    output_path,
    flip=False,
    slice_z=None,
    slice_output=None,
):
    mesh = load_mesh(input_path)
    oriented, report = auto_orient_mesh(mesh, input_path)

    if flip:
        oriented, report = flip_top_bottom(oriented, report)

    save_mesh(oriented, output_path)

    print(f"Saved: {output_path}")
    print(report.summary_text())

    if slice_z is not None or slice_output is not None:
        if slice_z is None or slice_output is None:
            raise ValueError(
                "Both --slice-z and --slice-output are required "
                "when exporting a command-line cross section."
            )

        data = extract_horizontal_cross_section(oriented, slice_z)
        export_cross_section(data, slice_output)
        print()
        print(f"Saved cross section: {slice_output}")
        print(data.summary_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input mesh file")
    parser.add_argument("--output", help="Output oriented mesh file")
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Flip top/bottom after auto-orienting",
    )
    parser.add_argument(
        "--slice-z",
        type=float,
        help="Optional horizontal slice Z height after orientation",
    )
    parser.add_argument(
        "--slice-output",
        help="Optional .png, .jpg, or .jpeg cross-section output path",
    )

    args = parser.parse_args()

    if args.input or args.output:
        if not args.input or not args.output:
            print(
                "Both --input and --output are required for command-line mode.",
                file=sys.stderr,
            )
            sys.exit(2)

        try:
            run_cli(
                args.input,
                args.output,
                args.flip,
                args.slice_z,
                args.slice_output,
            )
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        run_gui()


if __name__ == "__main__":
    main()
