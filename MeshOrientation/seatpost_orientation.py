#!/usr/bin/env python3
"""
Seatpost Orientation Tool - Fast Viewer Version

This version uses PyVista/VTK for a smoother hardware-accelerated mesh viewer.

Install:
    python -m pip install trimesh numpy pyvista pyvistaqt qtpy PySide6

Run:
    python seatpost_orientation_pyvista_gui.py

Workflow:
    1. Load Scan
    2. Auto Orient Vertical
    3. Flip Top/Bottom if needed
    4. Save Current Mesh

Notes:
    - The preview can be simplified without changing the saved mesh.
    - The saved mesh is always the full-resolution current mesh.
    - Move values are in the mesh's native units, probably millimeters for your scans.
    - This file intentionally uses qtpy + QT_API=pyside6 to avoid Qt binding mismatch errors.
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh


try:
    # Force pyvistaqt/qtpy to use the same Qt binding everywhere.
    # This avoids errors where pyvistaqt creates one type of Qt widget
    # while the app passes it a parent widget from another Qt binding.
    os.environ.setdefault("QT_API", "pyside6")

    import pyvista as pv
    from qtpy import QtCore, QtWidgets
    from pyvistaqt import QtInteractor
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

    def summary_text(self) -> str:
        lines = []

        if self.input_path:
            lines.append(f"File: {os.path.basename(self.input_path)}")
            lines.append("")

        if self.original_bounds is not None:
            lines.append("Original bounds:")
            lines.append(format_bounds(self.original_bounds))
            lines.append("")

        if self.detected_axis is None:
            lines.append("No auto-orientation has been applied yet.")
            lines.append("")
            lines.append("Load a mesh, then click Auto Orient Vertical.")
            return "\n".join(lines)

        lines.append("Auto-orientation:")
        lines.append(f"  Rotation angle to vertical: {self.rotation_angle_degrees:.3f}°")
        lines.append(f"  Detected post axis: {format_vector(self.detected_axis)}")
        lines.append(f"  Rotation axis: {format_vector(self.rotation_axis)}")
        lines.append(f"  Fit center used: {format_vector(self.fit_center)}")
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
            lines.append(f"Total transform translation column: {format_vector(self.total_transform[:3, 3])}")

        if self.current_bounds is not None:
            lines.append("")
            lines.append("Current bounds:")
            lines.append(format_bounds(self.current_bounds))

        lines.append("")
        lines.append("Note: move values are in the mesh's native units.")
        return "\n".join(lines)


def format_vector(v) -> str:
    if v is None:
        return "n/a"

    return f"[{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}]"


def format_matrix(m) -> list:
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


def normalize(v):
    n = np.linalg.norm(v)

    if n < 1e-12:
        raise ValueError("Cannot normalize a near-zero vector.")

    return v / n


def translation_matrix(vector):
    transform = np.eye(4)
    transform[:3, 3] = vector

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
           This reduces the effect of clamp bulges and bad scan edges.
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


def center_xy_and_level_z(mesh):
    """
    Put the mesh bottom at Z = 0 and center the X/Y bounding box around the origin.
    Returns the translation vector and the 4x4 translation matrix.
    """
    bounds = mesh.bounds
    xy_center = (bounds[0, :2] + bounds[1, :2]) / 2.0
    z_min = bounds[0, 2]

    move = np.array([-xy_center[0], -xy_center[1], -z_min])
    transform = translation_matrix(move)

    mesh.apply_transform(transform)

    return move, transform


def auto_orient_mesh(mesh, input_path=""):
    """
    Rotate the seatpost's main axis to world Z.
    """
    original_mesh = mesh
    mesh = mesh.copy()

    vertices = np.asarray(mesh.vertices)
    original_bounds = np.array(original_mesh.bounds, dtype=float)

    axis, center = estimate_seatpost_axis(vertices)

    rotation, rotation_axis, rotation_angle = rotation_matrix_from_vectors(axis, Z_AXIS)

    rotate_about_fit_center = np.eye(4)
    rotate_about_fit_center[:3, :3] = rotation
    rotate_about_fit_center[:3, 3] = -rotation @ center

    mesh.apply_transform(rotate_about_fit_center)

    level_move, level_transform = center_xy_and_level_z(mesh)

    total_transform = level_transform @ rotate_about_fit_center

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
    )

    return mesh, report


def flip_top_bottom(mesh, report):
    """
    Flip the vertical mesh 180 degrees so top and bottom swap.
    """
    mesh = mesh.copy()

    flip_rotation = np.eye(4)
    flip_rotation[:3, :3] = rotation_matrix_axis_angle(np.array([1.0, 0.0, 0.0]), np.pi)

    mesh.apply_transform(flip_rotation)

    level_move, level_transform = center_xy_and_level_z(mesh)

    flip_total = level_transform @ flip_rotation

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
    )

    return mesh, new_report


def save_mesh(mesh, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mesh.export(path)


def default_output_path(input_path, flipped=False):
    folder = os.path.dirname(os.path.abspath(input_path))
    base, ext = os.path.splitext(os.path.basename(input_path))

    suffix = "_vertical_flipped" if flipped else "_vertical"

    return os.path.join(folder, f"{base}{suffix}{ext}")


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


class SeatpostOrientationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Seatpost Orientation Tool - Fast Viewer")
        self.resize(1350, 850)

        self.input_path = None
        self.original_mesh = None
        self.current_mesh = None
        self.auto_oriented_mesh = None
        self.report = TransformReport()
        self.is_flipped = False

        self.preview_quality = "Fast Points"
        self.show_original = True
        self.show_current = True

        self.original_actor = None
        self.current_actor = None

        self._build_ui()
        self._update_transform_text()

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

        left_layout.addLayout(button_layout)

        option_layout = QtWidgets.QHBoxLayout()

        self.show_original_checkbox = QtWidgets.QCheckBox("Show original")
        self.show_original_checkbox.setChecked(True)
        self.show_original_checkbox.stateChanged.connect(self.on_display_options_changed)
        option_layout.addWidget(self.show_original_checkbox)

        self.show_current_checkbox = QtWidgets.QCheckBox("Show corrected/current")
        self.show_current_checkbox.setChecked(True)
        self.show_current_checkbox.stateChanged.connect(self.on_display_options_changed)
        option_layout.addWidget(self.show_current_checkbox)

        option_layout.addWidget(QtWidgets.QLabel("Preview quality:"))

        self.quality_combo = QtWidgets.QComboBox()
        self.quality_combo.addItems(["Fast Points", "Surface", "Surface + Edges"])
        self.quality_combo.setCurrentText("Fast Points")
        self.quality_combo.currentTextChanged.connect(self.on_quality_changed)
        option_layout.addWidget(self.quality_combo)

        self.reset_camera_button = QtWidgets.QPushButton("Reset Camera")
        self.reset_camera_button.clicked.connect(self.reset_camera)
        option_layout.addWidget(self.reset_camera_button)

        option_layout.addStretch()
        left_layout.addLayout(option_layout)

        self.plotter = QtInteractor(left_panel)
        left_layout.addWidget(self.plotter.interactor)

        self.status_label = QtWidgets.QLabel("Load a scan to begin.")
        left_layout.addWidget(self.status_label)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)

        title = QtWidgets.QLabel("Transform Readout")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        right_layout.addWidget(title)

        self.transform_text = QtWidgets.QPlainTextEdit()
        self.transform_text.setReadOnly(True)
        self.transform_text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        right_layout.addWidget(self.transform_text)

        self.copy_button = QtWidgets.QPushButton("Copy Transform Readout")
        self.copy_button.clicked.connect(self.copy_transform_readout)
        right_layout.addWidget(self.copy_button)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([920, 430])

        main_layout.addWidget(splitter)

        self._setup_plotter()

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

    def _redraw_scene(self, keep_camera=True):
        camera_position = None

        if keep_camera:
            try:
                camera_position = self.plotter.camera_position
            except Exception:
                camera_position = None

        self.plotter.clear()
        self.plotter.add_axes()
        self.plotter.show_grid()

        if self.original_mesh is None and self.current_mesh is None:
            self.plotter.add_text("Load a mesh to begin", position="upper_left", font_size=12)
            self.plotter.render()
            return

        if self.show_original and self.original_mesh is not None:
            original_preview = self._mesh_to_preview(self.original_mesh)
            self.original_actor = self._add_mesh_actor(original_preview, "Original", True)

        if self.show_current and self.current_mesh is not None:
            current_preview = self._mesh_to_preview(self.current_mesh)
            self.current_actor = self._add_mesh_actor(current_preview, "Current", False)

        try:
            self.plotter.add_legend(size=(0.18, 0.12))
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

    def _update_transform_text(self):
        self.transform_text.setPlainText(self.report.summary_text())

    def on_quality_changed(self, text):
        self.preview_quality = text
        self._set_status(f"Preview quality changed to {text}. Saved mesh is still full resolution.")
        self._redraw_scene(keep_camera=True)

    def on_display_options_changed(self):
        self.show_original = self.show_original_checkbox.isChecked()
        self.show_current = self.show_current_checkbox.isChecked()
        self._redraw_scene(keep_camera=True)

    def reset_camera(self):
        self.plotter.reset_camera()
        self.plotter.view_isometric()
        self.plotter.render()

    def load_scan(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select seatpost scan",
            "",
            "Mesh files (*.obj *.stl *.ply *.off *.glb *.gltf);;OBJ files (*.obj);;STL files (*.stl);;PLY files (*.ply);;All files (*.*)"
        )

        if not path:
            return

        try:
            mesh = load_mesh(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))
            return

        self.input_path = path
        self.original_mesh = mesh
        self.current_mesh = mesh.copy()
        self.auto_oriented_mesh = None
        self.is_flipped = False

        self.report = TransformReport(
            input_path=path,
            original_bounds=np.array(mesh.bounds, dtype=float),
            current_bounds=np.array(mesh.bounds, dtype=float),
        )

        self._set_status(f"Loaded: {path}")
        self._redraw_scene(keep_camera=False)
        self._update_transform_text()

    def auto_orient(self):
        if self.original_mesh is None:
            QtWidgets.QMessageBox.warning(self, "No scan loaded", "Load a scan first.")
            return

        self._set_status("Auto-orienting...")
        QtWidgets.QApplication.processEvents()

        try:
            mesh, report = auto_orient_mesh(self.original_mesh, self.input_path or "")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Auto orientation failed", str(exc))
            self._set_status("Auto orientation failed.")
            return

        self.auto_oriented_mesh = mesh.copy()
        self.current_mesh = mesh.copy()
        self.report = report
        self.is_flipped = False

        self._set_status(
            f"Auto-oriented vertical. Rotated {report.rotation_angle_degrees:.3f}°. "
            f"Save when it looks correct."
        )

        self._redraw_scene(keep_camera=False)
        self._update_transform_text()

    def flip_orientation(self):
        if self.original_mesh is None:
            QtWidgets.QMessageBox.warning(self, "No mesh loaded", "Load a scan first.")
            return

        if self.auto_oriented_mesh is None or self.current_mesh is None or self.report.detected_axis is None:
            self.auto_orient()

            if self.auto_oriented_mesh is None:
                return

        try:
            mesh, report = flip_top_bottom(self.current_mesh, self.report)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Flip failed", str(exc))
            return

        self.current_mesh = mesh
        self.report = report
        self.is_flipped = bool(report.flip_count % 2)

        state = "flipped" if self.is_flipped else "unflipped"
        self._set_status(f"Orientation {state}. Save when it looks correct.")

        self._redraw_scene(keep_camera=True)
        self._update_transform_text()

    def reset_orientation(self):
        if self.auto_oriented_mesh is not None:
            self.current_mesh = self.auto_oriented_mesh.copy()

            try:
                _, report = auto_orient_mesh(self.original_mesh, self.input_path or "")
                self.report = report
            except Exception:
                pass

            self.is_flipped = False
            self._set_status("Reset to auto-oriented vertical mesh.")
        elif self.original_mesh is not None:
            self.current_mesh = self.original_mesh.copy()

            self.report = TransformReport(
                input_path=self.input_path or "",
                original_bounds=np.array(self.original_mesh.bounds, dtype=float),
                current_bounds=np.array(self.current_mesh.bounds, dtype=float),
            )

            self.is_flipped = False
            self._set_status("Reset to original loaded mesh.")
        else:
            self._set_status("Nothing to reset.")

        self._redraw_scene(keep_camera=False)
        self._update_transform_text()

    def save_current(self):
        if self.current_mesh is None:
            QtWidgets.QMessageBox.warning(self, "No mesh to save", "Load and orient a scan first.")
            return

        if self.input_path:
            suggested_path = default_output_path(self.input_path, self.is_flipped)
        else:
            suggested_path = os.path.join(os.getcwd(), "seatpost_vertical.obj")

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save oriented mesh",
            suggested_path,
            "OBJ files (*.obj);;STL files (*.stl);;PLY files (*.ply);;All files (*.*)"
        )

        if not path:
            return

        try:
            save_mesh(self.current_mesh, path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))
            return

        self._set_status(f"Saved: {path}")
        QtWidgets.QMessageBox.information(self, "Saved", f"Saved oriented mesh:\n{path}")

    def copy_transform_readout(self):
        text = self.report.summary_text()
        QtWidgets.QApplication.clipboard().setText(text)
        self._set_status("Copied transform readout to clipboard.")


def run_gui():
    app = QtWidgets.QApplication(sys.argv)
    window = SeatpostOrientationWindow()
    window.show()

    sys.exit(app.exec())


def run_cli(input_path, output_path, flip=False):
    mesh = load_mesh(input_path)
    oriented, report = auto_orient_mesh(mesh, input_path)

    if flip:
        oriented, report = flip_top_bottom(oriented, report)

    save_mesh(oriented, output_path)

    print(f"Saved: {output_path}")
    print(report.summary_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input mesh file")
    parser.add_argument("--output", help="Output mesh file")
    parser.add_argument("--flip", action="store_true", help="Flip top/bottom after auto-orienting")

    args = parser.parse_args()

    if args.input or args.output:
        if not args.input or not args.output:
            print("Both --input and --output are required for command-line mode.", file=sys.stderr)
            sys.exit(2)

        run_cli(args.input, args.output, args.flip)
    else:
        run_gui()


if __name__ == "__main__":
    main()
