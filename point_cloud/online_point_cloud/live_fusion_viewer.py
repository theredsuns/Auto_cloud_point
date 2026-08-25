#!/usr/bin/env python3
"""Interactive Open3D online fusion viewer with per-frame highlight buttons."""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering


class LiveFusionWindow:
    """One rotatable Open3D window; controls are embedded at its right side."""
    MAX_FRAME_BUTTONS = 24

    def __init__(self, scan: Path):
        self.scan = scan.resolve()
        self.fused_path = self.scan / "fusion" / "fused_icp.ply"
        self.status_path = self.scan / "fusion" / "fusion_status.json"
        self.report_path = self.scan / "fusion" / "icp_registration_report.json"
        self.last_mtime = None
        self.selected_frame_id = None
        self.entries = []
        self.running = True
        self.initial_camera = True

        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window("在线 ICP 融合点云", 1280, 820)
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self.window.add_child(self.scene_widget)

        self.panel = gui.Vert(6, gui.Margins(10, 10, 10, 10))
        self.title = gui.Label("Fusion: waiting for first cloud...")
        self.panel.add_child(self.title)
        self.panel.add_child(gui.Label("Select a frame to highlight it in blue"))
        self.clear_button = gui.Button("Clear selection")
        self.clear_button.set_on_clicked(self.clear_highlight)
        self.panel.add_child(self.clear_button)
        self.panel.add_child(gui.Label("Fused frames"))
        self.frame_buttons = []
        for index in range(self.MAX_FRAME_BUTTONS):
            button = gui.Button("")
            button.visible = False
            button.set_on_clicked(lambda index=index: self.highlight_index(index))
            self.frame_buttons.append(button)
            self.panel.add_child(button)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self.layout)
        self.window.set_on_close(self.close)

    def layout(self, context):
        rect = self.window.content_rect
        panel_width = 205
        self.scene_widget.frame = gui.Rect(rect.x, rect.y, rect.width - panel_width, rect.height)
        self.panel.frame = gui.Rect(rect.get_right() - panel_width, rect.y, panel_width, rect.height)

    def close(self):
        self.running = False
        return True

    def update_frame_buttons(self):
        for index, button in enumerate(self.frame_buttons):
            if index < len(self.entries):
                entry = self.entries[index]
                marker = "● " if entry.get("frame_id") == self.selected_frame_id else ""
                button.text = f"{marker}Frame {index + 1}"
                button.visible = True
            else:
                button.visible = False

    def highlight_index(self, index):
        if index >= len(self.entries):
            return
        self.selected_frame_id = self.entries[index]["frame_id"]
        self.update_frame_buttons()
        self.refresh_scene()

    def clear_highlight(self):
        self.selected_frame_id = None
        self.update_frame_buttons()
        self.refresh_scene()

    def load_status(self):
        try:
            status = json.loads(self.status_path.read_text(encoding="utf-8"))
            count = int(status.get("frame_count", 0))
            total = int(status.get("input_frame_count", count))
        except (OSError, ValueError, json.JSONDecodeError):
            count = total = 0
        try:
            report = json.loads(self.report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            report = []
        self.entries = [
            entry for entry in report
            if entry.get("status") in {"camera_pose_anchor", "camera_pose_plus_icp", "anchor", "accepted"}
        ]
        if self.selected_frame_id not in {entry["frame_id"] for entry in self.entries}:
            self.selected_frame_id = None
        selected = "none" if self.selected_frame_id is None else str(self.selected_frame_id)
        self.title.text = f"Fusion: {count}/{total} frames | selected: {selected}"
        self.update_frame_buttons()

    def selected_overlay(self):
        if self.selected_frame_id is None:
            return None
        entry = next((item for item in self.entries if item["frame_id"] == self.selected_frame_id), None)
        if entry is None:
            return None
        path = self.scan / "clouds" / f"frame_{entry['frame_id']}.ply"
        if not path.is_file():
            return None
        cloud = o3d.io.read_point_cloud(str(path))
        if cloud.is_empty():
            return None
        cloud.transform(np.asarray(entry["root_from_frame"], dtype=np.float64))
        cloud = cloud.voxel_down_sample(.004)
        cloud.paint_uniform_color((0.03, 0.18, 1.0))
        return cloud

    def refresh_scene(self):
        if not self.fused_path.is_file():
            return
        cloud = o3d.io.read_point_cloud(str(self.fused_path))
        self.scene_widget.scene.clear_geometry()
        if cloud.is_empty():
            return
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = 2.5
        self.scene_widget.scene.add_geometry("fused_original_rgb", cloud, material)
        overlay = self.selected_overlay()
        if overlay is not None and not overlay.is_empty():
            selected_material = rendering.MaterialRecord()
            selected_material.shader = "defaultUnlit"
            selected_material.point_size = 4.0
            self.scene_widget.scene.add_geometry("selected_frame_blue", overlay, selected_material)
        if self.initial_camera:
            bounds = cloud.get_axis_aligned_bounding_box()
            self.scene_widget.setup_camera(60.0, bounds, bounds.get_center())
            self.initial_camera = False

    def refresh_from_disk(self):
        if not self.fused_path.is_file():
            return
        mtime = self.fused_path.stat().st_mtime_ns
        if mtime == self.last_mtime:
            return
        self.last_mtime = mtime
        self.load_status()
        self.refresh_scene()

    def watch(self):
        while self.running:
            try:
                changed = self.fused_path.is_file() and self.fused_path.stat().st_mtime_ns != self.last_mtime
            except OSError:
                changed = False
            if changed:
                self.app.post_to_main_thread(self.window, self.refresh_from_disk)
            time.sleep(.25)

    def run(self):
        threading.Thread(target=self.watch, daemon=True).start()
        self.refresh_from_disk()
        self.app.run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan", type=Path)
    args = parser.parse_args()
    LiveFusionWindow(args.scan).run()


if __name__ == "__main__":
    main()
