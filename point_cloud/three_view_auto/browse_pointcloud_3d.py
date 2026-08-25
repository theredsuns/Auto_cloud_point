#!/usr/bin/env python3
"""Browse generated PLY point clouds in an interactive Open3D window."""

import argparse
from pathlib import Path

import open3d as o3d


ROOT = Path(__file__).resolve().parent


def latest_run() -> Path:
    runs = [
        path for path in ROOT.iterdir()
        if path.is_dir() and any(path.glob("*.ply"))
    ]
    if not runs:
        raise SystemExit("No PLY point clouds found in object_pointcloud_output.")
    return max(runs, key=lambda path: path.stat().st_mtime)


class CloudBrowser:
    def __init__(self, files: list[Path]):
        self.files = files
        self.index = 0

    def show(self, viewer: o3d.visualization.VisualizerWithKeyCallback) -> bool:
        path = self.files[self.index]
        cloud = o3d.io.read_point_cloud(str(path))
        if cloud.is_empty():
            print(f"[warning] Empty cloud: {path.name}")
            return False
        viewer.clear_geometries()
        viewer.add_geometry(cloud, reset_bounding_box=True)
        print(f"[{self.index + 1}/{len(self.files)}] {path.name}: {len(cloud.points):,} points")
        return False

    def previous(self, viewer):
        self.index = max(0, self.index - 1)
        return self.show(viewer)

    def next(self, viewer):
        self.index = min(len(self.files) - 1, self.index + 1)
        return self.show(viewer)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive PLY browser: A=previous, B=next."
    )
    parser.add_argument(
        "--input", type=Path,
        help="output run folder; defaults to the latest run containing PLY files",
    )
    parser.add_argument("--point-size", type=float, default=2.0)
    args = parser.parse_args()
    if args.point_size <= 0:
        parser.error("--point-size must be positive")

    folder = (args.input or latest_run()).resolve()
    files = sorted(folder.glob("*.ply"))
    if not files:
        raise SystemExit(f"No .ply files in {folder}")

    browser = CloudBrowser(files)
    viewer = o3d.visualization.VisualizerWithKeyCallback()
    viewer.create_window(
        "3D point clouds | A=previous, B=next, mouse=orbit/zoom",
        width=1280,
        height=900,
    )
    options = viewer.get_render_option()
    options.point_size = args.point_size
    options.background_color = (0.08, 0.08, 0.08)
    viewer.register_key_callback(ord("A"), browser.previous)
    viewer.register_key_callback(ord("B"), browser.next)
    viewer.register_key_callback(ord("a"), browser.previous)
    viewer.register_key_callback(ord("b"), browser.next)
    browser.show(viewer)
    viewer.run()
    viewer.destroy_window()


if __name__ == "__main__":
    main()
