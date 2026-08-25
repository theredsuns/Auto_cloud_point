#!/usr/bin/env python3
"""Open the validated global-pose ZED reconstruction in Open3D."""

from pathlib import Path

import open3d as o3d


ROOT = Path(__file__).resolve().parent
POINT_CLOUD = (
    ROOT
    / "datasets/local_offline_capture/scan_20260725_141357"
    / "pointcloud_body_wing/fused_colmap_depth.ply"
)


def main():
    cloud = o3d.io.read_point_cloud(str(POINT_CLOUD))
    if cloud.is_empty():
        raise SystemExit(f"Point cloud is empty: {POINT_CLOUD}")
    o3d.visualization.draw_geometries(
        [cloud],
        window_name="ZED reconstruction - global camera poses",
        width=1280,
        height=900,
        point_show_normal=False,
    )


if __name__ == "__main__":
    main()
