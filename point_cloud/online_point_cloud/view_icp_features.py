#!/usr/bin/env python3
"""Open the saved ICP correspondence/support points for one online scan.

The exported clouds are directly usable in CloudCompare too:
  fusion/icp_debug/*_icp_correspondences.ply
Red: transformed new-frame point. Green: accumulated-model nearest point.
Blue: midpoint of a pair accepted by the final ICP distance gate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import open3d as o3d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan", type=Path, help="online_point_cloud/scan_... folder")
    parser.add_argument("--frame", help="only show one frame ID")
    args = parser.parse_args()
    folder = args.scan.resolve() / "fusion" / "icp_debug"
    paths = sorted(folder.glob("*_icp_correspondences.ply"))
    if args.frame:
        paths = [path for path in paths if path.name.startswith(args.frame + "_")]
    if not paths:
        raise SystemExit(f"No ICP debug cloud found in {folder}")
    geometries = [o3d.io.read_point_cloud(str(path)) for path in paths]
    geometries = [cloud for cloud in geometries if not cloud.is_empty()]
    if not geometries:
        raise SystemExit("The selected registration had no final-stage correspondences.")
    print("red=new frame, green=existing fused cloud, blue=accepted correspondence midpoint")
    o3d.visualization.draw_geometries(geometries, window_name="ICP correspondence/support points")


if __name__ == "__main__":
    main()
