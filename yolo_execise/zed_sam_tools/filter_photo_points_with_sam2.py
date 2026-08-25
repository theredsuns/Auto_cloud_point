#!/usr/bin/env python3
"""Keep only COLMAP points that lie inside SAM2 masks in multiple photos."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from zed_yolo_sam2_icp_reconstruct import select_target
from zed_yolo_sam2_live import ROOT, load_sam2, load_yolo


def read_images(path):
    records = []
    lines = [line.strip() for line in path.read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    for header, observations in zip(lines[::2], lines[1::2]):
        data = header.split()
        obs = observations.split()
        points = [(float(obs[i]), float(obs[i + 1]), int(obs[i + 2]))
                  for i in range(0, len(obs), 3) if int(obs[i + 2]) >= 0]
        records.append((data[-1], points))
    return records


def read_points(path):
    positions, colors = {}, {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        point_id = int(fields[0])
        positions[point_id] = [float(v) for v in fields[1:4]]
        colors[point_id] = [int(v) for v in fields[4:7]]
    return positions, colors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ROOT / "datasets/object_01")
    ap.add_argument("--max-images", type=int, default=18)
    ap.add_argument("--target-class", default="Wing and Body")
    ap.add_argument("--min-hits", type=int, default=2)
    ap.add_argument("--min-ratio", type=float, default=0.55)
    args = ap.parse_args()
    recon = args.dataset / "reconstruction_clear"
    images = read_images(recon / "sparse_txt/images.txt")
    indices = np.linspace(0, len(images) - 1, min(args.max_images, len(images)), dtype=int)
    selected_images = [images[i] for i in sorted(set(indices))]
    positions, colors = read_points(recon / "sparse_txt/points3D.txt")
    hits, seen = {}, {}
    yolo = load_yolo(ROOT / "best.pt")
    sam2 = load_sam2(ROOT / "sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
                     ROOT / "sam2/checkpoints/sam2.1_hiera_tiny.pt")
    for index, (name, observations) in enumerate(selected_images, 1):
        bgr = cv2.imread(str(args.dataset / "images" / name))
        if bgr is None:
            continue
        result = yolo.predict(bgr, conf=0.65, verbose=False)[0]
        detected = select_target(result, sam2, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                                 args.target_class)
        if detected is None:
            print(f"[{index}/{len(selected_images)}] no target: {name}")
            continue
        mask = detected[3]
        height, width = mask.shape
        for x, y, point_id in observations:
            if point_id not in positions or not (0 <= x < width and 0 <= y < height):
                continue
            seen[point_id] = seen.get(point_id, 0) + 1
            if mask[int(round(y)), int(round(x))]:
                hits[point_id] = hits.get(point_id, 0) + 1
        print(f"[{index}/{len(selected_images)}] SAM2 mask: {name}")
    keep = [point_id for point_id, count in hits.items()
            if count >= args.min_hits and count / seen[point_id] >= args.min_ratio]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray([positions[i] for i in keep]))
    cloud.colors = o3d.utility.Vector3dVector(np.asarray([colors[i] for i in keep]) / 255.0)
    if len(cloud.points):
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=12, std_ratio=1.5)
    out = recon / "object_sam2_sparse.ply"
    o3d.io.write_point_cloud(str(out), cloud, write_ascii=False, compressed=True)
    print(f"saved {len(cloud.points)} SAM2-filtered object points to {out}")
    o3d.visualization.draw_geometries([cloud], window_name="Offline photos — SAM2-filtered object points",
                                      width=1200, height=850)


if __name__ == "__main__":
    main()
