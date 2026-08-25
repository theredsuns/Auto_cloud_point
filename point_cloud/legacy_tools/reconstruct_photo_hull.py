#!/usr/bin/env python3
"""Build a CPU-only coarse photo hull from COLMAP poses and YOLO detections."""

import argparse
from collections import deque
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def qvec_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


def read_colmap(model):
    camera_line = next(
        line for line in (model / "cameras.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ).split()
    width, height = int(camera_line[2]), int(camera_line[3])
    fx, fy, cx, cy = map(float, camera_line[4:8])
    views = []
    lines = (model / "images.txt").read_text().splitlines()
    data = [line for line in lines if line and not line.startswith("#")]
    for line in data[::2]:
        p = line.split()
        views.append({
            "id": int(p[0]),
            "R": qvec_to_rot(np.array(list(map(float, p[1:5])))),
            "t": np.array(list(map(float, p[5:8]))),
            "name": p[9],
        })
    return (width, height, fx, fy, cx, cy), views


def detect_boxes(images, views, weights, cache):
    if cache.exists():
        return json.loads(cache.read_text())
    # Keep the system OpenCV/NumPy ABI loaded before adding the local YOLO runtime.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "yolo_runtime"))
    from ultralytics import YOLO
    model = YOLO(str(weights))
    paths = [str(images / view["name"]) for view in views]
    results = model(paths, device="cpu", verbose=False, stream=True)
    boxes = {}
    for view, result in zip(views, results):
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        conf = result.boxes.conf.detach().cpu().numpy()
        selected = xyxy[conf >= 0.35]
        if len(selected):
            # The detector sometimes splits the body and wing; their union is the object.
            x1, y1 = selected[:, :2].min(axis=0)
            x2, y2 = selected[:, 2:].max(axis=0)
            pad_x, pad_y = 0.015 * (x2-x1), 0.025 * (y2-y1)
            boxes[view["name"]] = [
                float(x1-pad_x), float(y1-pad_y), float(x2+pad_x), float(y2+pad_y)
            ]
    cache.write_text(json.dumps(boxes, indent=2))
    return boxes


def read_points(path):
    pts = []
    for line in path.read_text().splitlines():
        if line and not line.startswith("#"):
            p = line.split()
            pts.append(list(map(float, p[1:4])))
    return np.asarray(pts)


def score_points(points, views, boxes, intr):
    width, height, fx, fy, cx, cy = intr
    inside = np.zeros(len(points), np.int16)
    visible = np.zeros(len(points), np.int16)
    for view in views:
        if view["name"] not in boxes:
            continue
        cam = points @ view["R"].T + view["t"]
        z = cam[:, 2]
        u = fx * cam[:, 0] / np.maximum(z, 1e-9) + cx
        v = fy * cam[:, 1] / np.maximum(z, 1e-9) + cy
        valid = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        x1, y1, x2, y2 = boxes[view["name"]]
        hit = valid & (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        visible += valid
        inside += hit
    return inside / np.maximum(visible, 1)


def voxel_hull(bounds, views, boxes, intr, resolution):
    lo, hi = bounds
    shape = np.maximum(12, np.ceil((hi-lo) / ((hi-lo).max()/resolution)).astype(int))
    axes = [np.linspace(lo[i], hi[i], shape[i]) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    points = grid.reshape(-1, 3)
    votes = np.zeros(len(points), np.int16)
    seen = np.zeros(len(points), np.int16)
    width, height, fx, fy, cx, cy = intr
    for view in views:
        if view["name"] not in boxes:
            continue
        cam = points @ view["R"].T + view["t"]
        z = cam[:, 2]
        u = fx * cam[:, 0] / np.maximum(z, 1e-9) + cx
        v = fy * cam[:, 1] / np.maximum(z, 1e-9) + cy
        valid = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        x1, y1, x2, y2 = boxes[view["name"]]
        votes += valid & (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        seen += valid
    occupancy = (seen >= 8) & (votes >= np.maximum(6, np.ceil(seen * 0.72)))
    return occupancy.reshape(tuple(shape)), axes


def keep_largest_component(occ):
    """Remove detached false-positive hulls without requiring SciPy."""
    remaining = set(map(tuple, np.argwhere(occ)))
    largest = []
    neighbors = ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))
    while remaining:
        seed = remaining.pop()
        component = [seed]
        queue = deque([seed])
        while queue:
            p = queue.popleft()
            for d in neighbors:
                q = (p[0]+d[0], p[1]+d[1], p[2]+d[2])
                if q in remaining:
                    remaining.remove(q)
                    component.append(q)
                    queue.append(q)
        if len(component) > len(largest):
            largest = component
    clean = np.zeros_like(occ)
    if largest:
        clean[tuple(np.asarray(largest).T)] = True
    return clean


def write_surface_obj(path, occ, axes):
    # Emit only exposed voxel faces. Shared lattice vertices keep the outline crisp.
    vertex_ids, vertices, faces = {}, [], []
    dirs = [
        ((-1,0,0), ((0,0,0),(0,0,1),(0,1,1),(0,1,0))),
        ((1,0,0), ((1,0,0),(1,1,0),(1,1,1),(1,0,1))),
        ((0,-1,0), ((0,0,0),(1,0,0),(1,0,1),(0,0,1))),
        ((0,1,0), ((0,1,0),(0,1,1),(1,1,1),(1,1,0))),
        ((0,0,-1), ((0,0,0),(0,1,0),(1,1,0),(1,0,0))),
        ((0,0,1), ((0,0,1),(1,0,1),(1,1,1),(0,1,1))),
    ]
    step = np.array([a[1]-a[0] for a in axes])
    origin = np.array([a[0] for a in axes]) - step/2
    sx, sy, sz = occ.shape
    for i, j, k in np.argwhere(occ):
        for (di,dj,dk), corners in dirs:
            ni, nj, nk = i+di, j+dj, k+dk
            if 0 <= ni < sx and 0 <= nj < sy and 0 <= nk < sz and occ[ni,nj,nk]:
                continue
            face = []
            for corner in corners:
                key = (i+corner[0], j+corner[1], k+corner[2])
                if key not in vertex_ids:
                    vertex_ids[key] = len(vertices) + 1
                    vertices.append(origin + step*np.array(key))
                face.append(vertex_ids[key])
            faces.append(face)
    with path.open("w") as out:
        out.write("# CPU photo hull; coordinates follow the COLMAP reconstruction.\n")
        for v in vertices:
            out.write(f"v {v[0]:.7f} {v[1]:.7f} {v[2]:.7f}\n")
        for f in faces:
            out.write("f " + " ".join(map(str, f)) + "\n")
    return len(vertices), len(faces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("datasets/object_01"))
    ap.add_argument("--resolution", type=int, default=115)
    args = ap.parse_args()
    out = args.dataset / "reconstruction_clear"
    intr, views = read_colmap(out / "sparse_txt")
    boxes = detect_boxes(args.dataset/"images", views, Path("best.pt"), out/"boxes.json")
    sparse = read_points(out/"sparse_txt"/"points3D.txt")
    score = score_points(sparse, views, boxes, intr)
    object_points = sparse[score >= 0.62]
    if len(object_points) < 30:
        raise RuntimeError("Could not isolate enough object points")
    lo, hi = np.quantile(object_points, [0.02, 0.98], axis=0)
    margin = (hi-lo) * np.array([0.12, 0.18, 0.12])
    occ, axes = voxel_hull((lo-margin, hi+margin), views, boxes, intr, args.resolution)
    occ = keep_largest_component(occ)
    verts, faces = write_surface_obj(out/"object_clear.obj", occ, axes)
    summary = {
        "registered_photos": len(views), "photos_with_detection": len(boxes),
        "support_points": len(object_points), "occupied_voxels": int(occ.sum()),
        "vertices": verts, "faces": faces,
    }
    (out/"reconstruction_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
