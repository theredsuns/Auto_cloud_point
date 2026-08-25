#!/usr/bin/env python3
"""Build an offline visual-hull point cloud from SAM2 masks and COLMAP poses."""
import argparse
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from reconstruct_photo_hull import read_colmap, read_points, qvec_to_rot
from zed_yolo_sam2_icp_reconstruct import select_target
from zed_yolo_sam2_live import ROOT, load_sam2, load_yolo


def mask_for(bgr, yolo, sam, target):
    result = yolo.predict(bgr, conf=0.65, verbose=False)[0]
    selected = select_target(result, sam, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), target)
    return None if selected is None else selected[3]


def project(points, view, intr):
    width, height, fx, fy, cx, cy = intr
    cam = points @ view["R"].T + view["t"]
    z = cam[:, 2]; u = fx * cam[:, 0] / np.maximum(z, 1e-9) + cx; v = fy * cam[:, 1] / np.maximum(z, 1e-9) + cy
    return u, v, (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', type=Path, default=ROOT/'datasets/object_01')
    ap.add_argument('--views', type=int, default=18); ap.add_argument('--resolution', type=int, default=95)
    ap.add_argument('--target-class', default='Wing and Body'); args = ap.parse_args()
    recon = args.dataset/'reconstruction_clear'; intr, views = read_colmap(recon/'sparse_txt')
    selected = [views[i] for i in np.linspace(0, len(views)-1, min(args.views, len(views)), dtype=int)]
    yolo = load_yolo(ROOT/'best.pt'); sam = load_sam2(ROOT/'sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml', ROOT/'sam2/checkpoints/sam2.1_hiera_tiny.pt')
    masks = []
    for i, view in enumerate(selected, 1):
        bgr = cv2.imread(str(args.dataset/'images'/view['name'])); mask = None if bgr is None else mask_for(bgr, yolo, sam, args.target_class)
        if mask is not None: masks.append((view, mask)); print(f'[{i}/{len(selected)}] mask {view["name"]}')
    sparse = read_points(recon/'sparse_txt'/'points3D.txt')
    votes = np.zeros(len(sparse), np.int16); seen = np.zeros(len(sparse), np.int16)
    for view, mask in masks:
        u,v,valid = project(sparse, view, intr); ui=np.clip(np.rint(u).astype(int),0,mask.shape[1]-1); vi=np.clip(np.rint(v).astype(int),0,mask.shape[0]-1)
        seen += valid; votes += valid & mask[vi,ui]
    support = sparse[(votes >= 1) & (seen > 0)]
    if len(support) < 20: raise RuntimeError(f'Only {len(support)} supported points; photos lack usable object texture.')
    lo,hi=np.quantile(support,[.03,.97],axis=0); pad=(hi-lo)*.12; lo-=pad; hi+=pad
    step=(hi-lo).max()/args.resolution; shape=np.maximum(12,np.ceil((hi-lo)/step).astype(int)); axes=[lo[i]+(np.arange(shape[i])+.5)*step for i in range(3)]
    grid=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3); votes=np.zeros(len(grid),np.int16); seen=np.zeros(len(grid),np.int16)
    for view,mask in masks:
        u,v,valid=project(grid,view,intr); ui=np.clip(np.rint(u).astype(int),0,mask.shape[1]-1); vi=np.clip(np.rint(v).astype(int),0,mask.shape[0]-1)
        seen+=valid; votes+=valid & mask[vi,ui]
    occ=(seen>=max(4,len(masks)//3)) & (votes>=np.ceil(seen*.70))
    points=grid[occ]; cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)); cloud.paint_uniform_color((0.1,.75,.95))
    if len(points) < 1000:
        raise RuntimeError(
            f"SAM2 visual hull has only {len(points)} points; refusing to display an invalid model. "
            "Capture more viewpoints or save ZED depth with the RGB images."
        )
    out=recon/'object_sam2_visual_hull.ply'; o3d.io.write_point_cloud(str(out),cloud,compressed=True)
    print(f'SAM2 views={len(masks)} support={len(support)} hull points={len(points)} saved={out}')
    o3d.visualization.draw_geometries([cloud],window_name='SAM2 multi-view visual hull',width=1200,height=850)

if __name__=='__main__': main()
