#!/usr/bin/env python3
"""Convert the offline photo-hull OBJ into a dense, viewable surface cloud."""

from pathlib import Path

import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parent
OBJ = ROOT / "datasets/object_01/reconstruction_clear/object_clear.obj"
OUT = ROOT / "datasets/object_01/reconstruction_clear/object_surface_dense.ply"


def load_quad_obj(path: Path) -> o3d.geometry.TriangleMesh:
    vertices, triangles = [], []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
        elif line.startswith("f "):
            face = [int(value.split("/")[0]) - 1 for value in line.split()[1:]]
            for index in range(1, len(face) - 1):
                triangles.append([face[0], face[index], face[index + 1]])
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices)),
        o3d.utility.Vector3iVector(np.asarray(triangles)),
    )
    mesh.compute_vertex_normals()
    return mesh


def main():
    mesh = load_quad_obj(OBJ)
    cloud = mesh.sample_points_uniformly(number_of_points=180_000)
    cloud.paint_uniform_color((0.08, 0.68, 0.92))
    o3d.io.write_point_cloud(str(OUT), cloud, write_ascii=False, compressed=True)
    print(f"surface triangles: {len(mesh.triangles)}; dense points: {len(cloud.points)}")
    o3d.visualization.draw_geometries(
        [cloud], window_name="Offline photo model — dense surface point cloud",
        width=1200, height=850,
    )


if __name__ == "__main__":
    main()
