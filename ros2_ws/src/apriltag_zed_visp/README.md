# ZED rigid-body scanner

`zed_rigid_scanner` builds an OBJ or PLY mesh with the ZED spatial mapping API.
During capture it opens:

- `ZED 3D Scanner`: the left-camera image and tracking state.
- `ZED Live 3D Reconstruction`: the current depth points, accumulated modeling
  point cloud, growing blue mesh, world axes, camera pose, and camera trajectory.
  Drag to orbit and use the mouse wheel to zoom. The larger points show the
  model already fused in world coordinates; the smaller moving points show what
  the camera currently sees; the translucent blue surface is the generated mesh.

## Build and run

```bash
cd ~/object_seek/ros2_ws
colcon build --packages-select apriltag_zed_visp --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
ros2 run apriltag_zed_visp zed_rigid_scanner \
  --output rigid_body.obj
```

Keep the object still, leave a textured stationary background in view, and move
the camera slowly around the object. Scanning continues without a time or frame
limit. Press `B` in either viewer window when the reconstruction looks complete;
the scanner then extracts and saves the model. `Ctrl-C` and closing the 3D window
remain available as safety exits.

Useful options:

```text
--mesh-update-ms 750          3D refresh interval (minimum 100 ms)
--max-display-triangles 200000
                              live display limit; saved mesh is unaffected
--max-rviz-triangles 50000    RViz filled-surface limit; saved mesh is unaffected
--max-display-points 120000   live color point-cloud display limit
--point-cloud-every 3         refresh point cloud every 3 valid frames
--no-preview                  disable the 2D camera window
--no-3d                       disable the interactive 3D window
--image-only                  publish images only; disable depth, mapping, and model saving
--texture                     generate an OBJ texture
--svo recording.svo2          scan from a recording
```

If the live view is sluggish, first increase `--mesh-update-ms` and then lower
`--max-display-triangles`. These settings only change visualization performance,
not the final mesh resolution.

## RViz2 live point clouds

The scanner publishes the current colored depth cloud on
`/zed_scanner/live_cloud` and the accumulated reconstruction on
`/zed_scanner/built_cloud`. The filled reconstruction surface is published as a
triangle marker on `/zed_scanner/mesh`. The rectified ZED left image is published on
`/zed_scanner/left_image` at 1280x720 and about 10 FPS by default. A JPEG stream
is also published on `/zed_scanner/left_image/compressed`. Start RViz2 in
another terminal with:

```bash
source /home/nkk/object_seek/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=36
export ROS_LOCALHOST_ONLY=0
rviz2 -d /home/nkk/object_seek/ros2_ws/install/apriltag_zed_visp/share/apriltag_zed_visp/config/zed_scanner.rviz
```

For a dedicated resizable color-image window, use:

```bash
ros2 run rqt_image_view rqt_image_view --clear-config /zed_scanner/left_image
```

Keep the scanner's own 3D window open: press `B` there to finish and save.
