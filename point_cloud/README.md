# 点云工具

从此目录运行：

```bash
cd ~/object_seek/point_cloud/object_pointcloud_input
python3 run_pointcloud.py
```

深度浏览：

```bash
python3 ../view_depth_image.py depth
```

单帧/融合点云的 3D 浏览程序位于 `object_pointcloud_output/browse_pointcloud_3d.py`。

用 `body.pt` 自动生成并审核 mask：

```bash
python3 auto_mask_review.py
```

回车保存当前 mask 后进入下一张；`P` 重新画框并用 SAM2 修正。

远程 ZED 画面显示：

```bash
python3 remote_zed_viewer.py
```

`local_zed_capture.py` 与 ZED/SAM2 的共享采集依赖暂保留在项目根目录，避免破坏相机采集环境；它采集的数据可直接复制到本目录的 `object_pointcloud_input/`。
