# ZED 实时物体识别与点云

运行 [zed_yolo_sam2_live.py](zed_yolo_sam2_live.py)，ZED 左目窗口会显示类别和置信度，另一个窗口显示仅属于检测实例的彩色点云。`best.pt` 当前类别为 `Wing and Body`。

先确认相机已连接，然后运行：

```bash
cd /home/nkk/object_seek
python3 zed_yolo_sam2_live.py
```

按 `Q` 或 `Esc` 退出。没有 SAM2 时程序仍可启动，但会在窗口明确显示 `YOLO box fallback`；此模式只使用检测框，不能保证边缘精确。

要启用 SAM2 精确掩码，必须补齐压缩包中未包含的 SAM2 软件包、配置和 checkpoint，然后运行：

```bash
python3 zed_yolo_sam2_live.py \
  --sam2-config /absolute/path/to/sam2.1_hiera_t.yaml \
  --sam2-checkpoint /absolute/path/to/sam2.1_hiera_tiny.pt
```

SAM2 的 CPU 模式可能较慢；实时使用建议安装 NVIDIA 驱动和 CUDA 后使用 GPU。点云距离单位为米，`--point-stride 3` 可降低点数以提高显示流畅度。

## 远程 ZED、本地识别和建模

远端 `/home/skki/zed_code` 已改为在 `/zed_scanner/live_cloud` 中发布 `u/v` 像素坐标。本机无需连接或打开 ZED，运行：

```bash
source /home/nkk/object_seek/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=<远端使用的域 ID>
export ROS_LOCALHOST_ONLY=0
python3 /home/nkk/object_seek/remote_zed_yolo_sam2_icp.py \
  --sam2-config /absolute/path/to/sam2.1_hiera_t.yaml \
  --sam2-checkpoint /absolute/path/to/sam2.1_hiera_tiny.pt \
  --mesh-on-save
```

本地程序以同步的远端左目图像和点云为输入，YOLO 标注类别、SAM2 生成实例掩码，随后依据 `u/v` 保留准确对应的 3D 点并进行质量门控 ICP 融合。无需在远端安装 YOLO、SAM2 或 Open3D。

## 高质量多帧模型

使用严格的 SAM2 掩码与 ICP 融合程序：

```bash
python3 zed_yolo_sam2_icp_reconstruct.py \
  --sam2-config /absolute/path/to/sam2.1_hiera_t.yaml \
  --sam2-checkpoint /absolute/path/to/sam2.1_hiera_tiny.pt \
  --target-class 'Wing and Body' \
  --mesh-on-save
```

它不会把仅 YOLO 检测框的深度加入模型；每帧都会先完成 YOLO 类别标记和 SAM2 掩码，再以多尺度 ICP 对齐。ICP 的拟合度低于 0.55 或误差高于 2.5 cm 时，该帧会被拒绝，避免错误拼接。缓慢绕物体移动、相邻帧保持 50% 以上重叠，并让物体保持静止。按 `S` 保存 `models/recognized_object.ply` 和（加 `--mesh-on-save` 时）OBJ；按 `R` 从头扫描。
