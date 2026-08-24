# Auto Cloud Point — 可运行版本一

PiPER-L 机械臂、Tracer5 小车和远程 ZED 的本机控制客户端。

## 功能

- 受限低速的 PiPER-L 关节/末端控制与 RViz 预览；
- Tracer5 有线 DDS IMU 融合定位和路径控制；
- 自动抓拍规划：机械臂与小车步骤可混排、复制、保存并在重启后加载；
- YOLO + SAM2 抓拍与后台串行 ICP 点云融合；无合格目标时保存非目标帧并继续；
- 全局抓拍设置：到位停顿、无目标等待和最低识别度。

## 运行环境

- Ubuntu 22.04 + ROS 2 Humble；
- Python 3，包含 `numpy`、`opencv-python`、`Pillow`、`matplotlib`；
- 本机 ROS 工作区：`~/robot_arm/install/setup.bash`（机械臂/RViz 功能需要）；
- 远端 AGX：`skki@192.168.50.55`，具有 ROS、CAN、ZED 与 `~/zed_code/arm_control`；
- 小车有线地址：本机默认 `192.168.50.169`，Tracer5 默认 `192.168.50.55`。

自动点云抓拍还需要同级目录的既有 `point_cloud` 工程：

```text
object_seek/
├── Auto_cloud_point/
└── point_cloud/
```

其中应包含 `remote_zed_sam2_capture.py` 和 `online_point_cloud/online_icp_pipeline.py`。

## 快速启动

```bash
git clone https://github.com/theredsuns/Auto_cloud_point.git
cd Auto_cloud_point
chmod +x start_complete_arm_tcp_system.sh
./start_complete_arm_tcp_system.sh
```

启动过程只启动/复用远端服务和本机面板，不发送机械臂或小车运动命令。

仅启动 Tracer5 面板：

```bash
./start_tracer5_control_panel.sh
```

仅启动机械臂与 ZED 控制界面：

```bash
./start_zed_arm_control.sh
```

## 自动抓拍规划

1. 保存机械臂 TCP 位姿或 Tracer5 路径点。
2. 在“自动抓拍规划”生成队列，按需复制、移除、调整顺序。
3. 输入名称并“保存规划”；重启后从“已保存”加载。
4. 在顶部“全局设置”设定抓拍停顿、无目标等待和最低识别度，并保存。
5. 勾选安全确认后运行规划。

识别到目标但最高置信度低于阈值时，系统立即只保存 RGB 图片到当前扫描目录的 `non_targets/images/`，不会生成掩码或加入 ICP 融合，并继续后续步骤；完全未检测到目标时才按“无目标等待”设置重试。

## 本地运行数据

以下文件包含现场坐标或运行结果，默认不提交 Git：

- `motion_records.json`、`vehicle_motion_records.json`；
- `auto_capture_plans.json`、`control_settings.json`；
- `scan_*`、点云、日志和 Python 缓存。

如需共享规划，请单独导出并确认其中坐标适用于目标现场。

## 安全说明

运行前确认机械臂周围无人员、机械臂 CAN 反馈正常、小车周边无障碍物。所有实际运动都需要界面安全确认；请始终准备使用全局停止或“小车即停”。
