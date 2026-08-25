# PiPER-L 本机控制界面

简单按钮面板启动：

```bash
cd ~/object_seek/arm_control_client
python3 piper_control_panel.py
```

界面通过 SSH 使用远程机 `skki@192.168.50.55` 的 `~/zed_code/arm_control` 驱动。首次使用先刷新状态；六个滑条可在 PiPER 厂商关节限位内选择目标，速度固定 5%。

带三维预览的滑条控制：

```bash
cd ~/object_seek/arm_control_client
chmod +x start_piper_slider_preview.sh
./start_piper_slider_preview.sh
```

该命令会打开 RViz 三维模型和滑条控制窗口。滑条每次变化都会立即更新模型的**目标姿态**；只有点击“发送低速目标”后，才会向真实机械臂发送命令。

ZED 实时画面 + 三维预览 + 机械臂滑条控制：

```bash
cd ~/object_seek/arm_control_client
chmod +x start_zed_arm_control.sh
./start_zed_arm_control.sh
```

该联合脚本会先启动远程 ZED 的低延迟图像发布程序，然后打开 ZED 实时画面窗口、RViz 目标姿态预览和机械臂滑条控制窗口。启动过程本身不发送机械臂运动目标。

三维图形控制（RViz + 关节滑条）：

```bash
cd ~/object_seek/arm_control_client
chmod +x start_piper_rviz_control.sh
./start_piper_rviz_control.sh
```

启动后会出现两个窗口：RViz 显示真实机械臂姿态，`joint_state_publisher_gui` 是六个关节滑条。首次先等待约 5 秒，桥接会自动把滑条启动位置记录为“保持当前姿态”的基准。之后只移动一个滑条约 `+1°` 或 `-1°`，表示该关节相对当前位置小步移动；单关节不得超过 ±5°。关闭终端或按 `Ctrl+C` 会停止桥接程序，不会使机械臂继续运动。

MoveIt 末端拖拽控制：

```bash
cd ~/object_seek/arm_control_client
chmod +x start_piper_moveit_control.sh
./start_piper_moveit_control.sh
```

在 RViz 的 `MotionPlanning` 面板中选择 `Planning`，拖动末端交互标记后点击 `Plan & Execute`。桥接会拒绝任何单关节超过 5 度的规划，速度固定 5%。
