# 吸盘圆形检测与圆心距离

球窝是固定的半球结构时，优先使用下方的**几何测量**，不需要 YOLO 训练或大量数据。YOLO 流程仅保留给后续画面中出现多种、位置不固定的圆形目标。

## 球窝：直接几何测量（推荐）

```bash
python3 sucker_seek/measure_socket_geometry.py sucker_seek \
  --output sucker_seek/geometry_results
```

脚本按“**白色外圆环 + 中间黑色圆**”筛选并拟合球窝，输出：球窝口径、两个圆心坐标和圆心偏差。结果位于 `geometry_results/socket_measurements.csv`，并附带可视化结果图。

相机或球窝位置固定时，建议限制区域以避免背景误检，例如第二张样图：

```bash
python3 sucker_seek/measure_socket_geometry.py sucker_seek/20260824_164825_961599.jpg \
  --roi 440 40 410 370 --mm-per-pixel 0.125
```

这会同时输出以毫米表示的球窝口径和圆心偏差。标定值必须来自同一拍摄平面；球窝倾斜明显时，画面里的口沿会成为椭圆，应固定正视角或进行相机标定后再做精密尺寸判定。

## 实时测量

```bash
python3 sucker_seek/live_socket_measure.py --camera 0
```

使用 ZED 左目：

```bash
python3 sucker_seek/live_socket_measure.py --zed
```

使用 Intel RealSense（彩色画面与深度画面对齐）：

```bash
python3 sucker_seek/live_socket_measure.py --realsense
```

外圈已默认设为 **60 mm**、内圈默认设为 **40 mm**。默认仅做完整图案识别：外、内圆之间的环带为白色，内圆内部为黑色，并且外圆必须完整位于画面内；任何被画面边缘截断的圆都会忽略。加入 `--use-depth-gate` 后，RealSense 深度只负责排除已成功测得且真实 `Z >= 500 mm`（0.5 m）的目标；近距离无有效深度时不会拒绝颜色识别。最终显示的 `model Z` 始终由 60 mm / 40 mm 圆环尺寸和相机内参计算，不使用深度传感器数值作为测量结果。RealSense 深度过滤优先使用白色环带的深度中值，避免黑色中心的深度空洞造成漏检。

近距离测距公式为：`Zouter = fx × 60 / 外圆像素直径`，`Zinner = fx × 40 / 内圆像素直径`，`model Z` 为两者的融合值。只有完整的白环黑心结构才进入此计算；圆环平面应尽量正对相机，倾斜会带来比例误差。

默认不按尺寸距离过滤，以优先保证完整白环黑心目标能识别。只有明确提供 `--max-model-z-mm` 时，才会反推该距离处的最小投影面积并排除过小的圆。注意：60 mm 圆在 30 mm 的投影通常会大于 RealSense 画面宽度，不能和“完整外圆在画面内”同时满足；因此不应把 30 mm 作为默认筛选门槛。

## 手动尺寸标定

将完整球窝放在画面中，再运行：

```bash
python3 sucker_seek/manual_socket_calibration.py
```

按空格冻结画面，再拖动鼠标框住完整球窝并按 Enter 确认。程序会统计框内白色像素、黑色像素及其比例，并保存为 `socket_color_calibration.json`；把打印出的 JSON 发给我，我会将这些数值接入实时筛选。

实时程序会自动加载 `socket_color_calibration.json`。它会按标定框的长宽比例缩放候选区域，并比较白色、黑色像素数量；两者都在标定值的容差范围内才输出识别结果。标定框应只包含完整球窝，避免把手、衣物或大面积背景框进去。

标定中的白、黑像素现在互斥：白色为低饱和且亮度至少 90，黑色为亮度低于 70；中间灰色不计入任一类。重新标定后，白色比例与黑色比例之和应不大于 100%。

实时识别的颜色关系为：低饱和灰白环带包围连续的低亮度黑色中心。程序分别检查白色环形区和黑色中心区的像素比例，并要求黑色块接近环心；二维码等黑白格无法满足该圆形空间关系，会被忽略。

按 `Q` 或 `Esc` 关闭。球窝在画面中位置固定时，使用 `--roi X Y W H` 限定球窝区域会更稳定；加入 `--mm-per-pixel 0.125` 可同时显示毫米单位。

其中 `X Y W H` 必须替换为实际的像素整数，例如 `--roi 400 100 500 500`。不确定坐标时，直接使用交互式鼠标框选：

```bash
python3 sucker_seek/live_socket_measure.py --realsense --select-roi
```

在首帧拖动鼠标框住球窝，按 Enter 确认。

球窝在 0.5 m 处的外圆直径通常只有约 70–100 像素，程序已支持该尺寸。若背景有多个白色/黑色物体，务必使用 `--roi X Y W H` 限定球窝工作区域，这会显著减少漏检和误检。

默认无需框选，直接全画面识别。为保持画面流畅，圆检测默认保留 1280 像素宽的小圆细节、每 5 帧执行一次；候选评分仅扫描局部圆形区域。若电脑性能较好，可用 `--detect-every 2` 提高检测频率；若仍卡顿，用 `--detect-every 8` 降低 CPU 占用。

白环阈值已放宽为“低饱和、亮度大于 70”，黑心阈值为亮度小于 130，以适应偏灰的现场图像。需要查看灰度输入时加入 `--show-gray`。

## YOLO 圆检测（仅在需要识别不固定目标时）

每一个需要测量的圆都是 `circle` 类。模型输出的检测框中心就是圆心估计；脚本会计算同一张图所有圆心的两两欧氏距离。

当前仅有 2 张样图，不能训练出可靠模型。请采集至少 100 张（推荐 300+），覆盖距离、旋转、倾斜、光照、遮挡和不同背景；每一张都应标出所有需要测距的圆。

## 1. 标注圆心和半径

```bash
python3 sucker_seek/annotate_circles.py sucker_seek
```

每个圆点击两次：先圆心、再圆周。`U` 撤销，`S` 保存，`N` 下一张，`Q` 保存并退出。标注会写入 `sucker_seek/circle_annotations.json`。

## 2. 构建数据集并训练

```bash
python3 sucker_seek/build_yolo_dataset.py sucker_seek sucker_seek/circle_annotations.json \
  --output sucker_seek/dataset
chmod +x sucker_seek/train_circle_yolo.sh
./sucker_seek/train_circle_yolo.sh "$PWD/sucker_seek/dataset"
```

模型输出在 `sucker_seek/runs/circle/weights/best.pt`。训练使用项目自带的 CPU YOLO 运行库和已有的检测预训练权重 `body.pt`，无须另行安装或下载 PyTorch/Ultralytics；如安装 GPU 版运行库则会自动使用 GPU。

## 3. 推理及距离计算

```bash
python3 sucker_seek/detect_circle_distances.py sucker_seek \
  --weights sucker_seek/runs/circle/weights/best.pt \
  --output sucker_seek/results
```

输出的 `results/distances.csv` 含像素距离，结果图叠加圆心和连线。若需毫米距离，先用同一拍摄平面上的已知长度标定 `mm_per_pixel = 实际毫米 / 对应像素`，再加入：

```bash
python3 sucker_seek/detect_circle_distances.py sucker_seek --mm-per-pixel 0.125
```

注意：单目相机在画面倾斜时像素距离不等于平面真实距离。要测量精度，应固定相机正对工件，或用棋盘格完成相机标定及透视矫正；有 ZED 深度数据时可进一步测三维距离。
