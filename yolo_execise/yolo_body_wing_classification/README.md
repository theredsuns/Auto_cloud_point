# 机体 / 机翼照片分类训练

这是“按文件夹放照片”的 YOLO 分类训练工具。它不会要求 mask。

把照片放到：

```text
yolo_body_wing_classification/
├── input/
│   ├── background/ # 空场景、桌面、其他物体；不含机体/机翼
│   ├── body/       # 只放机体照片
│   └── wing/       # 只放机翼照片
```

每类至少加入 100 张不同场景。`background` 是实时检测拒绝空画面的必要类别；没有它，模型必然会把空画面误判为 body 或 wing。

同一张照片中如果同时有机体和机翼，不要放进这里；应使用 `yolo_body_wing_dataset/` 的分割训练流程。分类模型只能判断整张画面类别，不能框出真实物体边界；要“圈住”物体必须用带 mask/边界框的分割或检测模型。

## 创建数据集

```bash
cd ~/object_seek
python3 yolo_body_wing_classification/prepare_classification_dataset.py
```

脚本会生成 `dataset/train/body`、`dataset/train/wing`、`dataset/val/body`、`dataset/val/wing`。

## 训练

本项目已经有本地 CPU 版 YOLO 运行库，**不需要安装 Ultralytics，也不要继续下载 CUDA/NCCL 包**。

然后：

```bash
cd ~/object_seek/yolo_body_wing_classification
./train_yolo_classify.sh "$PWD/dataset"
```

结果模型是 `runs/body_wing_classifier/weights/best.pt`。

## 测试已训练模型

把未参与训练的新照片放进 `test_images/body` 或 `test_images/wing`（文件夹只用于整理，模型不会读取文件夹名），然后运行：

```bash
python3 test_body_wing_model.py
```

窗口中按 `A` 上一张、`B` 下一张、`S` 保存带预测文字的结果、`Q` 退出。

## 实时摄像头检测

```bash
python3 live_body_wing_camera.py
```

使用 ZED 时运行 `python3 live_body_wing_camera.py --zed`。按 `Q` 退出。USB 摄像头可使用 `--camera 1` 选择编号。
