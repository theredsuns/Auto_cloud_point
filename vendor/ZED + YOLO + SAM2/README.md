# ZED Camera + YOLO + SAM2 RGB-D 数据采集系统说明文档

## 1. 项目简介

本文档介绍如何使用 **Stereolabs ZED 深度相机** 替代 Intel RealSense，实现：

* RGB 图像采集
* 深度图采集
* 相机内参保存
* YOLO目标检测
* SAM2实例分割
* Mask后处理优化
* RGB-D-Mask数据保存

最终生成的数据可用于：

* BundleSDF
* BundleTrack
* NeRF重建
* 点云生成
* CAD模型配准

整体流程：

```
                 ZED Camera
                     |
          -----------------------
          |                     |
        RGB Image          Depth Image
          |
          |
      YOLO Detection
          |
          |
       Bounding Box
          |
          |
        SAM2 Segmentation
          |
          |
     Mask Post Processing
          |
          |
 -------------------------------
 |             |                |
RGB          Depth            Mask

```

---

# 2. 系统环境

## 2.1 硬件要求

推荐配置：

| 设备     | 要求                         |
| ------ | -------------------------- |
| GPU    | NVIDIA CUDA GPU            |
| 显存     | >=8GB                      |
| Camera | ZED / ZED2 / ZED2i / ZED X |
| CUDA   | >=11.8                     |
| Python | 3.10+                      |

---

# 3. 软件依赖

## 3.1 ZED SDK安装

首先安装：

```
ZED SDK
```

安装完成后测试：

```python
import pyzed.sl as sl

print(sl.Camera())
```

无报错说明 ZED Python接口安装成功。

---

## 3.2 Python环境

推荐：

```
Python 3.10
CUDA 12.x
PyTorch 2.x
```

创建环境：

```bash
conda create -n zed_sam2 python=3.10

conda activate zed_sam2
```

---

## 3.3 安装Python依赖

```bash
pip install numpy

pip install opencv-python

pip install ultralytics

pip install pyzed

pip install torch torchvision
```

安装SAM2：

```bash
cd sam2

pip install -e .
```

---

# 4. 项目目录结构

推荐：

```
ZED_YOLO_SAM2/

│
├── capture_zed_yolo_sam2.py
│
├── sam2/
│
├── checkpoints/
│      └── sam2.1_hiera_tiny.pt
│
├── weights/
│      └── best.pt
│
└── my_zed_data/

```

---

# 5. 数据保存结构

运行程序后：

```
my_zed_data/

├── rgb/
│    ├── 000000.png
│    ├── 000001.png
│
├── depth/
│    ├── 000000.png
│    ├── 000001.png
│
├── masks/
│    ├── 000000.png
│    ├── 000001.png
│
└── cam_K.txt

```

---

# 6. RGB数据说明

保存格式：

```
PNG
```

颜色格式：

```
BGR
```

尺寸：

例如：

```
1280 × 720
```

读取：

```python
rgb=cv2.imread(
    "rgb/000000.png"
)
```

---

# 7. Depth数据说明

ZED输出：

```
float32
```

单位：

```
毫米(mm)
```

例如：

```
1200
```

表示：

```
1.2m
```

保存：

```python
depth.astype(np.uint16)
```

读取：

```python
depth=cv2.imread(
    "depth/000000.png",
    -1
)
```

---

# 8. Mask数据说明

格式：

```
uint8
```

像素值：

背景：

```
0
```

目标：

```
255
```

读取：

```python
mask>0
```

表示有效区域。

---

# 9. 相机内参

保存：

```
cam_K.txt
```

格式：

```
fx 0 cx
0 fy cy
0 0 1
```

对应：

[
K=
\begin{bmatrix}
f_x&0&c_x\
0&f_y&c_y\
0&0&1
\end{bmatrix}
]

---

# 10. ZED相机初始化

初始化：

```python
init = sl.InitParameters()
```

分辨率：

```python
init.camera_resolution =
sl.RESOLUTION.HD720
```

输出：

```
1280×720
```

FPS：

```python
init.camera_fps=30
```

---

# 11. 深度模式选择

## PERFORMANCE

速度最快：

```python
sl.DEPTH_MODE.PERFORMANCE
```

适合：

* 实时跟踪
* 在线重建

---

## QUALITY

精度最高：

```python
sl.DEPTH_MODE.QUALITY
```

适合：

* 三维重建
* CAD匹配

推荐：

```
QUALITY
```

---

# 12. YOLO检测流程

输入：

```
RGB Image
```

输出：

```
Bounding Box
```

格式：

```
[x1,y1,x2,y2]
```

代码：

```python
results=model.predict(
    image
)
```

---

# 13. SAM2分割流程

YOLO输出：

```
Bounding Box
```

作为SAM2：

```
Box Prompt
```

生成：

```
Binary Mask
```

流程：

```
YOLO bbox

      |

SAM2

      |

Object Mask

```

---

# 14. Mask后处理

用于减少：

* 边缘锯齿
* 内部空洞
* 小噪声

处理步骤：

## 14.1 最大连通域

保留：

```
最大目标区域
```

删除：

```
背景
小碎片
```

---

## 14.2 Morphological Close

作用：

```
填充mask孔洞
```

---

## 14.3 Morphological Open

作用：

```
去除边缘毛刺
```

---

## 14.4 Gaussian Blur

作用：

```
平滑mask边界
```

---

# 15. RGB-D生成点云

相机模型：

[
X=(u-c_x)Z/f_x
]

[
Y=(v-c_y)Z/f_y
]

[
Z=depth
]

---

# 16. BundleSDF兼容要求

BundleSDF需要：

```
RGB

Depth

Mask

Camera Intrinsic

```

对应：

```
rgb/

depth/

masks/

cam_K.txt

```

要求：

所有图像尺寸一致：

```
RGB:
1280×720


Depth:
1280×720


Mask:
1280×720

```

---

# 17. 运行方法

## 17.1 激活环境

```bash
conda activate zed_sam2
```

---

## 17.2 检查ZED连接

运行：

```bash
python -c "import pyzed.sl as sl; print('ZED OK')"
```

输出：

```
ZED OK
```

说明驱动正常。

---

## 17.3 检查GPU

```bash
python
```

进入：

```python
import torch

print(torch.cuda.is_available())

print(torch.cuda.get_device_name())
```

正常：

```
True

NVIDIA GPU名称
```

---

## 17.4 修改模型路径

打开：

```
capture_zed_yolo_sam2.py
```

修改：

YOLO：

```python
yolo_model=YOLO(
"your/best.pt"
)
```

SAM2配置：

```python
sam2=build_sam2(

"sam2.1_hiera_t.yaml",

"sam2.1_hiera_tiny.pt"

)
```

---

## 17.5 启动采集程序

进入代码目录：

```bash
cd ZED_YOLO_SAM2
```

运行：

```bash
python capture_zed_yolo_sam2.py
```

成功后显示：

```
ZED opened

Camera intrinsic saved
```

并弹出窗口：

```
ZED YOLO SAM2
```

---

# 18. 数据采集操作

窗口运行时：

## 保存当前帧

按：

```
S
```

保存：

```
rgb/
depth/
masks/
```

例如：

```
Saved 000000.png
```

---

## 退出程序

按：

```
ESC
```

程序退出：

```
zed.close()
```

---

# 19. 输出数据检查

查看文件：

```bash
ls my_zed_data
```

应该包含：

```
rgb

depth

masks

cam_K.txt
```

检查数量：

```bash
ls rgb | wc -l

ls depth | wc -l

ls masks | wc -l
```

三个数量必须一致。

---

# 20. 常见问题

## 问题1

```
cloud before:0
```

原因：

Depth无有效值。

检查：

```python
print(depth.max())
```

正常：

```
>0
```

---

## 问题2

Mask存在但是点云为空

检查：

```python
rgb.shape

depth.shape

mask.shape
```

必须：

```
(H,W)
```

一致。

---

## 问题3

深度单位错误

错误：

```
毫米当米
```

例如：

```
1200
```

应该转换：

```python
depth_m=depth/1000
```

---

## 问题4

SAM2显存不足

降低：

```python
camera_resolution
```

或者：

```python
sam2.1_hiera_tiny
```

替代：

```
large模型
```

---

# 21. 推荐采集参数

针对：

* BundleSDF
* NeRF
* CAD点云匹配

推荐：

```
Resolution:

HD720


FPS:

30


Depth Mode:

QUALITY


Unit:

MILLIMETER


YOLO confidence:

0.7


SAM2:

box prompt

```
