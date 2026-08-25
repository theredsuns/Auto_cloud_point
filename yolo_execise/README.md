# YOLO 训练工具

- `yolo_body_wing_classification/`：把 body 与 wing 照片分别放入两个文件夹后进行分类训练。
- `yolo_body_wing_dataset/`：使用 body/wing mask 进行分割训练。
- `libraries/yolo_runtime/`：本地 CPU YOLO 运行库，训练时自动使用，无需安装 CUDA。
- `libraries/sam2/` 与 `libraries/sam2_runtime/`：SAM2 分割软件库与运行依赖。
- `libraries/venv_yolo/`：旧的 Python 虚拟环境；当前本地训练不依赖它。

`install_gpu_torch.sh` 会安装 CUDA 加速版 PyTorch；安装后训练脚本会自动使用 NVIDIA GPU。
