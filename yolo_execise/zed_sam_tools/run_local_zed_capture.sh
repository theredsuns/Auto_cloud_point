#!/usr/bin/env bash
# 编译检查并运行本机 ZED 手动机身+机翼采集程序。
# 每次采集会保存：原图、深度、深度置信度、机身/机翼/组合掩码和遮罩预览图。

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CAPTURE_PROGRAM="${SCRIPT_DIR}/local_zed_capture.py"
SAM2_CONFIG="${SCRIPT_DIR}/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_CHECKPOINT="${SCRIPT_DIR}/sam2/checkpoints/sam2.1_hiera_tiny.pt"

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "缺少必需文件：$1" >&2
        exit 1
    fi
}

require_file "${CAPTURE_PROGRAM}"
require_file "${SAM2_CONFIG}"
require_file "${SAM2_CHECKPOINT}"

cd "${SCRIPT_DIR}"

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${SCRIPT_DIR}/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"

echo "[1/2] 编译检查 Python 程序..."
python3 -m py_compile \
    local_zed_capture.py \
    zed_yolo_sam2_live.py \
    zed_yolo_sam2_icp_reconstruct.py

echo "[2/2] 启动本机 ZED 手动采集程序..."
echo "启动后选择：1=仅机翼，2=仅机体，3=机翼+机体。"
echo "默认完全手动框选：仅机翼模式绝不会自动添加 BODY；需要自动 BODY 框时加 --auto-body。"
echo "操作：主窗口按 Enter 冻结画面。"
echo "实时会显示 RGB 主窗口、Depth 深度伪彩窗口；每次保存后会后台更新并打开 ICP 融合点云窗口。"
echo "每帧会记录 /tracer5/IMU_data 的姿态与加速度；ICP 使用相邻帧 IMU yaw 约束旋转，不积分加速度。"
echo "按所选模式拖框 BODY（机身）和/或 WING（机翼），每次按 Enter 确认框。"
echo "遮罩预览：双部件模式下 B/W 切换部件；左键补漏，右键去误选，U 撤销。"
echo "C 清当前部件全部点，X 只清红叉，A 清当前模式下的全部点。"
echo "机身=绿色，机翼=橙色；修正后 Enter/S/空格 保存并进入下一张。"
echo "自动质量不足只会警告，不再拦截你的人工确认；R 重画框，Q 取消。"
echo "提示：不需要激活虚拟环境；首次加载 SAM2 时请等待进度提示。"
exec python3 -u "${CAPTURE_PROGRAM}" "$@"
