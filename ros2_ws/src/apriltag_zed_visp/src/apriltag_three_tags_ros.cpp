#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <deque>
#include <algorithm>
#include <memory>
#include <chrono>
#include <mutex>
#include <cstdio>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <sl/Camera.hpp>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2/LinearMath/Quaternion.hpp"
#include "tf2/LinearMath/Matrix3x3.hpp"

using namespace std;
using namespace cv;
using namespace std::chrono_literals;

// ============================================================
// 系统常量配置
// ============================================================
// AprilTag 标签尺寸（单位：米）- 必须与实际标签尺寸一致
const double TAG_SIZE_ID0 = 0.064;  // ID0 基准标签：15厘米
const double TAG_SIZE_ID1 = 0.064;  // ID1 辅助标签：15厘米
const double TAG_SIZE_ID2 = 0.064;  // ID2 目标标签：6厘米

// 标签 ID 定义
const int BASE_TAG_ID_0 = 0;       // 主基准标签 ID
const int BASE_TAG_ID_1 = 1;       // 辅助基准标签 ID（用于稳定性和标尺校正）
const int TARGET_TAG_ID = 2;       // 目标测量标签 ID

// ROS2 配置
const int ROS_DOMAIN_ID = 36;      // ROS2 域 ID，避免与其他节点冲突
const string TOPIC_NAME = "Trace5_zed_relative";  // 发布话题名称

// ============================================================
// 调试变量（用于诊断 Z 轴波动问题）
// ============================================================
double g_cap_x=0,g_cap_y=0,g_cap_z=0,g_cap_rx=0,g_cap_ry=0,g_cap_rz=0,g_cap_d=0,g_cap_r1x=0,g_cap_r1y=0,g_cap_r1z=0,g_cap_r1rx=0,g_cap_r1ry=0,g_cap_r1rz=0,g_cap_r1d=0;
double g_cap_prev_d=0;
bool g_bref_ok=false; double g_bref_d01=0, g_bref_d02=0;
double g_dcam_mv=0, g_dcam_dd=0;
bool g_ref_ok=false; Vec3d g_ref_t(0,0,0); Mat g_ref_R=Mat::eye(3,3,CV_64F);
bool g_cap_ready=false;
double g_dbg_pnpz0 = 0, g_dbg_zedz0 = -1, g_dbg_pnpz2 = 0, g_dbg_zedz2 = -1;
int g_dbg_frame = 0;

// 标定畸变系数全局变量
// ============================================================
bool g_use_calib_dist = false;
vector<double> g_calib_dist_coeffs;

// ============================================================
// ID0 与 ID1 之间的已知物理距离（单位：米）
// 这是标定好的刚体上两个标签的固定相对位置
// ============================================================
double ID0_TO_ID1_X = 0;  // auto-calibrated from first 50 frames
double g_d01_sum=0; int g_d01_n=0; bool g_d01_ready=false;
const double ID0_TO_ID1_Y = 0.000;   // Y 轴距离：0厘米（共面）
const double ID0_TO_ID1_Z = 0.000;   // Z 轴距离：0厘米（共面）

// ============================================================
// 全局显示字符串（启动时填充，用于终端显示）
// ============================================================
string g_intrinsics_source = "???";      // 内参来源：标定文件 或 ZED出厂
string g_intrinsics_values = "";         // 内参数值显示
string g_distortion_info = "";           // 畸变处理信息

// ============================================================
// 一维自适应卡尔曼滤波器类
// 根据残差动态调整过程噪声和测量噪声，实现自适应滤波
// ============================================================
class KalmanFilter1D {
private:
    double x_est;          // 状态估计值
    double p_est;          // 估计协方差
    double q;              // 当前过程噪声（动态调整）
    double r;              // 当前测量噪声（动态调整）
    bool initialized;      // 是否已初始化

    // 自适应参数
    double q_base;         // 基础过程噪声（初始值）
    double r_base;         // 基础测量噪声（初始值）
    double q_min;          // 过程噪声最小值（防止过小）
    double q_max;          // 过程噪声最大值（防止过大）
    double r_min;          // 测量噪声最小值
    double r_max;          // 测量噪声最大值
    double adapt_factor;   // 自适应调整系数（越大调整越快）

    // 残差统计（用于自适应判断）
    double residual_sum;   // 残差累积和
    int residual_count;    // 残差累积次数
    double last_residual;  // 上一帧残差

public:
    // 构造函数：设置基础噪声参数
    // process_noise: 基础过程噪声（越小越相信模型）
    // measurement_noise: 基础测量噪声（越小越相信测量）
    // adapt_f: 自适应调整系数（建议 0.05~0.2，越大响应越快）
    KalmanFilter1D(double process_noise = 0.001, double measurement_noise = 0.01, double adapt_f = 0.1)
        : q(process_noise), r(measurement_noise),
          q_base(process_noise), r_base(measurement_noise),
          q_min(process_noise * 0.1), q_max(process_noise * 20),
          r_min(measurement_noise * 0.1), r_max(measurement_noise * 20),
          adapt_factor(adapt_f),
          residual_sum(0), residual_count(0), last_residual(0),
          initialized(false) {}

    // 自适应噪声调整：根据残差大小动态调整 q 和 r
    // residual: 当前残差（测量值 - 预测值）
    void adaptNoise(double residual) {
        // 更新残差统计
        residual_sum += std::abs(residual);
        residual_count++;
        if (residual_count > 10) {
            residual_count = 0;
            residual_sum = 0;
        }

        // 计算当前残差的归一化幅度
        double residual_norm = std::abs(residual);

        // 自适应策略：
        // 残差大 → 测量值与预测值差距大 → 说明模型假设可能不成立（物体在移动）
        //          → 增大 q（相信测量，允许状态快速变化）
        //          → 减小 r（不相信模型预测）
        // 残差小 → 测量值与预测值一致 → 模型假设成立（物体静止）
        //          → 减小 q（相信模型，保持状态稳定）
        //          → 增大 r（不相信测量噪声）

        // 计算调整因子（优化：静止时更强滤波，运动时更快响应）
        double q_factor, r_factor;
        if (residual_norm > 0.008) {
            // 残差较大：物体在移动，快速响应
            q_factor = 1.0 + adapt_factor * 10;
            r_factor = 1.0 - adapt_factor * 5;
        } else if (residual_norm > 0.002) {
            // 残差中等：轻微调整
            q_factor = 1.0 + adapt_factor * 3;
            r_factor = 1.0 - adapt_factor * 2;
        } else {
            // 残差较小：物体静止，强力平滑滤波
            q_factor = 1.0 - adapt_factor * 2;
            r_factor = 1.0 + adapt_factor * 2;
        }

        // 应用调整并限制范围
        q = std::max(q_min, std::min(q_max, q * q_factor));
        r = std::max(r_min, std::min(r_max, r * r_factor));

        // 缓回归到基础值（防止参数过度偏离）
        q = q * 0.95 + q_base * 0.05;
        r = r * 0.95 + r_base * 0.05;

        last_residual = residual;
    }

    // 滤波处理：输入测量值，返回滤波后的估计值
    double filter(double measurement) {
        // 首次初始化：直接使用测量值作为初始估计
        if (!initialized) {
            x_est = measurement;
            p_est = 1.0;
            initialized = true;
            return x_est;
        }

        // 预测阶段：基于上一状态预测当前状态
        double x_pred = x_est;           // 状态预测（假设静止模型）
        double p_pred = p_est + q;       // 协方差预测（过程噪声累积）

        // 计算残差（用于自适应调整）
        double residual = measurement - x_pred;

        // 更新阶段：卡尔曼增益计算 + 状态修正
        double k = p_pred / (p_pred + r);              // 卡尔曼增益
        x_est = x_pred + k * residual;                 // 更新状态估计
        p_est = (1 - k) * p_pred;                      // 更新协方差

        // 自适应噪声调整（在状态更新后调用）
        adaptNoise(residual);

        return x_est;
    }

    // 获取当前噪声参数（用于调试）
    double getCurrentQ() const { return q; }
    double getCurrentR() const { return r; }
};

// ============================================================
// 高级多维度滤波器类
// 结合了：裁剪均值滤波 + 卡尔曼滤波 + EMA低通滤波
// 用于对 6D 位姿（x,y,z,rx,ry,rz）和距离进行综合滤波
// ============================================================
class AdvancedFilter {
private:
    // 历史数据队列（存储最近多帧数据）
    deque<double> history;       // 距离历史
    deque<double> history_x;     // X 坐标历史
    deque<double> history_y;     // Y 坐标历史
    deque<double> history_z;     // Z 坐标历史
    deque<double> history_rx;    // 绕X轴旋转历史
    deque<double> history_ry;    // 绕Y轴旋转历史
    deque<double> history_rz;    // 绕Z轴旋转历史

    int max_size;                // 历史队列最大长度
    double alpha;                // EMA 平滑因子（0~1，越大响应越快）
    bool initialized;            // 是否已初始化

    // 滤波后的最新值（用于 EMA 和跳变检测）
    double last_filtered;        // 距离
    double last_x, last_y, last_z;
    double last_rx, last_ry, last_rz;

    mutex mtx;                   // 线程互斥锁（保护历史数据）

    // 各维度独立的卡尔曼滤波器
    KalmanFilter1D kf_x;
    KalmanFilter1D kf_y;
    KalmanFilter1D kf_z;
    KalmanFilter1D kf_rx;
    KalmanFilter1D kf_ry;
    KalmanFilter1D kf_rz;
    KalmanFilter1D kf_dist;

public:
    // 构造函数：初始化滤波器参数
    // size: 历史队列长度（越大滤波越重，响应越慢）
    // smooth_factor: EMA 平滑因子（0.1=慢响应稳定，0.5=快响应）
    AdvancedFilter(int size = 30, double smooth_factor = 0.3) 
        : max_size(size), alpha(smooth_factor), initialized(false),
          kf_x(0.0002, 0.002), kf_y(0.0002, 0.002), kf_z(0.0002, 0.002),
          kf_rx(0.0005, 0.005), kf_ry(0.0005, 0.005), kf_rz(0.0005, 0.005),
          kf_dist(0.0002, 0.002) {
        last_filtered = last_x = last_y = last_z = 0;
        last_rx = last_ry = last_rz = 0;
    }

    // 添加一帧新数据到历史队列
    // x,y,z: 位置坐标
    // rx,ry,rz: 旋转向量（弧度）
    // distance: 三维距离
    void add(double x, double y, double z, double rx, double ry, double rz, double distance) {
        lock_guard<mutex> lock(mtx);  // 加锁保护线程安全

        // 首次初始化：直接记录初始值
        if (!initialized) {
            last_filtered = distance;
            last_x = x; last_y = y; last_z = z;
            last_rx = rx; last_ry = ry; last_rz = rz;
            initialized = true;
            history.push_back(distance);
            history_x.push_back(x);
            history_y.push_back(y);
            history_z.push_back(z);
            history_rx.push_back(rx);
            history_ry.push_back(ry);
            history_rz.push_back(rz);
            // 初始化卡尔曼滤波器
            kf_x.filter(x);
            kf_y.filter(y);
            kf_z.filter(z);
            kf_rx.filter(rx);
            kf_ry.filter(ry);
            kf_rz.filter(rz);
            kf_dist.filter(distance);
            return;
        }

        // 计算与上一帧的跳变幅度（用于检测异常）
        double jump_dist = sqrt(pow(x - last_x, 2) + pow(y - last_y, 2) + pow(z - last_z, 2)) * 1000.0;  // mm
        double jump_rot = sqrt(pow(rx - last_rx, 2) + pow(ry - last_ry, 2) + pow(rz - last_rz, 2)) * 180.0 / M_PI;  // deg

        // 添加到历史队列
        history.push_back(distance);
        history_x.push_back(x);
        history_y.push_back(y);
        history_z.push_back(z);
        history_rx.push_back(rx);
        history_ry.push_back(ry);
        history_rz.push_back(rz);

        // 保持队列长度不超过最大值（移除最旧数据）
        if ((int)history.size() > max_size) {
            history.pop_front();
            history_x.pop_front();
            history_y.pop_front();
            history_z.pop_front();
            history_rx.pop_front();
            history_ry.pop_front();
            history_rz.pop_front();
        }
    }

    // 计算标准差（用于评估数据稳定性）
    double getStdDev(const deque<double>& data) {
        if (data.size() < 2) return 0.0;
        double mean = 0.0;
        for (double v : data) mean += v;
        mean /= data.size();
        double var = 0.0;
        for (double v : data) var += (v - mean) * (v - mean);
        return sqrt(var / (data.size() - 1));
    }

    // 裁剪均值滤波：去除前后各 trim_percent% 的极端值后取平均
    // 有效抑制离群值（如标签检测跳变）
    double getTrimmedMean(const deque<double>& data, double trim_percent = 0.3) {
        if (data.empty()) return 0;
        deque<double> sorted = data;
        sort(sorted.begin(), sorted.end());
        
        int trim_count = (int)(sorted.size() * trim_percent / 2);
        double sum = 0;
        int count = 0;
        
        for (int i = trim_count; i < (int)sorted.size() - trim_count; i++) {
            sum += sorted[i];
            count++;
        }
        
        if (count > 0) return sum / count;
        return sorted[sorted.size() / 2];
    }

    // EMA 低通滤波（指数移动平均）
    // new_val: 新测量值
    // last_val: 上一帧滤波值
    // a: 平滑因子（0~1），越大越跟随新值，越小越平滑
    double lowPass(double new_val, double last_val, double a) {
        return a * new_val + (1 - a) * last_val;
    }

    // 加权均值：越新的数据权重越大（平方权重）
    double getWeightedMean(const deque<double>& data) {
        if (data.empty()) return 0;
        double sum = 0;
        double weight_sum = 0;
        int n = data.size();
        for (int i = 0; i < n; i++) {
            double w = (i + 1) * (i + 1);  // 平方权重
            sum += data[i] * w;
            weight_sum += w;
        }
        return sum / weight_sum;
    }

    // 获取综合滤波后的平滑值
    // 滤波流程：裁剪均值 → 卡尔曼滤波 → EMA低通
    void getSmoothed(double& x, double& y, double& z, 
                    double& rx, double& ry, double& rz, double& distance) {
        lock_guard<mutex> lock(mtx);
        
        if (history.empty()) {
            x = y = z = rx = ry = rz = distance = 0;
            return;
        }

        // 第一步：裁剪均值滤波（去除极端值）
        double raw_z = getTrimmedMean(history_z);
        double raw_x = getTrimmedMean(history_x);
        double raw_y = getTrimmedMean(history_y);
        double raw_rx = getTrimmedMean(history_rx);
        double raw_ry = getTrimmedMean(history_ry);
        double raw_rz = getTrimmedMean(history_rz);
        double raw_dist = getTrimmedMean(history);

        // 第二步：卡尔曼滤波（最优估计）
        z = kf_z.filter(raw_z);
        x = kf_x.filter(raw_x);
        y = kf_y.filter(raw_y);
        rx = kf_rx.filter(raw_rx);
        ry = kf_ry.filter(raw_ry);
        rz = kf_rz.filter(raw_rz);
        distance = kf_dist.filter(raw_dist);

        // 第三步：EMA 低通滤波（进一步平滑）
        z = lowPass(z, last_z, alpha);
        x = lowPass(x, last_x, alpha);
        y = lowPass(y, last_y, alpha);
        rx = lowPass(rx, last_rx, alpha);
        ry = lowPass(ry, last_ry, alpha);
        rz = lowPass(rz, last_rz, alpha);
        distance = lowPass(distance, last_filtered, alpha);

        // 更新上一帧滤波值（用于下一帧 EMA）
        last_x = x;
        last_y = y;
        last_z = z;
        last_rx = rx;
        last_ry = ry;
        last_rz = rz;
        last_filtered = distance;
    }

    // 清空历史数据（重置滤波器）
    void clear() {
        lock_guard<mutex> lock(mtx);
        history.clear();
        history_x.clear();
        history_y.clear();
        history_z.clear();
        history_rx.clear();
        history_ry.clear();
        history_rz.clear();
        initialized = false;
    }

    // 判断滤波器是否已稳定（历史数据达到阈值）
    int getCount() { lock_guard<mutex> lk(mtx); return (int)history.size(); }
    int getMaxSize() const { return max_size; }
    bool isStable() {
        lock_guard<mutex> lock(mtx);
        return history.size() >= max_size * 0.6;
    }
};

// ============================================================
// 位姿转换工具函数
// ============================================================

// 旋转向量 → 旋转矩阵（使用 Rodrigues 变换）
Mat rvecToMatrix(const Vec3d& rvec) {
    Mat R;
    Rodrigues(rvec, R);
    return R;
}

// 旋转矩阵 → 旋转向量（使用 Rodrigues 变换）
Vec3d matrixToRvec(const Mat& R) {
    Vec3d rvec;
    Rodrigues(R, rvec);
    return rvec;
}

// 计算变换矩阵的逆矩阵
// R: 旋转矩阵 (3x3)
// t: 平移向量 (3x1)
// 返回: 4x4 齐次变换矩阵的逆
Mat inverseTransform(const Mat& R, const Mat& t) {
    Mat T = Mat::eye(4, 4, CV_64F);
    R.copyTo(T(Rect(0, 0, 3, 3)));
    t.copyTo(T(Rect(3, 0, 1, 3)));
    return T.inv();
}

// 应用变换矩阵到三维点
// T: 4x4 齐次变换矩阵
// x,y,z: 输入点坐标
// ox,oy,oz: 输出变换后坐标
void transformPoint(const Mat& T, double x, double y, double z, 
                   double& ox, double& oy, double& oz) {
    Mat p = (Mat_<double>(4, 1) << x, y, z, 1.0);
    Mat tp = T * p;
    ox = tp.at<double>(0, 0);
    oy = tp.at<double>(1, 0);
    oz = tp.at<double>(2, 0);
}

// 相乘两个变换矩阵
Mat multiplyTransforms(const Mat& T1, const Mat& T2) {
    return T1 * T2;
}

// 从齐次变换矩阵中提取旋转矩阵和平移向量
void extractTransform(const Mat& T, Mat& R, Mat& t) {
    R = T(Rect(0, 0, 3, 3)).clone();
    t = T(Rect(3, 0, 1, 3)).clone();
}

// ============================================================
// ROS2 节点类：三标签基准系统
// 负责发布相对位姿数据到 ROS2 话题
// ============================================================
class ThreeTagSystemNode : public rclcpp::Node {
public:
    // 构造函数：初始化 ROS2 节点和发布者
    ThreeTagSystemNode() : Node("three_tag_system") {
        // 创建话题发布者，消息类型为 Float64MultiArray，队列长度为 10
        publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(TOPIC_NAME, 10);
        
        // 打印启动信息
        RCLCPP_INFO(this->get_logger(), "三标签基准系统已启动");
        RCLCPP_INFO(this->get_logger(), "DOMAIN ID: %d", ROS_DOMAIN_ID);
        RCLCPP_INFO(this->get_logger(), "话题名: %s", TOPIC_NAME.c_str());
        RCLCPP_INFO(this->get_logger(), "基准标签: ID0 (主), ID1 (辅助)");
        RCLCPP_INFO(this->get_logger(), "目标标签: ID2");
    }

    // 发布相对位姿数据
    // x,y,z: 相对位置（毫米）
    // rx,ry,rz: 相对姿态（弧度）
    // distance: 相对距离（毫米）
    // id0_found/id1_found/id2_found: 标签检测状态
    void publishRelative(double x, double y, double z, 
                        double rx, double ry, double rz,
                        double distance, bool id0_found, bool id1_found, bool id2_found) {
        auto message = std_msgs::msg::Float64MultiArray();
        message.data = {
            x, y, z, distance,
            rx, ry, rz,
            id0_found ? 1.0 : 0.0,
            id1_found ? 1.0 : 0.0,
            id2_found ? 1.0 : 0.0
        };
        publisher_->publish(message);
    }

private:
    // ROS2 发布者指针
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_;
};

// ============================================================
// 绘制 3D 坐标轴（用于可视化标签位姿）
// ============================================================
void draw3DAxis(Mat& image, const Vec3d& rvec, const Vec3d& tvec, 
                const Mat& camera_matrix, const Mat& dist_coeffs, float length) {
    // 定义 3D 坐标轴点：原点 + X/Y/Z 轴端点
    vector<Point3f> axis_points = {
        Point3f(0, 0, 0),
        Point3f(length, 0, 0),    // X 轴（红色）
        Point3f(0, length, 0),    // Y 轴（绿色）
        Point3f(0, 0, length)     // Z 轴（蓝色）
    };

    // 将 3D 点投影到图像平面
    vector<Point2f> image_points;
    projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs, image_points);

    // 绘制带箭头的坐标轴
    if (image_points.size() >= 4) {
        arrowedLine(image, image_points[0], image_points[1], Scalar(0, 0, 255), 2);  // X: 红色
        arrowedLine(image, image_points[0], image_points[2], Scalar(0, 255, 0), 2);  // Y: 绿色
        arrowedLine(image, image_points[0], image_points[3], Scalar(255, 0, 0), 2);  // Z: 蓝色
    }
}

// ============================================================
// 终端打印系统信息（实时显示位姿数据）
// ============================================================
void printSystemInfo(bool id0_found, bool id1_found, bool id2_found,
                    double id0_x, double id0_y, double id0_z,
                    double id1_x, double id1_y, double id1_z,
                    double rel_x, double rel_y, double rel_z,
                    double rel_rx, double rel_ry, double rel_rz,
                    double rel_distance, bool stable,
                    double rel1_x, double rel1_y, double rel1_z,
                    double rel1_rx, double rel1_ry, double rel1_rz,
                    double rel1_dist,
                    bool id0_r, bool id2_r,
                    double rrel_x, double rrel_y, double rrel_z) {
    // 打印标题和系统状态（不刷新屏幕，直接追加）
    cout << "\n\n==================================================" << endl;
    cout << "   三标签基准系统 (ID0+ID1 -> ID2)              " << endl;
    cout << "==================================================" << endl;
    cout << "  内参来源: " << g_intrinsics_source << endl;
    cout << "  内参值:   " << g_intrinsics_values << endl;
    cout << "  " << g_distortion_info << endl;
    cout << "  基准标签: ID0 (主) + ID1 (辅助稳定)" << endl;
    cout << "  目标标签: ID2" << endl;
    cout << "  ROS DOMAIN ID: " << ROS_DOMAIN_ID << endl;
    cout << "  话题名: " << TOPIC_NAME << endl;
    cout << "  稳定状态: " << (stable ? "✅ 已稳定" : "⏳ 收敛中...") << endl;
    cout << "==================================================" << endl;
    cout << endl;
    cout << fixed << setprecision(3);  // 输出精度：小数点后3位
    cout << endl;

    // 打印标签检测状态
    cout << "  标签检测状态:" << endl;
    cout << "    ID0: " << (id0_found ? "✅ 检测到" : "❌ 未检测") << endl;
    cout << "    ID1: " << (id1_found ? "✅ 检测到" : "❌ 未检测") << endl;
    cout << "    ID2: " << (id2_found ? "✅ 检测到" : "❌ 未检测") << endl;
    cout << endl;
    
    // 打印 ID0 在相机坐标系中的位置（毫米）
    if (id0_found) {
        cout << "  ID0 相机坐标系 (mm):" << endl;
        cout << "    X: " << id0_x * 1000 << "  Y: " << id0_y * 1000 << "  Z: " << id0_z * 1000 << endl;
    }
    // 打印 ID1 在相机坐标系中的位置（毫米）
    if (id1_found) {
        cout << "  ID1 相机坐标系 (mm):" << endl;
        cout << "    X: " << id1_x * 1000 << "  Y: " << id1_y * 1000 << "  Z: " << id1_z * 1000 << endl;
    }
    
    cout << endl;
    // 打印 ID2 相对于 ID0 的位姿（基准坐标系）
    if (id0_found && id2_found) {
        cout << "  ID2 相对于 ID0 的位姿 (基准坐标系):" << endl;
        cout << "    位置 (mm):  X: " << rel_x * 1000 << endl;
        cout << "                Y: " << rel_y * 1000 << endl;
        cout << "                Z: " << rel_z * 1000 << endl;
        cout << "    距离:      " << rel_distance * 1000 << " mm" << endl;
        cout << "    姿态 (度): Rx: " << rel_rx * 180.0 / CV_PI << "°" << endl;
        cout << "                Ry: " << rel_ry * 180.0 / CV_PI << "°" << endl;
        cout << "                Rz: " << rel_rz * 180.0 / CV_PI << "°" << endl;
    } else {
        cout << "  请确保 ID0 和 ID2 标签在视野内..." << endl;
    }

    // 打印 ID1 相对于 ID0 的位姿（用于校准和稳定性参考）
    if (id0_found && id1_found) {
        cout << endl;
        cout << "  ID1 相对于 ID0 的位姿 (基准坐标系):" << endl;
        cout << "    位置 (mm):  X: " << rel1_x * 1000 << endl;
        cout << "                Y: " << rel1_y * 1000 << endl;
        cout << "                Z: " << rel1_z * 1000 << endl;
        cout << "    距离:      " << rel1_dist * 1000 << " mm" << endl;
        cout << "    姿态 (度): Rx: " << rel1_rx * 180.0 / CV_PI << "°" << endl;
        cout << "                Ry: " << rel1_ry * 180.0 / CV_PI << "°" << endl;
        cout << "                Rz: " << rel1_rz * 180.0 / CV_PI << "°" << endl;
    }
    cout << endl;

    // 打印右目相机的相对位姿（用于对比验证）
    if (id0_r && id2_r) {
        double rdist = sqrt(rrel_x*rrel_x + rrel_y*rrel_y + rrel_z*rrel_z);
        cout << endl;
        cout << "  [右眼] ID2 相对于 ID0 的位姿:" << endl;
        cout << "    位置 (mm):  X: " << rrel_x*1000 << endl;
        cout << "                Y: " << rrel_y*1000 << endl;
        cout << "                Z: " << rrel_z*1000 << endl;
        cout << "    距离:      " << rdist*1000 << " mm" << endl;
    }
    cout << endl;

    // 打印话题数据格式说明
    cout << "  话题数据: [x, y, z, distance, rx, ry, rz, id0_ok, id1_ok, id2_ok]" << endl;
    cout << "==================================================" << endl;
    cout << "  按 ESC 键退出" << endl;
    cout << "==================================================" << endl;
}

// ============================================================
// 主函数：三标签基准测量系统
// 核心流程：
// 1. 初始化 ZED 相机
// 2. 加载相机内参（优先使用标定文件，否则用 ZED 出厂参数）
// 3. 创建去畸变映射表
// 4. 主循环：采集图像 → 检测标签 → PnP 解算 → 相对位姿计算 → 滤波 → 发布
// ============================================================
int main(int argc, char** argv) {
    // 设置 ROS2 域 ID（避免与其他 ROS2 系统冲突）
    setenv("ROS_DOMAIN_ID", to_string(ROS_DOMAIN_ID).c_str(), 1);
    
    // 初始化 ROS2 上下文
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ThreeTagSystemNode>();
    
    cout << "正在初始化 ZED 相机..." << endl;
    
    // 创建 ZED 相机对象
    sl::Camera zed;
    sl::InitParameters init_params;
    init_params.camera_resolution = sl::RESOLUTION::HD720;  // 分辨率：1280x720
    init_params.depth_mode = sl::DEPTH_MODE::PERFORMANCE;   // 深度模式：性能优先
    init_params.camera_fps = 30;                             // 帧率：30fps
    init_params.sensors_required = true;

    // 打开 ZED 相机
    sl::ERROR_CODE err = zed.open(init_params);
    if (err != sl::ERROR_CODE::SUCCESS) {
        cerr << "无法打开 ZED 相机: " << sl::toString(err) << endl;
        rclcpp::shutdown();
        return -1;
    }

    cout << "ZED 相机打开成功!" << endl;

    // 获取 ZED 左目相机的出厂内参
    auto zed_params = zed.getCameraInformation().camera_configuration.calibration_parameters.left_cam;
    cout << "ZED 出厂内参: fx=" << zed_params.fx << " fy=" << zed_params.fy
         << " cx=" << zed_params.cx << " cy=" << zed_params.cy << endl;

    // ============================================================
    // 使用自定义标定文件内参
    // ============================================================
    string calib_file = "/home/nkk/camera_calibration/version_1.yaml";
    double fx, fy, cx, cy;

    ifstream cf(calib_file);
    if (cf.is_open()) {
        string line;
        vector<double> K(9, 0);
        vector<double> D;
        while (getline(cf, line)) {
            if (line.empty() || line[0] == '#') continue;
            if (line.find("camera_matrix:") != string::npos) {
                getline(cf, line); getline(cf, line); getline(cf, line);
                size_t s = line.find('['), e = line.find(']');
                if (s != string::npos && e != string::npos) {
                    stringstream ss(line.substr(s+1, e-s-1));
                    string v; int i = 0;
                    while (getline(ss, v, ',') && i < 9) {
                        v.erase(0, v.find_first_not_of(" \t"));
                        K[i++] = stod(v);
                    }
                }
            } else if (line.find("distortion_coefficients:") != string::npos) {
                getline(cf, line); getline(cf, line); getline(cf, line);
                size_t s = line.find('['), e = line.find(']');
                if (s != string::npos && e != string::npos) {
                    stringstream ss(line.substr(s+1, e-s-1));
                    string v;
                    while (getline(ss, v, ',')) {
                        v.erase(0, v.find_first_not_of(" \t"));
                        if (!v.empty()) D.push_back(stod(v));
                    }
                }
            }
        }
        cf.close();
        if (K[0] > 0) {
            fx = K[0]; fy = K[4]; cx = K[2]; cy = K[5];
            
            cout << "\n📊 内参对比:" << endl;
            cout << "┌──────────┬─────────────┬─────────────┬──────────┐" << endl;
            cout << "│ 参数     │ 出厂内参     │ 标定内参     │ 差值(px) │" << endl;
            cout << "├──────────┼─────────────┼─────────────┼──────────┤" << endl;
            printf("│ fx       │ %11.2f │ %11.2f │ %+8.2f │\n", zed_params.fx, fx, fx - zed_params.fx);
            printf("│ fy       │ %11.2f │ %11.2f │ %+8.2f │\n", zed_params.fy, fy, fy - zed_params.fy);
            printf("│ cx       │ %11.2f │ %11.2f │ %+8.2f │\n", zed_params.cx, cx, cx - zed_params.cx);
            printf("│ cy       │ %11.2f │ %11.2f │ %+8.2f │\n", zed_params.cy, cy, cy - zed_params.cy);
            cout << "└──────────┴─────────────┴─────────────┴──────────┘" << endl;
            
            cout << "✅ 使用标定内参 (重投影误差: " << 0.133322 << "px)" << endl;
            g_intrinsics_source = "标定 (version_1.yaml)";
            
            if (!D.empty()) {
                cout << "✅ 使用标定畸变系数: ";
                for (size_t i = 0; i < D.size(); ++i) {
                    cout << D[i];
                    if (i < D.size()-1) cout << ",";
                }
                cout << endl;
                g_use_calib_dist = true;
                g_calib_dist_coeffs = D;
            }
        } else {
            fx = zed_params.fx; fy = zed_params.fy; cx = zed_params.cx; cy = zed_params.cy;
            cout << "⚠️ 标定文件解析失败，使用 ZED 出厂内参" << endl;
            g_intrinsics_source = "ZED 出厂 (factory)";
        }
    } else {
        fx = zed_params.fx; fy = zed_params.fy; cx = zed_params.cx; cy = zed_params.cy;
        cout << "⚠️ 标定文件不存在，使用 ZED 出厂内参" << endl;
        g_intrinsics_source = "ZED 出厂 (factory)";
    }

    // 更新全局显示字符串
    {
        stringstream ss;
        ss << fixed << setprecision(2);
        ss << "fx=" << fx << " fy=" << fy << " cx=" << cx << " cy=" << cy;
        g_intrinsics_values = ss.str();
    }
    g_distortion_info = "去畸变: ON (remap + 零畸变PnP)";

    // ============================================================
    // 构建相机内参矩阵和去畸变配置
    // ============================================================
    // 3x3 内参矩阵：[fx, 0, cx; 0, fy, cy; 0, 0, 1]
    Mat camera_matrix = (Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    // 选择去畸变用的畸变系数
    Mat remap_dist;
    if (g_use_calib_dist && !g_calib_dist_coeffs.empty()) {
        remap_dist = Mat(static_cast<int>(g_calib_dist_coeffs.size()), 1, CV_64F);
        for (size_t i = 0; i < g_calib_dist_coeffs.size(); ++i) {
            remap_dist.at<double>(static_cast<int>(i)) = g_calib_dist_coeffs[i];
        }
        cout << "✅ 去畸变使用标定畸变系数" << endl;
    } else {
        remap_dist = (Mat_<double>(5, 1) <<
            zed_params.disto[0], zed_params.disto[1],
            zed_params.disto[2], zed_params.disto[3],
            zed_params.disto[4]);
        cout << "✅ 去畸变使用 ZED 出厂畸变系数" << endl;
    }

    // ============================================================
    // 预计算去畸变映射表（加速实时处理）
    // ============================================================
    auto cam_info = zed.getCameraInformation().camera_configuration.resolution;
    int img_w = static_cast<int>(cam_info.width);
    int img_h = static_cast<int>(cam_info.height);
    Mat undist_map_x, undist_map_y;
    cv::initUndistortRectifyMap(camera_matrix, remap_dist, Mat(), camera_matrix,
                                Size(img_w, img_h),
                                CV_16SC2, undist_map_x, undist_map_y);

    // PnP 解算使用零畸变模型（因为图像已经过 remap 去畸变）
    Mat dist_coeffs = Mat::zeros(5, 1, CV_64F);
    cout << "✅ 去畸变映射已计算 → PnP 使用零畸变模型" << endl;

    // ============================================================
    // 右目相机配置（独立的内参和去畸变）
    // ============================================================
    auto zed_params_r = zed.getCameraInformation().camera_configuration.calibration_parameters.right_cam;
    Mat camera_matrix_r = (Mat_<double>(3, 3) << zed_params_r.fx, 0, zed_params_r.cx,
                                                      0, zed_params_r.fy, zed_params_r.cy,
                                                      0, 0, 1);
    Mat dist_coeffs_r = (Mat_<double>(5, 1) << zed_params_r.disto[0], zed_params_r.disto[1],
                                                zed_params_r.disto[2], zed_params_r.disto[3],
                                                zed_params_r.disto[4]);
    Mat undist_map_rx, undist_map_ry;
    cv::initUndistortRectifyMap(camera_matrix_r, dist_coeffs_r, Mat(), camera_matrix_r,
                                Size(img_w, img_h), CV_16SC2, undist_map_rx, undist_map_ry);
    cout << "✅ 右目内参: fx=" << zed_params_r.fx << " fy=" << zed_params_r.fy << endl;

    // ============================================================
    // 初始化 AprilTag 检测器
    // ============================================================
    // 使用 AprilTag 36h11 家族（最常用，平衡编码长度和检测鲁棒性）
    Ptr<aruco::Dictionary> dictionary = 
        aruco::getPredefinedDictionary(aruco::DICT_APRILTAG_36h11);

    // 检测器参数：启用亚像素角点精化（提高角点定位精度）
    Ptr<aruco::DetectorParameters> params = aruco::DetectorParameters::create();
    params->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX;  // 亚像素精化
    params->cornerRefinementMaxIterations = 100;                     // 最大迭代次数
    params->cornerRefinementMinAccuracy = 0.001;                     // 收敛精度

    // ============================================================
    // 创建滤波器实例（加强滤波，减小静态波动）
    // ============================================================
    AdvancedFilter relative_filter(60, 0.05);    // ID2→ID0：窗口60帧，EMA=0.05（更稳，响应稍慢）
    AdvancedFilter id1_filter(60, 0.05);         // ID1→ID0：同样参数

    // ============================================================
    // 运行时变量
    // ============================================================
    int frame_count = 0;
    bool id0_found = false, id1_found = false, id2_found = false;  // 左目标签检测状态
    Vec3d id0_tvec, id0_rvec;  // ID0 位姿（平移+旋转）
    Vec3d id1_tvec, id1_rvec;  // ID1 位姿
    Vec3d id2_tvec, id2_rvec;  // ID2 位姿

    Mat frame;
    sl::Mat zed_img, zed_img_r;
    
    // 右目相机标签位姿
    bool id0_r=false, id1_r=false, id2_r=false;
    Vec3d id0_tvec_r, id0_rvec_r, id1_tvec_r, id1_rvec_r, id2_tvec_r, id2_rvec_r;

    // 创建显示窗口
    namedWindow("三标签基准系统 (ID0+ID1 -> ID2)", WINDOW_NORMAL);
    resizeWindow("三标签基准系统 (ID0+ID1 -> ID2)", 1280, 720);
    cout << "✅ 显示窗口已创建，进入主循环..." << endl;

    // ============================================================
    // 主循环：图像采集 → 标签检测 → PnP → 相对位姿 → 滤波 → 发布
    // ============================================================
    ofstream csv_f("/home/nkk/coordate_change/measurement_log.csv");
    csv_f<<"frame,X2,Y2,Z2,D2,X1,Y1,Z1,D1"<<endl;
    int csv_n=0;
    ofstream lf("/home/nkk/coordate_change/measurement_log.txt");
        lf<<"N | ID2_X ID2_Y ID2_Z ID2_Rx ID2_Ry ID2_Rz ID2_D | ID1_X ID1_Y ID1_Z ID1_Rx ID1_Ry ID1_Rz ID1_D"<<endl;
    int ln=0;
    sl::SensorsData imu_data;
    while (rclcpp::ok()) {
        // 采集一帧图像（阻塞式，等待新帧）
        if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
            // 获取左目图像和深度图
            zed.retrieveImage(zed_img, sl::VIEW::LEFT);
            sl::Mat depth_map;
            zed.retrieveMeasure(depth_map, sl::MEASURE::DEPTH);
            
            // 将 ZED 深度图转换为 OpenCV 格式
            Mat depth_raw(depth_map.getHeight(), depth_map.getWidth(), CV_32FC1, depth_map.getPtr<float>());
            Mat depth_undist;
            // 对深度图应用去畸变（与图像去畸变对齐）
            cv::remap(depth_raw, depth_undist, undist_map_x, undist_map_y, cv::INTER_LINEAR);

            // 将 ZED 图像转换为 OpenCV 格式（BGRA → BGR）
            Mat cv_img(zed_img.getHeight(), zed_img.getWidth(), CV_8UC4, zed_img.getPtr<sl::uchar1>());
            cvtColor(cv_img, frame, COLOR_BGRA2BGR);

            // 对图像进行去畸变（消除镜头畸变影响）
            cv::remap(frame, frame, undist_map_x, undist_map_y, cv::INTER_LINEAR);
            Mat frame_left = frame.clone();  // 保存左目图像用于最终显示
                // Center ROI box
                int rx=img_w/8, ry=img_h/8, rw=img_w*3/4, rh=img_h*3/4;
                cv::rectangle(frame_left, Point(rx,ry), Point(rx+rw,ry+rh), Scalar(100,100,100), 2);

            // ============================================================
            // 左目相机：AprilTag 检测
            // ============================================================
            vector<int> ids;
            vector<vector<Point2f>> corners;

            // 检测图像中的 AprilTag
            aruco::detectMarkers(frame, dictionary, corners, ids, params);

            // 重置检测状态
            id0_found = id1_found = id2_found = false;

            // 如果检测到标签
            if (!ids.empty()) {
                // 在图像上绘制检测到的标签边框（红色，便于观察角点跳动）
                aruco::drawDetectedMarkers(frame, corners, ids, Scalar(0, 0, 255));

                // 存储 ID0 和 ID1 的角点（用于后续计算）
                vector<Point2f> c0, c1;
                
                // 遍历所有检测到的标签
                for (size_t i = 0; i < ids.size(); i++) {
                    Scalar axis_color; string tag_label; double tag_sz;
                    Vec3d rv, tv;  // 旋转向量和平移向量

                    // 根据标签 ID 设置参数
                    if (ids[i] == BASE_TAG_ID_0) {
                        id0_found = true; tag_sz = TAG_SIZE_ID0; c0 = corners[i];
                        axis_color = Scalar(0, 255, 0); tag_label = "ID0";
                    } else if (ids[i] == BASE_TAG_ID_1) {
                        id1_found = true; tag_sz = TAG_SIZE_ID1; c1 = corners[i];
                        axis_color = Scalar(255, 0, 255); tag_label = "ID1";
                    } else if (ids[i] == TARGET_TAG_ID) {
                        id2_found = true; tag_sz = TAG_SIZE_ID2;
                        axis_color = Scalar(0, 0, 255); tag_label = "ID2 (6cm)";
                    } else { continue; }  // 忽略其他 ID 的标签

                    // ============================================================
                    // PnP 解算：从 2D 角点估计 3D 位姿
                    // ============================================================
                    // 构建标签的 3D 世界坐标（以标签中心为原点的平面模型）
                    double h = tag_sz / 2;
                    vector<Point3f> obj = {{-h,h,0},{h,h,0},{h,-h,0},{-h,-h,0}};
                    
                    // 使用 IPPE 算法（针对平面目标优化的 PnP 算法）
                    solvePnP(obj, corners[i], camera_matrix, dist_coeffs, rv, tv, false, SOLVEPNP_IPPE);
                    Mat R_tag; Rodrigues(rv, R_tag); double cth = std::abs(R_tag.at<double>(2,2)); if (cth < 0.95 && cth > 0.3) tv[2] *= cth;

                    // ===== DEBUG: 记录原始 PnP Z 值（用于诊断 Z 轴波动）=====
                    {
                        double _pnpz = tv[2];
                        if (ids[i] == BASE_TAG_ID_0) g_dbg_pnpz0 = _pnpz;
                        else if (ids[i] == TARGET_TAG_ID) g_dbg_pnpz2 = _pnpz;
                    }
                    // ===== END DEBUG =====

                    // ============================================================
                    // 深度融合已禁用（ZED深度噪声导致波动增大）
                    // 使用纯 PnP 解算，通过滤波和双路径融合提高稳定性
                    // ============================================================

                    // Compute tag center (used for depth+display)
                    Point2f ctr(0,0); for(auto& c : corners[i]) ctr+=c; ctr*=0.25f;

                    // ZED depth: sample 9x9 patch around tag center, median
                    double depth_z = -1;
                    vector<float> dvals;
                    for(int dy=-4;dy<=4;dy++) for(int dx=-4;dx<=4;dx++) {
                        int px=(int)ctr.x+dx, py=(int)ctr.y+dy;
                        if(px>=0 && px<depth_undist.cols && py>=0 && py<depth_undist.rows) {
                            float d = depth_undist.at<float>(py, px);
                            if(d > 0.001f && std::isfinite(d)) dvals.push_back(d);
                        }
                    }
                    if(!dvals.empty()) { sort(dvals.begin(),dvals.end()); depth_z=dvals[dvals.size()/2]; }
                    if (ids[i]==BASE_TAG_ID_0) g_dbg_zedz0=depth_z;
                    else if (ids[i]==TARGET_TAG_ID) g_dbg_zedz2=depth_z;

                    // 保存标签位姿
                    if (ids[i] == BASE_TAG_ID_0) { id0_rvec = rv; id0_tvec = tv; }
                    else if (ids[i] == BASE_TAG_ID_1) { id1_rvec = rv; id1_tvec = tv; }
                    // ZED raw depth at tag center
                    Point2f ctr2(0,0); for(auto& c : corners[i]) ctr2+=c; ctr2*=0.25f;
                    int rx=(int)ctr2.x, ry=(int)ctr2.y;
                    float rd = (rx>=0&&rx<depth_raw.cols&&ry>=0&&ry<depth_raw.rows) ? depth_raw.at<float>(ry,rx) : -1;
                    if (ids[i]==BASE_TAG_ID_0) g_dbg_zedz0=rd;
                    else if (ids[i]==TARGET_TAG_ID) g_dbg_zedz2=rd;
                    else { id2_rvec = rv; id2_tvec = tv; }
                        double cth2 = fabs(R_tag.at<double>(2,2)); g_dbg_zedz2 = acos(cth2)*180/M_PI;

                    // 在图像上绘制 3D 坐标轴和标签名称
                    aruco::drawAxis(frame_left, camera_matrix, dist_coeffs, rv, tv, tag_sz * 0.5);
                    // Draw center cross on tag
                    cv::line(frame_left, Point(ctr.x-8,ctr.y), Point(ctr.x+8,ctr.y), axis_color, 2);
                    cv::line(frame_left, Point(ctr.x,ctr.y-8), Point(ctr.x,ctr.y+8), axis_color, 2);
                    cv::circle(frame_left, ctr, 5, axis_color, 2);
                    // Mark TL corner (depth sampling point) with a filled square
                    putText(frame_left, tag_label, Point(corners[i][0].x, corners[i][0].y - 10),
                           FONT_HERSHEY_SIMPLEX, 0.5, axis_color, 2);
                }

                // ============================================================
                // 右目相机：独立的标签检测和 PnP（用于双目融合提高稳定性）
                // ============================================================
                zed.retrieveImage(zed_img_r, sl::VIEW::RIGHT);
                Mat cv_img_r(zed_img_r.getHeight(), zed_img_r.getWidth(), CV_8UC4, zed_img_r.getPtr<sl::uchar1>());
                cvtColor(cv_img_r, frame, COLOR_BGRA2BGR);
                cv::remap(frame, frame, undist_map_rx, undist_map_ry, cv::INTER_LINEAR);

                vector<int> ids_r; vector<vector<Point2f>> corners_r;
                aruco::detectMarkers(frame, dictionary, corners_r, ids_r, params);
                id0_r = id1_r = id2_r = false;
                
                for (size_t i = 0; i < ids_r.size(); i++) {
                    double tag_sz;
                    if (ids_r[i] == BASE_TAG_ID_0) tag_sz = TAG_SIZE_ID0;
                    else if (ids_r[i] == BASE_TAG_ID_1) tag_sz = TAG_SIZE_ID1;
                    else if (ids_r[i] == TARGET_TAG_ID) tag_sz = TAG_SIZE_ID2;
                    else continue;
                    
                    double h = tag_sz / 2;
                    vector<Point3f> obj = {{-h,h,0},{h,h,0},{h,-h,0},{-h,-h,0}};
                    Vec3d rv, tv;
                    solvePnP(obj, corners_r[i], camera_matrix_r, dist_coeffs, rv, tv, false, SOLVEPNP_IPPE);
                    
                    if (ids_r[i] == BASE_TAG_ID_0) { id0_r=true; id0_tvec_r=tv; id0_rvec_r=rv; }
                    else if (ids_r[i] == BASE_TAG_ID_1) { id1_r=true; id1_tvec_r=tv; id1_rvec_r=rv; }
                    else { id2_r=true; id2_tvec_r=tv; id2_rvec_r=rv; }
                }

                // ============================================================
                // 相对位姿计算：ID2 相对于 ID0 的位姿
                // 条件：必须同时检测到 ID0 和 ID2
                // ============================================================
                if (id0_found && id2_found) {
                    frame_count++;

                    // ============================================================
                    // 标尺校正：已禁用（ID0-ID1距离未知）
                    // ID0和ID1仅用于旋转平均和双路径融合提高稳定性
                    // ============================================================
                    double scale_factor = 1.0;
                    bool has_scale = false;

                    // ============================================================
                    // 相对旋转计算：R_rel = R_id0^T * R_id2
                    // 将 ID2 的旋转从相机坐标系转换到 ID0 坐标系
                    // ============================================================
                    // EMA on ID0 rotation to suppress angular jitter
                    static Vec3d ema_r0(0,0,0); static bool ema_r0_ok=false;
                    if(!ema_r0_ok){ema_r0=id0_rvec;ema_r0_ok=true;}
                    ema_r0 = 0.05*id0_rvec + 0.95*ema_r0;
                    Mat R_id0 = rvecToMatrix(ema_r0);
                    Mat R_id2 = rvecToMatrix(id2_rvec);
                    
                    // 如果检测到 ID1，平均 ID0 和 ID1 的旋转（同刚体，噪声抵消）
                    if (id1_found) {
                        Vec3d rv_avg = (id0_rvec + id1_rvec) * 0.5;
                        R_id0 = rvecToMatrix(rv_avg);  // 使用平均旋转作为参考
                    }

                    Mat R_rel = R_id0.t() * R_id2;

                    // ============================================================
                    // 相对平移计算：t_rel = R_id0^T * (t_id2 - t_id0)
                    // 将 ID2 的平移从相机坐标系转换到 ID0 坐标系
                    // ============================================================
                    Vec3d t_rel_raw_vec(id2_tvec[0] - id0_tvec[0],
                                        id2_tvec[1] - id0_tvec[1],
                                        id2_tvec[2] - id0_tvec[2]);
                    Mat t_rel_raw = R_id0.t() * Mat(t_rel_raw_vec);
                    // Scale correction: use ID1→ID0 measured distance to fix PnP scale bias
                    if(id1_found) {
                        static double d01_true = 0; static bool d01_set = false;
                        double d01_now = norm(id1_tvec - id0_tvec);
                        if(!d01_set) { d01_true = d01_now; d01_set = true; }
                        d01_true = 0.005*d01_now + 0.995*d01_true;
                        double sf = d01_true / d01_now;
                        if(sf>0.9 && sf<1.1) t_rel_raw = t_rel_raw * sf;
                    }
                    // 3-frame median pre-filter on raw relative Z
                    static deque<double> z_pre3(3,0); static int z_pre3_n=0;
                    double myrawz = t_rel_raw.at<double>(2);
                    z_pre3[z_pre3_n%3] = myrawz; z_pre3_n++;
                    if(z_pre3_n>=3){ vector<double> zs3(z_pre3.begin(),z_pre3.end()); sort(zs3.begin(),zs3.end());
                        if(fabs(myrawz-zs3[1])>0.005) t_rel_raw.at<double>(2)=zs3[1]; }

                    // ============================================================
                    // 双路径融合：同时通过 ID0 和 ID1 计算相对位姿，取平均
                    // 原理：两条路径独立，噪声可以抵消
                    // ============================================================
                    if (id1_found) {
                        Mat R_id1 = rvecToMatrix(id1_rvec);
                        Mat R_rel2 = R_id1.t() * R_id2;
                        Vec3d r1 = matrixToRvec(R_rel);
                        Vec3d r2 = matrixToRvec(R_rel2);
                        R_rel = rvecToMatrix((r1 + r2) * 0.5);  // 平均旋转

                        // 通过 ID1 计算的平移（需要加上 ID0→ID1 的偏移）
                        Mat t2 = R_id1.t() * Mat(Vec3d(id2_tvec[0]-id1_tvec[0],
                                                       id2_tvec[1]-id1_tvec[1],
                                                       id2_tvec[2]-id1_tvec[2]));
                        t2.at<double>(0) += ID0_TO_ID1_X;
                        t2.at<double>(1) += ID0_TO_ID1_Y;
                        t2.at<double>(2) += ID0_TO_ID1_Z;
                        t_rel_raw = (t_rel_raw + t2) * 0.5;  // 平均平移
                    }

                    // ============================================================
                    // 双目融合：左目 + 右目结果取平均（进一步降低噪声）
                    // ============================================================
                    if (id0_r && id2_r) {
                        Mat R_id0_r = rvecToMatrix(id0_rvec_r);
                        Mat R_id2_r = rvecToMatrix(id2_rvec_r);
                        Mat R_rel_r = R_id0_r.t() * R_id2_r;
                        Mat t_r = R_id0_r.t() * Mat(Vec3d(id2_tvec_r[0]-id0_tvec_r[0],
                                                          id2_tvec_r[1]-id0_tvec_r[1],
                                                          id2_tvec_r[2]-id0_tvec_r[2]));

                        // 右目也使用双路径（通过 ID1）
                        if (id1_r) {
                            Mat R_id1_r = rvecToMatrix(id1_rvec_r);
                            Mat R_rel2_r = R_id1_r.t() * R_id2_r;
                            Vec3d r1r = matrixToRvec(R_rel_r);
                            Vec3d r2r = matrixToRvec(R_rel2_r);
                            R_rel_r = rvecToMatrix((r1r + r2r) * 0.5);
                            Mat t2_r = R_id1_r.t() * Mat(Vec3d(id2_tvec_r[0]-id1_tvec_r[0],
                                                               id2_tvec_r[1]-id1_tvec_r[1],
                                                               id2_tvec_r[2]-id1_tvec_r[2]));
                            t2_r.at<double>(0)+=ID0_TO_ID1_X; t2_r.at<double>(1)+=ID0_TO_ID1_Y; t2_r.at<double>(2)+=ID0_TO_ID1_Z;
                            t_r = (t_r + t2_r) * 0.5;
                        }

                        // 左目和右目结果取平均
                        Vec3d rv_l = matrixToRvec(R_rel);
                        Vec3d rv_r = matrixToRvec(R_rel_r);
                        R_rel = rvecToMatrix((rv_l + rv_r) * 0.5);
                        t_rel_raw = (t_rel_raw + t_r) * 0.5;
                    }

                    // ============================================================
                    // 应用标尺校正（如果有有效尺度因子）
                    // ============================================================
                    Mat t_rel;
                    if (has_scale) {
                        t_rel = scale_factor * t_rel_raw;
                    } else {
                        t_rel = t_rel_raw.clone();
                    }

                    // ===== DEBUG: 输出 CSV 调试数据到 stderr (已禁用) =====
                    // {
                    //     g_dbg_frame++;
                    //     double relz_raw = t_rel_raw.at<double>(2,0);
                    //     double relz_final = t_rel.at<double>(2,0);
                    //     double sf = has_scale ? scale_factor : 1.0;
                    //     fprintf(stderr, "DBGZ,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    //             g_dbg_frame, g_dbg_pnpz0, g_dbg_zedz0, g_dbg_pnpz2, g_dbg_zedz2,
                    //             relz_raw, sf, relz_final);
                    // }
                    // ===== END DEBUG =====

                    // ============================================================
                    // 稳定性评估：计算帧间变化量
                    // ============================================================
                    Vec3d rel_rvec = matrixToRvec(R_rel);
                    double rel_distance = norm(t_rel);

                    static Vec3d prev_t_rel(0,0,0);
                    static bool prev_init = false;
                    double delta_t = 0;
                    if (prev_init) {
                        delta_t = norm(t_rel - Mat(prev_t_rel)) * 1000.0;  // 变化量（毫米）
                    }
                    prev_t_rel = Vec3d(t_rel.at<double>(0,0), t_rel.at<double>(1,0), t_rel.at<double>(2,0));
                    prev_init = true;

                    // 稳定性评分（0~1，越高越稳定）
                    static double stability_score = 1.0;
                    if (delta_t < 5.0) {
                        stability_score = 0.95 * stability_score + 0.05 * 1.0;
                    } else {
                        stability_score = 0.95 * stability_score + 0.05 * 0.0;
                    }

                    // ============================================================
                    // 滤波处理：将当前位姿加入滤波器
                    // ============================================================
                    // Zero-reference correction using ID1→ID0
                    if(id1_found && g_ref_ok) {
                        Mat R0r=rvecToMatrix(id0_rvec); Mat t1r=R0r.t()*(Mat(id1_tvec)-Mat(id0_tvec));
                    // Apply both translation and rotation corrections
                    Vec3d err_t(t1r.at<double>(0)-g_ref_t[0], t1r.at<double>(1)-g_ref_t[1], t1r.at<double>(2)-g_ref_t[2]);
                    Mat R1r=rvecToMatrix(id1_rvec); Mat R_err=R0r.t()*R1r*g_ref_R.t();
                    t_rel = R_err.t() * t_rel - Mat(err_t);
                    double ref_d = sqrt(g_ref_t[0]*g_ref_t[0]+g_ref_t[1]*g_ref_t[1]+g_ref_t[2]*g_ref_t[2]);
                    double cur_d = sqrt(t1r.at<double>(0)*t1r.at<double>(0)+t1r.at<double>(1)*t1r.at<double>(1)+t1r.at<double>(2)*t1r.at<double>(2));
                    if(fabs(cur_d-ref_d)/ref_d > 0.05) putText(frame_left, "OUT OF RANGE", Point(img_w/2-80,60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,255), 2);
                    }
                    // Scale correction using ID1→ID0 as real-time reference
                    if(g_bref_ok && id1_found) {
                        Mat R0b=rvecToMatrix(id0_rvec); Mat t1b=R0b.t()*(Mat(id1_tvec)-Mat(id0_tvec));
                        double cur_d01=sqrt(t1b.at<double>(0)*t1b.at<double>(0)+t1b.at<double>(1)*t1b.at<double>(1)+t1b.at<double>(2)*t1b.at<double>(2));
                    double mv = fabs((cur_d01-g_bref_d01)*1000);
                        double sf = g_bref_d01 / cur_d01;
                        if(sf>0.85 && sf<1.15) { t_rel = t_rel * sf; static int dbg=0; if(++dbg%30==0) fprintf(stderr,"SCALE: sf=%.4f cur=%.1f ref=%.1f\n",sf,cur_d01*1000,g_bref_d01*1000); }
                    }
                    relative_filter.add(
                        t_rel.at<double>(0, 0), t_rel.at<double>(1, 0), t_rel.at<double>(2, 0),
                        rel_rvec[0], rel_rvec[1], rel_rvec[2],
                        rel_distance
                    );

                    // 检测刚体是否在移动（变化量 > 5mm 视为移动）
                    bool assembly_moving = (delta_t > 5.0);

                    // 获取滤波后的平滑值
                    double smooth_x, smooth_y, smooth_z;
                    double smooth_rx, smooth_ry, smooth_rz, smooth_dist;
                    relative_filter.getSmoothed(smooth_x, smooth_y, smooth_z,
                                               smooth_rx, smooth_ry, smooth_rz, smooth_dist);

                    // ============================================================
                    // 姿态额外平滑：对角度再应用一次强 EMA（提高角度稳定性）
                    // ============================================================
                    static double arx=0, ary=0, arz=0;
                    static bool aang_init = false;
                    if (!aang_init) { arx=smooth_rx; ary=smooth_ry; arz=smooth_rz; aang_init=true; }
                    const double AA = 0.04;  // 强平滑，响应慢但稳定
                    arx = AA * smooth_rx + (1-AA) * arx;
                    ary = AA * smooth_ry + (1-AA) * ary;
                    arz = AA * smooth_rz + (1-AA) * arz;
                    smooth_rx = arx; smooth_ry = ary; smooth_rz = arz;

                    // ============================================================
                    // 计算 ID1→ID0 相对位姿（用于相机运动补偿）
                    // ============================================================
                    double r1x=0, r1y=0, r1z=0, r1rx=0, r1ry=0, r1rz=0, r1d=0;
                    if (id1_found) {
                        Mat R_id0_m = rvecToMatrix(ema_r0);
                        Mat R_id1_m = rvecToMatrix(id1_rvec);
                        // ID1 相对于 ID0 的平移（在 ID0 坐标系下）
                        Mat t1 = R_id0_m.t() * (Mat(id1_tvec) - Mat(id0_tvec));
                        double raw_x=t1.at<double>(0,0), raw_y=t1.at<double>(1,0), raw_z=t1.at<double>(2,0);
                        double raw_d = sqrt(raw_x*raw_x + raw_y*raw_y + raw_z*raw_z);
                        // ID1 相对于 ID0 的旋转
                        Mat R_rel1 = R_id0_m.t() * R_id1_m;
                        Vec3d rr1 = matrixToRvec(R_rel1);
                        // 添加到滤波器
                        id1_filter.add(raw_x, raw_y, raw_z, rr1[0], rr1[1], rr1[2], raw_d);
                        double fd;
                        id1_filter.getSmoothed(r1x, r1y, r1z, r1rx, r1ry, r1rz, fd);
                        r1d = sqrt(r1x*r1x + r1y*r1y + r1z*r1z);
                    }

                    // ============================================================
                    // 相机运动补偿：
                    // 原理：ID0 和 ID1 是固定在刚体上的，它们的相对位置不应改变
                    // 当相机移动时，ID1→ID0 的原始测量会变化，这个变化量就是相机运动引起的
                    // 将这个变化量从 ID2→ID0 的结果中减去，消除相机运动的影响
                    // ============================================================
                    // Auto-calibrate ID0→ID1 distance
                    if (!g_d01_ready && id1_found) {
                        Mat R0c=rvecToMatrix(ema_r0); Mat t1c=R0c.t()*(Mat(id1_tvec)-Mat(id0_tvec));
                        g_d01_sum+=sqrt(t1c.at<double>(0)*t1c.at<double>(0)+t1c.at<double>(1)*t1c.at<double>(1)+t1c.at<double>(2)*t1c.at<double>(2));
                        g_d01_n++;
                        if(g_d01_n>=50){ ID0_TO_ID1_X=g_d01_sum/g_d01_n; g_d01_ready=true; cout<<"\nCALIB: ID0-ID1="<<ID0_TO_ID1_X*1000<<"mm (50 frames)"<<endl; }
                    }
                    double corr_x = smooth_x, corr_y = smooth_y, corr_z = smooth_z;
                    double corr_rx = smooth_rx, corr_ry = smooth_ry, corr_rz = smooth_rz;

                    // 距离的额外强 EMA 平滑
                    double cur_dist = sqrt(corr_x*corr_x+corr_y*corr_y+corr_z*corr_z);
                    if (id1_found && id1_filter.isStable()) {
                        // 当前帧 ID1→ID0 的原始测量值（相机坐标系）
                        double raw1_x = id1_tvec[0] - id0_tvec[0];
                        double raw1_y = id1_tvec[1] - id0_tvec[1];
                        double raw1_z = id1_tvec[2] - id0_tvec[2];
                        // 相机运动偏移 = 原始值 - 滤波值（滤波值是"真值"）
                        // 将反向偏移应用到 ID2→ID0 的结果中
                        corr_x -= (raw1_x - r1x);
                        corr_y -= (raw1_y - r1y);
                        corr_z -= (raw1_z - r1z);
                    }

                    // ============================================================
                    // 终端显示和 ROS2 发布
                    // ============================================================
                    if (true) {
                        // 计算右目相对位姿（用于对比验证）
                        double rrx=0, rry=0, rrz=0;
                        if (id0_r && id2_r) {
                            Mat R0r = rvecToMatrix(id0_rvec_r);
                            Mat tr = R0r.t() * Mat(Vec3d(id2_tvec_r[0]-id0_tvec_r[0],
                                                          id2_tvec_r[1]-id0_tvec_r[1],
                                                          id2_tvec_r[2]-id0_tvec_r[2]));
                            rrx=tr.at<double>(0); rry=tr.at<double>(1); rrz=tr.at<double>(2);
                        }

                        // 打印系统信息到终端（每10帧打印一次，约0.3秒）
                        static int print_count = 0;
                        if (print_count++ % 10 == 0) {
                            cout << fixed << setprecision(3);
                            cout << "\n[Frame " << frame_count << "] ";
                            cout << "ID2->ID0: ";
                            cout << "X=" << corr_x*1000 << " ";
                            cout << "Y=" << corr_y*1000 << " ";
                            cout << "Z=" << corr_z*1000 << " mm | ";
                            cout << "dist=" << smooth_dist*1000 << " mm ";
                            cout << (relative_filter.isStable() ? "✅" : "⏳") << endl;
                        }
                    }

                    // 发布 ROS2 话题（单位：毫米）
                    node->publishRelative(
                        corr_x * 1000, corr_y * 1000, corr_z * 1000,
                        corr_rx, corr_ry, corr_rz,
                        smooth_dist * 1000,
                        id0_found, id1_found, id2_found
                    );

                    // ============================================================
                    // Capture for log (after all corrections applied)
                    g_cap_x=corr_x; g_cap_y=corr_y; g_cap_z=corr_z; g_cap_rx=corr_rx; g_cap_ry=corr_ry; g_cap_rz=corr_rz;
                    g_cap_d=sqrt(corr_x*corr_x+corr_y*corr_y+corr_z*corr_z);
                    if(id1_found){g_cap_r1x=r1x;g_cap_r1y=r1y;g_cap_r1z=r1z;g_cap_r1rx=r1rx;g_cap_r1ry=r1ry;g_cap_r1rz=r1rz;g_cap_r1d=r1d;g_cap_ready=true;}

                    // 图像显示：绘制位姿信息
                    // ============================================================
                    Vec3d smooth_rvec_vis(corr_rx, corr_ry, corr_rz);

                    // 显示距离和移动状态
                    stringstream ss;
                    ss << fixed << setprecision(1);
                    if(!g_d01_ready) putText(frame_left, "CALIB "+to_string(g_d01_n)+"/50", Point(img_w/2-80,30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,255), 2);
                    ss << "L: " << sqrt(corr_x*corr_x+corr_y*corr_y+corr_z*corr_z)*1000 << "mm"
                       << (assembly_moving ? " [MOV]" : "");
                    putText(frame, ss.str(), Point(20, 30),
                           FONT_HERSHEY_SIMPLEX, 0.7,
                           assembly_moving ? Scalar(0, 255, 0) : Scalar(0, 255, 255), 2);

                    // 显示坐标值
                    stringstream ss2;
                    ss2 << "XYZ=" << corr_x*1000 << " " << corr_y*1000 << " " << corr_z*1000
                        << "mm  R=" << corr_rx*180/M_PI << " " << corr_ry*180/M_PI << " " << corr_rz*180/M_PI << "deg";
                    putText(frame_left, ss2.str(), Point(20, 55),
                           FONT_HERSHEY_SIMPLEX, 0.45, Scalar(255, 255, 0), 2);


                    // ID1→ID0 reference (prominent)
                    if (id1_found) {
                        double err_mm = fabs(r1d - sqrt(ID0_TO_ID1_X*ID0_TO_ID1_X))*1000;
                        Scalar clr = err_mm<1.0 ? Scalar(0,255,0) : (err_mm<3.0 ? Scalar(0,255,255) : Scalar(0,0,255));
                        stringstream ss3;
                        ss3 << fixed << setprecision(1);
                        ss3 << "ID1->ID0:" << r1d*1000 << "mm (" << err_mm << "mm err)";
                        putText(frame_left, ss3.str(), Point(20, 75),
                               FONT_HERSHEY_SIMPLEX, 0.65, clr, 2);
                        stringstream ss4;
                        ss4 << "XYZ=" << r1x*1000 << " " << r1y*1000 << " " << r1z*1000
                            << "mm  R=" << r1rx*180/M_PI << " " << r1ry*180/M_PI << " " << r1rz*180/M_PI << "deg";
                        putText(frame_left, ss4.str(), Point(20, 95),
                               FONT_HERSHEY_SIMPLEX, 0.4, clr, 1);
                    }
                }
            }

            // 显示图像（无论是否检测到标签都显示）
            stringstream sd; sd<<fixed<<setprecision(1)<<"ViewAngle: ID0="<<g_dbg_zedz0<<" ID2="<<g_dbg_zedz2<<"deg"; putText(frame_left,sd.str(),Point(20,130),FONT_HERSHEY_SIMPLEX,0.45,Scalar(0,255,255),1);
            // Check if all 3 tag centers are in ROI
            bool in_roi = false;
            if (id0_found && id1_found && id2_found) {
                auto px = [&](double X, double Y, double Z) -> Point2f {
                    double u = camera_matrix.at<double>(0,0)*X/Z + camera_matrix.at<double>(0,2);
                    double v = camera_matrix.at<double>(1,1)*Y/Z + camera_matrix.at<double>(1,2);
                    return Point2f(u,v);
                };
                Point2f p0=px(id0_tvec[0],id0_tvec[1],id0_tvec[2]);
                Point2f p1=px(id1_tvec[0],id1_tvec[1],id1_tvec[2]);
                Point2f p2=px(id2_tvec[0],id2_tvec[1],id2_tvec[2]);
            bool dist_ok = (id0_tvec[2]>0.2 && id0_tvec[2]<0.8 && id2_tvec[2]>0.2 && id2_tvec[2]<0.8);
                in_roi = p0.x>rx&&p0.x<rx+rw&&p0.y>ry&&p0.y<ry+rh
                      && p1.x>rx&&p1.x<rx+rw&&p1.y>ry&&p1.y<ry+rh
                      && p2.x>rx&&p2.x<rx+rw&&p2.y>ry&&p2.y<ry+rh;
            }
            cv::putText(frame_left, in_roi?"VALID (20-80cm)":"OUT OF RANGE", Point(rx+5,ry+20), FONT_HERSHEY_SIMPLEX, 0.7, in_roi?Scalar(0,255,0):Scalar(0,0,255), 2);
            if(!g_bref_ok) putText(frame_left,"Press B to set reference",Point(img_w/2-110,img_h-30),FONT_HERSHEY_SIMPLEX,0.6,Scalar(0,255,255),2);
            else { double mv=0,dd=0; if(id1_found){ Mat R0b=rvecToMatrix(id0_rvec); Mat t1b=R0b.t()*(Mat(id1_tvec)-Mat(id0_tvec)); double d01=sqrt(t1b.at<double>(0)*t1b.at<double>(0)+t1b.at<double>(1)*t1b.at<double>(1)+t1b.at<double>(2)*t1b.at<double>(2)); mv=fabs((d01-g_bref_d01)*1000); } dd=fabs((g_cap_d-g_bref_d02)*1000); stringstream sb; sb<<fixed<<setprecision(1)<<"dCAM:"<<mv<<"mm dID2:"<<dd<<"mm"; Scalar cl=id1_found?(mv<2?Scalar(0,255,0):Scalar(0,0,255)):Scalar(0,200,200); putText(frame_left,sb.str(),Point(img_w-230,50),FONT_HERSHEY_SIMPLEX,0.45,cl,2); }
            zed.getSensorsData(imu_data, sl::TIME_REFERENCE::CURRENT);
            auto a=imu_data.imu.linear_acceleration; auto g=imu_data.imu.angular_velocity;
            double acc=sqrt(a.x*a.x+a.y*a.y+a.z*a.z), gyr=sqrt(g.x*g.x+g.y*g.y+g.z*g.z);
            stringstream simu; simu<<fixed<<setprecision(3)<<"IMU a:"<<acc<<" g:"<<gyr;
            putText(frame_left,simu.str(),Point(20,img_h-20),FONT_HERSHEY_SIMPLEX,0.4,Scalar(0,255,0),1);
            imshow("三标签基准系统 (ID0+ID1 -> ID2)", frame_left);

            // 按 ESC 键退出
            char key = waitKey(10);
            if(key=='b'&&id1_found&&g_cap_ready){ Mat R0b=rvecToMatrix(id0_rvec); Mat t1b=R0b.t()*(Mat(id1_tvec)-Mat(id0_tvec)); g_bref_d01=sqrt(t1b.at<double>(0)*t1b.at<double>(0)+t1b.at<double>(1)*t1b.at<double>(1)+t1b.at<double>(2)*t1b.at<double>(2)); g_bref_d02=g_cap_d; g_bref_ok=true; fprintf(stderr,"B_PRESSED d01=%.1f d02=%.1f\n",g_bref_d01*1000,g_bref_d02*1000); cout<<"REF SET d01="<<g_bref_d01*1000<<"mm d02="<<g_bref_d02*1000<<"mm"<<endl; }
            if(key == 'z' && id1_found) { Mat R0r=rvecToMatrix(id0_rvec); Mat t1r=R0r.t()*(Mat(id1_tvec)-Mat(id0_tvec)); g_ref_t=Vec3d(t1r.at<double>(0),t1r.at<double>(1),t1r.at<double>(2)); Mat R1r=rvecToMatrix(id1_rvec); g_ref_R=R0r.t()*R1r; g_ref_ok=true; cout<<"ZEROED"<<endl; }
            if ((key == 13 || key == 32) && ln < 100 && g_cap_ready) {
                double dd = fabs(g_cap_d - g_cap_r1d);
            lf << setw(2) << ln << " | "
               << setw(7) << fixed << setprecision(1) << g_cap_x*1000 << " " << setw(7) << g_cap_y*1000 << " " << setw(7) << g_cap_z*1000 << " "
               << setw(6) << g_cap_rx*180/M_PI << " " << setw(6) << g_cap_ry*180/M_PI << " " << setw(6) << g_cap_rz*180/M_PI << " " << setw(7) << g_cap_d*1000 << " | "
               << setw(7) << g_cap_r1x*1000 << " " << setw(7) << g_cap_r1y*1000 << " " << setw(7) << g_cap_r1z*1000 << " "
               << setw(6) << g_cap_r1rx*180/M_PI << " " << setw(6) << g_cap_r1ry*180/M_PI << " " << setw(6) << g_cap_r1rz*180/M_PI << " " << setw(7) << g_cap_r1d*1000  << " " << setw(5) << g_dcam_mv << " " << setw(5) << g_dcam_dd << endl;
                g_cap_prev_d = g_cap_d;
                ln++;
                cout << "Captured " << ln << "/100  ID2=" << g_cap_d*1000 << "mm R=(" << g_cap_rx*180/M_PI << "," << g_cap_ry*180/M_PI << "," << g_cap_rz*180/M_PI << ")deg | ID1=" << g_cap_r1d*1000 << "mm" << endl;
            }
            if (key == 27) break;
        }
    }

    // 释放资源
    zed.close();
    rclcpp::shutdown();
    return 0;
}