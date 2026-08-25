#ifndef APRILTAG_DETECTOR_NODE_H
#define APRILTAG_DETECTOR_NODE_H

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.hpp>

#include "apriltag_zed_visp/apriltag_36h11_detector.h"
#include "apriltag_zed_visp/pose_estimator.h"
#include "apriltag_zed_visp/error_compensation.h"
#include "apriltag_zed_visp/object_detector.h"

#include <memory>
#include <string>
#include <mutex>
#include <deque>
#include <algorithm>

#ifdef HAVE_ZED_SDK
#include <sl/Camera.hpp>
#endif

namespace apriltag_zed_visp {

// Simple sliding-window filter for relative pose smoothing
struct RelativePoseFilter {
    std::deque<double> hist_x, hist_y, hist_z, hist_rx, hist_ry, hist_rz;
    int max_size;
    double alpha;
    double last_x, last_y, last_z, last_rx, last_ry, last_rz;
    bool initialized;

    RelativePoseFilter(int size = 30, double smooth = 0.15)
        : max_size(size), alpha(smooth), initialized(false) {
        last_x = last_y = last_z = last_rx = last_ry = last_rz = 0.0;
    }

    void reset() {
        hist_x.clear(); hist_y.clear(); hist_z.clear();
        hist_rx.clear(); hist_ry.clear(); hist_rz.clear();
        initialized = false;
    }

    double trimmedMean(const std::deque<double>& data, double trim = 0.2) {
        if (data.empty()) return 0.0;
        std::deque<double> sorted = data;
        std::sort(sorted.begin(), sorted.end());
        int trim_n = static_cast<int>(sorted.size() * trim / 2);
        double sum = 0.0; int n = 0;
        for (int i = trim_n; i < static_cast<int>(sorted.size()) - trim_n; ++i) {
            sum += sorted[i]; ++n;
        }
        return (n > 0) ? sum / n : sorted[sorted.size() / 2];
    }

    double lowPass(double val, double& last) {
        last = alpha * val + (1.0 - alpha) * last;
        return last;
    }

    void add(double x, double y, double z, double rx, double ry, double rz) {
        hist_x.push_back(x); hist_y.push_back(y); hist_z.push_back(z);
        hist_rx.push_back(rx); hist_ry.push_back(ry); hist_rz.push_back(rz);
        if (static_cast<int>(hist_x.size()) > max_size) {
            hist_x.pop_front(); hist_y.pop_front(); hist_z.pop_front();
            hist_rx.pop_front(); hist_ry.pop_front(); hist_rz.pop_front();
        }
        initialized = true;
    }

    void getFiltered(double& x, double& y, double& z,
                     double& rx, double& ry, double& rz) {
        if (!initialized || hist_x.empty()) { x=y=z=rx=ry=rz=0.0; return; }
        x = lowPass(trimmedMean(hist_x), last_x);
        y = lowPass(trimmedMean(hist_y), last_y);
        z = lowPass(trimmedMean(hist_z), last_z);
        rx = lowPass(trimmedMean(hist_rx), last_rx);
        ry = lowPass(trimmedMean(hist_ry), last_ry);
        rz = lowPass(trimmedMean(hist_rz), last_rz);
    }

    double stdDev(const std::deque<double>& data) {
        if (data.size() < 2) return 0.0;
        double mean = 0.0;
        for (double v : data) mean += v;
        mean /= data.size();
        double var = 0.0;
        for (double v : data) var += (v - mean) * (v - mean);
        return std::sqrt(var / (data.size() - 1));
    }

    void getFluctuation(double& std_x, double& std_y, double& std_z,
                        double& std_rx, double& std_ry, double& std_rz) {
        std_x = stdDev(hist_x);
        std_y = stdDev(hist_y);
        std_z = stdDev(hist_z);
        std_rx = stdDev(hist_rx);
        std_ry = stdDev(hist_ry);
        std_rz = stdDev(hist_rz);
    }

    bool isReady() const { return static_cast<int>(hist_x.size()) >= max_size / 3; }
};

class AprilTagDetectorNode : public rclcpp::Node {
public:
    AprilTagDetectorNode();
    ~AprilTagDetectorNode();

private:
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);
    void leftImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
    void rightImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
    
    void processStereoImages();
    
    void publishDetection(const AprilTagDetection& detection,
                          const TagPose& pose,
                          const std_msgs::msg::Header& header);
    
    void publishMarker(const TagPose& pose,
                       const std_msgs::msg::Header& header);
    
    void publishTransform(const TagPose& pose,
                          const std_msgs::msg::Header& header);
    
    void printPosePrecise(const TagPose& pose, double confidence);
    void printPoseComparison(const TagPose& pose_factory, const TagPose& pose_calib,
                             double conf_factory, double conf_calib);
    bool loadCalibrationYAML(const std::string& filepath, CameraParameters& params);
    void processDetection(const AprilTagDetection& det,
                          const std_msgs::msg::Header& header);

    // Multi-tag relative pose
    TagPose estimateSingleTag(const std::vector<cv::Point2f>& corners,
                              PoseEstimator& estimator, double& confidence);
    void computeRelativePose(const TagPose& pose_base, const TagPose& pose_target,
                             cv::Vec3d& rel_t, cv::Vec3d& rel_r, double& distance);
    void printRelativePose(const TagPose& pose_id0, const TagPose& pose_id1,
                           const TagPose& pose_id2,
                           bool id0_found, bool id1_found, bool id2_found);
    void publishRelativePose(const cv::Vec3d& rel_t, const cv::Vec3d& rel_r,
                             double distance, bool id0_found, bool id1_found, bool id2_found,
                             const std_msgs::msg::Header& header);
    
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr right_image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr left_camera_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr right_camera_info_sub_;
    
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr relative_pose_pub_;
    
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    std::unique_ptr<AprilTag36h11Detector> detector_;
    std::unique_ptr<PoseEstimator> pose_estimator_;
    std::unique_ptr<PoseEstimator> pose_estimator_calib_;  // calibrated intrinsics
    std::unique_ptr<ErrorCompensation> error_compensation_;
    std::unique_ptr<ObjectDetector> object_detector_;

    // Undistortion maps (computed from calibrated intrinsics)
    cv::Mat undist_map_x_, undist_map_y_;
    bool undist_map_ready_;
    bool enable_undistort_;
    
    sensor_msgs::msg::Image::ConstSharedPtr last_left_image_;
    sensor_msgs::msg::Image::ConstSharedPtr last_right_image_;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr last_camera_info_;
    
    std::mutex image_mutex_;
    
    std::string camera_frame_;
    std::string tag_frame_;
    int target_tag_id_;
    int base_tag_id_0_;
    int base_tag_id_1_;
    double tag_size_;
    bool use_stereo_;
    bool publish_tf_;
    bool publish_marker_;
    bool enable_compensation_;
    bool print_precise_pose_;
    bool enable_relative_pose_;
    
    bool camera_info_received_;
    bool camera_params_set_;
    bool calib_params_loaded_;
    bool prefer_calibrated_;
    std::string calib_file_path_;
    CameraParameters calib_params_;
    CameraParameters factory_params_original_;  // saved for reference

    // Relative pose filtering
    int filter_window_;
    double filter_alpha_;
    RelativePoseFilter rel_filter_;
    
    void declareParameters();
    void loadBasicParameters();
    void loadAdvancedParameters();
    
    void cameraCaptureLoop();
    
#ifdef HAVE_ZED_SDK
    void zedCaptureLoop();
#endif
    
    bool use_direct_camera_;
    int camera_device_id_;
    std::thread camera_thread_;
    std::atomic<bool> running_;
    
    bool enable_object_detection_;
    bool enable_color_filter_;
    double object_min_area_;
    double object_max_area_;
    
    bool enable_display_;
    
    std::string image_topic_;
    std::string camera_info_topic_;
    std::string left_image_topic_;
    std::string right_image_topic_;
    std::string left_camera_info_topic_;
    
    bool use_zed_sdk_;
    std::string zed_camera_model_;
    int zed_camera_serial_;
    
#ifdef HAVE_ZED_SDK
    std::shared_ptr<sl::Camera> zed_camera_;
    sl::InitParameters zed_init_params_;
#endif
};

} 

#endif 
