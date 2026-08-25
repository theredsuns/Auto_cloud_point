#include "apriltag_zed_visp/apriltag_detector_node.h"
#include <iomanip>
#include <sstream>
#include <functional>
#include <thread>
#include <fstream>

using namespace std::placeholders;

namespace apriltag_zed_visp {

AprilTagDetectorNode::AprilTagDetectorNode()
    : Node("apriltag_detector")
    , camera_info_received_(false)
    , camera_params_set_(false)
    , calib_params_loaded_(false)
    , undist_map_ready_(false)
    , enable_undistort_(true)
    , running_(true)
    , use_direct_camera_(false)
    , camera_device_id_(0)
    , target_tag_id_(2)
    , base_tag_id_0_(0)
    , base_tag_id_1_(1)
    , tag_size_(0.06)
    , use_stereo_(false)
    , publish_tf_(true)
    , publish_marker_(true)
    , enable_compensation_(true)
    , print_precise_pose_(true)
    , enable_relative_pose_(true)
    , prefer_calibrated_(true)
    , filter_window_(30)
    , filter_alpha_(0.15)
    , rel_filter_(30, 0.15)
{
    RCLCPP_INFO(this->get_logger(), "Initializing AprilTag detector node...");
    
    RCLCPP_INFO(this->get_logger(), "Declaring parameters...");
    declareParameters();
    RCLCPP_INFO(this->get_logger(), "Parameters declared");
    
    RCLCPP_INFO(this->get_logger(), "Loading basic parameters...");
    loadBasicParameters();
    RCLCPP_INFO(this->get_logger(), "Basic parameters loaded");
    
    try {
        detector_ = std::make_unique<AprilTag36h11Detector>(target_tag_id_, 1.0, 0.0, true);
        RCLCPP_INFO(this->get_logger(), "Detector created");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create detector: %s", e.what());
    }
    
    try {
        pose_estimator_ = std::make_unique<PoseEstimator>(tag_size_);
        RCLCPP_INFO(this->get_logger(), "Pose estimator created");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create pose estimator: %s", e.what());
    }
    
    try {
        error_compensation_ = std::make_unique<ErrorCompensation>();
        RCLCPP_INFO(this->get_logger(), "Error compensation created");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create error compensation: %s", e.what());
    }
    
    try {
        object_detector_ = std::make_unique<ObjectDetector>();
        RCLCPP_INFO(this->get_logger(), "Object detector created");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create object detector: %s", e.what());
    }
    
    RCLCPP_INFO(this->get_logger(), "Loading advanced parameters...");
    loadAdvancedParameters();
    RCLCPP_INFO(this->get_logger(), "Advanced parameters loaded");

    // Create second pose estimator for calibrated intrinsics
    try {
        pose_estimator_calib_ = std::make_unique<PoseEstimator>(tag_size_);
        RCLCPP_INFO(this->get_logger(), "Calibrated pose estimator created");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create calibrated pose estimator: %s", e.what());
    }

    // Load calibration YAML if provided
    calib_file_path_ = this->get_parameter("calib_file").as_string();
    if (!calib_file_path_.empty()) {
        if (loadCalibrationYAML(calib_file_path_, calib_params_)) {
            // If undistortion is enabled, zero out distortion (image is remapped)
            if (enable_undistort_) {
                calib_params_.dist_coeffs = cv::Mat::zeros(
                    calib_params_.dist_coeffs.rows, 1, CV_64F);
            }
            if (pose_estimator_calib_) {
                pose_estimator_calib_->setCameraParameters(calib_params_);
            }
            calib_params_loaded_ = true;
            RCLCPP_INFO(this->get_logger(), "Calibrated intrinsics loaded from: %s",
                        calib_file_path_.c_str());
            RCLCPP_INFO(this->get_logger(), "  Calib fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f",
                        calib_params_.fx(), calib_params_.fy(),
                        calib_params_.cx(), calib_params_.cy());
            RCLCPP_INFO(this->get_logger(), "  Calib distortion zeroed (undistortion enabled)");
        } else {
            RCLCPP_WARN(this->get_logger(), "Failed to load calibration file: %s",
                        calib_file_path_.c_str());
        }
    } else {
        RCLCPP_INFO(this->get_logger(), "No calibration file specified — comparison disabled");
    }
    
    try {
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        RCLCPP_INFO(this->get_logger(), "TF broadcaster created");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create TF broadcaster: %s", e.what());
    }
    
    try {
        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "tag_pose", 10);
        RCLCPP_INFO(this->get_logger(), "Pose publisher created");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create pose publisher: %s", e.what());
    }
    
    try {
        marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
            "tag_marker", 10);
        RCLCPP_INFO(this->get_logger(), "Marker publisher created");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create marker publisher: %s", e.what());
    }
    
    try {
        debug_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "debug_image", 10);
        RCLCPP_INFO(this->get_logger(), "Debug image publisher created");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create debug image publisher: %s", e.what());
    }

    try {
        relative_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "relative_pose", 10);
        RCLCPP_INFO(this->get_logger(), "Relative pose publisher created");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create relative pose publisher: %s", e.what());
    }
    
    if (use_zed_sdk_) {
        RCLCPP_INFO(this->get_logger(), "Using ZED SDK direct camera mode");
#ifdef HAVE_ZED_SDK
        zed_camera_ = std::make_shared<sl::Camera>();
        
        zed_init_params_.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
        zed_init_params_.coordinate_units = sl::UNIT::METER;
        zed_init_params_.depth_mode = sl::DEPTH_MODE::ULTRA;
        zed_init_params_.camera_resolution = sl::RESOLUTION::HD720;
        zed_init_params_.camera_fps = 30;
        zed_init_params_.enable_image_enhancement = true;
        
        if (zed_camera_serial_ > 0) {
            zed_init_params_.input.setFromSerialNumber(zed_camera_serial_);
            RCLCPP_INFO(this->get_logger(), "Opening ZED camera with serial: %d", zed_camera_serial_);
        }
        
        sl::ERROR_CODE err = zed_camera_->open(zed_init_params_);
        if (err == sl::ERROR_CODE::SUCCESS) {
            RCLCPP_INFO(this->get_logger(), "ZED camera opened successfully");
            
            sl::CameraInformation cam_info = zed_camera_->getCameraInformation();
            sl::CalibrationParameters zed_calib = cam_info.camera_configuration.calibration_parameters;
            sl::CameraParameters calib = zed_calib.left_cam;
            
            CameraParameters params;
            params.camera_matrix(0,0) = calib.fx;
            params.camera_matrix(0,1) = 0;
            params.camera_matrix(0,2) = calib.cx;
            params.camera_matrix(1,0) = 0;
            params.camera_matrix(1,1) = calib.fy;
            params.camera_matrix(1,2) = calib.cy;
            params.camera_matrix(2,0) = 0;
            params.camera_matrix(2,1) = 0;
            params.camera_matrix(2,2) = 1;
            params.dist_coeffs = cv::Mat(5, 1, CV_64F);
            params.dist_coeffs.at<double>(0) = calib.disto[0];
            params.dist_coeffs.at<double>(1) = calib.disto[1];
            params.dist_coeffs.at<double>(2) = calib.disto[2];
            params.dist_coeffs.at<double>(3) = calib.disto[3];
            params.dist_coeffs.at<double>(4) = calib.disto[4];
            params.image_width = cam_info.camera_configuration.resolution.width;
            params.image_height = cam_info.camera_configuration.resolution.height;
            
            factory_params_original_ = params;
            
            cv::Mat K(params.camera_matrix);
            cv::initUndistortRectifyMap(
                K, params.dist_coeffs, cv::Mat(), K,
                cv::Size(params.image_width, params.image_height),
                CV_16SC2, undist_map_x_, undist_map_y_);
            undist_map_ready_ = true;
            
            CameraParameters params_undist = params;
            params_undist.dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
            pose_estimator_->setCameraParameters(params_undist);
            error_compensation_->setImageDimensions(params.image_width, params.image_height);
            
            camera_params_set_ = true;
            camera_info_received_ = true;
            camera_frame_ = "zed_camera_frame";
            
            RCLCPP_INFO(this->get_logger(), "ZED camera params: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f",
                params.fx(), params.fy(), params.cx(), params.cy());
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            camera_thread_ = std::thread(&AprilTagDetectorNode::zedCaptureLoop, this);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to open ZED camera: %s", sl::toString(err).c_str());
            RCLCPP_WARN(this->get_logger(), "Falling back to OpenCV camera capture mode...");
            use_direct_camera_ = true;
        }
#else
        RCLCPP_ERROR(this->get_logger(), "ZED SDK not available. Please rebuild with ZED SDK support.");
        RCLCPP_WARN(this->get_logger(), "Falling back to OpenCV camera capture mode...");
        use_direct_camera_ = true;
#endif
    }
    
    if (use_direct_camera_) {
        RCLCPP_INFO(this->get_logger(), "Using direct camera capture mode");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        camera_thread_ = std::thread(&AprilTagDetectorNode::cameraCaptureLoop, this);
    } else if (use_stereo_) {
        RCLCPP_INFO(this->get_logger(), "Using stereo camera mode (topic subscription)");
        RCLCPP_INFO(this->get_logger(), "  Subscribing to: %s", left_image_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Subscribing to: %s", right_image_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Subscribing to: %s", left_camera_info_topic_.c_str());
        
        left_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            left_image_topic_, 10,
            std::bind(&AprilTagDetectorNode::leftImageCallback, this, _1));
        right_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            right_image_topic_, 10,
            std::bind(&AprilTagDetectorNode::rightImageCallback, this, _1));
        left_camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            left_camera_info_topic_, 10,
            std::bind(&AprilTagDetectorNode::cameraInfoCallback, this, _1));
    } else {
        RCLCPP_INFO(this->get_logger(), "Using monocular camera mode (topic subscription)");
        RCLCPP_INFO(this->get_logger(), "  Subscribing to: %s", image_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Subscribing to: %s", camera_info_topic_.c_str());
        
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            image_topic_, 10,
            std::bind(&AprilTagDetectorNode::imageCallback, this, _1));
        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            camera_info_topic_, 10,
            std::bind(&AprilTagDetectorNode::cameraInfoCallback, this, _1));
    }
    
    RCLCPP_INFO(this->get_logger(), "AprilTag ZED ViSP Detector initialized");
    RCLCPP_INFO(this->get_logger(), "Target Tag ID: %d", target_tag_id_);
    RCLCPP_INFO(this->get_logger(), "Tag Size: %.3f meters (6cm)", tag_size_);
    if (use_direct_camera_) {
        RCLCPP_INFO(this->get_logger(), "Camera Device ID: %d", camera_device_id_);
    }
}

AprilTagDetectorNode::~AprilTagDetectorNode() {
    running_ = false;
    if (camera_thread_.joinable()) {
        camera_thread_.join();
    }
}

void AprilTagDetectorNode::declareParameters() {
    this->declare_parameter("camera_frame", "camera_frame");
    this->declare_parameter("tag_frame", "tag_36h11_id2");
    this->declare_parameter("target_tag_id", 2);
    this->declare_parameter("base_tag_id_0", 0);
    this->declare_parameter("base_tag_id_1", 1);
    this->declare_parameter("tag_size", 0.06);
    this->declare_parameter("use_stereo", false);
    this->declare_parameter("use_direct_camera", true);
    this->declare_parameter("camera_device_id", 0);
    this->declare_parameter("camera_fx", 500.0);
    this->declare_parameter("camera_fy", 500.0);
    this->declare_parameter("camera_cx", 320.0);
    this->declare_parameter("camera_cy", 240.0);
    this->declare_parameter("image_width", 640);
    this->declare_parameter("image_height", 480);
    this->declare_parameter("publish_tf", true);
    this->declare_parameter("publish_marker", true);
    this->declare_parameter("enable_compensation", true);
    this->declare_parameter("print_precise_pose", true);
    this->declare_parameter("calib_file", "");
    this->declare_parameter("enable_relative_pose", true);
    this->declare_parameter("prefer_calibrated", true);
    this->declare_parameter("filter_window", 30);
    this->declare_parameter("filter_alpha", 0.15);
    this->declare_parameter("enable_undistort", true);
    this->declare_parameter("enable_object_detection", false);
    this->declare_parameter("enable_color_filter", false);
    this->declare_parameter("object_min_area", 500.0);
    this->declare_parameter("object_max_area", 100000.0);
    this->declare_parameter("enable_display", false);
    
    this->declare_parameter("image_topic", "image_rect");
    this->declare_parameter("camera_info_topic", "camera_info");
    this->declare_parameter("left_image_topic", "zed/left/image_rect_color");
    this->declare_parameter("right_image_topic", "zed/right/image_rect_color");
    this->declare_parameter("left_camera_info_topic", "zed/left/camera_info");
    
    this->declare_parameter("use_zed_sdk", false);
    this->declare_parameter("zed_camera_model", "zed2");
    this->declare_parameter("zed_camera_serial", 0);
}

void AprilTagDetectorNode::loadBasicParameters() {
    camera_frame_ = this->get_parameter("camera_frame").as_string();
    RCLCPP_INFO(this->get_logger(), "camera_frame: %s", camera_frame_.c_str());
    
    tag_frame_ = this->get_parameter("tag_frame").as_string();
    
    target_tag_id_ = this->get_parameter("target_tag_id").as_int();
    RCLCPP_INFO(this->get_logger(), "target_tag_id: %d", target_tag_id_);

    base_tag_id_0_ = this->get_parameter("base_tag_id_0").as_int();
    base_tag_id_1_ = this->get_parameter("base_tag_id_1").as_int();
    RCLCPP_INFO(this->get_logger(), "base_tag_ids: ID0=%d, ID1=%d", base_tag_id_0_, base_tag_id_1_);

    tag_size_ = this->get_parameter("tag_size").as_double();
    RCLCPP_INFO(this->get_logger(), "tag_size: %.3f", tag_size_);
    
    use_stereo_ = this->get_parameter("use_stereo").as_bool();
    RCLCPP_INFO(this->get_logger(), "use_stereo: %s", use_stereo_ ? "true" : "false");
    
    use_direct_camera_ = this->get_parameter("use_direct_camera").as_bool();
    RCLCPP_INFO(this->get_logger(), "use_direct_camera: %s", use_direct_camera_ ? "true" : "false");
    
    camera_device_id_ = this->get_parameter("camera_device_id").as_int();
    RCLCPP_INFO(this->get_logger(), "camera_device_id: %d", camera_device_id_);
    
    image_topic_ = this->get_parameter("image_topic").as_string();
    camera_info_topic_ = this->get_parameter("camera_info_topic").as_string();
    left_image_topic_ = this->get_parameter("left_image_topic").as_string();
    right_image_topic_ = this->get_parameter("right_image_topic").as_string();
    left_camera_info_topic_ = this->get_parameter("left_camera_info_topic").as_string();
    
    RCLCPP_INFO(this->get_logger(), "Image topic: %s", image_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "Camera info topic: %s", camera_info_topic_.c_str());
    if (use_stereo_) {
        RCLCPP_INFO(this->get_logger(), "Left image topic: %s", left_image_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "Right image topic: %s", right_image_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "Left camera info topic: %s", left_camera_info_topic_.c_str());
    }
    
    use_zed_sdk_ = this->get_parameter("use_zed_sdk").as_bool();
    zed_camera_model_ = this->get_parameter("zed_camera_model").as_string();
    zed_camera_serial_ = this->get_parameter("zed_camera_serial").as_int();
    
    RCLCPP_INFO(this->get_logger(), "Use ZED SDK: %s", use_zed_sdk_ ? "true" : "false");
    if (use_zed_sdk_) {
        RCLCPP_INFO(this->get_logger(), "ZED camera model: %s", zed_camera_model_.c_str());
        RCLCPP_INFO(this->get_logger(), "ZED camera serial: %d", zed_camera_serial_);
    }
}

void AprilTagDetectorNode::loadAdvancedParameters() {
    publish_tf_ = this->get_parameter("publish_tf").as_bool();
    publish_marker_ = this->get_parameter("publish_marker").as_bool();
    enable_compensation_ = this->get_parameter("enable_compensation").as_bool();
    print_precise_pose_ = this->get_parameter("print_precise_pose").as_bool();
    enable_relative_pose_ = this->get_parameter("enable_relative_pose").as_bool();
    prefer_calibrated_ = this->get_parameter("prefer_calibrated").as_bool();
    filter_window_ = this->get_parameter("filter_window").as_int();
    filter_alpha_ = this->get_parameter("filter_alpha").as_double();
    rel_filter_ = RelativePoseFilter(filter_window_, filter_alpha_);
    RCLCPP_INFO(this->get_logger(), "Filter: window=%d, alpha=%.2f", filter_window_, filter_alpha_);
    enable_undistort_ = this->get_parameter("enable_undistort").as_bool();
    RCLCPP_INFO(this->get_logger(), "Undistortion: %s", enable_undistort_ ? "ENABLED" : "disabled");
    
    enable_object_detection_ = this->get_parameter("enable_object_detection").as_bool();
    enable_color_filter_ = this->get_parameter("enable_color_filter").as_bool();
    object_min_area_ = this->get_parameter("object_min_area").as_double();
    object_max_area_ = this->get_parameter("object_max_area").as_double();
    
    RCLCPP_INFO(this->get_logger(), "Object Detection: %s", enable_object_detection_ ? "ENABLED" : "disabled");
    RCLCPP_INFO(this->get_logger(), "  Color Filter: %s", enable_color_filter_ ? "ENABLED" : "disabled");
    RCLCPP_INFO(this->get_logger(), "  Min Area: %.0f px", object_min_area_);
    RCLCPP_INFO(this->get_logger(), "  Max Area: %.0f px", object_max_area_);
    
    enable_display_ = this->get_parameter("enable_display").as_bool();
    RCLCPP_INFO(this->get_logger(), "Display Window: %s", enable_display_ ? "ENABLED" : "disabled");
    
    if (object_detector_) {
        object_detector_->enableColorFilter(enable_color_filter_);
        object_detector_->setMinArea(object_min_area_);
        object_detector_->setMaxArea(object_max_area_);
        object_detector_->enableTracking(true);
        object_detector_->setTrackingDistanceThreshold(50.0);
        object_detector_->enableCentroidFilter(true);
        object_detector_->setFilterAlpha(0.2);
        object_detector_->setMorphologyKernelSize(3);
    }
    
    if (use_direct_camera_) {
        CameraParameters params;
        double fx = this->get_parameter("camera_fx").as_double();
        double fy = this->get_parameter("camera_fy").as_double();
        double cx = this->get_parameter("camera_cx").as_double();
        double cy = this->get_parameter("camera_cy").as_double();
        int width = this->get_parameter("image_width").as_int();
        int height = this->get_parameter("image_height").as_int();
        
        params.camera_matrix = cv::Matx33d(
            fx, 0, cx,
            0, fy, cy,
            0, 0, 1
        );
        params.dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
        params.image_width = width;
        params.image_height = height;
        
        if (pose_estimator_) {
            pose_estimator_->setCameraParameters(params);
        }
        if (error_compensation_) {
            error_compensation_->setImageDimensions(width, height);
        }
        camera_params_set_ = true;
        
        RCLCPP_INFO(this->get_logger(), 
            "Direct camera params set: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f, %dx%d",
            fx, fy, cx, cy, width, height);
    }
}

void AprilTagDetectorNode::cameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
    
    if (!camera_params_set_) {
        CameraParameters params;
        params.camera_matrix = cv::Matx33d(
            msg->k[0], msg->k[1], msg->k[2],
            msg->k[3], msg->k[4], msg->k[5],
            msg->k[6], msg->k[7], msg->k[8]
        );

        params.dist_coeffs = cv::Mat(msg->d.size(), 1, CV_64F);
        for (size_t i = 0; i < msg->d.size(); ++i) {
            params.dist_coeffs.at<double>(i) = msg->d[i];
        }

        params.image_width = msg->width;
        params.image_height = msg->height;

        // Save original (for reference)
        factory_params_original_ = params;

        // Compute undistortion map and create zero-distortion copy
        cv::Mat K(params.camera_matrix);
        cv::initUndistortRectifyMap(
            K, params.dist_coeffs, cv::Mat(), K,
            cv::Size(msg->width, msg->height),
            CV_16SC2, undist_map_x_, undist_map_y_);
        undist_map_ready_ = true;

        // Set factory estimator to zero distortion (image will be remapped)
        CameraParameters params_undist = params;
        params_undist.dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
        pose_estimator_->setCameraParameters(params_undist);
        error_compensation_->setImageDimensions(msg->width, msg->height);

        camera_params_set_ = true;

        RCLCPP_INFO(this->get_logger(),
            "Camera params set: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f | distortion=%zu coeffs",
            params.fx(), params.fy(), params.cx(), params.cy(), msg->d.size());
        RCLCPP_INFO(this->get_logger(),
            "Undistortion map computed (%dx%d) — detection on undistorted image",
            msg->width, msg->height);
    }
    
    camera_frame_ = msg->header.frame_id;
    camera_info_received_ = true;
}

void AprilTagDetectorNode::imageCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr msg) {
    
    if (!camera_params_set_) {
        RCLCPP_WARN_ONCE(this->get_logger(), 
            "Waiting for camera info...");
        return;
    }
    
    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(
            msg, sensor_msgs::image_encodings::BGR8);

        cv::Mat image = cv_ptr->image;

        // === Undistort image: eliminates edge/skew distortion residuals ===
        if (enable_undistort_ && undist_map_ready_) {
            cv::remap(image, image, undist_map_x_, undist_map_y_, cv::INTER_LINEAR);
        }

        auto detections = detector_->detect(image);

        // Collect poses for all relevant tags
        TagPose pose_id0, pose_id1, pose_id2;
        TagPose pose_id0_calib, pose_id2_calib;
        bool id0_found = false, id1_found = false, id2_found = false;
        bool id0_calib_valid = false, id2_calib_valid = false;

        for (const auto& det : detections) {
            double conf;
            TagPose pose_f = estimateSingleTag(det.corners, *pose_estimator_, conf);

            if (det.id == base_tag_id_0_) {
                id0_found = pose_f.valid;
                pose_id0 = pose_f;
                // Calibrated
                if (calib_params_loaded_ && pose_estimator_calib_) {
                    double c2;
                    pose_id0_calib = estimateSingleTag(det.corners, *pose_estimator_calib_, c2);
                    id0_calib_valid = pose_id0_calib.valid;
                }
            } else if (det.id == base_tag_id_1_) {
                id1_found = pose_f.valid;
                pose_id1 = pose_f;
            } else if (det.id == target_tag_id_) {
                id2_found = pose_f.valid;
                pose_id2 = pose_f;
                // Calibrated
                if (calib_params_loaded_ && pose_estimator_calib_) {
                    double c2;
                    pose_id2_calib = estimateSingleTag(det.corners, *pose_estimator_calib_, c2);
                    id2_calib_valid = pose_id2_calib.valid;
                }
                // Publish target tag pose
                if (pose_f.valid) {
                    publishDetection(det, pose_f, msg->header);
                    if (publish_marker_) publishMarker(pose_f, msg->header);
                    if (publish_tf_) publishTransform(pose_f, msg->header);
                }
            }
        }

        // === Relative pose: ID2 relative to ID0 ===
        if (enable_relative_pose_ && id0_found && id2_found) {
            // Choose which estimator to use based on prefer_calibrated_
            TagPose& p0 = (prefer_calibrated_ && id0_calib_valid) ? pose_id0_calib : pose_id0;
            TagPose& p2 = (prefer_calibrated_ && id2_calib_valid) ? pose_id2_calib : pose_id2;

            cv::Vec3d rel_t, rel_r;
            double distance;
            computeRelativePose(p0, p2, rel_t, rel_r, distance);

            // Apply temporal filter
            rel_filter_.add(rel_t[0], rel_t[1], rel_t[2],
                           rel_r[0], rel_r[1], rel_r[2]);
            double fx, fy, fz, frx, fry, frz;
            rel_filter_.getFiltered(fx, fy, fz, frx, fry, frz);

            // Build filtered vectors for publishing
            cv::Vec3d rel_t_filt(fx, fy, fz);
            cv::Vec3d rel_r_filt(frx, fry, frz);
            double dist_filt = std::sqrt(fx*fx + fy*fy + fz*fz);

            publishRelativePose(rel_t_filt, rel_r_filt, dist_filt,
                               id0_found, id1_found, id2_found, msg->header);

            if (print_precise_pose_) {
                // Use original (unfiltered) values for print to see raw accuracy
                printRelativePose(pose_id0, pose_id1, pose_id2,
                                 id0_found, id1_found, id2_found);
                // Show filtered value + fluctuation
                std::stringstream ss;
                ss << std::fixed << std::setprecision(3);
                ss << "  Filtered (n=" << (rel_filter_.isReady() ? "✓" : "⏳") << "): "
                   << "X=" << std::setw(8) << fx*1000.0 << " mm"
                   << " Y=" << std::setw(8) << fy*1000.0 << " mm"
                   << " Z=" << std::setw(8) << fz*1000.0 << " mm"
                   << " dist=" << std::setw(8) << dist_filt*1000.0 << " mm";
                RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());

                // Show per-axis fluctuation (standard deviation)
                double sx, sy, sz, srx, sry, srz;
                rel_filter_.getFluctuation(sx, sy, sz, srx, sry, srz);
                std::stringstream sf;
                sf << std::fixed << std::setprecision(4);
                sf << "  Noise σ:  pos(XYZ)=" << std::setw(7) << sx*1000.0 << " "
                   << std::setw(7) << sy*1000.0 << " "
                   << std::setw(7) << sz*1000.0 << " mm  |  "
                   << "rot(RxRyRz)=" << std::setw(6) << srx*180.0/M_PI << "° "
                   << std::setw(6) << sry*180.0/M_PI << "° "
                   << std::setw(6) << srz*180.0/M_PI << "°";
                RCLCPP_INFO(this->get_logger(), "%s", sf.str().c_str());
                RCLCPP_INFO(this->get_logger(), "  Using: %s",
                            (prefer_calibrated_ && id0_calib_valid) ? "CALIBRATED" : "ZED Factory");
            }
        } else if (print_precise_pose_ && !(id0_found && id2_found)) {
            // Print individual tag status
            std::stringstream ss;
            ss << "Tags: ID0=" << (id0_found?"OK":"MISS")
               << " ID1=" << (id1_found?"OK":"MISS")
               << " ID2=" << (id2_found?"OK":"MISS");
            RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());
        }

        // Object detection
        std::vector<ObjectContour> object_contours;
        if (enable_object_detection_ && object_detector_) {
            object_contours = object_detector_->detect(image);
            static int frame_count = 0;
            frame_count++;
            if (frame_count % 30 == 0) {
                std::stringstream od_ss;
                od_ss << "Object Detection: " << object_contours.size() << " objects found";
                for (size_t i = 0; i < std::min(object_contours.size(), (size_t)5); ++i) {
                    od_ss << " | " << object_contours[i].shape 
                          << "(" << static_cast<int>(object_contours[i].area) << "px)";
                }
                RCLCPP_INFO(this->get_logger(), "%s", od_ss.str().c_str());
            }
        }

        // Debug image
        if (debug_image_pub_->get_subscription_count() > 0) {
            for (const auto& det : detections) {
                cv::Scalar color = cv::Scalar(0, 255, 0); // green default
                if (det.id == base_tag_id_0_) color = cv::Scalar(255, 0, 0);    // blue
                else if (det.id == base_tag_id_1_) color = cv::Scalar(255, 0, 255); // purple
                else if (det.id == target_tag_id_) color = cv::Scalar(0, 0, 255);   // red

                cv::polylines(image, std::vector<std::vector<cv::Point>>{
                    std::vector<cv::Point>(det.corners.begin(), det.corners.end())},
                    true, color, 2);
                for (const auto& corner : det.corners) {
                    cv::circle(image, corner, 3, color, -1);
                }
                std::string label;
                if (det.id == base_tag_id_0_) label = "ID0 BASE";
                else if (det.id == base_tag_id_1_) label = "ID1 AUX";
                else if (det.id == target_tag_id_) label = "ID2 TGT";
                else label = "ID" + std::to_string(det.id);
                cv::putText(image, label, det.center,
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
            }
            
            if (enable_object_detection_ && !object_contours.empty()) {
                object_detector_->drawContours(image, object_contours);
            }
            
            sensor_msgs::msg::Image::SharedPtr debug_msg =
                cv_bridge::CvImage(msg->header, "bgr8", image).toImageMsg();
            debug_image_pub_->publish(*debug_msg);
            
            if (enable_display_) {
                cv::namedWindow("AprilTag & Object Detection", cv::WINDOW_NORMAL);
                cv::imshow("AprilTag & Object Detection", image);
                if (cv::waitKey(1) == 27) {
                    running_ = false;
                    RCLCPP_INFO(this->get_logger(), "ESC pressed, stopping...");
                }
            }
        }

    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
}

void AprilTagDetectorNode::leftImageCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr msg) {
    std::lock_guard<std::mutex> lock(image_mutex_);
    last_left_image_ = msg;
    processStereoImages();
}

void AprilTagDetectorNode::rightImageCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr msg) {
    std::lock_guard<std::mutex> lock(image_mutex_);
    last_right_image_ = msg;
}

void AprilTagDetectorNode::processStereoImages() {
    if (!last_left_image_) return;
    
    imageCallback(last_left_image_);
}

void AprilTagDetectorNode::printPosePrecise(const TagPose& pose, double confidence) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(3);
    ss << "\n========================================" << std::endl;
    ss << "AprilTag ID " << target_tag_id_ << " Detection Results:" << std::endl;
    ss << "========================================" << std::endl;
    ss << "Position (meters):" << std::endl;
    ss << "  X: " << pose.translation[0] * 1000.0 << " mm" << std::endl;
    ss << "  Y: " << pose.translation[1] * 1000.0 << " mm" << std::endl;
    ss << "  Z: " << pose.translation[2] * 1000.0 << " mm" << std::endl;
    ss << "Rotation (radians):" << std::endl;
    ss << "  Rx: " << pose.rotation[0] << std::endl;
    ss << "  Ry: " << pose.rotation[1] << std::endl;
    ss << "  Rz: " << pose.rotation[2] << std::endl;
    ss << "Reprojection Error: " << std::setprecision(6) 
       << pose.reprojection_error << " pixels" << std::endl;
    ss << "Confidence: " << std::setprecision(3) 
       << confidence * 100.0 << "%" << std::endl;
    ss << "========================================" << std::endl;
    
    RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());
}

void AprilTagDetectorNode::publishDetection(
    const AprilTagDetection& detection,
    const TagPose& pose,
    const std_msgs::msg::Header& header) {
    
    auto msg = std::make_unique<geometry_msgs::msg::PoseStamped>();
    msg->header = header;
    msg->header.frame_id = camera_frame_;
    
    msg->pose.position.x = pose.translation[0];
    msg->pose.position.y = pose.translation[1];
    msg->pose.position.z = pose.translation[2];
    
    double angle = cv::norm(pose.rotation);
    if (angle > 1e-6) {
        cv::Vec3d axis = pose.rotation / angle;
        tf2::Quaternion q(
            axis[0] * sin(angle / 2),
            axis[1] * sin(angle / 2),
            axis[2] * sin(angle / 2),
            cos(angle / 2)
        );
        msg->pose.orientation.x = q.x();
        msg->pose.orientation.y = q.y();
        msg->pose.orientation.z = q.z();
        msg->pose.orientation.w = q.w();
    } else {
        msg->pose.orientation.w = 1.0;
    }
    
    pose_pub_->publish(std::move(msg));
}

void AprilTagDetectorNode::publishMarker(
    const TagPose& pose,
    const std_msgs::msg::Header& header) {
    
    auto marker = std::make_unique<visualization_msgs::msg::Marker>();
    marker->header = header;
    marker->header.frame_id = camera_frame_;
    marker->ns = "apriltag";
    marker->id = target_tag_id_;
    marker->type = visualization_msgs::msg::Marker::CUBE;
    marker->action = visualization_msgs::msg::Marker::ADD;
    
    marker->pose.position.x = pose.translation[0];
    marker->pose.position.y = pose.translation[1];
    marker->pose.position.z = pose.translation[2];
    
    double angle = cv::norm(pose.rotation);
    if (angle > 1e-6) {
        cv::Vec3d axis = pose.rotation / angle;
        tf2::Quaternion q(
            axis[0] * sin(angle / 2),
            axis[1] * sin(angle / 2),
            axis[2] * sin(angle / 2),
            cos(angle / 2)
        );
        marker->pose.orientation.x = q.x();
        marker->pose.orientation.y = q.y();
        marker->pose.orientation.z = q.z();
        marker->pose.orientation.w = q.w();
    } else {
        marker->pose.orientation.w = 1.0;
    }
    
    marker->scale.x = tag_size_;
    marker->scale.y = tag_size_;
    marker->scale.z = 0.01;
    
    marker->color.r = 0.0;
    marker->color.g = 1.0;
    marker->color.b = 0.0;
    marker->color.a = 0.8;
    
    marker_pub_->publish(std::move(marker));
}

void AprilTagDetectorNode::publishTransform(
    const TagPose& pose,
    const std_msgs::msg::Header& header) {
    
    geometry_msgs::msg::TransformStamped transform;
    
    transform.header = header;
    transform.header.frame_id = camera_frame_;
    transform.child_frame_id = tag_frame_;
    
    transform.transform.translation.x = pose.translation[0];
    transform.transform.translation.y = pose.translation[1];
    transform.transform.translation.z = pose.translation[2];
    
    double angle = cv::norm(pose.rotation);
    if (angle > 1e-6) {
        cv::Vec3d axis = pose.rotation / angle;
        tf2::Quaternion q(
            axis[0] * sin(angle / 2),
            axis[1] * sin(angle / 2),
            axis[2] * sin(angle / 2),
            cos(angle / 2)
        );
        transform.transform.rotation.x = q.x();
        transform.transform.rotation.y = q.y();
        transform.transform.rotation.z = q.z();
        transform.transform.rotation.w = q.w();
    } else {
        transform.transform.rotation.w = 1.0;
    }
    
    tf_broadcaster_->sendTransform(transform);
}

void AprilTagDetectorNode::cameraCaptureLoop() {
    RCLCPP_INFO(this->get_logger(), "Attempting to open camera device %d...", camera_device_id_);
    
    cv::VideoCapture cap;
    
    try {
        if (!cap.open(camera_device_id_, cv::CAP_V4L2)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open camera device %d with V4L2", camera_device_id_);
            if (!cap.open(camera_device_id_)) {
                RCLCPP_ERROR(this->get_logger(), "Failed to open camera device %d", camera_device_id_);
                RCLCPP_ERROR(this->get_logger(), "Please check if the camera is connected");
                RCLCPP_ERROR(this->get_logger(), "Check permissions: ls -la /dev/video*");
                RCLCPP_ERROR(this->get_logger(), "Try: sudo chmod 666 /dev/video%d", camera_device_id_);
                return;
            }
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception opening camera: %s", e.what());
        return;
    }
    
    RCLCPP_INFO(this->get_logger(), "Camera opened, setting parameters...");
    
    int width = this->get_parameter("image_width").as_int();
    int height = this->get_parameter("image_height").as_int();
    
    cap.set(cv::CAP_PROP_FRAME_WIDTH, static_cast<double>(width));
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(height));
    cap.set(cv::CAP_PROP_FPS, 30.0);
    
    int actual_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int actual_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double actual_fps = cap.get(cv::CAP_PROP_FPS);
    
    RCLCPP_INFO(this->get_logger(), "Camera initialized: %dx%d @ %.1ffps (requested: %dx%d @ 30fps)", 
                actual_width, actual_height, actual_fps, width, height);
    
    cv::Mat frame;
    auto last_time = std::chrono::steady_clock::now();
    
    while (running_ && rclcpp::ok()) {
        cap >> frame;
        if (frame.empty()) {
            RCLCPP_WARN(this->get_logger(), "Empty frame received");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - last_time).count();
        
        if (elapsed < 33) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        last_time = current_time;
        
        auto detections = detector_->detect(frame);

        // Collect poses for all relevant tags
        TagPose pose_id0, pose_id1, pose_id2;
        bool id0_found = false, id1_found = false, id2_found = false;

        for (const auto& det : detections) {
            double conf;
            TagPose pose_f = estimateSingleTag(det.corners, *pose_estimator_, conf);

            if (det.id == base_tag_id_0_) {
                id0_found = pose_f.valid; pose_id0 = pose_f;
            } else if (det.id == base_tag_id_1_) {
                id1_found = pose_f.valid; pose_id1 = pose_f;
            } else if (det.id == target_tag_id_) {
                id2_found = pose_f.valid; pose_id2 = pose_f;
                if (pose_f.valid) {
                    auto hdr = std_msgs::msg::Header();
                    hdr.stamp = this->now(); hdr.frame_id = camera_frame_;
                    publishDetection(det, pose_f, hdr);
                    if (publish_marker_) publishMarker(pose_f, hdr);
                    if (publish_tf_) publishTransform(pose_f, hdr);
                }
            }
        }

        // Relative pose (with filtering)
        if (enable_relative_pose_ && id0_found && id2_found) {
            cv::Vec3d rel_t, rel_r; double dist;
            computeRelativePose(pose_id0, pose_id2, rel_t, rel_r, dist);
            rel_filter_.add(rel_t[0], rel_t[1], rel_t[2], rel_r[0], rel_r[1], rel_r[2]);
            double fx, fy, fz, frx, fry, frz;
            rel_filter_.getFiltered(fx, fy, fz, frx, fry, frz);
            cv::Vec3d rel_t_f(fx, fy, fz), rel_r_f(frx, fry, frz);
            double dist_f = std::sqrt(fx*fx + fy*fy + fz*fz);
            auto hdr = std_msgs::msg::Header();
            hdr.stamp = this->now(); hdr.frame_id = camera_frame_;
            publishRelativePose(rel_t_f, rel_r_f, dist_f, id0_found, id1_found, id2_found, hdr);
            if (print_precise_pose_) {
                printRelativePose(pose_id0, pose_id1, pose_id2, id0_found, id1_found, id2_found);
            }
        }
        
        std::vector<ObjectContour> object_contours;
        if (enable_object_detection_ && object_detector_) {
            object_contours = object_detector_->detect(frame);
            static int frame_count_obj = 0;
            frame_count_obj++;
            if (frame_count_obj % 30 == 0) {
                std::stringstream od_ss;
                od_ss << "Object Detection: " << object_contours.size() << " objects found";
                for (size_t i = 0; i < std::min(object_contours.size(), (size_t)5); ++i) {
                    od_ss << " | " << object_contours[i].shape 
                          << "(" << static_cast<int>(object_contours[i].area) << "px)";
                }
                RCLCPP_INFO(this->get_logger(), "%s", od_ss.str().c_str());
            }
        }
        
        bool need_debug_frame = debug_image_pub_->get_subscription_count() > 0 || enable_display_;
        cv::Mat debug_frame;
        
        if (need_debug_frame) {
            debug_frame = frame.clone();
            for (const auto& det : detections) {
                std::vector<cv::Point> corners_int;
                for (const auto& c : det.corners) {
                    corners_int.emplace_back(cv::Point(static_cast<int>(c.x), static_cast<int>(c.y)));
                }
                cv::polylines(debug_frame, std::vector<std::vector<cv::Point>>{corners_int},
                             true, cv::Scalar(0, 255, 0), 2);
                
                for (const auto& corner : det.corners) {
                    cv::circle(debug_frame, corner, 3, cv::Scalar(0, 0, 255), -1);
                }
                
                std::stringstream ss;
                ss << "ID: " << det.id;
                cv::putText(debug_frame, ss.str(), det.center, 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
            }
            
            if (enable_object_detection_ && !object_contours.empty()) {
                object_detector_->drawContours(debug_frame, object_contours);
            }
        }
        
        if (debug_image_pub_->get_subscription_count() > 0) {
            auto header = std_msgs::msg::Header();
            header.stamp = this->now();
            header.frame_id = camera_frame_;
            
            sensor_msgs::msg::Image::SharedPtr debug_msg = 
                cv_bridge::CvImage(header, "bgr8", debug_frame).toImageMsg();
            debug_image_pub_->publish(*debug_msg);
        }
        
        if (enable_display_) {
            cv::namedWindow("AprilTag & Object Detection", cv::WINDOW_NORMAL);
            cv::imshow("AprilTag & Object Detection", debug_frame);
            if (cv::waitKey(1) == 27) {
                running_ = false;
                RCLCPP_INFO(this->get_logger(), "ESC pressed, stopping...");
            }
        }
    }
    
    cap.release();
    if (enable_display_) {
        cv::destroyAllWindows();
    }
    RCLCPP_INFO(this->get_logger(), "Camera capture loop stopped");
}

#ifdef HAVE_ZED_SDK
void AprilTagDetectorNode::zedCaptureLoop() {
    RCLCPP_INFO(this->get_logger(), "ZED camera capture loop started");
    
    if (!zed_camera_ || !zed_camera_->isOpened()) {
        RCLCPP_ERROR(this->get_logger(), "ZED camera is not opened");
        return;
    }
    
    sl::Mat zed_image;
    auto last_time = std::chrono::steady_clock::now();
    
    while (running_ && rclcpp::ok()) {
        sl::ERROR_CODE err = zed_camera_->grab();
        if (err != sl::ERROR_CODE::SUCCESS) {
            RCLCPP_WARN(this->get_logger(), "ZED grab error: %s", sl::toString(err).c_str());
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        zed_camera_->retrieveImage(zed_image, sl::VIEW::LEFT);
        
        cv::Mat frame(zed_image.getHeight(), zed_image.getWidth(), CV_8UC4, zed_image.getPtr<sl::uchar1>(sl::MEM::CPU));
        cv::cvtColor(frame, frame, cv::COLOR_RGBA2BGR);
        
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - last_time).count();
        
        if (elapsed < 33) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        last_time = current_time;
        
        auto detections = detector_->detect(frame);
        
        TagPose pose_id0, pose_id1, pose_id2;
        bool id0_found = false, id1_found = false, id2_found = false;
        
        for (const auto& det : detections) {
            double conf;
            TagPose pose_f = estimateSingleTag(det.corners, *pose_estimator_, conf);
            
            if (det.id == base_tag_id_0_) {
                id0_found = pose_f.valid; pose_id0 = pose_f;
            } else if (det.id == base_tag_id_1_) {
                id1_found = pose_f.valid; pose_id1 = pose_f;
            } else if (det.id == target_tag_id_) {
                id2_found = pose_f.valid; pose_id2 = pose_f;
                if (pose_f.valid) {
                    auto hdr = std_msgs::msg::Header();
                    hdr.stamp = this->now(); hdr.frame_id = camera_frame_;
                    publishDetection(det, pose_f, hdr);
                    if (publish_marker_) publishMarker(pose_f, hdr);
                    if (publish_tf_) publishTransform(pose_f, hdr);
                }
            }
        }
        
        if (enable_relative_pose_ && id0_found && id2_found) {
            cv::Vec3d rel_t, rel_r; double dist;
            computeRelativePose(pose_id0, pose_id2, rel_t, rel_r, dist);
            rel_filter_.add(rel_t[0], rel_t[1], rel_t[2], rel_r[0], rel_r[1], rel_r[2]);
            double fx, fy, fz, frx, fry, frz;
            rel_filter_.getFiltered(fx, fy, fz, frx, fry, frz);
            cv::Vec3d rel_t_f(fx, fy, fz), rel_r_f(frx, fry, frz);
            double dist_f = std::sqrt(fx*fx + fy*fy + fz*fz);
            auto hdr = std_msgs::msg::Header();
            hdr.stamp = this->now(); hdr.frame_id = camera_frame_;
            publishRelativePose(rel_t_f, rel_r_f, dist_f, id0_found, id1_found, id2_found, hdr);
            if (print_precise_pose_) {
                printRelativePose(pose_id0, pose_id1, pose_id2, id0_found, id1_found, id2_found);
            }
        }
        
        std::vector<ObjectContour> object_contours;
        if (enable_object_detection_ && object_detector_) {
            object_contours = object_detector_->detect(frame);
            static int frame_count_obj = 0;
            frame_count_obj++;
            if (frame_count_obj % 30 == 0) {
                std::stringstream od_ss;
                od_ss << "Object Detection: " << object_contours.size() << " objects found";
                for (size_t i = 0; i < std::min(object_contours.size(), (size_t)5); ++i) {
                    od_ss << " | " << object_contours[i].shape 
                          << "(" << static_cast<int>(object_contours[i].area) << "px)";
                }
                RCLCPP_INFO(this->get_logger(), "%s", od_ss.str().c_str());
            }
        }
        
        bool need_debug_frame = debug_image_pub_->get_subscription_count() > 0 || enable_display_;
        cv::Mat debug_frame;
        
        if (need_debug_frame) {
            debug_frame = frame.clone();
            
            for (const auto& det : detections) {
                std::vector<cv::Point> corners_int;
                for (const auto& c : det.corners) {
                    corners_int.push_back(cv::Point(cvRound(c.x), cvRound(c.y)));
                }
                cv::polylines(debug_frame, std::vector<std::vector<cv::Point>>{corners_int},
                             true, cv::Scalar(0, 255, 0), 2);
                
                for (const auto& corner : det.corners) {
                    cv::circle(debug_frame, corner, 3, cv::Scalar(0, 0, 255), -1);
                }
                
                std::stringstream ss;
                ss << "ID: " << det.id;
                cv::putText(debug_frame, ss.str(), det.center, 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
            }
            
            if (enable_object_detection_ && !object_contours.empty()) {
                object_detector_->drawContours(debug_frame, object_contours);
            }
        }
        
        if (debug_image_pub_->get_subscription_count() > 0) {
            auto header = std_msgs::msg::Header();
            header.stamp = this->now();
            header.frame_id = camera_frame_;
            
            sensor_msgs::msg::Image::SharedPtr debug_msg = 
                cv_bridge::CvImage(header, "bgr8", debug_frame).toImageMsg();
            debug_image_pub_->publish(*debug_msg);
        }
        
        if (enable_display_) {
            cv::namedWindow("AprilTag & Object Detection", cv::WINDOW_NORMAL);
            cv::imshow("AprilTag & Object Detection", debug_frame);
            if (cv::waitKey(1) == 27) {
                running_ = false;
                RCLCPP_INFO(this->get_logger(), "ESC pressed, stopping...");
            }
        }
    }
    
    if (zed_camera_) {
        zed_camera_->close();
    }
    if (enable_display_) {
        cv::destroyAllWindows();
    }
    RCLCPP_INFO(this->get_logger(), "ZED camera capture loop stopped");
}
#endif

bool AprilTagDetectorNode::loadCalibrationYAML(const std::string& filepath, CameraParameters& params) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        RCLCPP_ERROR(this->get_logger(), "Cannot open calibration file: %s", filepath.c_str());
        return false;
    }

    std::string line;
    int img_w = 0, img_h = 0;
    std::vector<double> K(9), D;

    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;

        // Parse image_width / image_height
        if (line.find("image_width:") != std::string::npos) {
            img_w = std::stoi(line.substr(line.find(':') + 1));
        } else if (line.find("image_height:") != std::string::npos) {
            img_h = std::stoi(line.substr(line.find(':') + 1));
        }
        // Parse camera_matrix data array
        else if (line.find("camera_matrix:") != std::string::npos) {
            // Read next line: "  rows: 3"
            std::getline(file, line);
            // Read next line: "  cols: 3"
            std::getline(file, line);
            // Read next line: "  data: [...]"
            std::getline(file, line);
            size_t start = line.find('[');
            size_t end = line.find(']');
            if (start != std::string::npos && end != std::string::npos) {
                std::string data = line.substr(start + 1, end - start - 1);
                std::stringstream ss(data);
                std::string val;
                int idx = 0;
                while (std::getline(ss, val, ',') && idx < 9) {
                    // Trim whitespace
                    val.erase(0, val.find_first_not_of(" \t"));
                    K[idx++] = std::stod(val);
                }
            }
        }
        // Parse distortion_coefficients data array
        else if (line.find("distortion_coefficients:") != std::string::npos) {
            std::getline(file, line); // rows
            std::getline(file, line); // cols
            std::getline(file, line); // data: [...]
            size_t start = line.find('[');
            size_t end = line.find(']');
            if (start != std::string::npos && end != std::string::npos) {
                std::string data = line.substr(start + 1, end - start - 1);
                std::stringstream ss(data);
                std::string val;
                while (std::getline(ss, val, ',')) {
                    val.erase(0, val.find_first_not_of(" \t"));
                    if (!val.empty()) D.push_back(std::stod(val));
                }
            }
        }
    }
    file.close();

    if (img_w == 0 || img_h == 0 || K[0] == 0.0) {
        RCLCPP_ERROR(this->get_logger(), "Invalid calibration data in file: %s", filepath.c_str());
        return false;
    }

    params.camera_matrix = cv::Matx33d(
        K[0], K[1], K[2],
        K[3], K[4], K[5],
        K[6], K[7], K[8]
    );
    params.image_width = img_w;
    params.image_height = img_h;

    params.dist_coeffs = cv::Mat(static_cast<int>(D.size()), 1, CV_64F);
    for (size_t i = 0; i < D.size(); ++i) {
        params.dist_coeffs.at<double>(static_cast<int>(i)) = D[i];
    }

    RCLCPP_INFO(this->get_logger(), "Loaded %zu distortion coefficients", D.size());
    return true;
}

void AprilTagDetectorNode::printPoseComparison(
    const TagPose& pose_factory, const TagPose& pose_calib,
    double conf_factory, double conf_calib) {

    // Compute delta between two estimations
    double dx = (pose_factory.translation[0] - pose_calib.translation[0]) * 1000.0;
    double dy = (pose_factory.translation[1] - pose_calib.translation[1]) * 1000.0;
    double dz = (pose_factory.translation[2] - pose_calib.translation[2]) * 1000.0;
    double dist_diff = std::sqrt(dx*dx + dy*dy + dz*dz);

    double drx = (pose_factory.rotation[0] - pose_calib.rotation[0]) * 180.0 / M_PI;
    double dry = (pose_factory.rotation[1] - pose_calib.rotation[1]) * 180.0 / M_PI;
    double drz = (pose_factory.rotation[2] - pose_calib.rotation[2]) * 180.0 / M_PI;

    std::stringstream ss;
    ss << std::fixed << std::setprecision(3);
    ss << "\n╔══════════════════════════════════════════════════════════════╗\n";
    ss << "║  AprilTag ID " << target_tag_id_
       << " — Intrinsics Comparison                      ║\n";
    ss << "╠══════════════════════════════════════════════════════════════╣\n";
    ss << "║              │  ZED Factory  │  Calibrated   │   Δ (abs)    ║\n";
    ss << "╠══════════════════════════════════════════════════════════════╣\n";
    ss << "║  X (mm)      │  " << std::setw(10) << pose_factory.translation[0] * 1000.0
       << "  │  " << std::setw(10) << pose_calib.translation[0] * 1000.0
       << "  │  " << std::setw(10) << std::abs(dx) << "  ║\n";
    ss << "║  Y (mm)      │  " << std::setw(10) << pose_factory.translation[1] * 1000.0
       << "  │  " << std::setw(10) << pose_calib.translation[1] * 1000.0
       << "  │  " << std::setw(10) << std::abs(dy) << "  ║\n";
    ss << "║  Z (mm)      │  " << std::setw(10) << pose_factory.translation[2] * 1000.0
       << "  │  " << std::setw(10) << pose_calib.translation[2] * 1000.0
       << "  │  " << std::setw(10) << std::abs(dz) << "  ║\n";
    ss << "║  3D Δ (mm)   │              │               │  "
       << std::setw(10) << dist_diff << "  ║\n";
    ss << "╠══════════════════════════════════════════════════════════════╣\n";
    ss << "║  Rx (deg)    │  " << std::setw(10) << pose_factory.rotation[0] * 180.0 / M_PI
       << "  │  " << std::setw(10) << pose_calib.rotation[0] * 180.0 / M_PI
       << "  │  " << std::setw(10) << std::abs(drx) << "  ║\n";
    ss << "║  Ry (deg)    │  " << std::setw(10) << pose_factory.rotation[1] * 180.0 / M_PI
       << "  │  " << std::setw(10) << pose_calib.rotation[1] * 180.0 / M_PI
       << "  │  " << std::setw(10) << std::abs(dry) << "  ║\n";
    ss << "║  Rz (deg)    │  " << std::setw(10) << pose_factory.rotation[2] * 180.0 / M_PI
       << "  │  " << std::setw(10) << pose_calib.rotation[2] * 180.0 / M_PI
       << "  │  " << std::setw(10) << std::abs(drz) << "  ║\n";
    ss << "╠══════════════════════════════════════════════════════════════╣\n";
    ss << "║  Reproj (px) │  " << std::setw(10) << std::setprecision(4) << pose_factory.reprojection_error
       << "  │  " << std::setw(10) << std::setprecision(4) << pose_calib.reprojection_error
       << "  │              ║\n";
    ss << std::setprecision(3);
    ss << "║  Confidence  │  " << std::setw(10) << conf_factory * 100.0 << "%"
       << "  │  " << std::setw(10) << conf_calib * 100.0 << "%"
       << "  │              ║\n";
    ss << "╚══════════════════════════════════════════════════════════════╝\n";

    RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());
}

void AprilTagDetectorNode::processDetection(
    const AprilTagDetection& det,
    const std_msgs::msg::Header& header) {
    (void)det;
    (void)header;
}

TagPose AprilTagDetectorNode::estimateSingleTag(
    const std::vector<cv::Point2f>& corners,
    PoseEstimator& estimator, double& confidence) {

    confidence = 1.0;
    TagPose pose = estimator.estimatePoseIterative(corners);

    if (pose.valid && enable_compensation_) {
        TagQualityMetrics metrics = error_compensation_->computeQualityMetrics(
            corners,
            cv::Mat(estimator.getCameraParameters().camera_matrix));
        pose.translation = error_compensation_->compensateAll(
            pose.translation, pose.rotation, corners,
            cv::Mat(estimator.getCameraParameters().camera_matrix));
        confidence = metrics.confidence;
    }
    return pose;
}

void AprilTagDetectorNode::computeRelativePose(
    const TagPose& pose_base, const TagPose& pose_target,
    cv::Vec3d& rel_t, cv::Vec3d& rel_r, double& distance) {

    // T_cam→base = [R_base | t_base], we want T_base→target
    // R_rel = R_base^T * R_target
    // t_rel = R_base^T * (t_target - t_base)

    cv::Mat R_base, R_target;
    cv::Rodrigues(pose_base.rotation, R_base);
    cv::Rodrigues(pose_target.rotation, R_target);

    cv::Mat R_rel = R_base.t() * R_target;
    cv::Mat t_rel = R_base.t() * (cv::Mat(pose_target.translation) - cv::Mat(pose_base.translation));

    cv::Rodrigues(R_rel, rel_r);

    rel_t = cv::Vec3d(t_rel.at<double>(0), t_rel.at<double>(1), t_rel.at<double>(2));
    distance = std::sqrt(rel_t[0]*rel_t[0] + rel_t[1]*rel_t[1] + rel_t[2]*rel_t[2]);
}

void AprilTagDetectorNode::printRelativePose(
    const TagPose& pose_id0, const TagPose& pose_id1,
    const TagPose& pose_id2,
    bool id0_found, bool id1_found, bool id2_found) {

    cv::Vec3d rel_t, rel_r;
    double distance;
    computeRelativePose(pose_id0, pose_id2, rel_t, rel_r, distance);

    std::stringstream ss;
    ss << std::fixed << std::setprecision(3);
    ss << "\n╔══════════════════════════════════════════════════════════════╗\n";
    ss << "║  3-Tag System: ID" << base_tag_id_0_ << "(base) → ID" << target_tag_id_ << "(target)";
    if (id1_found) ss << "  [ID" << base_tag_id_1_ << " aux]";
    ss << "\n";
    ss << "╠══════════════════════════════════════════════════════════════╣\n";
    ss << "║  Tags: ID0=" << (id0_found?"✅":"❌")
       << "  ID1=" << (id1_found?"✅":"❌")
       << "  ID2=" << (id2_found?"✅":"❌") << "\n";
    ss << "╠══════════════════════════════════════════════════════════════╣\n";

    // ID0 camera-frame pose
    ss << "║  Camera → ID0:  X=" << std::setw(8) << pose_id0.translation[0]*1000.0 << " mm"
       << "  Y=" << std::setw(8) << pose_id0.translation[1]*1000.0 << " mm"
       << "  Z=" << std::setw(8) << pose_id0.translation[2]*1000.0 << " mm\n";
    ss << "║  Camera → ID2:  X=" << std::setw(8) << pose_id2.translation[0]*1000.0 << " mm"
       << "  Y=" << std::setw(8) << pose_id2.translation[1]*1000.0 << " mm"
       << "  Z=" << std::setw(8) << pose_id2.translation[2]*1000.0 << " mm\n";
    ss << "╠══════════════════════════════════════════════════════════════╣\n";
    ss << "║  ★ ID2 → ID0 (relative):\n";
    ss << "║     Position:  X=" << std::setw(8) << rel_t[0]*1000.0 << " mm"
       << "  Y=" << std::setw(8) << rel_t[1]*1000.0 << " mm"
       << "  Z=" << std::setw(8) << rel_t[2]*1000.0 << " mm\n";
    ss << "║     Distance:  " << std::setw(8) << distance*1000.0 << " mm\n";
    ss << "║     Rotation:  Rx=" << std::setw(7) << rel_r[0]*180.0/M_PI << "°"
       << "  Ry=" << std::setw(7) << rel_r[1]*180.0/M_PI << "°"
       << "  Rz=" << std::setw(7) << rel_r[2]*180.0/M_PI << "°\n";

    if (id1_found) {
        // Also compute ID0→ID1 for verification
        cv::Mat R_base, R_id1;
        cv::Rodrigues(pose_id0.rotation, R_base);
        cv::Rodrigues(pose_id1.rotation, R_id1);
        cv::Mat t_check = R_base.t() * (cv::Mat(pose_id1.translation) - cv::Mat(pose_id0.translation));
        double check_dist = cv::norm(t_check);
        ss << "║  ID0→ID1 (check): " << std::setw(7) << check_dist*1000.0 << " mm";
        if (std::abs(check_dist - 0.1587) < 0.02) ss << " ✓";
        else ss << " ⚠ (expected ~158.7mm)";
        ss << "\n";
    }
    ss << "╚══════════════════════════════════════════════════════════════╝\n";

    RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());
}

void AprilTagDetectorNode::publishRelativePose(
    const cv::Vec3d& rel_t, const cv::Vec3d& rel_r,
    double distance, bool id0_found, bool id1_found, bool id2_found,
    const std_msgs::msg::Header& header) {

    auto msg = std::make_unique<geometry_msgs::msg::PoseStamped>();
    msg->header = header;
    msg->header.frame_id = "base_tag_" + std::to_string(base_tag_id_0_);

    msg->pose.position.x = rel_t[0];
    msg->pose.position.y = rel_t[1];
    msg->pose.position.z = rel_t[2];

    double angle = cv::norm(rel_r);
    if (angle > 1e-6) {
        cv::Vec3d axis = rel_r / angle;
        tf2::Quaternion q(
            axis[0] * sin(angle / 2), axis[1] * sin(angle / 2),
            axis[2] * sin(angle / 2), cos(angle / 2));
        msg->pose.orientation.x = q.x();
        msg->pose.orientation.y = q.y();
        msg->pose.orientation.z = q.z();
        msg->pose.orientation.w = q.w();
    } else {
        msg->pose.orientation.w = 1.0;
    }

    relative_pose_pub_->publish(std::move(msg));
}

}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<apriltag_zed_visp::AprilTagDetectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
