#include "apriltag_zed_visp/error_compensation.h"
#include <algorithm>
#include <cmath>

namespace apriltag_zed_visp {

ErrorCompensation::ErrorCompensation()
    : image_width_(640)
    , image_height_(480)
    , image_center_(320.0f, 240.0f)
    , tag_size_(0.06)
{}

ErrorCompensation::~ErrorCompensation() {}

void ErrorCompensation::setCompensationParams(const CompensationParams& params) {
    params_ = params;
}

CompensationParams ErrorCompensation::getCompensationParams() const {
    return params_;
}

void ErrorCompensation::setImageDimensions(int width, int height) {
    image_width_ = width;
    image_height_ = height;
    image_center_ = cv::Point2f(width / 2.0f, height / 2.0f);
}

TagQualityMetrics ErrorCompensation::computeQualityMetrics(
    const std::vector<cv::Point2f>& corners,
    const cv::Mat& camera_matrix) {
    
    TagQualityMetrics metrics;
    
    metrics.skew_angle = computeSkewAngle(corners, camera_matrix);
    
    metrics.edge_distance_ratio = computeEdgeDistanceRatio(corners);
    
    cv::Point2f center = computeTagCenter(corners);
    metrics.center_offset_x = (center.x - image_center_.x) / (image_width_ / 2.0);
    metrics.center_offset_y = (center.y - image_center_.y) / (image_height_ / 2.0);
    
    double area = polygonArea(corners);
    metrics.area_ratio = area / (image_width_ * image_height_);
    
    double side1 = cv::norm(corners[0] - corners[1]);
    double side2 = cv::norm(corners[1] - corners[2]);
    metrics.aspect_ratio = std::min(side1, side2) / std::max(side1, side2);
    
    metrics.confidence = computeConfidence(metrics);
    
    return metrics;
}

double ErrorCompensation::computeSkewAngle(const std::vector<cv::Point2f>& corners,
                                            const cv::Mat& camera_matrix) {
    if (corners.size() != 4) return 0.0;
    
    cv::Vec3d v1 = cv::Vec3d(corners[1].x - corners[0].x, 
                              corners[1].y - corners[0].y, 0.0);
    cv::Vec3d v2 = cv::Vec3d(corners[3].x - corners[0].x, 
                              corners[3].y - corners[0].y, 0.0);
    
    double dot = v1.dot(v2);
    double norm_v1 = cv::norm(v1);
    double norm_v2 = cv::norm(v2);
    
    if (norm_v1 < 1e-6 || norm_v2 < 1e-6) return M_PI / 2.0;
    
    double cos_angle = dot / (norm_v1 * norm_v2);
    cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
    
    double angle = std::acos(cos_angle);
    
    return std::abs(angle - M_PI / 2.0);
}

double ErrorCompensation::computeEdgeDistanceRatio(
    const std::vector<cv::Point2f>& corners) {
    
    if (corners.size() != 4) return 0.0;
    
    double min_dist = image_width_;
    for (const auto& corner : corners) {
        double dist_left = corner.x;
        double dist_right = image_width_ - corner.x;
        double dist_top = corner.y;
        double dist_bottom = image_height_ - corner.y;
        
        min_dist = std::min({min_dist, dist_left, dist_right, dist_top, dist_bottom});
    }
    
    return min_dist / (std::min(image_width_, image_height_) / 2.0);
}

double ErrorCompensation::polygonArea(const std::vector<cv::Point2f>& corners) {
    if (corners.size() < 3) return 0.0;
    
    double area = 0.0;
    int n = corners.size();
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        area += corners[i].x * corners[j].y;
        area -= corners[j].x * corners[i].y;
    }
    
    return std::abs(area) / 2.0;
}

cv::Point2f ErrorCompensation::computeTagCenter(
    const std::vector<cv::Point2f>& corners) {
    
    cv::Point2f center(0, 0);
    for (const auto& corner : corners) {
        center += corner;
    }
    center *= 1.0f / corners.size();
    
    return center;
}

cv::Vec3d ErrorCompensation::compensateSkewError(const cv::Vec3d& translation,
                                                   const TagQualityMetrics& metrics) {
    double skew_factor = metrics.skew_angle;
    
    double dx = -params_.skew_error_gain_x * skew_factor * translation[0];
    double dy = -params_.skew_error_gain_y * skew_factor * translation[1];
    double dz = -params_.skew_error_gain_z * skew_factor * translation[2];
    
    return cv::Vec3d(dx, dy, dz);
}

cv::Vec3d ErrorCompensation::compensateEdgeError(const cv::Vec3d& translation,
                                                  const TagQualityMetrics& metrics) {
    double edge_factor = 1.0 - metrics.edge_distance_ratio;
    
    double dx = -params_.edge_error_gain_x * edge_factor * metrics.center_offset_x * translation[2];
    double dy = -params_.edge_error_gain_y * edge_factor * metrics.center_offset_y * translation[2];
    double dz = -params_.edge_error_gain_z * edge_factor * translation[2];
    
    return cv::Vec3d(dx, dy, dz);
}

std::vector<cv::Point2f> ErrorCompensation::compensateLensDistortion(
    const std::vector<cv::Point2f>& corners,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs) {
    
    std::vector<cv::Point2f> undistorted;
    cv::undistortPoints(corners, undistorted, camera_matrix, dist_coeffs, 
                         cv::noArray(), camera_matrix);
    
    return undistorted;
}

cv::Vec3d ErrorCompensation::compensateAll(const cv::Vec3d& translation,
                                            const cv::Vec3d& rotation,
                                            const std::vector<cv::Point2f>& corners,
                                            const cv::Mat& camera_matrix) {
    TagQualityMetrics metrics = computeQualityMetrics(corners, camera_matrix);
    
    cv::Vec3d skew_correction = compensateSkewError(translation, metrics);
    cv::Vec3d edge_correction = compensateEdgeError(translation, metrics);
    
    double confidence = metrics.confidence;
    cv::Vec3d total_correction = 
        skew_correction * confidence + edge_correction * confidence;
    
    return translation + total_correction;
}

double ErrorCompensation::computeConfidence(const TagQualityMetrics& metrics) {
    double skew_confidence = gaussianWeight(metrics.skew_angle, 0.3);
    double edge_confidence = gaussianWeight(1.0 - metrics.edge_distance_ratio, 0.3);
    double aspect_confidence = gaussianWeight(1.0 - metrics.aspect_ratio, 0.15);
    
    return (skew_confidence + edge_confidence + aspect_confidence) / 3.0;
}

double ErrorCompensation::gaussianWeight(double x, double sigma) {
    return std::exp(-0.5 * x * x / (sigma * sigma));
}

void ErrorCompensation::refineCornersSubpixel(const cv::Mat& image,
                                               std::vector<cv::Point2f>& corners,
                                               int window_size) {
    if (corners.empty()) return;
    
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }
    
    cv::cornerSubPix(gray, corners, cv::Size(window_size, window_size),
                     cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                                      100, 0.0001));
}

std::vector<cv::Point2f> ErrorCompensation::undistortPointsCustom(
    const std::vector<cv::Point2f>& points,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs) {
    
    std::vector<cv::Point2f> result;
    result.reserve(points.size());
    
    double fx = camera_matrix.at<double>(0, 0);
    double fy = camera_matrix.at<double>(1, 1);
    double cx = camera_matrix.at<double>(0, 2);
    double cy = camera_matrix.at<double>(1, 2);
    
    double k1 = 0.0, k2 = 0.0, p1 = 0.0, p2 = 0.0;
    if (!dist_coeffs.empty() && dist_coeffs.total() >= 4) {
        k1 = dist_coeffs.at<double>(0);
        k2 = dist_coeffs.at<double>(1);
        p1 = dist_coeffs.at<double>(2);
        p2 = dist_coeffs.at<double>(3);
    }
    
    for (const auto& pt : points) {
        double x = (pt.x - cx) / fx;
        double y = (pt.y - cy) / fy;
        
        for (int i = 0; i < 20; ++i) {
            double r2 = x * x + y * y;
            double r4 = r2 * r2;
            
            double radial_distortion = 1.0 + k1 * r2 + k2 * r4;
            double x_distort = x * radial_distortion + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
            double y_distort = y * radial_distortion + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
            
            x += (x - x_distort) * 0.5;
            y += (y - y_distort) * 0.5;
        }
        
        result.emplace_back(x * fx + cx, y * fy + cy);
    }
    
    return result;
}

} 
