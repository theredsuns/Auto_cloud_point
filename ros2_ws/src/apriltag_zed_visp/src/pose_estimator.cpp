#include "apriltag_zed_visp/pose_estimator.h"
#include <limits>

namespace apriltag_zed_visp {

PoseEstimator::PoseEstimator(double tag_size)
    : tag_size_(tag_size)
    , camera_params_set_(false)
{
    initializeObjectPoints();
}

PoseEstimator::~PoseEstimator() {}

void PoseEstimator::initializeObjectPoints() {
    double half_size = tag_size_ / 2.0;
    object_points_.clear();
    object_points_.emplace_back(-half_size, -half_size, 0.0);
    object_points_.emplace_back( half_size, -half_size, 0.0);
    object_points_.emplace_back( half_size,  half_size, 0.0);
    object_points_.emplace_back(-half_size,  half_size, 0.0);
}

void PoseEstimator::setCameraParameters(const CameraParameters& params) {
    camera_params_ = params;
    camera_params_set_ = true;
}

std::vector<cv::Point3f> PoseEstimator::getTagObjectPoints() const {
    return object_points_;
}

TagPose PoseEstimator::estimatePose(const std::vector<cv::Point2f>& corners) {
    TagPose pose;
    pose.valid = false;
    
    if (!camera_params_set_ || corners.size() != 4) {
        return pose;
    }
    
    cv::Vec3d rvec, tvec;
    
    bool success = cv::solvePnP(
        object_points_,
        corners,
        cv::Mat(camera_params_.camera_matrix),
        camera_params_.dist_coeffs,
        rvec,
        tvec,
        false,
        cv::SOLVEPNP_IPPE
    );
    
    if (!success) {
        success = cv::solvePnP(
            object_points_,
            corners,
            cv::Mat(camera_params_.camera_matrix),
            camera_params_.dist_coeffs,
            rvec,
            tvec,
            false,
            cv::SOLVEPNP_ITERATIVE
        );
    }
    
    if (success) {
        pose.translation = tvec;
        pose.rotation = rvec;
        rodriguesToMatrix(rvec, pose.rotation_matrix);
        pose.reprojection_error = computeReprojectionError(corners, pose);
        pose.valid = true;
    }
    
    return pose;
}

TagPose PoseEstimator::estimatePoseIterative(const std::vector<cv::Point2f>& corners,
                                         int max_iterations, double eps) {
    TagPose pose = estimatePose(corners);
    
    if (!pose.valid) {
        return pose;
    }
    
    refinePoseWithLM(corners, pose);
    
    return pose;
}

double PoseEstimator::computeReprojectionError(const std::vector<cv::Point2f>& corners,
                                            const TagPose& pose) {
    if (!pose.valid) return std::numeric_limits<double>::max();
    
    std::vector<cv::Point2f> projected;
    projectPoints(object_points_, pose, projected);
    
    double total_error = 0.0;
    for (size_t i = 0; i < corners.size(); ++i) {
        double dx = corners[i].x - projected[i].x;
        double dy = corners[i].y - projected[i].y;
        total_error += dx * dx + dy * dy;
    }
    
    return std::sqrt(total_error / corners.size());
}

void PoseEstimator::refinePoseWithLM(const std::vector<cv::Point2f>& corners,
                                     TagPose& pose) {
    if (!pose.valid) return;
    
    cv::Vec3d rvec = pose.rotation;
    cv::Vec3d tvec = pose.translation;
    
    cv::solvePnPRefineLM(
        object_points_,
        corners,
        cv::Mat(camera_params_.camera_matrix),
        camera_params_.dist_coeffs,
        rvec,
        tvec,
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 1e-10)
    );
    
    pose.rotation = rvec;
    pose.translation = tvec;
    rodriguesToMatrix(rvec, pose.rotation_matrix);
    pose.reprojection_error = computeReprojectionError(corners, pose);
}

void PoseEstimator::rodriguesToMatrix(const cv::Vec3d& rvec, cv::Matx33d& R) {
    cv::Mat R_mat;
    cv::Rodrigues(rvec, R_mat);
    R = cv::Matx33d(R_mat);
}

void PoseEstimator::matrixToRodrigues(const cv::Matx33d& R, cv::Vec3d& rvec) {
    cv::Mat R_mat(R);
    cv::Rodrigues(R_mat, rvec);
}

void PoseEstimator::projectPoints(const std::vector<cv::Point3f>& points,
                                    const TagPose& pose,
                                    std::vector<cv::Point2f>& projected) {
    projected.clear();
    
    for (const auto& pt : points) {
        cv::Point3d p3d(pt.x, pt.y, pt.z);
        
        cv::Point3d p_cam = pose.rotation_matrix * p3d + 
                             cv::Point3d(pose.translation[0], 
                                            pose.translation[1], 
                                            pose.translation[2]);
        
        if (p_cam.z <= 0) {
            projected.emplace_back(-1, -1);
            continue;
        }
        
        double u = camera_params_.fx() * p_cam.x / p_cam.z + camera_params_.cx();
        double v = camera_params_.fy() * p_cam.y / p_cam.z + camera_params_.cy();
        
        projected.emplace_back(u, v);
    }
}

void PoseEstimator::computeJacobian(const std::vector<cv::Point3f>& points,
                                   const TagPose& pose,
                                   Eigen::MatrixXd& J) {
    int n = points.size();
    J.resize(2 * n, 6);
    
    for (int i = 0; i < n; ++i) {
        cv::Point3d p(points[i].x, points[i].y, points[i].z);
        
        cv::Point3d p_cam = pose.rotation_matrix * p + 
                             cv::Point3d(pose.translation[0], 
                                        pose.translation[1], 
                                        pose.translation[2]);
        
        double X = p_cam.x;
        double Y = p_cam.y;
        double Z = p_cam.z;
        double Z2 = Z * Z;
        
        double fx = camera_params_.fx();
        double fy = camera_params_.fy();
        
        J(2*i, 0) = fx * X * Y / Z2;
        J(2*i, 1) = -fx * (1 + X * X / Z2);
        J(2*i, 2) = fx * Y / Z;
        J(2*i, 3) = -fx / Z;
        J(2*i, 4) = 0;
        J(2*i, 5) = fx * X / Z2;
        
        J(2*i+1, 0) = fy * (1 + Y * Y / Z2);
        J(2*i+1, 1) = -fy * X * Y / Z2;
        J(2*i+1, 2) = -fy * X / Z;
        J(2*i+1, 3) = 0;
        J(2*i+1, 4) = -fy / Z;
        J(2*i+1, 5) = fy * Y / Z2;
    }
}

} 
