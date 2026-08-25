#ifndef POSE_ESTIMATOR_H
#define POSE_ESTIMATOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include <cmath>
#include <Eigen/Dense>

namespace apriltag_zed_visp {

struct TagPose {
    cv::Vec3d translation;
    cv::Vec3d rotation;
    cv::Matx33d rotation_matrix;
    double reprojection_error;
    bool valid;
};

struct CameraParameters {
    cv::Matx33d camera_matrix;
    cv::Mat dist_coeffs;
    int image_width;
    int image_height;
    double fx() const { return camera_matrix(0, 0); }
    double fy() const { return camera_matrix(1, 1); }
    double cx() const { return camera_matrix(0, 2); }
    double cy() const { return camera_matrix(1, 2); }
};

class PoseEstimator {
public:
    PoseEstimator(double tag_size = 0.162);
    
    ~PoseEstimator();
    
    void setCameraParameters(const CameraParameters& params);
    
    void setTagSize(double size) { tag_size_ = size; }
    double getTagSize() const { return tag_size_; }
    
    const CameraParameters& getCameraParameters() const { return camera_params_; }
    
    TagPose estimatePose(const std::vector<cv::Point2f>& corners);
    
    TagPose estimatePoseIterative(const std::vector<cv::Point2f>& corners, 
                                   int max_iterations = 50, 
                                   double eps = 1e-8);
    
    std::vector<cv::Point3f> getTagObjectPoints() const;
    
    double computeReprojectionError(const std::vector<cv::Point2f>& corners,
                                     const TagPose& pose);
    
    void refinePoseWithLM(const std::vector<cv::Point2f>& corners, TagPose& pose);
    
private:
    double tag_size_;
    CameraParameters camera_params_;
    bool camera_params_set_;
    
    std::vector<cv::Point3f> object_points_;
    
    void initializeObjectPoints();
    
    void rodriguesToMatrix(const cv::Vec3d& rvec, cv::Matx33d& R);
    
    void matrixToRodrigues(const cv::Matx33d& R, cv::Vec3d& rvec);
    
    void projectPoints(const std::vector<cv::Point3f>& points,
                       const TagPose& pose,
                       std::vector<cv::Point2f>& projected);
    
    void computeJacobian(const std::vector<cv::Point3f>& points,
                         const TagPose& pose,
                         Eigen::MatrixXd& J);
};

} 

#endif 
