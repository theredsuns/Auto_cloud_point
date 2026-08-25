#ifndef ERROR_COMPENSATION_H
#define ERROR_COMPENSATION_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

namespace apriltag_zed_visp {

struct CompensationParams {
    double skew_error_gain_x;
    double skew_error_gain_y;
    double skew_error_gain_z;
    double edge_error_gain_x;
    double edge_error_gain_y;
    double edge_error_gain_z;
    double lens_distortion_k1;
    double lens_distortion_k2;
    double lens_distortion_p1;
    double lens_distortion_p2;
    
    CompensationParams()
        : skew_error_gain_x(0.001)
        , skew_error_gain_y(0.001)
        , skew_error_gain_z(0.002)
        , edge_error_gain_x(0.005)
        , edge_error_gain_y(0.005)
        , edge_error_gain_z(0.01)
        , lens_distortion_k1(0.0)
        , lens_distortion_k2(0.0)
        , lens_distortion_p1(0.0)
        , lens_distortion_p2(0.0)
    {}
};

struct TagQualityMetrics {
    double skew_angle;
    double edge_distance_ratio;
    double center_offset_x;
    double center_offset_y;
    double area_ratio;
    double aspect_ratio;
    double confidence;
};

class ErrorCompensation {
public:
    ErrorCompensation();
    ~ErrorCompensation();
    
    void setCompensationParams(const CompensationParams& params);
    CompensationParams getCompensationParams() const;
    
    void setImageDimensions(int width, int height);
    
    TagQualityMetrics computeQualityMetrics(const std::vector<cv::Point2f>& corners,
                                             const cv::Mat& camera_matrix);
    
    cv::Vec3d compensateSkewError(const cv::Vec3d& translation,
                                   const TagQualityMetrics& metrics);
    
    cv::Vec3d compensateEdgeError(const cv::Vec3d& translation,
                                   const TagQualityMetrics& metrics);
    
    std::vector<cv::Point2f> compensateLensDistortion(
        const std::vector<cv::Point2f>& corners,
        const cv::Mat& camera_matrix,
        const cv::Mat& dist_coeffs);
    
    cv::Vec3d compensateAll(const cv::Vec3d& translation,
                            const cv::Vec3d& rotation,
                            const std::vector<cv::Point2f>& corners,
                            const cv::Mat& camera_matrix);
    
    double computeConfidence(const TagQualityMetrics& metrics);
    
    void refineCornersSubpixel(const cv::Mat& image,
                               std::vector<cv::Point2f>& corners,
                               int window_size = 11);
    
    std::vector<cv::Point2f> undistortPointsCustom(
        const std::vector<cv::Point2f>& points,
        const cv::Mat& camera_matrix,
        const cv::Mat& dist_coeffs);
    
private:
    CompensationParams params_;
    int image_width_;
    int image_height_;
    cv::Point2f image_center_;
    double tag_size_;
    
    double computeSkewAngle(const std::vector<cv::Point2f>& corners,
                             const cv::Mat& camera_matrix);
    
    double computeEdgeDistanceRatio(const std::vector<cv::Point2f>& corners);
    
    double polygonArea(const std::vector<cv::Point2f>& corners);
    
    cv::Point2f computeTagCenter(const std::vector<cv::Point2f>& corners);
    
    double gaussianWeight(double x, double sigma);
};

} 

#endif 
