#ifndef APRILTAG_36H11_DETECTOR_H
#define APRILTAG_36H11_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include <cmath>

namespace apriltag_zed_visp {

struct AprilTagDetection {
    int id;
    std::vector<cv::Point2f> corners;
    cv::Point2f center;
    double hamming_distance;
    double decision_margin;
};

class AprilTag36h11Detector {
public:
    AprilTag36h11Detector(int target_id = 2, double quad_decimate = 1.0, 
                          double quad_sigma = 0.0, bool refine_edges = true);
    
    ~AprilTag36h11Detector();
    
    std::vector<AprilTagDetection> detect(const cv::Mat& image);
    
    void setTargetId(int id) { target_id_ = id; }
    int getTargetId() const { return target_id_; }
    
    void setRefineEdges(bool refine) { refine_edges_ = refine; }
    
private:
    std::vector<std::vector<cv::Point>> findQuads(const cv::Mat& image);
    
    bool isQuadValid(const std::vector<cv::Point>& quad, const cv::Mat& image);
    
    cv::Mat extractTagRegion(const cv::Mat& image, const std::vector<cv::Point>& quad, 
                             int tag_size = 6);
    
    int decodeTag36h11(const cv::Mat& tag_image, double& hamming_dist);
    
    int hammingDistance6x6(uint64_t code1, uint64_t code2);
    
    uint64_t rotateCode6x6(uint64_t code, int rotations);
    
    void refineCorners(const cv::Mat& image, std::vector<cv::Point2f>& corners,
                       int window_size = 5, int iterations = 10);
    
    bool checkTagFamilies(int code);
    
    const int TARGET_TAG_ID = 2;
    const int TAG_WIDTH = 6;
    const int BLACK_BORDER = 1;
    
    int target_id_;
    double quad_decimate_;
    double quad_sigma_;
    bool refine_edges_;
    
    static const uint64_t VALID_CODES_36H11[];
    static const int NUM_VALID_CODES;
};

} 

#endif 
