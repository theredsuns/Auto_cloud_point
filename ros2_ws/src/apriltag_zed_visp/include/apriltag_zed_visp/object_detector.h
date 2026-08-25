#ifndef OBJECT_DETECTOR_H
#define OBJECT_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>

namespace apriltag_zed_visp {

struct ObjectContour {
    std::vector<cv::Point> contour;
    cv::Rect bounding_box;
    cv::Point2f center;
    double area;
    double perimeter;
    double aspect_ratio;
    double circularity;
    std::string shape;
    cv::Scalar color;
    double confidence;
};

struct ColorRange {
    std::string name;
    cv::Scalar lower;
    cv::Scalar upper;
    cv::Scalar display_color;
};

class ObjectDetector {
public:
    ObjectDetector();
    ~ObjectDetector();

    void setImageDimensions(int width, int height);
    
    void enableColorFilter(bool enable);
    void addColorRange(const std::string& name, 
                       const cv::Scalar& lower, 
                       const cv::Scalar& upper,
                       const cv::Scalar& display_color);
    void clearColorRanges();
    void setDefaultColorRanges();

    void setMinArea(double min_area);
    void setMaxArea(double max_area);
    void setContourApproximation(double epsilon_ratio);
    void enableMorphologicalOperations(bool enable);
    void setMorphologyKernelSize(int size);

    std::vector<ObjectContour> detect(const cv::Mat& image);
    
    void drawContours(cv::Mat& image, const std::vector<ObjectContour>& contours);
    
    std::vector<ObjectContour> getLastDetections() const;
    
    void enableTracking(bool enable);
    void setTrackingDistanceThreshold(double threshold);
    void enableCentroidFilter(bool enable);
    void setFilterAlpha(double alpha);

private:
    std::vector<ColorRange> color_ranges_;
    bool enable_color_filter_;
    
    double min_area_;
    double max_area_;
    double contour_epsilon_ratio_;
    
    bool enable_morphology_;
    int morph_kernel_size_;
    
    int image_width_;
    int image_height_;
    
    std::vector<ObjectContour> last_detections_;
    
    bool enable_tracking_;
    double tracking_distance_threshold_;
    int max_missing_frames_;
    std::map<int, ObjectContour> tracked_objects_;
    std::map<int, int> missing_frames_;
    int next_object_id_;
    
    bool enable_centroid_filter_;
    double filter_alpha_;
    std::map<int, cv::Point2f> filtered_centroids_;
    std::map<int, cv::Rect> filtered_bboxes_;
    std::map<int, std::vector<cv::Point>> filtered_contours_;

    cv::Mat preprocessImage(const cv::Mat& image);
    
    cv::Mat createColorMask(const cv::Mat& hsv_image);
    
    std::vector<std::vector<cv::Point>> findContours(const cv::Mat& mask);
    
    ObjectContour analyzeContour(const std::vector<cv::Point>& contour);
    
    std::string classifyShape(const std::vector<cv::Point>& contour, 
                             double aspect_ratio, 
                             double circularity);
    
    double computeCircularity(double area, double perimeter);
    
    cv::Scalar getColorFromMask(const cv::Mat& image, const cv::Rect& bbox);
};

} 

#endif 