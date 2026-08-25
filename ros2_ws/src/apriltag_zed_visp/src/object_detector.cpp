#include "apriltag_zed_visp/object_detector.h"
#include <algorithm>

namespace apriltag_zed_visp {

ObjectDetector::ObjectDetector()
    : enable_color_filter_(false)
    , min_area_(500.0)
    , max_area_(100000.0)
    , contour_epsilon_ratio_(0.02)
    , enable_morphology_(true)
    , morph_kernel_size_(3)
    , image_width_(640)
    , image_height_(480)
    , enable_tracking_(true)
    , tracking_distance_threshold_(50.0)
    , max_missing_frames_(3)
    , next_object_id_(1)
    , enable_centroid_filter_(true)
    , filter_alpha_(0.2) {
    setDefaultColorRanges();
}

ObjectDetector::~ObjectDetector() {}

void ObjectDetector::setImageDimensions(int width, int height) {
    image_width_ = width;
    image_height_ = height;
}

void ObjectDetector::enableColorFilter(bool enable) {
    enable_color_filter_ = enable;
}

void ObjectDetector::addColorRange(const std::string& name,
                                   const cv::Scalar& lower,
                                   const cv::Scalar& upper,
                                   const cv::Scalar& display_color) {
    ColorRange range;
    range.name = name;
    range.lower = lower;
    range.upper = upper;
    range.display_color = display_color;
    color_ranges_.push_back(range);
}

void ObjectDetector::clearColorRanges() {
    color_ranges_.clear();
}

void ObjectDetector::setDefaultColorRanges() {
    clearColorRanges();
    addColorRange("red",     cv::Scalar(0, 100, 100),   cv::Scalar(10, 255, 255),   cv::Scalar(0, 0, 255));
    addColorRange("red2",    cv::Scalar(160, 100, 100), cv::Scalar(180, 255, 255),  cv::Scalar(0, 0, 255));
    addColorRange("green",   cv::Scalar(40, 50, 50),    cv::Scalar(80, 255, 255),   cv::Scalar(0, 255, 0));
    addColorRange("blue",    cv::Scalar(90, 50, 50),    cv::Scalar(130, 255, 255),  cv::Scalar(255, 0, 0));
    addColorRange("yellow",  cv::Scalar(20, 100, 100),  cv::Scalar(35, 255, 255),   cv::Scalar(0, 255, 255));
    addColorRange("orange",  cv::Scalar(10, 100, 100),  cv::Scalar(20, 255, 255),   cv::Scalar(0, 165, 255));
    addColorRange("purple",  cv::Scalar(130, 50, 50),   cv::Scalar(160, 255, 255),  cv::Scalar(128, 0, 128));
}

void ObjectDetector::setMinArea(double min_area) {
    min_area_ = min_area;
}

void ObjectDetector::setMaxArea(double max_area) {
    max_area_ = max_area;
}

void ObjectDetector::setContourApproximation(double epsilon_ratio) {
    contour_epsilon_ratio_ = epsilon_ratio;
}

void ObjectDetector::enableMorphologicalOperations(bool enable) {
    enable_morphology_ = enable;
}

void ObjectDetector::setMorphologyKernelSize(int size) {
    morph_kernel_size_ = size;
}

void ObjectDetector::enableTracking(bool enable) {
    enable_tracking_ = enable;
}

void ObjectDetector::setTrackingDistanceThreshold(double threshold) {
    tracking_distance_threshold_ = threshold;
}

void ObjectDetector::enableCentroidFilter(bool enable) {
    enable_centroid_filter_ = enable;
}

void ObjectDetector::setFilterAlpha(double alpha) {
    filter_alpha_ = std::max(0.0, std::min(1.0, alpha));
}

cv::Mat ObjectDetector::preprocessImage(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.5);
    
    cv::Mat adaptive_thresh;
    cv::adaptiveThreshold(blurred, adaptive_thresh, 255, 
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                          cv::THRESH_BINARY_INV, 11, 2);
    
    cv::Mat edges;
    cv::Canny(blurred, edges, 30, 100);
    
    cv::Mat combined = adaptive_thresh & edges;
    
    return combined;
}

cv::Mat ObjectDetector::createColorMask(const cv::Mat& hsv_image) {
    cv::Mat mask = cv::Mat::zeros(hsv_image.size(), CV_8UC1);
    
    if (!enable_color_filter_ || color_ranges_.empty()) {
        return mask;
    }
    
    for (const auto& range : color_ranges_) {
        cv::Mat range_mask;
        cv::inRange(hsv_image, range.lower, range.upper, range_mask);
        mask = mask | range_mask;
    }
    
    return mask;
}

std::vector<std::vector<cv::Point>> ObjectDetector::findContours(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    
    cv::findContours(mask, contours, hierarchy, 
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    return contours;
}

double ObjectDetector::computeCircularity(double area, double perimeter) {
    if (perimeter == 0) return 0.0;
    return (4 * CV_PI * area) / (perimeter * perimeter);
}

std::string ObjectDetector::classifyShape(const std::vector<cv::Point>& contour,
                                          double aspect_ratio,
                                          double circularity) {
    std::vector<cv::Point> approx;
    double epsilon = contour_epsilon_ratio_ * cv::arcLength(contour, true);
    cv::approxPolyDP(contour, approx, epsilon, true);
    
    int vertices = approx.size();
    
    if (circularity > 0.8) {
        return "circle";
    } else if (circularity > 0.5) {
        return "ellipse";
    } else if (vertices == 3) {
        return "triangle";
    } else if (vertices == 4) {
        if (fabs(aspect_ratio - 1.0) < 0.15) {
            return "square";
        } else {
            return "rectangle";
        }
    } else if (vertices >= 5) {
        return "polygon";
    } else {
        return "unknown";
    }
}

cv::Scalar ObjectDetector::getColorFromMask(const cv::Mat& image, const cv::Rect& bbox) {
    cv::Mat roi = image(bbox);
    cv::Scalar mean_color = cv::mean(roi);
    return mean_color;
}

ObjectContour ObjectDetector::analyzeContour(const std::vector<cv::Point>& contour) {
    ObjectContour result;
    result.contour = contour;
    
    result.bounding_box = cv::boundingRect(contour);
    result.area = cv::contourArea(contour);
    result.perimeter = cv::arcLength(contour, true);
    
    cv::Moments moments = cv::moments(contour);
    if (moments.m00 != 0) {
        result.center.x = moments.m10 / moments.m00;
        result.center.y = moments.m01 / moments.m00;
    }
    
    result.aspect_ratio = static_cast<double>(result.bounding_box.width) / result.bounding_box.height;
    result.circularity = computeCircularity(result.area, result.perimeter);
    result.shape = classifyShape(contour, result.aspect_ratio, result.circularity);
    result.confidence = std::min(1.0, result.circularity * 0.5 + (1.0 / result.aspect_ratio) * 0.3 + 0.2);
    
    return result;
}

std::vector<ObjectContour> ObjectDetector::detect(const cv::Mat& image) {
    std::vector<ObjectContour> detections;
    
    cv::Mat edges = preprocessImage(image);
    
    cv::Mat mask = edges;
    
    if (enable_color_filter_) {
        cv::Mat hsv;
        cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
        cv::Mat color_mask = createColorMask(hsv);
        mask = mask | color_mask;
    }
    
    if (enable_morphology_) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, 
                                                   cv::Size(morph_kernel_size_, morph_kernel_size_));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    }
    
    std::vector<std::vector<cv::Point>> contours = findContours(mask);
    
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < min_area_ || area > max_area_) {
            continue;
        }
        
        ObjectContour obj = analyzeContour(contour);
        obj.color = getColorFromMask(image, obj.bounding_box);
        detections.push_back(obj);
    }
    
    std::sort(detections.begin(), detections.end(), 
              [](const ObjectContour& a, const ObjectContour& b) {
                  return a.area > b.area;
              });
    
    if (enable_tracking_) {
        std::map<int, ObjectContour> new_tracked_objects;
        std::vector<bool> matched_detections(detections.size(), false);
        
        for (const auto& tracked : tracked_objects_) {
            int best_det_idx = -1;
            double best_score = 1e9;
            
            for (size_t i = 0; i < detections.size(); ++i) {
                if (matched_detections[i]) continue;
                
                const auto& obj = detections[i];
                const auto& prev_obj = tracked.second;
                
                double dx = obj.center.x - prev_obj.center.x;
                double dy = obj.center.y - prev_obj.center.y;
                double dist_score = std::sqrt(dx*dx + dy*dy);
                
                double area_ratio = std::max(obj.area, prev_obj.area) / std::min(obj.area, prev_obj.area);
                double area_score = std::abs(area_ratio - 1.0) * 10;
                
                double shape_score = (obj.shape != prev_obj.shape) ? 50 : 0;
                
                double total_score = dist_score + area_score + shape_score;
                
                if (total_score < best_score && dist_score < tracking_distance_threshold_) {
                    best_score = total_score;
                    best_det_idx = i;
                }
            }
            
            if (best_det_idx >= 0) {
                matched_detections[best_det_idx] = true;
                ObjectContour& obj = detections[best_det_idx];
                
                obj.confidence = std::min(1.0, obj.confidence + 0.1);
                
                if (enable_centroid_filter_) {
                    cv::Point2f prev_center = filtered_centroids_[tracked.first];
                    obj.center.x = filter_alpha_ * obj.center.x + (1.0 - filter_alpha_) * prev_center.x;
                    obj.center.y = filter_alpha_ * obj.center.y + (1.0 - filter_alpha_) * prev_center.y;
                    filtered_centroids_[tracked.first] = obj.center;
                }
                
                if (enable_centroid_filter_) {
                    cv::Rect prev_bbox = filtered_bboxes_[tracked.first];
                    obj.bounding_box.x = static_cast<int>(filter_alpha_ * obj.bounding_box.x + (1.0 - filter_alpha_) * prev_bbox.x);
                    obj.bounding_box.y = static_cast<int>(filter_alpha_ * obj.bounding_box.y + (1.0 - filter_alpha_) * prev_bbox.y);
                    obj.bounding_box.width = static_cast<int>(filter_alpha_ * obj.bounding_box.width + (1.0 - filter_alpha_) * prev_bbox.width);
                    obj.bounding_box.height = static_cast<int>(filter_alpha_ * obj.bounding_box.height + (1.0 - filter_alpha_) * prev_bbox.height);
                    filtered_bboxes_[tracked.first] = obj.bounding_box;
                }
                
                if (enable_centroid_filter_ && !obj.contour.empty()) {
                    auto prev_contour_it = filtered_contours_.find(tracked.first);
                    if (prev_contour_it != filtered_contours_.end()) {
                        const auto& prev_contour = prev_contour_it->second;
                        std::vector<cv::Point> smoothed_contour;
                        
                        size_t min_size = std::min(obj.contour.size(), prev_contour.size());
                        for (size_t j = 0; j < min_size; ++j) {
                            int x = static_cast<int>(filter_alpha_ * obj.contour[j].x + (1.0 - filter_alpha_) * prev_contour[j].x);
                            int y = static_cast<int>(filter_alpha_ * obj.contour[j].y + (1.0 - filter_alpha_) * prev_contour[j].y);
                            smoothed_contour.push_back(cv::Point(x, y));
                        }
                        
                        if (obj.contour.size() > min_size) {
                            for (size_t j = min_size; j < obj.contour.size(); ++j) {
                                smoothed_contour.push_back(obj.contour[j]);
                            }
                        }
                        
                        obj.contour = smoothed_contour;
                    }
                    filtered_contours_[tracked.first] = obj.contour;
                }
                
                missing_frames_[tracked.first] = 0;
                new_tracked_objects[tracked.first] = obj;
            } else {
                missing_frames_[tracked.first]++;
                
                if (missing_frames_[tracked.first] <= max_missing_frames_) {
                    ObjectContour obj = tracked.second;
                    obj.confidence = std::max(0.1, obj.confidence - 0.1);
                    new_tracked_objects[tracked.first] = obj;
                }
            }
        }
        
        for (size_t i = 0; i < detections.size(); ++i) {
            if (!matched_detections[i]) {
                ObjectContour& obj = detections[i];
                obj.confidence = 0.5;
                
                if (enable_centroid_filter_) {
                    filtered_centroids_[next_object_id_] = obj.center;
                    filtered_bboxes_[next_object_id_] = obj.bounding_box;
                    filtered_contours_[next_object_id_] = obj.contour;
                }
                
                missing_frames_[next_object_id_] = 0;
                new_tracked_objects[next_object_id_] = obj;
                next_object_id_++;
            }
        }
        
        tracked_objects_ = new_tracked_objects;
        
        detections.clear();
        for (const auto& tracked : tracked_objects_) {
            detections.push_back(tracked.second);
        }
        
        std::sort(detections.begin(), detections.end(), 
                  [](const ObjectContour& a, const ObjectContour& b) {
                      return a.area > b.area;
                  });
    }
    
    last_detections_ = detections;
    return detections;
}

void ObjectDetector::drawContours(cv::Mat& image, const std::vector<ObjectContour>& contours) {
    for (size_t i = 0; i < contours.size(); ++i) {
        const auto& obj = contours[i];
        
        cv::Scalar draw_color = obj.color;
        if (obj.color[0] < 50 && obj.color[1] < 50 && obj.color[2] < 50) {
            draw_color = cv::Scalar(255, 255, 255);
        }
        
        cv::drawContours(image, std::vector<std::vector<cv::Point>>{obj.contour}, 
                         0, draw_color, 2);
        
        cv::rectangle(image, obj.bounding_box, draw_color, 2);
        
        cv::circle(image, obj.center, 3, cv::Scalar(0, 0, 255), -1);
        
        std::string label = obj.shape + " (" + std::to_string(static_cast<int>(obj.area)) + "px)";
        cv::putText(image, label, 
                    cv::Point(obj.bounding_box.x, obj.bounding_box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2);
    }
}

std::vector<ObjectContour> ObjectDetector::getLastDetections() const {
    return last_detections_;
}

} 