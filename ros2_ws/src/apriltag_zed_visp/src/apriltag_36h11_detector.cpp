#include "apriltag_zed_visp/apriltag_36h11_detector.h"
#include <algorithm>
#include <numeric>

namespace apriltag_zed_visp {

const uint64_t AprilTag36h11Detector::VALID_CODES_36H11[] = {
    0x00000000000L, 0x0000010fc15L, 0x0000021e42aL, 0x0000031183fL,
    0x00000439854L, 0x00000536441L, 0x00000627c7eL, 0x0000072806bL,
    0x000008710a8L, 0x0000097ecbdL, 0x00000a6f482L, 0x00000b60897L,
    0x00000c488fcL, 0x00000d474e9L, 0x00000e56cd6L, 0x00000f590c3L,
    0x000010e2150L, 0x000011edd45L, 0x000012fc57aL, 0x000013f196fL,
    0x000014db904L, 0x000015d4511L, 0x000016c5d2eL, 0x000017ca13bL,
    0x000018931f8L, 0x0000199cdedL, 0x00001a8d5d2L, 0x00001b829c7L,
    0x00001caa9acL, 0x00001da55b9L, 0x00001eb4d86L, 0x00001fbb193L
};

const int AprilTag36h11Detector::NUM_VALID_CODES = 32;

AprilTag36h11Detector::AprilTag36h11Detector(int target_id, double quad_decimate,
                                               double quad_sigma, bool refine_edges)
    : target_id_(target_id)
    , quad_decimate_(quad_decimate)
    , quad_sigma_(quad_sigma)
    , refine_edges_(refine_edges)
{}

AprilTag36h11Detector::~AprilTag36h11Detector() {}

std::vector<AprilTagDetection> AprilTag36h11Detector::detect(const cv::Mat& image) {
    std::vector<AprilTagDetection> detections;
    
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    if (quad_sigma_ > 0) {
        cv::GaussianBlur(gray, gray, cv::Size(0, 0), quad_sigma_);
    }
    
    if (quad_decimate_ > 1.0) {
        cv::resize(gray, gray, cv::Size(), 1.0/quad_decimate_, 1.0/quad_decimate_, cv::INTER_AREA);
    }
    
    auto quads = findQuads(gray);
    
    for (const auto& quad : quads) {
        if (!isQuadValid(quad, gray)) continue;
        
        std::vector<cv::Point2f> quad_f;
        for (const auto& p : quad) {
            quad_f.emplace_back(p.x * quad_decimate_, p.y * quad_decimate_);
        }
        
        cv::Mat tag_image = extractTagRegion(gray, quad, TAG_WIDTH + 2 * BLACK_BORDER);
        
        double hamming_dist;
        int tag_id = decodeTag36h11(tag_image, hamming_dist);
        
        if (tag_id >= 0 && tag_id == target_id_) {
            AprilTagDetection det;
            det.id = tag_id;
            det.corners = quad_f;
            det.center = cv::Point2f(
                (quad_f[0].x + quad_f[1].x + quad_f[2].x + quad_f[3].x) / 4.0,
                (quad_f[0].y + quad_f[1].y + quad_f[2].y + quad_f[3].y) / 4.0
            );
            det.hamming_distance = hamming_dist;
            det.decision_margin = 1.0 - hamming_dist / 11.0;
            
            if (refine_edges_) {
                refineCorners(image, det.corners);
            }
            
            detections.push_back(det);
        }
    }
    
    return detections;
}

std::vector<std::vector<cv::Point>> AprilTag36h11Detector::findQuads(const cv::Mat& image) {
    std::vector<std::vector<cv::Point>> quads;
    
    cv::Mat edges;
    cv::Canny(image, edges, 80, 200, 3);
    
    cv::Mat dilated;
    cv::dilate(edges, dilated, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)));
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(dilated, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    
    for (const auto& contour : contours) {
        if (contour.size() < 4) continue;
        
        double perimeter = cv::arcLength(contour, true);
        if (perimeter < 50) continue;
        
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, 0.02 * perimeter, true);
        
        if (approx.size() != 4) continue;
        
        if (!cv::isContourConvex(approx)) continue;
        
        std::sort(approx.begin(), approx.end(), 
            [](const cv::Point& a, const cv::Point& b) {
                return a.y < b.y;
            });
        
        if (approx[0].x > approx[1].x) std::swap(approx[0], approx[1]);
        if (approx[2].x < approx[3].x) std::swap(approx[2], approx[3]);
        
        quads.push_back(approx);
    }
    
    return quads;
}

bool AprilTag36h11Detector::isQuadValid(const std::vector<cv::Point>& quad, const cv::Mat& image) {
    if (quad.size() != 4) return false;
    
    double area = cv::contourArea(quad);
    if (area < 100 || area > image.total() * 0.9) return false;
    
    double min_dist = 1e10, max_dist = 0;
    for (int i = 0; i < 4; ++i) {
        int j = (i + 1) % 4;
        double dist = cv::norm(quad[i] - quad[j]);
        min_dist = std::min(min_dist, dist);
        max_dist = std::max(max_dist, dist);
    }
    
    if (max_dist / min_dist > 4.0) return false;
    
    return true;
}

cv::Mat AprilTag36h11Detector::extractTagRegion(const cv::Mat& image, 
                                                  const std::vector<cv::Point>& quad,
                                                  int tag_size) {
    std::vector<cv::Point2f> src_pts;
    std::vector<cv::Point2f> dst_pts;
    
    for (const auto& p : quad) {
        src_pts.emplace_back(p);
    }
    
    int size = tag_size * 50;
    dst_pts.emplace_back(0, 0);
    dst_pts.emplace_back(size - 1, 0);
    dst_pts.emplace_back(size - 1, size - 1);
    dst_pts.emplace_back(0, size - 1);
    
    cv::Mat M = cv::getPerspectiveTransform(src_pts, dst_pts);
    cv::Mat warped;
    cv::warpPerspective(image, warped, M, cv::Size(size, size));
    
    cv::Mat resized;
    cv::resize(warped, resized, cv::Size(tag_size, tag_size), 0, 0, cv::INTER_NEAREST);
    
    return resized;
}

int AprilTag36h11Detector::decodeTag36h11(const cv::Mat& tag_image, double& hamming_dist) {
    int tag_size = TAG_WIDTH + 2 * BLACK_BORDER;
    cv::Mat binary;
    cv::threshold(tag_image, binary, 0, 1, cv::THRESH_BINARY | cv::THRESH_OTSU);
    
    for (int i = 0; i < BLACK_BORDER; ++i) {
        for (int j = 0; j < tag_size; ++j) {
            if (binary.at<uchar>(i, j) != 0 || 
                binary.at<uchar>(tag_size - 1 - i, j) != 0 ||
                binary.at<uchar>(j, i) != 0 || 
                binary.at<uchar>(j, tag_size - 1 - i) != 0) {
                hamming_dist = 100;
                return -1;
            }
        }
    }
    
    uint64_t code = 0;
    for (int i = 0; i < TAG_WIDTH; ++i) {
        for (int j = 0; j < TAG_WIDTH; ++j) {
            int idx = i * TAG_WIDTH + j;
            int pixel = binary.at<uchar>(i + BLACK_BORDER, j + BLACK_BORDER);
            if (pixel > 0) {
                code |= (1ULL << idx);
            }
        }
    }
    
    int best_id = -1;
    int min_hamming = 11;
    
    for (int rot = 0; rot < 4; ++rot) {
        uint64_t rotated_code = rotateCode6x6(code, rot);
        
        for (int i = 0; i < NUM_VALID_CODES; ++i) {
            int ham = hammingDistance6x6(rotated_code, VALID_CODES_36H11[i]);
            if (ham < min_hamming) {
                min_hamming = ham;
                best_id = i;
            }
        }
    }
    
    hamming_dist = min_hamming;
    
    if (min_hamming <= 3) {
        return best_id;
    }
    
    return -1;
}

int AprilTag36h11Detector::hammingDistance6x6(uint64_t code1, uint64_t code2) {
    uint64_t diff = code1 ^ code2;
    int count = 0;
    uint64_t mask = (1ULL << 36) - 1;
    diff &= mask;
    
    while (diff) {
        count += diff & 1;
        diff >>= 1;
    }
    
    return count;
}

uint64_t AprilTag36h11Detector::rotateCode6x6(uint64_t code, int rotations) {
    rotations %= 4;
    if (rotations == 0) return code;
    
    uint64_t result = 0;
    for (int r = 0; r < 4; ++r) {
        for (int i = 0; i < TAG_WIDTH; ++i) {
            for (int j = 0; j < TAG_WIDTH; ++j) {
                int ni, nj;
                if (rotations == 1) { ni = j; nj = TAG_WIDTH - 1 - i; }
                else if (rotations == 2) { ni = TAG_WIDTH - 1 - i; nj = TAG_WIDTH - 1 - j; }
                else { ni = TAG_WIDTH - 1 - j; nj = i; }
                
                int bit = (code >> (i * TAG_WIDTH + j)) & 1;
                if (bit) {
                    result |= (1ULL << (ni * TAG_WIDTH + nj));
                }
            }
        }
    }
    
    return result;
}

void AprilTag36h11Detector::refineCorners(const cv::Mat& image, 
                                            std::vector<cv::Point2f>& corners,
                                            int window_size, int iterations) {
    if (corners.size() != 4) return;
    
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    cv::cornerSubPix(gray, corners, cv::Size(window_size, window_size), 
                      cv::Size(-1, -1),
                      cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 
                                       iterations, 0.001));
}

bool AprilTag36h11Detector::checkTagFamilies(int code) {
    return code >= 0 && code < NUM_VALID_CODES;
}

} 
