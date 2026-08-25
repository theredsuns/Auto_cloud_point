#include <sl/Camera.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/viz.hpp>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <csignal>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using PointCloudPublisher = rclcpp::Publisher<sensor_msgs::msg::PointCloud2>;
using ImagePublisher = rclcpp::Publisher<sensor_msgs::msg::Image>;
using CompressedImagePublisher = rclcpp::Publisher<sensor_msgs::msg::CompressedImage>;
using MarkerPublisher = rclcpp::Publisher<visualization_msgs::msg::Marker>;

std::atomic<bool> keep_running{true};

void signalHandler(int) {
    keep_running = false;
}

struct Options {
    std::string output = "rigid_body.obj";
    std::string svo_path;
    int serial_number = 0;
    int frames = 0;
    int mesh_update_ms = 750;
    int max_display_triangles = 200000;
    int max_rviz_triangles = 50000;
    int max_display_points = 120000;
    int point_cloud_every = 3;
    int image_every = 3;
    int jpeg_quality = 80;
    bool texture = false;
    bool preview = true;
    bool viewer_3d = true;
    bool image_only = false;
};

void printUsage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n"
        << "  --output PATH    Output .obj or .ply file (default: rigid_body.obj)\n"
        << "  --frames N       Optional safety limit; 0 means run until B (default: 0)\n"
        << "  --serial N       Open a ZED camera by serial number\n"
        << "  --svo PATH       Build the model from an SVO recording\n"
        << "  --texture        Generate a texture (best used with OBJ output)\n"
        << "  --no-preview     Disable the live camera preview window\n"
        << "  --no-3d          Disable the live interactive 3D model window\n"
        << "  --image-only     Publish camera images only; no depth, mesh, or model file\n"
        << "  --mesh-update-ms N  3D mesh refresh interval (default: 750)\n"
        << "  --max-display-triangles N  Display limit; saved mesh stays full resolution\n"
        << "  --max-rviz-triangles N  RViz filled-surface limit (default: 50000)\n"
        << "  --max-display-points N  Live point-cloud display limit (default: 120000)\n"
        << "  --point-cloud-every N  Refresh point cloud every N valid frames (default: 3)\n"
        << "  --image-every N  Publish image every N grabbed frames (default: 3)\n"
        << "  --jpeg-quality N  JPEG quality from 1 to 100 (default: 80)\n"
        << "  --help           Show this message\n";
}

bool parsePositiveInt(const char* text, int& value) {
    try {
        std::size_t used = 0;
        const int parsed = std::stoi(text, &used);
        if (used != std::string(text).size() || parsed < 0) {
            return false;
        }
        value = parsed;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool parseOptions(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help") {
            printUsage(argv[0]);
            return false;
        }
        if (arg == "--texture") {
            options.texture = true;
            continue;
        }
        if (arg == "--no-preview") {
            options.preview = false;
            continue;
        }
        if (arg == "--no-3d") {
            options.viewer_3d = false;
            continue;
        }
        if (arg == "--image-only") {
            options.image_only = true;
            continue;
        }
        if (i + 1 >= argc) {
            std::cerr << "Missing value after " << arg << '\n';
            return false;
        }
        if (arg == "--output") {
            options.output = argv[++i];
        } else if (arg == "--svo") {
            options.svo_path = argv[++i];
        } else if (arg == "--serial") {
            if (!parsePositiveInt(argv[++i], options.serial_number)) {
                std::cerr << "Invalid serial number\n";
                return false;
            }
        } else if (arg == "--frames") {
            if (!parsePositiveInt(argv[++i], options.frames)) {
                std::cerr << "Invalid frame count\n";
                return false;
            }
        } else if (arg == "--mesh-update-ms") {
            if (!parsePositiveInt(argv[++i], options.mesh_update_ms) || options.mesh_update_ms < 100) {
                std::cerr << "Mesh update interval must be at least 100 ms\n";
                return false;
            }
        } else if (arg == "--max-display-triangles") {
            if (!parsePositiveInt(argv[++i], options.max_display_triangles) ||
                options.max_display_triangles < 1000) {
                std::cerr << "Display triangle limit must be at least 1000\n";
                return false;
            }
        } else if (arg == "--max-rviz-triangles") {
            if (!parsePositiveInt(argv[++i], options.max_rviz_triangles) ||
                options.max_rviz_triangles < 1000) {
                std::cerr << "RViz triangle limit must be at least 1000\n";
                return false;
            }
        } else if (arg == "--max-display-points") {
            if (!parsePositiveInt(argv[++i], options.max_display_points) ||
                options.max_display_points < 1000) {
                std::cerr << "Display point limit must be at least 1000\n";
                return false;
            }
        } else if (arg == "--point-cloud-every") {
            if (!parsePositiveInt(argv[++i], options.point_cloud_every) ||
                options.point_cloud_every < 1) {
                std::cerr << "Point-cloud frame interval must be at least 1\n";
                return false;
            }
        } else if (arg == "--image-every") {
            if (!parsePositiveInt(argv[++i], options.image_every) || options.image_every < 1) {
                std::cerr << "Image frame interval must be at least 1\n";
                return false;
            }
        } else if (arg == "--jpeg-quality") {
            if (!parsePositiveInt(argv[++i], options.jpeg_quality) ||
                options.jpeg_quality < 1 || options.jpeg_quality > 100) {
                std::cerr << "JPEG quality must be between 1 and 100\n";
                return false;
            }
        } else {
            std::cerr << "Unknown option: " << arg << '\n';
            return false;
        }
    }
    return true;
}

bool hasSupportedExtension(const std::string& path) {
    const auto dot = path.find_last_of('.');
    if (dot == std::string::npos) {
        return false;
    }
    std::string extension = path.substr(dot);
    for (char& c : extension) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return extension == ".obj" || extension == ".ply";
}

void viewerKeyboardCallback(const cv::viz::KeyboardEvent& event, void*) {
    if (event.action == cv::viz::KeyboardEvent::KEY_DOWN &&
        (event.code == 'b' || event.code == 'B')) {
        keep_running = false;
    }
}

cv::Affine3d toCvPose(const sl::Transform& transform) {
    cv::Matx44d matrix = cv::Matx44d::eye();
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            matrix(row, column) = transform(row, column);
        }
    }
    return cv::Affine3d(matrix);
}

cv::viz::WMesh makeDisplayMesh(const sl::Mesh& mesh, int max_triangles,
                               std::size_t& displayed_triangles) {
    std::size_t total_triangles = 0;
    for (const auto& chunk : mesh.chunks) {
        total_triangles += chunk.triangles.size();
    }

    const std::size_t stride = std::max<std::size_t>(
        1, (total_triangles + static_cast<std::size_t>(max_triangles) - 1) /
               static_cast<std::size_t>(max_triangles));
    displayed_triangles = (total_triangles + stride - 1) / stride;

    cv::Mat points(1, static_cast<int>(displayed_triangles * 3), CV_32FC3);
    cv::Mat polygons(1, static_cast<int>(displayed_triangles * 4), CV_32SC1);
    auto* point_data = points.ptr<cv::Vec3f>();
    auto* polygon_data = polygons.ptr<int>();

    std::size_t source_triangle = 0;
    std::size_t output_triangle = 0;
    for (const auto& chunk : mesh.chunks) {
        for (const auto& triangle : chunk.triangles) {
            if (source_triangle++ % stride != 0) {
                continue;
            }
            const std::size_t indices[3] = {triangle.x, triangle.y, triangle.z};
            polygon_data[output_triangle * 4] = 3;
            for (std::size_t corner = 0; corner < 3; ++corner) {
                const auto& vertex = chunk.vertices[indices[corner]];
                const int point_index = static_cast<int>(output_triangle * 3 + corner);
                point_data[point_index] = cv::Vec3f(vertex.x, vertex.y, vertex.z);
                polygon_data[output_triangle * 4 + corner + 1] = point_index;
            }
            ++output_triangle;
        }
    }

    return cv::viz::WMesh(points, polygons);
}

std::size_t publishMeshMarker(const MarkerPublisher::SharedPtr& publisher,
                              const rclcpp::Node::SharedPtr& node,
                              const sl::Mesh& mesh, int max_triangles) {
    // Avoid constructing and serializing a large marker when RViz is not connected.
    if (publisher->get_subscription_count() == 0) {
        return 0;
    }

    std::size_t total_triangles = 0;
    for (const auto& chunk : mesh.chunks) {
        total_triangles += chunk.triangles.size();
    }
    if (total_triangles == 0) {
        return 0;
    }

    const std::size_t stride = std::max<std::size_t>(
        1, (total_triangles + static_cast<std::size_t>(max_triangles) - 1) /
               static_cast<std::size_t>(max_triangles));

    visualization_msgs::msg::Marker marker;
    marker.header.stamp = node->now();
    marker.header.frame_id = "map";
    marker.ns = "zed_reconstruction";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 1.0;
    marker.scale.y = 1.0;
    marker.scale.z = 1.0;
    marker.color.r = 0.27f;
    marker.color.g = 0.69f;
    marker.color.b = 0.92f;
    marker.color.a = 0.9f;
    marker.points.reserve(std::min<std::size_t>(total_triangles, max_triangles) * 3);

    std::size_t source_triangle = 0;
    for (const auto& chunk : mesh.chunks) {
        for (const auto& triangle : chunk.triangles) {
            if (source_triangle++ % stride != 0) {
                continue;
            }
            const std::size_t indices[3] = {triangle.x, triangle.y, triangle.z};
            if (std::any_of(std::begin(indices), std::end(indices),
                            [&chunk](std::size_t index) {
                                return index >= chunk.vertices.size();
                            })) {
                continue;
            }
            for (const std::size_t index : indices) {
                const auto& vertex = chunk.vertices[index];
                geometry_msgs::msg::Point point;
                point.x = vertex.x;
                point.y = vertex.y;
                point.z = vertex.z;
                marker.points.push_back(point);
            }
        }
    }

    publisher->publish(marker);
    return marker.points.size() / 3;
}

void updateTrajectory(cv::viz::Viz3d& viewer, const std::vector<cv::Vec3f>& trajectory) {
    if (trajectory.size() < 2) {
        return;
    }
    cv::Mat points(1, static_cast<int>(trajectory.size()), CV_32FC3);
    std::copy(trajectory.begin(), trajectory.end(), points.ptr<cv::Vec3f>());
    viewer.showWidget("camera_path", cv::viz::WPolyLine(points, cv::viz::Color::orange()));
}

void publishPointCloud(const PointCloudPublisher::SharedPtr& publisher,
                       const rclcpp::Node::SharedPtr& node,
                       const std::vector<cv::Vec3f>& points,
                       const std::vector<cv::Vec3b>& colors,
                       const cv::Affine3d& pose) {
    sensor_msgs::msg::PointCloud2 message;
    message.header.stamp = node->now();
    message.header.frame_id = "map";
    message.height = 1;
    message.width = static_cast<std::uint32_t>(points.size());
    message.is_dense = true;

    sensor_msgs::PointCloud2Modifier modifier(message);
    modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
    modifier.resize(points.size());

    sensor_msgs::PointCloud2Iterator<float> x_iterator(message, "x");
    sensor_msgs::PointCloud2Iterator<float> y_iterator(message, "y");
    sensor_msgs::PointCloud2Iterator<float> z_iterator(message, "z");
    sensor_msgs::PointCloud2Iterator<float> rgb_iterator(message, "rgb");
    const cv::Matx33d rotation = pose.rotation();
    const cv::Vec3d translation = pose.translation();
    for (std::size_t index = 0; index < points.size(); ++index,
                     ++x_iterator, ++y_iterator, ++z_iterator, ++rgb_iterator) {
        const cv::Vec3d local(points[index][0], points[index][1], points[index][2]);
        const cv::Vec3d world = rotation * local + translation;
        *x_iterator = static_cast<float>(world[0]);
        *y_iterator = static_cast<float>(world[1]);
        *z_iterator = static_cast<float>(world[2]);

        const std::uint32_t packed_rgb =
            (static_cast<std::uint32_t>(colors[index][2]) << 16) |
            (static_cast<std::uint32_t>(colors[index][1]) << 8) |
            static_cast<std::uint32_t>(colors[index][0]);
        float rgb_float = 0.0f;
        std::memcpy(&rgb_float, &packed_rgb, sizeof(rgb_float));
        *rgb_iterator = rgb_float;
    }
    publisher->publish(message);
}

void publishLeftImage(const ImagePublisher::SharedPtr& publisher,
                      const CompressedImagePublisher::SharedPtr& compressed_publisher,
                      const rclcpp::Node::SharedPtr& node,
                      const sl::Mat& image,
                      int jpeg_quality) {
    cv::Mat bgra(static_cast<int>(image.getHeight()), static_cast<int>(image.getWidth()),
                 CV_8UC4, image.getPtr<sl::uchar1>(sl::MEM::CPU),
                 image.getStepBytes(sl::MEM::CPU));
    cv::Mat bgr;
    cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);

    std_msgs::msg::Header header;
    header.stamp = node->now();
    header.frame_id = "zed_scanner_camera";

    if (publisher) {
        sensor_msgs::msg::Image message;
        message.header = header;
        message.height = image.getHeight();
        message.width = image.getWidth();
        message.encoding = "bgr8";
        message.is_bigendian = false;
        message.step = message.width * 3;
        message.data.resize(static_cast<std::size_t>(message.step) * message.height);
        for (std::size_t row = 0; row < message.height; ++row) {
            std::memcpy(message.data.data() + row * message.step,
                        bgr.ptr<unsigned char>(static_cast<int>(row)), message.step);
        }
        publisher->publish(message);
    }

    sensor_msgs::msg::CompressedImage compressed;
    compressed.header = header;
    compressed.format = "jpeg";
    cv::imencode(".jpg", bgr, compressed.data,
                 {cv::IMWRITE_JPEG_QUALITY, jpeg_quality});
    compressed_publisher->publish(compressed);
}

bool updatePointCloud(cv::viz::Viz3d* viewer, sl::Camera& camera, sl::Mat& point_cloud,
                      const cv::Affine3d& camera_pose, int max_points,
                      std::size_t& displayed_points,
                      const PointCloudPublisher::SharedPtr& publisher,
                      const rclcpp::Node::SharedPtr& node) {
    if (!viewer && publisher->get_subscription_count() == 0) {
        displayed_points = 0;
        return false;
    }

    const sl::Resolution resolution(640, 360);
    const auto status = camera.retrieveMeasure(
        point_cloud, sl::MEASURE::XYZBGRA, sl::MEM::CPU, resolution);
    if (status != sl::ERROR_CODE::SUCCESS) {
        return false;
    }

    const std::size_t pixel_count =
        static_cast<std::size_t>(point_cloud.getWidth()) * point_cloud.getHeight();
    const std::size_t stride = std::max<std::size_t>(
        1, (pixel_count + static_cast<std::size_t>(max_points) - 1) /
               static_cast<std::size_t>(max_points));

    std::vector<cv::Vec3f> points;
    std::vector<cv::Vec3b> colors;
    points.reserve(std::min<std::size_t>(pixel_count, max_points));
    colors.reserve(points.capacity());

    std::size_t pixel_index = 0;
    for (std::size_t y = 0; y < point_cloud.getHeight(); ++y) {
        const auto* row = reinterpret_cast<const sl::float4*>(
            reinterpret_cast<const unsigned char*>(point_cloud.getPtr<sl::float1>(sl::MEM::CPU)) +
            y * point_cloud.getStepBytes(sl::MEM::CPU));
        for (std::size_t x = 0; x < point_cloud.getWidth(); ++x, ++pixel_index) {
            if (pixel_index % stride != 0) {
                continue;
            }
            const auto& point = row[x];
            if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
                !std::isfinite(point.z)) {
                continue;
            }

            std::uint32_t bgra = 0;
            std::memcpy(&bgra, &point.w, sizeof(bgra));
            points.emplace_back(point.x, point.y, point.z);
            colors.emplace_back(static_cast<unsigned char>(bgra & 0xff),
                                static_cast<unsigned char>((bgra >> 8) & 0xff),
                                static_cast<unsigned char>((bgra >> 16) & 0xff));
        }
    }
    if (points.empty()) {
        return false;
    }

    cv::Mat point_matrix(1, static_cast<int>(points.size()), CV_32FC3, points.data());
    cv::Mat color_matrix(1, static_cast<int>(colors.size()), CV_8UC3, colors.data());
    publishPointCloud(publisher, node, points, colors, camera_pose);
    if (viewer) {
        auto cloud_widget = cv::viz::WCloud(point_matrix, color_matrix);
        cloud_widget.setRenderingProperty(cv::viz::POINT_SIZE, 2.0);
        viewer->showWidget("live_point_cloud", cloud_widget, camera_pose);
    }
    displayed_points = points.size();
    return true;
}

std::size_t updateReconstructionCloud(cv::viz::Viz3d* viewer, const sl::Mesh& mesh,
                                      int max_points,
                                      const PointCloudPublisher::SharedPtr& publisher,
                                      const rclcpp::Node::SharedPtr& node) {
    if (!viewer && publisher->get_subscription_count() == 0) {
        return 0;
    }

    std::size_t total_vertices = 0;
    for (const auto& chunk : mesh.chunks) {
        total_vertices += chunk.vertices.size();
    }
    if (total_vertices == 0) {
        return 0;
    }

    const std::size_t stride = std::max<std::size_t>(
        1, (total_vertices + static_cast<std::size_t>(max_points) - 1) /
               static_cast<std::size_t>(max_points));
    std::vector<cv::Vec3f> points;
    std::vector<cv::Vec3b> colors;
    points.reserve(std::min<std::size_t>(total_vertices, max_points));
    colors.reserve(points.capacity());

    std::size_t source_vertex = 0;
    for (const auto& chunk : mesh.chunks) {
        const bool has_colors = chunk.colors.size() == chunk.vertices.size();
        for (std::size_t index = 0; index < chunk.vertices.size(); ++index) {
            if (source_vertex++ % stride != 0) {
                continue;
            }
            const auto& vertex = chunk.vertices[index];
            if (!std::isfinite(vertex.x) || !std::isfinite(vertex.y) ||
                !std::isfinite(vertex.z)) {
                continue;
            }
            points.emplace_back(vertex.x, vertex.y, vertex.z);
            if (has_colors) {
                const auto& color = chunk.colors[index];
                colors.emplace_back(color.x, color.y, color.z);
            } else {
                colors.emplace_back(40, 210, 255);
            }
        }
    }
    if (points.empty()) {
        return 0;
    }

    cv::Mat point_matrix(1, static_cast<int>(points.size()), CV_32FC3, points.data());
    cv::Mat color_matrix(1, static_cast<int>(colors.size()), CV_8UC3, colors.data());
    publishPointCloud(publisher, node, points, colors, cv::Affine3d::Identity());
    if (viewer) {
        auto cloud_widget = cv::viz::WCloud(point_matrix, color_matrix);
        cloud_widget.setRenderingProperty(cv::viz::POINT_SIZE, 3.0);
        viewer->showWidget("reconstruction_cloud", cloud_widget);
    }
    return points.size();
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parseOptions(argc, argv, options)) {
        return argc > 1 && std::string(argv[1]) == "--help" ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (!options.image_only && !hasSupportedExtension(options.output)) {
        std::cerr << "Output path must end in .obj or .ply\n";
        return EXIT_FAILURE;
    }

    rclcpp::init(argc, argv);
    auto ros_node = std::make_shared<rclcpp::Node>("zed_rigid_scanner");
    PointCloudPublisher::SharedPtr live_cloud_publisher;
    PointCloudPublisher::SharedPtr built_cloud_publisher;
    MarkerPublisher::SharedPtr mesh_publisher;
    if (!options.image_only) {
        live_cloud_publisher = ros_node->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/zed_scanner/live_cloud", rclcpp::SensorDataQoS());
        built_cloud_publisher = ros_node->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/zed_scanner/built_cloud", rclcpp::SensorDataQoS());
        mesh_publisher = ros_node->create_publisher<visualization_msgs::msg::Marker>(
            "/zed_scanner/mesh", rclcpp::SensorDataQoS());
    }
    ImagePublisher::SharedPtr image_publisher;
    if (!options.image_only) {
        image_publisher = ros_node->create_publisher<sensor_msgs::msg::Image>(
            "/zed_scanner/left_image", rclcpp::SensorDataQoS());
    }
    auto compressed_image_publisher =
        ros_node->create_publisher<sensor_msgs::msg::CompressedImage>(
            "/zed_scanner/left_image/compressed", rclcpp::SensorDataQoS());

    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    sl::InitParameters init;
    init.camera_resolution = sl::RESOLUTION::HD720;
    init.camera_fps = 30;
    init.depth_mode =
        options.image_only ? sl::DEPTH_MODE::NONE : sl::DEPTH_MODE::ULTRA;
    init.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    init.coordinate_units = sl::UNIT::METER;
    init.depth_minimum_distance = 0.20f;
    init.depth_maximum_distance = 2.0f;

    if (!options.svo_path.empty()) {
        init.input.setFromSVOFile(options.svo_path.c_str());
        init.svo_real_time_mode = false;
    } else if (options.serial_number > 0) {
        init.input.setFromSerialNumber(options.serial_number);
    }

    sl::Camera camera;
    sl::ERROR_CODE status = camera.open(init);
    if (status != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "Failed to open ZED camera: " << sl::toString(status) << '\n';
        return EXIT_FAILURE;
    }

    if (!options.image_only) {
        sl::PositionalTrackingParameters tracking;
        tracking.enable_area_memory = true;
        status = camera.enablePositionalTracking(tracking);
        if (status != sl::ERROR_CODE::SUCCESS) {
            std::cerr << "Failed to enable positional tracking: " << sl::toString(status) << '\n';
            camera.close();
            return EXIT_FAILURE;
        }

        sl::SpatialMappingParameters mapping;
        mapping.map_type = sl::SpatialMappingParameters::SPATIAL_MAP_TYPE::MESH;
        mapping.set(sl::SpatialMappingParameters::MAPPING_RESOLUTION::HIGH);
        mapping.set(sl::SpatialMappingParameters::MAPPING_RANGE::SHORT);
        mapping.save_texture = options.texture;
        mapping.use_chunk_only = false;
        mapping.stability_counter = 4;

        status = camera.enableSpatialMapping(mapping);
        if (status != sl::ERROR_CODE::SUCCESS) {
            std::cerr << "Failed to enable spatial mapping: " << sl::toString(status) << '\n';
            camera.disablePositionalTracking();
            camera.close();
            return EXIT_FAILURE;
        }
    }

    sl::RuntimeParameters runtime;
    runtime.confidence_threshold = 35;
    runtime.texture_confidence_threshold = 50;

    if (options.image_only) {
        std::cout << "ZED image-only stream started.\n"
                  << "Depth, point clouds, spatial mapping, mesh publication, and model "
                     "saving are disabled.\n"
                  << "ROS 2 topic: /zed_scanner/left_image/compressed\n";
    } else {
        std::cout << "ZED rigid-body scan started.\n"
                  << "Keep the object still and move the camera slowly around it.\n"
                  << "Keep a textured, stationary background visible for camera tracking.\n"
                  << "Press B in a viewer window when the model is complete, then it will be saved.\n"
                  << "ROS 2 topics: /zed_scanner/live_cloud, /zed_scanner/built_cloud, "
                     "/zed_scanner/mesh, /zed_scanner/left_image, "
                     "/zed_scanner/left_image/compressed\n"
                  << "Output: " << options.output << '\n';
    }

    std::unique_ptr<cv::viz::Viz3d> viewer;
    if (options.viewer_3d && !options.image_only) {
        try {
            viewer = std::make_unique<cv::viz::Viz3d>("ZED Live 3D Reconstruction");
            viewer->setBackgroundColor(cv::viz::Color(28, 32, 38));
            viewer->showWidget("world", cv::viz::WCoordinateSystem(0.25));
            // OpenCV 4.5 has no (scale, color) overload. Passing a Color as the
            // second argument is interpreted as an image and aborts viewer setup.
            viewer->showWidget("camera", cv::viz::WCameraPosition(0.12));
            viewer->showWidget(
                "help", cv::viz::WText("B: finish & save | Large: built | Small: live",
                                        cv::Point(20, 20), 18, cv::viz::Color::white()));
            viewer->registerKeyboardCallback(viewerKeyboardCallback);
            viewer->spinOnce(1, true);
        } catch (const cv::Exception& error) {
            std::cerr << "3D viewer disabled: " << error.what() << '\n';
            viewer.reset();
        }
    }

    int captured = 0;
    int failed_grabs = 0;
    bool preview_enabled = options.preview;
    bool preview_error_reported = false;
    sl::Pose pose;
    sl::Mat left_image;
    sl::Mat stream_image;
    sl::Mat point_cloud;
    sl::Mesh live_mesh;
    bool mesh_request_pending = false;
    auto last_mesh_request = std::chrono::steady_clock::now() -
                             std::chrono::milliseconds(options.mesh_update_ms);
    std::vector<cv::Vec3f> camera_trajectory;
    std::size_t displayed_live_points = 0;
    std::size_t displayed_model_points = 0;
    int grabbed_frames = 0;
    bool viewer_fitted_to_cloud = false;
    while (keep_running && (options.frames == 0 || captured < options.frames)) {
        status = camera.grab(runtime);
        if (status == sl::ERROR_CODE::END_OF_SVOFILE_REACHED) {
            std::cout << "\nReached the end of the SVO file.\n";
            break;
        }
        if (status != sl::ERROR_CODE::SUCCESS) {
            ++failed_grabs;
            if (failed_grabs > 100) {
                std::cerr << "\nToo many consecutive camera grab failures: "
                          << sl::toString(status) << '\n';
                break;
            }
            continue;
        }
        failed_grabs = 0;
        ++grabbed_frames;
        rclcpp::spin_some(ros_node);

        const auto tracking_state =
            options.image_only
                ? sl::POSITIONAL_TRACKING_STATE::OFF
                : camera.getPosition(pose, sl::REFERENCE_FRAME::WORLD);

        if (viewer) {
            if (viewer->wasStopped()) {
                keep_running = false;
            } else {
                viewer->spinOnce(1, true);
            }
        }

        if (!options.image_only &&
            (grabbed_frames == 1 || grabbed_frames % options.point_cloud_every == 0)) {
            try {
                const cv::Affine3d cloud_pose =
                    tracking_state == sl::POSITIONAL_TRACKING_STATE::OK
                        ? toCvPose(pose.pose_data)
                        : cv::Affine3d::Identity();
                if (updatePointCloud(viewer.get(), camera, point_cloud, cloud_pose,
                                     options.max_display_points, displayed_live_points,
                                     live_cloud_publisher, ros_node)) {
                    if (viewer && !viewer_fitted_to_cloud) {
                        viewer->resetCamera();
                        viewer_fitted_to_cloud = true;
                    }
                    if (viewer) {
                        std::ostringstream depth_status;
                        depth_status << "Live depth: " << displayed_live_points
                                     << " points | Tracking: "
                                     << static_cast<const char*>(sl::toString(tracking_state));
                        viewer->showWidget(
                            "status", cv::viz::WText(depth_status.str(), cv::Point(20, 50), 18,
                                                     cv::viz::Color::green()));
                    }
                }
            } catch (const cv::Exception& error) {
                std::cerr << "\nPoint-cloud update failed: " << error.what() << '\n';
            }
        }

        if (grabbed_frames == 1 || grabbed_frames % options.image_every == 0) {
            const auto stream_status = camera.retrieveImage(
                stream_image, sl::VIEW::LEFT, sl::MEM::CPU, sl::Resolution(1280, 720));
            if (stream_status == sl::ERROR_CODE::SUCCESS) {
                publishLeftImage(image_publisher, compressed_image_publisher, ros_node,
                                 stream_image, options.jpeg_quality);
            } else if (!preview_error_reported) {
                std::cerr << "\nUnable to retrieve stream image: "
                          << sl::toString(stream_status) << '\n';
                preview_error_reported = true;
            }
        }

        if (preview_enabled) {
            const auto image_status = camera.retrieveImage(
                left_image, sl::VIEW::LEFT, sl::MEM::CPU);
            if (image_status == sl::ERROR_CODE::SUCCESS) {
                cv::Mat bgra(
                    static_cast<int>(left_image.getHeight()),
                    static_cast<int>(left_image.getWidth()),
                    CV_8UC4,
                    left_image.getPtr<sl::uchar1>(sl::MEM::CPU),
                    left_image.getStepBytes(sl::MEM::CPU));
                cv::Mat preview;
                cv::cvtColor(bgra, preview, cv::COLOR_BGRA2BGR);

                const cv::Scalar status_color =
                    tracking_state == sl::POSITIONAL_TRACKING_STATE::OK
                        ? cv::Scalar(80, 220, 80)
                        : cv::Scalar(40, 180, 255);
                std::string frame_text = "Frames: " + std::to_string(captured);
                if (options.frames > 0) {
                    frame_text += "/" + std::to_string(options.frames);
                }
                cv::putText(preview, frame_text, cv::Point(24, 38),
                            cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 255, 255), 2,
                            cv::LINE_AA);
                const std::string tracking_text =
                    std::string("Tracking: ") + static_cast<const char*>(sl::toString(tracking_state));
                cv::putText(preview, tracking_text,
                            cv::Point(24, 76), cv::FONT_HERSHEY_SIMPLEX, 0.75,
                            status_color, 2, cv::LINE_AA);
                cv::putText(preview, "B: finish and save",
                            cv::Point(24, static_cast<int>(left_image.getHeight()) - 24),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2,
                            cv::LINE_AA);

                try {
                    cv::imshow("ZED 3D Scanner", preview);
                    const int key = cv::waitKey(1) & 0xff;
                    if (key == 'b' || key == 'B') {
                        keep_running = false;
                    }
                } catch (const cv::Exception& error) {
                    std::cerr << "\nPreview disabled: " << error.what() << '\n';
                    preview_enabled = false;
                    cv::destroyAllWindows();
                }
            } else if (!preview_error_reported) {
                std::cerr << "\nUnable to retrieve preview image: "
                          << sl::toString(image_status) << '\n';
                preview_error_reported = true;
            }
        }

        if (!keep_running) {
            break;
        }
        if (options.image_only) {
            ++captured;
            if (captured == 1 || captured % 300 == 0) {
                std::cout << "\rStreamed " << captured << " camera frames    " << std::flush;
            }
            continue;
        }
        if (tracking_state != sl::POSITIONAL_TRACKING_STATE::OK) {
            if (captured % 30 == 0) {
                std::cout << "\rWaiting for stable positional tracking..." << std::flush;
            }
            continue;
        }

        ++captured;
        {
            const cv::Affine3d camera_pose = toCvPose(pose.pose_data);
            if (viewer) {
            viewer->setWidgetPose("camera", camera_pose);
            if (captured == 1 || captured % 5 == 0) {
                const cv::Vec3d translation = camera_pose.translation();
                camera_trajectory.emplace_back(
                    static_cast<float>(translation[0]), static_cast<float>(translation[1]),
                    static_cast<float>(translation[2]));
            }
            }

            const auto now = std::chrono::steady_clock::now();
            if (!mesh_request_pending &&
                now - last_mesh_request >= std::chrono::milliseconds(options.mesh_update_ms)) {
                camera.requestSpatialMapAsync();
                mesh_request_pending = true;
            }
            if (mesh_request_pending &&
                camera.getSpatialMapRequestStatusAsync() == sl::ERROR_CODE::SUCCESS) {
                const auto mesh_status = camera.retrieveSpatialMapAsync(live_mesh);
                mesh_request_pending = false;
                last_mesh_request = now;
                if (mesh_status == sl::ERROR_CODE::SUCCESS) {
                    try {
                        std::size_t total_triangles = 0;
                        for (const auto& chunk : live_mesh.chunks) {
                            total_triangles += chunk.triangles.size();
                        }
                        if (total_triangles == 0) {
                            continue;
                        }
                        std::size_t displayed_triangles = 0;
                        if (viewer) {
                            auto mesh_widget = makeDisplayMesh(
                                live_mesh, options.max_display_triangles, displayed_triangles);
                            mesh_widget.setColor(cv::viz::Color(70, 175, 235));
                            mesh_widget.setRenderingProperty(cv::viz::OPACITY, 0.48);
                            viewer->showWidget("live_mesh", mesh_widget);
                        } else {
                            displayed_triangles = total_triangles;
                        }
                        displayed_model_points = updateReconstructionCloud(
                            viewer.get(), live_mesh, options.max_display_points,
                            built_cloud_publisher, ros_node);
                        const std::size_t rviz_triangles = publishMeshMarker(
                            mesh_publisher, ros_node, live_mesh, options.max_rviz_triangles);
                        if (viewer) {
                            updateTrajectory(*viewer, camera_trajectory);

                            std::ostringstream status_text;
                            status_text << "Frames: " << captured;
                            if (options.frames > 0) {
                                status_text << '/' << options.frames;
                            }
                            status_text << " | live: " << displayed_live_points
                                        << " | built: " << displayed_model_points
                                        << " | mesh: " << displayed_triangles << " triangles"
                                        << " | RViz: " << rviz_triangles;
                            viewer->showWidget(
                                "status", cv::viz::WText(status_text.str(), cv::Point(20, 50), 18,
                                                         cv::viz::Color::green()));
                        }
                    } catch (const cv::Exception& error) {
                        std::cerr << "\n3D mesh update failed; viewer disabled: " << error.what()
                                  << '\n';
                        viewer.reset();
                    }
                }
            }
        }
        if (captured == 1 || captured % 30 == 0) {
            std::cout << "\rCaptured " << captured;
            if (options.frames > 0) {
                std::cout << '/' << options.frames;
            }
            std::cout << " frames | mapping=" << camera.getSpatialMappingState()
                      << "    " << std::flush;
        }
    }
    if (preview_enabled) {
        cv::destroyAllWindows();
    }
    if (options.image_only) {
        camera.close();
        rclcpp::shutdown();
        std::cout << "\nImage-only stream stopped. No model file was created.\n";
        return EXIT_SUCCESS;
    }
    std::cout << "\nExtracting mesh from " << captured << " valid frames...\n";

    sl::Mesh mesh;
    status = camera.extractWholeSpatialMap(mesh);
    if (status != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "Failed to extract mesh: " << sl::toString(status) << '\n';
        camera.disableSpatialMapping();
        camera.disablePositionalTracking();
        camera.close();
        rclcpp::shutdown();
        return EXIT_FAILURE;
    }

    std::cout << "Filtering mesh...\n";
    if (!mesh.filter(sl::MeshFilterParameters::MESH_FILTER::MEDIUM)) {
        std::cerr << "Mesh filtering failed; saving the unfiltered mesh.\n";
    }

    if (options.texture) {
        std::cout << "Generating texture...\n";
        if (!mesh.applyTexture(sl::MESH_TEXTURE_FORMAT::RGB)) {
            std::cerr << "Texture generation failed; saving geometry only.\n";
        }
    }

    std::cout << "Saving mesh to " << options.output << "...\n";
    const bool saved = mesh.save(options.output.c_str());

    camera.disableSpatialMapping();
    camera.disablePositionalTracking();
    camera.close();
    rclcpp::shutdown();

    if (!saved) {
        std::cerr << "Failed to save mesh. Check the output path and permissions.\n";
        return EXIT_FAILURE;
    }

    std::cout << "Done. Open the model in MeshLab or Blender for cropping and cleanup.\n";
    return EXIT_SUCCESS;
}
