from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    
    camera_frame = LaunchConfiguration('camera_frame')
    tag_frame = LaunchConfiguration('tag_frame')
    target_tag_id = LaunchConfiguration('target_tag_id')
    tag_size = LaunchConfiguration('tag_size')
    publish_tf = LaunchConfiguration('publish_tf')
    publish_marker = LaunchConfiguration('publish_marker')
    enable_compensation = LaunchConfiguration('enable_compensation')
    print_precise_pose = LaunchConfiguration('print_precise_pose')
    prefer_calibrated = LaunchConfiguration('prefer_calibrated')
    filter_window = LaunchConfiguration('filter_window')
    filter_alpha = LaunchConfiguration('filter_alpha')
    base_tag_id_0 = LaunchConfiguration('base_tag_id_0')
    base_tag_id_1 = LaunchConfiguration('base_tag_id_1')
    enable_relative_pose = LaunchConfiguration('enable_relative_pose')
    enable_object_detection = LaunchConfiguration('enable_object_detection')
    enable_color_filter = LaunchConfiguration('enable_color_filter')
    enable_display = LaunchConfiguration('enable_display')
    object_min_area = LaunchConfiguration('object_min_area')
    object_max_area = LaunchConfiguration('object_max_area')
    
    use_direct_camera = LaunchConfiguration('use_direct_camera')
    use_stereo = LaunchConfiguration('use_stereo')
    camera_device_id = LaunchConfiguration('camera_device_id')
    
    image_topic = LaunchConfiguration('image_topic')
    camera_info_topic = LaunchConfiguration('camera_info_topic')
    left_image_topic = LaunchConfiguration('left_image_topic')
    right_image_topic = LaunchConfiguration('right_image_topic')
    left_camera_info_topic = LaunchConfiguration('left_camera_info_topic')
    
    apriltag_share = get_package_share_directory('apriltag_zed_visp')
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_frame',
            default_value='camera_frame',
            description='Camera frame ID'
        ),
        DeclareLaunchArgument(
            'tag_frame',
            default_value='tag_36h11_id2',
            description='Tag frame ID'
        ),
        DeclareLaunchArgument(
            'target_tag_id',
            default_value='2',
            description='Target AprilTag ID (36h11 family)'
        ),
        DeclareLaunchArgument(
            'tag_size',
            default_value='0.06',
            description='Tag size in meters (6cm)'
        ),
        DeclareLaunchArgument(
            'publish_tf',
            default_value='true',
            description='Publish TF transform'
        ),
        DeclareLaunchArgument(
            'publish_marker',
            default_value='true',
            description='Publish visualization marker'
        ),
        DeclareLaunchArgument(
            'enable_compensation',
            default_value='true',
            description='Enable skew and edge error compensation'
        ),
        DeclareLaunchArgument(
            'print_precise_pose',
            default_value='true',
            description='Print precise pose with 3 decimal places'
        ),
        DeclareLaunchArgument(
            'prefer_calibrated',
            default_value='true',
            description='Use calibrated intrinsics when available'
        ),
        DeclareLaunchArgument(
            'filter_window',
            default_value='30',
            description='Sliding window size for relative pose filtering'
        ),
        DeclareLaunchArgument(
            'filter_alpha',
            default_value='0.15',
            description='Low-pass filter strength (0=heavy, 1=raw)'
        ),
        DeclareLaunchArgument(
            'base_tag_id_0',
            default_value='0',
            description='Base reference tag ID (default: 0)'
        ),
        DeclareLaunchArgument(
            'base_tag_id_1',
            default_value='1',
            description='Auxiliary verification tag ID (default: 1)'
        ),
        DeclareLaunchArgument(
            'enable_relative_pose',
            default_value='true',
            description='Compute and publish ID2 pose relative to ID0'
        ),
        DeclareLaunchArgument(
            'enable_object_detection',
            default_value='false',
            description='Enable object contour detection'
        ),
        DeclareLaunchArgument(
            'enable_color_filter',
            default_value='false',
            description='Enable color-based filtering for object detection'
        ),
        DeclareLaunchArgument(
            'enable_display',
            default_value='false',
            description='Enable OpenCV display window'
        ),
        DeclareLaunchArgument(
            'object_min_area',
            default_value='1000',
            description='Minimum contour area in pixels'
        ),
        DeclareLaunchArgument(
            'object_max_area',
            default_value='50000',
            description='Maximum contour area in pixels'
        ),
        
        DeclareLaunchArgument(
            'use_direct_camera',
            default_value='false',
            description='Use direct camera capture via OpenCV (true) or ROS topics (false)'
        ),
        DeclareLaunchArgument(
            'use_stereo',
            default_value='false',
            description='Use stereo camera mode (ZED, RealSense, etc.)'
        ),
        DeclareLaunchArgument(
            'camera_device_id',
            default_value='0',
            description='Camera device ID for direct capture mode'
        ),
        
        DeclareLaunchArgument(
            'image_topic',
            default_value='image_rect',
            description='Monocular image topic (for use_direct_camera=false, use_stereo=false)'
        ),
        DeclareLaunchArgument(
            'camera_info_topic',
            default_value='camera_info',
            description='Camera info topic'
        ),
        DeclareLaunchArgument(
            'left_image_topic',
            default_value='left/image_rect_color',
            description='Left image topic (for stereo mode)'
        ),
        DeclareLaunchArgument(
            'right_image_topic',
            default_value='right/image_rect_color',
            description='Right image topic (for stereo mode)'
        ),
        DeclareLaunchArgument(
            'left_camera_info_topic',
            default_value='left/camera_info',
            description='Left camera info topic (for stereo mode)'
        ),
        
        Node(
            package='apriltag_zed_visp',
            executable='apriltag_detector',
            name='apriltag_detector',
            output='screen',
            parameters=[{
                'camera_frame': camera_frame,
                'tag_frame': tag_frame,
                'target_tag_id': target_tag_id,
                'tag_size': tag_size,
                'use_direct_camera': use_direct_camera,
                'use_stereo': use_stereo,
                'camera_device_id': camera_device_id,
                'publish_tf': publish_tf,
                'publish_marker': publish_marker,
                'enable_compensation': enable_compensation,
                'print_precise_pose': print_precise_pose,
                'prefer_calibrated': prefer_calibrated,
                'filter_window': filter_window,
                'filter_alpha': filter_alpha,
                'base_tag_id_0': base_tag_id_0,
                'base_tag_id_1': base_tag_id_1,
                'enable_relative_pose': enable_relative_pose,
                'enable_undistort': True,
                'enable_object_detection': enable_object_detection,
                'enable_color_filter': enable_color_filter,
                'enable_display': enable_display,
                'object_min_area': object_min_area,
                'object_max_area': object_max_area,
                'image_topic': image_topic,
                'camera_info_topic': camera_info_topic,
                'left_image_topic': left_image_topic,
                'right_image_topic': right_image_topic,
                'left_camera_info_topic': left_camera_info_topic,
            }]
        )
    ])