from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    
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
    
    apriltag_share = get_package_share_directory('apriltag_zed_visp')
    
    return LaunchDescription([
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
            default_value='true',
            description='Enable object contour detection'
        ),
        DeclareLaunchArgument(
            'enable_color_filter',
            default_value='false',
            description='Enable color-based filtering for object detection'
        ),
        DeclareLaunchArgument(
            'enable_display',
            default_value='true',
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
        
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='realsense2_camera',
            output='screen',
            parameters=[{
                'enable_color': True,
                'enable_depth': True,
                'enable_infra1': False,
                'enable_infra2': False,
                'color_width': 1920,
                'color_height': 1080,
                'color_fps': 30,
                'depth_width': 1280,
                'depth_height': 720,
                'depth_fps': 30,
            }]
        ),
        
        Node(
            package='apriltag_zed_visp',
            executable='apriltag_detector',
            name='apriltag_detector',
            output='screen',
            parameters=[{
                'camera_frame': 'camera_color_optical_frame',
                'tag_frame': 'tag_36h11_id2',
                'target_tag_id': target_tag_id,
                'tag_size': tag_size,
                'use_direct_camera': False,
                'use_stereo': False,
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
                'image_topic': '/camera/color/image_raw',
                'camera_info_topic': '/camera/color/camera_info',
            }]
        )
    ])