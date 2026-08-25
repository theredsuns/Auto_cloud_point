from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    
    camera_frame = LaunchConfiguration('camera_frame')
    tag_frame = LaunchConfiguration('tag_frame')
    target_tag_id = LaunchConfiguration('target_tag_id')
    tag_size = LaunchConfiguration('tag_size')
    use_direct_camera = LaunchConfiguration('use_direct_camera')
    publish_tf = LaunchConfiguration('publish_tf')
    publish_marker = LaunchConfiguration('publish_marker')
    enable_compensation = LaunchConfiguration('enable_compensation')
    print_precise_pose = LaunchConfiguration('print_precise_pose')
    camera_model = LaunchConfiguration('camera_model')
    zed_node_name = LaunchConfiguration('zed_node_name')
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_frame',
            default_value='zed_left_camera_frame',
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
            'use_direct_camera',
            default_value='false',
            description='Use ZED ROS topics instead of direct camera capture'
        ),
        DeclareLaunchArgument(
            'camera_model',
            default_value='zed2',
            description='ZED camera model: zed, zedm, zed2, zed2i, zedx, zedxm, virtual'
        ),
        DeclareLaunchArgument(
            'zed_node_name',
            default_value='zed_node',
            description='ZED node name'
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
                'use_stereo': True,
                'publish_tf': publish_tf,
                'publish_marker': publish_marker,
                'enable_compensation': enable_compensation,
                'print_precise_pose': print_precise_pose,
            }],
            remappings=[
                ('zed/left/image_rect_color', ['/', zed_node_name, '/left/image_rect_color']),
                ('zed/right/image_rect_color', ['/', zed_node_name, '/right/image_rect_color']),
                ('zed/left/camera_info', ['/', zed_node_name, '/left/camera_info']),
            ]
        )
    ])
