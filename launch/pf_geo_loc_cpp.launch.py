import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('particle_filter_loc_cpp')
    default_config = os.path.join(pkg_share, 'config', 'pf_config.yaml')

    config_arg = DeclareLaunchArgument(
        'config_path',
        default_value=default_config,
        description='Path to pf_config.yaml'
    )

    node = Node(
        package='particle_filter_loc_cpp',
        executable='pf_geo_loc_node',
        name='pf_geo_loc_node',
        parameters=[{'config_path': LaunchConfiguration('config_path')}],
        output='screen',
        additional_env={
            # Force OpenCV 4.12 to resolve before system 4.5d (cv_bridge dependency)
            'LD_PRELOAD': '/usr/local/lib/libopencv_imgproc.so.412'
                          ':/usr/local/lib/libopencv_imgcodecs.so.412'
                          ':/usr/local/lib/libopencv_highgui.so.412'
                          ':/usr/local/lib/libopencv_core.so.412',
        },
    )

    return LaunchDescription([config_arg, node])
