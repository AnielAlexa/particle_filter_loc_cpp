"""Full stack launch: camera + VIO + particle filter + MAVLink bridge.

Boot order:
  t=0    camera + bridge  — camera publishes frames, bridge opens UART and
                            starts HEARTBEAT exchange with the FC.
  t=10   VIO              — gives camera time to settle before VIO starts
                            consuming frames.
  t=20   PF               — starts after VIO is producing /ov_srvins/poseimu
                            (PF's QR lock waits for VIO anyway).

Override anything via launch args, e.g.:
  ros2 launch particle_filter_loc_cpp full_stack.launch.py \
      uart_device:=/dev/ttyUSB0 baudrate:=115200
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pf_share     = get_package_share_directory('particle_filter_loc_cpp')
    bridge_share = get_package_share_directory('pf_mavlink_bridge')
    vio_share    = get_package_share_directory('ov_srvins')

    default_pf_config     = os.path.join(pf_share, 'config', 'pf_config.yaml')
    default_bridge_config = os.path.join(bridge_share, 'config', 'bridge.yaml')
    vio_launch_file       = os.path.join(vio_share, 'launch', 'subscribe.launch.py')

    # ── Launch arguments ──────────────────────────────────────────
    args = [
        DeclareLaunchArgument('pf_config',     default_value=default_pf_config),
        DeclareLaunchArgument('bridge_config', default_value=default_bridge_config),
        DeclareLaunchArgument('uart_device',   default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('baudrate',      default_value='921600'),
        DeclareLaunchArgument('start_camera',  default_value='true'),
        DeclareLaunchArgument('start_vio',     default_value='true'),
        DeclareLaunchArgument('start_bridge',  default_value='true'),
        DeclareLaunchArgument('pf_delay_s',    default_value='2.0'),
    ]

    # ── 1. MAVLink bridge — first so latched topics get delivered ──
    bridge = Node(
        package='pf_mavlink_bridge',
        executable='pf_mavlink_bridge_node',
        name='pf_mavlink_bridge',
        parameters=[
            LaunchConfiguration('bridge_config'),
            {
                'device':   LaunchConfiguration('uart_device'),
                'baudrate': LaunchConfiguration('baudrate'),
            },
        ],
        output='screen',
    )

    # ── 2. Camera ──────────────────────────────────────────────────
    camera = Node(
        package='camera_display_node',
        executable='camera_display_node',
        name='camera_display_node',
        output='screen',
    )

    # ── 3. VIO (include its existing launch) — delayed +10 s ───────
    vio = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(vio_launch_file),
    )
    vio_delayed = TimerAction(period=10.0, actions=[vio])

    # ── 4. Particle filter — delayed +20 s (10 s after VIO) ────────
    pf = Node(
        package='particle_filter_loc_cpp',
        executable='pf_geo_loc_node',
        name='pf_geo_loc_node',
        parameters=[{'config_path': LaunchConfiguration('pf_config')}],
        output='screen',
        additional_env={
            'LD_PRELOAD': '/usr/local/lib/libopencv_imgproc.so.412'
                          ':/usr/local/lib/libopencv_imgcodecs.so.412'
                          ':/usr/local/lib/libopencv_core.so.412',
        },
    )
    pf_delayed = TimerAction(period=20.0, actions=[pf])

    return LaunchDescription(args + [
        bridge,        # t=0
        camera,        # t=0
        vio_delayed,   # t=10
        pf_delayed,    # t=20
    ])
