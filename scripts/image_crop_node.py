#!/usr/bin/env python3
"""ROS2 node: center-crop mono8 image to square, resize to target size.

Subscribes to raw camera images (e.g. 1280x720 mono8), center-crops to the
largest inscribed square (720x720), then resizes to output_size (default 320x320).
Publishes the result on a separate topic for downstream nodes that expect
square, pre-cropped input.

Usage:
    python3 image_crop_node.py
    python3 image_crop_node.py --ros-args -p input_topic:=/camera/image_mono \
                                          -p output_topic:=/camera/image_rect \
                                          -p output_size:=320
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
import numpy as np
import cv2


class ImageCropNode(Node):
    def __init__(self):
        super().__init__("image_crop_node")

        self.declare_parameter("input_topic", "/camera/image_mono")
        self.declare_parameter("output_topic", "/camera/image_rect")
        self.declare_parameter("output_size", 320)
        self.declare_parameter("subsample", 2)

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        self.output_size = self.get_parameter("output_size").value
        self.subsample = self.get_parameter("subsample").value
        self.frame_count = 0

        # Subscribe reliable (matches bag replay QoS)
        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # Publish best_effort (matches PF node QoS)
        qos_pub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.pub = self.create_publisher(Image, output_topic, qos_pub)
        self.sub = self.create_subscription(
            Image, input_topic, self.callback, qos_sub
        )

        self.get_logger().info(
            f"Crop node: {input_topic} -> {output_topic} "
            f"(crop to square, resize {self.output_size}x{self.output_size}, "
            f"subsample={self.subsample})"
        )

    def callback(self, msg: Image):
        self.frame_count += 1
        if self.frame_count % self.subsample != 0:
            return

        h = msg.height
        w = msg.width

        # Decode mono8 from flat buffer
        if msg.encoding == "mono8":
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w)
        else:
            self.get_logger().warn(
                f"Expected mono8, got {msg.encoding}; skipping", once=True
            )
            return

        # Center-crop to largest inscribed square
        s = min(h, w)
        x0 = (w - s) // 2
        y0 = (h - s) // 2
        cropped = img[y0 : y0 + s, x0 : x0 + s]

        # Resize to output_size x output_size
        out_size = self.output_size
        if s != out_size:
            resized = cv2.resize(
                cropped, (out_size, out_size), interpolation=cv2.INTER_LINEAR
            )
        else:
            resized = cropped

        # Build output message, preserving original timestamp
        out_msg = Image()
        out_msg.header = msg.header
        out_msg.height = out_size
        out_msg.width = out_size
        out_msg.encoding = "mono8"
        out_msg.is_bigendian = 0
        out_msg.step = out_size
        out_msg.data = resized.tobytes()

        self.pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImageCropNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
