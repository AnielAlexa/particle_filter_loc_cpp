#!/usr/bin python3
"""ROS2 node: resize mono8 image to the matcher's native input size.

Subscribes to raw camera images (1280x720 mono8) and resizes directly to
output_width x output_height (default 512x288, matching the LiteSAM
288x512 engine). The source aspect ratio (16:9) matches the target, so no
crop is needed — a plain resize preserves scale.
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
        self.declare_parameter("output_width", 512)
        self.declare_parameter("output_height", 288)
        self.declare_parameter("subsample", 2)

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        self.output_width = self.get_parameter("output_width").value
        self.output_height = self.get_parameter("output_height").value
        self.subsample = self.get_parameter("subsample").value
        self.frame_count = 0

        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
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
            f"Resize node: {input_topic} -> {output_topic} "
            f"({self.output_width}x{self.output_height}, "
            f"subsample={self.subsample})"
        )

    def callback(self, msg: Image):
        self.frame_count += 1
        if self.frame_count % self.subsample != 0:
            return

        h = msg.height
        w = msg.width

        if msg.encoding == "mono8":
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w)
        else:
            self.get_logger().warn(
                f"Expected mono8, got {msg.encoding}; skipping", once=True
            )
            return

        out_w = self.output_width
        out_h = self.output_height
        if w != out_w or h != out_h:
            resized = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = img

        out_msg = Image()
        out_msg.header = msg.header
        out_msg.height = out_h
        out_msg.width = out_w
        out_msg.encoding = "mono8"
        out_msg.is_bigendian = 0
        out_msg.step = out_w
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
