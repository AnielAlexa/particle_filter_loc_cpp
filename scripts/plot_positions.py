#!/usr/bin/python3.10
"""
Plot RTK ground truth vs PF estimated positions on a satellite patch mosaic.
Subscribes to /m300/rtk/fix and /pf_geo_loc/position, saves plot every N seconds.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, PoseStamped

import numpy as np
import cv2
import os
import json
import time
import sys
import argparse
import yaml

MODEL_DIR = "/home/jetson/ros2_ws/src/particle_filter_loc_cpp/model"
OUTPUT_PATH = "/home/jetson/ros2_ws/src/particle_filter_loc_cpp/results/pf_trajectory.png"
CSV_PATH = "/home/jetson/ros2_ws/src/particle_filter_loc_cpp/results/pf_results.csv"
BIN_DIR = "/home/jetson/ros2_ws/src/particle_filter_loc_cpp/scripts/ardupilot_bins"
CONFIG_PATH = "/home/jetson/ros2_ws/src/particle_filter_loc_cpp/config/pf_config.yaml"
METERS_PER_DEG = 111319.5


def load_gps_from_bin(bin_path):
    """Read GPS messages from an ArduPilot .bin log. Returns [(lat, lon, t_sec), ...]."""
    from pymavlink import mavutil
    m = mavutil.mavlink_connection(bin_path)
    out = []
    while True:
        msg = m.recv_match(type=["GPS"], blocking=False)
        if msg is None:
            break
        # Status 3 = 3D fix, 4 = 3D + DGPS, 5 = RTK float, 6 = RTK fixed
        if getattr(msg, "Status", 0) < 3:
            continue
        lat = float(msg.Lat)
        lon = float(msg.Lng)
        if lat == 0.0 and lon == 0.0:
            continue
        t = float(msg.TimeUS) * 1e-6  # seconds since boot
        out.append((lat, lon, t))
    return out


class PositionPlotter(Node):
    def __init__(self, bin_path=None):
        super().__init__("position_plotter")

        # Storage
        self.rtk_positions = []   # [(lat, lon, time)]
        self.pf_positions = []    # [(lat, lon, time)]
        self.vio_positions = []   # [(lat, lon, time)]
        self.fc_positions = []    # [(lat, lon, time)] from /pf_geo_loc/fc_local_position (NED)
        self.pf_state = "UNINIT"
        self.rtk_source = "topic"

        # Load enu_origin from config for NED→lat/lon conversion
        try:
            with open(CONFIG_PATH) as f:
                cfg = yaml.safe_load(f)
            self.origin_lat = float(cfg["enu_origin"]["lat"])
            self.origin_lon = float(cfg["enu_origin"]["lon"])
            self.get_logger().info(
                f"Loaded enu_origin: lat={self.origin_lat:.6f}, lon={self.origin_lon:.6f}")
        except Exception as e:
            self.get_logger().warn(f"Could not load enu_origin from {CONFIG_PATH}: {e}")
            self.origin_lat = None
            self.origin_lon = None

        # Optional: preload RTK from ArduPilot .bin log
        if bin_path:
            self.get_logger().info(f"Loading GPS from {bin_path} ...")
            self.rtk_positions = load_gps_from_bin(bin_path)
            self.rtk_source = f"bin:{os.path.basename(bin_path)}"
            self.get_logger().info(f"Loaded {len(self.rtk_positions)} GPS fixes from log")

        # Load GPS metadata for patch boundaries
        gps_path = os.path.join(MODEL_DIR, "data/patches/gps_metadata.json")
        with open(gps_path) as f:
            self.gps_meta = json.load(f)

        # Compute patch coverage bounds
        lats = [v["lat"] for v in self.gps_meta.values()]
        lons = [v["lon"] for v in self.gps_meta.values()]
        self.bounds = {
            "min_lat": min(v["bounds"]["min_lat"] for v in self.gps_meta.values()),
            "max_lat": max(v["bounds"]["max_lat"] for v in self.gps_meta.values()),
            "min_lon": min(v["bounds"]["min_lon"] for v in self.gps_meta.values()),
            "max_lon": max(v["bounds"]["max_lon"] for v in self.gps_meta.values()),
        }
        self.get_logger().info(
            f"Patch bounds: lat=[{self.bounds['min_lat']:.6f}, {self.bounds['max_lat']:.6f}] "
            f"lon=[{self.bounds['min_lon']:.6f}, {self.bounds['max_lon']:.6f}]")

        # Build background mosaic from patches
        self.bg_img, self.img_bounds = self._build_mosaic()

        # Subscriptions
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=10)

        self.sub_rtk = self.create_subscription(
            NavSatFix, "/m300/rtk/fix", self.rtk_cb, qos_reliable)
        self.sub_pf = self.create_subscription(
            NavSatFix, "/pf_geo_loc/position", self.pf_cb, qos_reliable)
        self.sub_vio = self.create_subscription(
            NavSatFix, "/pf_geo_loc/vio_position", self.vio_cb, qos_reliable)
        self.sub_state = self.create_subscription(
            String, "/pf_geo_loc/state", self.state_cb, qos_reliable)
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10)
        # Topic type varies across bags: PointStamped on some, PoseStamped on others.
        # Probe the publisher type at runtime, then create the matching subscription.
        self._fc_topic = "/pf_geo_loc/fc_local_position"
        self._fc_qos = qos_best_effort
        self.sub_fc = None
        self._fc_probe_timer = self.create_timer(0.5, self._probe_fc_type)

        # Ensure output dir
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

        # Log counter
        self.log_timer = self.create_timer(5.0, self.log_status)

        # Live display timer
        self.display_timer = self.create_timer(1.0, self.update_display)

        self.get_logger().info(f"Plotter ready. Ctrl+C to save final image to {OUTPUT_PATH}")

    def _build_mosaic(self):
        """Build a background mosaic image from satellite patches."""
        b = self.bounds
        # Add margin
        margin = 0.001  # ~100m
        min_lat = b["min_lat"] - margin
        max_lat = b["max_lat"] + margin
        min_lon = b["min_lon"] - margin
        max_lon = b["max_lon"] + margin

        # Target image size
        img_h, img_w = 1200, 1200
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon

        bg = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        bg[:] = (40, 40, 40)  # dark gray background

        patches_dir = os.path.join(MODEL_DIR, "data/patches")
        loaded = 0
        for name, meta in self.gps_meta.items():
            pb = meta["bounds"]
            patch_path = os.path.join(patches_dir, f"{name}.png")
            if not os.path.exists(patch_path):
                continue

            # Map patch bounds to pixel coords
            x0 = int((pb["min_lon"] - min_lon) / lon_range * img_w)
            x1 = int((pb["max_lon"] - min_lon) / lon_range * img_w)
            y0 = int((max_lat - pb["max_lat"]) / lat_range * img_h)
            y1 = int((max_lat - pb["min_lat"]) / lat_range * img_h)

            if x1 <= x0 or y1 <= y0:
                continue
            if x0 >= img_w or y0 >= img_h or x1 <= 0 or y1 <= 0:
                continue

            patch_img = cv2.imread(patch_path)
            if patch_img is None:
                continue

            resized = cv2.resize(patch_img, (x1 - x0, y1 - y0))

            # Clip to image bounds
            sx0 = max(0, -x0)
            sy0 = max(0, -y0)
            dx0 = max(0, x0)
            dy0 = max(0, y0)
            sx1 = min(resized.shape[1], img_w - x0)
            sy1 = min(resized.shape[0], img_h - y0)

            if sx1 > sx0 and sy1 > sy0:
                bg[dy0:dy0 + sy1 - sy0, dx0:dx0 + sx1 - sx0] = resized[sy0:sy1, sx0:sx1]
                loaded += 1

        self.get_logger().info(f"Mosaic built: {loaded} patches, {img_w}x{img_h}")
        return bg, {"min_lat": min_lat, "max_lat": max_lat,
                     "min_lon": min_lon, "max_lon": max_lon}

    def _latlon_to_px(self, lat, lon):
        """Convert lat/lon to pixel coordinates on the mosaic."""
        b = self.img_bounds
        h, w = self.bg_img.shape[:2]
        x = int((lon - b["min_lon"]) / (b["max_lon"] - b["min_lon"]) * w)
        y = int((b["max_lat"] - lat) / (b["max_lat"] - b["min_lat"]) * h)
        return x, y

    def rtk_cb(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.rtk_positions.append((msg.latitude, msg.longitude, t))

    def pf_cb(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.pf_positions.append((msg.latitude, msg.longitude, t))

    def vio_cb(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.vio_positions.append((msg.latitude, msg.longitude, t))

    def state_cb(self, msg):
        self.pf_state = msg.data

    def _probe_fc_type(self):
        """Inspect the publisher's type and create a matching subscription once seen."""
        if self.sub_fc is not None:
            return
        infos = self.get_publishers_info_by_topic(self._fc_topic)
        if not infos:
            return
        type_name = infos[0].topic_type  # e.g. 'geometry_msgs/msg/PoseStamped'
        if type_name.endswith("PoseStamped"):
            msg_cls, cb = PoseStamped, self.fc_pose_cb
        elif type_name.endswith("PointStamped"):
            msg_cls, cb = PointStamped, self.fc_point_cb
        else:
            self.get_logger().warn(
                f"Unsupported type '{type_name}' for {self._fc_topic}; FC trail disabled")
            self._fc_probe_timer.cancel()
            return
        self.sub_fc = self.create_subscription(msg_cls, self._fc_topic, cb, self._fc_qos)
        self.get_logger().info(f"Subscribed to {self._fc_topic} as {type_name}")
        self._fc_probe_timer.cancel()

    def _append_fc_ned(self, north, east, stamp):
        if self.origin_lat is None:
            return
        lat = self.origin_lat + north / METERS_PER_DEG
        lon = self.origin_lon + east / (METERS_PER_DEG * np.cos(np.radians(self.origin_lat)))
        t = stamp.sec + stamp.nanosec * 1e-9
        self.fc_positions.append((lat, lon, t))

    def fc_point_cb(self, msg):
        # NED: x = North, y = East, z = Down
        self._append_fc_ned(msg.point.x, msg.point.y, msg.header.stamp)

    def fc_pose_cb(self, msg):
        self._append_fc_ned(msg.pose.position.x, msg.pose.position.y, msg.header.stamp)

    def log_status(self):
        self.get_logger().info(
            f"Collecting... RTK={len(self.rtk_positions)} PF={len(self.pf_positions)} state={self.pf_state}")

    def update_display(self):
        """Redraw live OpenCV window."""
        if not self.pf_positions and not self.rtk_positions and not self.fc_positions:
            return
        img = self._render()
        cv2.imshow("PF Live Trajectory", img)
        cv2.waitKey(1)

    def _render(self):
        """Render current trajectories onto mosaic background."""
        img = self.bg_img.copy()

        # Draw RTK trail (green)
        if len(self.rtk_positions) >= 2:
            pts = [self._latlon_to_px(lat, lon) for lat, lon, _ in self.rtk_positions]
            for i in range(1, len(pts)):
                cv2.line(img, pts[i - 1], pts[i], (0, 220, 0), 2, cv2.LINE_AA)
            cv2.circle(img, pts[0], 10, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img, "S", (pts[0][0] - 5, pts[0][1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(img, pts[-1], 8, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(img, pts[-1], 8, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw FC ground-truth trail (yellow, from NED /pf_geo_loc/fc_local_position)
        if len(self.fc_positions) >= 2:
            pts = [self._latlon_to_px(lat, lon) for lat, lon, _ in self.fc_positions]
            for i in range(1, len(pts)):
                cv2.line(img, pts[i - 1], pts[i], (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(img, pts[0], 10, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(img, pts[-1], 7, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, pts[-1], 7, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw VIO trail (blue, dead-reckoned)
        if len(self.vio_positions) >= 2:
            pts = [self._latlon_to_px(lat, lon) for lat, lon, _ in self.vio_positions]
            for i in range(1, len(pts)):
                cv2.line(img, pts[i - 1], pts[i], (255, 120, 0), 2, cv2.LINE_AA)
            cv2.circle(img, pts[-1], 6, (255, 120, 0), -1, cv2.LINE_AA)
            cv2.circle(img, pts[-1], 6, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw PF trail (red)
        if len(self.pf_positions) >= 2:
            pts = [self._latlon_to_px(lat, lon) for lat, lon, _ in self.pf_positions]
            for i in range(1, len(pts)):
                cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(img, pts[-1], 8, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(img, pts[-1], 8, (255, 255, 255), 2, cv2.LINE_AA)

        # Compute errors
        errors = self._compute_errors()

        # Legend
        y0 = 30
        cv2.putText(img, f"RTK ({self.rtk_source})", (15, y0), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 220, 0), 2, cv2.LINE_AA)
        cv2.putText(img, "PF (estimated)", (15, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "VIO (dead-reckoned)", (15, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 120, 0), 2, cv2.LINE_AA)
        cv2.putText(img, "FC NED (ground truth, error ref)", (15, y0 + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"State: {self.pf_state}", (15, y0 + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, f"RTK:{len(self.rtk_positions)} PF:{len(self.pf_positions)} "
                    f"VIO:{len(self.vio_positions)} FC:{len(self.fc_positions)}",
                    (15, y0 + 125), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)

        if errors:
            median_err = np.median(errors)
            mean_err = np.mean(errors)
            max_err = np.max(errors)
            cv2.putText(img, f"Median error: {median_err:.1f}m", (15, y0 + 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f"Mean: {mean_err:.1f}m | Max: {max_err:.1f}m", (15, y0 + 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        return img

    def _compute_errors(self):
        """PF error against FC ground truth (NED-derived), time-nearest match."""
        errors = []
        if self.fc_positions and self.pf_positions:
            gt_arr = np.array([(lat, lon) for lat, lon, _ in self.fc_positions])
            gt_times = np.array([t for _, _, t in self.fc_positions])
            for plat, plon, pt in self.pf_positions:
                idx = np.argmin(np.abs(gt_times - pt))
                glat, glon = gt_arr[idx]
                dlat = (plat - glat) * METERS_PER_DEG
                dlon = (plon - glon) * METERS_PER_DEG * np.cos(np.radians(glat))
                errors.append(np.sqrt(dlat ** 2 + dlon ** 2))
        return errors

    def save_plot(self):
        if not self.pf_positions and not self.rtk_positions:
            self.get_logger().warn("No positions collected, nothing to save")
            return

        img = self._render()
        cv2.imwrite(OUTPUT_PATH, img)

        # Save CSV: PF aligned to FC (ground truth) and nearest RTK, with PF-vs-FC error
        with open(CSV_PATH, "w") as f:
            f.write("time,fc_lat,fc_lon,rtk_lat,rtk_lon,pf_lat,pf_lon,error_m\n")
            if self.pf_positions:
                have_fc = len(self.fc_positions) > 0
                have_rtk = len(self.rtk_positions) > 0
                if have_fc:
                    fc_times = np.array([t for _, _, t in self.fc_positions])
                    fc_arr = np.array([(lat, lon) for lat, lon, _ in self.fc_positions])
                if have_rtk:
                    rtk_times = np.array([t for _, _, t in self.rtk_positions])
                    rtk_arr = np.array([(lat, lon) for lat, lon, _ in self.rtk_positions])
                for plat, plon, pt in self.pf_positions:
                    if have_fc:
                        gi = int(np.argmin(np.abs(fc_times - pt)))
                        glat, glon = fc_arr[gi]
                        dlat = (plat - glat) * METERS_PER_DEG
                        dlon = (plon - glon) * METERS_PER_DEG * np.cos(np.radians(glat))
                        err = float(np.sqrt(dlat ** 2 + dlon ** 2))
                        fc_lat_s, fc_lon_s = f"{glat:.8f}", f"{glon:.8f}"
                        err_s = f"{err:.2f}"
                    else:
                        fc_lat_s, fc_lon_s, err_s = "", "", ""
                    if have_rtk:
                        ri = int(np.argmin(np.abs(rtk_times - pt)))
                        rlat, rlon = rtk_arr[ri]
                        rtk_lat_s, rtk_lon_s = f"{rlat:.8f}", f"{rlon:.8f}"
                    else:
                        rtk_lat_s, rtk_lon_s = "", ""
                    f.write(f"{pt:.3f},{fc_lat_s},{fc_lon_s},{rtk_lat_s},{rtk_lon_s},"
                            f"{plat:.8f},{plon:.8f},{err_s}\n")

        self.get_logger().info(f"Saved {OUTPUT_PATH} and {CSV_PATH}")


def main(args=None):
    parser = argparse.ArgumentParser(description="Plot RTK/PF/VIO trajectories. Optional ArduPilot .bin for RTK source.")
    parser.add_argument("bin", nargs="?", default=None,
                        help="Path to ArduPilot .bin log (optional, replaces /m300/rtk/fix as RTK source)")
    cli_args, ros_args = parser.parse_known_args(args=sys.argv[1:] if args is None else args)

    bin_path = cli_args.bin
    if bin_path and not os.path.isabs(bin_path) and not os.path.exists(bin_path):
        bin_path = os.path.join(BIN_DIR, bin_path)

    rclpy.init(args=ros_args)
    node = PositionPlotter(bin_path=bin_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Final save
        node.save_plot()
        cv2.destroyAllWindows()
        node.get_logger().info(f"Final plot saved to {OUTPUT_PATH}")
        node.get_logger().info(f"CSV saved to {CSV_PATH}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
