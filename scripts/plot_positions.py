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

import numpy as np
import cv2
import os
import json
import time

MODEL_DIR = "/home/jetson/particle_filter_ws/src/particle_filter_loc_cpp/model"
OUTPUT_PATH = "/home/jetson/particle_filter_ws/src/particle_filter_loc_cpp/results/pf_trajectory.png"
CSV_PATH = "/home/jetson/particle_filter_ws/src/particle_filter_loc_cpp/results/pf_results.csv"


class PositionPlotter(Node):
    def __init__(self):
        super().__init__("position_plotter")

        # Storage
        self.rtk_positions = []   # [(lat, lon, time)]
        self.pf_positions = []    # [(lat, lon, time)]
        self.vio_positions = []   # [(lat, lon, time)]
        self.pf_state = "UNINIT"

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

    def log_status(self):
        self.get_logger().info(
            f"Collecting... RTK={len(self.rtk_positions)} PF={len(self.pf_positions)} state={self.pf_state}")

    def update_display(self):
        """Redraw live OpenCV window."""
        if not self.pf_positions and not self.rtk_positions:
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
        cv2.putText(img, "RTK (ground truth)", (15, y0), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 220, 0), 2, cv2.LINE_AA)
        cv2.putText(img, "PF (estimated)", (15, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "VIO (dead-reckoned)", (15, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 120, 0), 2, cv2.LINE_AA)
        cv2.putText(img, f"State: {self.pf_state}", (15, y0 + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, f"RTK:{len(self.rtk_positions)} PF:{len(self.pf_positions)} VIO:{len(self.vio_positions)}",
                    (15, y0 + 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)

        if errors:
            median_err = np.median(errors)
            mean_err = np.mean(errors)
            max_err = np.max(errors)
            cv2.putText(img, f"Median error: {median_err:.1f}m", (15, y0 + 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f"Mean: {mean_err:.1f}m | Max: {max_err:.1f}m", (15, y0 + 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        return img

    def _compute_errors(self):
        errors = []
        if self.rtk_positions and self.pf_positions:
            rtk_arr = np.array([(lat, lon) for lat, lon, _ in self.rtk_positions])
            rtk_times = np.array([t for _, _, t in self.rtk_positions])
            for plat, plon, pt in self.pf_positions:
                idx = np.argmin(np.abs(rtk_times - pt))
                rlat, rlon = rtk_arr[idx]
                dlat = (plat - rlat) * 111320
                dlon = (plon - rlon) * 111320 * np.cos(np.radians(rlat))
                errors.append(np.sqrt(dlat ** 2 + dlon ** 2))
        return errors

    def save_plot(self):
        if not self.pf_positions and not self.rtk_positions:
            self.get_logger().warn("No positions collected, nothing to save")
            return

        img = self._render()
        cv2.imwrite(OUTPUT_PATH, img)

        # Save CSV
        with open(CSV_PATH, "w") as f:
            f.write("time,rtk_lat,rtk_lon,pf_lat,pf_lon,error_m\n")
            if errors:
                rtk_times = np.array([t for _, _, t in self.rtk_positions])
                rtk_arr = np.array([(lat, lon) for lat, lon, _ in self.rtk_positions])
                for i, (plat, plon, pt) in enumerate(self.pf_positions):
                    idx = np.argmin(np.abs(rtk_times - pt))
                    rlat, rlon = rtk_arr[idx]
                    f.write(f"{pt:.3f},{rlat:.8f},{rlon:.8f},{plat:.8f},{plon:.8f},{errors[i]:.2f}\n")

        self.get_logger().info(f"Saved {OUTPUT_PATH} and {CSV_PATH}")


def main(args=None):
    rclpy.init(args=args)
    node = PositionPlotter()
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
