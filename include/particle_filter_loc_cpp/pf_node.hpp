#pragma once

#include <atomic>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>

#include <builtin_interfaces/msg/time.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/range.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/u_int8.hpp>

#include <optional>

#include "config.hpp"
#include "geo_utils.hpp"
#include "observation_model.hpp"
#include "particle_filter.hpp"
#include "qr_init_detector.hpp"
#include "trust_model.hpp"
#include "vio_motion_model.hpp"

namespace pf {

class PFGeoLocNode : public rclcpp::Node {
public:
    explicit PFGeoLocNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

private:
    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
    void rtk_callback(const sensor_msgs::msg::NavSatFix::ConstSharedPtr& msg);
    void yaw_callback(const std_msgs::msg::Float64::ConstSharedPtr& msg);
    void alt_callback(const sensor_msgs::msg::Range::ConstSharedPtr& msg);
    void vio_pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr& msg);
    void try_initialize(float median_alt);
    void try_lock_yaw();
    void try_qr_init(const uint8_t* mono_data, int width, int height);
    void process_frame(const uint8_t* mono_data, int width, int height,
                       const builtin_interfaces::msg::Time& stamp);

    // Config
    FullConfig cfg_;
    ENUFrame enu_;

    // Components
    std::unique_ptr<ParticleFilter> pf_;
    std::unique_ptr<ObservationModel> obs_;
    std::unique_ptr<TrustTracker> trust_;
    std::unique_ptr<QrInitDetector> qr_init_;
    VIOMotionModel vio_motion_;

    // QR init stability buffers (filled in image_callback while !initialized_).
    std::deque<double> qr_lat_buf_;
    std::deque<double> qr_lon_buf_;
    std::deque<double> qr_yaw_buf_;
    // Locked QR pose: captured on the ground once samples are stable; used at
    // takeoff (altitude > init_altitude_m) to seed the PF.
    bool qr_locked_ = false;
    double qr_lock_lat_ = 0.0;
    double qr_lock_lon_ = 0.0;
    double qr_lock_yaw_compass_deg_ = 0.0;

    // ROS
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_image_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr sub_rtk_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr sub_yaw_;
    rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr sub_alt_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_vio_pose_;

    rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr pub_position_;
    rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr pub_vio_position_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pub_ess_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_state_;
    // Bridge-facing local odometry: takeoff-relative ENU pose for the MAVLink
    // bridge (or any REP-103 ROS consumer).
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_local_;
    // Latched origin (lat/lon at takeoff, from QR) and reset_counter — both
    // transient_local QoS so a late-starting bridge gets the most recent.
    rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr pub_origin_;
    rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr pub_reset_counter_;
    uint8_t reset_counter_ = 0;

    // Preflight status (JSON over std_msgs/String) — feeds the phone UI.
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_preflight_;
    rclcpp::TimerBase::SharedPtr preflight_timer_;
    void publish_preflight();
    // Per-frame snapshot of the last QR decode (cleared each image callback
    // before try_qr_init runs, so the UI sees "no QR" when a frame had none).
    bool   last_qr_seen_ = false;
    double last_qr_lat_ = 0.0;
    double last_qr_lon_ = 0.0;
    double last_qr_yaw_compass_deg_ = 0.0;
    double last_qr_side_px_ = 0.0;
    // Last compass yaw published to /odom_local (updated each process_frame
    // after init). This is the value the bridge converts to NED yaw.
    double last_pub_yaw_compass_deg_ = 0.0;

    // VIO→ENU mapping (cached at init for dead-reckoned VIO trajectory publication)
    double vio_ref_x_ = 0.0, vio_ref_y_ = 0.0;
    double vio_align_cos_ = 1.0, vio_align_sin_ = 0.0;
    double init_enu_e_ = 0.0, init_enu_n_ = 0.0;

    // State
    std::deque<float> altitude_buf_;
    std::atomic<bool> processing_{false};
    std::mutex processing_mutex_;
    bool initialized_ = false;
    bool first_init_camera_ = true;
    int frame_count_ = 0;
    double gt_lat_ = 0, gt_lon_ = 0;
    bool has_gt_ = false;
    std::optional<double> last_rtk_yaw_deg_;

    // Latest VIO sample cache (used for init alignment and delta predict)
    bool has_vio_ = false;
    double last_vio_x_ = 0.0, last_vio_y_ = 0.0, last_vio_yaw_deg_ = 0.0;
    int64_t last_vio_ts_ns_ = 0;

    // Pre-takeoff yaw-lock ring buffers (see config.pf.yaw_lock_*)
    std::deque<double> yaw_lock_vio_buf_;
    std::deque<double> yaw_lock_rtk_buf_;
    bool yaw_locked_pretakeoff_ = false;
    double locked_enu_yaw_compass_deg_ = 0.0;  // stored for seeding particles at init

    // Stats
    struct FineStats {
        int satellite_ok = 0, satellite_tried = 0;
        int mosaic_ok = 0, mosaic_tried = 0;
        int patch_ok = 0, patch_tried = 0;
        int fine_frames = 0;      // frames where fine was attempted
        int coarse_only = 0;      // frames with only coarse
        int skip_coarse = 0;      // fine frames that skipped coarse
        int total_sat_inliers = 0, total_mosaic_inliers = 0, total_patch_inliers = 0;
    } stats_;
    rclcpp::TimerBase::SharedPtr stats_timer_;
    void print_stats();

    // Diagnostic CSV
    std::ofstream diag_csv_;
    void init_diag_csv();
    void log_diag_row(const builtin_interfaces::msg::Time& stamp,
                      int phase, double est_e, double est_n, double est_h,
                      double spread, double ess,
                      double top1_sim, int n_coarse_candidates,
                      const std::string& fine_source, int fine_inliers,
                      double fine_corr_dist, float flow_consistency,
                      float flow_mag_cv, float inlier_ratio,
                      double bias_x, double bias_y,
                      double rtk_lat, double rtk_lon,
                      double pnp_altitude, double baro_altitude,
                      const std::string& fine_method,
                      double fine_lat = 0.0, double fine_lon = 0.0,
                      int sat_inliers = 0, double sat_lat = 0.0, double sat_lon = 0.0, const std::string& sat_method = "",
                      int mos_inliers = 0, double mos_lat = 0.0, double mos_lon = 0.0, const std::string& mos_method = "");
};

}  // namespace pf
