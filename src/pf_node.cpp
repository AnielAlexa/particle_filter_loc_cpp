#include "particle_filter_loc_cpp/pf_node.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>

namespace pf {

PFGeoLocNode::PFGeoLocNode(const rclcpp::NodeOptions& options)
    : Node("pf_geo_loc_node", options)
{
    // Load config (default: installed pf_config.yaml)
    std::string default_config = ament_index_cpp::get_package_share_directory("particle_filter_loc_cpp")
                                 + "/config/pf_config.yaml";
    this->declare_parameter<std::string>("config_path", default_config);
    std::string config_path = this->get_parameter("config_path").as_string();

    cfg_ = load_config(config_path);
    enu_ = ENUFrame(cfg_.enu_origin_lat, cfg_.enu_origin_lon);

    // Initialize components
    pf_ = std::make_unique<ParticleFilter>(cfg_.pf);
    obs_ = std::make_unique<ObservationModel>(cfg_.matchers, enu_);
    trust_ = std::make_unique<TrustTracker>(cfg_.trust);
    qr_init_ = std::make_unique<QrInitDetector>(
        cfg_.camera.heading_offset_deg,
        cfg_.init.qr_min_side_px,
        cfg_.init.qr_max_aspect_skew,
        cfg_.init.qr_phi_offset_deg);
    vio_motion_.set_jump_params(cfg_.pf.vio_jump_threshold_m, cfg_.pf.vio_jump_velocity_ema_alpha);

    // QoS
    rclcpp::QoS qos_sensor(1);
    qos_sensor.best_effort();

    // Subscriptions
    sub_image_ = create_subscription<sensor_msgs::msg::Image>(
        cfg_.replay.camera_topic, qos_sensor,
        std::bind(&PFGeoLocNode::image_callback, this, std::placeholders::_1));

    sub_rtk_ = create_subscription<sensor_msgs::msg::NavSatFix>(
        cfg_.replay.rtk_topic, 10,
        std::bind(&PFGeoLocNode::rtk_callback, this, std::placeholders::_1));

    sub_yaw_ = create_subscription<std_msgs::msg::Float64>(
        cfg_.replay.yaw_topic, 10,
        std::bind(&PFGeoLocNode::yaw_callback, this, std::placeholders::_1));

    sub_alt_ = create_subscription<sensor_msgs::msg::Range>(
        cfg_.replay.altimeter_topic, 10,
        std::bind(&PFGeoLocNode::alt_callback, this, std::placeholders::_1));

    sub_vio_pose_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        cfg_.replay.vio_pose_topic, 50,
        std::bind(&PFGeoLocNode::vio_pose_callback, this, std::placeholders::_1));

    // Publishers
    pub_position_ = create_publisher<sensor_msgs::msg::NavSatFix>("/pf_geo_loc/position", 10);
    pub_vio_position_ = create_publisher<sensor_msgs::msg::NavSatFix>("/pf_geo_loc/vio_position", 10);
    pub_ess_ = create_publisher<std_msgs::msg::Float32>("/pf_geo_loc/ess", 10);
    pub_state_ = create_publisher<std_msgs::msg::String>("/pf_geo_loc/state", 10);
    pub_odom_local_ = create_publisher<nav_msgs::msg::Odometry>("/pf_geo_loc/odom_local", 10);

    rclcpp::QoS latched(rclcpp::KeepLast(1));
    latched.transient_local().reliable();
    pub_origin_ = create_publisher<sensor_msgs::msg::NavSatFix>("/pf_geo_loc/origin", latched);
    pub_reset_counter_ = create_publisher<std_msgs::msg::UInt8>("/pf_geo_loc/reset_counter", latched);

    pub_preflight_ = create_publisher<std_msgs::msg::String>("/pf_geo_loc/preflight_status", 10);
    preflight_timer_ = create_wall_timer(std::chrono::milliseconds(200),
        std::bind(&PFGeoLocNode::publish_preflight, this));

    // Stats timer — print every 10s
    stats_timer_ = create_wall_timer(std::chrono::seconds(10),
        std::bind(&PFGeoLocNode::print_stats, this));

    init_diag_csv();
    RCLCPP_INFO(get_logger(), "PFGeoLocNode (C++) ready.");
}

void PFGeoLocNode::init_diag_csv() {
    std::string path = "/home/jetson/ros2_ws/src/particle_filter_loc_cpp/results/pf_diag.csv";
    diag_csv_.open(path, std::ios::trunc);
    if (diag_csv_.is_open()) {
        diag_csv_ << "timestamp,frame,phase,est_e,est_n,est_h,"
                  << "spread,ess,top1_sim,n_coarse_cand,"
                  << "fine_source,fine_inliers,fine_corr_dist,"
                  << "flow_consistency,flow_mag_cv,inlier_ratio,"
                  << "bias_x,bias_y,"
                  << "rtk_lat,rtk_lon,pnp_alt,baro_alt,fine_method,"
                  << "fine_lat,fine_lon,fine_gt_err,"
                  << "sat_inliers,sat_gt_err,sat_method,"
                  << "mos_inliers,mos_gt_err,mos_method\n";
        RCLCPP_INFO(get_logger(), "Diagnostic CSV: %s", path.c_str());
    }
}

void PFGeoLocNode::log_diag_row(
    const builtin_interfaces::msg::Time& stamp,
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
    double fine_lat, double fine_lon,
    int sat_inliers, double sat_lat, double sat_lon, const std::string& sat_method,
    int mos_inliers, double mos_lat, double mos_lon, const std::string& mos_method)
{
    if (!diag_csv_.is_open()) return;
    double ts = stamp.sec + stamp.nanosec * 1e-9;

    auto gt_err = [&](double lat, double lon) -> double {
        if (lat == 0.0 || rtk_lat == 0.0) return 0.0;
        double cl = std::cos(rtk_lat * M_PI / 180.0);
        double de = (lon - rtk_lon) * cl * 111319.5;
        double dn = (lat - rtk_lat) * 111319.5;
        return std::sqrt(de * de + dn * dn);
    };

    diag_csv_ << std::fixed << std::setprecision(3) << ts << ","
              << frame_count_ << "," << phase << ","
              << std::setprecision(2) << est_e << "," << est_n << ","
              << std::setprecision(1) << est_h << ","
              << spread << "," << std::setprecision(0) << ess << ","
              << std::setprecision(4) << top1_sim << ","
              << n_coarse_candidates << ","
              << fine_source << "," << fine_inliers << ","
              << std::setprecision(2) << fine_corr_dist << ","
              << std::setprecision(3) << flow_consistency << ","
              << flow_mag_cv << "," << inlier_ratio << ","
              << std::setprecision(4) << bias_x << "," << bias_y << ","
              << std::setprecision(8) << rtk_lat << "," << rtk_lon << ","
              << std::setprecision(1) << pnp_altitude << "," << baro_altitude << ","
              << fine_method << ","
              << std::setprecision(8) << fine_lat << "," << fine_lon << ","
              << std::setprecision(2) << gt_err(fine_lat, fine_lon) << ","
              << sat_inliers << "," << std::setprecision(2) << gt_err(sat_lat, sat_lon) << "," << sat_method << ","
              << mos_inliers << "," << std::setprecision(2) << gt_err(mos_lat, mos_lon) << "," << mos_method << "\n";
    diag_csv_.flush();
}

void PFGeoLocNode::publish_preflight() {
    // Emit a tiny JSON-as-String. No third-party JSON dep needed; the consumer
    // is the preflight web UI.
    std::ostringstream os;
    os.setf(std::ios::fixed);

    auto bool_str = [](bool b) { return b ? "true" : "false"; };
    auto num_or_null = [&](double v, bool valid, int prec) {
        if (!valid) { os << "null"; return; }
        os.precision(prec); os << v;
    };

    auto circ_std = [](const std::deque<double>& buf) -> double {
        if (buf.empty()) return 0.0;
        double sx = 0.0, sy = 0.0;
        for (double v : buf) {
            double r = v * M_PI / 180.0;
            sx += std::cos(r); sy += std::sin(r);
        }
        double R = std::sqrt(sx * sx + sy * sy) / static_cast<double>(buf.size());
        double Rc = std::min(std::max(R, 1e-9), 1.0);
        return std::sqrt(-2.0 * std::log(Rc)) * 180.0 / M_PI;
    };
    auto lin_stats = [](const std::deque<double>& buf, double& mean, double& stddev) {
        if (buf.empty()) { mean = 0.0; stddev = 0.0; return; }
        double s = 0.0;
        for (double v : buf) s += v;
        mean = s / static_cast<double>(buf.size());
        double ss = 0.0;
        for (double v : buf) ss += (v - mean) * (v - mean);
        stddev = std::sqrt(ss / static_cast<double>(buf.size()));
    };

    double yaw_std = circ_std(qr_yaw_buf_);
    double pos_std_m = 0.0;
    if (qr_lat_buf_.size() > 1) {
        double lat_mean, lat_std, lon_mean, lon_std;
        lin_stats(qr_lat_buf_, lat_mean, lat_std);
        lin_stats(qr_lon_buf_, lon_mean, lon_std);
        pos_std_m = std::hypot(
            lat_std * 111319.5,
            lon_std * 111319.5 * std::cos(lat_mean * M_PI / 180.0));
    }

    const int target_n = std::max(1, cfg_.init.qr_stable_n);
    const int have_n   = static_cast<int>(qr_lat_buf_.size());
    const bool yaw_aligned = vio_motion_.is_yaw_locked();

    os << "{";
    os << "\"qr_detected\":" << bool_str(last_qr_seen_) << ",";
    os << "\"qr_last_lat\":";          num_or_null(last_qr_lat_,            last_qr_seen_, 7); os << ",";
    os << "\"qr_last_lon\":";          num_or_null(last_qr_lon_,            last_qr_seen_, 7); os << ",";
    os << "\"qr_last_yaw_compass_deg\":"; num_or_null(last_qr_yaw_compass_deg_, last_qr_seen_, 2); os << ",";
    os << "\"qr_lock_samples\":[" << have_n << "," << target_n << "],";
    os.precision(3); os << "\"qr_yaw_stddev_deg\":" << yaw_std << ",";
    os.precision(4); os << "\"qr_pos_stddev_m\":"   << pos_std_m << ",";
    os << "\"qr_locked\":" << bool_str(qr_locked_) << ",";
    os << "\"qr_lock_lat\":";          num_or_null(qr_lock_lat_,            qr_locked_, 7); os << ",";
    os << "\"qr_lock_lon\":";          num_or_null(qr_lock_lon_,            qr_locked_, 7); os << ",";
    os << "\"qr_lock_yaw_compass_deg\":"; num_or_null(qr_lock_yaw_compass_deg_, qr_locked_, 2); os << ",";
    os << "\"yaw_aligned\":" << bool_str(yaw_aligned) << ",";
    // After init we report the live PF heading (what the bridge will send to
    // the FC). Pre-init but post-yaw-lock, the locked QR yaw is the alignment.
    {
        bool have_aligned = yaw_aligned;
        double aligned = initialized_ ? last_pub_yaw_compass_deg_ : qr_lock_yaw_compass_deg_;
        os << "\"aligned_yaw_compass_deg\":";
        num_or_null(aligned, have_aligned, 2);
        os << ",";
    }
    os << "\"vio_seen\":"      << bool_str(has_vio_) << ",";
    os << "\"altimeter_seen\":"<< bool_str(!altitude_buf_.empty()) << ",";
    os << "\"initialized\":"   << bool_str(initialized_) << ",";
    os << "\"phase\":\""       << phase_name(pf_->phase()) << "\"";
    os << "}";

    std_msgs::msg::String msg;
    msg.data = os.str();
    pub_preflight_->publish(msg);
}

void PFGeoLocNode::print_stats() {
    if (frame_count_ == 0) return;
    auto& s = stats_;
    RCLCPP_INFO(get_logger(),
        "STATS | frames=%d fine=%d coarse_only=%d | "
        "sat=%d/%d (avg_inl=%.1f) mosaic=%d/%d (avg_inl=%.1f) patch=%d/%d (avg_inl=%.1f) | "
        "skip_coarse=%d",
        frame_count_, s.fine_frames, s.coarse_only,
        s.satellite_ok, s.satellite_tried,
        s.satellite_ok > 0 ? (double)s.total_sat_inliers / s.satellite_ok : 0.0,
        s.mosaic_ok, s.mosaic_tried,
        s.mosaic_ok > 0 ? (double)s.total_mosaic_inliers / s.mosaic_ok : 0.0,
        s.patch_ok, s.patch_tried,
        s.patch_ok > 0 ? (double)s.total_patch_inliers / s.patch_ok : 0.0,
        s.skip_coarse);

    // Append to stats CSV
    std::string stats_path = "/home/jetson/ros2_ws/src/particle_filter_loc_cpp/results/pf_stats.csv";
    bool file_exists = std::ifstream(stats_path).good();
    std::ofstream f(stats_path, std::ios::app);
    if (f.is_open()) {
        if (!file_exists) {
            f << "mosaic_ctx_scale,frames,fine_frames,coarse_only,"
              << "sat_ok,sat_tried,sat_avg_inl,"
              << "mosaic_ok,mosaic_tried,mosaic_avg_inl,"
              << "patch_ok,patch_tried,patch_avg_inl,"
              << "skip_coarse\n";
        }
        f << cfg_.matchers.mosaic_context_scale << ","
          << frame_count_ << "," << s.fine_frames << "," << s.coarse_only << ","
          << s.satellite_ok << "," << s.satellite_tried << ","
          << (s.satellite_ok > 0 ? (double)s.total_sat_inliers / s.satellite_ok : 0.0) << ","
          << s.mosaic_ok << "," << s.mosaic_tried << ","
          << (s.mosaic_ok > 0 ? (double)s.total_mosaic_inliers / s.mosaic_ok : 0.0) << ","
          << s.patch_ok << "," << s.patch_tried << ","
          << (s.patch_ok > 0 ? (double)s.total_patch_inliers / s.patch_ok : 0.0) << ","
          << s.skip_coarse << "\n";
    }
}

void PFGeoLocNode::alt_callback(const sensor_msgs::msg::Range::ConstSharedPtr& msg) {
    altitude_buf_.push_back(msg->range);
    if (altitude_buf_.size() > 10) altitude_buf_.pop_front();
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
        "DBG alt_cb: range=%.2f buf_size=%zu initialized=%d",
        msg->range, altitude_buf_.size(), initialized_);

    // Init is now driven by QR lock in try_qr_init; altitude_buf_ is still
    // consumed by process_frame to set obs.altitude_m for fine-match gating.
}

void PFGeoLocNode::try_lock_yaw() {
    // Pre-takeoff yaw lock against RTK is no longer used: with sky-lign QR
    // init, the QR provides world-aligned yaw directly. Kept as a stub so
    // existing callers compile; remove once vio_pose_callback is rewritten.
    return;

    if (yaw_locked_pretakeoff_ || initialized_) return;
    if (!has_vio_ || !last_rtk_yaw_deg_.has_value()) return;
    if (cfg_.pf.init_yaw != 0.0) return;  // fixed-config yaw: no need to lock from RTK

    // Require ground-level altitude to avoid locking mid-climb
    if (!altitude_buf_.empty()) {
        std::vector<float> sorted(altitude_buf_.begin(), altitude_buf_.end());
        std::sort(sorted.begin(), sorted.end());
        float median_alt = sorted[sorted.size() / 2];
        if (median_alt > cfg_.pf.yaw_lock_max_altitude_m) return;
    }

    // Skip zero-ish samples (VIO can emit identity on first frames)
    if (std::abs(last_vio_yaw_deg_) < 1e-6 && std::abs(last_vio_x_) < 1e-9
        && std::abs(last_vio_y_) < 1e-9) return;

    yaw_lock_vio_buf_.push_back(last_vio_yaw_deg_);
    yaw_lock_rtk_buf_.push_back(*last_rtk_yaw_deg_);
    size_t cap = static_cast<size_t>(cfg_.pf.yaw_lock_buffer_size);
    while (yaw_lock_vio_buf_.size() > cap) yaw_lock_vio_buf_.pop_front();
    while (yaw_lock_rtk_buf_.size() > cap) yaw_lock_rtk_buf_.pop_front();
    if (yaw_lock_vio_buf_.size() < cap) return;

    // Circular stats on each buffer (handles wrap at 0/360)
    auto circ_stats = [](const std::deque<double>& buf, double& mean_deg, double& stddev_deg) {
        double sx = 0.0, sy = 0.0;
        for (double v : buf) {
            double r = v * M_PI / 180.0;
            sx += std::cos(r);
            sy += std::sin(r);
        }
        double n = static_cast<double>(buf.size());
        double R = std::sqrt(sx * sx + sy * sy) / n;
        double m = std::atan2(sy, sx) * 180.0 / M_PI;
        if (m < 0.0) m += 360.0;
        mean_deg = m;
        // R clamped to avoid log(0); stddev via circular formula
        double R_clamped = std::min(std::max(R, 1e-9), 1.0);
        stddev_deg = std::sqrt(-2.0 * std::log(R_clamped)) * 180.0 / M_PI;
    };

    double vio_mean, vio_std, rtk_mean, rtk_std;
    circ_stats(yaw_lock_vio_buf_, vio_mean, vio_std);
    circ_stats(yaw_lock_rtk_buf_, rtk_mean, rtk_std);

    if (vio_std > cfg_.pf.yaw_lock_max_stddev_deg ||
        rtk_std > cfg_.pf.yaw_lock_max_stddev_deg) {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
            "Yaw lock waiting: vio_std=%.2f rtk_std=%.2f (thr=%.2f)",
            vio_std, rtk_std, cfg_.pf.yaw_lock_max_stddev_deg);
        return;
    }

    double vio_yaw_adj = wrap360(vio_mean + cfg_.pf.vio_yaw_offset_deg);
    vio_motion_.lock_yaw(vio_yaw_adj, rtk_mean);
    locked_enu_yaw_compass_deg_ = rtk_mean;
    yaw_locked_pretakeoff_ = true;
    RCLCPP_INFO(get_logger(),
        "YAW LOCKED pre-takeoff: vio_mean=%.2f (std=%.2f) rtk_mean=%.2f (std=%.2f) "
        "-> align_theta=%.2f",
        vio_mean, vio_std, rtk_mean, rtk_std, vio_motion_.align_theta_deg());
}

namespace {

void circ_stats_deg(const std::deque<double>& buf, double& mean_deg, double& stddev_deg) {
    double sx = 0.0, sy = 0.0;
    for (double v : buf) {
        double r = v * M_PI / 180.0;
        sx += std::cos(r);
        sy += std::sin(r);
    }
    double n = static_cast<double>(buf.size());
    double R = std::sqrt(sx * sx + sy * sy) / n;
    double m = std::atan2(sy, sx) * 180.0 / M_PI;
    if (m < 0.0) m += 360.0;
    mean_deg = m;
    double R_clamped = std::min(std::max(R, 1e-9), 1.0);
    stddev_deg = std::sqrt(-2.0 * std::log(R_clamped)) * 180.0 / M_PI;
}

void mean_stddev(const std::deque<double>& buf, double& mean, double& stddev) {
    double s = 0.0;
    for (double v : buf) s += v;
    mean = s / buf.size();
    double ss = 0.0;
    for (double v : buf) ss += (v - mean) * (v - mean);
    stddev = std::sqrt(ss / buf.size());
}

}  // namespace

void PFGeoLocNode::try_qr_init(const uint8_t* mono_data, int width, int height) {
    if (initialized_ || qr_locked_) return;

    // Wrap the ROS-owned mono8 buffer (no copy).
    cv::Mat mono(height, width, CV_8UC1, const_cast<uint8_t*>(mono_data));
    auto qr = qr_init_->detect(mono);

    // Cache last detection (or absence of one) for the preflight UI.
    last_qr_seen_ = qr.has_value();
    if (qr.has_value()) {
        last_qr_lat_ = qr->lat;
        last_qr_lon_ = qr->lon;
        last_qr_yaw_compass_deg_ = qr->yaw_compass_deg;
        last_qr_side_px_ = qr->side_px;
    }

    if (!qr.has_value()) {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
            "QR init: no valid QR (need geo:LAT,LON payload, drone above iPhone)");
        return;
    }

    qr_lat_buf_.push_back(qr->lat);
    qr_lon_buf_.push_back(qr->lon);
    qr_yaw_buf_.push_back(qr->yaw_compass_deg);
    const size_t cap = static_cast<size_t>(std::max(1, cfg_.init.qr_stable_n));
    while (qr_lat_buf_.size() > cap) qr_lat_buf_.pop_front();
    while (qr_lon_buf_.size() > cap) qr_lon_buf_.pop_front();
    while (qr_yaw_buf_.size() > cap) qr_yaw_buf_.pop_front();

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
        "QR lock: lat=%.7f lon=%.7f yaw=%.2f side=%.0fpx (%zu/%zu samples)",
        qr->lat, qr->lon, qr->yaw_compass_deg, qr->side_px,
        qr_lat_buf_.size(), cap);

    if (qr_lat_buf_.size() < cap) return;

    if (!has_vio_) {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
            "QR lock: stable QR, waiting for VIO pose to lock yaw");
        return;
    }

    double yaw_mean, yaw_std;
    circ_stats_deg(qr_yaw_buf_, yaw_mean, yaw_std);
    if (yaw_std > cfg_.init.qr_yaw_max_stddev_deg) {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
            "QR lock: yaw not stable (std=%.2f, thr=%.2f)",
            yaw_std, cfg_.init.qr_yaw_max_stddev_deg);
        return;
    }

    double lat_mean, lat_std, lon_mean, lon_std;
    mean_stddev(qr_lat_buf_, lat_mean, lat_std);
    mean_stddev(qr_lon_buf_, lon_mean, lon_std);
    double pos_std_m = std::hypot(
        lat_std * 111319.5,
        lon_std * 111319.5 * std::cos(lat_mean * M_PI / 180.0));
    if (pos_std_m > cfg_.init.qr_pos_max_stddev_m) {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
            "QR lock: position not stable (std=%.3f m, thr=%.3f)",
            pos_std_m, cfg_.init.qr_pos_max_stddev_m);
        return;
    }

    qr_lock_lat_ = lat_mean;
    qr_lock_lon_ = lon_mean;
    qr_lock_yaw_compass_deg_ = yaw_mean;
    qr_locked_ = true;

    // Full init at QR lock: yaw + position alignment + PF seed. Predict loop
    // (in vio_pose_callback) starts running immediately; fine matching stays
    // gated by altitude inside process_frame.
    double vio_yaw_adj = wrap360(last_vio_yaw_deg_ + cfg_.pf.vio_yaw_offset_deg);
    vio_motion_.lock_yaw(vio_yaw_adj, yaw_mean);
    vio_motion_.pin_position(last_vio_x_, last_vio_y_);

    auto [e, n] = enu_.wgs84_to_enu(lat_mean, lon_mean);
    pf_->seed_from_position(e, n, yaw_mean,
        cfg_.pf.init_sigma_pos, cfg_.pf.init_sigma_hdg);

    vio_ref_x_ = last_vio_x_;
    vio_ref_y_ = last_vio_y_;
    double alpha = vio_motion_.align_theta_deg() * M_PI / 180.0;
    vio_align_cos_ = std::cos(alpha);
    vio_align_sin_ = std::sin(alpha);
    init_enu_e_ = e;
    init_enu_n_ = n;

    initialized_ = true;
    reset_counter_ = static_cast<uint8_t>(reset_counter_ + 1);

    // Latched origin + reset_counter for the MAVLink bridge.
    {
        sensor_msgs::msg::NavSatFix origin_msg;
        origin_msg.header.stamp = now();
        origin_msg.header.frame_id = "odom";
        origin_msg.latitude = qr_lock_lat_;
        origin_msg.longitude = qr_lock_lon_;
        origin_msg.altitude = 0.0;
        origin_msg.status.status = sensor_msgs::msg::NavSatStatus::STATUS_FIX;
        pub_origin_->publish(origin_msg);

        std_msgs::msg::UInt8 rc_msg;
        rc_msg.data = reset_counter_;
        pub_reset_counter_->publish(rc_msg);
    }

    RCLCPP_INFO(get_logger(),
        "INIT (QR lock on ground): lat=%.7f lon=%.7f -> ENU(%.2f, %.2f) | "
        "yaw=%.2f deg (std=%.2f, pos_std=%.3fm) | VIO ref=(%.2f, %.2f) vio_yaw=%.1f "
        "-> align_theta=%.1f | reset_counter=%u",
        lat_mean, lon_mean, e, n, yaw_mean, yaw_std, pos_std_m,
        last_vio_x_, last_vio_y_, last_vio_yaw_deg_,
        vio_motion_.align_theta_deg(),
        static_cast<unsigned>(reset_counter_));
}

void PFGeoLocNode::try_initialize(float /*median_alt*/) {
    // Legacy altimeter-driven init path. Init is now performed in try_qr_init
    // at the QR-lock event; this stub is kept only to satisfy the existing
    // header declaration and any leftover callers.
}

void PFGeoLocNode::yaw_callback(const std_msgs::msg::Float64::ConstSharedPtr& msg) {
    last_rtk_yaw_deg_ = msg->data * 10.0;
}

void PFGeoLocNode::rtk_callback(const sensor_msgs::msg::NavSatFix::ConstSharedPtr& msg) {
    gt_lat_ = msg->latitude;
    gt_lon_ = msg->longitude;
    has_gt_ = true;
}

void PFGeoLocNode::vio_pose_callback(
    const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr& msg) {
    int64_t ts_ns = static_cast<int64_t>(msg->header.stamp.sec) * 1000000000LL +
                    msg->header.stamp.nanosec;
    const auto& p = msg->pose.pose.position;
    const auto& q = msg->pose.pose.orientation;
    double yaw_deg = quat_to_yaw_deg(q.x, q.y, q.z, q.w);

    last_vio_x_ = p.x;
    last_vio_y_ = p.y;
    last_vio_yaw_deg_ = yaw_deg;
    last_vio_ts_ns_ = ts_ns;
    has_vio_ = true;

    try_lock_yaw();

    if (initialized_ && vio_motion_.is_aligned() && pf_->phase() != Phase::UNINIT) {
        double yaw_adj = wrap360(yaw_deg + cfg_.pf.vio_yaw_offset_deg);
        auto delta = vio_motion_.update(ts_ns, p.x, p.y, yaw_adj);
        if (delta.has_value()) {
            pf_->predict(*delta);
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                "[VIO] raw=(%.2f, %.2f, yaw_math=%.1f) enu_delta=(%+.3f, %+.3f) "
                "hdg_compass=%.1f rtk_compass=%.1f",
                p.x, p.y, yaw_deg, delta->dx_m, delta->dy_m,
                delta->heading_deg,
                last_rtk_yaw_deg_.value_or(-1.0));
        }

        // Publish dead-reckoned VIO position in WGS84 for plotting/diagnostics
        double dx_v = p.x - vio_ref_x_;
        double dy_v = p.y - vio_ref_y_;
        double enu_e = init_enu_e_ + vio_align_cos_ * dx_v - vio_align_sin_ * dy_v;
        double enu_n = init_enu_n_ + vio_align_sin_ * dx_v + vio_align_cos_ * dy_v;
        auto [vlat, vlon] = enu_.enu_to_wgs84(enu_e, enu_n);
        sensor_msgs::msg::NavSatFix vio_msg;
        vio_msg.header.stamp = msg->header.stamp;
        vio_msg.header.frame_id = "vio_aligned";
        vio_msg.latitude = vlat;
        vio_msg.longitude = vlon;
        vio_msg.altitude = 0.0;
        pub_vio_position_->publish(vio_msg);
    }
}

void PFGeoLocNode::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    // Drop frame if still processing previous
    if (processing_.exchange(true)) {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 3000,
            "DBG img_cb: still processing, dropping frame");
        return;
    }

    RCLCPP_INFO(get_logger(), "DBG img_cb: got frame %dx%d enc=%s",
        msg->width, msg->height, msg->encoding.c_str());

    try {
        // Get mono8 data directly from ROS message (zero-copy if possible)
        const uint8_t* data;
        int width, height;
        cv_bridge::CvImageConstPtr cv_ptr;  // keeps converted buffer alive

        if (msg->encoding == "mono8") {
            data = msg->data.data();
            width = msg->width;
            height = msg->height;
        } else {
            cv_ptr = cv_bridge::toCvShare(msg, "mono8");
            data = cv_ptr->image.data;
            width = cv_ptr->image.cols;
            height = cv_ptr->image.rows;
        }

        if (!initialized_) {
            // Reset before detection so a frame with no QR shows up as
            // "not detected this frame" in /pf_geo_loc/preflight_status.
            last_qr_seen_ = false;
            try_qr_init(data, width, height);
            processing_.store(false);
            return;
        }

        process_frame(data, width, height, msg->header.stamp);
    } catch (const std::exception& e) {
        RCLCPP_WARN(get_logger(), "Frame processing error: %s", e.what());
    }

    processing_.store(false);
}

void PFGeoLocNode::process_frame(const uint8_t* mono_data, int width, int height,
                                 const builtin_interfaces::msg::Time& stamp) {
    auto& pf = *pf_;
    auto& obs = *obs_;

    frame_count_++;
    RCLCPP_INFO(get_logger(), "DBG process_frame F%d: phase=%s %dx%d",
        frame_count_, phase_name(pf.phase()), width, height);

    // Check first few pixel values to verify data is not blank
    if (frame_count_ <= 3) {
        int sum = 0;
        for (int i = 0; i < std::min(100, width * height); ++i) sum += mono_data[i];
        RCLCPP_INFO(get_logger(), "DBG pixel check: avg(first100)=%.1f", sum / 100.0);
    }

    if (pf.phase() == Phase::UNINIT) return;

    // Update altitude
    if (!altitude_buf_.empty()) {
        std::vector<float> sorted(altitude_buf_.begin(), altitude_buf_.end());
        std::sort(sorted.begin(), sorted.end());
        obs.altitude_m = sorted[sorted.size() / 2];
    }

    auto [est_e, est_n, est_h] = pf.estimate();
    double search_radius = pf.get_search_radius();
    bool should_fine = pf.should_run_fine();
    bool fine_succeeded = false;

    // Diagnostic accumulators
    double diag_top1_sim = 0.0;
    int diag_n_coarse_cand = 0;
    std::string diag_fine_source = "none";
    std::string diag_fine_method = "none";
    int diag_fine_inliers = 0;
    double diag_corr_dist = 0.0;
    double diag_pnp_alt = 0.0;
    double diag_fine_lat = 0.0, diag_fine_lon = 0.0;
    float diag_flow_con = 0.0f, diag_flow_cv = 0.0f, diag_inl_ratio = 0.0f;
    // Per-stage bench
    int diag_sat_inl = 0; double diag_sat_lat = 0, diag_sat_lon = 0; std::string diag_sat_method;
    int diag_mos_inl = 0; double diag_mos_lat = 0, diag_mos_lon = 0; std::string diag_mos_method;

    // ── Fine frames: try reconstruction-based matching first (no coarse needed) ──
    if (should_fine) {
        stats_.fine_frames++;
        const int early_exit = cfg_.pf.fine_early_exit_inliers;
        std::optional<FineResult> best_fine;

        auto [rlat, rlon] = enu_.enu_to_wgs84(est_e, est_n);
        double heading_for_recon = est_h + cfg_.camera.heading_offset_deg
                                         + cfg_.camera.recon_extra_heading_deg;

        // Build footprint reconstruction (adaptive context scale from particle spread)
        double adaptive_sat_ctx = pf.get_satellite_context_scale(
            cfg_.matchers.satellite_context_scale_min,
            cfg_.matchers.satellite_context_scale_max);
        std::optional<FootprintReconstruction> fp_recon;
        if (obs.altitude_m > 20.0) {
            try {
                fp_recon = obs.footprint().reconstruct(
                    rlat, rlon, obs.altitude_m, heading_for_recon,
                    cfg_.matchers.camera_fx, cfg_.matchers.camera_fy,
                    cfg_.matchers.matcher_width, cfg_.matchers.matcher_height,
                    cv::Size(cfg_.matchers.matcher_width, cfg_.matchers.matcher_height),
                    cfg_.matchers.mosaic_context_scale,
                    adaptive_sat_ctx);
            } catch (const std::exception& e) {
                RCLCPP_WARN(get_logger(), "Recon exception: %s", e.what());
            }
        }

        // Debug viz: drone image vs reconstructed satellite crop (toggle via debug.viz_drone_vs_sat)
        if (cfg_.debug.viz_drone_vs_sat) {
            RCLCPP_INFO_ONCE(get_logger(), "debug viz enabled (DISPLAY=%s)",
                             std::getenv("DISPLAY") ? std::getenv("DISPLAY") : "(unset)");
        }
        if (cfg_.debug.viz_drone_vs_sat && fp_recon.has_value() && !fp_recon->satellite_crop.empty()) {
            int W = cfg_.matchers.matcher_width;
            int H = cfg_.matchers.matcher_height;
            cv::Mat drone(height, width, CV_8UC1, const_cast<uint8_t*>(mono_data));
            cv::Mat drone_rs, sat_rs;
            cv::resize(drone, drone_rs, cv::Size(W, H));
            cv::resize(fp_recon->satellite_crop, sat_rs, cv::Size(W, H));
            cv::Mat drone_bgr;
            cv::cvtColor(drone_rs, drone_bgr, cv::COLOR_GRAY2BGR);
            if (sat_rs.channels() == 1) cv::cvtColor(sat_rs, sat_rs, cv::COLOR_GRAY2BGR);
            cv::Mat side_by_side;
            cv::hconcat(drone_bgr, sat_rs, side_by_side);
            cv::putText(side_by_side, "drone", {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 2);
            cv::putText(side_by_side, "sat_recon", {W + 10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 2);
            cv::imshow("pf_debug: drone | satellite_recon", side_by_side);
            cv::waitKey(1);
        }

        // Stage A: Satellite perspective-warped (PnP only, no homography)
        if (fp_recon.has_value() && !fp_recon->satellite_crop.empty() && !fp_recon->warp_M_inv.empty()) {
            stats_.satellite_tried++;
            auto sat_fine = obs.fine_match_on_satellite(
                mono_data, width, height,
                fp_recon->satellite_crop,
                fp_recon->mosaic_meta,
                fp_recon->warp_M_inv);

            // Refinement: if primary used wide context and produced a hit, re-match
            // at context_min centered on the candidate for an accurate position.
            // Normal match — no extra gates; downstream pipeline continues to judge it.
            const double ctx_min = cfg_.matchers.satellite_context_scale_min;
            if (sat_fine.has_value() && adaptive_sat_ctx > ctx_min + 0.05) {
                double refine_heading = sat_fine->heading_deg.value_or(est_h)
                                        + cfg_.camera.heading_offset_deg
                                        + cfg_.camera.recon_extra_heading_deg;
                try {
                    auto refine_recon = obs.footprint().reconstruct(
                        sat_fine->lat, sat_fine->lon, obs.altitude_m, refine_heading,
                        cfg_.matchers.camera_fx, cfg_.matchers.camera_fy,
                        cfg_.matchers.matcher_width, cfg_.matchers.matcher_height,
                        cv::Size(cfg_.matchers.matcher_width, cfg_.matchers.matcher_height),
                        cfg_.matchers.mosaic_context_scale,
                        ctx_min);
                    if (refine_recon.has_value()
                        && !refine_recon->satellite_crop.empty()
                        && !refine_recon->warp_M_inv.empty()) {
                        auto refined = obs.fine_match_on_satellite(
                            mono_data, width, height,
                            refine_recon->satellite_crop,
                            refine_recon->mosaic_meta,
                            refine_recon->warp_M_inv);
                        if (refined.has_value()) {
                            RCLCPP_INFO(get_logger(),
                                "F%d sat refine ctx %.2f->%.2f inl %d->%d",
                                frame_count_, adaptive_sat_ctx, ctx_min,
                                sat_fine->inliers, refined->inliers);
                            sat_fine = refined;
                        }
                    }
                } catch (const std::exception& e) {
                    RCLCPP_WARN(get_logger(), "sat refine exception: %s", e.what());
                }
            }

            if (sat_fine.has_value()) {
                stats_.satellite_ok++;
                stats_.total_sat_inliers += sat_fine->inliers;
                if (!best_fine || sat_fine->inliers > best_fine->inliers)
                    best_fine = sat_fine;
            }
        }
        // Stage B: Mosaic disabled — satellite only

        // Upstream gates: reject bad matches before they can poison the voting cluster.
        // 1. PnP altitude gate: reject if PnP altitude disagrees with barometer.
        // 2. Inlier-ratio gate: correct matches have median ratio ~0.70, wrong ~0.51 —
        //    floor kills a chunk of surviving wrong matches that altitude alone can't.
        auto altitude_gate_ok = [&](const auto& fine) -> bool {
            if (fine.method == "pnp" && fine.pnp_altitude > 0.0 && obs.altitude_m > 0.0) {
                double alt_delta = std::abs(fine.pnp_altitude - obs.altitude_m);
                if (alt_delta > cfg_.pf.pnp_altitude_gate_m) {
                    RCLCPP_WARN(get_logger(), "F%d PnP alt gate: pnp=%.1f baro=%.1f delta=%.1f > %.1f — rejected",
                        frame_count_, fine.pnp_altitude, obs.altitude_m, alt_delta, cfg_.pf.pnp_altitude_gate_m);
                    return false;
                }
            }
            if (cfg_.pf.min_inlier_ratio > 0.0f && fine.inlier_ratio > 0.0f &&
                fine.inlier_ratio < cfg_.pf.min_inlier_ratio) {
                RCLCPP_WARN(get_logger(), "F%d inlier_ratio gate: %.2f < %.2f — rejected",
                    frame_count_, fine.inlier_ratio, cfg_.pf.min_inlier_ratio);
                return false;
            }
            return true;
        };

        // If satellite succeeded with enough inliers, apply and skip coarse
        if (best_fine.has_value() && best_fine->inliers >= early_exit && altitude_gate_ok(*best_fine)) {
            auto [fe, fn] = enu_.wgs84_to_enu(best_fine->lat, best_fine->lon);
            double corr_dist = std::sqrt((fe - est_e) * (fe - est_e) + (fn - est_n) * (fn - est_n));
            (void)corr_dist;  // logged in CSV
            RCLCPP_INFO(get_logger(), "F%d fine %s inliers=%d %s — skip coarse (drift=%.2f)",
                frame_count_, best_fine->patch_name.c_str(), best_fine->inliers, best_fine->method.c_str(),
                corr_dist);
            bool accepted = pf.update_fine(fe, fn, best_fine->inliers, best_fine->heading_deg,
                          std::nullopt, std::nullopt, best_fine->method, obs.altitude_m,
                          best_fine->flow_magnitude_cv);
            fine_succeeded = accepted;
            stats_.skip_coarse++;
            diag_fine_source = accepted ? best_fine->patch_name : ("rej:" + best_fine->patch_name);
            diag_fine_method = best_fine->method;
            diag_fine_inliers = accepted ? best_fine->inliers : -best_fine->inliers;
            diag_corr_dist = corr_dist;
            diag_pnp_alt = best_fine->pnp_altitude;
            diag_fine_lat = best_fine->lat;
            diag_fine_lon = best_fine->lon;
            diag_flow_con = best_fine->flow_consistency;
            diag_flow_cv = best_fine->flow_magnitude_cv;
            diag_inl_ratio = best_fine->inlier_ratio;
        }

        // Stage C disabled: coarse + single-patch fallback had ~0% success in practice.
        // Coarse is still used for LOST recovery below.
        if (false) {
            // Run coarse match
            const std::vector<int>* candidates = nullptr;
            std::vector<int> candidate_vec;
            if (pf.phase() != Phase::LOST) {
                candidate_vec = obs.get_indices_within_radius(est_e, est_n, search_radius);
                if (candidate_vec.size() < 5)
                    candidate_vec = obs.get_indices_within_radius(est_e, est_n, search_radius * 2.0);
                candidates = &candidate_vec;
            }
            auto coarse = obs.coarse_match(mono_data, width, height, candidates, cfg_.pf.top_k_coarse);

            // Coarse weight update
            std::vector<std::tuple<double, double, float>> coarse_obs;
            for (size_t i = 0; i < coarse.top_k_names.size(); ++i) {
                auto [e, n] = obs.get_patch_center_enu(coarse.top_k_names[i]);
                coarse_obs.emplace_back(e, n, coarse.top_k_sims[i]);
            }
            if (pf.phase() == Phase::LOST)
                pf.update_coarse(coarse_obs);
            if (!coarse.top_k_sims.empty())
                diag_top1_sim = coarse.top_k_sims[0];
            diag_n_coarse_cand = static_cast<int>(coarse.top_k_names.size());

            // Single patch fine (stage C)
            if (!coarse.top_k_names.empty()) {
                stats_.patch_tried++;
                double ctx_frac = pf.get_context_fraction();
                int fine_top_k = pf.get_fine_top_k();
                for (int i = 0; i < fine_top_k && i < static_cast<int>(coarse.top_k_names.size()); ++i) {
                    auto patch_fine = obs.fine_match(mono_data, width, height,
                                                      coarse.top_k_names[i], ctx_frac);
                    if (patch_fine.has_value()) {
                        if (!best_fine || patch_fine->inliers > best_fine->inliers)
                            best_fine = patch_fine;
                        if (best_fine->inliers >= early_exit) break;
                    }
                }
            }

            // Apply best fine result (from any stage)
            if (best_fine.has_value() && altitude_gate_ok(*best_fine)) {
                // Track which stage won
                if (best_fine->patch_name != "satellite")
                    { stats_.patch_ok++; stats_.total_patch_inliers += best_fine->inliers; }

                auto [fe, fn] = enu_.wgs84_to_enu(best_fine->lat, best_fine->lon);
                double corr_dist = std::sqrt((fe - est_e) * (fe - est_e) + (fn - est_n) * (fn - est_n));
                (void)corr_dist;  // logged in CSV
                RCLCPP_INFO(get_logger(), "F%d fine %s inliers=%d %s + coarse (drift=%.2f)",
                    frame_count_, best_fine->patch_name.c_str(), best_fine->inliers, best_fine->method.c_str(),
                    corr_dist);
                bool accepted = pf.update_fine(fe, fn, best_fine->inliers, best_fine->heading_deg,
                              std::nullopt, std::nullopt, best_fine->method, obs.altitude_m,
                              best_fine->flow_magnitude_cv);
                diag_fine_source = accepted ? best_fine->patch_name : ("rej:" + best_fine->patch_name);
                diag_fine_method = best_fine->method;
                diag_fine_inliers = accepted ? best_fine->inliers : -best_fine->inliers;
                diag_corr_dist = corr_dist;
                diag_pnp_alt = best_fine->pnp_altitude;
                diag_fine_lat = best_fine->lat;
                diag_fine_lon = best_fine->lon;
                diag_flow_con = best_fine->flow_consistency;
                diag_flow_cv = best_fine->flow_magnitude_cv;
                diag_inl_ratio = best_fine->inlier_ratio;
            }
        }
    }

    // ── Non-fine frames: coast on RTK predict only (no coarse) ──
    if (!should_fine) {
        stats_.coarse_only++;
    }

    pf.resample_if_needed();
    pf.check_transitions();

    // ── LOST recovery: coarse(extended) → fine → double verify → re-inject ──
    if (pf.phase() == Phase::LOST && should_fine && !fine_succeeded && cfg_.matchers.coarse_enabled) {
        RCLCPP_WARN(get_logger(), "F%d LOST recovery: running extended coarse+fine", frame_count_);

        // Extended search: no radius filter (full DB)
        auto coarse = obs.coarse_match(mono_data, width, height, nullptr, 1);
        if (!coarse.top_k_names.empty() && coarse.top_k_sims[0] >= 0.3f) {
            // Fine match on top-1 coarse result
            double ctx_frac = pf.get_context_fraction();
            auto primary_fine = obs.fine_match(mono_data, width, height,
                                                coarse.top_k_names[0], ctx_frac);

            if (primary_fine.has_value())
            {
                // Double verification: reconstruct satellite at candidate position, re-match
                // Primary can be weak — verification match must confirm with enough inliers
                double verify_heading = primary_fine->heading_deg.value_or(est_h)
                                        + cfg_.camera.heading_offset_deg
                                        + cfg_.camera.recon_extra_heading_deg;
                bool verified = false;
                double agreement_m = 999.0;

                try {
                    auto verify_recon_opt = obs.footprint().reconstruct(
                        primary_fine->lat, primary_fine->lon, obs.altitude_m, verify_heading,
                        cfg_.matchers.camera_fx, cfg_.matchers.camera_fy,
                        cfg_.matchers.matcher_width, cfg_.matchers.matcher_height,
                        cv::Size(cfg_.matchers.matcher_width, cfg_.matchers.matcher_height),
                        cfg_.matchers.mosaic_context_scale,
                        cfg_.matchers.satellite_context_scale_min);

                    if (verify_recon_opt.has_value() &&
                        !verify_recon_opt->satellite_crop.empty() &&
                        !verify_recon_opt->warp_M_inv.empty()) {
                        auto verify_fine = obs.fine_match_on_satellite(
                            mono_data, width, height,
                            verify_recon_opt->satellite_crop,
                            verify_recon_opt->mosaic_meta,
                            verify_recon_opt->warp_M_inv);

                        if (verify_fine.has_value() &&
                            verify_fine->inliers >= cfg_.pf.lost_recovery_min_inliers) {
                            auto [ve, vn] = enu_.wgs84_to_enu(verify_fine->lat, verify_fine->lon);
                            auto [pe, pn] = enu_.wgs84_to_enu(primary_fine->lat, primary_fine->lon);
                            agreement_m = std::sqrt((ve-pe)*(ve-pe) + (vn-pn)*(vn-pn));
                            verified = agreement_m < cfg_.pf.lost_recovery_verify_agreement_m;
                        }
                    }
                } catch (const std::exception& e) {
                    RCLCPP_WARN(get_logger(), "LOST verify exception: %s", e.what());
                }

                if (verified) {
                    auto [fe, fn] = enu_.wgs84_to_enu(primary_fine->lat, primary_fine->lon);
                    RCLCPP_WARN(get_logger(),
                        "F%d LOST RECOVERED: inliers=%d agreement=%.1fm → re-inject at (%.1f, %.1f)",
                        frame_count_, primary_fine->inliers, agreement_m, fe, fn);
                    pf.apply_global_correction(fe, fn, primary_fine->heading_deg,
                                               0.0, cfg_.pf.lost_recovery_sigma_pos);
                } else {
                    RCLCPP_WARN(get_logger(),
                        "F%d LOST verify failed: inliers=%d agreement=%.1fm",
                        frame_count_, primary_fine->inliers, agreement_m);
                }
            }
        }
    }

    // Bias debug (every 3s)
    auto est_full = pf.estimate_full();
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 3000,
        "[BIAS] bias_x=%.3f bias_y=%.3f spread=%.1f phase=%s",
        est_full.bias_x, est_full.bias_y,
        pf.weighted_spread(), phase_name(pf.phase()));

    // Publish estimate
    auto [pub_e, pub_n, pub_hdg] = pf.estimate();
    auto [pub_lat, pub_lon] = enu_.enu_to_wgs84(pub_e, pub_n);
    last_pub_yaw_compass_deg_ = pub_hdg;

    // Jump detection: a fine-match update that yanks the estimate by more
    // than `pf.jump_reset_distance_m` (or yaw by `pf.jump_reset_yaw_deg`)
    // signals a re-localization, not a smooth correction. Bump reset_counter
    // so AP's EKF re-anchors instead of treating it as a 20m glitch.
    double dx_jump = pub_e - est_e;
    double dy_jump = pub_n - est_n;
    double dist_jump = std::sqrt(dx_jump * dx_jump + dy_jump * dy_jump);
    double dyaw_jump = std::abs(wrap360(pub_hdg - est_h + 180.0) - 180.0);
    if (dist_jump > cfg_.pf.jump_reset_distance_m ||
        dyaw_jump > cfg_.pf.jump_reset_yaw_deg) {
        reset_counter_ = static_cast<uint8_t>(reset_counter_ + 1);
        std_msgs::msg::UInt8 rc_msg;
        rc_msg.data = reset_counter_;
        pub_reset_counter_->publish(rc_msg);
        RCLCPP_WARN(get_logger(),
            "JUMP detected (matcher relocalization): Δpos=%.2f m, Δyaw=%.2f deg "
            "→ reset_counter=%u",
            dist_jump, dyaw_jump, static_cast<unsigned>(reset_counter_));
    }

    sensor_msgs::msg::NavSatFix pos_msg;
    pos_msg.header.stamp = stamp;  // use bag timestamp for correct time alignment
    pos_msg.header.frame_id = "gps_rtk_antenna";
    pos_msg.latitude = pub_lat;
    pos_msg.longitude = pub_lon;
    pub_position_->publish(pos_msg);

    // Local odometry (REP-103 ENU/FLU) for MAVLink bridge consumption.
    {
        // Weighted variance of east, north, yaw over particles.
        const auto& P = pf.particles();
        const auto& W = pf.weights();
        double mean_e = 0.0, mean_n = 0.0;
        double sx = 0.0, sy = 0.0;
        for (int i = 0; i < P.rows(); ++i) {
            mean_e += W(i) * P(i, COL_X);
            mean_n += W(i) * P(i, COL_Y);
            double r = P(i, COL_YAW) * M_PI / 180.0;
            sx += W(i) * std::cos(r);
            sy += W(i) * std::sin(r);
        }
        double var_e = 0.0, var_n = 0.0;
        for (int i = 0; i < P.rows(); ++i) {
            double de = P(i, COL_X) - mean_e;
            double dn = P(i, COL_Y) - mean_n;
            var_e += W(i) * de * de;
            var_n += W(i) * dn * dn;
        }
        double R = std::sqrt(sx * sx + sy * sy);
        double R_clamped = std::min(std::max(R, 1e-9), 1.0);
        double var_yaw_rad = -2.0 * std::log(R_clamped);  // circular variance (rad²)

        // Compass yaw → ROS ENU yaw (REP-103: 0=East, +CCW)
        double yaw_compass_rad = pub_hdg * M_PI / 180.0;
        double yaw_enu = M_PI_2 - yaw_compass_rad;

        nav_msgs::msg::Odometry odom;
        odom.header.stamp = stamp;
        odom.header.frame_id = "odom";
        odom.child_frame_id = "base_link";
        odom.pose.pose.position.x = pub_e - init_enu_e_;  // east, takeoff-relative
        odom.pose.pose.position.y = pub_n - init_enu_n_;  // north, takeoff-relative
        odom.pose.pose.position.z = 0.0;
        odom.pose.pose.orientation.x = 0.0;
        odom.pose.pose.orientation.y = 0.0;
        odom.pose.pose.orientation.z = std::sin(yaw_enu * 0.5);
        odom.pose.pose.orientation.w = std::cos(yaw_enu * 0.5);

        // Pose covariance (6x6 row-major): [x,y,z,roll,pitch,yaw]
        for (int i = 0; i < 36; ++i) odom.pose.covariance[i] = 0.0;
        odom.pose.covariance[0]  = var_e;
        odom.pose.covariance[7]  = var_n;
        odom.pose.covariance[14] = std::numeric_limits<double>::quiet_NaN();
        odom.pose.covariance[21] = std::numeric_limits<double>::quiet_NaN();
        odom.pose.covariance[28] = std::numeric_limits<double>::quiet_NaN();
        odom.pose.covariance[35] = var_yaw_rad;

        // Twist not estimated by PF — mark unknown.
        for (int i = 0; i < 36; ++i) odom.twist.covariance[i] = 0.0;
        odom.twist.covariance[0] = -1.0;  // ROS convention: unknown

        pub_odom_local_->publish(odom);
    }

    std_msgs::msg::Float32 ess_msg;
    ess_msg.data = static_cast<float>(pf.effective_sample_size());
    pub_ess_->publish(ess_msg);

    std_msgs::msg::String state_msg;
    state_msg.data = phase_name(pf.phase());
    pub_state_->publish(state_msg);

    // Diagnostic CSV row
    log_diag_row(stamp, static_cast<int>(pf.phase()), pub_e, pub_n, pub_hdg,
                 pf.weighted_spread(), pf.effective_sample_size(),
                 diag_top1_sim, diag_n_coarse_cand,
                 diag_fine_source, diag_fine_inliers, diag_corr_dist,
                 diag_flow_con, diag_flow_cv, diag_inl_ratio,
                 est_full.bias_x, est_full.bias_y,
                 gt_lat_, gt_lon_,
                 diag_pnp_alt, obs.altitude_m, diag_fine_method,
                 diag_fine_lat, diag_fine_lon,
                 diag_sat_inl, diag_sat_lat, diag_sat_lon, diag_sat_method,
                 diag_mos_inl, diag_mos_lat, diag_mos_lon, diag_mos_method);

    RCLCPP_INFO(get_logger(),
        "=== F%d | %s | lat=%.6f lon=%.6f | ESS=%.0f | spread=%.1f ===",
        frame_count_, phase_name(pf.phase()),
        pub_lat, pub_lon,
        pf.effective_sample_size(), pf.weighted_spread());
}

}  // namespace pf
