#include "particle_filter_loc_cpp/pf_node.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <numeric>

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

    // Stats timer — print every 10s
    stats_timer_ = create_wall_timer(std::chrono::seconds(10),
        std::bind(&PFGeoLocNode::print_stats, this));

    init_diag_csv();
    RCLCPP_INFO(get_logger(), "PFGeoLocNode (C++) ready.");
}

void PFGeoLocNode::init_diag_csv() {
    std::string path = "/home/jetson/particle_filter_ws/src/particle_filter_loc_cpp/results/pf_diag.csv";
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
    std::string stats_path = "/home/jetson/particle_filter_ws/src/particle_filter_loc_cpp/results/pf_stats.csv";
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

    if (!initialized_ && altitude_buf_.size() >= 5) {
        std::vector<float> sorted(altitude_buf_.begin(), altitude_buf_.end());
        std::sort(sorted.begin(), sorted.end());
        float median_alt = sorted[sorted.size() / 2];
        try_initialize(median_alt);
    }
}

void PFGeoLocNode::try_lock_yaw() {
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

void PFGeoLocNode::try_initialize(float median_alt) {
    if (initialized_) return;
    if (!pf_->try_init(median_alt)) return;
    if (!has_vio_) {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
            "Init: altitude OK (%.1f m) - waiting for VIO pose", median_alt);
        return;
    }

    double enu_yaw_deg = 0.0;
    std::string yaw_source;
    if (cfg_.pf.init_yaw != 0.0) {
        enu_yaw_deg = cfg_.pf.init_yaw;
        yaw_source = "config";
    } else if (yaw_locked_pretakeoff_) {
        enu_yaw_deg = locked_enu_yaw_compass_deg_;
        yaw_source = "pre-locked";
    } else if (last_rtk_yaw_deg_.has_value()) {
        enu_yaw_deg = *last_rtk_yaw_deg_;
        yaw_source = "RTK";
    } else {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
            "Init: altitude+VIO OK - waiting for RTK yaw (init_yaw=0)");
        return;
    }

    double seed_lat = 0.0, seed_lon = 0.0;
    std::string seed_source;
    if (cfg_.pf.init_lat != 0.0 && cfg_.pf.init_lon != 0.0) {
        seed_lat = cfg_.pf.init_lat;
        seed_lon = cfg_.pf.init_lon;
        seed_source = "config";
    } else if (has_gt_) {
        seed_lat = gt_lat_;
        seed_lon = gt_lon_;
        seed_source = "RTK";
    } else {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
            "Init: waiting for RTK fix (init_lat/lon=0)");
        return;
    }

    auto [e, n] = enu_.wgs84_to_enu(seed_lat, seed_lon);
    pf_->seed_from_position(e, n, enu_yaw_deg,
        cfg_.pf.init_sigma_pos, cfg_.pf.init_sigma_hdg);

    double vio_yaw_adj = wrap360(last_vio_yaw_deg_ + cfg_.pf.vio_yaw_offset_deg);
    if (vio_motion_.is_yaw_locked()) {
        // Rotation was captured on the ground; only pin the VIO origin here
        vio_motion_.pin_position(last_vio_x_, last_vio_y_);
    } else {
        vio_motion_.align(last_vio_x_, last_vio_y_, vio_yaw_adj, enu_yaw_deg);
    }

    // Cache VIO→ENU mapping for dead-reckoned publication
    vio_ref_x_ = last_vio_x_;
    vio_ref_y_ = last_vio_y_;
    double alpha = vio_motion_.align_theta_deg() * M_PI / 180.0;
    vio_align_cos_ = std::cos(alpha);
    vio_align_sin_ = std::sin(alpha);
    init_enu_e_ = e;
    init_enu_n_ = n;

    initialized_ = true;
    RCLCPP_INFO(get_logger(),
        "INIT: alt=%.1f m | pos(%s) lat=%.6f lon=%.6f -> ENU(%.1f, %.1f) | "
        "yaw(%s)=%.1f deg | VIO ref=(%.2f, %.2f) vio_yaw=%.1f -> align_theta=%.1f",
        median_alt, seed_source.c_str(), seed_lat, seed_lon, e, n,
        yaw_source.c_str(), enu_yaw_deg,
        last_vio_x_, last_vio_y_, last_vio_yaw_deg_,
        vio_motion_.align_theta_deg());
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
    if (!initialized_) {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 3000,
            "DBG img_cb: not initialized yet, skipping");
        return;
    }

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

        if (msg->encoding == "mono8") {
            data = msg->data.data();
            width = msg->width;
            height = msg->height;
        } else {
            // Convert via cv_bridge
            auto cv_ptr = cv_bridge::toCvShare(msg, "mono8");
            data = cv_ptr->image.data;
            width = cv_ptr->image.cols;
            height = cv_ptr->image.rows;
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
        double heading_for_recon = est_h + cfg_.camera.heading_offset_deg;

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
                    cfg_.matchers.matcher_resolution, cfg_.matchers.matcher_resolution,
                    cv::Size(cfg_.matchers.matcher_resolution, cfg_.matchers.matcher_resolution),
                    cfg_.matchers.mosaic_context_scale,
                    adaptive_sat_ctx);
            } catch (const std::exception& e) {
                RCLCPP_WARN(get_logger(), "Recon exception: %s", e.what());
            }
        }

        // Debug visualization disabled (cv::imshow/resize segfault with OpenCV 4.5d on this device)

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
                                        + cfg_.camera.heading_offset_deg;
                try {
                    auto refine_recon = obs.footprint().reconstruct(
                        sat_fine->lat, sat_fine->lon, obs.altitude_m, refine_heading,
                        cfg_.matchers.camera_fx, cfg_.matchers.camera_fy,
                        cfg_.matchers.matcher_resolution, cfg_.matchers.matcher_resolution,
                        cv::Size(cfg_.matchers.matcher_resolution, cfg_.matchers.matcher_resolution),
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
    if (pf.phase() == Phase::LOST && should_fine && !fine_succeeded) {
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
                                        + cfg_.camera.heading_offset_deg;
                bool verified = false;
                double agreement_m = 999.0;

                try {
                    auto verify_recon_opt = obs.footprint().reconstruct(
                        primary_fine->lat, primary_fine->lon, obs.altitude_m, verify_heading,
                        cfg_.matchers.camera_fx, cfg_.matchers.camera_fy,
                        cfg_.matchers.matcher_resolution, cfg_.matchers.matcher_resolution,
                        cv::Size(cfg_.matchers.matcher_resolution, cfg_.matchers.matcher_resolution),
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

    sensor_msgs::msg::NavSatFix pos_msg;
    pos_msg.header.stamp = stamp;  // use bag timestamp for correct time alignment
    pos_msg.header.frame_id = "gps_rtk_antenna";
    pos_msg.latitude = pub_lat;
    pos_msg.longitude = pub_lon;
    pub_position_->publish(pos_msg);

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
