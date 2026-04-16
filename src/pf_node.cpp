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
    motion_.set_jump_params(cfg_.pf.vio_jump_threshold_m, cfg_.pf.vio_jump_velocity_ema_alpha);

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

    // Publishers
    pub_position_ = create_publisher<sensor_msgs::msg::NavSatFix>("/pf_geo_loc/position", 10);
    pub_ess_ = create_publisher<std_msgs::msg::Float32>("/pf_geo_loc/ess", 10);
    pub_state_ = create_publisher<std_msgs::msg::String>("/pf_geo_loc/state", 10);

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

    if (!initialized_ && altitude_buf_.size() >= 5) {
        std::vector<float> sorted(altitude_buf_.begin(), altitude_buf_.end());
        std::sort(sorted.begin(), sorted.end());
        float median_alt = sorted[sorted.size() / 2];

        if (pf_->try_init(median_alt)) {
            initialized_ = true;
            RCLCPP_INFO(get_logger(), "Altitude gate passed: %.1f m", median_alt);

            // Seed PF: preconfigured position > RTK fallback
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
            }

            if (seed_lat != 0.0) {
                auto [e, n] = enu_.wgs84_to_enu(seed_lat, seed_lon);
                pf_->seed_from_position(e, n, 0.0,
                    cfg_.pf.init_sigma_pos, cfg_.pf.init_sigma_hdg);
                RCLCPP_INFO(get_logger(), "%s init: lat=%.6f lon=%.6f -> ENU(%.1f, %.1f)",
                    seed_source.c_str(), seed_lat, seed_lon, e, n);
            }
        }
    }
}

void PFGeoLocNode::yaw_callback(const std_msgs::msg::Float64::ConstSharedPtr& msg) {
    motion_.set_yaw(msg->data);
}

void PFGeoLocNode::rtk_callback(const sensor_msgs::msg::NavSatFix::ConstSharedPtr& msg) {
    gt_lat_ = msg->latitude;
    gt_lon_ = msg->longitude;
    has_gt_ = true;

    int64_t ts_ns = static_cast<int64_t>(msg->header.stamp.sec) * 1000000000LL +
                    msg->header.stamp.nanosec;

    if (initialized_ && pf_->phase() != Phase::UNINIT) {
        auto delta = motion_.update(ts_ns, msg->latitude, msg->longitude);
        if (delta.has_value())
            pf_->predict(delta.value());
    } else {
        motion_.update(ts_ns, msg->latitude, msg->longitude);
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

        // Build footprint reconstruction
        std::optional<FootprintReconstruction> fp_recon;
        if (obs.altitude_m > 20.0) {
            try {
                fp_recon = obs.footprint().reconstruct(
                    rlat, rlon, obs.altitude_m, heading_for_recon,
                    cfg_.matchers.camera_fx, cfg_.matchers.camera_fy,
                    cfg_.matchers.matcher_resolution, cfg_.matchers.matcher_resolution,
                    cv::Size(cfg_.matchers.matcher_resolution, cfg_.matchers.matcher_resolution),
                    cfg_.matchers.mosaic_context_scale,
                    cfg_.matchers.satellite_context_scale);
            } catch (const std::exception& e) {
                RCLCPP_WARN(get_logger(), "Recon exception: %s", e.what());
            }
        }

        // Debug: show images
        {
            int res = cfg_.matchers.matcher_resolution;
            cv::Mat mono_wrap(height, width, CV_8UC1, const_cast<uint8_t*>(mono_data));
            cv::Mat uav_show;
            cv::resize(mono_wrap, uav_show, cv::Size(res, res));
            cv::imshow("UAV", uav_show);
            if (fp_recon.has_value() && !fp_recon->satellite_crop.empty())
                cv::imshow("Satellite", fp_recon->satellite_crop);
            if (fp_recon.has_value() && !fp_recon->mosaic_rotated.empty()) {
                cv::Mat mosaic_show;
                cv::resize(fp_recon->mosaic_rotated, mosaic_show, cv::Size(res, res));
                cv::imshow("Mosaic", mosaic_show);
            }
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
            if (sat_fine.has_value()) {
                stats_.satellite_ok++;
                stats_.total_sat_inliers += sat_fine->inliers;
                if (!best_fine || sat_fine->inliers > best_fine->inliers)
                    best_fine = sat_fine;
            }
        }
        // Stage B: Mosaic disabled — satellite only

        // PnP altitude gate: reject if PnP altitude disagrees with barometer
        auto altitude_gate_ok = [&](const auto& fine) -> bool {
            if (fine.method != "pnp" || fine.pnp_altitude <= 0.0 || obs.altitude_m <= 0.0)
                return true;
            double alt_delta = std::abs(fine.pnp_altitude - obs.altitude_m);
            if (alt_delta > cfg_.pf.pnp_altitude_gate_m) {
                RCLCPP_WARN(get_logger(), "F%d PnP alt gate: pnp=%.1f baro=%.1f delta=%.1f > %.1f — rejected",
                    frame_count_, fine.pnp_altitude, obs.altitude_m, alt_delta, cfg_.pf.pnp_altitude_gate_m);
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
            pf.update_fine(fe, fn, best_fine->inliers, best_fine->heading_deg,
                          std::nullopt, std::nullopt, best_fine->method);
            fine_succeeded = true;
            stats_.skip_coarse++;
            diag_fine_source = best_fine->patch_name;
            diag_fine_method = best_fine->method;
            diag_fine_inliers = best_fine->inliers;
            diag_corr_dist = corr_dist;
            diag_pnp_alt = best_fine->pnp_altitude;
            diag_fine_lat = best_fine->lat;
            diag_fine_lon = best_fine->lon;
            diag_flow_con = best_fine->flow_consistency;
            diag_flow_cv = best_fine->flow_magnitude_cv;
            diag_inl_ratio = best_fine->inlier_ratio;
        }

        // Stage C: Need coarse for single-patch fallback
        if (!fine_succeeded) {
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
                pf.update_fine(fe, fn, best_fine->inliers, best_fine->heading_deg,
                              std::nullopt, std::nullopt, best_fine->method);
                diag_fine_source = best_fine->patch_name;
                diag_fine_method = best_fine->method;
                diag_fine_inliers = best_fine->inliers;
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
                double verify_heading = primary_fine->heading_deg.value_or(
                    motion_.current_yaw()) + cfg_.camera.heading_offset_deg;
                bool verified = false;
                double agreement_m = 999.0;

                try {
                    auto verify_recon_opt = obs.footprint().reconstruct(
                        primary_fine->lat, primary_fine->lon, obs.altitude_m, verify_heading,
                        cfg_.matchers.camera_fx, cfg_.matchers.camera_fy,
                        cfg_.matchers.matcher_resolution, cfg_.matchers.matcher_resolution,
                        cv::Size(cfg_.matchers.matcher_resolution, cfg_.matchers.matcher_resolution),
                        cfg_.matchers.mosaic_context_scale,
                        cfg_.matchers.satellite_context_scale);

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
