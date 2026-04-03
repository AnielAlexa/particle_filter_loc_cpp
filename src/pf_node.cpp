#include "particle_filter_loc_cpp/pf_node.hpp"

#include <algorithm>
#include <numeric>

#include <cv_bridge/cv_bridge.h>

namespace pf {

PFGeoLocNode::PFGeoLocNode(const rclcpp::NodeOptions& options)
    : Node("pf_geo_loc_node", options)
{
    // Load config
    this->declare_parameter<std::string>("config_path", "");
    std::string config_path = this->get_parameter("config_path").as_string();
    if (config_path.empty()) {
        RCLCPP_ERROR(get_logger(), "config_path parameter required");
        throw std::runtime_error("config_path not set");
    }

    cfg_ = load_config(config_path);
    enu_ = ENUFrame(cfg_.enu_origin_lat, cfg_.enu_origin_lon);

    // Initialize components
    pf_ = std::make_unique<ParticleFilter>(cfg_.pf);
    obs_ = std::make_unique<ObservationModel>(cfg_.matchers, enu_);
    trust_ = std::make_unique<TrustTracker>(cfg_.trust);

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

    RCLCPP_INFO(get_logger(), "PFGeoLocNode (C++) ready.");
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

            // RTK init: seed PF from RTK position if available
            if (has_gt_) {
                auto [e, n] = enu_.wgs84_to_enu(gt_lat_, gt_lon_);
                pf_->seed_from_position(e, n, 0.0, 30.0, 180.0);
                RCLCPP_INFO(get_logger(), "RTK init: lat=%.6f lon=%.6f -> ENU(%.1f, %.1f)",
                    gt_lat_, gt_lon_, e, n);
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

        process_frame(data, width, height);
    } catch (const std::exception& e) {
        RCLCPP_WARN(get_logger(), "Frame processing error: %s", e.what());
    }

    processing_.store(false);
}

void PFGeoLocNode::process_frame(const uint8_t* mono_data, int width, int height) {
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

    // If PF not yet seeded (no RTK was available at altitude init), seed from coarse
    if (pf.phase() == Phase::UNINIT && first_init_camera_) {
        first_init_camera_ = false;
        RCLCPP_INFO(get_logger(), "DBG: No RTK init, running coarse SEED (top_k=%d)",
            cfg_.pf.top_k_coarse);
        auto coarse = obs.coarse_match(mono_data, width, height, nullptr, cfg_.pf.top_k_coarse);
        for (size_t i = 0; i < coarse.top_k_names.size(); ++i) {
            RCLCPP_INFO(get_logger(), "DBG:   [%zu] %s sim=%.4f",
                i, coarse.top_k_names[i].c_str(), coarse.top_k_sims[i]);
        }
        std::vector<std::pair<double, double>> centers;
        for (auto& name : coarse.top_k_names)
            centers.push_back(obs.get_patch_center_enu(name));
        pf.seed_from_coarse(centers, coarse.top_k_sims);
        RCLCPP_INFO(get_logger(), "DBG: coarse seed phase=%s spread=%.1f ESS=%.0f",
            phase_name(pf.phase()), pf.weighted_spread(), pf.effective_sample_size());
        return;
    }

    if (pf.phase() == Phase::UNINIT) return;

    auto [est_e, est_n, est_h] = pf.estimate();
    double search_radius = pf.get_search_radius();

    // Coarse matching with spatial constraint
    const std::vector<int>* candidates = nullptr;
    std::vector<int> candidate_vec;
    if (pf.phase() != Phase::DISPERSED) {
        candidate_vec = obs.get_indices_within_radius(est_e, est_n, search_radius);
        if (candidate_vec.size() < 5) {
            candidate_vec = obs.get_indices_within_radius(est_e, est_n, search_radius * 2.0);
        }
        candidates = &candidate_vec;
        RCLCPP_INFO(get_logger(), "DBG: spatial filter: %zu candidates within r=%.0fm",
            candidate_vec.size(), search_radius);
    } else {
        RCLCPP_INFO(get_logger(), "DBG: DISPERSED — no spatial filter, searching all %d patches",
            609);
    }

    RCLCPP_INFO(get_logger(), "DBG: running coarse_match...");
    auto coarse = obs.coarse_match(mono_data, width, height, candidates, cfg_.pf.top_k_coarse);
    RCLCPP_INFO(get_logger(), "DBG: coarse_match returned %zu results", coarse.top_k_names.size());
    for (size_t i = 0; i < coarse.top_k_names.size(); ++i) {
        RCLCPP_INFO(get_logger(), "DBG:   [%zu] %s sim=%.4f",
            i, coarse.top_k_names[i].c_str(), coarse.top_k_sims[i]);
    }

    // Coarse observation update
    std::vector<std::tuple<double, double, float>> coarse_obs;
    for (size_t i = 0; i < coarse.top_k_names.size(); ++i) {
        auto [e, n] = obs.get_patch_center_enu(coarse.top_k_names[i]);
        coarse_obs.emplace_back(e, n, coarse.top_k_sims[i]);
    }
    RCLCPP_INFO(get_logger(), "DBG: calling update_coarse with %zu obs (NOTE: filter drops sim<0.3!)",
        coarse_obs.size());
    pf.update_coarse(coarse_obs);
    RCLCPP_INFO(get_logger(), "DBG: after update_coarse: spread=%.1f ESS=%.0f",
        pf.weighted_spread(), pf.effective_sample_size());

    // Update altitude
    if (!altitude_buf_.empty()) {
        std::vector<float> sorted(altitude_buf_.begin(), altitude_buf_.end());
        std::sort(sorted.begin(), sorted.end());
        obs.altitude_m = sorted[sorted.size() / 2];
    }

    // Fine matching
    bool should_fine = pf.should_run_fine();
    RCLCPP_INFO(get_logger(), "DBG: should_run_fine=%d phase=%s",
        should_fine, phase_name(pf.phase()));
    if (should_fine) {
        int fine_top_k = pf.get_fine_top_k();
        double ctx_frac = pf.get_context_fraction();
        RCLCPP_INFO(get_logger(), "DBG: running fine_match top_k=%d ctx_frac=%.2f", fine_top_k, ctx_frac);
        for (int i = 0; i < fine_top_k && i < static_cast<int>(coarse.top_k_names.size()); ++i) {
            RCLCPP_INFO(get_logger(), "DBG:   fine_match on %s...", coarse.top_k_names[i].c_str());
            auto fine_result = obs.fine_match(mono_data, width, height,
                                              coarse.top_k_names[i], ctx_frac);
            if (fine_result.has_value()) {
                auto [fe, fn] = enu_.wgs84_to_enu(fine_result->lat, fine_result->lon);
                RCLCPP_INFO(get_logger(), "DBG:   fine OK: inliers=%d method=%s lat=%.6f lon=%.6f",
                    fine_result->inliers, fine_result->method.c_str(),
                    fine_result->lat, fine_result->lon);
                pf.update_fine(fe, fn, fine_result->inliers, fine_result->heading_deg);
                break;
            } else {
                RCLCPP_INFO(get_logger(), "DBG:   fine_match returned nullopt");
            }
        }
    }

    pf.resample_if_needed();
    pf.check_transitions();

    // Publish estimate
    auto [pub_e, pub_n, pub_hdg] = pf.estimate();
    auto [pub_lat, pub_lon] = enu_.enu_to_wgs84(pub_e, pub_n);

    sensor_msgs::msg::NavSatFix pos_msg;
    pos_msg.header.stamp = this->now();
    pos_msg.latitude = pub_lat;
    pos_msg.longitude = pub_lon;
    pub_position_->publish(pos_msg);

    std_msgs::msg::Float32 ess_msg;
    ess_msg.data = static_cast<float>(pf.effective_sample_size());
    pub_ess_->publish(ess_msg);

    std_msgs::msg::String state_msg;
    state_msg.data = phase_name(pf.phase());
    pub_state_->publish(state_msg);

    RCLCPP_INFO(get_logger(),
        "=== F%d | %s | lat=%.6f lon=%.6f | top1=%s sim=%.3f | ESS=%.0f | spread=%.1f | r=%.0fm ===",
        frame_count_, phase_name(pf.phase()),
        pub_lat, pub_lon,
        coarse.top_k_names.empty() ? "?" : coarse.top_k_names[0].c_str(),
        coarse.top_k_sims.empty() ? 0.f : coarse.top_k_sims[0],
        pf.effective_sample_size(), pf.weighted_spread(), search_radius);
}

}  // namespace pf
