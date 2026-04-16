#pragma once

#include <string>
#include <vector>

namespace pf {

struct PFConfig {
    int n_dispersed = 300;
    int n_tracking = 150;
    double sigma_pos_dispersed = 2.0;
    double sigma_pos_tracking = 3.0;
    double sigma_hdg_dispersed = 10.0;
    double sigma_hdg_tracking = 2.0;
    double drift_noise_m_per_s = 0.0;
    double drift_noise_hdg_per_s = 0.0;
    double sigma_obs_coarse = 10.0;
    double sigma_obs_fine = 5.0;
    int top_k_coarse = 5;
    double ess_threshold_fraction = 0.5;
    double init_altitude_m = 50.0;
    double converge_spread_m = 40.0;
    double tracking_spread_m = 20.0;
    double lost_spread_m = 150.0;
    int fine_every_n_frames = 3;
    int fine_early_exit_inliers = 15;
    int fine_min_inliers_heading = 12;
    double min_search_radius_m = 40.0;
    double search_radius_multiplier = 2.5;
    double base_context_fraction = 0.2;
    double max_context_fraction = 1.0;
    bool altitude_sigma_enabled = false;
    double altitude_sigma_ref_m = 60.0;
    double altitude_sigma_scale = 0.5;
    double static_threshold_m = 0.15;
    double static_sigma_scale = 0.1;
    double fine_consistency_max_m = 40.0;
    double pnp_altitude_gate_m = 15.0;   // reject PnP if |pnp_alt - baro_alt| > this
    int fine_force_inliers = 0;
    bool roughen_enabled = true;
    double roughen_scale = 2.0;
    double likelihood_nu = 5.0;
    bool adaptive_enabled = true;
    double adaptive_ema_alpha = 0.2;
    double adaptive_drift_floor_m = 4.0;
    double adaptive_drift_ref_m = 10.0;
    int adaptive_min_inliers = 15;
    double adaptive_sigma_pos_rtk = 0.5;
    double adaptive_sigma_obs_fine_rtk = 5.0;
    double adaptive_roughen_scale_rtk = 0.5;
    double adaptive_sigma_pos_vio = 3.0;
    double adaptive_sigma_obs_fine_vio = 5.0;
    double adaptive_roughen_scale_vio = 2.0;

    // Bias model (Phase 2+)
    double sigma_bias_random_walk = 0.05;   // m/step bias drift
    double sigma_bias_init = 0.01;          // initial bias uncertainty
    double bias_clamp_m = 10.0;             // max bias magnitude (prevents runaway)
    double vio_jump_threshold_m = 20.0;     // displacement triggering jump detection (RTK: high)
    double vio_jump_velocity_ema_alpha = 0.3;
    int obs_min_inliers_apply = 10;         // min inliers to apply observation
    double inlier_tau = 25.0;               // trust curve: score = 1 - exp(-inliers/tau)

    // Initialization (Phase 4+)
    double init_sigma_pos = 0.2;
    double init_sigma_hdg = 2.0;

    // LOST recovery (Phase 5+)
    int lost_recovery_min_inliers = 15;
    double lost_recovery_verify_agreement_m = 20.0;
    double lost_recovery_sigma_pos = 2.0;
    double lost_recovery_sigma_hdg = 5.0;
};

struct TrustConfig {
    double inlier_tau = 25.0;
    double inlier_weight = 0.39;
    double sim_floor = 0.10;
    double sim_ceiling = 0.55;
    double sim_weight = 0.22;
    double consistency_weight = 0.11;
    double consistency_min_radius_m = 30.0;
    double altitude_weight = 0.11;
    double altitude_ref_m = 60.0;
    double geometry_weight = 0.17;
    double sigma_min_scale = 0.3;
    double sigma_max_scale = 5.0;
    double drift_pf_confidence_threshold = 0.3;
    double drift_recon_sim_threshold = 0.20;
    double drift_coarse_sim_min = 0.35;
    double drift_sigma_cap = 3.0;
    double recon_high_conf_sim_thr = 0.30;
    int recon_high_conf_inlier_thr = 25;
    double recon_high_conf_boost = 1.5;
    bool global_corr_enabled = false;
    double global_corr_recon_sim_threshold = 0.15;
    int global_corr_min_frames = 3;
    int global_corr_min_inliers = 10;
    double global_corr_max_spread_m = 50.0;
    double global_corr_position_consistency_m = 40.0;
    int global_corr_cooldown_frames = 10;
    double global_corr_teleport_fraction = 0.35;
    double global_corr_teleport_sigma = 10.0;
    double global_corr_min_spread_m = 15.0;
    double global_corr_coarse_validation_m = 40.0;
    double ema_alpha = 0.5;
    double min_confidence = 0.05;
    double coarse_sim_floor = 0.25;
    double coarse_sim_ceiling = 0.60;
    double coarse_max_teleport_fraction = 0.8;
    std::vector<double> coarse_teleport_sigma_range = {15.0, 30.0};
};

struct MatcherConfig {
    std::string script_dir;
    bool use_vlad_trt = true;
    std::string vlad_trt_engine_path;
    std::string vlad_database_path;
    std::string vlad_patch_names_path;
    std::string boq_engine_path;
    std::string boq_database_path;
    std::string patch_names_path;
    std::string fine_engine_path;
    std::string gps_metadata_path;
    std::string patches_dir;
    int image_size = 256;
    bool grayscale = true;
    int matcher_resolution = 320;
    double camera_fx = 501.8;
    double camera_fy = 502.1;
    double camera_cx = 151.1;
    double camera_cy = 221.9;
    bool context_enabled = true;
    double context_fraction = 0.2;
    double fine_conf_threshold = 0.20;
    int min_inliers_ransac = 4;
    int min_inliers = 8;
    int min_inliers_homography = 5;
    int ransac_max_iters = 1000;
    int patch_cache_size = 100;
    double mosaic_context_scale = 1.5;      // mosaic crop = footprint_diagonal * this
    double satellite_context_scale = 1.1;  // satellite crop expansion factor
};

struct InitConfig {
    int lock_n_frames = 4;
    double lock_sim_threshold = 0.3;
};

struct CameraConfig {
    double fx = 1129.0;
    double fy = 1130.0;
    int w = 1280;
    int h = 720;
    double heading_offset_deg = 270.0;
};

struct ReplayConfig {
    std::string camera_topic = "/camera/image_mono";
    std::string rtk_topic = "/m300/rtk/fix";
    std::string yaw_topic = "/m300/rtk/yaw";
    std::string altimeter_topic = "/altimeter/range";
};

struct FullConfig {
    CameraConfig camera;
    double enu_origin_lat = 45.7896;
    double enu_origin_lon = 21.1928;
    PFConfig pf;
    TrustConfig trust;
    MatcherConfig matchers;
    InitConfig init;
    ReplayConfig replay;
};

// Load from YAML file
FullConfig load_config(const std::string& yaml_path);

}  // namespace pf
