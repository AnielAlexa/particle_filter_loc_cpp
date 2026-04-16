#include "particle_filter_loc_cpp/config.hpp"
#include <yaml-cpp/yaml.h>
#include <stdexcept>

namespace pf {

namespace {

template <typename T>
T get(const YAML::Node& node, const std::string& key, T default_val) {
    if (node[key]) return node[key].as<T>();
    return default_val;
}

}  // namespace

FullConfig load_config(const std::string& yaml_path) {
    YAML::Node root = YAML::LoadFile(yaml_path);
    FullConfig cfg;

    // Camera
    if (auto c = root["camera_original"]) {
        cfg.camera.fx = get(c, "fx", cfg.camera.fx);
        cfg.camera.fy = get(c, "fy", cfg.camera.fy);
        cfg.camera.w = get(c, "w", cfg.camera.w);
        cfg.camera.h = get(c, "h", cfg.camera.h);
        cfg.camera.heading_offset_deg = get(c, "heading_offset_deg", cfg.camera.heading_offset_deg);
    }

    // ENU origin
    if (auto e = root["enu_origin"]) {
        cfg.enu_origin_lat = get(e, "lat", cfg.enu_origin_lat);
        cfg.enu_origin_lon = get(e, "lon", cfg.enu_origin_lon);
    }

    // Particle filter
    if (auto p = root["particle_filter"]) {
        auto& pf = cfg.pf;
        pf.n_dispersed = get(p, "n_dispersed", pf.n_dispersed);
        pf.n_tracking = get(p, "n_tracking", pf.n_tracking);
        pf.sigma_pos_dispersed = get(p, "sigma_pos_dispersed", pf.sigma_pos_dispersed);
        pf.sigma_pos_tracking = get(p, "sigma_pos_tracking", pf.sigma_pos_tracking);
        pf.sigma_hdg_dispersed = get(p, "sigma_hdg_dispersed", pf.sigma_hdg_dispersed);
        pf.sigma_hdg_tracking = get(p, "sigma_hdg_tracking", pf.sigma_hdg_tracking);
        pf.drift_noise_m_per_s = get(p, "drift_noise_m_per_s", pf.drift_noise_m_per_s);
        pf.drift_noise_hdg_per_s = get(p, "drift_noise_hdg_per_s", pf.drift_noise_hdg_per_s);
        pf.sigma_obs_coarse = get(p, "sigma_obs_coarse", pf.sigma_obs_coarse);
        pf.sigma_obs_fine = get(p, "sigma_obs_fine", pf.sigma_obs_fine);
        pf.top_k_coarse = get(p, "top_k_coarse", pf.top_k_coarse);
        pf.ess_threshold_fraction = get(p, "ess_threshold_fraction", pf.ess_threshold_fraction);
        pf.init_altitude_m = get(p, "init_altitude_m", pf.init_altitude_m);
        pf.converge_spread_m = get(p, "converge_spread_m", pf.converge_spread_m);
        pf.tracking_spread_m = get(p, "tracking_spread_m", pf.tracking_spread_m);
        pf.lost_spread_m = get(p, "lost_spread_m", pf.lost_spread_m);
        pf.fine_every_n_frames = get(p, "fine_every_n_frames", pf.fine_every_n_frames);
        pf.fine_early_exit_inliers = get(p, "fine_early_exit_inliers", pf.fine_early_exit_inliers);
        pf.fine_min_inliers_heading = get(p, "fine_min_inliers_heading", pf.fine_min_inliers_heading);
        pf.min_search_radius_m = get(p, "min_search_radius_m", pf.min_search_radius_m);
        pf.search_radius_multiplier = get(p, "search_radius_multiplier", pf.search_radius_multiplier);
        pf.base_context_fraction = get(p, "base_context_fraction", pf.base_context_fraction);
        pf.max_context_fraction = get(p, "max_context_fraction", pf.max_context_fraction);
        pf.altitude_sigma_enabled = get(p, "altitude_sigma_enabled", pf.altitude_sigma_enabled);
        pf.altitude_sigma_ref_m = get(p, "altitude_sigma_ref_m", pf.altitude_sigma_ref_m);
        pf.altitude_sigma_scale = get(p, "altitude_sigma_scale", pf.altitude_sigma_scale);
        pf.static_threshold_m = get(p, "static_threshold_m", pf.static_threshold_m);
        pf.static_sigma_scale = get(p, "static_sigma_scale", pf.static_sigma_scale);
        pf.fine_consistency_max_m = get(p, "fine_consistency_max_m", pf.fine_consistency_max_m);
        pf.pnp_altitude_gate_m = get(p, "pnp_altitude_gate_m", pf.pnp_altitude_gate_m);
        pf.fine_force_inliers = get(p, "fine_force_inliers", pf.fine_force_inliers);
        pf.roughen_enabled = get(p, "roughen_enabled", pf.roughen_enabled);
        pf.roughen_scale = get(p, "roughen_scale", pf.roughen_scale);
        pf.likelihood_nu = get(p, "likelihood_nu", pf.likelihood_nu);
        pf.adaptive_enabled = get(p, "adaptive_enabled", pf.adaptive_enabled);
        pf.adaptive_ema_alpha = get(p, "adaptive_ema_alpha", pf.adaptive_ema_alpha);
        pf.adaptive_drift_floor_m = get(p, "adaptive_drift_floor_m", pf.adaptive_drift_floor_m);
        pf.adaptive_drift_ref_m = get(p, "adaptive_drift_ref_m", pf.adaptive_drift_ref_m);
        pf.adaptive_min_inliers = get(p, "adaptive_min_inliers", pf.adaptive_min_inliers);
        pf.adaptive_sigma_pos_rtk = get(p, "adaptive_sigma_pos_rtk", pf.adaptive_sigma_pos_rtk);
        pf.adaptive_sigma_obs_fine_rtk = get(p, "adaptive_sigma_obs_fine_rtk", pf.adaptive_sigma_obs_fine_rtk);
        pf.adaptive_roughen_scale_rtk = get(p, "adaptive_roughen_scale_rtk", pf.adaptive_roughen_scale_rtk);
        pf.adaptive_sigma_pos_vio = get(p, "adaptive_sigma_pos_vio", pf.adaptive_sigma_pos_vio);
        pf.adaptive_sigma_obs_fine_vio = get(p, "adaptive_sigma_obs_fine_vio", pf.adaptive_sigma_obs_fine_vio);
        pf.adaptive_roughen_scale_vio = get(p, "adaptive_roughen_scale_vio", pf.adaptive_roughen_scale_vio);

        // Bias model
        pf.sigma_bias_random_walk = get(p, "sigma_bias_random_walk", pf.sigma_bias_random_walk);
        pf.sigma_bias_init = get(p, "sigma_bias_init", pf.sigma_bias_init);
        pf.vio_jump_threshold_m = get(p, "vio_jump_threshold_m", pf.vio_jump_threshold_m);
        pf.vio_jump_velocity_ema_alpha = get(p, "vio_jump_velocity_ema_alpha", pf.vio_jump_velocity_ema_alpha);
        pf.obs_min_inliers_apply = get(p, "obs_min_inliers_apply", pf.obs_min_inliers_apply);
        pf.inlier_tau = get(p, "inlier_tau", pf.inlier_tau);
        pf.init_sigma_pos = get(p, "init_sigma_pos", pf.init_sigma_pos);
        pf.init_sigma_hdg = get(p, "init_sigma_hdg", pf.init_sigma_hdg);
        pf.lost_recovery_min_inliers = get(p, "lost_recovery_min_inliers", pf.lost_recovery_min_inliers);
        pf.lost_recovery_verify_agreement_m = get(p, "lost_recovery_verify_agreement_m", pf.lost_recovery_verify_agreement_m);
        pf.lost_recovery_sigma_pos = get(p, "lost_recovery_sigma_pos", pf.lost_recovery_sigma_pos);
        pf.lost_recovery_sigma_hdg = get(p, "lost_recovery_sigma_hdg", pf.lost_recovery_sigma_hdg);
    }

    // Trust
    if (auto t = root["trust"]) {
        auto& tr = cfg.trust;
        tr.inlier_tau = get(t, "inlier_tau", tr.inlier_tau);
        tr.inlier_weight = get(t, "inlier_weight", tr.inlier_weight);
        tr.sim_floor = get(t, "sim_floor", tr.sim_floor);
        tr.sim_ceiling = get(t, "sim_ceiling", tr.sim_ceiling);
        tr.sim_weight = get(t, "sim_weight", tr.sim_weight);
        tr.consistency_weight = get(t, "consistency_weight", tr.consistency_weight);
        tr.consistency_min_radius_m = get(t, "consistency_min_radius_m", tr.consistency_min_radius_m);
        tr.altitude_weight = get(t, "altitude_weight", tr.altitude_weight);
        tr.altitude_ref_m = get(t, "altitude_ref_m", tr.altitude_ref_m);
        tr.geometry_weight = get(t, "geometry_weight", tr.geometry_weight);
        tr.sigma_min_scale = get(t, "sigma_min_scale", tr.sigma_min_scale);
        tr.sigma_max_scale = get(t, "sigma_max_scale", tr.sigma_max_scale);
        tr.drift_pf_confidence_threshold = get(t, "drift_pf_confidence_threshold", tr.drift_pf_confidence_threshold);
        tr.drift_recon_sim_threshold = get(t, "drift_recon_sim_threshold", tr.drift_recon_sim_threshold);
        tr.drift_coarse_sim_min = get(t, "drift_coarse_sim_min", tr.drift_coarse_sim_min);
        tr.drift_sigma_cap = get(t, "drift_sigma_cap", tr.drift_sigma_cap);
        tr.recon_high_conf_sim_thr = get(t, "recon_high_conf_sim_thr", tr.recon_high_conf_sim_thr);
        tr.recon_high_conf_inlier_thr = get(t, "recon_high_conf_inlier_thr", tr.recon_high_conf_inlier_thr);
        tr.recon_high_conf_boost = get(t, "recon_high_conf_boost", tr.recon_high_conf_boost);
        tr.global_corr_enabled = get(t, "global_corr_enabled", tr.global_corr_enabled);
        tr.ema_alpha = get(t, "ema_alpha", tr.ema_alpha);
        tr.min_confidence = get(t, "min_confidence", tr.min_confidence);
        tr.coarse_sim_floor = get(t, "coarse_sim_floor", tr.coarse_sim_floor);
        tr.coarse_sim_ceiling = get(t, "coarse_sim_ceiling", tr.coarse_sim_ceiling);
        tr.coarse_max_teleport_fraction = get(t, "coarse_max_teleport_fraction", tr.coarse_max_teleport_fraction);
        if (t["coarse_teleport_sigma_range"]) {
            tr.coarse_teleport_sigma_range.clear();
            for (const auto& v : t["coarse_teleport_sigma_range"])
                tr.coarse_teleport_sigma_range.push_back(v.as<double>());
        }
    }

    // Matchers
    if (auto m = root["matchers"]) {
        auto& mc = cfg.matchers;
        mc.script_dir = get<std::string>(m, "script_dir", mc.script_dir);
        mc.use_vlad_trt = get(m, "use_vlad_trt", mc.use_vlad_trt);
        mc.vlad_trt_engine_path = get<std::string>(m, "vlad_trt_engine_path", mc.vlad_trt_engine_path);
        mc.vlad_database_path = get<std::string>(m, "vlad_database_path", mc.vlad_database_path);
        mc.vlad_patch_names_path = get<std::string>(m, "vlad_patch_names_path", mc.vlad_patch_names_path);
        mc.boq_engine_path = get<std::string>(m, "boq_engine_path", mc.boq_engine_path);
        mc.boq_database_path = get<std::string>(m, "boq_database_path", mc.boq_database_path);
        mc.patch_names_path = get<std::string>(m, "patch_names_path", mc.patch_names_path);
        mc.fine_engine_path = get<std::string>(m, "fine_engine_path", mc.fine_engine_path);
        mc.gps_metadata_path = get<std::string>(m, "gps_metadata_path", mc.gps_metadata_path);
        mc.patches_dir = get<std::string>(m, "patches_dir", mc.patches_dir);
        mc.image_size = get(m, "image_size", mc.image_size);
        mc.grayscale = get(m, "grayscale", mc.grayscale);
        mc.matcher_resolution = get(m, "matcher_resolution", mc.matcher_resolution);
        mc.camera_fx = get(m, "camera_fx", mc.camera_fx);
        mc.camera_fy = get(m, "camera_fy", mc.camera_fy);
        mc.camera_cx = get(m, "camera_cx", mc.camera_cx);
        mc.camera_cy = get(m, "camera_cy", mc.camera_cy);
        mc.context_enabled = get(m, "context_enabled", mc.context_enabled);
        mc.context_fraction = get(m, "context_fraction", mc.context_fraction);
        mc.fine_conf_threshold = get(m, "fine_conf_threshold", mc.fine_conf_threshold);
        mc.min_inliers_ransac = get(m, "min_inliers_ransac", mc.min_inliers_ransac);
        mc.min_inliers = get(m, "min_inliers", mc.min_inliers);
        mc.min_inliers_homography = get(m, "min_inliers_homography", mc.min_inliers_homography);
        mc.ransac_max_iters = get(m, "ransac_max_iters", mc.ransac_max_iters);
        mc.patch_cache_size = get(m, "patch_cache_size", mc.patch_cache_size);
        mc.mosaic_context_scale = get(m, "mosaic_context_scale", mc.mosaic_context_scale);
        mc.satellite_context_scale = get(m, "satellite_context_scale", mc.satellite_context_scale);
    }

    // Init
    if (auto i = root["init"]) {
        cfg.init.lock_n_frames = get(i, "lock_n_frames", cfg.init.lock_n_frames);
        cfg.init.lock_sim_threshold = get(i, "lock_sim_threshold", cfg.init.lock_sim_threshold);
    }

    // Replay (topic names)
    if (auto r = root["replay"]) {
        cfg.replay.camera_topic = get<std::string>(r, "camera_topic", cfg.replay.camera_topic);
        cfg.replay.rtk_topic = get<std::string>(r, "rtk_topic", cfg.replay.rtk_topic);
        cfg.replay.yaw_topic = get<std::string>(r, "yaw_topic", cfg.replay.yaw_topic);
        cfg.replay.altimeter_topic = get<std::string>(r, "altimeter_topic", cfg.replay.altimeter_topic);
    }

    return cfg;
}

}  // namespace pf
