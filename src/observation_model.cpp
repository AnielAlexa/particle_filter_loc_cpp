#include "particle_filter_loc_cpp/observation_model.hpp"

#include <cmath>
#include <iostream>
#include <regex>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace pf {

namespace {

FlowMetrics compute_flow_metrics(const Eigen::MatrixXf& mkpts0, const Eigen::MatrixXf& mkpts1) {
    FlowMetrics fm;
    int n = mkpts0.rows();
    if (n < 3) return fm;

    Eigen::MatrixXf flow = mkpts1 - mkpts0;
    Eigen::VectorXf angles(n), mags(n);
    for (int i = 0; i < n; ++i) {
        angles(i) = std::atan2(flow(i, 1), flow(i, 0));
        mags(i) = flow.row(i).norm();
    }

    double mean_sin = 0, mean_cos = 0;
    for (int i = 0; i < n; ++i) {
        mean_sin += std::sin(angles(i));
        mean_cos += std::cos(angles(i));
    }
    mean_sin /= n;
    mean_cos /= n;
    fm.flow_consistency = static_cast<float>(std::sqrt(mean_sin * mean_sin + mean_cos * mean_cos));

    float mag_mean = mags.mean();
    float mag_std = std::sqrt((mags.array() - mag_mean).square().mean());
    fm.flow_magnitude_cv = (mag_mean > 1e-6f) ? mag_std / mag_mean : 1.0f;

    fm.flow_heading_deg = static_cast<float>(
        std::fmod(std::atan2(mean_sin, mean_cos) * 180.0 / M_PI, 360.0));

    return fm;
}

std::pair<int, int> parse_patch_name(const std::string& name) {
    std::regex re(R"(patch_(\d+)_(\d+))");
    std::smatch m;
    if (!std::regex_match(name, m, re))
        throw std::runtime_error("Cannot parse patch name: " + name);
    return {std::stoi(m[1].str()), std::stoi(m[2].str())};
}

}  // namespace

ObservationModel::ObservationModel(const MatcherConfig& cfg, const ENUFrame& enu)
    : cfg_(cfg), enu_(enu)
{
    script_dir_ = cfg.script_dir;
    patches_dir_ = script_dir_ + "/" + cfg.patches_dir;

    // Initialize VPI preprocessor
    vpi_ = std::make_unique<VpiPreprocessor>(cfg.image_size, cfg.matcher_resolution);

    // Database path
    std::string db_npy_path, names_path;
    if (cfg.use_vlad_trt) {
        db_npy_path = script_dir_ + "/" + cfg.vlad_database_path;
        names_path = script_dir_ + "/" + cfg.vlad_patch_names_path;
    } else {
        db_npy_path = script_dir_ + "/" + cfg.boq_database_path;
        names_path = script_dir_ + "/" + cfg.patch_names_path;
    }
    // Convert .pt to .npy if needed
    // C++17: use rfind instead of C++20 ends_with
    if (db_npy_path.size() >= 3 && db_npy_path.rfind(".pt") == db_npy_path.size() - 3)
        db_npy_path = db_npy_path.substr(0, db_npy_path.size() - 3) + ".npy";

    std::string gps_path = script_dir_ + "/" + cfg.gps_metadata_path;

    // Load descriptor database
    db_ = std::make_unique<DescriptorDatabase>(db_npy_path, names_path, gps_path, enu);

    // Load coarse matcher
    std::string coarse_engine;
    if (cfg.use_vlad_trt)
        coarse_engine = script_dir_ + "/" + cfg.vlad_trt_engine_path;
    else
        coarse_engine = script_dir_ + "/" + cfg.boq_engine_path;
    coarse_ = std::make_unique<CoarseMatcher>(coarse_engine, *db_, *vpi_, cfg.image_size);

    // Load fine matcher
    std::string fine_engine = script_dir_ + "/" + cfg.fine_engine_path;
    fine_ = std::make_unique<FineMatcher>(fine_engine, *vpi_, cfg.matcher_resolution, cfg.fine_conf_threshold);

    // Pose estimator
    pose_ = std::make_unique<PoseEstimator>(
        cfg.camera_fx, cfg.camera_fy, cfg.camera_cx, cfg.camera_cy,
        altitude_m, cfg.min_inliers_ransac, cfg.min_inliers, cfg.ransac_max_iters);

    // Footprint reconstructor
    // Build GPS metadata map from database
    std::unordered_map<std::string, PatchMeta> gps_meta;
    for (int i = 0; i < db_->num_patches(); ++i) {
        try {
            gps_meta[db_->patch_name(i)] = db_->patch_meta(db_->patch_name(i));
        } catch (...) {}
    }
    footprint_ = std::make_unique<SatelliteFootprintReconstructor>(gps_meta, patches_dir_, enu);

    std::cout << "[ObservationModel] Loaded " << db_->num_patches()
              << " patches, desc_dim=" << db_->descriptor_dim() << std::endl;
}

CoarseResult ObservationModel::coarse_match(
    const uint8_t* mono_data, int width, int height,
    const std::vector<int>* candidate_indices, int top_k)
{
    return coarse_->match(mono_data, width, height, candidate_indices, top_k);
}

cv::Mat ObservationModel::load_patch(const std::string& name) {
    auto it = patch_cache_.find(name);
    if (it != patch_cache_.end()) return it->second;

    std::string path = patches_dir_ + "/" + name + ".png";
    cv::Mat img = cv::imread(path);
    if (img.empty()) return {};

    if (static_cast<int>(patch_cache_.size()) >= cfg_.patch_cache_size && !cache_order_.empty()) {
        patch_cache_.erase(cache_order_.front());
        cache_order_.erase(cache_order_.begin());
    }
    patch_cache_[name] = img;
    cache_order_.push_back(name);
    return img;
}

std::optional<FineResult> ObservationModel::fine_match(
    const uint8_t* mono_data, int width, int height,
    const std::string& patch_name, double context_fraction)
{
    pose_->set_altitude(altitude_m);
    int res = cfg_.matcher_resolution;

    // Build context patch
    cv::Mat patch_bgr;
    FlatMeta flat_meta;

    if (cfg_.context_enabled) {
        // Stitch context patch from neighbors
        try {
            auto [row, col] = parse_patch_name(patch_name);
            auto& center_meta = db_->patch_meta(patch_name);
            auto& cb = center_meta.bounds;

            cv::Mat center_img = load_patch(patch_name);
            if (center_img.empty()) return std::nullopt;
            int tile_h = center_img.rows, tile_w = center_img.cols;

            double tile_lat_span = cb.max_lat - cb.min_lat;
            double tile_lon_span = cb.max_lon - cb.min_lon;
            double lat_per_px = tile_lat_span / (tile_h - 1);
            double lon_per_px = tile_lon_span / (tile_w - 1);

            double ext_lat = context_fraction * tile_lat_span;
            double ext_lon = context_fraction * tile_lon_span;
            double comp_min_lat = cb.min_lat - ext_lat;
            double comp_max_lat = cb.max_lat + ext_lat;
            double comp_min_lon = cb.min_lon - ext_lon;
            double comp_max_lon = cb.max_lon + ext_lon;

            int comp_h = std::max(1, static_cast<int>(std::round((comp_max_lat - comp_min_lat) / lat_per_px) + 1));
            int comp_w = std::max(1, static_cast<int>(std::round((comp_max_lon - comp_min_lon) / lon_per_px) + 1));

            cv::Mat composite = cv::Mat::zeros(comp_h, comp_w, CV_8UC3);

            for (int dr = -2; dr <= 2; ++dr) {
                for (int dc = -2; dc <= 2; ++dc) {
                    std::string nname = "patch_" + std::to_string(row + dr) + "_" + std::to_string(col + dc);
                    PatchMeta nb_meta;
                    try { nb_meta = db_->patch_meta(nname); } catch (...) { continue; }
                    auto& nb = nb_meta.bounds;

                    if (nb.max_lat <= comp_min_lat || nb.min_lat >= comp_max_lat ||
                        nb.max_lon <= comp_min_lon || nb.min_lon >= comp_max_lon) continue;

                    cv::Mat timg = (dr == 0 && dc == 0) ? center_img : load_patch(nname);
                    if (timg.empty()) continue;

                    int th = timg.rows, tw = timg.cols;
                    int x_off = static_cast<int>(std::round((nb.min_lon - comp_min_lon) / lon_per_px));
                    int y_off = static_cast<int>(std::round((comp_max_lat - nb.max_lat) / lat_per_px));

                    int src_x0 = std::max(0, -x_off), dst_x0 = std::max(0, x_off);
                    int src_y0 = std::max(0, -y_off), dst_y0 = std::max(0, y_off);
                    int src_x1 = std::min(tw, comp_w - x_off);
                    int src_y1 = std::min(th, comp_h - y_off);

                    if (src_x1 > src_x0 && src_y1 > src_y0) {
                        timg(cv::Rect(src_x0, src_y0, src_x1 - src_x0, src_y1 - src_y0))
                            .copyTo(composite(cv::Rect(dst_x0, dst_y0,
                                                       src_x1 - src_x0, src_y1 - src_y0)));
                    }
                }
            }

            patch_bgr = composite;
            flat_meta = {comp_min_lat, comp_max_lat, comp_min_lon, comp_max_lon, comp_h, comp_w};
        } catch (...) {
            return std::nullopt;
        }
    } else {
        patch_bgr = load_patch(patch_name);
        if (patch_bgr.empty()) return std::nullopt;
        auto& pm = db_->patch_meta(patch_name);
        flat_meta = {pm.bounds.min_lat, pm.bounds.max_lat,
                     pm.bounds.min_lon, pm.bounds.max_lon,
                     patch_bgr.rows, patch_bgr.cols};
    }

    // Run ELoFTR matching
    auto match_out = fine_->match(mono_data, width, height,
                                   patch_bgr.data, patch_bgr.cols, patch_bgr.rows);
    if (match_out.num_matches < cfg_.min_inliers_ransac) return std::nullopt;

    // Flow metrics
    auto flow = compute_flow_metrics(match_out.keypoints0, match_out.keypoints1);

    // Scale patch keypoints to composite pixel space
    Eigen::MatrixXf mkpts1_patch = match_out.keypoints1;
    mkpts1_patch.col(0) *= static_cast<float>(flat_meta.patch_w) / res;
    mkpts1_patch.col(1) *= static_cast<float>(flat_meta.patch_h) / res;

    // PnP
    auto pnp_result = pose_->solve_pnp(match_out.keypoints0, mkpts1_patch, flat_meta);
    if (pnp_result.has_value()) {
        auto& r = pnp_result.value();
        if (r.inliers >= cfg_.min_inliers) {
            return FineResult{r.lat, r.lon, r.inliers, "pnp", r.heading_deg,
                             patch_name, flow.flow_consistency, flow.flow_magnitude_cv,
                             static_cast<float>(r.inliers) / match_out.num_matches,
                             flow.flow_heading_deg, match_out.num_matches, r.estimated_altitude};
        }
    }

    return std::nullopt;
}

std::optional<FineResult> ObservationModel::fine_match_on_satellite(
    const uint8_t* mono_data, int width, int height,
    const cv::Mat& satellite_crop,
    const FlatMeta& mosaic_meta,
    const cv::Mat& warp_M_inv)
{
    pose_->set_altitude(altitude_m);
    int res = cfg_.matcher_resolution;
    int sh = satellite_crop.rows, sw = satellite_crop.cols;

    auto match_out = fine_->match(mono_data, width, height,
                                   satellite_crop.data, sw, sh);
    if (match_out.num_matches < cfg_.min_inliers_ransac) return std::nullopt;

    auto flow = compute_flow_metrics(match_out.keypoints0, match_out.keypoints1);

    // Scale keypoints from matcher res to satellite_crop pixel space
    Eigen::MatrixXf mkpts1_sat = match_out.keypoints1;
    mkpts1_sat.col(0) *= static_cast<float>(sw) / res;
    mkpts1_sat.col(1) *= static_cast<float>(sh) / res;

    // Un-warp: satellite_crop pixels → North-up mosaic pixels via perspective M_inv (3x3)
    int n = mkpts1_sat.rows();
    Eigen::MatrixXf mkpts1_northup(n, 2);
    for (int i = 0; i < n; ++i) {
        double x = mkpts1_sat(i, 0), y = mkpts1_sat(i, 1);
        double w_h = warp_M_inv.at<double>(2, 0) * x + warp_M_inv.at<double>(2, 1) * y + warp_M_inv.at<double>(2, 2);
        if (std::abs(w_h) < 1e-8) w_h = 1e-8;
        mkpts1_northup(i, 0) = static_cast<float>(
            (warp_M_inv.at<double>(0, 0) * x + warp_M_inv.at<double>(0, 1) * y + warp_M_inv.at<double>(0, 2)) / w_h);
        mkpts1_northup(i, 1) = static_cast<float>(
            (warp_M_inv.at<double>(1, 0) * x + warp_M_inv.at<double>(1, 1) * y + warp_M_inv.at<double>(1, 2)) / w_h);
    }

    FlatMeta north_meta = mosaic_meta;

    auto pnp_result = pose_->solve_pnp(match_out.keypoints0, mkpts1_northup, north_meta);
    if (pnp_result.has_value() && pnp_result->inliers >= cfg_.min_inliers) {
        auto& r = pnp_result.value();
        return FineResult{r.lat, r.lon, r.inliers, "pnp", r.heading_deg,
                         "satellite", flow.flow_consistency, flow.flow_magnitude_cv,
                         static_cast<float>(r.inliers) / match_out.num_matches,
                         flow.flow_heading_deg, match_out.num_matches, r.estimated_altitude};
    }

    // Homography fallback
    auto homo_result = pose_->solve_homography(match_out.keypoints0, mkpts1_northup, north_meta, res);
    if (homo_result.has_value() && homo_result->inliers >= cfg_.min_inliers) {
        auto& r = homo_result.value();
        return FineResult{r.lat, r.lon, r.inliers, "homography", std::nullopt,
                         "satellite", flow.flow_consistency, flow.flow_magnitude_cv,
                         static_cast<float>(r.inliers) / match_out.num_matches,
                         flow.flow_heading_deg, match_out.num_matches, 0.0};
    }

    return std::nullopt;
}

std::optional<FineResult> ObservationModel::fine_match_on_mosaic(
    const uint8_t* mono_data, int width, int height,
    const cv::Mat& mosaic_rotated,
    const FlatMeta& mosaic_meta,
    const cv::Mat& rot_crop_M_inv)
{
    pose_->set_altitude(altitude_m);
    int res = cfg_.matcher_resolution;
    int mh = mosaic_rotated.rows, mw = mosaic_rotated.cols;

    auto match_out = fine_->match(mono_data, width, height,
                                   mosaic_rotated.data, mw, mh);
    if (match_out.num_matches < cfg_.min_inliers_ransac) return std::nullopt;

    auto flow = compute_flow_metrics(match_out.keypoints0, match_out.keypoints1);

    // Scale to mosaic_rotated pixel space
    Eigen::MatrixXf mkpts1_rot = match_out.keypoints1;
    mkpts1_rot.col(0) *= static_cast<float>(mw) / res;
    mkpts1_rot.col(1) *= static_cast<float>(mh) / res;

    // Map back to North-up mosaic via inverse affine
    int n = mkpts1_rot.rows();
    Eigen::MatrixXf mkpts1_northup(n, 2);
    for (int i = 0; i < n; ++i) {
        double x = mkpts1_rot(i, 0), y = mkpts1_rot(i, 1);
        mkpts1_northup(i, 0) = static_cast<float>(
            rot_crop_M_inv.at<double>(0, 0) * x + rot_crop_M_inv.at<double>(0, 1) * y + rot_crop_M_inv.at<double>(0, 2));
        mkpts1_northup(i, 1) = static_cast<float>(
            rot_crop_M_inv.at<double>(1, 0) * x + rot_crop_M_inv.at<double>(1, 1) * y + rot_crop_M_inv.at<double>(1, 2));
    }

    FlatMeta north_meta = mosaic_meta;

    auto pnp_result = pose_->solve_pnp(match_out.keypoints0, mkpts1_northup, north_meta);
    if (pnp_result.has_value() && pnp_result->inliers >= cfg_.min_inliers) {
        auto& r = pnp_result.value();
        return FineResult{r.lat, r.lon, r.inliers, "pnp", r.heading_deg,
                         "mosaic", flow.flow_consistency, flow.flow_magnitude_cv,
                         static_cast<float>(r.inliers) / match_out.num_matches,
                         flow.flow_heading_deg, match_out.num_matches, r.estimated_altitude};
    }

    return std::nullopt;
}

std::pair<double, double> ObservationModel::get_patch_center_enu(const std::string& name) const {
    for (int i = 0; i < db_->num_patches(); ++i) {
        if (db_->patch_name(i) == name)
            return db_->patch_center_enu(i);
    }
    return {0.0, 0.0};
}

std::vector<int> ObservationModel::get_indices_within_radius(
    double east, double north, double radius_m) const
{
    return db_->indices_within_radius(east, north, radius_m);
}

}  // namespace pf
