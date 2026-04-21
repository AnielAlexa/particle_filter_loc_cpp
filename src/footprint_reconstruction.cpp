#include "particle_filter_loc_cpp/footprint_reconstruction.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "particle_filter_loc_cpp/camera_footprint.hpp"

namespace pf {

SatelliteFootprintReconstructor::SatelliteFootprintReconstructor(
    const std::unordered_map<std::string, PatchMeta>& gps_metadata,
    const std::string& patches_dir,
    const ENUFrame& enu)
    : gps_meta_(gps_metadata), patches_dir_(patches_dir), enu_(enu)
{
    for (auto& [name, meta] : gps_metadata) {
        tile_names_.push_back(name);
        tile_min_lat_.push_back(meta.bounds.min_lat);
        tile_max_lat_.push_back(meta.bounds.max_lat);
        tile_min_lon_.push_back(meta.bounds.min_lon);
        tile_max_lon_.push_back(meta.bounds.max_lon);
    }

    // Compute GSD from first tile
    if (!tile_names_.empty()) {
        auto& first_b = gps_metadata.at(tile_names_[0]).bounds;
        cv::Mat sample = load_tile(tile_names_[0]);
        if (!sample.empty()) {
            tile_h_ = sample.rows;
            tile_w_ = sample.cols;
        }
        lat_per_px_ = (first_b.max_lat - first_b.min_lat) / (tile_h_ - 1);
        lon_per_px_ = (first_b.max_lon - first_b.min_lon) / (tile_w_ - 1);
    }
}

cv::Mat SatelliteFootprintReconstructor::load_tile(const std::string& name) {
    auto it = tile_cache_.find(name);
    if (it != tile_cache_.end()) return it->second;

    std::string path = patches_dir_ + "/" + name + ".png";
    cv::Mat img = cv::imread(path);
    if (img.empty()) return {};

    if (static_cast<int>(tile_cache_.size()) >= MAX_CACHE && !cache_order_.empty()) {
        tile_cache_.erase(cache_order_.front());
        cache_order_.erase(cache_order_.begin());
    }
    tile_cache_[name] = img;
    cache_order_.push_back(name);
    return img;
}

std::tuple<cv::Mat, FlatMeta, std::vector<std::string>>
SatelliteFootprintReconstructor::stitch_mosaic(
    double min_lat, double max_lat, double min_lon, double max_lon)
{
    std::vector<int> overlapping;
    for (int i = 0; i < static_cast<int>(tile_names_.size()); ++i) {
        if (tile_max_lat_[i] > min_lat && tile_min_lat_[i] < max_lat &&
            tile_max_lon_[i] > min_lon && tile_min_lon_[i] < max_lon)
            overlapping.push_back(i);
    }

    if (overlapping.empty()) return {{}, {}, {}};

    int comp_h = std::max(1, static_cast<int>(std::round((max_lat - min_lat) / lat_per_px_) + 1));
    int comp_w = std::max(1, static_cast<int>(std::round((max_lon - min_lon) / lon_per_px_) + 1));

    cv::Mat composite = cv::Mat::zeros(comp_h, comp_w, CV_8UC3);
    std::vector<std::string> source_tiles;

    for (int idx : overlapping) {
        cv::Mat timg = load_tile(tile_names_[idx]);
        if (timg.empty()) continue;
        source_tiles.push_back(tile_names_[idx]);

        int th = timg.rows, tw = timg.cols;
        int x_off = static_cast<int>(std::round((tile_min_lon_[idx] - min_lon) / lon_per_px_));
        int y_off = static_cast<int>(std::round((max_lat - tile_max_lat_[idx]) / lat_per_px_));

        int src_x0 = std::max(0, -x_off), dst_x0 = std::max(0, x_off);
        int src_y0 = std::max(0, -y_off), dst_y0 = std::max(0, y_off);
        int src_x1 = std::min(tw, comp_w - x_off);
        int src_y1 = std::min(th, comp_h - y_off);
        int dst_x1 = dst_x0 + (src_x1 - src_x0);
        int dst_y1 = dst_y0 + (src_y1 - src_y0);

        if (src_x1 > src_x0 && src_y1 > src_y0) {
            timg(cv::Rect(src_x0, src_y0, src_x1 - src_x0, src_y1 - src_y0))
                .copyTo(composite(cv::Rect(dst_x0, dst_y0, dst_x1 - dst_x0, dst_y1 - dst_y0)));
        }
    }

    FlatMeta meta{min_lat, max_lat, min_lon, max_lon, comp_h, comp_w};
    return {composite, meta, source_tiles};
}

std::optional<FootprintReconstruction> SatelliteFootprintReconstructor::reconstruct(
    double lat, double lon,
    double altitude_m, double heading_deg,
    double fx, double fy, int img_w, int img_h,
    cv::Size output_size, double mosaic_context_scale,
    double satellite_context_scale)
{
    double fw = altitude_m * img_w / fx;
    double fh = altitude_m * img_h / fy;

    // Footprint corners
    auto corners = compute_footprint_corners_gps(fx, fy, img_w, img_h, altitude_m, heading_deg, lat, lon, enu_);

    // Scale satellite footprint corners outward from center if context > 1.0
    if (satellite_context_scale > 1.001) {
        for (auto& c : corners) {
            c[0] = lat + (c[0] - lat) * satellite_context_scale;
            c[1] = lon + (c[1] - lon) * satellite_context_scale;
        }
    }

    // Mosaic extent
    double diag = std::sqrt(fw * fw + fh * fh);
    double square_side = diag * mosaic_context_scale;
    double mosaic_radius = square_side * std::sqrt(2.0) / 2.0;

    double lat_rad = lat * M_PI / 180.0;
    double r_lat = mosaic_radius / 111320.0;
    double r_lon = mosaic_radius / (111320.0 * std::cos(lat_rad));

    auto [mosaic, meta, tiles] = stitch_mosaic(lat - r_lat, lat + r_lat, lon - r_lon, lon + r_lon);
    if (mosaic.empty()) return std::nullopt;

    int m_h = meta.patch_h, m_w = meta.patch_w;

    // Drone position in mosaic pixel space
    double center_px_x = (lon - meta.min_lon) / (meta.max_lon - meta.min_lon) * (m_w - 1);
    double center_px_y = (meta.max_lat - lat) / (meta.max_lat - meta.min_lat) * (m_h - 1);

    // Perspective warp for satellite_crop
    cv::Point2f src_pts[4], dst_pts[4];
    for (int i = 0; i < 4; ++i) {
        double c_lat = corners[i][0], c_lon = corners[i][1];
        float px = static_cast<float>((c_lon - meta.min_lon) / (meta.max_lon - meta.min_lon) * (m_w - 1));
        float py = static_cast<float>((meta.max_lat - c_lat) / (meta.max_lat - meta.min_lat) * (m_h - 1));
        src_pts[i] = {px, py};
    }
    dst_pts[0] = {0, 0};
    dst_pts[1] = {static_cast<float>(output_size.width - 1), 0};
    dst_pts[2] = {static_cast<float>(output_size.width - 1), static_cast<float>(output_size.height - 1)};
    dst_pts[3] = {0, static_cast<float>(output_size.height - 1)};

    // Validate src_pts to avoid degenerate perspective transforms that crash warpPerspective
    for (int i = 0; i < 4; ++i) {
        if (!std::isfinite(src_pts[i].x) || !std::isfinite(src_pts[i].y))
            return std::nullopt;
    }

    cv::Mat M = cv::getPerspectiveTransform(src_pts, dst_pts);
    cv::Mat M_inv = cv::getPerspectiveTransform(dst_pts, src_pts);

    // Validate M (getPerspectiveTransform can return NaN on near-degenerate input)
    if (!cv::checkRange(M) || !cv::checkRange(M_inv))
        return std::nullopt;

    cv::Mat warped;
    cv::warpPerspective(mosaic, warped, M, output_size);

    // Heading-aligned mosaic. Crop size matches output_size aspect so the
    // downstream matcher receives a rectangle with the drone's ground-footprint
    // aspect (no non-uniform stretch in VpiPreprocessor::prepare_fine_patch).
    double gsd_lon = (meta.max_lon - meta.min_lon) / (m_w - 1) * 111320.0 * std::cos(lat_rad);
    double gsd_lat = (meta.max_lat - meta.min_lat) / (m_h - 1) * 111320.0;
    double gsd = (gsd_lon + gsd_lat) / 2.0;
    double aspect = static_cast<double>(output_size.height) / output_size.width;
    int S_w = std::max(64, static_cast<int>(std::round(square_side / gsd)));
    int S_h = std::max(64, static_cast<int>(std::round(S_w * aspect)));

    cv::Mat rot_mat = cv::getRotationMatrix2D(
        {static_cast<float>(center_px_x), static_cast<float>(center_px_y)}, heading_deg, 1.0);
    rot_mat.at<double>(0, 2) += S_w / 2.0 - center_px_x;
    rot_mat.at<double>(1, 2) += S_h / 2.0 - center_px_y;

    cv::Mat mosaic_rot;
    cv::warpAffine(mosaic, mosaic_rot, rot_mat, {S_w, S_h});

    // Inverse affine
    cv::Mat M_fwd_3x3 = cv::Mat::zeros(3, 3, CV_64F);
    rot_mat.copyTo(M_fwd_3x3(cv::Rect(0, 0, 3, 2)));
    M_fwd_3x3.at<double>(2, 2) = 1.0;
    cv::Mat rot_crop_M_inv_3x3 = M_fwd_3x3.inv();
    cv::Mat rot_crop_M_inv = rot_crop_M_inv_3x3(cv::Rect(0, 0, 3, 2)).clone();

    FootprintReconstruction result;
    result.satellite_crop = warped;
    result.footprint_w_m = fw;
    result.footprint_h_m = fh;
    result.source_tiles = tiles;
    result.mosaic_bgr = mosaic;
    result.mosaic_meta = meta;
    result.mosaic_rotated = mosaic_rot;
    result.warp_M_inv = M_inv;
    result.rot_crop_M_inv = rot_crop_M_inv;
    result.heading_deg = heading_deg;
    return result;
}

}  // namespace pf
