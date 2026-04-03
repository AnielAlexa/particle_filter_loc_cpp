#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include "geo_utils.hpp"

namespace pf {

struct FootprintInfo {
    double altitude_m;
    double drone_gsd;
    double sat_gsd;
    double footprint_w_m;
    double footprint_h_m;
    double scale_ratio;
    double n_patches_spanned;
};

inline FootprintInfo compute_footprint(
    double fx, double fy, int img_w, int img_h, double altitude_m,
    double sat_gsd = 0.298, double patch_ground_size_m = 100.0)
{
    double drone_gsd = altitude_m / fx;
    double footprint_w = altitude_m * img_w / fx;
    double footprint_h = altitude_m * img_h / fy;
    double scale_ratio = drone_gsd / sat_gsd;
    double n_patches = std::max(footprint_w, footprint_h) / patch_ground_size_m;
    return {altitude_m, drone_gsd, sat_gsd, footprint_w, footprint_h,
            scale_ratio, n_patches};
}

inline double sigma_obs_coarse(const FootprintInfo& fp, double base_sigma = 50.0) {
    double floor = std::max(fp.footprint_w_m, fp.footprint_h_m) / 2.0;
    double scaled = base_sigma * std::max(1.0, fp.n_patches_spanned);
    return std::max(scaled, floor);
}

inline double sigma_obs_fine(const FootprintInfo& fp, double base_sigma = 5.0) {
    return base_sigma * std::max(1.0, fp.scale_ratio);
}

inline double context_fraction_from_footprint(
    const FootprintInfo& fp, double patch_ground_size_m = 100.0,
    double margin_factor = 1.3)
{
    double needed = std::max(fp.footprint_w_m, fp.footprint_h_m) * margin_factor;
    double extra = std::max(0.0, (needed - patch_ground_size_m) / 2.0);
    return extra / patch_ground_size_m;
}

// Returns [4][2] array of (lat, lon) for footprint corners: TL, TR, BR, BL
inline std::array<std::array<double, 2>, 4> compute_footprint_corners_gps(
    double fx, double fy, int img_w, int img_h,
    double altitude_m, double heading_deg,
    double center_lat, double center_lon, const ENUFrame& enu)
{
    double half_w = altitude_m * img_w / (2.0 * fx);
    double half_h = altitude_m * img_h / (2.0 * fy);

    // Corners in local body frame: TL, TR, BR, BL
    double corners_local[4][2] = {
        {-half_w,  half_h},
        { half_w,  half_h},
        { half_w, -half_h},
        {-half_w, -half_h},
    };

    double theta = heading_deg * M_PI / 180.0;
    double cos_t = std::cos(theta), sin_t = std::sin(theta);

    auto [center_e, center_n] = enu.wgs84_to_enu(center_lat, center_lon);

    std::array<std::array<double, 2>, 4> corners_gps;
    for (int i = 0; i < 4; ++i) {
        double x = corners_local[i][0], y = corners_local[i][1];
        double e = x * cos_t + y * sin_t;
        double n = -x * sin_t + y * cos_t;
        auto [lat, lon] = enu.enu_to_wgs84(center_e + e, center_n + n);
        corners_gps[i] = {lat, lon};
    }
    return corners_gps;
}

inline std::array<double, 4> scale_intrinsics(
    double fx, double fy, double cx, double cy,
    int orig_w, int orig_h, int target_w, int target_h)
{
    double sx = static_cast<double>(target_w) / orig_w;
    double sy = static_cast<double>(target_h) / orig_h;
    return {fx * sx, fy * sy, cx * sx, cy * sy};
}

}  // namespace pf
