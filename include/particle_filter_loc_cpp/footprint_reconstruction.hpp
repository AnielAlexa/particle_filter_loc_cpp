#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

#include "geo_utils.hpp"
#include "types.hpp"

namespace pf {

struct FootprintReconstruction {
    cv::Mat satellite_crop;
    double footprint_w_m, footprint_h_m;
    std::vector<std::string> source_tiles;
    cv::Mat mosaic_bgr;
    FlatMeta mosaic_meta;
    cv::Mat mosaic_rotated;
    cv::Mat warp_M_inv;        // 3x3: satellite_crop px -> mosaic px
    cv::Mat rot_crop_M_inv;    // 2x3: mosaic_rotated px -> mosaic px
    double heading_deg = 0.0;
};

class SatelliteFootprintReconstructor {
public:
    SatelliteFootprintReconstructor(
        const std::unordered_map<std::string, PatchMeta>& gps_metadata,
        const std::string& patches_dir,
        const ENUFrame& enu);

    std::optional<FootprintReconstruction> reconstruct(
        double lat, double lon,
        double altitude_m, double heading_deg,
        double fx, double fy, int img_w, int img_h,
        cv::Size output_size = {640, 480},
        double mosaic_context_scale = 2.0);

private:
    cv::Mat load_tile(const std::string& name);
    std::tuple<cv::Mat, FlatMeta, std::vector<std::string>> stitch_mosaic(
        double min_lat, double max_lat, double min_lon, double max_lon);

    const std::unordered_map<std::string, PatchMeta>& gps_meta_;
    std::string patches_dir_;
    ENUFrame enu_;

    // Tile info
    std::vector<std::string> tile_names_;
    std::vector<double> tile_min_lat_, tile_max_lat_, tile_min_lon_, tile_max_lon_;
    double lat_per_px_, lon_per_px_;
    int tile_h_ = 400, tile_w_ = 400;

    // LRU cache
    std::unordered_map<std::string, cv::Mat> tile_cache_;
    std::vector<std::string> cache_order_;
    static constexpr int MAX_CACHE = 200;
};

}  // namespace pf
