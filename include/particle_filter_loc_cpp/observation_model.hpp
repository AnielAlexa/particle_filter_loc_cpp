#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "coarse_matcher.hpp"
#include "config.hpp"
#include "descriptor_database.hpp"
#include "fine_matcher.hpp"
#include "footprint_reconstruction.hpp"
#include "geo_utils.hpp"
#include "pose_estimator.hpp"
#include "types.hpp"
#include "vpi_preprocessor.hpp"

namespace pf {

class ObservationModel {
public:
    ObservationModel(const MatcherConfig& cfg, const ENUFrame& enu);

    CoarseResult coarse_match(const uint8_t* mono_data, int width, int height,
                              const std::vector<int>* candidate_indices = nullptr,
                              int top_k = 5);

    std::optional<FineResult> fine_match(const uint8_t* mono_data, int width, int height,
                                         const std::string& patch_name,
                                         double context_fraction = 0.2);

    std::optional<FineResult> fine_match_on_satellite(
        const uint8_t* mono_data, int width, int height,
        const cv::Mat& satellite_crop,
        const FlatMeta& mosaic_meta,
        const cv::Mat& warp_M_inv);

    std::optional<FineResult> fine_match_on_mosaic(
        const uint8_t* mono_data, int width, int height,
        const cv::Mat& mosaic_rotated,
        const FlatMeta& mosaic_meta,
        const cv::Mat& rot_crop_M_inv);

    std::pair<double, double> get_patch_center_enu(const std::string& name) const;
    std::vector<int> get_indices_within_radius(double east, double north, double radius_m) const;

    SatelliteFootprintReconstructor& footprint() { return *footprint_; }

    double altitude_m = 50.0;
    const std::string& patches_dir() const { return patches_dir_; }

private:
    cv::Mat load_patch(const std::string& name);

    MatcherConfig cfg_;
    ENUFrame enu_;
    std::string script_dir_;
    std::string patches_dir_;

    std::unique_ptr<VpiPreprocessor> vpi_;
    std::unique_ptr<CoarseMatcher> coarse_;
    std::unique_ptr<FineMatcher> fine_;
    std::unique_ptr<PoseEstimator> pose_;
    std::unique_ptr<DescriptorDatabase> db_;
    std::unique_ptr<SatelliteFootprintReconstructor> footprint_;

    // Patch LRU cache
    std::unordered_map<std::string, cv::Mat> patch_cache_;
    std::vector<std::string> cache_order_;
};

}  // namespace pf
