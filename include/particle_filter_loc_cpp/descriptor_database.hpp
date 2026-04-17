#pragma once

#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "geo_utils.hpp"
#include "types.hpp"

namespace pf {

class DescriptorDatabase {
public:
    DescriptorDatabase(const std::string& npy_path,
                       const std::string& patch_names_path,
                       const std::string& gps_metadata_path,
                       const ENUFrame& enu,
                       bool load_descriptors = true);

    // Cosine similarity top-K search
    CoarseResult cosine_top_k(const float* query_desc, int desc_dim,
                              const std::vector<int>* candidate_indices,
                              int top_k) const;

    // Spatial query
    std::vector<int> indices_within_radius(double east, double north, double radius_m) const;

    const std::string& patch_name(int idx) const { return patch_names_[idx]; }
    std::pair<double, double> patch_center_enu(int idx) const {
        return {patch_enu_(idx, 0), patch_enu_(idx, 1)};
    }
    const PatchMeta& patch_meta(const std::string& name) const;
    int num_patches() const { return n_patches_; }
    int descriptor_dim() const { return desc_dim_; }

private:
    void load_npy(const std::string& path);
    void load_patch_names(const std::string& path);
    void load_gps_metadata(const std::string& path, const ENUFrame& enu);

    // Row-major to match numpy C-order layout from .npy files
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> descriptors_;  // [N, D] L2-normalized
    std::vector<std::string> patch_names_;
    Eigen::MatrixXd patch_enu_;    // [N, 2]
    std::unordered_map<std::string, PatchMeta> gps_meta_;
    int n_patches_ = 0;
    int desc_dim_ = 0;
};

}  // namespace pf
