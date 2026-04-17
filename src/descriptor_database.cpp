#include "particle_filter_loc_cpp/descriptor_database.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <nlohmann/json.hpp>

namespace pf {

void DescriptorDatabase::load_npy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Cannot open .npy file: " + path);

    // Parse numpy .npy format header
    char magic[6];
    f.read(magic, 6);
    if (magic[0] != '\x93' || std::string(magic + 1, 5) != "NUMPY")
        throw std::runtime_error("Invalid .npy magic: " + path);

    uint8_t major, minor;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);

    uint32_t header_len;
    if (major == 1) {
        uint16_t hl;
        f.read(reinterpret_cast<char*>(&hl), 2);
        header_len = hl;
    } else {
        f.read(reinterpret_cast<char*>(&header_len), 4);
    }

    std::string header(header_len, '\0');
    f.read(&header[0], header_len);

    // Parse shape from header: {'descr': '<f4', 'fortran_order': False, 'shape': (N, D), }
    auto parse_shape = [&]() -> std::pair<int, int> {
        auto pos = header.find("'shape'");
        if (pos == std::string::npos) pos = header.find("\"shape\"");
        pos = header.find('(', pos);
        auto end = header.find(')', pos);
        std::string shape_str = header.substr(pos + 1, end - pos - 1);
        // Remove spaces
        shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ' '), shape_str.end());
        auto comma = shape_str.find(',');
        int n = std::stoi(shape_str.substr(0, comma));
        int d = std::stoi(shape_str.substr(comma + 1));
        return {n, d};
    };

    auto [n, d] = parse_shape();
    n_patches_ = n;
    desc_dim_ = d;

    descriptors_.resize(n, d);
    f.read(reinterpret_cast<char*>(descriptors_.data()), n * d * sizeof(float));

    // L2 normalize rows
    for (int i = 0; i < n; ++i) {
        float norm = descriptors_.row(i).norm();
        if (norm > 1e-8f)
            descriptors_.row(i) /= norm;
    }

    std::cout << "[DB] Loaded " << n << " descriptors, dim=" << d << " from " << path << std::endl;
}

void DescriptorDatabase::load_patch_names(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open patch names: " + path);
    std::string line;
    while (std::getline(f, line)) {
        // Trim whitespace
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' '))
            line.pop_back();
        if (!line.empty())
            patch_names_.push_back(line);
    }
    if (static_cast<int>(patch_names_.size()) != n_patches_)
        throw std::runtime_error("Patch names count mismatch: " +
                                 std::to_string(patch_names_.size()) + " vs " + std::to_string(n_patches_));
}

void DescriptorDatabase::load_gps_metadata(const std::string& path, const ENUFrame& enu) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open GPS metadata: " + path);

    nlohmann::json j = nlohmann::json::parse(f);

    for (auto& [name, entry] : j.items()) {
        PatchMeta pm;
        pm.lat = entry["lat"].get<double>();
        pm.lon = entry["lon"].get<double>();
        auto& b = entry["bounds"];
        pm.bounds.min_lat = b["min_lat"].get<double>();
        pm.bounds.max_lat = b["max_lat"].get<double>();
        pm.bounds.min_lon = b["min_lon"].get<double>();
        pm.bounds.max_lon = b["max_lon"].get<double>();
        gps_meta_[name] = pm;
    }

    // Pre-compute ENU positions
    patch_enu_.resize(n_patches_, 2);
    for (int i = 0; i < n_patches_; ++i) {
        auto it = gps_meta_.find(patch_names_[i]);
        if (it != gps_meta_.end()) {
            auto [e, n] = enu.wgs84_to_enu(it->second.lat, it->second.lon);
            patch_enu_(i, 0) = e;
            patch_enu_(i, 1) = n;
        }
    }
}

DescriptorDatabase::DescriptorDatabase(
    const std::string& npy_path,
    const std::string& patch_names_path,
    const std::string& gps_metadata_path,
    const ENUFrame& enu,
    bool load_descriptors)
{
    if (load_descriptors) {
        load_npy(npy_path);
        load_patch_names(patch_names_path);
    } else {
        // Metadata-only mode: skip the large .npy descriptor matrix.
        // load_patch_names checks n_patches_, so derive it from the names file instead.
        std::ifstream f(patch_names_path);
        if (!f.is_open())
            throw std::runtime_error("Cannot open patch names: " + patch_names_path);
        std::string line;
        while (std::getline(f, line)) {
            while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' '))
                line.pop_back();
            if (!line.empty())
                patch_names_.push_back(line);
        }
        n_patches_ = static_cast<int>(patch_names_.size());
        desc_dim_ = 0;
        std::cout << "[DB] Metadata-only: " << n_patches_ << " patches (descriptors skipped)" << std::endl;
    }
    load_gps_metadata(gps_metadata_path, enu);
}

CoarseResult DescriptorDatabase::cosine_top_k(
    const float* query_desc, int desc_dim,
    const std::vector<int>* candidate_indices,
    int top_k) const
{
    Eigen::Map<const Eigen::VectorXf> query(query_desc, desc_dim);

    CoarseResult result;

    if (candidate_indices && !candidate_indices->empty()) {
        int nc = static_cast<int>(candidate_indices->size());
        int k = std::min(top_k, nc);

        Eigen::VectorXf sims(nc);
        for (int i = 0; i < nc; ++i)
            sims(i) = descriptors_.row((*candidate_indices)[i]).dot(query);

        // Partial sort for top-K
        std::vector<int> idx(nc);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int a, int b) { return sims(a) > sims(b); });

        for (int i = 0; i < k; ++i) {
            int local = idx[i];
            int global = (*candidate_indices)[local];
            result.top_k_indices.push_back(global);
            result.top_k_sims.push_back(sims(local));
            result.top_k_names.push_back(patch_names_[global]);
        }
    } else {
        int k = std::min(top_k, n_patches_);

        // Matrix-vector multiply: descriptors_ * query → sims
        Eigen::VectorXf sims = descriptors_ * query;

        std::vector<int> idx(n_patches_);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int a, int b) { return sims(a) > sims(b); });

        for (int i = 0; i < k; ++i) {
            result.top_k_indices.push_back(idx[i]);
            result.top_k_sims.push_back(sims(idx[i]));
            result.top_k_names.push_back(patch_names_[idx[i]]);
        }
    }

    return result;
}

std::vector<int> DescriptorDatabase::indices_within_radius(
    double east, double north, double radius_m) const
{
    double r2 = radius_m * radius_m;
    std::vector<int> result;
    for (int i = 0; i < n_patches_; ++i) {
        double dx = patch_enu_(i, 0) - east;
        double dy = patch_enu_(i, 1) - north;
        if (dx * dx + dy * dy <= r2)
            result.push_back(i);
    }
    return result;
}

const PatchMeta& DescriptorDatabase::patch_meta(const std::string& name) const {
    auto it = gps_meta_.find(name);
    if (it == gps_meta_.end())
        throw std::runtime_error("Unknown patch: " + name);
    return it->second;
}

}  // namespace pf
