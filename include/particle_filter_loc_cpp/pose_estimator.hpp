#pragma once

#include <optional>
#include <vector>

#include <Eigen/Dense>

#include "types.hpp"

namespace pf {

struct PnPResult {
    double lat, lon;
    double heading_deg;
    int inliers;
    double estimated_altitude = 0.0;
    std::vector<bool> inlier_mask;
};

struct HomographyResult {
    double lat, lon;
    int inliers;
    std::vector<bool> inlier_mask;
};

class PoseEstimator {
public:
    PoseEstimator(double fx, double fy, double cx, double cy,
                  double altitude_m = 50.0,
                  int min_inliers_ransac = 4,
                  int min_inliers = 8,
                  int ransac_max_iters = 1000);

    void set_altitude(double altitude_m) { altitude_m_ = altitude_m; }

    // PnP with RANSAC (PoseLib if available, OpenCV fallback)
    std::optional<PnPResult> solve_pnp(
        const Eigen::MatrixXf& mkpts_drone,
        const Eigen::MatrixXf& mkpts_patch_px,
        const FlatMeta& meta);

    // Homography with RANSAC
    std::optional<HomographyResult> solve_homography(
        const Eigen::MatrixXf& mkpts_drone,
        const Eigen::MatrixXf& mkpts_patch,
        const FlatMeta& meta,
        int drone_res);

private:
    double fx_, fy_, cx_, cy_;
    double altitude_m_;
    int min_inliers_ransac_, min_inliers_;
    int ransac_max_iters_;
};

}  // namespace pf
