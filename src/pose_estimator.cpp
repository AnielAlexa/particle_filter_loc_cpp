#include "particle_filter_loc_cpp/pose_estimator.hpp"

#include <cmath>
#include <iostream>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#ifdef HAS_POSELIB
#include <PoseLib/poselib.h>
#endif

namespace pf {

PoseEstimator::PoseEstimator(
    double fx, double fy, double cx, double cy,
    double altitude_m, int min_inliers_ransac, int min_inliers, int ransac_max_iters)
    : fx_(fx), fy_(fy), cx_(cx), cy_(cy), altitude_m_(altitude_m),
      min_inliers_ransac_(min_inliers_ransac), min_inliers_(min_inliers),
      ransac_max_iters_(ransac_max_iters) {}

std::optional<PnPResult> PoseEstimator::solve_pnp(
    const Eigen::MatrixXf& mkpts_drone,
    const Eigen::MatrixXf& mkpts_patch_px,
    const FlatMeta& meta)
{
    int n = mkpts_drone.rows();
    if (n < min_inliers_ransac_) return std::nullopt;

    // Convert patch pixel coords to local ENU (meters) with Z=0
    double center_lat = (meta.min_lat + meta.max_lat) / 2.0;
    double center_lon = (meta.min_lon + meta.max_lon) / 2.0;
    double cos_lat = std::cos(center_lat * M_PI / 180.0);

    std::vector<cv::Point3f> obj_pts(n);
    std::vector<cv::Point2f> img_pts(n);

    for (int i = 0; i < n; ++i) {
        double lat = meta.max_lat - mkpts_patch_px(i, 1) *
                     (meta.max_lat - meta.min_lat) / (meta.patch_h - 1);
        double lon = meta.min_lon + mkpts_patch_px(i, 0) *
                     (meta.max_lon - meta.min_lon) / (meta.patch_w - 1);
        double x = (lon - center_lon) * cos_lat * 111319.5;
        double y = (lat - center_lat) * 111319.5;
        obj_pts[i] = {static_cast<float>(x), static_cast<float>(y), 0.0f};
        img_pts[i] = {mkpts_drone(i, 0), mkpts_drone(i, 1)};
    }

#ifdef HAS_POSELIB
    // Use PoseLib for PnP (faster LO-RANSAC)
    std::vector<Eigen::Vector2d> pts2d(n);
    std::vector<Eigen::Vector3d> pts3d(n);
    for (int i = 0; i < n; ++i) {
        pts2d[i] = {mkpts_drone(i, 0), mkpts_drone(i, 1)};
        pts3d[i] = {obj_pts[i].x, obj_pts[i].y, 0.0};
    }

    poselib::Camera cam;
    cam.model_id = 0;  // SIMPLE_PINHOLE → use PINHOLE
    cam.width = static_cast<int>(cx_ * 2);
    cam.height = static_cast<int>(cy_ * 2);
    cam.params = {fx_, fy_, cx_, cy_};

    poselib::AbsolutePoseEstimateOptions opts;
    opts.ransac.max_iterations = ransac_max_iters_;
    opts.ransac.max_reproj_error = 8.0;

    poselib::CameraPose pose;
    std::vector<char> inlier_mask;
    auto stats = poselib::estimate_absolute_pose(pts2d, pts3d, cam, opts, &pose, &inlier_mask);

    int n_inliers = 0;
    for (auto c : inlier_mask) if (c) n_inliers++;
    if (n_inliers < min_inliers_ransac_) return std::nullopt;

    // Extract position and heading from PoseLib pose
    Eigen::Matrix3d R = pose.R();
    Eigen::Vector3d t = pose.t;
    Eigen::Vector3d cam_pos = -R.transpose() * t;

    if (!cam_pos.allFinite()) return std::nullopt;

    // Altitude sanity check
    double pnp_alt = cam_pos(2);
    if (altitude_m_ > 0 && std::abs(pnp_alt - altitude_m_) > altitude_m_ * 0.5)
        return std::nullopt;

    double cam_lat = center_lat + cam_pos(1) / 111319.5;
    double cam_lon = center_lon + cam_pos(0) / (111319.5 * cos_lat);
    double heading = std::fmod(std::atan2(R(1, 0), R(0, 0)) * 180.0 / M_PI, 360.0);
    if (heading < 0) heading += 360.0;

    PnPResult result;
    result.lat = cam_lat;
    result.lon = cam_lon;
    result.heading_deg = heading;
    result.inliers = n_inliers;
    result.inlier_mask.resize(n);
    for (int i = 0; i < n; ++i) result.inlier_mask[i] = inlier_mask[i] != 0;
    return result;

#else
    // OpenCV fallback
    cv::Mat K = (cv::Mat_<float>(3, 3) <<
        fx_, 0, cx_,
        0, fy_, cy_,
        0, 0, 1);
    cv::Mat tvec_init = (cv::Mat_<float>(3, 1) << 0, 0, altitude_m_);

    cv::Mat rvec, tvec, inlier_idx;
    bool ok = false;
    try {
        ok = cv::solvePnPRansac(obj_pts, img_pts, K, cv::noArray(),
                                rvec, tvec, false,
                                100, 8.0f, 0.99, inlier_idx,
                                cv::SOLVEPNP_ITERATIVE);
    } catch (...) { return std::nullopt; }

    if (!ok || inlier_idx.empty() || inlier_idx.rows < min_inliers_ransac_)
        return std::nullopt;

    // Check for NaN
    if (!std::isfinite(rvec.at<double>(0)) || !std::isfinite(tvec.at<double>(0))) {
        // Fallback: IPPE solver
        cv::Mat inlier_obj, inlier_img;
        for (int i = 0; i < inlier_idx.rows; ++i) {
            int idx = inlier_idx.at<int>(i);
            inlier_obj.push_back(obj_pts[idx]);
            inlier_img.push_back(img_pts[idx]);
        }
        try {
            ok = cv::solvePnP(inlier_obj, inlier_img, K, cv::noArray(),
                              rvec, tvec, false, cv::SOLVEPNP_IPPE);
        } catch (...) { return std::nullopt; }
        if (!ok || !std::isfinite(rvec.at<double>(0))) return std::nullopt;
    }

    // Refine on inliers
    cv::Mat inlier_obj_ref, inlier_img_ref;
    for (int i = 0; i < inlier_idx.rows; ++i) {
        int idx = inlier_idx.at<int>(i);
        inlier_obj_ref.push_back(obj_pts[idx]);
        inlier_img_ref.push_back(img_pts[idx]);
    }
    try {
        cv::Mat rv2 = rvec.clone(), tv2 = tvec.clone();
        bool ok2 = cv::solvePnP(inlier_obj_ref, inlier_img_ref, K, cv::noArray(),
                                rv2, tv2, true, cv::SOLVEPNP_ITERATIVE);
        if (ok2 && std::isfinite(tv2.at<double>(0))) {
            rvec = rv2; tvec = tv2;
        }
    } catch (...) {}

    cv::Mat R;
    cv::Rodrigues(rvec, R);
    cv::Mat cam_pos_cv = -R.t() * tvec;

    double cx = cam_pos_cv.at<double>(0);
    double cy_pos = cam_pos_cv.at<double>(1);
    double cz = cam_pos_cv.at<double>(2);

    if (!std::isfinite(cx) || !std::isfinite(cy_pos)) return std::nullopt;
    if (altitude_m_ > 0 && std::abs(cz - altitude_m_) > altitude_m_ * 0.5)
        return std::nullopt;

    double cam_lat = center_lat + cy_pos / 111319.5;
    double cam_lon = center_lon + cx / (111319.5 * cos_lat);
    double heading = std::fmod(std::atan2(R.at<double>(1, 0), R.at<double>(0, 0)) * 180.0 / M_PI, 360.0);
    if (heading < 0) heading += 360.0;

    PnPResult result;
    result.lat = cam_lat;
    result.lon = cam_lon;
    result.heading_deg = heading;
    result.inliers = inlier_idx.rows;
    result.inlier_mask.resize(n, false);
    for (int i = 0; i < inlier_idx.rows; ++i)
        result.inlier_mask[inlier_idx.at<int>(i)] = true;
    return result;
#endif
}

std::optional<HomographyResult> PoseEstimator::solve_homography(
    const Eigen::MatrixXf& mkpts_drone,
    const Eigen::MatrixXf& mkpts_patch,
    const FlatMeta& meta,
    int drone_res)
{
    int n = mkpts_drone.rows();
    if (n < 4) return std::nullopt;

#ifdef HAS_POSELIB
    // PoseLib homography
    std::vector<Eigen::Vector2d> pts1(n), pts2(n);
    for (int i = 0; i < n; ++i) {
        pts1[i] = {mkpts_drone(i, 0), mkpts_drone(i, 1)};
        pts2[i] = {mkpts_patch(i, 0), mkpts_patch(i, 1)};
    }

    Eigen::Matrix3d H;
    std::vector<char> inlier_mask;
    poselib::HomographyEstimateOptions opts;
    opts.ransac.max_iterations = ransac_max_iters_;
    opts.ransac.max_reproj_error = 5.0;
    poselib::estimate_homography(pts1, pts2, opts, &H, &inlier_mask);

    int n_inliers = 0;
    for (auto c : inlier_mask) if (c) n_inliers++;
    if (n_inliers < min_inliers_ransac_) return std::nullopt;

    // Project drone center through homography
    Eigen::Vector3d center(drone_res / 2.0, drone_res / 2.0, 1.0);
    Eigen::Vector3d proj = H * center;
    double px = proj(0) / proj(2);
    double py = proj(1) / proj(2);

#else
    // OpenCV homography
    std::vector<cv::Point2f> pts1(n), pts2(n);
    for (int i = 0; i < n; ++i) {
        pts1[i] = {mkpts_drone(i, 0), mkpts_drone(i, 1)};
        pts2[i] = {mkpts_patch(i, 0), mkpts_patch(i, 1)};
    }

    cv::Mat mask;
    cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, 5.0, mask, ransac_max_iters_);
    if (H.empty()) return std::nullopt;

    int n_inliers = cv::countNonZero(mask);
    if (n_inliers < min_inliers_ransac_) return std::nullopt;

    std::vector<cv::Point2f> pt_in = {{drone_res / 2.0f, drone_res / 2.0f}};
    std::vector<cv::Point2f> pt_out;
    cv::perspectiveTransform(pt_in, pt_out, H);
    double px = pt_out[0].x;
    double py = pt_out[0].y;

    std::vector<char> inlier_mask(n, 0);
    for (int i = 0; i < n; ++i)
        inlier_mask[i] = mask.at<uint8_t>(i);
#endif

    // Scale to patch pixel space and convert to GPS
    px = px * meta.patch_w / drone_res;
    py = py * meta.patch_h / drone_res;
    px = std::max(0.0, std::min(static_cast<double>(meta.patch_w - 1), px));
    py = std::max(0.0, std::min(static_cast<double>(meta.patch_h - 1), py));

    double lat = meta.max_lat - py * (meta.max_lat - meta.min_lat) / (meta.patch_h - 1);
    double lon = meta.min_lon + px * (meta.max_lon - meta.min_lon) / (meta.patch_w - 1);

    HomographyResult result;
    result.lat = lat;
    result.lon = lon;
    result.inliers = n_inliers;
    result.inlier_mask.resize(n);
    for (int i = 0; i < n; ++i) result.inlier_mask[i] = inlier_mask[i] != 0;
    return result;
}

}  // namespace pf
