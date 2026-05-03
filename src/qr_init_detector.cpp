#include "particle_filter_loc_cpp/qr_init_detector.hpp"

#include <algorithm>
#include <cmath>
#include <regex>

namespace pf {

namespace {

double wrap360(double deg) {
    double r = std::fmod(deg, 360.0);
    if (r < 0.0) r += 360.0;
    return r;
}

}  // namespace

QrInitDetector::QrInitDetector(double heading_offset_deg,
                               double min_side_px,
                               double max_aspect_skew)
    : heading_offset_deg_(heading_offset_deg),
      min_side_px_(min_side_px),
      max_aspect_skew_(max_aspect_skew) {}

std::optional<QrInitResult> QrInitDetector::detect(const cv::Mat& mono8) {
    if (mono8.empty()) return std::nullopt;

    std::vector<cv::Point2f> corners;
    std::string payload;
    try {
        payload = detector_.detectAndDecode(mono8, corners);
    } catch (const cv::Exception&) {
        return std::nullopt;
    }
    if (payload.empty() || corners.size() != 4) return std::nullopt;

    // Parse `geo:LAT,LON` (lat/lon may be signed decimals).
    static const std::regex re(R"(^\s*geo:\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$)");
    std::smatch m;
    if (!std::regex_match(payload, m, re)) return std::nullopt;

    double lat = std::stod(m[1].str());
    double lon = std::stod(m[2].str());

    // Corner convention from cv::QRCodeDetector: [TL, TR, BR, BL] in the QR's
    // own frame. The payload's "top" in the QR pattern points to true North in
    // world (sky-lign rotates the canvas via the device compass).
    const cv::Point2f TL = corners[0];
    const cv::Point2f TR = corners[1];
    const cv::Point2f BR = corners[2];
    const cv::Point2f BL = corners[3];

    auto len = [](cv::Point2f a, cv::Point2f b) {
        return std::hypot(a.x - b.x, a.y - b.y);
    };
    double s_top    = len(TL, TR);
    double s_right  = len(TR, BR);
    double s_bottom = len(BR, BL);
    double s_left   = len(BL, TL);
    double s_min = std::min({s_top, s_right, s_bottom, s_left});
    double s_max = std::max({s_top, s_right, s_bottom, s_left});
    if (s_min < min_side_px_) return std::nullopt;
    if ((s_max - s_min) / s_max > max_aspect_skew_) return std::nullopt;

    // QR-up direction in image pixel coordinates: from BL toward TL.
    // Image y points down, so flip y to express the vector in math axes.
    double up_x =  (TL.x - BL.x);
    double up_y = -(TL.y - BL.y);

    // φ_img = angle of QR-up vector measured CW from image-up (math: image-up
    // is +y after flipping). Equivalently atan2(x, y) gives CW-from-up.
    double phi_img_deg = std::atan2(up_x, up_y) * 180.0 / M_PI;

    // The reconstruction pipeline computes `heading_for_recon = drone_compass_yaw +
    // heading_offset_deg` where heading_for_recon is the world-compass angle of
    // image-up. Inverting: image-up world-compass = H + offset, and QR-up =
    // North = image-up rotated CW by φ_img → 0 = H + offset + φ_img (mod 360).
    double yaw_compass_deg = wrap360(-phi_img_deg - heading_offset_deg_);

    QrInitResult out;
    out.lat = lat;
    out.lon = lon;
    out.yaw_compass_deg = yaw_compass_deg;
    out.side_px = 0.25 * (s_top + s_right + s_bottom + s_left);
    out.center_px = cv::Point2f(
        0.25f * (TL.x + TR.x + BR.x + BL.x),
        0.25f * (TL.y + TR.y + BR.y + BL.y));
    return out;
}

}  // namespace pf
