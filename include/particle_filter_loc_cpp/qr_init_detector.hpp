#pragma once

#include <optional>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

namespace pf {

struct QrInitResult {
    double lat;
    double lon;
    double yaw_compass_deg;   // 0=N, 90=E — drone compass heading derived from QR orientation
    double side_px;           // mean QR side length in pixels (for diagnostics)
    cv::Point2f center_px;    // QR center in image (for diagnostics)
};

class QrInitDetector {
public:
    QrInitDetector(double heading_offset_deg,
                   double min_side_px,
                   double max_aspect_skew);

    // Runs cv::QRCodeDetector on a mono8 image and decodes a `geo:LAT,LON` payload.
    // Returns nullopt if no valid QR is found, payload is malformed, or sanity checks fail.
    std::optional<QrInitResult> detect(const cv::Mat& mono8);

private:
    double heading_offset_deg_;
    double min_side_px_;
    double max_aspect_skew_;
    cv::QRCodeDetector detector_;
};

}  // namespace pf
