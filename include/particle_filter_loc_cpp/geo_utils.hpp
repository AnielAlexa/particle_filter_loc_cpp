#pragma once

#include <cmath>
#include <utility>

namespace pf {

constexpr double METERS_PER_DEG = 111319.5;

class ENUFrame {
public:
    ENUFrame() = default;
    ENUFrame(double origin_lat, double origin_lon)
        : origin_lat_(origin_lat), origin_lon_(origin_lon),
          cos_lat_(std::cos(origin_lat * M_PI / 180.0)) {}

    std::pair<double, double> wgs84_to_enu(double lat, double lon) const {
        double east  = (lon - origin_lon_) * METERS_PER_DEG * cos_lat_;
        double north = (lat - origin_lat_) * METERS_PER_DEG;
        return {east, north};
    }

    std::pair<double, double> enu_to_wgs84(double east, double north) const {
        double lat = origin_lat_ + north / METERS_PER_DEG;
        double lon = origin_lon_ + east / (METERS_PER_DEG * cos_lat_);
        return {lat, lon};
    }

    double origin_lat() const { return origin_lat_; }
    double origin_lon() const { return origin_lon_; }

private:
    double origin_lat_ = 0.0;
    double origin_lon_ = 0.0;
    double cos_lat_ = 1.0;
};

inline double haversine_m(double lat1, double lon1, double lat2, double lon2) {
    constexpr double R = 6371000.0;
    double phi1 = lat1 * M_PI / 180.0;
    double phi2 = lat2 * M_PI / 180.0;
    double dphi = (lat2 - lat1) * M_PI / 180.0;
    double dlam = (lon2 - lon1) * M_PI / 180.0;
    double a = std::sin(dphi / 2) * std::sin(dphi / 2) +
               std::cos(phi1) * std::cos(phi2) *
               std::sin(dlam / 2) * std::sin(dlam / 2);
    return 2.0 * R * std::asin(std::sqrt(a));
}

inline double normalize_angle(double deg) {
    deg = std::fmod(deg, 360.0);
    if (deg < 0.0) deg += 360.0;
    if (deg >= 180.0) deg -= 360.0;
    return deg;
}

}  // namespace pf
