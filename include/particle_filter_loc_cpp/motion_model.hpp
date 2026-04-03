#pragma once

#include <cmath>
#include <optional>

#include "types.hpp"

namespace pf {

constexpr double METERS_PER_DEGREE_LAT = 111320.0;

inline std::pair<double, double> get_relative_movement(
    double lat_prev, double lon_prev, double lat_curr, double lon_curr)
{
    double dlat = lat_curr - lat_prev;
    double dlon = lon_curr - lon_prev;
    double lat_rad = lat_prev * M_PI / 180.0;
    double dy = dlat * METERS_PER_DEGREE_LAT;
    double dx = dlon * (METERS_PER_DEGREE_LAT * std::cos(lat_rad));
    return {dx, dy};
}

class RTKMotionModel {
public:
    void set_yaw(double yaw_raw) {
        yaw_deg_ = yaw_raw * 10.0;
    }

    std::optional<MotionDelta> update(int64_t timestamp_ns, double lat, double lon) {
        if (!has_prev_) {
            prev_lat_ = lat;
            prev_lon_ = lon;
            prev_ts_ = timestamp_ns;
            has_prev_ = true;
            return std::nullopt;
        }

        auto [dx, dy] = get_relative_movement(prev_lat_, prev_lon_, lat, lon);
        double dt_s = (timestamp_ns - prev_ts_) * 1e-9;

        prev_lat_ = lat;
        prev_lon_ = lon;
        prev_ts_ = timestamp_ns;

        if (dt_s <= 0.0)
            return std::nullopt;

        return MotionDelta{dx, dy, yaw_deg_, dt_s};
    }

private:
    bool has_prev_ = false;
    double prev_lat_ = 0.0;
    double prev_lon_ = 0.0;
    int64_t prev_ts_ = 0;
    double yaw_deg_ = 0.0;
};

}  // namespace pf
