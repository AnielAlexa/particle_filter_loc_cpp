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
    void set_jump_params(double threshold_m, double ema_alpha) {
        jump_threshold_m_ = threshold_m;
        velocity_ema_alpha_ = ema_alpha;
    }

    void set_yaw(double yaw_raw) {
        yaw_deg_ = yaw_raw * 10.0;
    }

    double current_yaw() const { return yaw_deg_; }

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

        // Jump detection
        double displacement = std::sqrt(dx * dx + dy * dy);
        bool is_jump = false;
        if (velocity_ema_ > 0.0) {
            // Jump if displacement exceeds threshold AND is much larger than recent velocity
            double expected = velocity_ema_ * dt_s;
            is_jump = displacement > jump_threshold_m_ &&
                      displacement > expected * 3.0;
        } else {
            is_jump = displacement > jump_threshold_m_;
        }

        // Update velocity EMA (only on non-jumps to keep it stable)
        if (!is_jump && dt_s > 0.0) {
            double speed = displacement / dt_s;
            velocity_ema_ = velocity_ema_alpha_ * speed +
                           (1.0 - velocity_ema_alpha_) * velocity_ema_;
        }

        return MotionDelta{dx, dy, yaw_deg_, dt_s, is_jump};
    }

private:
    bool has_prev_ = false;
    double prev_lat_ = 0.0;
    double prev_lon_ = 0.0;
    int64_t prev_ts_ = 0;
    double yaw_deg_ = 0.0;

    // Jump detection
    double velocity_ema_ = 0.0;
    double jump_threshold_m_ = 5.0;
    double velocity_ema_alpha_ = 0.3;
};

}  // namespace pf
