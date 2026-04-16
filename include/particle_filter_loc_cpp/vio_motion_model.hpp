#pragma once

#include <cmath>
#include <optional>

#include "types.hpp"

namespace pf {

// Extract yaw (rotation about Z) in degrees from a quaternion.
// Uses the ROS convention: positive yaw = CCW around Z.
inline double quat_to_yaw_deg(double qx, double qy, double qz, double qw) {
    double siny_cosp = 2.0 * (qw * qz + qx * qy);
    double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    double yaw_rad = std::atan2(siny_cosp, cosy_cosp);
    double deg = yaw_rad * 180.0 / M_PI;
    if (deg < 0.0) deg += 360.0;
    return deg;
}

inline double wrap360(double deg) {
    deg = std::fmod(deg, 360.0);
    if (deg < 0.0) deg += 360.0;
    return deg;
}

// Consumes VIO poses (in the VIO world frame) and emits ENU-frame motion
// deltas for the particle filter. The VIO frame's translation/orientation
// is arbitrary; calling align() once at init pins its origin and rotation
// into ENU using a known ENU heading (from config or captured RTK yaw).
class VIOMotionModel {
public:
    void set_jump_params(double threshold_m, double ema_alpha) {
        jump_threshold_m_ = threshold_m;
        velocity_ema_alpha_ = ema_alpha;
    }

    // enu_yaw_compass_deg: drone heading in compass convention at init
    //                     (N=0, CW positive — matches RTK/DJI compass & PF).
    // vio_yaw_math_deg:   drone body yaw extracted from VIO quaternion via
    //                     quat_to_yaw_deg (math convention: +X=East, CCW).
    // Internally we convert RTK to math yaw (ψ_math = 90 − ψ_compass) so the
    // rotation that maps VIO-world position deltas into ENU Cartesian is
    //     α_math = ψ_E_math − ψ_V_math
    // Heading returned by update() is reported back to the PF in compass
    // convention (ψ_compass = 90 − ψ_math).
    void align(double vio_x, double vio_y, double vio_yaw_math_deg,
               double enu_yaw_compass_deg) {
        double enu_yaw_math = wrap360(90.0 - enu_yaw_compass_deg);
        align_theta_deg_ = wrap360(enu_yaw_math - vio_yaw_math_deg);
        double t = align_theta_deg_ * M_PI / 180.0;
        cos_theta_ = std::cos(t);
        sin_theta_ = std::sin(t);
        prev_vio_x_ = vio_x;
        prev_vio_y_ = vio_y;
        aligned_ = true;
        has_prev_ = false;  // first update() after align() seeds prev timestamp
    }

    bool is_aligned() const { return aligned_; }
    double align_theta_deg() const { return align_theta_deg_; }

    std::optional<MotionDelta> update(int64_t timestamp_ns,
                                      double vio_x, double vio_y,
                                      double vio_yaw_math_deg) {
        double vio_yaw_deg = vio_yaw_math_deg;
        if (!aligned_) return std::nullopt;

        if (!has_prev_) {
            prev_vio_x_ = vio_x;
            prev_vio_y_ = vio_y;
            prev_ts_ = timestamp_ns;
            has_prev_ = true;
            return std::nullopt;
        }

        double dx_vio = vio_x - prev_vio_x_;
        double dy_vio = vio_y - prev_vio_y_;
        double dt_s = (timestamp_ns - prev_ts_) * 1e-9;

        prev_vio_x_ = vio_x;
        prev_vio_y_ = vio_y;
        prev_ts_ = timestamp_ns;

        if (dt_s <= 0.0) return std::nullopt;

        // Rotate VIO-frame position delta into ENU Cartesian (E, N)
        double dx = cos_theta_ * dx_vio - sin_theta_ * dy_vio;
        double dy = sin_theta_ * dx_vio + cos_theta_ * dy_vio;
        // Body yaw in ENU math frame, then back to compass for the PF
        double yaw_math = wrap360(vio_yaw_deg + align_theta_deg_);
        double heading_deg = wrap360(90.0 - yaw_math);

        // Jump detection (mirrors RTKMotionModel)
        double displacement = std::sqrt(dx * dx + dy * dy);
        bool is_jump = false;
        if (velocity_ema_ > 0.0) {
            double expected = velocity_ema_ * dt_s;
            is_jump = displacement > jump_threshold_m_ &&
                      displacement > expected * 3.0;
        } else {
            is_jump = displacement > jump_threshold_m_;
        }
        if (!is_jump && dt_s > 0.0) {
            double speed = displacement / dt_s;
            velocity_ema_ = velocity_ema_alpha_ * speed +
                           (1.0 - velocity_ema_alpha_) * velocity_ema_;
        }

        return MotionDelta{dx, dy, heading_deg, dt_s, is_jump};
    }

private:
    bool aligned_ = false;
    bool has_prev_ = false;
    double align_theta_deg_ = 0.0;
    double cos_theta_ = 1.0;
    double sin_theta_ = 0.0;
    double prev_vio_x_ = 0.0;
    double prev_vio_y_ = 0.0;
    int64_t prev_ts_ = 0;

    double velocity_ema_ = 0.0;
    double jump_threshold_m_ = 20.0;
    double velocity_ema_alpha_ = 0.3;
};

}  // namespace pf
