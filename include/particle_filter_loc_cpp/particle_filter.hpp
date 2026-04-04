#pragma once

#include <optional>
#include <random>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

#include "config.hpp"
#include "types.hpp"

namespace pf {

class ParticleFilter {
public:
    explicit ParticleFilter(const PFConfig& cfg, uint64_t rng_seed = 42);

    // Initialization
    bool try_init(double altitude_m);
    void seed_from_coarse(const std::vector<std::pair<double, double>>& centers_enu,
                          const std::vector<float>& similarities,
                          std::optional<double> sigma_override = std::nullopt);
    void seed_from_position(double east, double north, double heading_deg = 0.0,
                            double sigma_pos = 5.0, double sigma_hdg = 10.0);

    // Prediction
    void predict(const MotionDelta& delta);

    // Observation updates
    double get_obs_sigma_coarse(double altitude_m = 0.0) const;
    void update_coarse(const std::vector<std::tuple<double, double, float>>& top_k_patches,
                       double altitude_m = 0.0, double temperature = 0.05);
    void inject_coarse_trust(double east, double north, float sim,
                             std::optional<double> frac_override = std::nullopt,
                             std::optional<double> sigma_override = std::nullopt);
    bool update_fine(double fine_east, double fine_north, int inliers,
                     std::optional<double> heading_deg = std::nullopt,
                     std::optional<double> sigma_override = std::nullopt,
                     std::optional<double> kappa_override = std::nullopt);

    // Adaptive parameter interpolation
    void feed_correction_distance(double correction_dist_m);
    double adaptive_sigma_pos() const;
    double adaptive_sigma_obs_fine() const;
    double adaptive_roughen_scale() const;

    // Global correction
    void apply_global_correction(double east, double north,
                                 std::optional<double> heading_deg = std::nullopt,
                                 double teleport_fraction = 0.35,
                                 double teleport_sigma = 10.0);

    // Resampling
    void resample_if_needed();
    void check_transitions();

    // Output
    std::tuple<double, double, double> estimate() const;
    double effective_sample_size() const;
    double weighted_spread() const;
    bool should_run_fine();
    int get_fine_top_k() const;
    double get_context_fraction(double base = 0.25) const;
    double get_search_radius() const;

    // State
    Phase phase() const { return phase_; }
    bool is_static() const { return is_static_; }
    double drift_factor() const { return drift_factor_; }
    double correction_ema() const { return correction_ema_; }
    int num_particles() const { return particles_.rows(); }
    const Eigen::MatrixXd& particles() const { return particles_; }
    const Eigen::VectorXd& weights() const { return weights_; }

private:
    void systematic_resample();
    void reduce_particles(int target_n);
    void expand_particles(int target_n);

    PFConfig cfg_;
    Phase phase_ = Phase::UNINIT;
    Eigen::MatrixXd particles_;  // [N, 3]: east, north, heading
    Eigen::VectorXd weights_;    // [N]
    std::mt19937_64 rng_;
    int frame_count_ = 0;
    bool is_static_ = false;
    bool motion_detected_ = false;
    double cov_sigma_ema_ = 100.0;
    double correction_ema_ = 0.0;
    double drift_factor_ = 0.0;
    bool initialized_ = false;
};

}  // namespace pf
