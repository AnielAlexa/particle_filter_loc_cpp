#pragma once

#include <deque>
#include <optional>
#include <random>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

#include "config.hpp"
#include "types.hpp"

namespace pf {

// Particle state column indices
static constexpr int COL_X = 0;
static constexpr int COL_Y = 1;
static constexpr int COL_YAW = 2;
static constexpr int COL_BIAS_X = 3;
static constexpr int COL_BIAS_Y = 4;
static constexpr int STATE_DIM = 5;

class ParticleFilter {
public:
    explicit ParticleFilter(const PFConfig& cfg, uint64_t rng_seed = 42);

    // Initialization
    bool try_init(double altitude_m);
    void seed_from_position(double east, double north, double heading_deg = 0.0,
                            double sigma_pos = 0.2, double sigma_hdg = 2.0);

    // Prediction
    void predict(const MotionDelta& delta);

    // Observation updates
    double get_obs_sigma_coarse(double altitude_m = 0.0) const;
    void update_coarse(const std::vector<std::tuple<double, double, float>>& top_k_patches,
                       double altitude_m = 0.0, double temperature = 0.05);
    bool update_fine(double fine_east, double fine_north, int inliers,
                     std::optional<double> heading_deg = std::nullopt,
                     std::optional<double> sigma_override = std::nullopt,
                     std::optional<double> kappa_override = std::nullopt,
                     const std::string& method = "",
                     double altitude_m = 0.0,
                     float flow_mag_cv = 1.0f);

    // Global correction (LOST recovery)
    void apply_global_correction(double east, double north,
                                 std::optional<double> heading_deg = std::nullopt,
                                 double teleport_fraction = 0.35,
                                 double teleport_sigma = 10.0);

    // Resampling
    void resample_if_needed();
    void check_transitions();

    // Output
    std::tuple<double, double, double> estimate() const;  // bias-corrected (east, north, heading)
    PFEstimate estimate_full() const;                      // full estimate with bias
    double effective_sample_size() const;
    double weighted_spread() const;
    bool should_run_fine();
    int get_fine_top_k() const;
    double get_context_fraction(double base = 0.25) const;
    double get_search_radius() const;

    // State
    Phase phase() const { return phase_; }
    bool is_static() const { return is_static_; }
    int num_particles() const { return particles_.rows(); }
    const Eigen::MatrixXd& particles() const { return particles_; }
    const Eigen::VectorXd& weights() const { return weights_; }

private:
    void systematic_resample();
    void reduce_particles(int target_n);
    void expand_particles(int target_n);

    PFConfig cfg_;
    Phase phase_ = Phase::UNINIT;
    Eigen::MatrixXd particles_;  // [N, 5]: x, y, yaw, bias_x, bias_y
    Eigen::VectorXd weights_;    // [N]
    std::mt19937_64 rng_;
    int frame_count_ = 0;
    int frames_since_fine_ = 0;  // for staleness-based LOST detection
    bool is_static_ = false;
    bool motion_detected_ = false;
    bool initialized_ = false;

    // Sliding-window match voting: keep recent fine matches (regardless of
    // whether they agreed with PF) so we can apply the cluster centroid
    // rather than trust a single observation.
    struct RecentMatch {
        double e, n;
        int inliers;
        int age;  // camera-frame count since push
    };
    std::deque<RecentMatch> recent_matches_;
};

}  // namespace pf
