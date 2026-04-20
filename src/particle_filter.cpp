#include "particle_filter_loc_cpp/particle_filter.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace pf {

namespace {
constexpr double NEG_INF = -1e30;

double clamp(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

// Fill Eigen vector with normal random values
void fill_normal(Eigen::VectorXd& vec, double mean, double sigma, std::mt19937_64& rng) {
    std::normal_distribution<double> dist(mean, sigma);
    for (int i = 0; i < vec.size(); ++i) vec(i) = dist(rng);
}

}  // namespace

ParticleFilter::ParticleFilter(const PFConfig& cfg, uint64_t rng_seed)
    : cfg_(cfg), rng_(rng_seed) {}

// ─── Initialization ─────────────────────────────────────────────

bool ParticleFilter::try_init(double altitude_m) {
    if (phase_ != Phase::UNINIT) return true;
    return altitude_m > cfg_.init_altitude_m;
}

void ParticleFilter::seed_from_position(
    double east, double north, double heading_deg,
    double sigma_pos, double sigma_hdg)
{
    int n = cfg_.n_tracking;
    particles_.resize(n, STATE_DIM);
    particles_.setZero();
    weights_.setConstant(n, 1.0 / n);

    std::normal_distribution<double> pos_d(0.0, sigma_pos);
    std::normal_distribution<double> hdg_d(heading_deg, sigma_hdg);
    std::normal_distribution<double> bias_d(0.0, cfg_.sigma_bias_init);

    for (int i = 0; i < n; ++i) {
        particles_(i, COL_X) = east + pos_d(rng_);
        particles_(i, COL_Y) = north + pos_d(rng_);
        particles_(i, COL_YAW) = std::fmod(hdg_d(rng_), 360.0);
        if (particles_(i, COL_YAW) < 0) particles_(i, COL_YAW) += 360.0;
        particles_(i, COL_BIAS_X) = bias_d(rng_);
        particles_(i, COL_BIAS_Y) = bias_d(rng_);
    }

    phase_ = Phase::TRACKING;
    frame_count_ = 0;
    initialized_ = true;
}

// ─── Prediction ─────────────────────────────────────────────────

void ParticleFilter::predict(const MotionDelta& delta) {
    if (!initialized_) return;
    int n = particles_.rows();

    double move_m = std::sqrt(delta.dx_m * delta.dx_m + delta.dy_m * delta.dy_m);
    is_static_ = move_m < cfg_.static_threshold_m;
    if (!is_static_) motion_detected_ = true;

    double sigma_pos = (phase_ == Phase::TRACKING)
        ? cfg_.sigma_pos_tracking : cfg_.sigma_pos_dispersed;
    double sigma_hdg = (phase_ == Phase::TRACKING)
        ? cfg_.sigma_hdg_tracking : cfg_.sigma_hdg_dispersed;
    double sigma_bias = cfg_.sigma_bias_random_walk;

    if (is_static_) {
        sigma_pos *= cfg_.static_sigma_scale;
        sigma_hdg *= cfg_.static_sigma_scale;
        sigma_bias *= cfg_.static_sigma_scale;
    }

    std::normal_distribution<double> pos_noise(0.0, sigma_pos);
    std::normal_distribution<double> hdg_noise(0.0, sigma_hdg);
    std::normal_distribution<double> bias_noise(0.0, sigma_bias);

    if (delta.is_jump) {
        // VIO jump: absorb into bias, don't move position
        for (int i = 0; i < n; ++i) {
            particles_(i, COL_BIAS_X) += delta.dx_m + bias_noise(rng_);
            particles_(i, COL_BIAS_Y) += delta.dy_m + bias_noise(rng_);
            particles_(i, COL_YAW) = delta.heading_deg + hdg_noise(rng_);
        }
    } else {
        // Normal VIO motion update
        for (int i = 0; i < n; ++i) {
            particles_(i, COL_X) += delta.dx_m + pos_noise(rng_);
            particles_(i, COL_Y) += delta.dy_m + pos_noise(rng_);
            particles_(i, COL_YAW) = delta.heading_deg + hdg_noise(rng_);
            // Bias random walk
            particles_(i, COL_BIAS_X) += bias_noise(rng_);
            particles_(i, COL_BIAS_Y) += bias_noise(rng_);
        }
    }

    // Clamp bias magnitude to prevent runaway
    if (cfg_.bias_clamp_m > 0.0) {
        for (int i = 0; i < n; ++i) {
            double bx = particles_(i, COL_BIAS_X);
            double by = particles_(i, COL_BIAS_Y);
            double bmag = std::sqrt(bx * bx + by * by);
            if (bmag > cfg_.bias_clamp_m) {
                double scale = cfg_.bias_clamp_m / bmag;
                particles_(i, COL_BIAS_X) *= scale;
                particles_(i, COL_BIAS_Y) *= scale;
            }
        }
    }

    // Legacy drift noise (will be removed in Phase 6)
    if (!is_static_ && !delta.is_jump && cfg_.drift_noise_m_per_s > 0.0 && delta.dt_s > 0.0) {
        double drift_sigma = cfg_.drift_noise_m_per_s * std::sqrt(delta.dt_s);
        std::normal_distribution<double> drift(0.0, drift_sigma);
        for (int i = 0; i < n; ++i) {
            particles_(i, COL_X) += drift(rng_);
            particles_(i, COL_Y) += drift(rng_);
        }
    }

    // Normalize heading to [0, 360)
    for (int i = 0; i < n; ++i) {
        particles_(i, COL_YAW) = std::fmod(particles_(i, COL_YAW), 360.0);
        if (particles_(i, COL_YAW) < 0) particles_(i, COL_YAW) += 360.0;
    }
}

// ─── Observation updates ────────────────────────────────────────

double ParticleFilter::get_obs_sigma_coarse(double altitude_m) const {
    if (!cfg_.altitude_sigma_enabled || altitude_m <= 0.0)
        return cfg_.sigma_obs_coarse;
    double scale = (altitude_m / cfg_.altitude_sigma_ref_m) * cfg_.altitude_sigma_scale;
    scale = clamp(scale, 0.5, 2.0);
    return cfg_.sigma_obs_coarse * scale;
}

void ParticleFilter::update_coarse(
    const std::vector<std::tuple<double, double, float>>& top_k_patches,
    double altitude_m, double temperature)
{
    if (!initialized_ || top_k_patches.empty()) return;
    int N = particles_.rows();

    // Filter: drop candidates with sim < 0.3
    std::vector<std::tuple<double, double, float>> filtered;
    for (auto& [e, n, s] : top_k_patches) {
        if (s >= 0.3f) filtered.emplace_back(e, n, s);
    }
    std::cout << "[PF] DBG update_coarse: input=" << top_k_patches.size()
              << " after sim>=0.3 filter=" << filtered.size();
    if (!top_k_patches.empty())
        std::cout << " (top sim=" << std::get<2>(top_k_patches[0]) << ")";
    std::cout << std::endl;
    if (filtered.empty()) return;

    // If top-1 clearly dominant, use only top-1
    if (filtered.size() >= 2) {
        float gap = std::get<2>(filtered[0]) - std::get<2>(filtered[1]);
        if (gap > 0.05f) filtered.resize(1);
    }

    int K = static_cast<int>(filtered.size());

    // Softmax mixture weights
    Eigen::VectorXd mix_w(K);
    double max_sim = -1e30;
    for (int k = 0; k < K; ++k)
        max_sim = std::max(max_sim, static_cast<double>(std::get<2>(filtered[k])) / temperature);
    for (int k = 0; k < K; ++k)
        mix_w(k) = std::exp(std::get<2>(filtered[k]) / temperature - max_sim);
    mix_w /= mix_w.sum();

    double sigma = get_obs_sigma_coarse(altitude_m);
    double sigma2 = 2.0 * sigma * sigma;
    double dist_cap2 = 9.0 * sigma * sigma;
    double nu = cfg_.likelihood_nu;
    bool use_student_t = nu < 100.0;

    // Compute log-likelihood for each particle via logsumexp over K
    Eigen::MatrixXd log_comp(N, K);
    for (int k = 0; k < K; ++k) {
        double east = std::get<0>(filtered[k]);
        double north = std::get<1>(filtered[k]);
        double log_mix = std::log(mix_w(k) + 1e-300);

        for (int i = 0; i < N; ++i) {
            double gx = particles_(i, COL_X) - particles_(i, COL_BIAS_X);
            double gy = particles_(i, COL_Y) - particles_(i, COL_BIAS_Y);
            double dx = gx - east;
            double dy = gy - north;
            double d2 = dx * dx + dy * dy;

            double lc;
            if (use_student_t)
                lc = log_mix - (nu + 2.0) / 2.0 * std::log(1.0 + d2 / (nu * sigma2));
            else
                lc = log_mix - d2 / sigma2;

            log_comp(i, k) = (d2 > dist_cap2) ? NEG_INF : lc;
        }
    }

    // logsumexp over K for each particle
    Eigen::VectorXd log_likelihood(N);
    for (int i = 0; i < N; ++i) {
        double lse_max = log_comp.row(i).maxCoeff();
        if (lse_max <= NEG_INF) {
            log_likelihood(i) = 0.0;  // neutral — no update
            continue;
        }
        double sum_exp = 0.0;
        for (int k = 0; k < K; ++k)
            sum_exp += std::exp(log_comp(i, k) - lse_max);
        log_likelihood(i) = lse_max + std::log(sum_exp + 1e-300);
    }

    // Update weights
    Eigen::VectorXd log_w = weights_.array().max(1e-300).log() + log_likelihood.array();
    log_w.array() -= log_w.maxCoeff();
    weights_ = log_w.array().exp();
    double total = weights_.sum();
    if (total > 0)
        weights_ /= total;
    else
        weights_.setConstant(N, 1.0 / N);
}

bool ParticleFilter::update_fine(
    double fine_east, double fine_north, int inliers,
    std::optional<double> heading_deg,
    std::optional<double> sigma_override,
    std::optional<double> kappa_override,
    const std::string& method,
    double altitude_m,
    float flow_mag_cv)
{
    if (!initialized_) return false;
    if (std::isnan(fine_east) || std::isnan(fine_north)) return false;

    // Adaptive inlier gating: scale threshold with altitude
    int effective_threshold = cfg_.obs_min_inliers_apply;
    if (altitude_m > 0.0 && cfg_.obs_min_inliers_ref_alt > 0.0) {
        double scale = altitude_m / cfg_.obs_min_inliers_ref_alt;
        effective_threshold = static_cast<int>(cfg_.obs_min_inliers_apply * scale);
        effective_threshold = std::min(effective_threshold, cfg_.obs_min_inliers_max);
        effective_threshold = std::max(effective_threshold, cfg_.obs_min_inliers_apply);
    }
    if (inliers < effective_threshold && !sigma_override.has_value())
        return false;

    int N = particles_.rows();

    // Inlier trust score: smooth exponential curve
    double trust = 1.0 - std::exp(-static_cast<double>(inliers) / cfg_.inlier_tau);

    // ── Sliding-window match voting (replaces gate + consensus) ──
    // Record this match into the recent window, then cluster: only apply the
    // observation if ≥K recent matches (including this one) agree within
    // `agreement_m` of each other. The applied position is the cluster centroid.
    // PF no longer gates matches against its own estimate — it smooths over
    // observation clusters. Singleton matches update nothing.
    double vote_fine_east = fine_east;
    double vote_fine_north = fine_north;
    if (!sigma_override.has_value() && cfg_.vote_enabled) {
        recent_matches_.push_back({fine_east, fine_north, inliers, 0});
        while ((int)recent_matches_.size() > cfg_.vote_window_size)
            recent_matches_.pop_front();

        // Warm-up: if the window is too small to possibly form a quorum,
        // accept the match directly. Prevents startup lockout where the
        // first K-1 matches are silently dropped and PF never initializes
        // off VIO predict.
        bool warm_up = (int)recent_matches_.size() < cfg_.vote_min_agree;

        // Find the largest cluster within agreement_m of the current match
        int agree = 1;
        double sum_e = fine_east, sum_n = fine_north;
        double sum_w = static_cast<double>(inliers);
        double sum_we = fine_east * sum_w, sum_wn = fine_north * sum_w;
        for (auto it = recent_matches_.begin(); it != recent_matches_.end(); ++it) {
            // skip the last entry (current match, already counted)
            if (it + 1 == recent_matches_.end()) continue;
            double de = it->e - fine_east;
            double dn = it->n - fine_north;
            if (std::sqrt(de * de + dn * dn) <= cfg_.vote_agreement_m) {
                agree++;
                sum_e += it->e;
                sum_n += it->n;
                double w = static_cast<double>(it->inliers);
                sum_w += w;
                sum_we += it->e * w;
                sum_wn += it->n * w;
            }
        }
        if (!warm_up && agree < cfg_.vote_min_agree) {
            // Not enough agreement — treat as noise, don't update PF
            return false;
        }
        if (!warm_up) {
            // Use inlier-weighted centroid of the agreeing cluster as the observation
            if (sum_w > 0.0) {
                vote_fine_east = sum_we / sum_w;
                vote_fine_north = sum_wn / sum_w;
            } else {
                vote_fine_east = sum_e / agree;
                vote_fine_north = sum_n / agree;
            }
        }
    }
    fine_east = vote_fine_east;
    fine_north = vote_fine_north;

    // Determine sigma: smooth scaling from trust score
    // trust=0 → sigma = base/0.2 = 5x base (very loose)
    // trust=0.5 → sigma = base/0.6 = 1.67x base
    // trust=1.0 → sigma = base/1.0 = base (tight)
    double flow_penalty = 1.0;
    if (!sigma_override.has_value() && cfg_.flow_mag_cv_ref > 0.0) {
        double cv = std::max(static_cast<double>(flow_mag_cv), 0.02);
        flow_penalty = cfg_.flow_mag_cv_ref / cv;
        flow_penalty = std::clamp(flow_penalty, 1.0, cfg_.flow_mag_cv_max_penalty);
    }

    // Sanity brake: inflate sigma when the voted centroid is far from the PF's
    // current bias-corrected weighted mean. Scene aliasing can produce tight,
    // consistent wrong clusters; this softly down-weights them without a hard
    // reject. Quadratic beyond the gate, capped.
    double corr_penalty = 1.0;
    if (!sigma_override.has_value() && cfg_.corr_dist_sigma_gate_m > 0.0) {
        double sum_w = weights_.sum();
        if (sum_w > 0.0) {
            Eigen::VectorXd w_norm = weights_ / sum_w;
            double mean_e = ((particles_.col(COL_X) - particles_.col(COL_BIAS_X)).array() * w_norm.array()).sum();
            double mean_n = ((particles_.col(COL_Y) - particles_.col(COL_BIAS_Y)).array() * w_norm.array()).sum();
            double corr_d = std::hypot(fine_east - mean_e, fine_north - mean_n);
            if (corr_d > cfg_.corr_dist_sigma_gate_m) {
                double r = corr_d / cfg_.corr_dist_sigma_gate_m;
                corr_penalty = std::clamp(r * r, 1.0, cfg_.corr_dist_sigma_max);
            }
        }
    }

    double sigma;
    if (sigma_override.has_value()) {
        sigma = sigma_override.value();
    } else {
        double scale = 1.0 / (0.2 + 0.8 * trust);
        sigma = cfg_.sigma_obs_fine * scale * flow_penalty * corr_penalty;
    }

    double sigma2 = 2.0 * sigma * sigma;
    double nu = cfg_.likelihood_nu;

    Eigen::VectorXd log_lik(N);
    for (int i = 0; i < N; ++i) {
        // Bias-corrected global position
        double gx = particles_(i, COL_X) - particles_(i, COL_BIAS_X);
        double gy = particles_(i, COL_Y) - particles_(i, COL_BIAS_Y);
        double dx = gx - fine_east;
        double dy = gy - fine_north;
        double d2 = dx * dx + dy * dy;
        if (nu < 100.0)
            log_lik(i) = -(nu + 2.0) / 2.0 * std::log(1.0 + d2 / (nu * sigma2));
        else
            log_lik(i) = -d2 / sigma2;
    }

    // Von Mises heading update
    if (heading_deg.has_value() && inliers >= cfg_.fine_min_inliers_heading) {
        double kappa;
        if (kappa_override.has_value()) {
            kappa = kappa_override.value();
        } else {
            // Smooth kappa from trust: trust=0.45 → κ≈7, trust=0.63 → κ≈9.5, trust=0.95 → κ≈14
            kappa = 15.0 * trust;
        }
        if (kappa > 0) {
            double hdg = heading_deg.value();
            for (int i = 0; i < N; ++i) {
                double diff_rad = (particles_(i, COL_YAW) - hdg) * M_PI / 180.0;
                log_lik(i) += kappa * std::cos(diff_rad);
            }
        }
    }

    // Update weights
    Eigen::VectorXd log_w = weights_.array().max(1e-300).log() + log_lik.array();
    log_w.array() -= log_w.maxCoeff();
    weights_ = log_w.array().exp();
    double total = weights_.sum();
    if (total > 0)
        weights_ /= total;
    else
        weights_.setConstant(N, 1.0 / N);
    frames_since_fine_ = 0;
    return true;
}

void ParticleFilter::apply_global_correction(
    double east, double north, std::optional<double> heading_deg,
    double teleport_fraction, double teleport_sigma)
{
    if (!initialized_) return;

    // LOST recovery: re-seed all particles at verified position
    double hdg = heading_deg.value_or(0.0);
    double sigma_pos = (teleport_sigma > 0.0) ? teleport_sigma : cfg_.lost_recovery_sigma_pos;
    double sigma_hdg = cfg_.lost_recovery_sigma_hdg;

    seed_from_position(east, north, hdg, sigma_pos, sigma_hdg);
    phase_ = Phase::TRACKING;
}

// ─── Resampling ─────────────────────────────────────────────────

void ParticleFilter::resample_if_needed() {
    if (!initialized_) return;
    double ess = effective_sample_size();
    double threshold = cfg_.ess_threshold_fraction * particles_.rows();
    if (ess < threshold) systematic_resample();
}

void ParticleFilter::systematic_resample() {
    int n = particles_.rows();
    Eigen::VectorXd cumsum(n);
    cumsum(0) = weights_(0);
    for (int i = 1; i < n; ++i)
        cumsum(i) = cumsum(i - 1) + weights_(i);
    cumsum(n - 1) = 1.0;

    std::uniform_real_distribution<double> u01(0.0, 1.0);
    double u0 = u01(rng_);

    Eigen::MatrixXd new_particles(n, STATE_DIM);
    int j = 0;
    for (int i = 0; i < n; ++i) {
        double target = (u0 + i) / n;
        while (j < n - 1 && cumsum(j) < target) ++j;
        new_particles.row(i) = particles_.row(j);
    }
    particles_ = new_particles;
    weights_.setConstant(n, 1.0 / n);

    // Roughening
    if (cfg_.roughen_enabled) {
        double h = cfg_.roughen_scale;
        double spread = weighted_spread();
        double n_inv_third = std::pow(static_cast<double>(n), -1.0 / 3.0);
        double roughen_pos = std::max(0.3, h * spread * n_inv_third);
        double roughen_hdg = std::max(0.5, h * 10.0 * n_inv_third);

        std::normal_distribution<double> pos_d(0.0, roughen_pos);
        std::normal_distribution<double> hdg_d(0.0, roughen_hdg);
        std::normal_distribution<double> bias_d(0.0, 0.01);
        for (int i = 0; i < n; ++i) {
            particles_(i, COL_X) += pos_d(rng_);
            particles_(i, COL_Y) += pos_d(rng_);
            particles_(i, COL_YAW) += hdg_d(rng_);
            particles_(i, COL_YAW) = std::fmod(particles_(i, COL_YAW), 360.0);
            if (particles_(i, COL_YAW) < 0) particles_(i, COL_YAW) += 360.0;
            particles_(i, COL_BIAS_X) += bias_d(rng_);
            particles_(i, COL_BIAS_Y) += bias_d(rng_);
        }
    }
}

// ─── State transitions ──────────────────────────────────────────

void ParticleFilter::check_transitions() {
    if (!initialized_) return;
    double spread = weighted_spread();
    frames_since_fine_++;

    // Age recent match window — drop entries older than the voting window.
    // Measured in camera frames (check_transitions is called once per camera frame).
    for (auto& m : recent_matches_) m.age++;
    while (!recent_matches_.empty() &&
           recent_matches_.front().age > cfg_.vote_max_age_frames) {
        recent_matches_.pop_front();
    }

    switch (phase_) {
    case Phase::TRACKING:
        // LOST triggers: (a) spread blew up, or (b) no fine match for too long
        // (tight cloud at wrong location — classic "confidently wrong")
        if (spread > cfg_.lost_spread_m ||
            frames_since_fine_ > cfg_.lost_fine_stale_frames) {
            phase_ = Phase::LOST;
            frame_count_ = 0;
        }
        break;
    case Phase::LOST:
        if (spread < cfg_.tracking_spread_m) {
            phase_ = Phase::TRACKING;
            frame_count_ = 0;
        }
        break;
    default:
        break;
    }
}

void ParticleFilter::reduce_particles(int target_n) {
    if (particles_.rows() <= target_n) return;
    systematic_resample();
    int n = particles_.rows();
    if (n <= target_n) return;

    // Random subset
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng_);
    idx.resize(target_n);

    Eigen::MatrixXd new_p(target_n, STATE_DIM);
    for (int i = 0; i < target_n; ++i)
        new_p.row(i) = particles_.row(idx[i]);
    particles_ = new_p;
    weights_.setConstant(target_n, 1.0 / target_n);
}

void ParticleFilter::expand_particles(int target_n) {
    int n = particles_.rows();
    if (n >= target_n) return;
    int n_add = target_n - n;

    // Weighted sampling with replacement
    std::discrete_distribution<int> dist(weights_.data(), weights_.data() + n);
    std::normal_distribution<double> pos_d(0.0, cfg_.sigma_pos_dispersed);
    std::normal_distribution<double> hdg_d(0.0, cfg_.sigma_hdg_dispersed);

    Eigen::MatrixXd new_p(target_n, STATE_DIM);
    new_p.setZero();
    new_p.topRows(n) = particles_;
    for (int i = 0; i < n_add; ++i) {
        int src = dist(rng_);
        new_p(n + i, COL_X) = particles_(src, COL_X) + pos_d(rng_);
        new_p(n + i, COL_Y) = particles_(src, COL_Y) + pos_d(rng_);
        new_p(n + i, COL_YAW) = std::fmod(particles_(src, COL_YAW) + hdg_d(rng_), 360.0);
        if (new_p(n + i, COL_YAW) < 0) new_p(n + i, COL_YAW) += 360.0;
        new_p(n + i, COL_BIAS_X) = particles_(src, COL_BIAS_X);
        new_p(n + i, COL_BIAS_Y) = particles_(src, COL_BIAS_Y);
    }
    particles_ = new_p;
    weights_.setConstant(target_n, 1.0 / target_n);
}

// ─── Output ─────────────────────────────────────────────────────

std::tuple<double, double, double> ParticleFilter::estimate() const {
    if (!initialized_) return {0.0, 0.0, 0.0};
    int n = particles_.rows();

    // Bias-corrected global position
    double east = 0.0, north = 0.0;
    for (int i = 0; i < n; ++i) {
        east += weights_(i) * (particles_(i, COL_X) - particles_(i, COL_BIAS_X));
        north += weights_(i) * (particles_(i, COL_Y) - particles_(i, COL_BIAS_Y));
    }

    // Circular mean for heading
    double sin_sum = 0.0, cos_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double rad = particles_(i, COL_YAW) * M_PI / 180.0;
        sin_sum += weights_(i) * std::sin(rad);
        cos_sum += weights_(i) * std::cos(rad);
    }
    double heading = std::fmod(std::atan2(sin_sum, cos_sum) * 180.0 / M_PI, 360.0);
    if (heading < 0) heading += 360.0;

    return {east, north, heading};
}

PFEstimate ParticleFilter::estimate_full() const {
    if (!initialized_) return {};
    auto [e, n, h] = estimate();
    int np = particles_.rows();
    double bx = weights_.dot(particles_.col(COL_BIAS_X));
    double by = weights_.dot(particles_.col(COL_BIAS_Y));
    return {e, n, h, bx, by};
}

double ParticleFilter::effective_sample_size() const {
    if (!initialized_) return 0.0;
    return 1.0 / (weights_.squaredNorm() + 1e-300);
}

double ParticleFilter::weighted_spread() const {
    if (!initialized_) return std::numeric_limits<double>::infinity();
    int n = particles_.rows();

    // Bias-corrected positions
    double mean_e = 0.0, mean_n = 0.0;
    for (int i = 0; i < n; ++i) {
        mean_e += weights_(i) * (particles_(i, COL_X) - particles_(i, COL_BIAS_X));
        mean_n += weights_(i) * (particles_(i, COL_Y) - particles_(i, COL_BIAS_Y));
    }

    double var_e = 0.0, var_n = 0.0;
    for (int i = 0; i < n; ++i) {
        double ge = particles_(i, COL_X) - particles_(i, COL_BIAS_X);
        double gn = particles_(i, COL_Y) - particles_(i, COL_BIAS_Y);
        double de = ge - mean_e;
        double dn = gn - mean_n;
        var_e += weights_(i) * de * de;
        var_n += weights_(i) * dn * dn;
    }
    return std::sqrt(var_e + var_n);
}

bool ParticleFilter::should_run_fine() {
    frame_count_++;
    if (phase_ == Phase::UNINIT) return false;
    if (phase_ == Phase::LOST) return true;  // Always run fine in LOST for recovery
    return (frame_count_ % cfg_.fine_every_n_frames) == 0;
}

int ParticleFilter::get_fine_top_k() const {
    double spread = weighted_spread();
    if (spread > 50.0) return 3;
    if (spread > 20.0) return 2;
    return 1;
}

double ParticleFilter::get_context_fraction(double base) const {
    double spread = weighted_spread();
    double frac = base + spread / 200.0;
    return clamp(frac, cfg_.base_context_fraction, cfg_.max_context_fraction);
}

double ParticleFilter::get_search_radius() const {
    double spread = weighted_spread();
    return std::max(spread * cfg_.search_radius_multiplier, cfg_.min_search_radius_m);
}

double ParticleFilter::get_satellite_context_scale(double min_scale, double max_scale) const {
    double spread = weighted_spread();
    double lo = cfg_.sat_ctx_spread_low_m;
    double hi = cfg_.sat_ctx_spread_high_m;
    double t = (hi > lo) ? (spread - lo) / (hi - lo) : 0.0;
    t = std::clamp(t, 0.0, 1.0);
    return min_scale + t * (max_scale - min_scale);
}

}  // namespace pf
