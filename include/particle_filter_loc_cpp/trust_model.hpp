#pragma once

#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "config.hpp"

namespace pf {

struct CandidateScore {
    double east_m, north_m;
    int inliers;
    std::optional<double> heading_deg;
    std::string source;
    double inlier_score = 0.0;
    double sim_score = 0.0;
    double consistency_score = 0.0;
    double altitude_score = 0.0;
    double geometry_score = 0.0;
    double confidence = 0.0;
};

struct FrameTrust {
    std::optional<CandidateScore> best;
    std::vector<CandidateScore> all_scores;
    bool drift_detected = false;
    double recon_sim_ema = 0.0;
    double pf_self_confidence_ema = 0.0;
};

struct GlobalCorrectionResult {
    double east_m, north_m;
    std::optional<double> heading_deg;
    int inliers_median;
    int n_consistent_frames;
    double distance_from_pf_m;
};

// Candidate input for evaluate_frame
struct CandidateInput {
    double east_m, north_m;
    int inliers;
    std::optional<double> heading_deg;
    std::string source;
    float flow_consistency = 0.0f;
    float flow_magnitude_cv = 1.0f;
    float inlier_ratio = 0.0f;
};

CandidateScore score_candidate(
    double east_m, double north_m, int inliers,
    std::optional<double> heading_deg, const std::string& source,
    double sim, double pf_east, double pf_north,
    double pf_spread, double lost_spread, double altitude_m,
    const TrustConfig& cfg,
    float flow_consistency = 0.0f,
    float flow_magnitude_cv = 1.0f,
    float inlier_ratio = 0.0f);

std::optional<std::pair<double, double>> evaluate_coarse_trust(
    double top1_sim, double pf_spread, double recon_sim_ema,
    const TrustConfig& cfg);

class TrustTracker {
public:
    explicit TrustTracker(const TrustConfig& cfg);

    FrameTrust evaluate_frame(
        const std::vector<CandidateInput>& fine_candidates,
        double recon_sim, double top1_sim,
        double pf_east, double pf_north,
        double pf_spread, double lost_spread,
        double altitude_m, double timestamp_s);

    double get_sigma_scale(double confidence) const;
    double get_effective_sigma(double confidence, double base_sigma, bool is_static = false) const;
    double get_effective_kappa(double confidence, bool is_static = false) const;

    std::optional<GlobalCorrectionResult> evaluate_global_correction(
        double recon_sim,
        std::optional<double> coarse_fine_east,
        std::optional<double> coarse_fine_north,
        int coarse_fine_inliers,
        std::optional<double> coarse_fine_heading,
        double pf_east, double pf_north, double pf_spread);

    double recon_sim_ema() const { return recon_sim_ema_; }
    double pf_self_confidence_ema() const { return pf_self_confidence_ema_; }

private:
    double ema(double old_val, double new_val) const;

    TrustConfig cfg_;
    double recon_sim_ema_ = 0.0;
    double pf_self_confidence_ema_ = 0.5;
    bool initialized_ = false;
    int recon_diverge_count_ = 0;
    std::vector<std::tuple<double, double, int, std::optional<double>>> coarse_fine_hits_;
    int global_correction_cooldown_ = 0;
};

}  // namespace pf
