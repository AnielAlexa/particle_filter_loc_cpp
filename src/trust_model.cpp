#include "particle_filter_loc_cpp/trust_model.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace pf {

namespace {
double clamp(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}
}  // namespace

CandidateScore score_candidate(
    double east_m, double north_m, int inliers,
    std::optional<double> heading_deg, const std::string& source,
    double sim, double pf_east, double pf_north,
    double pf_spread, double lost_spread, double altitude_m,
    const TrustConfig& cfg,
    float flow_consistency, float flow_magnitude_cv, float inlier_ratio)
{
    CandidateScore cs;
    cs.east_m = east_m;
    cs.north_m = north_m;
    cs.inliers = inliers;
    cs.heading_deg = heading_deg;
    cs.source = source;

    // Signal 1: Inlier score
    cs.inlier_score = 1.0 - std::exp(-inliers / cfg.inlier_tau);

    // Signal 2: Similarity score
    if (cfg.sim_ceiling > cfg.sim_floor)
        cs.sim_score = clamp((sim - cfg.sim_floor) / (cfg.sim_ceiling - cfg.sim_floor), 0.0, 1.0);
    else
        cs.sim_score = (sim >= cfg.sim_floor) ? 1.0 : 0.0;

    // Signal 3: PF consistency
    if (pf_spread > lost_spread) {
        cs.consistency_score = 1.0;
    } else {
        double eff_r = std::max(pf_spread * 2.0, cfg.consistency_min_radius_m);
        double dx = east_m - pf_east;
        double dy = north_m - pf_north;
        double dist = std::sqrt(dx * dx + dy * dy);
        cs.consistency_score = std::exp(-0.5 * (dist / eff_r) * (dist / eff_r));
    }

    // Signal 5: Altitude
    cs.altitude_score = clamp(cfg.altitude_ref_m / std::max(altitude_m, 1.0), 0.3, 1.0);

    // Signal 7: Flow geometry
    if (flow_consistency > 0.0f || inlier_ratio > 0.0f) {
        double mag_score = 1.0 / (1.0 + flow_magnitude_cv);
        if (source == "satellite" || source == "mosaic")
            cs.geometry_score = flow_consistency * mag_score * (0.5 + 0.5 * inlier_ratio);
        else
            cs.geometry_score = mag_score * (0.3 + 0.7 * inlier_ratio);
    } else {
        cs.geometry_score = 0.3;
    }

    // Combined: weighted geometric mean
    struct Signal { double score; double weight; };
    Signal signals[] = {
        {cs.inlier_score,     cfg.inlier_weight},
        {cs.sim_score,        cfg.sim_weight},
        {cs.consistency_score, cfg.consistency_weight},
        {cs.altitude_score,   cfg.altitude_weight},
        {cs.geometry_score,   cfg.geometry_weight},
    };
    double log_conf = 0.0;
    for (auto& s : signals)
        log_conf += s.weight * std::log(std::max(s.score, 1e-10));
    cs.confidence = std::exp(log_conf);

    // High-confidence joint gate
    if ((source == "satellite" || source == "mosaic") &&
        sim >= cfg.recon_high_conf_sim_thr &&
        inliers >= cfg.recon_high_conf_inlier_thr) {
        cs.confidence = std::min(1.0, cs.confidence * cfg.recon_high_conf_boost);
    }

    return cs;
}

std::optional<std::pair<double, double>> evaluate_coarse_trust(
    double top1_sim, double pf_spread, double recon_sim_ema,
    const TrustConfig& cfg)
{
    double denom = cfg.coarse_sim_ceiling - cfg.coarse_sim_floor;
    double sim_conf;
    if (denom <= 0)
        sim_conf = (top1_sim >= cfg.coarse_sim_floor) ? 1.0 : 0.0;
    else
        sim_conf = clamp((top1_sim - cfg.coarse_sim_floor) / denom, 0.0, 1.0);

    double spread_boost = clamp(pf_spread / 100.0, 0.0, 1.0);
    double recon_suppress = 1.0 - clamp(recon_sim_ema - 0.3, 0.0, 0.5) * 2.0;
    double coarse_conf = sim_conf * (0.5 + 0.5 * spread_boost) * recon_suppress;

    if (coarse_conf < 0.15) return std::nullopt;

    double frac = clamp(coarse_conf * 0.8, 0.1, cfg.coarse_max_teleport_fraction);
    double sigma_lo = cfg.coarse_teleport_sigma_range[0];
    double sigma_hi = cfg.coarse_teleport_sigma_range[1];
    double sigma = sigma_hi / (1.0 + coarse_conf);
    sigma = clamp(sigma, sigma_lo, sigma_hi);

    return std::make_pair(frac, sigma);
}

// ─── TrustTracker ───────────────────────────────────────────────

TrustTracker::TrustTracker(const TrustConfig& cfg) : cfg_(cfg) {}

double TrustTracker::ema(double old_val, double new_val) const {
    if (!initialized_) return new_val;
    double a = cfg_.ema_alpha;
    return a * new_val + (1.0 - a) * old_val;
}

FrameTrust TrustTracker::evaluate_frame(
    const std::vector<CandidateInput>& fine_candidates,
    double recon_sim, double top1_sim,
    double pf_east, double pf_north,
    double pf_spread, double lost_spread,
    double altitude_m, double /*timestamp_s*/)
{
    recon_sim_ema_ = ema(recon_sim_ema_, recon_sim);
    double pf_self_conf = 1.0 - clamp(pf_spread / lost_spread, 0.0, 1.0);
    pf_self_confidence_ema_ = ema(pf_self_confidence_ema_, pf_self_conf);
    initialized_ = true;

    bool drift_detected = (
        pf_self_confidence_ema_ < cfg_.drift_pf_confidence_threshold &&
        recon_sim_ema_ < cfg_.drift_recon_sim_threshold &&
        top1_sim >= cfg_.drift_coarse_sim_min
    );

    FrameTrust result;
    result.drift_detected = drift_detected;
    result.recon_sim_ema = recon_sim_ema_;
    result.pf_self_confidence_ema = pf_self_confidence_ema_;

    for (auto& c : fine_candidates) {
        if (drift_detected && (c.source == "satellite" || c.source == "mosaic"))
            continue;

        double sim = (c.source == "satellite" || c.source == "mosaic") ? recon_sim : top1_sim;
        double eff_spread = drift_detected ? (lost_spread + 1.0) : pf_spread;

        auto cs = score_candidate(
            c.east_m, c.north_m, c.inliers, c.heading_deg, c.source,
            sim, pf_east, pf_north, eff_spread, lost_spread, altitude_m, cfg_,
            c.flow_consistency, c.flow_magnitude_cv, c.inlier_ratio);
        result.all_scores.push_back(cs);
    }

    if (!result.all_scores.empty()) {
        auto best_it = std::max_element(result.all_scores.begin(), result.all_scores.end(),
            [](const CandidateScore& a, const CandidateScore& b) { return a.confidence < b.confidence; });
        if (best_it->confidence >= cfg_.min_confidence)
            result.best = *best_it;
    }

    return result;
}

double TrustTracker::get_sigma_scale(double confidence) const {
    return 1.0 / (0.2 + 0.8 * confidence);
}

double TrustTracker::get_effective_sigma(double confidence, double base_sigma, bool is_static) const {
    double scale = get_sigma_scale(confidence);
    if (scale > cfg_.drift_sigma_cap && pf_self_confidence_ema_ < cfg_.drift_pf_confidence_threshold)
        scale = cfg_.drift_sigma_cap;
    double sigma = base_sigma * clamp(scale, cfg_.sigma_min_scale, cfg_.sigma_max_scale);
    if (is_static)
        sigma = std::max(sigma, base_sigma * 2.0);
    return sigma;
}

double TrustTracker::get_effective_kappa(double confidence, bool is_static) const {
    double kappa = 15.0 * confidence;
    if (is_static) kappa *= 0.3;
    return kappa;
}

std::optional<GlobalCorrectionResult> TrustTracker::evaluate_global_correction(
    double recon_sim,
    std::optional<double> coarse_fine_east,
    std::optional<double> coarse_fine_north,
    int coarse_fine_inliers,
    std::optional<double> coarse_fine_heading,
    double pf_east, double pf_north, double pf_spread)
{
    if (!cfg_.global_corr_enabled) return std::nullopt;

    if (global_correction_cooldown_ > 0) global_correction_cooldown_--;

    if (pf_spread < cfg_.global_corr_min_spread_m) {
        recon_diverge_count_ = 0;
        coarse_fine_hits_.clear();
        return std::nullopt;
    }

    bool divergent = (recon_sim < cfg_.global_corr_recon_sim_threshold &&
                      pf_spread < cfg_.global_corr_max_spread_m);

    if (divergent) {
        recon_diverge_count_++;
        if (coarse_fine_east.has_value() && coarse_fine_north.has_value() &&
            coarse_fine_inliers >= cfg_.global_corr_min_inliers) {
            coarse_fine_hits_.emplace_back(
                coarse_fine_east.value(), coarse_fine_north.value(),
                coarse_fine_inliers, coarse_fine_heading);
        }
    } else {
        recon_diverge_count_ = 0;
        coarse_fine_hits_.clear();
    }

    if (recon_diverge_count_ < cfg_.global_corr_min_frames ||
        static_cast<int>(coarse_fine_hits_.size()) < cfg_.global_corr_min_frames ||
        global_correction_cooldown_ > 0) {
        return std::nullopt;
    }

    // Compute median position
    std::vector<double> easts, norths;
    std::vector<int> inliers_list;
    for (auto& [e, n, inl, hdg] : coarse_fine_hits_) {
        easts.push_back(e);
        norths.push_back(n);
        inliers_list.push_back(inl);
    }

    auto median = [](std::vector<double> v) -> double {
        std::sort(v.begin(), v.end());
        int n = v.size();
        return (n % 2 == 0) ? (v[n / 2 - 1] + v[n / 2]) / 2.0 : v[n / 2];
    };

    double med_e = median(easts);
    double med_n = median(norths);

    // Spatial consistency
    double var_e = 0, var_n = 0;
    for (auto e : easts) var_e += (e - med_e) * (e - med_e);
    for (auto n : norths) var_n += (n - med_n) * (n - med_n);
    var_e /= easts.size();
    var_n /= norths.size();
    double hit_spread = std::sqrt(var_e + var_n);
    if (hit_spread > cfg_.global_corr_position_consistency_m) return std::nullopt;

    double dx = med_e - pf_east, dy = med_n - pf_north;
    double dist_from_pf = std::sqrt(dx * dx + dy * dy);
    if (dist_from_pf < pf_spread * 2.0) return std::nullopt;

    // Heading: circular median
    std::optional<double> result_heading;
    std::vector<double> headings;
    for (auto& [e, n, inl, hdg] : coarse_fine_hits_)
        if (hdg.has_value()) headings.push_back(hdg.value());

    if (!headings.empty()) {
        double sin_sum = 0, cos_sum = 0;
        for (double h : headings) {
            double rad = h * M_PI / 180.0;
            sin_sum += std::sin(rad);
            cos_sum += std::cos(rad);
        }
        double mean_hdg = std::fmod(std::atan2(sin_sum, cos_sum) * 180.0 / M_PI, 360.0);
        if (mean_hdg < 0) mean_hdg += 360.0;
        double R = std::sqrt(sin_sum * sin_sum + cos_sum * cos_sum) / headings.size();
        double circ_std = std::sqrt(-2.0 * std::log(std::max(R, 1e-6))) * 180.0 / M_PI;
        if (circ_std < 30.0) result_heading = mean_hdg;
    }

    global_correction_cooldown_ = cfg_.global_corr_cooldown_frames;
    recon_diverge_count_ = 0;
    coarse_fine_hits_.clear();

    std::sort(inliers_list.begin(), inliers_list.end());
    int med_inl = inliers_list[inliers_list.size() / 2];

    return GlobalCorrectionResult{med_e, med_n, result_heading, med_inl,
                                   static_cast<int>(easts.size()), dist_from_pf};
}

}  // namespace pf
