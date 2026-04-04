#pragma once

#include <cmath>
#include <optional>
#include <string>
#include <vector>

namespace pf {

// ─── Motion ─────────────────────────────────────────────────────

struct MotionDelta {
    double dx_m = 0.0;
    double dy_m = 0.0;
    double heading_deg = 0.0;
    double dt_s = 0.0;
};

// ─── Coarse matching ────────────────────────────────────────────

struct CoarseResult {
    std::vector<std::string> top_k_names;
    std::vector<float> top_k_sims;
    std::vector<int> top_k_indices;
};

// ─── Fine matching ──────────────────────────────────────────────

struct FineResult {
    double lat = 0.0;
    double lon = 0.0;
    int inliers = 0;
    std::string method;  // "pnp" or "homography"
    std::optional<double> heading_deg;
    std::string patch_name;
    float flow_consistency = 0.0f;
    float flow_magnitude_cv = 1.0f;
    float inlier_ratio = 0.0f;
    std::optional<float> flow_heading_deg;
    int n_total_matches = 0;
    double pnp_altitude = 0.0;  // PnP-estimated altitude (0 = homography, no estimate)
};

// ─── Particle filter phases ─────────────────────────────────────

enum class Phase { UNINIT, DISPERSED, CONVERGING, TRACKING };

inline const char* phase_name(Phase p) {
    switch (p) {
        case Phase::UNINIT:     return "UNINIT";
        case Phase::DISPERSED:  return "DISPERSED";
        case Phase::CONVERGING: return "CONVERGING";
        case Phase::TRACKING:   return "TRACKING";
    }
    return "UNKNOWN";
}

// ─── GPS patch metadata ─────────────────────────────────────────

struct PatchBounds {
    double min_lat, max_lat, min_lon, max_lon;
};

struct PatchMeta {
    double lat, lon;
    PatchBounds bounds;
};

// Flat metadata used for PnP/homography GPS conversion
struct FlatMeta {
    double min_lat, max_lat, min_lon, max_lon;
    int patch_h, patch_w;
};

// ─── Flow geometry metrics ──────────────────────────────────────

struct FlowMetrics {
    float flow_consistency = 0.0f;
    float flow_magnitude_cv = 1.0f;
    std::optional<float> flow_heading_deg;
};

}  // namespace pf
