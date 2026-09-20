#pragma once
#include <random>
#include <cmath>
#include <algorithm>

namespace vox {

// ---------------------------------------------------------------
// 辅音发生器：擦音用带通整形白噪，爆破音用延迟脉冲包络，鼻音交给声道鼻支路
// ---------------------------------------------------------------
class ConsonantSource {
public:
    enum Type { None = 0, S, Sh, F, P, B, M };

    explicit ConsonantSource(double fs) : fs_(fs) {}
    void setSeed(unsigned s) { rng_.seed(s); }
    void reset() { z1_ = lp1_ = z2_ = 0.0; }

    double process(double t, int type) {
        if (type == None || type == M) return 0.0;
        const double x = (cdur_ > 1e-6) ? (t / cdur_) : 1.0;
        if (x < 0.0 || x >= 1.0) return 0.0;
        const double env = envelope(x, type);
        const double w = dist_(rng_);
        switch (type) {
            case S:  return env * 0.9 * hp(w, 0.80);
            case Sh: return env * 0.9 * bp(w, 0.45);
            case F:  return env * 0.5 * hp(w, 0.45);
            // 爆破：宽带瞬态 + 送气尾（hp 给高频冲击感，bp 给 1~4k 重心）
            case P:  return env * (1.60 * hp(w, 0.88) + 0.90 * bp(w, 0.65));
            case B:  return env * (0.80 * hp(w, 0.72) + 0.95 * bp(w, 0.42));
            default: return 0.0;
        }
    }

    void setDuration(double d) { cdur_ = d; }

    // 成阻期静默系数：爆破音在成阻期压掉声带，释放后平滑恢复
    // （硬切换会产生爆音，所以恢复段用 ~13ms 斜坡）
    double voicingGain(double t, int type) const {
        if (type != P && type != B) return 1.0;
        if (cdur_ <= 1e-6) return 1.0;
        const double x = t / cdur_;
        if (x <= 0.0 || x >= 1.0) return 1.0;
        const double onset = 0.55;
        if (x < onset) return 0.0;
        const double u = (x - onset) / (1.0 - onset);
        return std::min(1.0, u / 0.35);
    }

private:
    double envelope(double x, int type) const {
        if (type == P || type == B) {
            const double onset = 0.55;              // 成阻期：静默
            if (x < onset) return 0.0;
            const double u = (x - onset) / std::max(1e-9, 1.0 - onset);
            const double rise = std::min(1.0, (x - onset) / 0.02);   // ~1.6ms 起振，避免纯脉冲
            const double burst = std::exp(-u * 9.0);                 // 瞬态
            const double asp   = std::exp(-u * 2.2) * 0.35;          // 送气尾
            return rise * (burst + asp);
        }
        const double a = std::min(1.0, x / 0.15);
        const double r = std::min(1.0, (1.0 - x) / 0.20);
        return a * r;
    }
    double hp(double x, double c) { const double y = x - c * z1_; z1_ = x; return y; }
    double bp(double x, double c) {
        lp1_ = lp1_ + c * (x - lp1_);
        const double y = lp1_ - z2_;
        z2_ = lp1_;
        return y;
    }
    double fs_;
    double cdur_{0.08};
    double z1_{0.0}, lp1_{0.0}, z2_{0.0};
    std::mt19937 rng_{999u};
    std::uniform_real_distribution<double> dist_{-1.0, 1.0};
};

} // namespace vox
