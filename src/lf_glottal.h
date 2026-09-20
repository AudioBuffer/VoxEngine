#pragma once
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>

namespace vox {

inline constexpr double kPi = 3.14159265358979323846;

// ---------------------------------------------------------------
// LF (Liljencrants-Fant) 声门脉冲模型 —— 波表实现
//   开启段 0..Te : e^(alpha*t) * sin(wg*t)
//   返回段 Te..T0: 指数回收段，与开启段在 Te 处幅值连续
// 整周期去直流(净流量为0)并峰值归一，任意 f0 用线性插值查表。
// ---------------------------------------------------------------
class LFGlottal {
public:
    explicit LFGlottal(double fs, int tableSize = 4096)
        : fs_(fs), n_(tableSize) { build(0.35, 0.55, 0.02, 2.5); }

    // tp/te/ta 均为基音周期的比例
    // alphaT0: 开相指数增长率，单位是「每周期」。归一化时间下 alpha 就等于 alpha*T0。
    // 自然嗓音 2~5；原来写死的 1/(2*Ta)=25 属极端值，脉冲近似冲击 -> 声音发尖发毛躁。
    void build(double tp, double te, double ta, double alphaT0 = 2.5) {
        tp = std::clamp(tp, 0.15, 0.45);
        te = std::clamp(te, tp + 0.05, 0.90);
        ta = std::clamp(ta, 0.005, 0.10);

        const double wg    = kPi / (2.0 * tp);
        const double alpha = std::clamp(alphaT0, 0.5, 30.0);
        const double eps   = 1.0 / ta;

        const double openAtTe = std::exp(alpha * te) * std::sin(wg * te);
        const double retAtTe  = 1.0 - std::exp(-eps * (1.0 - te));
        const double scale    = (std::fabs(retAtTe) > 1e-12) ? (openAtTe / retAtTe) : 0.0;

        table_.assign(n_, 0.0);
        for (int i = 0; i < n_; ++i) {
            const double x = double(i) / double(n_);
            table_[i] = (x < te)
                ? std::exp(alpha * x) * std::sin(wg * x)
                : scale * (std::exp(-eps * (x - te)) - std::exp(-eps * (1.0 - te)));
        }
        double mean = 0.0;
        for (double v : table_) mean += v;
        mean /= double(n_);
        for (double& v : table_) v -= mean;
        double pk = 1e-12;
        for (double v : table_) pk = std::max(pk, std::fabs(v));
        for (double& v : table_) v /= pk;

        phase_ = 0.0;
        resetPeriod();
    }

    void reset() { phase_ = 0.0; resetPeriod(); }
    void setJitter(double j)  { jitter_  = std::clamp(j, 0.0, 0.20); }
    void setShimmer(double s) { shimmer_ = std::clamp(s, 0.0, 0.40); }
    void setSeed(unsigned s)  { rng_.seed(s); }

    // 当前周期的 jitter 缩放（供上层回传真实瞬时 F0）
    double lastPeriodScale() const { return periodScale_; }

    double process(double f0) {
        if (f0 < 1.0) f0 = 1.0;
        phase_ += (f0 * periodScale_) / fs_;
        const double out = lookup(phase_);
        if (phase_ >= 1.0) { phase_ -= std::floor(phase_); resetPeriod(); }
        return out * ampScale_;
    }

private:
    void resetPeriod() {
        if (jitter_ > 0.0) {
            std::uniform_real_distribution<double> d(-jitter_, jitter_);
            periodScale_ = 1.0 + d(rng_);
        } else periodScale_ = 1.0;
        if (shimmer_ > 0.0) {
            std::uniform_real_distribution<double> d(-shimmer_, shimmer_);
            ampScale_ = 1.0 + d(rng_);
        } else ampScale_ = 1.0;
    }
    double lookup(double ph) const {
        const double x = ph * double(n_);
        const int i0 = int(x) % n_;
        const int i1 = (i0 + 1) % n_;
        const double f = x - std::floor(x);
        return table_[i0] * (1.0 - f) + table_[i1] * f;
    }
    double fs_;
    int n_;
    std::vector<double> table_;
    double phase_{0.0}, periodScale_{1.0}, ampScale_{1.0};
    double jitter_{0.0}, shimmer_{0.0};
    std::mt19937 rng_{12345u};
};

} // namespace vox
