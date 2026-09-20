#pragma once
// 独立基频跟踪：与 GUI 内部分析同一套算法（粗搜+精搜+次谐波纠正+中值滤波+空隙插值）
#include <vector>
#include <cmath>
#include <algorithm>

namespace vox {

// 单帧基频（YIN：差值函数 + 累积均值归一化 + 取第一个谷）
// 不用「自相关全局最大」——那会因长 lag 的余弦凑巧接近峰值而选到 2~5 倍周期，
// 表现为 440Hz 被报成 220/146Hz（半频、三分频）。
inline double pitchAtFrame(const std::vector<float>& x, size_t off, int FRAME,
                           double fs, double fmin, double fmax) {
    const int minLag = std::max(2, (int)(fs / fmax));
    const int maxLag = std::min(FRAME - 2, (int)(fs / fmin));
    if (maxLag <= minLag) return 0.0;
    double e = 0;
    for (int i = 0; i < FRAME; ++i) e += (double)x[off + i] * x[off + i];
    if (std::sqrt(e / FRAME) < 0.008) return 0.0;                 // 静音

    // 1) 差值函数 d(lag) = Σ (x[i] - x[i+lag])^2
    std::vector<double> d((size_t)maxLag + 1, 0.0);
    for (int lag = 1; lag <= maxLag; ++lag) {
        double s = 0;
        for (int i = 0; i + lag < FRAME; ++i) {
            const double t = (double)x[off + i] - (double)x[off + i + lag];
            s += t * t;
        }
        d[(size_t)lag] = s;
    }
    // 2) 累积均值归一化 d'(lag) = d(lag) * lag / Σ_{j<=lag} d(j)
    std::vector<double> dp((size_t)maxLag + 1, 1.0);
    double run = 0.0;
    for (int lag = 1; lag <= maxLag; ++lag) {
        run += d[(size_t)lag];
        dp[(size_t)lag] = (run > 1e-12) ? d[(size_t)lag] * lag / run : 1.0;
    }
    // 3) 从 minLag 起取「第一个」低于阈值的谷 —— 这一步才是避免倍频的关键
    const double kThresh = 0.15;
    int bestLag = -1;
    for (int lag = minLag; lag <= maxLag; ++lag) {
        if (dp[(size_t)lag] < kThresh) {
            while (lag + 1 <= maxLag && dp[(size_t)(lag + 1)] < dp[(size_t)lag]) ++lag;
            bestLag = lag;
            break;
        }
    }
    if (bestLag < 0) {                                            // 都不达标则取全局最小
        double bv = 1e18;
        for (int lag = minLag; lag <= maxLag; ++lag)
            if (dp[(size_t)lag] < bv) { bv = dp[(size_t)lag]; bestLag = lag; }
    }
    if (bestLag <= 0) return 0.0;
    // 4) 抛物线插值精修（亚采样精度）
    double better = (double)bestLag;
    if (bestLag > minLag && bestLag < maxLag) {
        const double a = dp[(size_t)(bestLag - 1)], b = dp[(size_t)bestLag], c = dp[(size_t)(bestLag + 1)];
        const double den = 2.0 * (2.0 * b - a - c);
        if (std::fabs(den) > 1e-12) better = bestLag + (c - a) / den;
    }
    return (better > 0.0) ? fs / better : 0.0;
}

inline void trackF0(const std::vector<float>& pcm, double fs, std::vector<float>& f0,
                    double fmin = 60.0, double fmax = 500.0) {
    f0.clear();
    const int FRAME = (int)(fs * 0.030), HOP = (int)(fs * 0.015);
    if ((int)pcm.size() < FRAME * 2) return;
    const int nf = ((int)pcm.size() - FRAME) / HOP + 1;
    f0.resize((size_t)nf, 0.0f);
    for (int f = 0; f < nf; ++f)
        f0[(size_t)f] = (float)pitchAtFrame(pcm, (size_t)(f * HOP), FRAME, fs, fmin, fmax);

    // 5 点中值滤波
    if (f0.size() >= 5) {
        std::vector<float> tmp = f0;
        for (size_t i = 2; i + 2 < f0.size(); ++i) {
            float w[5] = { tmp[i-2], tmp[i-1], tmp[i], tmp[i+1], tmp[i+2] };
            std::sort(w, w + 5);
            f0[i] = w[2];
        }
    }
    // ≤3 帧空隙线性插值
    size_t i = 0;
    while (i < f0.size()) {
        if (f0[i] > 0) { ++i; continue; }
        size_t j = i;
        while (j < f0.size() && f0[j] <= 0) ++j;
        if (i > 0 && j < f0.size() && (j - i) <= 3) {
            for (size_t k = i; k < j; ++k) {
                const double t = double(k - i + 1) / double(j - i + 1);
                f0[k] = (float)(f0[i-1] * (1.0 - t) + f0[j] * t);
            }
        }
        i = j;
    }
}

} // namespace vox
