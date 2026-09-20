#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include "lf_glottal.h"

namespace vox {

// ---------------------------------------------------------------
// Kelly-Lochbaum 声道格型滤波器
//   1) 由 F1..F5 的谐振器级联出全极点多项式 A(z) = Π(1 + c1 z^-1 + c2 z^-2)
//   2) Levinson 步降递推精确解出反射系数 k_m
//   3) 用逆格型(合成格型)结构实现，传输函数即 1/A(z)
//      => 共振峰位置与设定值严格一致（非拟合）
// 鼻音分支：并联谐振器做相减，产生反共振零点。
// ---------------------------------------------------------------
class KLTract {
public:
    explicit KLTract(double fs) : fs_(fs) {}

    void configure(const double* formants, const double* bw, int nf,
                   double nasalFreq, double nasalBw, double nasalGain)
    {
        nf = std::min(nf, 5);

        std::vector<double> a(1, 1.0);
        for (int i = 0; i < nf; ++i) {
            const double r  = std::exp(-kPi * bw[i] / fs_);
            const double th = 2.0 * kPi * std::clamp(formants[i], 50.0, fs_ * 0.45) / fs_;
            const double c1 = -2.0 * r * std::cos(th);
            const double c2 = r * r;
            std::vector<double> nx(a.size() + 2, 0.0);
            for (size_t j = 0; j < a.size(); ++j) {
                nx[j]     += a[j];
                nx[j + 1] += a[j] * c1;
                nx[j + 2] += a[j] * c2;
            }
            a.swap(nx);
        }

        const int order = int(a.size()) - 1;
        a_.assign(a.begin(), a.end());      // 保留 A(z) 供直接型实现
        std::vector<double> cur = a;
        k_.assign(order, 0.0);
        for (int m = order; m >= 1; --m) {
            const double km = cur[m];
            k_[m - 1] = km;
            const double den = 1.0 - km * km;
            if (std::fabs(den) < 1e-9) break;
            std::vector<double> prev(m, 0.0);
            for (int i = 0; i < m; ++i) prev[i] = (cur[i] - km * cur[m - i]) / den;
            cur.swap(prev);
        }
        sections_ = int(k_.size());
        state_.assign(sections_, 0.0);
        order_ = order;
        yhist_.assign(order_, 0.0);

        nasalGain_ = std::clamp(nasalGain, 0.0, 1.0);
        const double rn  = std::exp(-kPi * nasalBw / fs_);
        const double thn = 2.0 * kPi * std::clamp(nasalFreq, 50.0, 1000.0) / fs_;
        nc1_ = -2.0 * rn * std::cos(thn);
        nc2_ = rn * rn;
        nz1_ = nz2_ = 0.0;

        // 级联在 F1 处的峰值增益，用于给并联支路定标
        {
            const double w0 = 2.0 * kPi * std::clamp(formants[0], 50.0, fs_ * 0.45) / fs_;
            double re = 1.0, im = 0.0;
            for (size_t j = 1; j < a.size(); ++j) {
                re += a[j] * std::cos(j * w0);
                im -= a[j] * std::sin(j * w0);
            }
            const double m = std::sqrt(re * re + im * im);
            cascGain_ = (m > 1e-12) ? 1.0 / m : 1.0;
        }

        // Klatt 式并联支路：F4/F5 由声源直接激励，幅度显式可控，
        // 用来补回级联结构固有的高频过度衰减。
        par_.clear();
        for (int i = 2; i < nf; ++i) {
            const double r  = std::exp(-kPi * bw[i] / fs_);
            const double w0 = 2.0 * kPi * std::clamp(formants[i], 50.0, fs_ * 0.45) / fs_;
            Par p;
            p.a1 = -2.0 * r * std::cos(w0);
            p.a2 = r * r;
            const double re = 1.0 + p.a1 * std::cos(w0) + p.a2 * std::cos(2.0 * w0);
            const double im = -p.a1 * std::sin(w0) - p.a2 * std::sin(2.0 * w0);
            p.b0 = std::sqrt(re * re + im * im);   // 峰值增益 = 1
            p.z1 = p.z2 = 0.0;
            par_.push_back(p);
        }
        // 权重由自然元音包络反解: F3:F4:F5 = 1 : 0.333 : 0.122
        static const double w3[3] = { 0.75, 0.25, 0.09 };
        for (size_t i = 0; i < par_.size() && i < 3; ++i) par_[i].g = w3[i];

        // 逐元音响度归一化：用冲激响应能量估计滤波器总增益
        {
            reset();
            normGain_ = 1.0;   // 关键：探针必须用未缩放状态测量，
                               // 否则每次 configure 会复合放大，换元音时把前面的音符压没
            double e = 0.0;
            for (int n = 0; n < 4096; ++n) {
                const double y = process(n == 0 ? 1.0 : 0.0);
                if (n > 32) e += y * y;
            }
            normGain_ = (e > 1e-30) ? 1.0 / std::sqrt(e) : 1.0;
            reset();
        }
        rx1_ = 0.0;
    }

    // u 为声门源。用与格型严格等价的全极点直接型实现（数值稳定）。
    double process(double u) {
        const double un = u * normGain_;   // 归一化源：级联与并联共用
        double f = un;
        for (int i = 0; i < order_; ++i) f -= a_[i + 1] * yhist_[i];
        for (int i = order_ - 1; i > 0; --i) yhist_[i] = yhist_[i - 1];
        if (order_ > 0) yhist_[0] = f;
        // 鼻音支路（并联相减 -> 反共振）
        const double nin  = u * nasalGain_;
        const double nout = nin - nc1_ * nz1_ - nc2_ * nz2_;
        nz2_ = nz1_; nz1_ = nout;
        double y = f - 0.35 * nout;

        // 唇端辐射：一阶高通(微分)
        double rad = y - rad_ * rx1_;
        rx1_ = y;
        // 倾斜补偿：再叠一级微分(+6dB/oct)
        if (tilt_ > 1e-6) {
            const double t = rad - tilt_ * tx1_;
            tx1_ = rad;
            rad = t;
        }
        // 并联支路：F4/F5 直接由声源激励，补回高频
        if (hfGain_ > 1e-6) {
            double par = 0.0;
            for (size_t i = 0; i < par_.size(); ++i) {
                Par& p = par_[i];
                const double y = p.b0 * un - p.a1 * p.z1 - p.a2 * p.z2;
                p.z2 = p.z1; p.z1 = y;
                par += p.g * y;   // g = 权重，实时乘 hfGain_
            }
            rad += par * hfGain_ * cascGain_;   // 与级联同量级
        }
        return rad;
    }

    // 唇端辐射高通系数：0=不辐射(全通)，0.95=强微分
    void setRadiation(double r) { rad_ = std::clamp(r, 0.0, 0.99); }
    // 高频倾斜补偿：0=关，1=再叠一级微分(+6dB/oct)
    void setTiltComp(double k) { tilt_ = std::clamp(k, 0.0, 2.0); }
    // 并联高频支路增益：0=纯级联(旧行为), 1=补足自然高频
    void setHfGain(double g) { hfGain_ = std::clamp(g, 0.0, 3.0); }

    void reset() {
        std::fill(state_.begin(), state_.end(), 0.0);
        std::fill(yhist_.begin(), yhist_.end(), 0.0);
        nz1_ = nz2_ = rx1_ = tx1_ = 0.0;
        for (size_t i = 0; i < par_.size(); ++i) { par_[i].z1 = par_[i].z2 = 0.0; }
    }
    int sections() const { return sections_; }

private:
    double fs_;
    int sections_{0}, order_{0};
    std::vector<double> k_, a_, state_, yhist_;
    double nasalGain_{0.0}, nc1_{0.0}, nc2_{0.0}, nz1_{0.0}, nz2_{0.0};
    double rx1_{0.0};
    double rad_{0.95};
    double tilt_{0.0}, tx1_{0.0};
    struct Par { double b0{1.0}, a1{0.0}, a2{0.0}, g{0.0}, z1{0.0}, z2{0.0}; };
    std::vector<Par> par_;
    double hfGain_{0.16};   // 默认并联支路增益（实测自然包络）
    double cascGain_{1.0};
    double normGain_{1.0};
};

} // namespace vox
