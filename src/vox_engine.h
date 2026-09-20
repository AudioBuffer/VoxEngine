#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include "lf_glottal.h"
#include "kl_tract.h"
#include "noise_source.h"
#include "onnx_engine.h"

namespace vox {

struct Vowel { const char* name; double f[5]; };

// a o e i u v(ü) —— F1..F5，F4/F5 提供高频空气感
inline const Vowel kSlots[6] = {
    {"a", { 800, 1200, 2500, 3500, 4500 }},
    {"o", { 450,  800, 2830, 3500, 4500 }},
    {"e", { 500, 1750, 2450, 3500, 4500 }},
    {"i", { 300, 2300, 3000, 3500, 4500 }},
    {"u", { 350,  600, 2400, 3500, 4500 }},
    {"v", { 300, 1700, 2160, 3500, 4500 }},
};
inline const double kBw[5] = { 60, 90, 120, 150, 200 };

struct Note {
    double start{0.0};
    double dur{1.0};
    int    vowel{0};
    double f0{220.0};
    int    consonant{0};
    double cdur{0.08};
};

class VoxEngine {
public:
    explicit VoxEngine(double fs = 44100.0)
        : fs_(fs), glottal_(fs), tract_(fs), cons_(fs) {}

    void setVoiceScale(double s)   { voiceScale_ = std::clamp(s, 0.8, 1.4); }
    void setBreath(double b)       { breath_ = std::clamp(b, 0.0, 1.0); }
    void setVibrato(double d, double rate = 5.0, double delay = 0.35) {
        vibDepth_ = d; vibRate_ = rate; vibDelay_ = delay;
    }
    void setJitterShimmer(double j, double s) { glottal_.setJitter(j); glottal_.setShimmer(s); }
    // 声门开相指数增长率（音色尖锐度/张力）；自然值 2~5
    void setGlottalAlpha(double a) { glottal_.build(0.35, 0.55, 0.02, a); }
    // 完整声门形状：tp/te/ta 为周期比例，alpha 为开相增长率
    void setGlottalShape(double tp, double te, double ta, double alpha) {
        glottal_.build(tp, te, ta, alpha);
    }
    // 共振峰带宽整体缩放（1.0 = kBw 原值）
    void setBandwidthScale(double s) {
        bwScale_ = std::clamp(s, 0.2, 4.0);
    }
    // 直通声道（只测声门源时用）
    void setBypassTract(bool b) { bypassTract_ = b; }
    // 唇端辐射高通系数（0=全通, 0.95=强微分）
    void setRadiation(double r)    { tract_.setRadiation(r); }
    void setRadiationFor(double r){ tract_.setRadiation(r); }
    void setTiltComp(double k)    { tract_.setTiltComp(k); }
    void setHfGain(double g)      { tract_.setHfGain(g); }
    void setAutoNormalize(bool b) { autoNorm_ = b; }
    // ONNX 后处理钩子。接口内无模型时 ready()==false，这里整段跳过 -> 纯参数合成。
    void setOnnxEngine(OnnxEngine* o) { onnx_ = o; }
    bool onnxActive() const { return onnx_ && onnx_->ready(); }
    void setLegato(double ms)      { glide_ = ms / 1000.0; }
    void setNasal(double gain)     { nasalGain_ = std::clamp(gain, 0.0, 1.0); }
    void setSeed(unsigned s)       { glottal_.setSeed(s); cons_.setSeed(s ^ 0x9e37u); }
    // 外部直接指定 F1..F5（GUI 滑块用）
    void setFormants(const double f[5]) {
        for (int i = 0; i < 5; ++i) fOverride_[i] = f[i];
        hasOverride_ = true;
    }

    void addNote(const Note& n)    { notes_.push_back(n); }
    void clear()                   { notes_.clear(); }

    // f0Out 非空时，同时回传逐样本的 F0 曲线（供绘图）
    std::vector<float> render(double totalSeconds, std::vector<float>* f0Out = nullptr) {
        const int n = int(totalSeconds * fs_);
        std::vector<float> out(n, 0.0f);
        if (f0Out) f0Out->assign(size_t(n), 0.0f);
        glottal_.reset(); tract_.reset(); cons_.reset();

        int curVowel = -1;
        const Note* curNote = nullptr;   // 当前音符，用于取「上一个音符」的音高
        double prevF0 = notes_.empty() ? 220.0 : notes_.front().f0;

        for (int i = 0; i < n; ++i) {
            const double t = double(i) / fs_;
            const Note* nt = nullptr;
            for (const auto& c : notes_)
                if (t >= c.start && t < c.start + c.dur) { nt = &c; break; }
            if (!nt) { out[i] = 0.0f; continue; }

            // 音符号切换：滑音必须从「上一个音符」的音高出发。
            // （原来 prevF0 只在循环外赋值一次，导致每个音符都从第一个音符的
            //   音高滑入；换元音/拖动音符后听起来就是「某个音不准」。）
            if (nt != curNote) {
                if (curNote) prevF0 = curNote->f0;
                curNote = nt;
            }
            if (nt->vowel != curVowel) {
                curVowel = nt->vowel;
                configureTract(curVowel);
            }

            const double tt = t - nt->start;
            double f0 = nt->f0;

            // Smoothstep 滑音
            if (glide_ > 1e-6 && tt < glide_) {
                const double x = tt / glide_;
                const double s = x * x * (3.0 - 2.0 * x);
                f0 = prevF0 + (nt->f0 - prevF0) * s;
            }
            // 延迟颤音（5Hz，淡入）
            if (vibDepth_ > 0.0 && tt > vibDelay_) {
                const double fade = std::min(1.0, (tt - vibDelay_) / 0.4);
                f0 *= 1.0 + vibDepth_ * fade * std::sin(2.0 * kPi * vibRate_ * tt);
            }

            // 回传设定轨迹（基频+滑音+颤音）。jitter 是逐周期微扰，
            // 混进来会让曲线逐点乱跳，不适合作为对比基准。
            if (f0Out) (*f0Out)[size_t(i)] = float(f0);
            const double voice = glottal_.process(f0);
            const double cons  = cons_.process(tt, nt->consonant);
            // 声带增益由辅音类型连续给出（爆破音成阻期真正静默），
            // 不再按 cons!=0 逐样本硬切换——那会在噪声过零处产生咔哒声。
            const double vg = cons_.voicingGain(tt, nt->consonant);
            const double src = voice * vg + cons;

            const double y = bypassTract_ ? src : tract_.process(src);
            // 气息噪声（呼吸感）
            const double breath = breath_ * 0.05 * cons_.process(tt, ConsonantSource::F);
            out[i] = float(y + breath);
        }
        // 输出峰值归一化（防止溢出/削波；可在外部关闭）
        if (autoNorm_ && !out.empty()) {
            double pk = 0.0;
            for (float v : out) pk = std::max(pk, (double)std::fabs(v));
            if (pk > 1e-12) {
                const float g = float(0.85 / pk);
                for (float& v : out) v *= g;
            }
        }
        // ONNX 润色：没有模型就直接放弃，保持纯参数合成结果
        if (onnx_ && onnx_->ready()) onnx_->process(out, fs_);
        return out;
    }

private:
    void configureTract(int vi) {
        double f[5];
        for (int i = 0; i < 5; ++i)
            f[i] = (hasOverride_ ? fOverride_[i] : kSlots[vi].f[i]) * voiceScale_;
        double bw[5];
        for (int i = 0; i < 5; ++i) bw[i] = kBw[i] * bwScale_;
        tract_.configure(f, bw, 5, 280.0, 120.0, nasalGain_);
    }

    double fs_;
    LFGlottal glottal_;
    KLTract tract_;
    ConsonantSource cons_;
    std::vector<Note> notes_;
    double voiceScale_{1.0}, breath_{0.0};
    double vibDepth_{0.0}, vibRate_{5.0}, vibDelay_{0.35};
    double glide_{0.06}, nasalGain_{0.0};
    double fOverride_[5]{800,1200,2500,3500,4500};
    bool hasOverride_{false};
    double bwScale_{1.0};        // 共振峰带宽缩放
    bool bypassTract_{false};    // 直通声道（只测声门源）
    bool autoNorm_{true};        // 输出峰值归一化
    OnnxEngine* onnx_{nullptr};  // 可为空：空则纯参数合成
};

} // namespace vox
