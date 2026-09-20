#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include "seq.h"
#include "vox_engine.h"
#include "wav_writer.h"

using namespace vox;

// 朴素 DFT 找谱峰（0.1-5kHz，25Hz 网格）——用于验证共振峰是否落在设定位置
static std::vector<std::pair<double,double>> spectrumPeaks(const std::vector<float>& x, double fs, int topN)
{
    const int N = std::min<int>(8192, int(x.size()));
    if (N < 512) return {};
    std::vector<std::pair<double,double>> mags;
    for (double f = 100.0; f <= 5000.0; f += 25.0) {
        double re = 0.0, im = 0.0;
        const double w = 2.0 * kPi * f / fs;
        for (int n = 0; n < N; ++n) {
            const double win = 0.5 - 0.5 * std::cos(2.0 * kPi * n / (N - 1));
            const double s = double(x[n]) * win;
            re += s * std::cos(w * n);
            im -= s * std::sin(w * n);
        }
        mags.push_back({ f, std::sqrt(re * re + im * im) });
    }
    std::vector<std::pair<double,double>> pk;
    for (size_t i = 1; i + 1 < mags.size(); ++i)
        if (mags[i].second > mags[i-1].second && mags[i].second >= mags[i+1].second)
            pk.push_back(mags[i]);
    std::sort(pk.begin(), pk.end(), [](auto& a, auto& b){ return a.second > b.second; });
    if (int(pk.size()) > topN) pk.resize(topN);
    std::sort(pk.begin(), pk.end(), [](auto& a, auto& b){ return a.first < b.first; });
    return pk;
}

int main(int argc, char** argv)
{
    // 连续发音入口：参数以 '[' 开头即视为 JSON 序列。核心产出是内存缓冲，
    // 加 --wav=path 才旁路落盘（不做成按钮/默认行为）。
    if (argc > 1 && argv[1][0] == '[') {
        std::vector<float> buf = renderSequenceJson(argv[1], 44100.0);
        for (int i = 2; i < argc; ++i) {
            if (std::strncmp(argv[i], "--wav=", 6) == 0) {
                if (writeSequenceWav(buf, argv[i] + 6)) std::printf("wav -> %s\n", argv[i] + 6);
            }
        }
        double pk = 0.0;
        for (float v : buf) pk = std::max(pk, std::fabs((double)v));
        std::printf("seq: %zu samples (%.2f s)  peak=%.0f  [内存直出]\n",
                    buf.size(), buf.size() / 44100.0, pk);
        return 0;
    }

    std::string vowel = "a";
    std::string outf  = "vox_out.wav";
    double seconds = 1.0, f0 = 220.0;
    double voice = 1.0, breath = 0.12, vib = 0.008, vibdelay = 0.35;
    double jitter = 0.004, shimmer = 0.02, nasal = 0.0, legato = 60.0;
    double cdur = 0.08; int cons = 0;
    double df[5] = {800, 1200, 2500, 3500, 4500};
    bool hasOverride = false;

    int pos = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const size_t eq = a.find('=');
        if (eq != std::string::npos) {
            const std::string k = a.substr(0, eq);
            const std::string vs = a.substr(eq + 1);
            const double v = std::atof(vs.c_str());
            if (k == "f0") f0 = v;
            else if (k == "dur") seconds = v;
            else if (k == "voice") voice = v;
            else if (k == "breath") breath = v;
            else if (k == "vib") vib = v;
            else if (k == "vibdelay") vibdelay = v;
            else if (k == "jitter") jitter = v;
            else if (k == "shimmer") shimmer = v;
            else if (k == "nasal") nasal = v;
            else if (k == "legato") legato = v;
            else if (k == "cons") cons = int(v);
            else if (k == "cdur") cdur = v;
            else if (k == "vowel") vowel = vs;
            else if (k == "out") outf = vs;
            else if (k.size() == 2 && k[0] == 'f' && k[1] >= '1' && k[1] <= '5') { df[k[1] - '1'] = v; hasOverride = true; }
            continue;
        }
        if (pos == 0) vowel = a;
        else if (pos == 1) seconds = std::atof(a.c_str());
        else if (pos == 2) f0 = std::atof(a.c_str());
        else if (pos == 3) outf = a;
        pos++;
    }

    const double fs = 44100.0;
    int vi = 0;
    for (int i = 0; i < 6; ++i) if (vowel == kSlots[i].name) vi = i;

    VoxEngine eng(fs);
    eng.setVoiceScale(voice);
    eng.setBreath(breath);
    eng.setVibrato(vib, 5.0, vibdelay);
    eng.setJitterShimmer(jitter, shimmer);
    eng.setLegato(legato);
    eng.setNasal(nasal);
    if (hasOverride) eng.setFormants(df);

    Note n;
    n.start = 0.0; n.dur = seconds; n.vowel = vi; n.f0 = f0; n.consonant = cons; n.cdur = cdur;
    eng.clear(); eng.addNote(n);

    std::vector<float> buf = eng.render(seconds);

    double pk = 1e-12, sum = 0.0;
    for (float v : buf) { pk = std::max(pk, std::fabs(double(v))); sum += double(v) * double(v); }
    const double rms = std::sqrt(sum / double(buf.size() ? buf.size() : 1));
    const double g = 0.85 / pk;
    for (float& v : buf) v = float(double(v) * g);

    if (!writeWav16(outf, buf, int(fs))) { std::fprintf(stderr, "写 WAV 失败\n"); return 1; }

    std::printf("== VoxEngine V2 物理骨架 ==\n");
    std::printf("元音=%s 时长=%.2fs f0=%.1fHz Fs=%.0f 样本=%zu\n",
                vowel.c_str(), seconds, f0, fs, buf.size());
    std::printf("峰值=%.4f RMS=%.4f -> %s\n", pk, rms, outf.c_str());
    std::printf("共振峰设定:");
    for (int i = 0; i < 5; ++i) std::printf(" %.0f", (hasOverride ? df[i] : kSlots[vi].f[i]) * voice);
    std::printf("\n频谱实测峰:");
    for (auto& p : spectrumPeaks(buf, fs, 5)) std::printf(" %.0fHz", p.first);
    std::printf("\n");
    return 0;
}
