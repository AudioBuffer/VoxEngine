#pragma once
#include "phonemes.h"
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <cctype>
#include <algorithm>
#include "vox_engine.h"
#include "wav_writer.h"

namespace vox {

struct SeqItem {
    std::string phoneme;
    double dur{0.5};
    double f0{180.0};
    int    vowel{0};
    int    cons{0};
    bool   ok{false};
};

/** 音素符号 -> (元音槽, 辅音类型) */
inline bool lookupPhoneme(const std::string& p, int& vowel, int& cons) {
    return lookupPhonemeAll(p, vowel, cons);   // 中/英/日 总表（phonemes.h）
}

inline double jsonNum(const std::string& obj, const char* key, double def) {
    const size_t p = obj.find(key);
    if (p == std::string::npos) return def;
    const size_t c = obj.find(':', p);
    if (c == std::string::npos) return def;
    return std::strtod(obj.c_str() + c + 1, nullptr);
}

/** 最小 JSON 解析：只认 [{"phoneme":..,"duration":..,"f0":..}, ...] */
inline std::vector<SeqItem> parseSeqJson(const std::string& j) {
    std::vector<SeqItem> out;
    size_t pos = 0;
    while (true) {
        const size_t a = j.find('{', pos);
        if (a == std::string::npos) break;
        const size_t b = j.find('}', a);
        if (b == std::string::npos) break;
        const std::string obj = j.substr(a, b - a + 1);
        pos = b + 1;

        SeqItem it;
        const size_t p = obj.find("\"phoneme\"");
        if (p != std::string::npos) {
            const size_t q1 = obj.find('"', obj.find(':', p));
            if (q1 != std::string::npos) {
                const size_t q2 = obj.find('"', q1 + 1);
                if (q2 != std::string::npos) it.phoneme = obj.substr(q1 + 1, q2 - q1 - 1);
            }
        }
        it.dur = jsonNum(obj, "duration", 0.5);
        it.f0  = jsonNum(obj, "f0", 180.0);
        it.ok  = lookupPhoneme(it.phoneme, it.vowel, it.cons);
        out.push_back(it);
    }
    return out;
}

/** 【核心接口】连续发音 -> 内存缓冲（供共享直出，不落盘）
 *  逐段调用单音素渲染，每段首尾 15ms 线性淡化，顺序拼接。
 *  先算总采样数一次性分配；段缓冲复用，不反复分配。 */
inline std::vector<float> renderSequence(const std::vector<SeqItem>& items, double fs,
                                         double voice = 1.0, double breath = 0.12)
{
    const double FADE_SEC = 0.015;                   // 15ms
    const size_t FADE_N = (size_t)std::llround(FADE_SEC * fs);

    std::vector<size_t> segLen(items.size(), 0);
    size_t total = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        const double d = items[i].ok ? items[i].dur : 0.0;
        segLen[i] = (size_t)std::llround(d * fs);
        total += segLen[i];
    }
    std::vector<float> out(total, 0.0f);             // 一次性分配
    if (total == 0) return out;

    std::vector<float> seg;                          // 复用段缓冲
    size_t off = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        const SeqItem& it = items[i];
        const size_t n = segLen[i];
        if (n == 0) continue;

        VoxEngine eng(fs);                           // 复用现有单音素渲染
        eng.setVoiceScale(voice);
        eng.setBreath(breath);
        eng.clear();
        Note nt;
        nt.start = 0.0; nt.dur = it.dur; nt.f0 = it.f0;
        nt.vowel = it.vowel; nt.consonant = it.cons; nt.cdur = 0.08;
        eng.addNote(nt);
        seg = eng.render(it.dur);
        seg.resize(n, 0.0f);

        size_t f = FADE_N;                           // 首尾线性淡化
        if (f * 2 > n) f = n / 2;
        if (f > 0) {
            for (size_t k = 0; k < f; ++k) {
                const float g = float(k) / float(f);
                seg[k] *= g;
                seg[n - 1 - k] *= g;
            }
        }
        std::copy(seg.begin(), seg.begin() + (std::ptrdiff_t)n,
                  out.begin() + (std::ptrdiff_t)off);
        off += n;
    }
    return out;
}

/** 便捷入口：JSON -> 内存缓冲 */
inline std::vector<float> renderSequenceJson(const std::string& json, double fs = 44100.0) {
    return renderSequence(parseSeqJson(json), fs);
}

/** 旁路：把内存缓冲写成 WAV。不是 UI 动作，只在需要落盘时调用。 */
inline bool writeSequenceWav(const std::vector<float>& buf, const std::string& path,
                             double fs = 44100.0)
{
    if (buf.empty()) return false;
    std::vector<float> t = buf;
    double pk = 1e-12;
    for (float v : t) pk = std::max(pk, std::fabs((double)v));
    if (pk > 1e-9) for (float& v : t) v = float(double(v) * (0.85 / pk));
    return writeWav16(path, t, (int)fs);
}

} // namespace vox
