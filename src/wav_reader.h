#pragma once
// 最小 WAV 读取：PCM16 / PCM32 / float32；多声道下混单声道；非目标采样率线性重采样
#include <string>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace vox {

inline uint32_t rd32(const unsigned char* p){ return p[0]|(p[1]<<8)|(p[2]<<16)|((uint32_t)p[3]<<24); }
inline uint16_t rd16(const unsigned char* p){ return (uint16_t)(p[0]|(p[1]<<8)); }

inline bool readWavMono(const std::string& path, std::vector<float>& out,
                        double targetSr = 44100.0) {
    out.clear();
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz < 44) { std::fclose(f); return false; }
    std::vector<unsigned char> buf((size_t)sz);
    if (std::fread(buf.data(), 1, (size_t)sz, f) != (size_t)sz) { std::fclose(f); return false; }
    std::fclose(f);
    if (std::memcmp(buf.data(), "RIFF", 4) != 0 || std::memcmp(buf.data()+8, "WAVE", 4) != 0) return false;

    int fmt = 0, ch = 0, bits = 0; double sr = 0;
    size_t dataOff = 0, dataLen = 0;
    size_t p = 12;
    while (p + 8 <= buf.size()) {
        const char* id = (const char*)(buf.data() + p);
        const uint32_t len = rd32(buf.data() + p + 4);
        const size_t body = p + 8;
        if (std::memcmp(id, "fmt ", 4) == 0 && body + 16 <= buf.size()) {
            fmt  = rd16(buf.data() + body);
            ch   = rd16(buf.data() + body + 2);
            sr   = rd32(buf.data() + body + 4);
            bits = rd16(buf.data() + body + 14);
        } else if (std::memcmp(id, "data", 4) == 0) {
            dataOff = body;
            dataLen = std::min<size_t>(len, buf.size() - body);
        }
        p = body + len + (len & 1);
    }
    if (!fmt || ch <= 0 || !sr || !dataOff || !dataLen) return false;
    const bool isFloat = (fmt == 3);
    const int bytes = bits / 8;
    if (bytes <= 0) return false;
    const size_t frames = dataLen / ((size_t)bytes * (size_t)ch);
    if (!frames) return false;

    std::vector<float> mono(frames, 0.0f);
    const unsigned char* d = buf.data() + dataOff;
    for (size_t i = 0; i < frames; ++i) {
        double acc = 0.0;
        for (int c = 0; c < ch; ++c) {
            const unsigned char* q = d + (i * (size_t)ch + (size_t)c) * (size_t)bytes;
            double v = 0.0;
            if (isFloat && bits == 32) {
                float fv; std::memcpy(&fv, q, 4); v = fv;
            } else if (bits == 16) {
                int16_t s; std::memcpy(&s, q, 2); v = s / 32768.0;
            } else if (bits == 32) {
                int32_t s; std::memcpy(&s, q, 4); v = s / 2147483648.0;
            } else if (bits == 8) {
                v = ((int)q[0] - 128) / 128.0;
            }
            acc += v;
        }
        mono[i] = (float)(acc / ch);
    }

    // 线性重采样到目标采样率
    if (std::fabs(sr - targetSr) > 1.0) {
        const double ratio = targetSr / sr;
        const size_t n = (size_t)(frames * ratio);
        out.resize(n);
        for (size_t i = 0; i < n; ++i) {
            const double x = i / ratio;
            const size_t i0 = (size_t)x;
            const size_t i1 = std::min(i0 + 1, frames - 1);
            const double t = x - i0;
            out[i] = (float)(mono[i0] * (1.0 - t) + mono[i1] * t);
        }
    } else {
        out.swap(mono);
    }
    return true;
}

} // namespace vox
