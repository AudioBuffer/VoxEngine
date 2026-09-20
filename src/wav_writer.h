#pragma once
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>

namespace vox {

// 生成 WAV 字节（内存版，用于管道播放，不落盘）
inline std::vector<unsigned char> makeWav16(const std::vector<float>& data, int sampleRate) {
    std::vector<unsigned char> out;
    const uint16_t channels = 1, bits = 16;
    const uint16_t blockAlign = uint16_t(channels * bits / 8);
    const uint32_t byteRate = uint32_t(sampleRate) * blockAlign;
    const uint32_t dataSize = uint32_t(data.size()) * blockAlign;
    auto p32 = [&](uint32_t v){ for (int i=0;i<4;i++) out.push_back((unsigned char)((v >> (8*i)) & 0xFF)); };
    auto p16 = [&](uint16_t v){ for (int i=0;i<2;i++) out.push_back((unsigned char)((v >> (8*i)) & 0xFF)); };
    auto tag = [&](const char* t){ for (int i=0;i<4;i++) out.push_back((unsigned char)t[i]); };
    tag("RIFF"); p32(36 + dataSize); tag("WAVE");
    tag("fmt "); p32(16); p16(1); p16(channels); p32(uint32_t(sampleRate)); p32(byteRate); p16(blockAlign); p16(bits);
    tag("data"); p32(dataSize);
    out.reserve(out.size() + dataSize);
    for (float s : data) {
        float v = s;
        if (v > 1.0f) v = 1.0f; else if (v < -1.0f) v = -1.0f;
        p16(uint16_t(int16_t(std::lround(double(v) * 32767.0))));
    }
    return out;
}

inline bool writeWav16(const std::string& path, const std::vector<float>& data, int sampleRate) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const uint16_t channels = 1, bits = 16;
    const uint16_t blockAlign = uint16_t(channels * bits / 8);
    const uint32_t byteRate = uint32_t(sampleRate) * blockAlign;
    const uint32_t dataSize = uint32_t(data.size()) * blockAlign;

    auto w32 = [&](uint32_t v) { std::fwrite(&v, 4, 1, f); };
    auto w16 = [&](uint16_t v) { std::fwrite(&v, 2, 1, f); };
    std::fwrite("RIFF", 1, 4, f); w32(36 + dataSize);
    std::fwrite("WAVE", 1, 4, f);
    std::fwrite("fmt ", 1, 4, f); w32(16); w16(1); w16(channels);
    w32(uint32_t(sampleRate)); w32(byteRate); w16(blockAlign); w16(bits);
    std::fwrite("data", 1, 4, f); w32(dataSize);

    for (float s : data) {
        float v = s;
        if (v > 1.0f) v = 1.0f; else if (v < -1.0f) v = -1.0f;
        const int16_t q = int16_t(std::lround(double(v) * 32767.0));
        std::fwrite(&q, 2, 1, f);
    }
    std::fclose(f);
    return true;
}

} // namespace vox
