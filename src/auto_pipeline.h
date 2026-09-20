#pragma once
// 全自动流水线：歌唱模板文件夹 + 说话样本文件夹 -> 提取音高 -> 造成对数据 -> 写训练集 -> 生成训练脚本
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <dirent.h>
#include <sys/stat.h>
#include "wav_reader.h"
#include "f0_track.h"
#include "vox_engine.h"

namespace vox {

struct PipeStats {
    int singFiles{0}, speechFiles{0}, okFiles{0};
    size_t f0Chunks{0}, polishChunks{0};
    double f0Min{0}, f0Max{0}, f0Mean{0};
    double seconds{0};
};

inline bool dirExists(const std::string& d){ struct stat st; return ::stat(d.c_str(),&st)==0 && S_ISDIR(st.st_mode); }

// ---- 目录扫描：分类 + 时长（WAV 读头，MP3 走 ffprobe，无则按码率估算）----
struct DirScan {
    std::vector<std::string> wavs, mp3s, others;
    double wavSec{0}, mp3Sec{0};
};

inline double fileSeconds(const std::string& path, bool isWav) {
    if (isWav) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return 0;
        unsigned char h[44];
        if (std::fread(h, 1, 44, f) != 44) { std::fclose(f); return 0; }
        const uint16_t ch = (uint16_t)(h[22] | (h[23] << 8));
        const uint32_t sr = (uint32_t)(h[24] | (h[25]<<8) | (h[26]<<16) | ((uint32_t)h[27]<<24));
        const uint16_t bits = (uint16_t)(h[34] | (h[35] << 8));
        std::fseek(f, 0, SEEK_END);
        const long sz = std::ftell(f);
        std::fclose(f);
        const double bytesPerSec = (double)sr * ch * (bits / 8.0);
        if (bytesPerSec <= 0) return 0;
        return std::max(0.0, (double)(sz - 44) / bytesPerSec);
    }
    // MP3：优先 ffprobe
    std::string cmd = "ffprobe -v error -show_entries format=duration -of csv=p=0 \"" + path + "\" 2>/dev/null";
    if (FILE* p = ::popen(cmd.c_str(), "r")) {
        char buf[128] = {0};
        if (std::fgets(buf, sizeof buf, p)) { ::pclose(p); double d = std::atof(buf); if (d > 0) return d; }
        else ::pclose(p);
    }
    // 兜底：按 128 kbps 估算
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return 0;
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fclose(f);
    return sz > 0 ? (double)sz * 8.0 / 128000.0 : 0.0;
}

inline DirScan scanDir(const std::string& dir) {
    DirScan d;
    DIR* dp = ::opendir(dir.c_str());
    if (!dp) return d;
    while (dirent* e = ::readdir(dp)) {
        const std::string n = e->d_name;
        if (n == "." || n == "..") continue;
        std::string low = n;
        for (auto& c : low) c = (char)std::tolower((unsigned char)c);
        const std::string full = dir + "/" + n;
        struct stat stx;
        if (::stat(full.c_str(), &stx) != 0 || !S_ISREG(stx.st_mode)) continue;
        if (low.size() >= 4 && low.rfind(".wav") == low.size() - 4) {
            d.wavs.push_back(full); d.wavSec += fileSeconds(full, true);
        } else if (low.size() >= 4 && low.rfind(".mp3") == low.size() - 4) {
            d.mp3s.push_back(full); d.mp3Sec += fileSeconds(full, false);
        } else {
            d.others.push_back(n);
        }
    }
    ::closedir(dp);
    std::sort(d.wavs.begin(), d.wavs.end());
    std::sort(d.mp3s.begin(), d.mp3s.end());
    std::sort(d.others.begin(), d.others.end());
    return d;
}

// ---- 输入校验 ----
enum class CheckResult { Ok, Refuse, NeedConfirmMp3 };

inline std::string fmtMin(double sec) {
    char b[64];
    std::snprintf(b, sizeof b, "%.1f 分钟", sec / 60.0);
    return b;
}

inline CheckResult validateInputs(const DirScan& sing, const DirScan& speech,
                                  double minSingMin, double minSpeechMin,
                                  std::string& report, bool& needMp3Confirm) {
    needMp3Confirm = false;
    report.clear();
    auto refuse = [&](const std::string& why) {
        report += "【拒绝生成】" + why + "\n";
        return CheckResult::Refuse;
    };
    // 规则 1：非 WAV 文件（MP3 除外）一律拒绝
    if (!sing.others.empty() || !speech.others.empty()) {
        std::string all;
        for (auto& f : sing.others)   all += "  歌唱模板/" + f + "\n";
        for (auto& f : speech.others) all += "  说话样本/" + f + "\n";
        return refuse("目录内存在非 WAV 文件（共 " +
                      std::to_string(sing.others.size() + speech.others.size()) +
                      " 个），请先清理或移走：\n" + all);
    }
    // 规则 2：存在 MP3 -> 需要用户确认后才继续
    if (!sing.mp3s.empty() || !speech.mp3s.empty()) {
        const size_t n = sing.mp3s.size() + speech.mp3s.size();
        report += "【需要确认】目录内有 " + std::to_string(n) + " 个 MP3 文件"
                  "（歌唱 " + std::to_string(sing.mp3s.size()) + "，说话 " +
                  std::to_string(speech.mp3s.size()) + "）。\n";
        report += "  合计时长约 " + fmtMin(sing.mp3Sec + speech.mp3Sec) + "。\n";
        report += "  确认后将用 ffmpeg 转为 44100Hz 单声道 WAV 再处理。\n";
        needMp3Confirm = true;
        return CheckResult::NeedConfirmMp3;
    }
    // 规则 3/4：时长门槛
    double singMin = (sing.wavSec + sing.mp3Sec) / 60.0;
    double spkMin  = (speech.wavSec + speech.mp3Sec) / 60.0;
    if (singMin < minSingMin) {
        char b[160];
        std::snprintf(b, sizeof b,
            "歌唱模板总时长不足：当前 %.1f 分钟，要求不少于 %.0f 分钟（差 %.1f 分钟）",
            singMin, minSingMin, minSingMin - singMin);
        return refuse(b);
    }
    if (spkMin < minSpeechMin) {
        char b[160];
        std::snprintf(b, sizeof b,
            "说话样本总时长不足：当前 %.1f 分钟，要求不少于 %.0f 分钟（差 %.1f 分钟）",
            spkMin, minSpeechMin, minSpeechMin - spkMin);
        return refuse(b);
    }
    char b[256];
    std::snprintf(b, sizeof b,
        "【校验通过】歌唱模板 %.1f 分钟（≥%.0f），说话样本 %.1f 分钟（≥%.0f），全部为 WAV",
        singMin, minSingMin, spkMin, minSpeechMin);
    report = b;
    return CheckResult::Ok;
}

// ---- MP3 -> WAV 转换 ----
// 递归建目录
inline bool mkdirs(const std::string& path) {
    if (path.empty()) return false;
    std::string cur;
    for (size_t i = 0; i < path.size(); ++i) {
        cur += path[i];
        if (path[i] == '/' || i + 1 == path.size()) {
            if (cur == "/" || cur.empty()) continue;
            ::mkdir(cur.c_str(), 0755);
        }
    }
    struct stat stx;
    return ::stat(path.c_str(), &stx) == 0 && S_ISDIR(stx.st_mode);
}

inline int convertMp3s(const std::vector<std::string>& mp3s, const std::string& dstDir,
                       std::string& log) {
    if (mp3s.empty()) return 0;
    if (!mkdirs(dstDir)) { log += "  无法创建目录 " + dstDir + "\n"; return 0; }
    int n = 0;
    for (const auto& p : mp3s) {
        std::string base = p.substr(p.find_last_of('/') + 1);
        base = base.substr(0, base.find_last_of('.'));
        const std::string out = dstDir + "/" + base + ".wav";
        std::string cmd = "ffmpeg -y -v error -i \"" + p + "\" -ac 1 -ar 44100 -c:a pcm_s16le \"" + out + "\"";
        const int rc = std::system(cmd.c_str());
        struct stat stx;
        const bool made = (rc == 0) && ::stat(out.c_str(), &stx) == 0 && stx.st_size > 44;
        log += std::string(made ? "  转换 " : "  转换失败 ") + base + "\n";
        if (made) ++n;
    }
    return n;
}

inline std::vector<std::string> scanWavs(const std::string& dir) {
    std::vector<std::string> out;
    DIR* dp = ::opendir(dir.c_str());
    if (!dp) return out;
    while (dirent* e = ::readdir(dp)) {
        std::string n = e->d_name;
        if (n.size() < 5) continue;
        std::string low = n;
        for (auto& c : low) c = (char)std::tolower((unsigned char)c);
        if (low.rfind(".wav") == low.size() - 4) out.push_back(dir + "/" + n);
    }
    ::closedir(dp);
    std::sort(out.begin(), out.end());
    return out;
}

// ---- npy 写出（float32, C 顺序）----
inline bool writeNpy(const std::string& path, const std::vector<float>& data,
                     const std::vector<size_t>& shape) {
    std::ostringstream sh; sh << "(";
    for (size_t i = 0; i < shape.size(); ++i) { sh << shape[i]; if (i + 1 < shape.size()) sh << ", "; }
    if (shape.size() == 1) sh << ",";
    sh << ")";
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': " + sh.str() + ", }";
    const size_t pre = 6 + 2 + 2;                       // magic+ver+len
    size_t pad = 64 - ((pre + hdr.size() + 1) % 64);
    if (pad == 64) pad = 0;
    hdr += std::string(pad, ' ') + "\n";
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    const char magic[6] = { (char)0x93, 'N', 'U', 'M', 'P', 'Y' };
    f.write(magic, 6);
    f.put(1); f.put(0);
    uint16_t hl = (uint16_t)hdr.size();
    f.put((char)(hl & 0xff)); f.put((char)(hl >> 8));
    f.write(hdr.data(), (std::streamsize)hdr.size());
    f.write((const char*)data.data(), (std::streamsize)(data.size() * sizeof(float)));
    return true;
}

// ---- 由 F0 轨迹驱动引擎合成（与原始录音逐帧对齐）----
inline std::vector<float> synthFromF0(const std::vector<float>& f0, double fs) {
    const double HOP = 0.015;
    VoxEngine e(fs);
    e.clear();
    const size_t group = 4;                             // 每 60ms 一个音符号
    double t = 0.0;
    for (size_t i = 0; i < f0.size(); i += group) {
        double s = 0; int n = 0;
        for (size_t k = i; k < std::min(i + group, f0.size()); ++k)
            if (f0[k] > 0) { s += f0[k]; ++n; }
        if (n > 0) {
            Note nt;
            nt.start = t; nt.dur = HOP * group;
            nt.f0 = s / n;
            nt.vowel = 0; nt.consonant = 0; nt.cdur = 0.05;
            e.addNote(nt);
        }
        t += HOP * group;
    }
    const double total = std::max(0.2, t);
    std::vector<float> out = e.render(total);
    out.resize((size_t)(total * fs), 0.0f);
    return out;
}

// ---- 主流程 ----
inline bool runAutoPipelineFiles(const std::vector<std::string>& files,
                                 const std::string& outDir, PipeStats& st, std::string& log);

inline bool runAutoPipeline(const std::string& singDir, const std::string& speechDir,
                            const std::string& outDir, PipeStats& st, std::string& log,
                            double minSingMin = 60.0, double minSpeechMin = 30.0) {
    auto say = [&](const std::string& s){ log += s + "\n"; };
    st = PipeStats{};
    if (!dirExists(outDir) && !mkdirs(outDir)) { say("输出目录无法创建: " + outDir); return false; }

    // ---------- 输入校验 ----------
    DirScan sa = scanDir(singDir), sb = scanDir(speechDir);
    std::string rep; bool needMp3 = false;
    const CheckResult cr = validateInputs(sa, sb, minSingMin, minSpeechMin, rep, needMp3);
    say(rep);
    if (cr == CheckResult::Refuse) return false;

    std::vector<std::string> a = sa.wavs, b = sb.wavs;
    if (cr == CheckResult::NeedConfirmMp3) {
        // 非 GUI 场景（命令行/测试）默认视为已确认
        const char* force = std::getenv("VOX_MP3_CONFIRM");
        if (!(force && std::string(force) == "1")) {
            say("  未确认，已停止。确认后请重试（或设 VOX_MP3_CONFIRM=1）。");
            return false;
        }
        say("  已确认，开始转换 MP3 -> WAV");
        const int n1 = convertMp3s(sa.mp3s, outDir + "/_converted/sing", log);
        const int n2 = convertMp3s(sb.mp3s, outDir + "/_converted/speech", log);
        say("  共转换 " + std::to_string(n1 + n2) + " 个");
        auto a2 = scanDir(outDir + "/_converted/sing");
        auto b2 = scanDir(outDir + "/_converted/speech");
        a.insert(a.end(), a2.wavs.begin(), a2.wavs.end());
        b.insert(b.end(), b2.wavs.begin(), b2.wavs.end());
    }

    st.singFiles   = (int)a.size();
    st.speechFiles = (int)b.size();
    std::vector<std::string> files = a;
    files.insert(files.end(), b.begin(), b.end());
    say("歌唱模板 " + std::to_string(a.size()) + " 个文件，说话样本 " + std::to_string(b.size()) + " 个文件");
    if (files.empty()) { say("两个文件夹里都没有可用音频"); return false; }

    const size_t N = 16384, HOP = 8192, FHOP = 661, FLEN = 1323, T = 23;
    std::vector<float> fX, fY, pX, pY;
    double fmin = 1e9, fmax = -1e9, fsum = 0; size_t fcnt = 0;

    for (const auto& path : files) {
        std::vector<float> pcm;
        if (!readWavMono(path, pcm, 44100.0) || pcm.size() < N) {
            say("  跳过(读失败或过短): " + path); continue;
        }
        std::vector<float> f0;
        trackF0(pcm, 44100.0, f0, 60.0, 900.0);
        size_t voiced = 0;
        for (float v : f0) if (v > 0) { ++voiced; fmin = std::min(fmin,(double)v); fmax = std::max(fmax,(double)v); fsum += v; ++fcnt; }
        if (voiced < 20) { say("  跳过(几乎无浊音): " + path); continue; }
        ++st.okFiles;
        st.seconds += (double)pcm.size() / 44100.0;

        std::vector<float> synth = synthFromF0(f0, 44100.0);
        const size_t M = std::min(pcm.size(), synth.size());

        for (size_t s = 0; s + N <= M; s += HOP) {
            // F0 数据集
            for (size_t i = 0; i < N; ++i) fX.push_back(pcm[s + i]);
            for (size_t i = 0; i < T; ++i) {
                const size_t c = (s + FHOP * i + FLEN / 2) / FHOP;
                fY.push_back(c < f0.size() ? f0[c] : 0.0f);
            }
            ++st.f0Chunks;
            // 润色成对数据集（引擎合成 = 输入，真人录音 = 目标）
            for (size_t i = 0; i < N; ++i) { pX.push_back(synth[s + i]); pY.push_back(pcm[s + i]); }
            ++st.polishChunks;
        }
        say("  " + path.substr(path.find_last_of('/') + 1) + "  ok");
    }
    if (!st.f0Chunks) { say("没有产出任何样本块"); return false; }

    if (!writeNpy(outDir + "/f0_X.npy", fX, {st.f0Chunks, 1, N}) ||
        !writeNpy(outDir + "/f0_Y.npy", fY, {st.f0Chunks, T}) ||
        !writeNpy(outDir + "/polish_X.npy", pX, {st.polishChunks, 1, N}) ||
        !writeNpy(outDir + "/polish_Y.npy", pY, {st.polishChunks, 1, N})) {
        say("写 npy 失败"); return false;
    }
    st.f0Min = (fcnt ? fmin : 0); st.f0Max = (fcnt ? fmax : 0);
    st.f0Mean = fcnt ? fsum / fcnt : 0;
    say("F0 数据集: " + std::to_string(st.f0Chunks) + " 块   F0 范围 " +
        std::to_string((int)st.f0Min) + "~" + std::to_string((int)st.f0Max) + "Hz  均值 " +
        std::to_string((int)st.f0Mean) + "Hz");
    say("润色数据集: " + std::to_string(st.polishChunks) + " 块（引擎合成 -> 真人录音，逐帧对齐）");
    say("已写出 f0_X/f0_Y/polish_X/polish_Y.npy 到 " + outDir);
    return true;
}

} // namespace vox
