// VoxEngine C++ TUI —— 界面与 DSP 都在 C++ 层；推理层交 JS(node) 子进程
// 用法: ./build/vox_tui            交互
//       ./build/vox_tui --selftest 非交互自检
#include <cstdio>
#include "i18n.h"
using vox::T;
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <termios.h>
#include <unistd.h>
#include "vox_engine.h"
#include "wav_writer.h"

using namespace vox;

// ---------------- 参数表 ----------------
struct EnumParam { const char* label; std::vector<const char*> vals; int i{0}; };
struct NumParam  { const char* label; double mn, mx, step, v; const char* unit; };

static const char* kVowels[] = {"a","o","e","i","u","v"};
static const char* kCons[]   = {T("无", "none"),"s","sh","f","p","b","m"};

struct UI {
    int vowel{0}, cons{0};
    NumParam n[13] = {
        {T("基频 F0", "Pitch F0"),   80,   400,  5,    220,   "Hz"},
        {T("时长", "Duration"),      0.3,  5,    0.1,  1.0,   "s" },
        {T("声线缩放", "Voice scale"),  0.8,  1.4,  0.02, 1.0,   "x" },
        {"F1",        200,  1100, 10,   800,   "Hz"},
        {"F2",        600,  2600, 20,   1200,  "Hz"},
        {"F3",        1800, 3400, 20,   2500,  "Hz"},
        {T("F4 空气感", "F4 air"), 2800, 4200, 20,   3500,  "Hz"},
        {T("F5 空气感", "F5 air"), 3800, 5200, 20,   4500,  "Hz"},
        {T("气息感", "Breathiness"),    0,    1,    0.02, 0.12,  ""  },
        {T("颤音深度", "Vibrato depth"),  0,    0.05, 0.002,0.008, ""  },
        {T("抖动", "Jitter"),    0,    0.03, 0.002,0.004, ""  },
        {T("微幅", "Shimmer"),   0,    0.15, 0.01, 0.02,  ""  },
        {T("鼻音耦合", "Nasal coupling"),  0,    1,    0.05, 0.0,   ""  },
    };
    NumParam inf[3] = {
        {T("LDM 步数", "LDM steps"), 0, 6, 1,    3,    ""},
        {T("人味", "Humanity"),     0, 1, 0.05, 0.35, ""},
        {T("推理开关", "Inference"), 0, 1, 1,    1,    ""},
    };
    int sel{0};
    int total() const { return 2 + 13 + 3; }   // 元音 + 辅音 + 13个数值 + 推理3项
};

// 按可执行文件位置定位推理脚本，避免 cwd 不同导致失败
static std::string exeDir() {
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return std::string(".");
    buf[n] = 0;
    std::string s(buf);
    const size_t p = s.find_last_of('/');
    return (p == std::string::npos) ? std::string(".") : s.substr(0, p);
}
static std::string inferScript() {
    const std::string d = exeDir();          // .../VoxEngine/build
    const std::string cands[3] = { d + "/../js/infer.mjs", d + "/js/infer.mjs", "js/infer.mjs" };
    for (const std::string& c : cands) {
        FILE* f = std::fopen(c.c_str(), "r");
        if (f) { std::fclose(f); return c; }
    }
    return std::string("js/infer.mjs");
}

static termios g_orig;
static void rawOn()  { termios r; tcgetattr(STDIN_FILENO,&g_orig); r=g_orig; r.c_lflag &= ~(ICANON|ECHO); r.c_cc[VMIN]=1; r.c_cc[VTIME]=0; tcsetattr(STDIN_FILENO,TCSANOW,&r); }
static void rawOff() { tcsetattr(STDIN_FILENO,TCSANOW,&g_orig); }

static std::string bar(double v, double mn, double mx, int w=18) {
    int n = int(std::round((v-mn)/(mx-mn) * w));
    n = std::max(0, std::min(w, n));
    std::string s = " [";
    for (int i=0;i<w;i++) s += (i<n) ? "\u2588" : "\u00b7";
    s += "]";
    return s;
}
static std::string fmt(double v) {
    char b[32];
    if (std::fabs(v - std::round(v)) < 1e-9) std::snprintf(b,sizeof b,"%.0f",v);
    else std::snprintf(b,sizeof b,"%.3f",v);
    return b;
}

static void draw(const UI& u, const std::string& status) {
    std::string o = "\x1b[2J\x1b[H";
    o += T("\x1b[1;36m VoxEngine \x1b[0m  ↑↓ 选择   ←→ 调节   r 存文件   p 播放(不落盘)   q 退出\n", "\x1b[1;36m VoxEngine \x1b[0m  up/down select   left/right adjust   r save file   p play (no file)   q quit\n");
    o += "\x1b[90m----------------------------------------------\x1b[0m\n";
    int row = 0;
    auto line = [&](const std::string& text) { o += (row == u.sel ? "\x1b[7m> " : "  ") + text + "\x1b[0m\n"; row++; };

    line(std::string("\xe5\x85\x83\xe9\x9f\xb3         ").substr(0,0) + "\xe5\x85\x83\xe9\x9f\xb3         " + kVowels[u.vowel]);
    line(std::string("\xe8\xbe\x85\xe9\x9f\xb3         ") + kCons[u.cons]);
    for (int i=0;i<13;i++) {
        const NumParam& p = u.n[i];
        line(std::string(p.label) + std::string(std::max<int>(1, 14 - std::strlen(p.label)), ' ')
             + fmt(p.v) + " " + p.unit + bar(p.v, p.mn, p.mx));
    }
    for (int i=0;i<3;i++) {
        const NumParam& p = u.inf[i];
        line(std::string("[JS] ") + p.label + std::string(std::max<int>(1, 9 - std::strlen(p.label)), ' ')
             + fmt(p.v) + " " + bar(p.v, p.mn, p.mx));
    }
    o += "\x1b[90m----------------------------------------------\x1b[0m\n";
    o += status;
    std::fputs(o.c_str(), stdout);
    std::fflush(stdout);
}

static NumParam& curNum(UI& u, int sel) {
    if (sel >= 2 && sel < 15) return u.n[sel-2];
    return u.inf[sel-15];
}

// 切换元音时，把该元音的共振峰载入 F1..F5（否则元音选择器是死的）
static void loadVowelFormants(UI& u) {
    for (int i = 0; i < 5; ++i) u.n[3 + i].v = kSlots[u.vowel].f[i];
}

static bool renderToBuf(UI& u, std::vector<float>& out) {
    const double fs = 44100.0;
    VoxEngine eng(fs);
    eng.setVoiceScale(u.n[2].v);
    eng.setBreath(u.n[8].v);
    eng.setVibrato(u.n[9].v, 5.0, 0.35);
    eng.setJitterShimmer(u.n[10].v, u.n[11].v);
    eng.setNasal(u.n[12].v);
    double f[5] = { u.n[3].v, u.n[4].v, u.n[5].v, u.n[6].v, u.n[7].v };
    eng.setFormants(f);
    Note nt; nt.start = 0; nt.dur = u.n[1].v; nt.vowel = u.vowel;
    nt.f0 = u.n[0].v; nt.consonant = u.cons; nt.cdur = 0.08;
    eng.clear(); eng.addNote(nt);
    out = eng.render(u.n[1].v);
    double pk = 1e-12;
    for (float v : out) pk = std::max(pk, std::fabs(double(v)));
    if (pk < 1e-9) return false;
    for (float& v : out) v = float(double(v) * (0.85 / pk));
    return true;
}

static int renderAll(UI& u) {
    std::vector<float> buf;
    if (!renderToBuf(u, buf)) return 0;
    if (!writeWav16("tui_cpp.wav", buf, 44100)) return 0;
    return 1;
}

int main(int argc, char** argv) {
    UI u;
    loadVowelFormants(u);
    const bool self = (argc > 1 && std::strcmp(argv[1], "--selftest") == 0);
    for (int i = 2; i < argc; ++i) {
        if (std::strncmp(argv[i], "--vowel=", 8) == 0) { u.vowel = std::atoi(argv[i] + 8) % 6; loadVowelFormants(u); }
    }

    if (self) {
        u.vowel = 1; u.n[0].v = 180; u.n[8].v = 0.25; u.cons = 1;
        if (!renderAll(u)) { std::printf(T("渲染失败\n", "render failed\n")); return 1; }
        std::printf(T("C++ 层渲染 OK -> tui_cpp.wav\n", "C++ layer render OK -> tui_cpp.wav\n"));
        const int steps = int(u.inf[0].v);
        if (steps > 0) {
            std::string cmd = "node \"" + inferScript() + "\" tui_cpp.wav tui_cpp_ref.wav --steps=" + fmt(u.inf[0].v)
                            + " --human=" + fmt(u.inf[1].v) + " --f0=" + fmt(u.n[0].v);
            std::printf(T("调用 JS 推理层: %s\n", "calling JS inference layer: %s\n"), cmd.c_str());
            const int rc = std::system(cmd.c_str());
            std::printf(T("推理层返回 %d\n", "inference layer returned %d\n"), rc);
        }
        for (int i = 2; i < argc; ++i) {
            if (std::strcmp(argv[i], "--pipe") == 0) {
                std::vector<float> b2;
                if (!renderToBuf(u, b2)) { std::printf(T("pipe: 渲染失败\n", "pipe: render failed\n")); return 1; }
                const std::vector<unsigned char> w = makeWav16(b2, 44100);
                const std::string pc = "node \"" + inferScript() + "\" - - --steps=2 --human=0.3 --f0=220 > pipe_out.wav";
                FILE* pp = popen(pc.c_str(), "w");
                if (!pp) { std::printf(T("pipe: popen 失败\n", "pipe: popen failed\n")); return 1; }
                std::fwrite(w.data(), 1, w.size(), pp);
                const int rc = pclose(pp);
                std::printf(T("pipe 测试: 内存 %zu 字节 -> JS(stdin/stdout) -> pipe_out.wav  rc=%d\n", "pipe test: memory %zu bytes -> JS(stdin/stdout) -> pipe_out.wav  rc=%d\n"), w.size(), rc);
            }
        }
        std::printf(T("产物: tui_cpp.wav, tui_cpp_ref.wav\n", "outputs: tui_cpp.wav, tui_cpp_ref.wav\n"));
        return 0;
    }

    if (!isatty(STDIN_FILENO)) { std::printf(T("需要 TTY；非交互请用 --selftest\n", "TTY required; use --selftest for non-interactive\n")); return 0; }
    rawOn();
    std::atexit(rawOff);
    draw(u, T("\x1b[90m就绪\x1b[0m", "\x1b[90mready\x1b[0m"));

    while (true) {
        char c = 0;
        if (read(STDIN_FILENO, &c, 1) != 1) break;
        if (c == 'q' || c == 3) break;
        if (c == 'r') {
            draw(u, T("\x1b[33m渲染中…\x1b[0m", "\x1b[33mrendering...\x1b[0m"));
            if (!renderAll(u)) { draw(u, T("\x1b[31m渲染失败\x1b[0m", "\x1b[31mrender failed\x1b[0m")); continue; }
            std::string st = T("\x1b[32m完成 -> tui_cpp.wav\x1b[0m", "\x1b[32mdone -> tui_cpp.wav\x1b[0m");
            if (int(u.inf[0].v) > 0 && int(u.inf[2].v) == 1) {
                std::string cmd = "node \"" + inferScript() + "\" tui_cpp.wav tui_cpp_ref.wav --steps=" + fmt(u.inf[0].v)
                                + " --human=" + fmt(u.inf[1].v) + " --f0=" + fmt(u.n[0].v) + " >/dev/null 2>&1";
                int rc = std::system(cmd.c_str());
                st = (rc == 0) ? T("\x1b[32m完成 -> tui_cpp_ref.wav (JS推理)\x1b[0m", "\x1b[32mdone -> tui_cpp_ref.wav (JS inference)\x1b[0m")
                               : T("\x1b[31mJS 推理层失败（骨架已存 tui_cpp.wav）\x1b[0m", "\x1b[31mJS inference layer failed (skeleton saved to tui_cpp.wav)\x1b[0m");
            }
            draw(u, st + T("   \x1b[90m按 p 播放\x1b[0m", "   \x1b[90mpress p to play\x1b[0m"));
            continue;
        }
        if (c == 'p') {
            // 渲染到内存 -> 管道 -> mpv，全程不创建文件
            draw(u, T("\x1b[33m渲染中(内存)…\x1b[0m", "\x1b[33mrendering (in memory)...\x1b[0m"));
            std::vector<float> buf;
            if (!renderToBuf(u, buf)) { draw(u, T("\x1b[31m渲染失败\x1b[0m", "\x1b[31mrender failed\x1b[0m")); continue; }
            const std::vector<unsigned char> wav = makeWav16(buf, 44100);
            std::string cmd;
            if (int(u.inf[0].v) > 0 && int(u.inf[2].v) == 1) {
                cmd = "node \"" + inferScript() + "\" - - --steps=" + fmt(u.inf[0].v)
                    + " --human=" + fmt(u.inf[1].v) + " --f0=" + fmt(u.n[0].v)
                    + " | mpv --no-video --really-quiet --no-terminal - 2>/dev/null";
            } else {
                cmd = "mpv --no-video --really-quiet --no-terminal - 2>/dev/null";
            }
            FILE* pp = popen(cmd.c_str(), "w");
            if (pp) { std::fwrite(wav.data(), 1, wav.size(), pp); ::pclose(pp); }
            draw(u, T("\x1b[36m播放完成（未创建文件）\x1b[0m", "\x1b[36mplayback done (no file created)\x1b[0m"));
            continue;
        }
        if (c == '\x1b') {
            char seq[2] = {0,0};
            if (read(STDIN_FILENO,&seq[0],1)!=1) break;
            if (read(STDIN_FILENO,&seq[1],1)!=1) break;
            if (seq[0]=='[') {
                if (seq[1]=='A') u.sel = (u.sel - 1 + u.total()) % u.total();
                else if (seq[1]=='B') u.sel = (u.sel + 1) % u.total();
                else if (seq[1]=='C' || seq[1]=='D') {
                    const int d = (seq[1]=='C') ? 1 : -1;
                    if (u.sel == 0) { u.vowel = (u.vowel + d + 6) % 6; loadVowelFormants(u); }
                    else if (u.sel == 1) u.cons = (u.cons + d + 7) % 7;
                    else {
                        NumParam& p = curNum(u, u.sel);
                        p.v = std::min(p.mx, std::max(p.mn, p.v + d * p.step));
                    }
                }
            }
            draw(u, T("\x1b[90m就绪\x1b[0m", "\x1b[90mready\x1b[0m"));
        }
    }
    rawOff();
    std::fputs("\x1b[2J\x1b[H", stdout);
    return 0;
}
