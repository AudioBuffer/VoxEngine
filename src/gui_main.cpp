// VoxEngine X11 原生 GUI（GTK3 + cairo）—— 不用 WebView
// 布局：顶部图表栏 | 三区（歌手/卷帘/轨道）| 底部控制栏
#include <gtk/gtk.h>
#include <cairo.h>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <cstdint>
#include <unistd.h>
#include <glib-unix.h>
#include <sys/socket.h>
#include <sys/un.h>
#include "vox_engine.h"
#include "seq.h"
#include "onnx_engine.h"

using namespace vox;

// ---------------- 播放器常驻（消除 mpv 每次冷启动的 ~1.8s）----------------
static const char* kMpvSock = "/tmp/vox-mpv.sock";
static bool g_mpvUp = false;

static bool mpvIpc(const std::string& json) {
    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return false;
    sockaddr_un a{}; a.sun_family = AF_UNIX;
    std::strncpy(a.sun_path, kMpvSock, sizeof(a.sun_path)-1);
    if (::connect(fd, (sockaddr*)&a, sizeof(a)) < 0) { ::close(fd); return false; }
    std::string msg = json + "\n";
    ssize_t r = ::write(fd, msg.data(), msg.size());
    ::close(fd);
    return r > 0;
}

// 启动常驻 mpv（进程活着、空闲等命令，不再退出）
static void mpvStart() {
    ::unlink(kMpvSock);
    g_spawn_command_line_async(
        "mpv --idle=yes --no-video --really-quiet --no-terminal "
        "--audio-display=no --no-resume-playback --input-ipc-server=/tmp/vox-mpv.sock",
        nullptr);
    g_mpvUp = true;
}

// 非阻塞预热：启动常驻进程后由主循环轮询，绝不卡住界面
static gboolean mpvWarmupTick(gpointer) {
    if (mpvIpc("{\"command\":[\"get_property\",\"idle-active\"]}")) { g_mpvUp = true; return G_SOURCE_REMOVE; }
    return G_SOURCE_CONTINUE;
}
static void mpvEnsureAsync() {
    if (g_mpvUp) return;
    // 已有常驻实例就直接复用 —— 否则每次启动 GUI 都会新起一个 mpv，
    // 旧的在异常退出时变成孤儿，越堆越多。
    if (mpvIpc("{\"command\":[\"get_property\",\"idle-active\"]}")) { g_mpvUp = true; return; }
    mpvStart();                                   // 起进程，不等
    g_timeout_add(200, mpvWarmupTick, nullptr);   // 后台轮询
}

static void mpvQuit() { if (g_mpvUp) mpvIpc("{\"command\":[\"quit\"]}"); }

static void mpvPlay(const std::string& path) {
    std::string cmd = "{\"command\":[\"loadfile\",\"" + path + "\",\"replace\"]}";
    if (mpvIpc(cmd)) return;                      // 常驻进程命中：~10ms
    mpvEnsureAsync();                             // 没命中：后台补起，不阻塞
    // 兜底：直接冷启一次（同样不阻塞主循环）
    std::string c2 = "mpv --no-video --really-quiet --no-terminal --audio-display=no \"" + path + "\"";
    g_spawn_command_line_async(c2.c_str(), nullptr);
}

// ---------------- 全局状态 ----------------
static std::vector<float> g_pcm;
static std::vector<float> g_f0;
static std::vector<float> g_spec;                 // 0..5kHz 归一幅度
static float  g_formants[5] = { 800, 1200, 2500, 3500, 4500 };
static double g_fs = 44100.0;
static double g_dur = 0.0;
static double g_alpha = 2.5;      // 声门开相增长率，自然值 2~5
static char   g_status[256] = "就绪";

// ---------------- 设置项状态 ----------------
static const char* kEngineVersion = "0.4.0-alpha";
static OnnxEngine  g_onnx;
static bool        g_onnxOn   = false;
static std::string g_onnxPath, g_jsPath, g_bankPath;
static bool        g_jsOn     = false;
static GtkWidget*  g_guiOnnxStatus = nullptr;
static GtkWidget *g_daFormant, *g_daSpec, *g_daPitch, *g_daRoll;

// ---------------- FFT ----------------
static void fft(std::vector<double>& re, std::vector<double>& im) {
    int n = (int)re.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * M_PI / len;
        double wr = std::cos(ang), wi = std::sin(ang);
        for (int i = 0; i < n; i += len) {
            double cwr = 1, cwi = 0;
            for (int k = 0; k < len / 2; k++) {
                int u = i + k, v = i + k + len / 2;
                double xr = re[v]*cwr - im[v]*cwi;
                double xi = re[v]*cwi + im[v]*cwr;
                re[v] = re[u] - xr; im[v] = im[u] - xi;
                re[u] += xr;        im[u] += xi;
                double nwr = cwr*wr - cwi*wi;
                cwi = cwr*wi + cwi*wr; cwr = nwr;
            }
        }
    }
}

static void computeSpectrum() {
    g_spec.clear();
    const int N = 2048;
    if ((int)g_pcm.size() < N) return;
    std::vector<double> re(N,0), im(N,0);
    int off = std::min((int)g_pcm.size() - N, (int)g_pcm.size()/4);
    for (int i = 0; i < N; i++) {
        double w = 0.5 - 0.5*std::cos(2*M_PI*i/(N-1));
        re[i] = g_pcm[off+i] * w;
    }
    fft(re, im);
    int bins = (int)(5000.0 * N / g_fs);
    g_spec.resize(std::max(2,bins));
    double mx = -200;
    for (int i = 0; i < bins; i++) {
        double m = 10*std::log10(re[i]*re[i] + im[i]*im[i] + 1e-12);
        g_spec[i] = (float)m;
        if (m > mx) mx = m;
    }
    for (auto& v : g_spec) v = (float)std::max(0.0, (v - (mx-60.0))/60.0);
}

static void computePitch() {
    g_f0.clear();
    const int FRAME = (int)(g_fs*0.030), HOP = (int)(g_fs*0.015);
    if ((int)g_pcm.size() < FRAME*2) return;
    const int minLag = (int)(g_fs/500), maxLag = (int)(g_fs/60);
    int nf = ((int)g_pcm.size() - FRAME)/HOP + 1;
    g_f0.resize(nf, 0.0f);
    for (int f = 0; f < nf; f++) {
        int off = f*HOP;
        double e = 0;
        for (int i = 0; i < FRAME; i++) e += (double)g_pcm[off+i]*g_pcm[off+i];
        if (std::sqrt(e/FRAME) < 0.01) { g_f0[f] = 0; continue; }
        double best = -1e18; int bestLag = 0;
        for (int lag = minLag; lag <= maxLag; lag += 4) {
            double r = 0;
            for (int i = 0; i + lag < FRAME; i++) r += (double)g_pcm[off+i]*g_pcm[off+i+lag];
            r /= (FRAME-lag);
            if (r > best) { best = r; bestLag = lag; }
        }
        for (int lag = std::max(minLag,bestLag-4); lag <= std::min(maxLag,bestLag+4); lag++) {
            double r = 0;
            for (int i = 0; i + lag < FRAME; i++) r += (double)g_pcm[off+i]*g_pcm[off+i+lag];
            r /= (FRAME-lag);
            if (r > best) { best = r; bestLag = lag; }
        }
        // 次谐波纠正：自相关常把 N 个周期当成 1 个（220Hz 测成 110/73Hz）。
        // 若 lag/2、lag/3 处有强度相近的峰，那个才是真实基频。
        for (int div = 2; div <= 3; div++) {
            int cand = bestLag / div;
            if (cand < minLag) break;
            double bc = -1e18; int bcl = 0;
            for (int lag = std::max(minLag,cand-3); lag <= std::min(maxLag,cand+3); lag++) {
                double r = 0;
                for (int i = 0; i + lag < FRAME; i++) r += (double)g_pcm[off+i]*g_pcm[off+i+lag];
                r /= (FRAME-lag);
                if (r > bc) { bc = r; bcl = lag; }
            }
            if (bc >= 0.85 * best) { best = bc; bestLag = bcl; break; }
        }
        g_f0[f] = (bestLag > 0) ? (float)(g_fs/bestLag) : 0.0f;
    }
}

// 音高后处理：中值滤波去倍频/半频误判 + 短空档插值
static void smoothPitch(std::vector<float>& f0) {
    if (f0.size() < 3) return;
    // 1) 5 点中值滤波（只在有声邻域内取中值，避免把静音拉成有声）
    std::vector<float> t = f0;
    for (size_t i = 0; i < f0.size(); i++) {
        std::vector<float> w;
        for (int k = -2; k <= 2; k++) {
            long j = (long)i + k;
            if (j >= 0 && j < (long)t.size() && t[j] > 1.0f) w.push_back(t[j]);
        }
        if (w.empty()) { f0[i] = 0.0f; continue; }
        std::sort(w.begin(), w.end());
        f0[i] = w[w.size() / 2];
    }
    // 2) 短空档（<=3 帧）线性插值补上，长空档保持断开
    size_t i = 0;
    while (i < f0.size()) {
        if (f0[i] > 1.0f) { i++; continue; }
        size_t a = i;
        while (i < f0.size() && f0[i] <= 1.0f) i++;
        size_t b = i;                       // [a, b) 是空档
        if (a == 0 || b >= f0.size()) continue;
        size_t len = b - a;
        if (len <= 3) {
            float v0 = f0[a-1], v1 = f0[b];
            for (size_t k = 0; k < len; k++)
                f0[a+k] = v0 + (v1-v0) * (float)(k+1) / (float)(len+1);
        }
    }
}

// ---------------- 绘图辅助 ----------------
static void bg(cairo_t* cr, int w, int h) {
    cairo_set_source_rgb(cr, 0.98, 0.98, 0.99);
    cairo_paint(cr);
    cairo_set_source_rgb(cr, 0.80, 0.80, 0.82);
    cairo_set_line_width(cr, 1.0);
    cairo_rectangle(cr, 0.5, 0.5, w-1, h-1);
    cairo_stroke(cr);
    cairo_set_source_rgb(cr, 0.90, 0.90, 0.92);
    for (int i = 1; i <= 4; i++) {
        double x = w * i / 5.0;
        cairo_move_to(cr, x, 2); cairo_line_to(cr, x, h-2); cairo_stroke(cr);
    }
}
static void label(cairo_t* cr, const char* s, double x, double y) {
    cairo_set_source_rgb(cr, 0.35, 0.35, 0.38);
    cairo_set_font_size(cr, 10);
    cairo_move_to(cr, x, y);
    cairo_show_text(cr, s);
}

// ---------------- 三个图表 ----------------
static gboolean onDrawFormant(GtkWidget* w, cairo_t* cr, gpointer) {
    int W = gtk_widget_get_allocated_width(w), H = gtk_widget_get_allocated_height(w);
    bg(cr, W, H);
    const float bw[5] = {60,90,120,150,200};
    std::vector<double> v(W, 0.0);
    double mx = 1e-6;
    for (int i = 0; i < W; i++) {
        double hz = 5000.0 * i / (W-1), s = 0;
        for (int k = 0; k < 5; k++) {
            double d = (hz - g_formants[k]) / std::max(30.0, bw[k]*0.5);
            s += 1.0/(1.0 + d*d);
        }
        v[i] = s; if (s > mx) mx = s;
    }
    cairo_set_source_rgb(cr, 0.25, 0.32, 0.71);
    cairo_set_line_width(cr, 2.0);
    for (int i = 0; i < W; i++) {
        double y = (H-6) - (v[i]/mx)*(H-14);
        if (i == 0) cairo_move_to(cr, i, y); else cairo_line_to(cr, i, y);
    }
    cairo_stroke(cr);
    for (int k = 0; k < 5; k++) {
        double x = (W-1) * std::min(1.0f, g_formants[k]/5000.0f);
        cairo_set_source_rgb(cr, 0.6, 0.6, 0.65);
        cairo_set_line_width(cr, 1.0);
        cairo_move_to(cr, x, 2); cairo_line_to(cr, x, H-14); cairo_stroke(cr);
        char b[8]; std::snprintf(b, sizeof b, "F%d", k+1);
        label(cr, b, x+2, 11);
    }
    label(cr, "共振峰", 4, H-4);
    return TRUE;
}

static gboolean onDrawSpec(GtkWidget* w, cairo_t* cr, gpointer) {
    int W = gtk_widget_get_allocated_width(w), H = gtk_widget_get_allocated_height(w);
    bg(cr, W, H);
    if (g_spec.size() > 1) {
        cairo_set_source_rgb(cr, 0.0, 0.54, 0.48);
        cairo_set_line_width(cr, 1.6);
        int n = (int)g_spec.size();
        for (int i = 0; i < n; i++) {
            double x = 4 + (W-8) * i / (double)(n-1);
            double y = (H-6) - g_spec[i]*(H-14);
            if (i == 0) cairo_move_to(cr, x, y); else cairo_line_to(cr, x, y);
        }
        cairo_stroke(cr);
    } else label(cr, "渲染后显示", 6, H/2.0);
    label(cr, "频谱", 4, H-4);
    return TRUE;
}

static gboolean onDrawPitch(GtkWidget* w, cairo_t* cr, gpointer) {
    int W = gtk_widget_get_allocated_width(w), H = gtk_widget_get_allocated_height(w);
    bg(cr, W, H);
    double mn = 1e9, mx = -1e9;
    for (float v : g_f0) if (v > 1) { mn = std::min(mn,(double)v); mx = std::max(mx,(double)v); }
    if (mn <= mx) {
        double mean = (mn+mx)/2, span = std::max(6.0, mean*0.30);
        mn = mean - span/2;
        cairo_set_source_rgb(cr, 0.90, 0.32, 0.0);
        cairo_set_line_width(cr, 1.8);
        int n = (int)g_f0.size();
        bool started = false;
        for (int i = 0; i < n; i++) {
            double x = 4 + (W-8) * i / (double)(n-1);
            if (g_f0[i] <= 1) { started = false; continue; }
            double t = (g_f0[i]-mn)/span;
            double y = (H-6) - t*(H-14);
            if (!started) { cairo_move_to(cr, x, y); started = true; } else cairo_line_to(cr, x, y);
        }
        cairo_stroke(cr);
        char b[32]; std::snprintf(b, sizeof b, "%.0f-%.0fHz", mn, mn+span);
        label(cr, b, 4, 11);
    } else label(cr, "渲染后显示", 6, H/2.0);
    label(cr, "音高", 4, H-4);
    return TRUE;
}

// ---------------- 主编辑区（音符网格，暂为静态网格）----------------
// ---------------- 可编辑卷帘（OpenUtau 式）----------------
struct RollNote {
    double start{0.0}, dur{0.4}, f0{220.0};
    int vowel{0}, cons{0};
    std::string ph{"a"};
};
static std::vector<RollNote> g_notes;          // 当前轨道的音符

// 每轨独立的声音参数
struct TrackParams {
    int    voicebank{0};     // 声库索引
    double f0Shift{0.0};     // 基频细调（半音）
    double alpha{2.5};       // 声门张力
    double breath{0.0};      // 气息
    double gain{1.0};        // 音量
};

// ---- 多轨道 ----
struct Track { std::string name; std::vector<RollNote> notes; TrackParams p; };
static std::vector<Track> g_tracks;
static int  g_curTrack = 0;
static void trackSave() {
    if (g_curTrack >= 0 && g_curTrack < (int)g_tracks.size()) g_tracks[g_curTrack].notes = g_notes;
}
static void trackLoad(int i) {
    trackSave();
    if (i < 0 || i >= (int)g_tracks.size()) return;
    g_curTrack = i;
    g_notes = g_tracks[(size_t)i].notes;
}

// ---- 声库（内置男声 / 女声）----
struct VoicebankDef { const char* name; const char* desc; double scale; };
static const VoicebankDef kVoices[] = {
    { "男声（内置）", "共振峰 ×0.95　低沉", 0.95 },
    { "女声（内置）", "共振峰 ×1.14　明亮", 1.14 },
    { "童声（内置）", "共振峰 ×1.30　清亮", 1.30 },
};
static const int kVoiceCount = (int)(sizeof(kVoices) / sizeof(kVoices[0]));
static const char* voicebankName(int v)  { return kVoices[std::clamp(v,0,kVoiceCount-1)].name; }
static double      voicebankScale(int v) { return kVoices[std::clamp(v,0,kVoiceCount-1)].scale; }

static int g_voicebank = 0;  // 兼容旧引用：始终等于当前轨的声库

// ---- 中键框选 ----
static std::vector<int> g_sel2;                 // 多选集合
static bool   g_boxOn = false;
static double g_boxX0 = 0, g_boxY0 = 0, g_boxX1 = 0, g_boxY1 = 0;
static double g_mouseT = 0, g_mouseF = 220;     // 鼠标在卷帘里的位置（粘贴锚点）

// ---- 剪贴板 ----
static std::vector<RollNote> g_clip;
static GtkWidget* g_trackBox = nullptr;
static std::string g_lastText;
static int g_sel = -1, g_dragMode = 0, g_dragNote = -1;
static double g_dragT0 = 0, g_dragY0 = 0;
static double g_origStart = 0, g_origDur = 0.4, g_origF0 = 220;

static const double kRollFmin = 80.0, kRollFmax = 800.0, kKeyW = 46.0;
static double g_pxPerSec = 130.0;

static double rollT2X(double t) { return kKeyW + t * g_pxPerSec; }
static double rollX2T(double x) { return (x - kKeyW) / g_pxPerSec; }
static double rollF2Y(double f, int H) {
    const double lo = std::log2(kRollFmin), hi = std::log2(kRollFmax);
    f = std::clamp(f, kRollFmin, kRollFmax);
    return 6.0 + (H - 12.0) * (hi - std::log2(f)) / (hi - lo);
}
static double rollY2F(double y, int H) {
    const double lo = std::log2(kRollFmin), hi = std::log2(kRollFmax);
    double t = std::clamp((y - 6.0) / (H - 12.0), 0.0, 1.0);
    return std::pow(2.0, hi - t * (hi - lo));
}
static double rollSnapF(double f) {
    return 440.0 * std::pow(2.0, std::round(12.0 * std::log2(f / 440.0)) / 12.0);
}
static double rollSnapT(double t) { return std::max(0.0, std::round(t / 0.05) * 0.05); }

static int rollHit(double x, double y, int H) {
    for (int i = (int)g_notes.size() - 1; i >= 0; --i) {
        double x0 = rollT2X(g_notes[i].start), x1 = rollT2X(g_notes[i].start + g_notes[i].dur);
        double y0 = rollF2Y(g_notes[i].f0, H) - 9, y1 = y0 + 18;
        if (x >= x0 && x <= x1 && y >= y0 && y <= y1) return i;
    }
    return -1;
}

static const char* kNoteName[12] = {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};
static void rollKeyName(double f, char* out, size_t n) {
    int midi = (int)std::lround(69.0 + 12.0 * std::log2(f / 440.0));
    std::snprintf(out, n, "%s%d", kNoteName[((midi % 12) + 12) % 12], midi / 12 - 1);
}

static gboolean onDrawRoll(GtkWidget* w, cairo_t* cr, gpointer) {
    int W = gtk_widget_get_allocated_width(w), H = gtk_widget_get_allocated_height(w);
    cairo_set_source_rgb(cr, 0.13, 0.14, 0.18);
    cairo_paint(cr);

    const int m0 = (int)std::ceil(12.0 * std::log2(kRollFmin / 440.0) + 69.0);
    const int m1 = (int)std::floor(12.0 * std::log2(kRollFmax / 440.0) + 69.0);
    const double rowH = (H - 12.0) / (double)(m1 - m0 + 1);
    for (int m = m0; m <= m1; ++m) {
        const double f = 440.0 * std::pow(2.0, (m - 69) / 12.0);
        const double y = rollF2Y(f, H);
        const int pc = ((m % 12) + 12) % 12;
        const bool black = (pc == 1 || pc == 3 || pc == 6 || pc == 8 || pc == 10);
        if (black) {
            cairo_set_source_rgb(cr, 0.16, 0.17, 0.21);
            cairo_rectangle(cr, kKeyW, y - rowH / 2, W - kKeyW, rowH);
            cairo_fill(cr);
        }
        cairo_set_source_rgb(cr, 0.22, 0.23, 0.28);
        cairo_set_line_width(cr, 1.0);
        cairo_move_to(cr, kKeyW, y); cairo_line_to(cr, W, y); cairo_stroke(cr);
        cairo_set_source_rgb(cr, black ? 0.13 : 0.80, black ? 0.13 : 0.80, black ? 0.15 : 0.78);
        cairo_rectangle(cr, 0, y - rowH / 2 + 0.5, kKeyW - 2, rowH - 1);
        cairo_fill(cr);
        char nm[8]; rollKeyName(f, nm, sizeof nm);
        cairo_set_source_rgb(cr, black ? 0.65 : 0.15, black ? 0.65 : 0.15, black ? 0.65 : 0.15);
        cairo_set_font_size(cr, 8);
        cairo_move_to(cr, 3, y + 3);
        cairo_show_text(cr, nm);
    }
    // 时间网格
    for (double t = 0; rollT2X(t) < W; t += 0.25) {
        const double x = rollT2X(t);
        if (x < kKeyW) continue;
        cairo_set_source_rgb(cr, 0.22, 0.23, 0.28);
        cairo_set_line_width(cr, (std::fabs(t - std::round(t)) < 1e-6) ? 1.4 : 0.7);
        cairo_move_to(cr, x, 0); cairo_line_to(cr, x, H); cairo_stroke(cr);
    }
    // 多轨全览（借 OpenUtau：所有轨音符同一卷帘叠加，非当前轨压暗）
    if (g_tracks.size() > 1) {
        for (size_t t = 0; t < g_tracks.size(); ++t) {
            if ((int)t == g_curTrack) continue;
            for (const auto& n : g_tracks[t].notes) {
                const double x0 = rollT2X(n.start), x1 = rollT2X(n.start + n.dur);
                if (x1 < kKeyW || x0 > W) continue;
                const double y = rollF2Y(n.f0, H);
                cairo_set_source_rgba(cr, 0.62, 0.66, 0.78, 0.38);
                cairo_rectangle(cr, x0, y - 3, std::max(2.0, x1 - x0), 6);
                cairo_fill(cr);
            }
        }
    }
    // 音符
    for (size_t i = 0; i < g_notes.size(); ++i) {
        const RollNote& n = g_notes[i];
        const double x0 = rollT2X(n.start), x1 = rollT2X(n.start + n.dur);
        if (x1 < kKeyW || x0 > W) continue;
        const double y = rollF2Y(n.f0, H);
        const double wpx = std::max(3.0, x1 - x0);
        const bool sel = ((int)i == g_sel) ||
                         std::find(g_sel2.begin(), g_sel2.end(), (int)i) != g_sel2.end();
        cairo_set_source_rgb(cr, sel ? 0.96 : 0.29, sel ? 0.64 : 0.53, sel ? 0.16 : 0.91);
        cairo_rectangle(cr, x0, y - 9, wpx, 18);
        cairo_fill(cr);
        cairo_set_source_rgb(cr, 0.08, 0.09, 0.12);
        cairo_set_line_width(cr, 1.0);
        cairo_rectangle(cr, x0, y - 9, wpx, 18);
        cairo_stroke(cr);
        cairo_set_source_rgb(cr, 1, 1, 1);
        cairo_rectangle(cr, std::max(x0, x1 - 3.0), y - 9, std::min(3.0, wpx), 18);
        cairo_fill(cr);
        if (wpx > 26) {
            char nb[8]; rollKeyName(n.f0, nb, sizeof nb);
            cairo_set_source_rgb(cr, 0.05, 0.06, 0.09);
            cairo_set_font_size(cr, 9);
            cairo_move_to(cr, x0 + 4, y + 3);
            cairo_show_text(cr, n.ph.c_str());
            cairo_move_to(cr, x0 + 6 + 11.0 * n.ph.size(), y + 3);
            cairo_show_text(cr, nb);
        }
    }
    // 中键框选矩形
    if (g_boxOn) {
        cairo_set_source_rgba(cr, 0.95, 0.75, 0.25, 0.22);
        cairo_rectangle(cr, std::min(g_boxX0,g_boxX1), std::min(g_boxY0,g_boxY1),
                        std::fabs(g_boxX1-g_boxX0), std::fabs(g_boxY1-g_boxY0));
        cairo_fill_preserve(cr);
        cairo_set_source_rgb(cr, 0.98, 0.80, 0.30);
        cairo_set_line_width(cr, 1.2);
        cairo_stroke(cr);
    }
    cairo_set_source_rgb(cr, 0.55, 0.58, 0.65);
    cairo_set_font_size(cr, 10);
    char hint[160];
    std::snprintf(hint, sizeof hint,
        "左键加/拖音符 · 右键删 · 中键框选 · Ctrl+C 复制 / Ctrl+V 粘到鼠标处   [%s] %zu 音符 %zu 选中",
        g_tracks.empty() ? "轨道" : g_tracks[(size_t)g_curTrack].name.c_str(),
        g_notes.size(), g_sel2.size());
    cairo_move_to(cr, kKeyW + 4, H - 4);
    cairo_show_text(cr, hint);
    return TRUE;
}

// 默认旋律：有音高变化、有时值变化、音符之间留空隙（不再是同音高首尾硬接）
static void seedDemo() {
    struct { double t, d, f; const char* ph; } m[] = {
        {0.00, 0.45, 261.63, "a"}, {0.50, 0.45, 293.66, "a"},
        {1.00, 0.45, 329.63, "a"}, {1.50, 0.90, 392.00, "a"},
        {2.50, 0.45, 440.00, "a"}, {3.00, 0.45, 392.00, "a"},
        {3.50, 0.45, 329.63, "a"}, {4.00, 1.00, 261.63, "a"},
    };
    for (const auto& e : m) {
        RollNote n; n.start = e.t; n.dur = e.d; n.f0 = e.f;
        n.ph = e.ph;
        lookupPhoneme(e.ph, n.vowel, n.cons);
        g_notes.push_back(n);
    }
}

// 复制：以最早音符为基准存相对时间
static void doCopy() {
    std::vector<int> sel = g_sel2;
    if (sel.empty() && g_sel >= 0) sel.push_back(g_sel);
    if (sel.empty()) { std::snprintf(g_status, sizeof g_status, "未选中音符（中键框选）"); return; }
    double t0 = 1e18;
    for (int i : sel) if (i >= 0 && i < (int)g_notes.size()) t0 = std::min(t0, g_notes[i].start);
    g_clip.clear();
    for (int i : sel) {
        if (i < 0 || i >= (int)g_notes.size()) continue;
        RollNote n = g_notes[(size_t)i];
        n.start -= t0;
        g_clip.push_back(n);
    }
    std::snprintf(g_status, sizeof g_status, "已复制 %zu 个音符", g_clip.size());
    if (g_daRoll) gtk_widget_queue_draw(g_daRoll);
}

// 粘贴：锚在鼠标所在时间，写入「当前轨道」
static void doPaste() {
    if (g_clip.empty()) { std::snprintf(g_status, sizeof g_status, "剪贴板为空"); return; }
    const double anchor = std::max(0.0, g_mouseT);
    g_sel2.clear();
    for (RollNote n : g_clip) {
        n.start = rollSnapT(anchor + n.start);
        g_notes.push_back(n);
        g_sel2.push_back((int)g_notes.size() - 1);
    }
    trackSave();
    std::snprintf(g_status, sizeof g_status, "已粘贴 %zu 个音符 → %s",
                  g_clip.size(), g_tracks.empty() ? "轨道" : g_tracks[(size_t)g_curTrack].name.c_str());
    if (g_daRoll) gtk_widget_queue_draw(g_daRoll);
}

static gboolean onKey(GtkWidget*, GdkEventKey* e, gpointer) {
    if (!(e->state & GDK_CONTROL_MASK)) return FALSE;
    if (e->keyval == GDK_KEY_c || e->keyval == GDK_KEY_C) { doCopy();  return TRUE; }
    if (e->keyval == GDK_KEY_v || e->keyval == GDK_KEY_V) { doPaste(); return TRUE; }
    return FALSE;
}

// ---------------- 轨道侧栏 ----------------
static void rebuildTracks();
static void syncParamWidgets();

static void onTrackClick(GtkWidget*, gpointer data) {
    trackLoad((int)(intptr_t)data);
    g_sel = -1; g_sel2.clear();
    syncParamWidgets();
    rebuildTracks();
    if (g_daRoll) gtk_widget_queue_draw(g_daRoll);
    std::snprintf(g_status, sizeof g_status, "切换到 %s", g_tracks[(size_t)g_curTrack].name.c_str());
}

static void onNewTrack(GtkWidget*, gpointer) {
    trackSave();
    char nm[32];
    std::snprintf(nm, sizeof nm, "轨道%zu", g_tracks.size() + 1);
    g_tracks.push_back(Track{ nm, {} });
    g_curTrack = (int)g_tracks.size() - 1;
    g_notes.clear();
    g_sel = -1; g_sel2.clear();
    syncParamWidgets();
    rebuildTracks();
    if (g_daRoll) gtk_widget_queue_draw(g_daRoll);
    std::snprintf(g_status, sizeof g_status, "已新建 %s", nm);
}

static GtkWidget *g_vbRevealer=nullptr,*g_vbButton=nullptr,*g_overlay=nullptr,*g_vbList=nullptr;
static GtkWidget *g_scF0=nullptr,*g_scAlpha=nullptr,*g_scBreath=nullptr;
static bool g_syncing=false;
static TrackParams& curP() {
    static TrackParams dummy;
    if (g_tracks.empty() || g_curTrack < 0 || g_curTrack >= (int)g_tracks.size()) return dummy;
    return g_tracks[(size_t)g_curTrack].p;
}
static void onVbPick(GtkWidget*, gpointer);
static void fillVoicebankList();

static void syncParamWidgets() {
    if (g_tracks.empty()) return;
    const TrackParams& P = curP();
    g_syncing = true;
    if (g_scF0)    gtk_range_set_value(GTK_RANGE(g_scF0),    P.f0Shift);
    if (g_scAlpha) gtk_range_set_value(GTK_RANGE(g_scAlpha), P.alpha);
    if (g_scBreath)gtk_range_set_value(GTK_RANGE(g_scBreath),P.breath);
    g_syncing = false;
    if (g_vbButton) {
        char lbl[128];
        std::snprintf(lbl, sizeof lbl, "声库：%s\n（点击选择）", kVoices[P.voicebank].name);
        gtk_button_set_label(GTK_BUTTON(g_vbButton), lbl);
    }
}
static void onF0Changed(GtkRange* r, gpointer) {
    if (g_syncing || g_tracks.empty()) return;
    curP().f0Shift = gtk_range_get_value(r);
    std::snprintf(g_status, sizeof g_status, "本轨基频细调 %+.1f 半音", curP().f0Shift);
}
static void onAlphaChanged(GtkRange* r, gpointer) {
    if (g_syncing || g_tracks.empty()) return;
    curP().alpha = gtk_range_get_value(r);
    std::snprintf(g_status, sizeof g_status, "本轨张力 %.1f", curP().alpha);
}
static void onBreathChanged(GtkRange* r, gpointer) {
    if (g_syncing || g_tracks.empty()) return;
    curP().breath = gtk_range_get_value(r);
    std::snprintf(g_status, sizeof g_status, "本轨气息 %.2f", curP().breath);
}

static void fillVoicebankList() {
    if (!g_vbList) return;
    gtk_container_foreach(GTK_CONTAINER(g_vbList), (GtkCallback)gtk_widget_destroy, nullptr);
    const int cw = g_tracks.empty() ? 0 : curP().voicebank;
    for (int i = 0; i < kVoiceCount; ++i) {
        char lbl[200];
        std::snprintf(lbl, sizeof lbl, "%s%s\n%s", kVoices[i].name,
                      (i == cw) ? "　✔当前" : "", kVoices[i].desc);
        GtkWidget* b = gtk_button_new_with_label(lbl);
        if (i == cw) gtk_style_context_add_class(gtk_widget_get_style_context(b), "vox-vb-sel");
        g_signal_connect_data(b, "clicked", G_CALLBACK(onVbPick),
                              (gpointer)(intptr_t)i, nullptr, (GConnectFlags)0);
        gtk_box_pack_start(GTK_BOX(g_vbList), b, FALSE, FALSE, 0);
        gtk_widget_show(b);
    }
}
static void vbApply(int idx) {
    if (idx < 0 || idx >= kVoiceCount) return;
    if (!g_tracks.empty()) curP().voicebank = idx;
    g_voicebank = idx;
    std::snprintf(g_status, sizeof g_status, "本轨声库已切换为 %s", kVoices[idx].name);
    syncParamWidgets();
    fillVoicebankList();
    if (g_vbRevealer) gtk_revealer_set_reveal_child(GTK_REVEALER(g_vbRevealer), FALSE);
    if (g_daRoll) gtk_widget_queue_draw(g_daRoll);
}
static void onVbPick(GtkWidget*, gpointer data) { vbApply((int)(intptr_t)data); }
static void onVoicebank(GtkWidget* btn, gpointer) {
    g_vbButton = btn;
    if (!g_vbRevealer) return;
    const bool on = gtk_revealer_get_reveal_child(GTK_REVEALER(g_vbRevealer));
    if (!on) fillVoicebankList();
    gtk_revealer_set_reveal_child(GTK_REVEALER(g_vbRevealer), !on);
}

// ---------------- 应用级样式（弹出面板需要背景，否则按钮会孤零零浮在卷帘上）----------------
static void ensureCss() {
    GtkCssProvider* prov = gtk_css_provider_new();
    gtk_css_provider_load_from_data(prov,
        ".vox-panel {"
        "  background-image: none;"
        "  background-color: #e9eff9;"      /* 浅底，配黑字 */
        "  border: 2px solid #3a6fc4;"
        "  border-radius: 8px;"
        "  padding: 10px;"
        "}"
        ".vox-panel-title { color: #000000; font-weight: bold; font-size: 108%; }"
        ".vox-panel-hint  { color: #2b2b2b; font-size: 90%; }"
        ".vox-panel label { color: #000000; }"
        ".vox-panel button { color: #000000; padding: 5px 8px; }"
        ".vox-panel button label { color: #000000; }"
        ".vox-vb-sel { background-image: none; background-color: #8fc0f0; }"   /* 选中项：蓝 */
        , -1, nullptr);
    gtk_style_context_add_provider_for_screen(gdk_screen_get_default(),
        GTK_STYLE_PROVIDER(prov), GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
    g_object_unref(prov);
}

static void buildVoicebankPanel() {
    g_vbRevealer = gtk_revealer_new();
    gtk_revealer_set_transition_type(GTK_REVEALER(g_vbRevealer), GTK_REVEALER_TRANSITION_TYPE_SLIDE_RIGHT);
    gtk_revealer_set_transition_duration(GTK_REVEALER(g_vbRevealer), 180);
    gtk_widget_set_halign(g_vbRevealer, GTK_ALIGN_START);
    gtk_widget_set_valign(g_vbRevealer, GTK_ALIGN_START);
    gtk_widget_set_margin_start(g_vbRevealer, 6);
    gtk_widget_set_margin_top(g_vbRevealer, 104);

    // GtkBox 是无窗口控件，CSS 背景不会绘制；必须用带自身窗口的 GtkEventBox 承载
    GtkWidget* shell = gtk_event_box_new();
    gtk_event_box_set_visible_window(GTK_EVENT_BOX(shell), TRUE);
    gtk_style_context_add_class(gtk_widget_get_style_context(shell), "vox-panel");

    GtkWidget* panel = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    gtk_widget_set_size_request(panel, 220, -1);

    GtkWidget* title = gtk_label_new("选择声库");
    gtk_style_context_add_class(gtk_widget_get_style_context(title), "vox-panel-title");
    gtk_widget_set_halign(title, GTK_ALIGN_START);
    gtk_box_pack_start(GTK_BOX(panel), title, FALSE, FALSE, 0);

    g_vbList = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);
    gtk_box_pack_start(GTK_BOX(panel), g_vbList, FALSE, FALSE, 0);

    GtkWidget* hint = gtk_label_new("声库控制共振峰整体缩放，作用于当前轨道");
    gtk_style_context_add_class(gtk_widget_get_style_context(hint), "vox-panel-hint");
    gtk_label_set_line_wrap(GTK_LABEL(hint), TRUE);
    gtk_widget_set_halign(hint, GTK_ALIGN_START);
    gtk_box_pack_start(GTK_BOX(panel), hint, FALSE, FALSE, 0);

    gtk_container_add(GTK_CONTAINER(shell), panel);
    gtk_container_add(GTK_CONTAINER(g_vbRevealer), shell);
    gtk_overlay_add_overlay(GTK_OVERLAY(g_overlay), g_vbRevealer);
    fillVoicebankList();
    std::fprintf(stderr, "[声库面板] overlay=%p revealer=%p list=%p\n",
                 (void*)g_overlay, (void*)g_vbRevealer, (void*)g_vbList);
}

// 声库按钮下方的详细调节（作用于当前轨道）
static GtkWidget* makeParamRow(const char* title, GtkWidget* scale, GtkWidget* valLabel) {
    GtkWidget* b = gtk_box_new(GTK_ORIENTATION_VERTICAL, 1);
    gtk_box_pack_start(GTK_BOX(b), gtk_label_new(title), FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(b), scale, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(b), valLabel, FALSE, FALSE, 0);
    return b;
}

static void rebuildTracks() {
    if (!g_trackBox) return;
    gtk_container_foreach(GTK_CONTAINER(g_trackBox), (GtkCallback)gtk_widget_destroy, nullptr);
    for (size_t i = 0; i < g_tracks.size(); ++i) {
        char lbl[96];
        std::snprintf(lbl, sizeof lbl, "%zu. %s%s", i + 1, g_tracks[i].name.c_str(),
                      ((int)i == g_curTrack) ? "  ← 当前" : "");
        GtkWidget* b = gtk_button_new_with_label(lbl);
        g_signal_connect_data(b, "clicked", G_CALLBACK(onTrackClick),
                              (gpointer)(intptr_t)i, nullptr, (GConnectFlags)0);
        gtk_box_pack_start(GTK_BOX(g_trackBox), b, FALSE, FALSE, 0);
        gtk_widget_show(b);
    }
}

static gboolean onRollPress(GtkWidget* w, GdkEventButton* e, gpointer) {
    int H = gtk_widget_get_allocated_height(w);
    if (e->x < kKeyW) return TRUE;
    int hit = rollHit(e->x, e->y, H);
    if (e->button == 2) {                    // 中键：开始框选
        g_boxOn = true;
        g_boxX0 = g_boxX1 = e->x; g_boxY0 = g_boxY1 = e->y;
        g_sel2.clear();
        gtk_widget_queue_draw(w);
        return TRUE;
    }
    if (e->button == 3) {
        if (hit >= 0) {
            g_notes.erase(g_notes.begin() + hit);
            g_sel = -1;
            gtk_widget_queue_draw(w);
        }
        return TRUE;
    }
    if (e->button != 1) return TRUE;
    const double t = rollX2T(e->x);
    if (hit >= 0) {
        g_dragNote = hit; g_sel = hit;
        g_origStart = g_notes[hit].start;
        g_origDur   = g_notes[hit].dur;
        g_origF0    = g_notes[hit].f0;
        g_dragT0 = t; g_dragY0 = e->y;
        const double xr = rollT2X(g_origStart + g_origDur);
        g_dragMode = (std::fabs(e->x - xr) < 9.0) ? 2 : 1;
    } else {
        RollNote n;
        n.start = rollSnapT(t);
        n.dur = 0.4;
        n.f0 = rollSnapF(rollY2F(e->y, H));
        const char* def = "a";
        lookupPhoneme(def, n.vowel, n.cons);
        n.ph = def;
        g_notes.push_back(n);
        g_sel = (int)g_notes.size() - 1;
        g_dragNote = g_sel; g_dragMode = 1;
        g_origStart = n.start; g_origDur = n.dur; g_origF0 = n.f0;
        g_dragT0 = t; g_dragY0 = e->y;
    }
    gtk_widget_queue_draw(w);
    return TRUE;
}

static gboolean onRollMotion(GtkWidget* w, GdkEventMotion* e, gpointer) {
    int H = gtk_widget_get_allocated_height(w);
    g_mouseT = rollX2T(e->x);                       // 粘贴锚点
    g_mouseF = rollY2F(e->y, H);
    if (g_boxOn) {                                  // 中键框选：实时更新
        g_boxX1 = e->x; g_boxY1 = e->y;
        gtk_widget_queue_draw(w);
        return TRUE;
    }
    if (g_dragMode == 0 || g_dragNote < 0 || g_dragNote >= (int)g_notes.size()) return TRUE;
    const double t = rollX2T(e->x);
    RollNote& n = g_notes[g_dragNote];
    if (g_dragMode == 1) {
        n.start = rollSnapT(g_origStart + (t - g_dragT0));
        if (std::fabs(e->y - g_dragY0) > 4.0) n.f0 = rollSnapF(rollY2F(e->y, H));
    } else {
        n.dur = std::max(0.05, rollSnapT(g_origDur + (t - g_dragT0)));
    }
    gtk_widget_queue_draw(w);
    return TRUE;
}

static gboolean onRollRelease(GtkWidget* w, GdkEventButton* e, gpointer) {
    int H = gtk_widget_get_allocated_height(w);
    if (g_boxOn) {                                  // 收框：选中矩形内所有音符
        g_boxOn = false;
        const double x0 = std::min(g_boxX0,g_boxX1), x1 = std::max(g_boxX0,g_boxX1);
        const double y0 = std::min(g_boxY0,g_boxY1), y1 = std::max(g_boxY0,g_boxY1);
        g_sel2.clear();
        for (size_t i = 0; i < g_notes.size(); ++i) {
            const double nx0 = rollT2X(g_notes[i].start);
            const double nx1 = rollT2X(g_notes[i].start + g_notes[i].dur);
            const double ny  = rollF2Y(g_notes[i].f0, H);
            if (nx1 >= x0 && nx0 <= x1 && ny >= y0 - 9 && ny <= y1 + 9)
                g_sel2.push_back((int)i);
        }
        std::snprintf(g_status, sizeof g_status, "框中 %zu 个音符", g_sel2.size());
        gtk_widget_queue_draw(w);
        return TRUE;
    }
    (void)e;
    g_dragMode = 0; g_dragNote = -1;
    return TRUE;
}

// ---------------- 渲染 ----------------
// 按每个音符的绝对时间混音（不再是首尾硬接）
// 音素连续化（借 OpenUtau 的做法）：
//   整段交给同一个 VoxEngine 连续渲染 —— 声门相位、声道状态跨音符保持，
//   音高由引擎的 smoothstep 滑音衔接；只有「乐句首尾」才淡化，
//   段内音符之间不淡化，元音得以延续而不是每个音符重新起音。
static void renderRoll() {
    g_pcm.clear();
    if (g_notes.empty()) return;

    std::vector<RollNote> ns = g_notes;
    std::sort(ns.begin(), ns.end(),
              [](const RollNote& a, const RollNote& b) { return a.start < b.start; });

    // 连续段划分：间隔 < 20ms 视为连音（同一次发声）
    std::vector<std::pair<size_t,size_t>> runs;
    for (size_t i = 0; i < ns.size(); ) {
        size_t j = i;
        while (j + 1 < ns.size() &&
               ns[j+1].start <= ns[j].start + ns[j].dur + 0.02) ++j;
        runs.push_back({i, j});
        i = j + 1;
    }

    double end = 0.0;
    for (const auto& n : ns) end = std::max(end, n.start + n.dur);
    const double totalSec = end + 0.2;

    const TrackParams P = g_tracks.empty() ? TrackParams{} : curP();
    VoxEngine eng(g_fs);
    eng.setGlottalAlpha(P.alpha);                      // 本轨张力
    eng.setBreath(P.breath);                           // 本轨气息
    eng.setVoiceScale(voicebankScale(P.voicebank));    // 本轨声库：共振峰整体缩放
    eng.setOnnxEngine(g_onnxOn ? &g_onnx : nullptr);   // 关闭或无模型 -> 纯参数合成
    eng.clear();
    for (const auto& n : ns) {
        Note nt;
        nt.start = n.start; nt.dur = n.dur;
        nt.f0 = n.f0 * std::pow(2.0, P.f0Shift / 12.0);   // 本轨基频细调（半音）
        nt.vowel = n.vowel; nt.consonant = n.cons;
        nt.cdur = std::min(0.08, n.dur * 0.45);
        eng.addNote(nt);
    }
    std::vector<float> buf = eng.render(totalSec);

    // 只在连续段首尾淡化；段内不淡化 -> 连音不断开
    const size_t F = (size_t)std::llround(0.010 * g_fs);
    for (const auto& r : runs) {
        size_t a = (size_t)std::llround(ns[r.first].start * g_fs);
        size_t b = std::min(buf.size(),
                   (size_t)std::llround((ns[r.second].start + ns[r.second].dur) * g_fs));
        if (b <= a) continue;
        size_t f = std::min(F, (b - a) / 2);
        for (size_t k = 0; k < f; ++k) {
            float gg = (float)k / (float)f;
            buf[a + k] *= gg;
            buf[b - 1 - k] *= gg;
        }
    }
    double pk = 1e-12;
    for (float v : buf) pk = std::max(pk, (double)std::fabs(v));
    for (float& v : buf) v = (float)(v * 0.85 / pk);
    g_pcm = std::move(buf);
}


// ---------------- 设置面板 ----------------
static void settingsRefreshOnnx() {
    if (!g_guiOnnxStatus) return;
    char b[256];
    if (!g_onnxOn)             std::snprintf(b,sizeof b,"已关闭 —— 使用纯参数合成（DSP）");
    else if (g_onnx.loading()) std::snprintf(b,sizeof b,"模型加载中…（当前仍为纯参数合成）");
    else if (g_onnx.ready())   std::snprintf(b,sizeof b,"已启用 —— ONNX 推理生效");
    else                       std::snprintf(b,sizeof b,"无可用模型 —— 已放弃 ONNX，回落纯参数合成%s%s",
                                     g_onnx.lastError().empty()?"":"：", g_onnx.lastError().c_str());
    gtk_label_set_text(GTK_LABEL(g_guiOnnxStatus), b);
}
static void onOnnxToggled(GtkSwitch*, gboolean state, gpointer) {
    g_onnxOn = state;
    if (g_onnxOn && !g_onnxPath.empty()) g_onnx.loadAsync(g_onnxPath);
    else                                 g_onnx.unload();
    settingsRefreshOnnx();
}
static void onOnnxFile(GtkFileChooserButton* b, gpointer) {
    const char* f = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(b));
    g_onnxPath = f ? f : "";
    if (g_onnxOn) { if (g_onnxPath.empty()) g_onnx.unload(); else g_onnx.loadAsync(g_onnxPath); }
    settingsRefreshOnnx();
}
static void onJsToggled(GtkToggleButton* b, gpointer) { g_jsOn = gtk_toggle_button_get_active(b); }
static void onJsFile(GtkFileChooserButton* b, gpointer) {
    const char* f = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(b)); g_jsPath = f ? f : "";
}
static void onBankFile(GtkFileChooserButton* b, gpointer) {
    const char* f = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(b)); g_bankPath = f ? f : "";
}
static GtkWidget* fileRow(const char* title) {
    GtkWidget* fc = gtk_file_chooser_button_new(title, GTK_FILE_CHOOSER_ACTION_OPEN);
    gtk_widget_set_size_request(fc, 250, -1);
    return fc;
}
static void gridCfg(GtkWidget* g) {
    gtk_grid_set_row_spacing(GTK_GRID(g), 6);
    gtk_grid_set_column_spacing(GTK_GRID(g), 10);
    gtk_container_set_border_width(GTK_CONTAINER(g), 8);
}
static void onSettings(GtkWidget* w, gpointer) {
    GtkWidget* win = gtk_widget_get_toplevel(w);
    GtkWidget* dlg = gtk_dialog_new_with_buttons("设置",
        GTK_IS_WINDOW(win) ? GTK_WINDOW(win) : nullptr,
        GTK_DIALOG_MODAL, "关闭", GTK_RESPONSE_CLOSE, nullptr);
    gtk_window_set_default_size(GTK_WINDOW(dlg), 640, 420);
    GtkWidget* box = gtk_dialog_get_content_area(GTK_DIALOG(dlg));
    gtk_container_set_border_width(GTK_CONTAINER(box), 10);

    GtkWidget* f1 = gtk_frame_new("引擎版本");
    GtkWidget* g1 = gtk_grid_new(); gridCfg(g1);
    char vbuf[160];
    std::snprintf(vbuf, sizeof vbuf, "VoxEngine %s　　核心：C++ 物理声学骨架（DSP 参数合成）", kEngineVersion);
    gtk_grid_attach(GTK_GRID(g1), gtk_label_new(vbuf), 0, 0, 2, 1);
    gtk_container_add(GTK_CONTAINER(f1), g1);
    gtk_box_pack_start(GTK_BOX(box), f1, FALSE, FALSE, 0);

    GtkWidget* f2 = gtk_frame_new("JS 外挂插件");
    GtkWidget* g2 = gtk_grid_new(); gridCfg(g2);
    GtkWidget* jsOn = gtk_check_button_new_with_label("启用插件");
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(jsOn), g_jsOn);
    g_signal_connect(jsOn, "toggled", G_CALLBACK(onJsToggled), nullptr);
    gtk_grid_attach(GTK_GRID(g2), jsOn, 0, 0, 1, 1);
    GtkWidget* jsF = fileRow("选择 .js 插件");
    g_signal_connect(jsF, "file-set", G_CALLBACK(onJsFile), nullptr);
    gtk_grid_attach(GTK_GRID(g2), jsF, 1, 0, 1, 1);
    gtk_grid_attach(GTK_GRID(g2), gtk_label_new(
        "外部 .js 文件即插件，暴露 spectrum / f0Spectrum / f0Contour / duration 等钩子（运行于 QuickJS）"), 0, 1, 2, 1);
    gtk_container_add(GTK_CONTAINER(f2), g2);
    gtk_box_pack_start(GTK_BOX(box), f2, FALSE, FALSE, 0);

    GtkWidget* f3 = gtk_frame_new("ONNX 推理引擎");
    GtkWidget* g3 = gtk_grid_new(); gridCfg(g3);
    GtkWidget* sw = gtk_switch_new();
    gtk_switch_set_active(GTK_SWITCH(sw), g_onnxOn);
    g_signal_connect(sw, "state-set", G_CALLBACK(onOnnxToggled), nullptr);
    gtk_grid_attach(GTK_GRID(g3), gtk_label_new("推理开关"), 0, 0, 1, 1);
    gtk_grid_attach(GTK_GRID(g3), sw, 1, 0, 1, 1);
    GtkWidget* onxF = fileRow("选择 .onnx 模型");
    g_signal_connect(onxF, "file-set", G_CALLBACK(onOnnxFile), nullptr);
    gtk_grid_attach(GTK_GRID(g3), onxF, 2, 0, 1, 1);
    g_guiOnnxStatus = gtk_label_new("");
    gtk_label_set_line_wrap(GTK_LABEL(g_guiOnnxStatus), TRUE);
    gtk_grid_attach(GTK_GRID(g3), g_guiOnnxStatus, 0, 1, 3, 1);
    gtk_container_add(GTK_CONTAINER(f3), g3);
    gtk_box_pack_start(GTK_BOX(box), f3, FALSE, FALSE, 0);

    GtkWidget* f4 = gtk_frame_new("声库配置");
    GtkWidget* g4 = gtk_grid_new(); gridCfg(g4);
    GtkWidget* bkF = fileRow("选择声库 .zip");
    g_signal_connect(bkF, "file-set", G_CALLBACK(onBankFile), nullptr);
    gtk_grid_attach(GTK_GRID(g4), gtk_label_new("声库包"), 0, 0, 1, 1);
    gtk_grid_attach(GTK_GRID(g4), bkF, 1, 0, 1, 1);
    gtk_grid_attach(GTK_GRID(g4), gtk_label_new(
        "声库＝一个 zip，内含 voice.json 与 3 个 .onnx（Q8）。未配置则使用内置男声 / 女声。"), 0, 1, 2, 1);
    gtk_container_add(GTK_CONTAINER(f4), g4);
    gtk_box_pack_start(GTK_BOX(box), f4, FALSE, FALSE, 0);

    settingsRefreshOnnx();
    gtk_widget_show_all(dlg);
    gtk_dialog_run(GTK_DIALOG(dlg));
    gtk_widget_destroy(dlg);
    g_guiOnnxStatus = nullptr;
}

static void onRender(GtkWidget* btn, gpointer entryPtr) {
    (void)btn;
    const char* txt = gtk_entry_get_text(GTK_ENTRY(entryPtr));
    std::string s = txt ? txt : "";
    // 输入框内容变化时按音素重建卷帘；之后用鼠标编辑，不再覆盖
    if (!s.empty() && s != g_lastText) {
        std::vector<RollNote> built;
        double t = 0.0;
        size_t i = 0;
        while (i < s.size()) {
            while (i < s.size() && std::isspace((unsigned char)s[i])) i++;
            size_t j = i;
            while (j < s.size() && !std::isspace((unsigned char)s[j])) j++;
            if (j > i) {
                std::string tok = s.substr(i, j - i);
                int vw = 0, cs = 0;
                if (lookupPhoneme(tok, vw, cs)) {
                    RollNote n; n.start = t; n.dur = 0.4; n.f0 = 220.0;
                    n.vowel = vw; n.cons = cs; n.ph = tok;
                    built.push_back(n);
                    t += 0.4;
                }
            }
            i = j;
        }
        if (built.empty()) {
            std::snprintf(g_status, sizeof g_status, "无效音素，请重新输入");
            gtk_widget_queue_draw(g_daRoll);
            return;
        }
        g_notes = built;
        g_lastText = s;
        g_sel = -1;
    }
    if (g_notes.empty()) {
        std::snprintf(g_status, sizeof g_status, "卷帘为空：在网格上点击添加音符");
        gtk_widget_queue_draw(g_daRoll);
        return;
    }

    renderRoll();

    g_dur = g_pcm.size() / g_fs;
    computeSpectrum();
    computePitch();
    smoothPitch(g_f0);
    std::snprintf(g_status, sizeof g_status, "%.2fs  %zu 音符", g_dur, g_notes.size());
    gtk_widget_queue_draw(g_daFormant);
    gtk_widget_queue_draw(g_daSpec);
    gtk_widget_queue_draw(g_daPitch);
    gtk_widget_queue_draw(g_daRoll);
    const char* tmp = g_get_tmp_dir();
    std::string path = std::string(tmp) + "/vox_gui_seq.wav";
    writeSequenceWav(g_pcm, path, g_fs);
    mpvPlay(path);
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--selftest") {
        int fail = 0;
        auto chk = [&](bool ok, const char* what) {
            std::printf("%s %s\n", ok ? "  通过 " : "  失败 ", what);
            if (!ok) ++fail;
        };
        chk(std::fabs(rollX2T(rollT2X(1.25)) - 1.25) < 1e-9, "时间<->像素 往返");
        for (double f : {110.0, 220.0, 440.0, 660.0})
            chk(std::fabs(rollY2F(rollF2Y(f, 400), 400) - f) < 0.6, "音高<->像素 往返");
        chk(std::fabs(rollSnapF(223.0) - 220.0) < 0.6, "半音吸附 223 -> 220");
        chk(std::fabs(rollSnapF(233.0) - 233.08) < 0.6, "半音吸附 233 -> A#3");
        chk(std::fabs(rollSnapT(0.37) - 0.35) < 1e-9, "时间吸附 0.37 -> 0.35");

        g_notes.clear();
        RollNote a; a.start = 0.0; a.dur = 0.4; a.f0 = 220; g_notes.push_back(a);
        RollNote b; b.start = 1.0; b.dur = 0.3; b.f0 = 440; g_notes.push_back(b);
        const int H = 400;
        chk(rollHit(rollT2X(0.2), rollF2Y(220, H), H) == 0, "命中音符#1");
        chk(rollHit(rollT2X(1.1), rollF2Y(440, H), H) == 1, "命中音符#2");
        chk(rollHit(rollT2X(0.2), rollF2Y(440, H), H) == -1, "空白处不误命中");

        // 音符时间定位：3 个不连续音符，检查输出长度与各音符起始处的能量
        g_notes.clear();
        for (int k = 0; k < 3; ++k) {
            RollNote n; n.start = 0.5 * k; n.dur = 0.3; n.f0 = 220.0 * std::pow(2.0, k / 12.0);
            n.vowel = 0; n.cons = 0; n.ph = "a";
            g_notes.push_back(n);
        }
        renderRoll();
        const size_t want = (size_t)std::llround((1.0 + 0.3 + 0.2) * 44100.0);
        chk(g_pcm.size() == want, "输出长度 = 末尾音符结束 + 0.2s 尾巴");
        auto rms = [&](double t0, double t1) {
            size_t i0 = (size_t)(t0 * 44100.0), i1 = std::min((size_t)(t1 * 44100.0), g_pcm.size());
            double e = 0; for (size_t i = i0; i < i1; ++i) e += (double)g_pcm[i] * g_pcm[i];
            return std::sqrt(e / std::max<size_t>(1, i1 - i0));
        };
        chk(rms(0.10, 0.25) > 0.02, "音符#1 区间有声");
        chk(rms(0.60, 0.75) > 0.02, "音符#2 区间有声");
        chk(rms(1.10, 1.25) > 0.02, "音符#3 区间有声");
        chk(rms(0.35, 0.45) < 0.30 * rms(0.10, 0.25), "音符间空隙确实更安静");
        double pk = 0; for (float v : g_pcm) pk = std::max(pk, (double)std::fabs(v));
        chk(pk <= 0.87 && pk > 0.80, "峰值归一化到 0.85 附近");

        // 音素连续化：两个首尾相接的音符，交界处不应塌陷
        g_notes.clear();
        for (int k = 0; k < 2; ++k) {
            RollNote n; n.start = 0.5 * k; n.dur = 0.5; n.f0 = 220.0;
            n.vowel = 0; n.cons = 0; n.ph = "a";
            g_notes.push_back(n);
        }
        renderRoll();
        const double body = rms(0.20, 0.35);      // 第一个音符中段
        const double joint = rms(0.485, 0.515);   // 交界处 ±15ms
        chk(joint > 0.55 * body, "连音交界处不塌陷（连续化生效）");
        std::printf("        交界 RMS %.4f / 音符中段 RMS %.4f = %.2f\n", joint, body, joint / body);

        // 回归：三个「不同元音」的音符必须都发声（曾因探针复用旧 normGain 而只发一个音）
        g_notes.clear();
        for (int k = 0; k < 3; ++k) {
            RollNote n; n.start = 0.4 * k; n.dur = 0.4; n.f0 = 220.0;
            n.vowel = k; n.cons = 0; n.ph = "aoe" + k;
            g_notes.push_back(n);
        }
        renderRoll();
        double r1 = rms(0.10, 0.35), r2 = rms(0.50, 0.75), r3 = rms(0.90, 1.15);
        double rmax = std::max(r1, std::max(r2, r3));
        chk(r1 > 0.30 * rmax && r2 > 0.30 * rmax && r3 > 0.30 * rmax,
            "换元音不吞音（三个元音都发声）");
        std::printf("        a/o/e 三段 RMS %.4f / %.4f / %.4f\n", r1, r2, r3);
        // ---- 多轨：切换后各自内容不丢 ----
        g_tracks.clear(); g_notes.clear();
        g_tracks.push_back(Track{ "甲", {} });
        g_tracks.push_back(Track{ "乙", {} });
        g_curTrack = 0; g_notes.clear();
        RollNote n1; n1.start = 0; n1.dur = 0.4; n1.f0 = 220; g_notes.push_back(n1);
        trackLoad(1);
        chk(g_notes.empty(), "切到空轨道后卷帘为空");
        RollNote n2; n2.start = 0; n2.dur = 0.4; n2.f0 = 330; g_notes.push_back(n2);
        trackLoad(0);
        chk(g_notes.size() == 1 && std::fabs(g_notes[0].f0 - 220) < 1e-6, "切回甲轨内容仍在");
        trackLoad(1);
        chk(g_notes.size() == 1 && std::fabs(g_notes[0].f0 - 330) < 1e-6, "乙轨内容独立");

        // ---- 复制 / 粘贴 ----
        g_notes.clear(); g_sel2.clear();
        RollNote a1; a1.start = 0.2; a1.dur = 0.4; a1.f0 = 220; g_notes.push_back(a1);
        RollNote a2; a2.start = 0.7; a2.dur = 0.4; a2.f0 = 330; g_notes.push_back(a2);
        g_sel2.push_back(0); g_sel2.push_back(1);
        doCopy();
        chk(g_clip.size() == 2, "复制到 2 个音符");
        chk(std::fabs(g_clip[0].start) < 1e-9 && std::fabs(g_clip[1].start - 0.5) < 1e-9,
            "复制后相对时间正确");
        g_mouseT = 2.0;
        doPaste();
        chk(g_notes.size() == 4, "粘贴后共 4 个音符");
        chk(std::fabs(g_notes[2].start - 2.0) < 1e-6 && std::fabs(g_notes[3].start - 2.5) < 1e-6,
            "粘贴锚定到鼠标所在时间");

        // ---- 声库 ----
        chk(std::fabs(voicebankScale(1) - voicebankScale(0)) > 0.1, "男女声库参数确有差异");

        // ---- 每轨参数独立：互不污染 ----
        g_tracks.clear(); g_notes.clear();
        g_tracks.push_back(Track{ "甲", {}, TrackParams{} });
        g_tracks.push_back(Track{ "乙", {}, TrackParams{} });
        g_curTrack = 0;
        curP().f0Shift = 3.0;
        curP().voicebank = 2;
        curP().alpha = 7.5;
        trackLoad(1);
        chk(std::fabs(curP().f0Shift) < 1e-9, "切到乙轨 f0Shift 归零（未被污染）");
        chk(curP().voicebank == 0,          "切到乙轨声库独立");
        curP().alpha = 9.0;                 // 改乙轨
        trackLoad(0);
        chk(std::fabs(curP().f0Shift - 3.0) < 1e-9, "切回甲轨 f0Shift 保留");
        chk(curP().voicebank == 2,                  "切回甲轨声库保留");
        chk(std::fabs(curP().alpha - 7.5) < 1e-9,   "每轨参数互不污染（乙轨改动未影响甲轨）");

        std::printf("\n%s  (%d 项失败)\n", fail ? "自检失败" : "自检全部通过", fail);
        return fail ? 1 : 0;
    }

    gtk_init(&argc, &argv);
    ensureCss();
    GtkWidget* win = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(win), "VoxEngine");
    gtk_window_set_default_size(GTK_WINDOW(win), 1080, 600);
    gtk_window_move(GTK_WINDOW(win), 30, 20);        // 显式定位，避免 WM 丢到屏幕外
    g_signal_connect(win, "key-press-event", G_CALLBACK(onKey), nullptr);
    // 外部 kill 时也要回收常驻播放器，否则 mpv 会变成孤儿堆积
    g_unix_signal_add(SIGTERM, [](gpointer) -> gboolean { mpvQuit(); gtk_main_quit(); return G_SOURCE_REMOVE; }, nullptr);
    g_unix_signal_add(SIGINT,  [](gpointer) -> gboolean { mpvQuit(); gtk_main_quit(); return G_SOURCE_REMOVE; }, nullptr);
    g_signal_connect(win, "destroy", G_CALLBACK(+[](GtkWidget*, gpointer){ mpvQuit(); gtk_main_quit(); }), nullptr);

    GtkWidget* root = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);
    gtk_container_set_border_width(GTK_CONTAINER(root), 6);
    g_overlay = gtk_overlay_new();
    gtk_container_add(GTK_CONTAINER(win), g_overlay);
    gtk_container_add(GTK_CONTAINER(g_overlay), root);

    // ---- 顶部图表栏 ----
    GtkWidget* top = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    gtk_box_pack_start(GTK_BOX(root), top, FALSE, FALSE, 0);
    g_daFormant = gtk_drawing_area_new();
    gtk_widget_set_size_request(g_daFormant, -1, 70);
    g_signal_connect(g_daFormant, "draw", G_CALLBACK(onDrawFormant), nullptr);
    gtk_box_pack_start(GTK_BOX(top), g_daFormant, TRUE, TRUE, 0);
    g_daSpec = gtk_drawing_area_new();
    gtk_widget_set_size_request(g_daSpec, -1, 70);
    g_signal_connect(g_daSpec, "draw", G_CALLBACK(onDrawSpec), nullptr);
    gtk_box_pack_start(GTK_BOX(top), g_daSpec, TRUE, TRUE, 0);
    g_daPitch = gtk_drawing_area_new();
    gtk_widget_set_size_request(g_daPitch, -1, 70);
    g_signal_connect(g_daPitch, "draw", G_CALLBACK(onDrawPitch), nullptr);
    gtk_box_pack_start(GTK_BOX(top), g_daPitch, TRUE, TRUE, 0);

    // ---- 三区 ----
    GtkWidget* mid = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    gtk_box_pack_start(GTK_BOX(root), mid, TRUE, TRUE, 0);

    GtkWidget* singer = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);
    gtk_widget_set_size_request(singer, 156, -1);
    gtk_box_pack_start(GTK_BOX(mid), singer, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(singer), gtk_label_new("歌手 / 本轨参数"), FALSE, FALSE, 0);
    GtkWidget* av = gtk_button_new_with_label("声库：男声（内置）\n（点击选择）");
    g_signal_connect(av, "clicked", G_CALLBACK(onVoicebank), nullptr);
    g_vbButton = av;
    gtk_box_pack_start(GTK_BOX(singer), av, FALSE, FALSE, 0);

    // ---- 声库按钮往下：详细调节（写进当前轨道）----
    g_scF0 = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, -12.0, 12.0, 1.0);
    gtk_range_set_value(GTK_RANGE(g_scF0), 0.0);
    g_signal_connect(g_scF0, "value-changed", G_CALLBACK(onF0Changed), nullptr);
    gtk_box_pack_start(GTK_BOX(singer), gtk_label_new("基频细调（半音）"), FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(singer), g_scF0, FALSE, FALSE, 0);

    g_scAlpha = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 0.5, 25.0, 0.1);
    gtk_range_set_value(GTK_RANGE(g_scAlpha), 2.5);
    g_signal_connect(g_scAlpha, "value-changed", G_CALLBACK(onAlphaChanged), nullptr);
    gtk_box_pack_start(GTK_BOX(singer), gtk_label_new("张力 alpha"), FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(singer), g_scAlpha, FALSE, FALSE, 0);

    g_scBreath = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 0.0, 1.0, 0.05);
    gtk_range_set_value(GTK_RANGE(g_scBreath), 0.0);
    g_signal_connect(g_scBreath, "value-changed", G_CALLBACK(onBreathChanged), nullptr);
    gtk_box_pack_start(GTK_BOX(singer), gtk_label_new("气息"), FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(singer), g_scBreath, FALSE, FALSE, 0);

    g_daRoll = gtk_drawing_area_new();
    g_signal_connect(g_daRoll, "draw", G_CALLBACK(onDrawRoll), nullptr);
    gtk_widget_add_events(g_daRoll, GDK_BUTTON_PRESS_MASK | GDK_BUTTON_RELEASE_MASK | GDK_POINTER_MOTION_MASK);
    g_signal_connect(g_daRoll, "button-press-event",   G_CALLBACK(onRollPress),   nullptr);
    g_signal_connect(g_daRoll, "button-release-event", G_CALLBACK(onRollRelease), nullptr);
    g_signal_connect(g_daRoll, "motion-notify-event",  G_CALLBACK(onRollMotion),  nullptr);
    gtk_box_pack_start(GTK_BOX(mid), g_daRoll, TRUE, TRUE, 0);

    GtkWidget* track = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);
    gtk_widget_set_size_request(track, 120, -1);
    gtk_box_pack_start(GTK_BOX(mid), track, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(track), gtk_label_new("轨道"), FALSE, FALSE, 0);
    g_trackBox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);
    gtk_box_pack_start(GTK_BOX(track), g_trackBox, FALSE, FALSE, 0);
    GtkWidget* newTrk = gtk_button_new_with_label("＋ 新建空白轨道");
    g_signal_connect(newTrk, "clicked", G_CALLBACK(onNewTrack), nullptr);
    gtk_box_pack_start(GTK_BOX(track), newTrk, FALSE, FALSE, 0);

    // ---- 底部控制栏 ----
    GtkWidget* bot = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    gtk_box_pack_start(GTK_BOX(root), bot, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(bot), gtk_label_new("音素:"), FALSE, FALSE, 0);
    GtkWidget* entry = gtk_entry_new();
    gtk_entry_set_text(GTK_ENTRY(entry), "中/英/日：ba be bi / ni hao / ka ki");
    gtk_box_pack_start(GTK_BOX(bot), entry, TRUE, TRUE, 0);
    GtkWidget* gear = gtk_button_new_with_label("⚙ 设置");
    gtk_box_pack_start(GTK_BOX(bot), gear, FALSE, FALSE, 0);
    g_signal_connect(gear, "clicked", G_CALLBACK(onSettings), nullptr);
    GtkWidget* play = gtk_button_new_with_label("▶ 渲染并播放");
    gtk_box_pack_start(GTK_BOX(bot), play, FALSE, FALSE, 0);
    g_signal_connect(play, "clicked", G_CALLBACK(onRender), entry);
    GtkWidget* st = gtk_label_new(g_status);
    gtk_box_pack_start(GTK_BOX(bot), st, FALSE, FALSE, 0);

    // ---- 精细设置栏：声门开相增长率 alpha ----
    GtkWidget* fine = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    gtk_box_pack_start(GTK_BOX(root), fine, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(fine), gtk_label_new("精细设置  alpha(张力):"), FALSE, FALSE, 0);
    GtkWidget* sc = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 0.5, 25.0, 0.1);
    gtk_range_set_value(GTK_RANGE(sc), 2.5);
    gtk_widget_set_size_request(sc, 420, -1);
    gtk_box_pack_start(GTK_BOX(fine), sc, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(fine), gtk_label_new("2~5 自然 / 25 尖锐(旧默认)"), FALSE, FALSE, 0);
    g_signal_connect(sc, "value-changed", G_CALLBACK(+[](GtkRange* r, gpointer){ g_alpha = gtk_range_get_value(r); }), nullptr);

    buildVoicebankPanel();                 // 必须在 show_all 之前建好，否则子控件不显示
    gtk_widget_show_all(win);
    gtk_widget_show_all(g_vbRevealer);                                  // overlay 子控件需显式 show
    gtk_revealer_set_reveal_child(GTK_REVEALER(g_vbRevealer), FALSE);   // 初始收起
    if (g_tracks.empty()) g_tracks.push_back(Track{ "轨道一", {} });
    g_curTrack = 0;
    if (g_notes.empty()) seedDemo();   // 空卷帘时给一段可编辑的默认旋律
    trackSave();
    rebuildTracks();
    syncParamWidgets();
    gtk_widget_queue_draw(g_daRoll);
    mpvEnsureAsync();     // 非阻塞预热播放器
    gtk_main();
    return 0;
}
