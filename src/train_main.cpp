// VoxEngine 全自动训练流水线（C++ / GTK3）
// 选「歌唱模板文件夹」+「说话样本文件夹」-> 全自动提取音高、造成对数据、生成训练脚本、验证模型
#include <gtk/gtk.h>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include "i18n.h"
#include "auto_pipeline.h"
#include "onnx_engine.h"

using namespace vox;

static std::string g_sing, g_speech, g_out;
static GtkWidget *g_logView=nullptr, *g_btnRun=nullptr, *g_status=nullptr;
static GtkWidget *g_spSing=nullptr, *g_spSpeak=nullptr;
static std::mutex g_mx;
static std::vector<std::string> g_lines;
static bool g_busy=false;

static void push(const std::string& s){ std::lock_guard<std::mutex> lk(g_mx); g_lines.push_back(s); }

static gboolean drain(gpointer){
    std::vector<std::string> got;
    { std::lock_guard<std::mutex> lk(g_mx); got.swap(g_lines); }
    if(!got.empty() && g_logView){
        GtkTextBuffer* b = gtk_text_view_get_buffer(GTK_TEXT_VIEW(g_logView));
        GtkTextIter end; gtk_text_buffer_get_end_iter(b,&end);
        for(auto& s:got){ gtk_text_buffer_insert(b,&end,(s+"\n").c_str(),-1); }
        gtk_text_view_scroll_to_iter(GTK_TEXT_VIEW(g_logView),&end,0.0,FALSE,0.0,1.0);
    }
    return G_SOURCE_CONTINUE;
}

// ---- 自动生成的训练脚本（按 MODEL_CONTRACT.md）----
static const char* kTrainPy = R"PYEOF(
#!/usr/bin/env python3
"""VoxEngine 训练脚本（自动生成）—— 契约见 MODEL_CONTRACT.md
用法: python3 train.py --task f0        # 训练基频模型
      python3 train.py --task polish    # 训练润色模型
需要: torch numpy onnx
"""
import argparse, numpy as np, torch, torch.nn as nn

CHUNK, T = 16384, 23

class F0Net(nn.Module):
    """pcm_in[1,1,16384] -> f0[1,23,1]"""
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(
            nn.Conv1d(1, 32, 65, 4, 32), nn.ReLU(),
            nn.Conv1d(32, 64, 33, 4, 16), nn.ReLU(),
            nn.Conv1d(64, 96, 17, 2, 8), nn.ReLU(),
            nn.AdaptiveAvgPool1d(T),
        )
        s.head = nn.Linear(96, 1)
    def forward(s, x):
        h = s.net(x)                  # [B,96,23]
        h = h.transpose(1, 2)         # [B,23,96]
        return s.head(h)              # [B,23,1]

class PolishNet(nn.Module):
    """pcm_in[1,1,16384] -> pcm_out[1,1,16384]"""
    def __init__(s, c=48):
        super().__init__()
        s.enc = nn.Sequential(nn.Conv1d(1,c,17,1,8), nn.ReLU(),
                              nn.Conv1d(c,c,17,2,8), nn.ReLU(),
                              nn.Conv1d(c,c*2,17,2,8), nn.ReLU())
        s.dec = nn.Sequential(nn.ConvTranspose1d(c*2,c,4,2,1), nn.ReLU(),
                              nn.ConvTranspose1d(c,c,4,2,1), nn.ReLU(),
                              nn.Conv1d(c,1,17,1,8))
    def forward(s, x):
        return x + s.dec(s.enc(x))

def train(task, out):
    if task == 'f0':
        X = np.load('f0_X.npy'); Y = np.load('f0_Y.npy')
        net = F0Net(); onnx_name = 'f0.onnx'
        print(f'样本 {X.shape}  标签 {Y.shape}')
        opt = torch.optim.Adam(net.parameters(), 2e-4)
        for ep in range(60):
            perm = np.random.permutation(len(X)); tot=0
            for i in range(0, len(perm), 32):
                idx = perm[i:i+32]
                xb = torch.from_numpy(X[idx]); yb = torch.from_numpy(Y[idx])   # [B,1,N] [B,23]
                pred = net(xb).squeeze(-1)                                      # [B,23]
                m = (yb > 0).float()
                loss = (torch.abs(pred-yb)*m).sum()/m.sum().clamp(min=1) \
                     + 0.1*torch.nn.functional.binary_cross_entropy_with_logits(
                           torch.log(pred.clamp(min=1e-3)), m)
                opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
            print(f'  epoch {ep:3d}  loss {tot/max(1,len(perm)//32):.4f}')
        dummy = torch.zeros(1,1,CHUNK)
        torch.onnx.export(net, dummy, onnx_name, opset_version=13,
            input_names=['pcm_in'], output_names=['f0'],
            dynamic_axes=None)
    else:
        X = np.load('polish_X.npy'); Y = np.load('polish_Y.npy')
        net = PolishNet(); onnx_name = 'polish.onnx'
        print(f'样本 {X.shape}  目标 {Y.shape}')
        opt = torch.optim.Adam(net.parameters(), 1e-4)
        for ep in range(60):
            perm = np.random.permutation(len(X)); tot=0
            for i in range(0, len(perm), 16):
                idx = perm[i:i+16]
                xb = torch.from_numpy(X[idx]); yb = torch.from_numpy(Y[idx])
                pred = net(xb)
                l1 = torch.nn.functional.l1_loss(pred, yb)
                stft = 0
                for n in (512, 1024, 2048):        # 多分辨率 STFT 损失（单用 L1 会糊）
                    w = torch.hann_window(n)
                    a = torch.stft(pred[0,0], n, n//4, window=w, return_complex=True).abs()
                    b = torch.stft(yb[0,0],   n, n//4, window=w, return_complex=True).abs()
                    stft = stft + torch.nn.functional.l1_loss(torch.log(a+1e-5), torch.log(b+1e-5))
                loss = l1 + 0.5*stft
                opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
            print(f'  epoch {ep:3d}  loss {tot/max(1,len(perm)//16):.4f}')
        dummy = torch.zeros(1,1,CHUNK)
        torch.onnx.export(net, dummy, onnx_name, opset_version=13,
            input_names=['pcm_in'], output_names=['pcm_out'],
            dynamic_axes=None)
    print(f'已导出 {onnx_name}')

    # Q8 动态量化（需求 006）
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        q = onnx_name.replace('.onnx', '_q8.onnx')
        quantize_dynamic(onnx_name, q, weight_type=QuantType.QInt8)
        print(f'已量化 {q}')
    except Exception as e:
        print('量化跳过:', e)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='f0', choices=['f0','polish'])
    a = ap.parse_args()
    train(a.task, '.')
)PYEOF";

static void runPipelineThread(){
    std::string log;
    PipeStats st;
    push(T("=== 开始全自动处理 ===", "=== Auto processing started ==="));
    double minS = gtk_spin_button_get_value(GTK_SPIN_BUTTON(g_spSing));
    double minP = gtk_spin_button_get_value(GTK_SPIN_BUTTON(g_spSpeak));
    bool ok = runAutoPipeline(g_sing, g_speech, g_out, st, log, minS, minP);
    { std::istringstream is(log); std::string l; while(std::getline(is,l)) push(l); }
    std::ofstream(g_out + "/train.py") << kTrainPy;
    push(T("已生成训练脚本: ", "Training script written: ") + g_out + "/train.py");
    push(ok ? T("=== 处理完成，可以开始训练 ===", "=== Done - ready to train ===") : T("=== 处理失败 ===", "=== Processing failed ==="));
    g_busy = false;
    if(g_btnRun) gtk_widget_set_sensitive(g_btnRun, TRUE);
    if(g_status) gtk_label_set_text(GTK_LABEL(g_status), ok ? T("处理完成", "Done") : T("处理失败", "Failed"));
}

static void onRun(GtkWidget* btn, gpointer){
    if(g_busy) return;
    if(g_sing.empty() || g_speech.empty()){ push(T("请先选择两个文件夹", "Please choose both folders first")); return; }
    if(g_out.empty()){ push(T("请先选择输出文件夹", "Please choose the output folder first")); return; }

    const double minS = gtk_spin_button_get_value(GTK_SPIN_BUTTON(g_spSing));
    const double minP = gtk_spin_button_get_value(GTK_SPIN_BUTTON(g_spSpeak));

    // ---- 规则校验（在主线程做，便于弹确认框）----
    DirScan sa = scanDir(g_sing), sb = scanDir(g_speech);
    std::string rep; bool needMp3 = false;
    CheckResult cr = validateInputs(sa, sb, minS, minP, rep, needMp3);
    push(rep);

    if (cr == CheckResult::Refuse) {
        gtk_label_set_text(GTK_LABEL(g_status), T("已拒绝：输入不符合要求", "Rejected: input does not meet requirements"));
        return;
    }
    bool confirmed = false;
    if (cr == CheckResult::NeedConfirmMp3) {
        GtkWidget* top = gtk_widget_get_toplevel(btn);
        GtkWidget* d = gtk_message_dialog_new(
            GTK_IS_WINDOW(top) ? GTK_WINDOW(top) : nullptr,
            GTK_DIALOG_MODAL, GTK_MESSAGE_QUESTION, GTK_BUTTONS_YES_NO,
            T("检测到 MP3 文件，是否转换为 WAV 后继续？", "MP3 files found. Convert to WAV and continue?"));
        gtk_message_dialog_format_secondary_text(GTK_MESSAGE_DIALOG(d), "%s", rep.c_str());
        const int r = gtk_dialog_run(GTK_DIALOG(d));
        gtk_widget_destroy(d);
        if (r != GTK_RESPONSE_YES) {
            push(T("用户取消：未确认 MP3 转换，已停止。", "Cancelled: MP3 conversion not confirmed."));
            gtk_label_set_text(GTK_LABEL(g_status), T("已取消", "Cancelled"));
            return;
        }
        push(T("用户已确认转换 MP3。", "User confirmed MP3 conversion."));
        confirmed = true;
    }
    ::setenv("VOX_MP3_CONFIRM", confirmed ? "1" : "0", 1);

    g_busy = true; gtk_widget_set_sensitive(g_btnRun, FALSE);
    gtk_label_set_text(GTK_LABEL(g_status), T("处理中…", "Processing..."));
    std::thread(runPipelineThread).detach();
}

static void onFolder(GtkFileChooserButton* b, gpointer data){
    std::string* dst = (std::string*)data;
    const char* f = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(b));
    if(f) *dst = f;
}

static void onGenScript(GtkWidget*, gpointer){
    if(g_out.empty()){ push("请先选输出文件夹"); return; }
    std::ofstream(g_out + "/train.py") << kTrainPy;
    push(T("已生成 ", "Written ") + g_out + "/train.py");
}

static void onTrain(GtkWidget*, gpointer){
    if(g_out.empty()){ push("请先选输出文件夹"); return; }
    std::string cmd = "bash -lc 'cd \"" + g_out + "\" && python3 train.py --task f0 2>&1 | tail -40'";
    FILE* p = ::popen(cmd.c_str(), "r");
    if(!p){ push(T("无法启动 python3（训练需要 torch，本机大概率没有）", "Cannot start python3 (training needs torch, likely unavailable here)")); return; }
    char buf[512];
    while(std::fgets(buf, sizeof buf, p)) push(std::string(buf));
    ::pclose(p);
    push(T("训练进程结束", "Training process finished"));
}

static void onVerify(GtkWidget*, gpointer){
    GtkWidget* d = gtk_file_chooser_dialog_new(T("选择 .onnx 模型", "Choose .onnx model"), nullptr,
        GTK_FILE_CHOOSER_ACTION_OPEN, T("取消", "Cancel"), GTK_RESPONSE_CANCEL, T("打开", "Open"), GTK_RESPONSE_ACCEPT, nullptr);
    if(gtk_dialog_run(GTK_DIALOG(d)) == GTK_RESPONSE_ACCEPT){
        char* f = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(d));
        OnnxEngine e; e.loadAsync(f); e.wait();
        push(std::string(T("验证 ", "Verifying ")) + f + (e.ready() ? T("  加载成功", "  loaded OK") : T("  加载失败: ", "  load failed: ") + e.lastError()));
        if(e.ready()){
            std::vector<float> pcm(44100, 0.0f);
            for(size_t i=0;i<pcm.size();++i) pcm[i]=(float)(0.5*std::sin(2*M_PI*220*i/44100.0));
            std::vector<float> f0;
            if(e.predictF0(pcm,44100.0,f0))
                push(T("  按 F0 模型解释成功，输出 ", "  interpreted as F0 model, output ") + std::to_string(f0.size()) + T(" 帧", " frame(s)"));
            else
                push(T("  按 F0 模型解释失败（可能是润色模型）: ", "  not an F0 model (maybe a polish model): ") + e.lastError());
        }
        g_free(f);
    }
    gtk_widget_destroy(d);
}

static void onQuit(){ gtk_main_quit(); }

int main(int argc, char** argv){
    gtk_init(&argc, &argv);
    GtkWidget* win = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(win), T("VoxEngine 全自动训练流水线", "VoxEngine Auto Training Pipeline"));
    gtk_window_set_default_size(GTK_WINDOW(win), 860, 560);
    g_signal_connect(win, "destroy", G_CALLBACK(onQuit), nullptr);

    GtkWidget* root = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    gtk_container_set_border_width(GTK_CONTAINER(root), 10);
    gtk_container_add(GTK_CONTAINER(win), root);

    const char* labels[3] = { T("歌唱模板文件夹", "Singing template folder"), T("说话样本文件夹", "Speech sample folder"), T("输出文件夹", "Output folder") };
    std::string* dsts[3] = { &g_sing, &g_speech, &g_out };
    for(int i=0;i<3;i++){
        GtkWidget* row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
        GtkWidget* lb = gtk_label_new(labels[i]);
        gtk_widget_set_size_request(lb, 130, -1);
        gtk_box_pack_start(GTK_BOX(row), lb, FALSE, FALSE, 0);
        GtkWidget* fc = gtk_file_chooser_button_new(T("选择文件夹", "Choose a folder"), GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER);
        gtk_widget_set_size_request(fc, 520, -1);
        g_signal_connect(fc, "file-set", G_CALLBACK(onFolder), dsts[i]);
        gtk_box_pack_start(GTK_BOX(row), fc, TRUE, TRUE, 0);
        gtk_box_pack_start(GTK_BOX(root), row, FALSE, FALSE, 0);
    }

    GtkWidget* req = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    gtk_box_pack_start(GTK_BOX(req), gtk_label_new(T("时长门槛：歌唱不少于", "Duration: singing at least")), FALSE, FALSE, 0);
    g_spSing = gtk_spin_button_new_with_range(1, 600, 1);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(g_spSing), 60);
    gtk_box_pack_start(GTK_BOX(req), g_spSing, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(req), gtk_label_new(T("分钟，说话不少于", "min, speech at least")), FALSE, FALSE, 0);
    g_spSpeak = gtk_spin_button_new_with_range(1, 600, 1);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(g_spSpeak), 30);
    gtk_box_pack_start(GTK_BOX(req), g_spSpeak, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(req), gtk_label_new(T("分钟", "min")), FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(root), req, FALSE, FALSE, 0);

    GtkWidget* bar = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    g_btnRun = gtk_button_new_with_label(T("① 全自动处理（提取音高 + 造成对数据）", "(1) Auto process (extract pitch + build pairs)"));
    g_signal_connect(g_btnRun, "clicked", G_CALLBACK(onRun), nullptr);
    gtk_box_pack_start(GTK_BOX(bar), g_btnRun, FALSE, FALSE, 0);
    GtkWidget* b2 = gtk_button_new_with_label(T("② 生成训练脚本", "(2) Generate training script"));
    g_signal_connect(b2, "clicked", G_CALLBACK(onGenScript), nullptr);
    gtk_box_pack_start(GTK_BOX(bar), b2, FALSE, FALSE, 0);
    GtkWidget* b3 = gtk_button_new_with_label(T("③ 开始训练", "(3) Start training"));
    g_signal_connect(b3, "clicked", G_CALLBACK(onTrain), nullptr);
    gtk_box_pack_start(GTK_BOX(bar), b3, FALSE, FALSE, 0);
    GtkWidget* b4 = gtk_button_new_with_label(T("④ 验证 ONNX", "(4) Verify ONNX"));
    g_signal_connect(b4, "clicked", G_CALLBACK(onVerify), nullptr);
    gtk_box_pack_start(GTK_BOX(bar), b4, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(root), bar, FALSE, FALSE, 0);

    GtkWidget* sc = gtk_scrolled_window_new(nullptr, nullptr);
    g_logView = gtk_text_view_new();
    gtk_text_view_set_editable(GTK_TEXT_VIEW(g_logView), FALSE);
    gtk_text_view_set_monospace(GTK_TEXT_VIEW(g_logView), TRUE);
    gtk_container_add(GTK_CONTAINER(sc), g_logView);
    gtk_box_pack_start(GTK_BOX(root), sc, TRUE, TRUE, 0);

    g_status = gtk_label_new(T("就绪：选两个文件夹，点①", "Ready: pick two folders, then click (1)"));
    gtk_box_pack_start(GTK_BOX(root), g_status, FALSE, FALSE, 0);

    g_timeout_add(100, drain, nullptr);
    gtk_widget_show_all(win);
    gtk_main();
    return 0;
}
