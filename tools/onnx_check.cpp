#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include "onnx_engine.h"
int main(int argc, char** argv){
    if (argc < 2) { std::printf("用法: t_onnx <model.onnx>\n"); return 1; }
    vox::OnnxEngine e;
    e.loadAsync(argv[1]);
    e.wait();
    std::printf("加载: ready=%d loading=%d  err=%s\n", (int)e.ready(), (int)e.loading(), e.lastError().c_str());
    if (!e.ready()) return 1;

    std::vector<float> pcm(40000);
    for (size_t i=0;i<pcm.size();++i) pcm[i]=(float)(std::sin(2*M_PI*440*i/44100.0)*0.8);
    std::vector<float> orig = pcm;
    bool ok = e.process(pcm, 44100.0);
    std::printf("推理: process=%d\n", (int)ok);
    if (!ok) { std::printf("err=%s\n", e.lastError().c_str()); return 2; }
    double maxerr=0; for(size_t i=0;i<pcm.size();++i) maxerr=std::max(maxerr,std::fabs((double)pcm[i]-0.5*(double)orig[i]));
    std::printf("判定: 输出应为输入×0.5, 最大误差=%.3e\n", maxerr);
    std::printf("%s\n", maxerr<1e-5 ? "★ ONNX 加载+推理 实测通过" : "★ 判定失败");
    return maxerr<1e-5?0:3;
}
