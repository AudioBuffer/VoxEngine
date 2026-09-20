#pragma once
// ---------------------------------------------------------------
// ONNX 推理接口
//   设计原则：**接口内没有任何模型时，放弃 ONNX，回落到纯参数合成**。
//   因此 ready()==false 时所有调用都是零副作用的空操作，
//   引擎完全走 DSP 参数合成路径，绝不等待、绝不报错、绝不半边输出。
//   启用真实推理：编译时加 -DVOX_WITH_ONNX 并链接 onnxruntime。
// ---------------------------------------------------------------
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <cstdio>
#include <cmath>
#include <algorithm>

#ifdef VOX_WITH_ONNX
#include <onnxruntime_c_api.h>
#include <dlfcn.h>
#include <cstdlib>
#endif

namespace vox {

class OnnxEngine {
public:
    OnnxEngine() = default;
    ~OnnxEngine() { unload(); }
    OnnxEngine(const OnnxEngine&) = delete;
    OnnxEngine& operator=(const OnnxEngine&) = delete;

    // 异步加载：不阻塞界面与渲染。失败即保持未就绪 -> 纯参数合成。
    void loadAsync(const std::string& path) {
        unload();
        {
            std::lock_guard<std::mutex> lk(m_);
            path_ = path;
            err_.clear();
        }
        loading_.store(true);
#ifdef VOX_WITH_ONNX
        worker_ = std::thread([this, path] {
            bool ok = false;
            try { ok = loadImpl(path); }
            catch (const std::exception& e) { std::lock_guard<std::mutex> lk(m_); err_ = e.what(); }
            catch (...) { std::lock_guard<std::mutex> lk(m_); err_ = "未知错误"; }
            if (!ok) { std::lock_guard<std::mutex> lk(m_); if (err_.empty()) err_ = "模型加载失败"; }
            ready_.store(ok);
            loading_.store(false);
        });
#else
        { std::lock_guard<std::mutex> lk(m_); err_ = "未编译 ONNX 支持（纯参数合成）"; }
        loading_.store(false);
        ready_.store(false);
#endif
    }

    void wait() { if (worker_.joinable()) worker_.join(); }
    bool ready()   const { return ready_.load(); }
    bool loading() const { return loading_.load(); }
    bool enabled() const { return hasModel_.load(); }   // 接口里到底有没有模型
    std::string lastError() const { std::lock_guard<std::mutex> lk(m_); return err_; }

    // 清空：立刻回到纯参数合成
    void unload() {
        wait();
#ifdef VOX_WITH_ONNX
        releaseImpl();
#endif
        ready_.store(false);
        hasModel_.store(false);
    }

    // 波形 -> F0 轨迹（Hz，0 = 清音）。
    // 按契约：块 16384、50% 重叠；每块输出 T 帧，映射到帧移 661 样本的全局帧。
    // 返回 false 表示未使用 ONNX，调用方应保留自己的 DSP 分析结果。
    bool predictF0(const std::vector<float>& pcm, double fs, std::vector<float>& f0Hz) {
        (void)fs;
        f0Hz.clear();
        if (!ready_.load() || pcm.size() < 4) return false;
#ifdef VOX_WITH_ONNX
        if (!api_ || !sess_ || !mi_) return false;
        const size_t N = chunk_;
        const size_t hop = N / 2;
        const size_t FHOP = 661;                       // 契约帧移
        std::vector<float> acc, wsum;
        for (size_t s0 = 0; s0 < pcm.size(); s0 += hop) {
            std::vector<float> buf(N, 0.0f);
            for (size_t i = 0; i < N; ++i)
                buf[i] = (s0 + i < pcm.size()) ? pcm[s0 + i] : 0.0f;
            std::vector<float> out; std::vector<int64_t> shp;
            if (!runChunk(buf, shp, out)) return false;
            if (out.empty()) return false;
            const size_t T = out.size();
            const size_t base = (s0 + FHOP / 2) / FHOP; // 该块第 0 帧对应的全局帧号
            if (acc.size() < base + T) { acc.resize(base + T, 0.0f); wsum.resize(base + T, 0.0f); }
            for (size_t i = 0; i < T; ++i) { acc[base + i] += out[i]; wsum[base + i] += 1.0f; }
            if (s0 + N >= pcm.size()) break;
        }
        f0Hz.resize(acc.size());
        for (size_t i = 0; i < acc.size(); ++i) f0Hz[i] = (wsum[i] > 0.0f) ? acc[i] / wsum[i] : 0.0f;
        return true;
#else
        (void)pcm; return false;
#endif
    }

    // 就地后处理（润色 / 降噪 / 声码器）。
    // 返回 false 表示「本帧未使用 ONNX」，调用方照常使用 DSP 结果。
    bool process(std::vector<float>& pcm, double fs) {
        (void)fs;
        if (!ready_.load() || pcm.empty()) return false;
#ifdef VOX_WITH_ONNX
        return processImpl(pcm, fs);
#else
        (void)pcm;
        return false;
#endif
    }

private:
#ifdef VOX_WITH_ONNX
    // ---------------- ONNX Runtime C API 实现 ----------------
    // 按需加载 onnxruntime：不硬链接，未启用 ONNX 时进程里完全没有它
    void* openRuntime() {
        const char* env = std::getenv("VOX_ORT_PATH");
        const char* cands[6];
        int n = 0;
        cands[n++] = "libonnxruntime.so";                 // 安卓 app 原生库目录 / 系统搜索路径
        if (env && *env) cands[n++] = env;
        const char* home = std::getenv("HOME");
        static std::string s1, s2, s3;
        if (home) {
            s1 = std::string(home) + "/onnxdev/jni/arm64-v8a/libonnxruntime.so";
            s2 = std::string(home) + "/onnxdev/lib/libonnxruntime.so";
            cands[n++] = s1.c_str(); cands[n++] = s2.c_str();
        }
        cands[n++] = "/data/data/com.termux/files/usr/lib/libonnxruntime.so";
        for (int i = 0; i < n; ++i) {
            void* h = ::dlopen(cands[i], RTLD_NOW | RTLD_LOCAL);
            if (h) return h;
        }
        return nullptr;
    }

    bool loadImpl(const std::string& path) {
        lib_ = openRuntime();
        if (!lib_) {
            std::lock_guard<std::mutex> lk(m_);
            err_ = "找不到 libonnxruntime.so（未安装运行时）";
            return false;
        }
        typedef const OrtApiBase* (*GetBaseFn)(void);
        GetBaseFn getBase = (GetBaseFn)::dlsym(lib_, "OrtGetApiBase");
        if (!getBase) { std::lock_guard<std::mutex> lk(m_); err_ = "缺少 OrtGetApiBase 符号"; return false; }
        api_ = getBase()->GetApi(ORT_API_VERSION);
        if (!api_) { std::lock_guard<std::mutex> lk(m_); err_ = "无法获取 ONNX Runtime API"; return false; }
        if (api_->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "vox", &env_) != nullptr) {
            std::lock_guard<std::mutex> lk(m_); err_ = "创建环境失败"; return false; }
        if (api_->CreateSessionOptions(&so_) != nullptr) {
            std::lock_guard<std::mutex> lk(m_); err_ = "创建会话选项失败"; return false; }
        api_->SetIntraOpNumThreads(so_, 2);
        api_->SetSessionGraphOptimizationLevel(so_, ORT_ENABLE_ALL);
        OrtStatus* st = api_->CreateSession(env_, path.c_str(), so_, &sess_);
        if (st != nullptr) {
            std::lock_guard<std::mutex> lk(m_);
            err_ = std::string("模型载入失败: ") + api_->GetErrorMessage(st);
            api_->ReleaseStatus(st);
            return false;
        }
        OrtAllocator* alloc = nullptr;
        api_->GetAllocatorWithDefaultOptions(&alloc);
        char* nm = nullptr;
        if (api_->SessionGetInputName(sess_, 0, alloc, &nm) != nullptr) {
            std::lock_guard<std::mutex> lk(m_); err_ = "取输入名失败"; return false; }
        inName_ = nm; api_->AllocatorFree(alloc, nm);
        if (api_->SessionGetOutputName(sess_, 0, alloc, &nm) != nullptr) {
            std::lock_guard<std::mutex> lk(m_); err_ = "取输出名失败"; return false; }
        outName_ = nm; api_->AllocatorFree(alloc, nm);

        // 从模型输入形状推出分块长度（动态维则用默认 16384）
        OrtTypeInfo* ti = nullptr;
        if (api_->SessionGetInputTypeInfo(sess_, 0, &ti) == nullptr && ti) {
            const OrtTensorTypeAndShapeInfo* sh = nullptr;
            if (api_->CastTypeInfoToTensorInfo(ti, &sh) == nullptr && sh) {
                size_t nd = 0;
                if (api_->GetDimensionsCount(sh, &nd) == nullptr && nd > 0) {
                    std::vector<int64_t> d(nd);
                    if (api_->GetDimensions(sh, d.data(), nd) == nullptr) {
                        const int64_t last = d[nd - 1];
                        if (last > 0) chunk_ = (size_t)last;
                        size_t tot = 1;
                        for (size_t i = 0; i < nd; ++i) tot *= (d[i] > 0 ? (size_t)d[i] : 1);
                        hasModel_.store(true);
                        (void)tot;
                    }
                }
            }
            api_->ReleaseTypeInfo(ti);
        }
        api_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi_);
        std::lock_guard<std::mutex> lk(m_);
        err_.clear();
        return true;
    }

    // 单块推理：输入 N 个样本(float32)，输出展平的张量数据与形状
    bool runChunk(const std::vector<float>& buf, std::vector<int64_t>& shapeOut,
                  std::vector<float>& dataOut) {
        if (!api_ || !sess_ || !mi_) return false;
        const size_t N = buf.size();
        int64_t shape[3] = { 1, 1, (int64_t)N };
        OrtValue* in = nullptr;
        OrtStatus* st = api_->CreateTensorWithDataAsOrtValue(
            mi_, const_cast<float*>(buf.data()), N * sizeof(float), shape, 3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in);
        if (st != nullptr) { api_->ReleaseStatus(st); return false; }
        const char* inNames[1]  = { inName_.c_str() };
        const char* outNames[1] = { outName_.c_str() };
        OrtValue* ov = nullptr;
        st = api_->Run(sess_, nullptr, inNames,
                       (const OrtValue* const*)&in, 1, outNames, 1, &ov);
        api_->ReleaseValue(in);
        if (st != nullptr) {
            std::lock_guard<std::mutex> lk(m_);
            err_ = std::string("推理失败: ") + api_->GetErrorMessage(st);
            api_->ReleaseStatus(st);
            if (ov) api_->ReleaseValue(ov);
            return false;
        }
        float* od = nullptr;
        if (api_->GetTensorMutableData(ov, (void**)&od) != nullptr || !od) {
            api_->ReleaseValue(ov); return false;
        }
        shapeOut.clear();
        OrtTensorTypeAndShapeInfo* info = nullptr;
        size_t total = N;
        if (api_->GetTensorTypeAndShape(ov, &info) == nullptr && info) {
            size_t nd = 0;
            if (api_->GetDimensionsCount(info, &nd) == nullptr && nd > 0) {
                shapeOut.resize(nd);
                if (api_->GetDimensions(info, shapeOut.data(), nd) == nullptr) {
                    total = 1;
                    for (size_t i = 0; i < nd; ++i)
                        total *= (shapeOut[i] > 0 ? (size_t)shapeOut[i] : 1);
                }
            }
            api_->ReleaseTensorTypeAndShapeInfo(info);
        }
        dataOut.assign(od, od + total);
        api_->ReleaseValue(ov);
        return true;
    }

    // 50% 重叠分块 + 三角窗交叉淡化（避免块边界爆音）
    bool processImpl(std::vector<float>& pcm, double fs) {
        (void)fs;
        if (!api_ || !sess_ || !mi_) return false;
        const size_t N = chunk_;
        if (N < 4 || pcm.size() < 4) return false;
        const size_t hop = N / 2;
        std::vector<float> acc(pcm.size(), 0.0f), wsum(pcm.size(), 0.0f);
        std::vector<float> buf(N, 0.0f);

        for (size_t s = 0; s < pcm.size(); s += hop) {
            for (size_t i = 0; i < N; ++i) {
                const size_t j = s + i;
                buf[i] = (j < pcm.size()) ? pcm[j] : 0.0f;
            }
            std::vector<float> od; std::vector<int64_t> shp;
            if (!runChunk(buf, shp, od) || od.empty()) return false;
            for (size_t i = 0; i < N && i < od.size(); ++i) {
                const size_t j = s + i;
                if (j >= pcm.size()) break;
                float w = 1.0f - std::fabs(2.0f * (float)i / (float)(N - 1) - 1.0f);
                if (w < 1e-3f) w = 1e-3f;
                acc[j]  += od[i] * w;
                wsum[j] += w;
            }
            if (s + N >= pcm.size()) break;
        }
        for (size_t i = 0; i < pcm.size(); ++i)
            if (wsum[i] > 1e-6f) pcm[i] = acc[i] / wsum[i];
        return true;
    }

    void releaseImpl() {
        if (api_) {
            if (sess_) { api_->ReleaseSession(sess_); sess_ = nullptr; }
            if (so_)   { api_->ReleaseSessionOptions(so_); so_ = nullptr; }
            if (mi_)   { api_->ReleaseMemoryInfo(mi_); mi_ = nullptr; }
            if (env_)  { api_->ReleaseEnv(env_); env_ = nullptr; }
        }
        inName_.clear(); outName_.clear();
        api_ = nullptr;
        if (lib_) { ::dlclose(lib_); lib_ = nullptr; }   // 释放运行时，进程回到零占用
    }

    void* lib_{nullptr};
    const OrtApi* api_{nullptr};
    OrtEnv* env_{nullptr};
    OrtSessionOptions* so_{nullptr};
    OrtSession* sess_{nullptr};
    OrtMemoryInfo* mi_{nullptr};
    std::string inName_, outName_;
    size_t chunk_{16384};
#endif
    mutable std::mutex m_;
    std::string path_, err_;
    std::atomic<bool> ready_{false}, loading_{false}, hasModel_{false};
    std::thread worker_;
};

} // namespace vox
