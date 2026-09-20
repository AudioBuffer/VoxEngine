# VoxEngine

**全本地、离线的 C++ 歌声合成引擎（SVS）**，面向 Termux / Android ARM64。

不依赖任何在线服务：核心是物理声学参数的 DSP 合成，ONNX 模型为**可选**的润色/增强层，
没有模型时自动回落到纯参数合成，不影响任何功能。

## 特性

| 模块 | 说明 |
|---|---|
| 声学骨架 | 声门源用 **LF 模型**（波表、去直流、峰值归一化）+ jitter/shimmer + 延迟颤音 |
| 声道 | 5 共振峰 **全极点级联** + **Klatt 式并联高频支路**（补回级联固有的过陡滚降） |
| 辅音 | 带通噪声源：擦音 / 塞擦音 / 爆破音（成阻期真正静音 + 平滑释放防爆音） |
| 音素 | **中英日共 110 个**（汉语拼音 / 英语 ARPABET / 日语罗马字），CV 音节自动拆分 |
| 连续发音 | 整段交给同一引擎连续渲染，声门相位与声道状态跨音符保持；音高 smoothstep 滑音 |
| 编辑器 | X11 GTK3 卷帘：多轨、每轨独立参数、中键框选、Ctrl+C/V、多轨全览 |
| 声库 | 内置男声/女声/童声；弹出侧栏选择 |
| ONNX | 可选润色 + 基频预测，**按需 dlopen**，未启用时零开销 |

## 构建

### Termux GUI（X11）

```bash
# 依赖
pkg install clang make gtk3 pkg-config

# 主界面
clang++ -O2 -std=c++17 -Isrc -o bin/vox_gui src/gui_main.cpp \
  $(pkg-config --cflags --libs gtk+-3.0) -lm

# 自检（33 项断言，不需要开窗口）
./bin/vox_gui --selftest
```

### 全自动训练流水线

```bash
clang++ -O2 -std=c++17 -Isrc -o bin/vox_train src/train_main.cpp \
  $(pkg-config --cflags --libs gtk+-3.0) -lm -pthread
```

### 安卓 APK（不需要 SDK/NDK）

Termux 本身就是 aarch64-android，用 `clang++ → javac → d8 → aapt2 → apksigner` 直接打包：

```bash
~/apkbuild/build.sh
```

需要 `android.jar`（API 34）与打包脚本，见 `~/apkbuild/`。

## ONNX（可选）

启用真实推理需要两步（**不启用也能跑，走纯参数合成**）：

1. 头文件：从 ONNX Runtime AAR 取 `headers/onnxruntime_c_api.h`
2. 运行时：`libonnxruntime.so`（arm64-v8a）

```bash
# 放在 ~/onnxdev/ 下即可被自动找到（也支持 VOX_ORT_PATH 环境变量）
unzip ort.aar 'headers/*' 'jni/arm64-v8a/*' -d ~/onnxdev

clang++ -O2 -std=c++17 -DVOX_WITH_ONNX -Isrc -I$HOME/onnxdev/headers \
  -o bin/vox_gui src/gui_main.cpp $(pkg-config --cflags --libs gtk+-3.0) -lm
```

**按需加载**：onnxruntime 通过 `dlopen` 加载，未启用 ONNX 的进程里完全没有它，
零启动开销、零内存占用（这是设计决定，见 `src/onnx_engine.h` 顶部注释）。

模型契约见 [`docs/MODEL_CONTRACT.md`](docs/MODEL_CONTRACT.md)，训练方案见 [`docs/TRAINING.md`](docs/TRAINING.md)。

## 训练（全自动）

`bin/vox_train` 提供图形界面的全自动流水线：

1. 选「歌唱模板文件夹」+「说话样本文件夹」
2. 点「① 全自动处理」

在校验通过后自动：提取音高 → 造成对数据 → 写 `.npy` → 生成 `train.py`。

### 输入校验规则

| 规则 | 行为 |
|---|---|
| 目录内有非 WAV 文件 | **拒绝**，并列出违规文件 |
| 目录内有 MP3 | **弹窗确认**，确认后 ffmpeg 转 44100Hz 单声道 WAV 再继续 |
| 歌唱模板总时长 < 60 分钟 | **拒绝**，报出实际时长与差额（阈值可调） |
| 说话样本总时长 < 30 分钟 | **拒绝**，同上 |

### 成对数据的来源

引擎是**参数驱动**的，所以把从录音提取的 F0 喂给引擎，
合成结果与录音**逐帧对齐**，直接构成 `(引擎输出, 真人录音)` 训练对。

## 目录结构

```
README.md           项目愿景与技术路线
BUILD.md            构建与开发指南（本文件）
LICENSE             MIT
docs/
  MODEL_CONTRACT.md  ONNX 模型张量契约（训练必须严格对齐）
  TRAINING.md        训练方案（数据、提音高、切块、导出）
  REQUIREMENTS.md    需求演进记录（含每轮实测数据与失败复盘）
src/                引擎与界面源码（头文件实现为主）
  lf_glottal.h        LF 声门源
  kl_tract.h          声道（级联 + Klatt 并联高频支路）
  noise_source.h      辅音噪声源
  vox_engine.h        引擎主类
  phonemes.h          中英日音素总表（110 个）
  f0_track.h          YIN 基频跟踪
  wav_reader.h        WAV 读取 + 重采样
  wav_writer.h        WAV 写出
  onnx_engine.h       ONNX 推理接口（按需 dlopen + 降级）
  auto_pipeline.h     全自动训练流水线核心
  gui_main.cpp        X11 GUI（主界面）
  train_main.cpp      训练流水线 GUI
  tui_main.cpp        终端 UI
  main.cpp            命令行测试入口
  seq.h               序列渲染
js/                 JS 插件层（QuickJS）
tools/              数据集准备与模型校验
  make_dataset.py     训练机上生成数据集
  onnx_check.cpp      本机验证模型能否加载推理
```

## 已知限制

- **英文单词**（如 `hello`）查不到，需要 G2P 词典；目前请直接输入音素（`hh eh l ow`）
- 训练必须在 PC / Colab（本机无 PyTorch）
- 安卓端布局尚未适配新编辑器功能（多轨、每轨参数等目前仅 X11 GUI）

## 版本历史与需求记录

需求演进记录见 [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md)（含每轮的实测数据与失败复盘）。
