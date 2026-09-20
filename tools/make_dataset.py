#!/usr/bin/env python3
"""把 WAV 切成 VoxEngine 模型契约要求的训练张量（在训练机上跑，需要 numpy+librosa）

用法:
  python3 make_dataset.py out_dir  a.wav  b.wav ...
  python3 make_dataset.py out_dir  --fmin 100 --fmax 1000  *.wav   # 女声

产出:
  out_dir/X.npy  float32 (N, 1, 16384)   输入波形块
  out_dir/Y.npy  float32 (N, 23)         每块 23 帧 F0(Hz)，0 表示清音
与 MODEL_CONTRACT.md 严格一致：44100Hz 单声道，块 16384，步进 8192，
帧长 1323、帧移 661、T=23。
"""
import sys, os, glob
import numpy as np
import librosa

SR, CHUNK, HOP = 44100, 16384, 8192
FLEN, FHOP, T = 1323, 661, 23

def extract_f0(y, fmin, fmax):
    """librosa.pyin 提取 F0，再做引擎同款后处理"""
    f0, voiced, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=SR,
                                 frame_length=FLEN, hop_length=FHOP)
    f0 = np.nan_to_num(f0, nan=0.0)
    f0[~voiced] = 0.0
    # 1) 5 点中值滤波（去掉孤立跳变）
    if len(f0) >= 5:
        from scipy.signal import medfilt
        f0 = medfilt(f0, 5)
    # 2) 半频/倍频纠正
    for i in range(len(f0)):
        if f0[i] <= 0: continue
        for div in (2, 3):
            cand = f0[i] / div
            if cand < fmin: break
            if i > 0 and f0[i-1] > 0 and abs(f0[i-1]-cand)/cand < 0.06:
                f0[i] = cand; break
    # 3) 短空隙插值
    idx = np.where(f0 > 0)[0]
    if len(idx) > 1:
        f0 = np.interp(np.arange(len(f0)), idx, f0[idx])
    return f0

def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    out_dir = sys.argv[1]
    args = sys.argv[2:]
    fmin, fmax = 80.0, 800.0
    if args and args[0] == "--fmin":
        fmin, fmax = float(args[1]), float(args[2]); args = args[3:]
    files = []
    for a in args:
        files += glob.glob(a)
    if not files:
        print("没有找到输入 WAV"); sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)

    X, Y = [], []
    for path in files:
        y, sr = librosa.load(path, sr=SR, mono=True)
        if len(y) < CHUNK: 
            print(f"  跳过(太短) {path}"); continue
        f0 = extract_f0(y, fmin, fmax)
        n = 0
        for s in range(0, len(y) - CHUNK + 1, HOP):
            seg = y[s:s+CHUNK]
            # 第 i 帧 = 样本区间 [661*i, 661*i+1323) 相对块首
            lab = np.zeros(T, dtype=np.float32)
            for i in range(T):
                c = (s + FHOP*i + FLEN//2) // FHOP      # 块内第 i 帧中心 → 全局帧号
                if 0 <= c < len(f0): lab[i] = f0[c]
            X.append(seg.reshape(1, CHUNK)); Y.append(lab); n += 1
        print(f"  {os.path.basename(path)}: {n} 块")
    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.float32)
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "Y.npy"), Y)
    print(f"\n完成: X={X.shape}  Y={Y.shape}  → {out_dir}/")
    print(f"消音比例: {float((Y==0).mean()):.1%}（清音帧占比，用于定 loss 权重）")

if __name__ == "__main__":
    main()
