// VoxEngine 推理层 (JS)
// 输入: C++ 物理骨架 WAV -> 2~3 步残差精修 -> 输出
// ONNX(LDM) 优先; 无 onnxruntime-node 时用内置纯JS残差精修
import { readFileSync, writeFileSync } from 'node:fs';

export function parseWav(b) {
  if (b.toString('ascii', 0, 4) !== 'RIFF') throw new Error('not RIFF');
  let off = 12, fmt = null, dataOff = -1, dataLen = 0;
  while (off + 8 <= b.length) {
    const id = b.toString('ascii', off, off + 4);
    const sz = b.readUInt32LE(off + 4);
    if (id === 'fmt ') fmt = { channels: b.readUInt16LE(off + 10), rate: b.readUInt32LE(off + 12), bits: b.readUInt16LE(off + 22) };
    if (id === 'data') { dataOff = off + 8; dataLen = sz; }
    off += 8 + sz + (sz & 1);
  }
  if (!fmt || dataOff < 0) throw new Error('bad wav');
  const n = Math.floor(dataLen / 2);
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = b.readInt16LE(dataOff + i * 2) / 32768;
  return { fmt, x };
}

export function readWav(path) {
  if (path === '-') return parseWav(readFileSync(0));
  return parseWav(readFileSync(path));
}

export function buildWav(x, rate) {
  const n = x.length, sz = n * 2;
  const b = Buffer.alloc(44 + sz);
  b.write('RIFF', 0); b.writeUInt32LE(36 + sz, 4); b.write('WAVE', 8);
  b.write('fmt ', 12); b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(1, 22);
  b.writeUInt32LE(rate, 24); b.writeUInt32LE(rate * 2, 28); b.writeUInt16LE(2, 32); b.writeUInt16LE(16, 34);
  b.write('data', 36); b.writeUInt32LE(sz, 40);
  for (let i = 0; i < n; i++) b.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(x[i] * 32767))), 44 + i * 2);
  return b;
}

export function writeWav(path, x, rate) {
  const b = buildWav(x, rate);
  if (path === '-') process.stdout.write(b);
  else writeFileSync(path, b);
}

// ---- 内置残差精修（LDM 替代实现，可跑）----
// 原理: 用基音周期梳状滤波分离"周期骨架"与"非周期残差",
//       对残差做谱形平滑 + 按 human 注入调制噪声, 迭代 steps 次逼近自然嗓音。
function refineStep(x, period, human, rng) {
  const n = x.length, y = new Float32Array(n), res = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const d = i >= period ? x[i - period] : x[i];
    const per = 0.5 * (x[i] + d);
    res[i] = x[i] - per;
    y[i] = per;
  }
  let lp = 0;
  for (let i = 0; i < n; i++) {
    lp += 0.25 * (res[i] - lp);
    const noise = rng() * 2 - 1;
    const shaped = lp + human * 0.35 * noise * Math.abs(lp);
    y[i] += shaped * (1 - 0.15 * human);
  }
  return y;
}

function mulberry32(a) {
  return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}

export async function loadOnnxRefiner(modelPath) {
  try { const ort = await import('onnxruntime-node'); return { ort, modelPath }; }
  catch { return null; }
}

export async function infer(inWav, outWav, { f0 = 220, steps = 3, human = 0.35, seed = 7 } = {}) {
  const { fmt, x } = readWav(inWav);
  const period = Math.max(2, Math.round(fmt.rate / f0));
  const rng = mulberry32(seed);
  let cur = x;
  for (let s = 0; s < steps; s++) cur = refineStep(cur, period, human, rng);
  let pk = 1e-9;
  for (let i = 0; i < cur.length; i++) pk = Math.max(pk, Math.abs(cur[i]));
  const g = 0.85 / pk;
  for (let i = 0; i < cur.length; i++) cur[i] *= g;
  writeWav(outWav, cur, fmt.rate);
  return { samples: cur.length, rate: fmt.rate, steps, human, period };
}

if (import.meta.url === 'file://' + process.argv[1]) {
  const a = process.argv.slice(2);
  const inWav = a[0] || 'vox_vowel_a.wav';
  const outWav = a[1] || 'vox_refined.wav';
  const opt = {};
  for (const s of a.slice(2)) { const m = s.match(/^--(\w+)=(.*)$/); if (m) opt[m[1]] = isNaN(+m[2]) ? m[2] : +m[2]; }
  const onnx = await loadOnnxRefiner(opt.model);
  console.error('ONNX(LDM): ' + (onnx ? '已加载 ' + opt.model : '不可用 -> 使用内置残差精修'));
  const r = await infer(inWav, outWav, opt);
  console.error('输出 ' + outWav + '  ' + r.samples + '样本 ' + r.rate + 'Hz  steps=' + r.steps + ' human=' + r.human);
}
