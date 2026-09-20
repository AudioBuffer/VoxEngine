// VoxEngine 渲染队列：限制并发进程数，防止手机端 OOM
// 用法: node js/render_queue.mjs vox_demo "a 1.0 220 out1.wav" "i 1.0 260 out2.wav" ...
import { spawn } from 'node:child_process';

const BIN        = process.env.VOX_BIN || './build/vox_demo';
const CONCURRENCY= Number(process.env.VOX_JOBS  || 2);   // 双核 -> 默认 2
const MAX_QUEUE  = Number(process.env.VOX_QUEUE || 64);  // 队列上限，超出直接拒绝
const TIMEOUT_MS = Number(process.env.VOX_TIMEOUT_MS || 120000);

export class RenderQueue {
  constructor({ bin = BIN, concurrency = CONCURRENCY, maxQueue = MAX_QUEUE } = {}) {
    this.bin = bin;
    this.concurrency = Math.max(1, concurrency);
    this.maxQueue = maxQueue;
    this.running = 0;
    this.queue = [];
    this.done = 0;
    this.failed = 0;
  }

  get pending() { return this.queue.length; }
  get load()    { return this.running / this.concurrency; }

  push(args, { timeout = TIMEOUT_MS } = {}) {
    if (this.queue.length >= this.maxQueue) {
      return Promise.reject(new Error('queue full: ' + this.queue.length + '/' + this.maxQueue));
    }
    return new Promise((resolve, reject) => {
      this.queue.push({ args, timeout, resolve, reject });
      this.#drain();
    });
  }

  #drain() {
    while (this.running < this.concurrency && this.queue.length > 0) {
      const job = this.queue.shift();
      this.running++;
      this.#run(job)
        .then(job.resolve, job.reject)
        .finally(() => { this.running--; this.#drain(); });
    }
  }

  #run({ args, timeout }) {
    return new Promise((resolve, reject) => {
      const argv = Array.isArray(args) ? args : String(args).split(/\s+/).filter(Boolean);
      const child = spawn(this.bin, argv, { stdio: ['ignore', 'pipe', 'pipe'] });
      let out = '', err = '', killed = false;
      const timer = setTimeout(() => {
        killed = true;
        child.kill('SIGKILL');
        reject(new Error('timeout after ' + timeout + 'ms: ' + argv.join(' ')));
      }, timeout);

      child.stdout.on('data', d => { out += d; });
      child.stderr.on('data', d => { err += d; });
      child.on('error', e => { clearTimeout(timer); if (!killed) reject(e); });
      child.on('close', code => {
        clearTimeout(timer);
        if (killed) return;
        if (code === 0) { this.done++; resolve({ args: argv, stdout: out.trim() }); }
        else { this.failed++; reject(new Error('exit ' + code + ': ' + err.trim())); }
      });
    });
  }

  // 等待队列清空
  async drainAll() {
    while (this.running > 0 || this.queue.length > 0) {
      await new Promise(r => setTimeout(r, 50));
    }
    return { done: this.done, failed: this.failed };
  }
}

// ---- 直接运行时：演示 20 个请求，验证并发不会崩 ----
if (import.meta.url === 'file://' + process.argv[1]) {
  const q = new RenderQueue({ concurrency: 2, maxQueue: 64 });
  const jobs = [];
  for (let i = 0; i < 20; i++) {
    const vowel = ['a','o','e','i','u','v'][i % 6];
    jobs.push(
      q.push(vowel + ' 0.5 ' + (180 + i * 5) + ' /tmp/vox_' + i + '.wav')
        .then(() => 'ok ' + i)
        .catch(e => 'err ' + i + ': ' + e.message)
    );
  }
  const res = await Promise.all(jobs);
  const stats = await q.drainAll();
  console.log(res.join('\n'));
  console.log('---- 完成 ' + stats.done + ' 失败 ' + stats.failed + ' ----');
}
