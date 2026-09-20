// VoxEngine TUI —— 纯终端界面，零依赖
// 用法: node js/tui.mjs          交互调参
//       node js/tui.mjs --demo   非交互跑一次全链路(自检)
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';

const BIN   = process.env.VOX_BIN || './build/vox_demo';
const OUT   = process.env.VOX_OUT || 'tui_out.wav';
const REF   = 'tui_refined.wav';

const P = [
  { k:'vowel',  label:'元音',      kind:'enum', vals:['a','o','e','i','u','v'], i:0 },
  { k:'f0',     label:'基频 F0',   min:80,  max:400, step:5,     v:220,  unit:'Hz' },
  { k:'dur',    label:'时长',      min:0.3, max:5,   step:0.1,   v:1.0,  unit:'s'  },
  { k:'voice',  label:'声线缩放',  min:0.8, max:1.4, step:0.02,  v:1.0,  unit:'x'  },
  { k:'f1',     label:'F1',        min:200, max:1100,step:10,    v:800,  unit:'Hz' },
  { k:'f2',     label:'F2',        min:600, max:2600,step:20,    v:1200, unit:'Hz' },
  { k:'f3',     label:'F3',        min:1800,max:3400,step:20,    v:2500, unit:'Hz' },
  { k:'f4',     label:'F4 空气感', min:2800,max:4200,step:20,    v:3500, unit:'Hz' },
  { k:'f5',     label:'F5 空气感', min:3800,max:5200,step:20,    v:4500, unit:'Hz' },
  { k:'breath', label:'气息感',    min:0,   max:1,   step:0.02,  v:0.12, unit:''   },
  { k:'vib',    label:'颤音深度',  min:0,   max:0.05,step:0.002, v:0.008,unit:''   },
  { k:'jitter', label:'Jitter',    min:0,   max:0.03,step:0.002, v:0.004,unit:''   },
  { k:'shimmer',label:'Shimmer',   min:0,   max:0.15,step:0.01,  v:0.02, unit:''   },
  { k:'nasal',  label:'鼻音耦合',  min:0,   max:1,   step:0.05,  v:0.0,  unit:''   },
  { k:'cons',   label:'辅音',      kind:'enum', vals:['无','s','sh','f','p','b','m'], i:0 },
  { k:'steps',  label:'LDM 步数',  min:0,   max:6,   step:1,     v:3,    unit:''   },
  { k:'human',  label:'人味',      min:0,   max:1,   step:0.05,  v:0.35, unit:''   },
];

const get = k => { const p = P.find(x => x.k === k); return p.kind === 'enum' ? p.vals[p.i] : p.v; };
const set = (k, d) => {
  const p = P.find(x => x.k === k);
  if (p.kind === 'enum') { p.i = (p.i + d + p.vals.length) % p.vals.length; return; }
  let v = p.v + d * p.step;
  v = Math.min(p.max, Math.max(p.min, v));
  p.v = Math.round(v * 1e6) / 1e6;
};

function run(cmd, args) {
  return new Promise((res, rej) => {
    const c = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let o = '', e = '';
    c.stdout.on('data', d => o += d);
    c.stderr.on('data', d => e += d);
    c.on('error', rej);
    c.on('close', code => code === 0 ? res(o) : rej(new Error(cmd + ' exit ' + code + '\n' + e)));
  });
}

const CONS = { '无':0, s:1, sh:2, f:3, p:4, b:5, m:6 };

export async function render() {
  const args = [
    'out=' + OUT,
    'vowel=' + get('vowel'), 'dur=' + get('dur'), 'f0=' + get('f0'),
    'f1=' + get('f1'), 'f2=' + get('f2'), 'f3=' + get('f3'),
    'f4=' + get('f4'), 'f5=' + get('f5'),
    'voice=' + get('voice'), 'breath=' + get('breath'), 'vib=' + get('vib'),
    'jitter=' + get('jitter'), 'shimmer=' + get('shimmer'), 'nasal=' + get('nasal'),
    'cons=' + CONS[get('cons')],
  ];
  const eng = await run(BIN, args);
  const steps = get('steps') | 0;
  let inf = '';
  if (steps > 0) {
    inf = await run('node', ['js/infer.mjs', OUT, REF, '--steps=' + steps, '--human=' + get('human'), '--f0=' + get('f0')]);
  }
  return { eng, inf, file: steps > 0 ? REF : OUT };
}

// ---------------- 非交互自检 ----------------
if (process.argv.includes('--demo')) {
  console.log('TUI 自检: 调参数 -> 渲染 -> 推理');
  set('vowel', 1); set('f0', -8); set('breath', 4); set('cons', 1);
  const r = await render();
  console.log(r.eng.trim());
  if (r.inf) console.log(r.inf.trim());
  console.log('产物: ' + r.file + (existsSync(r.file) ? '  OK' : '  缺失!'));
  process.exit(0);
}

// ---------------- 交互界面 ----------------
let sel = 0;
const esc = s => '\x1b[' + s;

function draw(status) {
  const out = [];
  out.push(esc('2J') + esc('H'));
  out.push('\x1b[1;36m VoxEngine TUI \x1b[0m  ↑↓ 选择   ←→ 调节   r 渲染   p 播放   q 退出\n');
  out.push('\x1b[90m' + '─'.repeat(46) + '\x1b[0m');
  P.forEach((p, i) => {
    const on = i === sel;
    const mark = on ? '\x1b[7m>' : ' ';
    const val = p.kind === 'enum' ? p.vals[p.i]
              : (p.v === Math.round(p.v) ? p.v : p.v.toFixed(3).replace(/0+$/, '').replace(/\.$/, ''));
    let bar = '';
    if (p.kind !== 'enum') {
      const w = 18;
      const t = (p.v - p.min) / (p.max - p.min);
      const n = Math.round(t * w);
      bar = ' [' + '█'.repeat(n) + '·'.repeat(w - n) + ']';
    }
    out.push(mark + ' ' + p.label.padEnd(12) + String(val).padStart(8) + (p.unit || '').padEnd(3) + bar + (on ? '\x1b[0m' : ''));
  });
  out.push('\x1b[90m' + '─'.repeat(46) + '\x1b[0m');
  out.push(status || '\x1b[90m就绪\x1b[0m');
  process.stdout.write(out.join('\n'));
}

async function act(key) {
  if (key === 'q' || key === '\x03') { process.stdout.write('\x1b[2J\x1b[H'); process.exit(0); }
  if (key === 'r') {
    draw('\x1b[33m渲染中…\x1b[0m');
    try { const r = await render(); draw('\x1b[32m完成 -> ' + r.file + '\x1b[0m  (按 p 播放)'); }
    catch (e) { draw('\x1b[31m失败: ' + String(e.message).slice(0, 60) + '\x1b[0m'); }
    return;
  }
  if (key === 'p') {
    const f = existsSync(REF) ? REF : OUT;
    spawn('mpv', ['--no-video', '--really-quiet', f], { detached: true, stdio: 'ignore' }).unref();
    draw('\x1b[36m播放 ' + f + '\x1b[0m');
    return;
  }
  draw();
}

if (!process.stdin.isTTY) { console.log('非 TTY：请用 node js/tui.mjs --demo'); process.exit(0); }
process.stdin.setRawMode(true);
process.stdin.resume();
process.stdin.setEncoding('utf8');
draw();
process.stdin.on('data', async (k) => {
  if (k === '\x1b[A') { sel = (sel - 1 + P.length) % P.length; draw(); }
  else if (k === '\x1b[B') { sel = (sel + 1) % P.length; draw(); }
  else if (k === '\x1b[C') { set(P[sel].k, 1); draw(); }
  else if (k === '\x1b[D') { set(P[sel].k, -1); draw(); }
  else await act(k);
});
