// node examples/tiny-cnn.js > spec.json — a real forward + backward pass of a tiny CNN, written out as a spec.
// Weights are hand-built (edge kernels + class templates), not trained; every number in the spec is computed here.
const bmp = s => s.trim().split(/\s+/).flatMap(r => [...r].map(c => +(c === '#')));
const PROTO = {
  '0': bmp(`........ ..####.. .#....#. .#....#. .#....#. .#....#. ..####.. ........`),
  '1': bmp(`........ ...#.... ..##.... ...#.... ...#.... ...#.... ..###... ........`),
  '7': bmp(`........ .######. .....#.. ....#... ...#.... ...#.... ...#.... ........`),
};
const X = bmp(`........ ..#####. ......#. .....#.. ....#... ....#... ...#.... ........`);
const CLASSES = Object.keys(PROTO), TARGET = CLASSES.indexOf('7');
const K = [[1, 1, 1, 0, 0, 0, -1, -1, -1], [1, 0, -1, 1, 0, -1, 1, 0, -1]]; // horizontal edge, vertical edge

const conv = x => K.flatMap(k => Array.from({ length: 36 }, (_, i) => {
  const r = Math.floor(i / 6), c = i % 6; let s = 0;
  for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) s += k[a * 3 + b] * x[(r + a) * 8 + c + b];
  return s;
}));
const relu = x => x.map(v => Math.max(0, v));
const pool = x => Array.from({ length: 18 }, (_, i) => {
  const ch = Math.floor(i / 9), r = Math.floor((i % 9) / 3), c = i % 3, at = (a, b) => x[ch * 36 + (r * 2 + a) * 6 + c * 2 + b];
  return Math.max(at(0, 0), at(0, 1), at(1, 0), at(1, 1));
});
const feats = x => pool(relu(conv(x)));
const P = CLASSES.map(c => feats(PROTO[c])), mean = P[0].map((_, j) => (P[0][j] + P[1][j] + P[2][j]) / 3);
const Wt = P.map(p => p.map((v, j) => (v - mean[j]) * 0.15));
const fc = x => Wt.map(w => w.reduce((s, v, j) => s + v * x[j], 0));
const softmax = x => { const m = Math.max(...x), e = x.map(v => Math.exp(v - m)), s = e.reduce((a, b) => a + b); return e.map(v => v / s); };
const ce = p => [-Math.log(p[TARGET])];

const L = [conv, relu, pool, x => x, fc, softmax, ce];
const acts = [X]; for (const fn of L) acts.push(fn(acts.at(-1)));
const lossFrom = (i, a) => L.slice(i).reduce((v, fn) => fn(v), a)[0];
// central finite differences: d(loss)/d(acts[i])
const grad = i => acts[i].map((_, j) => { const a = [...acts[i]], e = 1e-4; a[j] += e; const up = lossFrom(i, a); a[j] -= 2 * e; return (up - lossFrom(i, a)) / (2 * e); });

const r = (a, n = 3) => a.map(v => +v.toFixed(n));
const T = (i, shape, extra) => ({ out: { shape, data: r(acts[i]), ...extra }, grad: { shape, data: r(grad(i), 4) } });
const [, , , , , logits, probs, loss] = acts;
const pred = CLASSES[probs.indexOf(Math.max(...probs))], pct = v => (v * 100).toFixed(1) + '%';
if (pred !== '7') throw new Error('demo broke: predicted ' + pred);

console.log(JSON.stringify({
  title: 'TinyCNN 数据流',
  summary: '一个最小的卷积分类器：两个边缘检测卷积核，加一个全连接层，把 8×8 的手写图案分成 0、1、7 三类。权重是手工设定的，没有经过训练。',
  example: '一张 8×8 的手写“7”（和模板相比向右偏了一格），正确答案是“7”。',
  levels: ['整体', '两个阶段', '每一层'],
  root: { name: 'root',
    children: [
      { name: '输入图像', type: 'Input', ...T(0, [8, 8]),
        note: '1 表示有笔画，0 表示空白。下面的梯度图显示哪些像素一变，损失就跟着变，也就是模型判断时最在意的位置。' },
      { name: 'TinyCNN', type: 'Model', desc: '整个模型：先提取边缘特征，再根据特征打分。', out: T(6, [3], { labels: CLASSES }).out,
        note: `模型认为这张图是“${pred}”，把握 ${pct(probs[TARGET])}。`,
        children: [
          { name: '特征提取', type: 'Sequential', desc: '把像素变成“哪里有横边、哪里有竖边”的粗略地图。',
            note: '64 个像素被压缩成 18 个数：2 种边缘 × 3×3 个区域。',
            children: [
              { name: 'conv', type: 'Conv2d', ...T(1, [2, 6, 6]),
                note: '第 1 张图是横向边缘检测的结果：顶上那一横的上沿是负值，下沿是正值。第 2 张图是竖向边缘：笔画左侧为负，右侧为正。3×3 的窗口不补边，所以 8×8 变成 6×6。' },
              { name: 'relu', type: 'ReLU', ...T(2, [2, 6, 6]), note: '所有负值都被清零，只留下“上亮下暗”和“左亮右暗”这两种方向的边缘。' },
              { name: 'pool', type: 'MaxPool2d', ...T(3, [2, 3, 3]), note: '每 2×2 取最大值，6×6 变成 3×3。图案偏移一格时，池化后的结果变化不大，这就是它能容忍位置偏移的原因。' },
            ] },
          { name: '分类头', type: 'Sequential', desc: '根据特征给每个类别打分，再把分数变成概率。',
            children: [
              { name: 'flatten', type: 'Flatten', ...T(4, [18]), note: '2×3×3 排成 18 个数，数值和上一步完全相同。' },
              { name: 'fc', type: 'Linear', ...T(5, [3], { labels: CLASSES }),
                note: `三个类别各得一个分数：${CLASSES.map((c, i) => `“${c}” ${logits[i].toFixed(2)}`).join('，')}。这里的权重是每个类别模板的特征（减去平均值），所以特征越像哪个模板，哪个分数越高。` },
              { name: 'softmax', type: 'Softmax', ...T(6, [3], { labels: CLASSES }),
                note: `分数变成概率：${CLASSES.map((c, i) => `“${c}” ${pct(probs[i])}`).join('，')}。` },
            ] },
        ] },
      { name: 'loss', type: 'CrossEntropyLoss', out: { shape: [], data: r(loss, 4) },
        note: `正确类别“7”的概率是 ${pct(probs[TARGET])}，损失 = −ln(${probs[TARGET].toFixed(3)}) = ${loss[0].toFixed(3)}。概率越接近 100%，损失越接近 0。` },
    ] },
}));
