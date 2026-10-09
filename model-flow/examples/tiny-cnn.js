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

// A picture drawn from the same arrays, as an SVG data URI: the input beside the three templates the weights were built from.
const grid = (px, x0, title) => `<text x="${x0 + 36}" y="11" text-anchor="middle" font-size="10" fill="#1b2a41">${title}</text>` +
  px.map((v, i) => `<rect x="${x0 + (i % 8) * 9}" y="${16 + Math.floor(i / 8) * 9}" width="9" height="9" fill="${v ? '#1b2a41' : '#fff'}" stroke="#d6dae0" stroke-width=".5"/>`).join('');
const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 372 94" width="372" height="94" font-family="sans-serif"><rect width="372" height="94" fill="#fff"/>` +
  grid(X, 6, '输入') + CLASSES.map((c, k) => grid(PROTO[c], 102 + k * 90, `模板“${c}”`)).join('') + `</svg>`;
const TEMPLATES = { image: 'data:image/svg+xml;base64,' + Buffer.from(svg).toString('base64'), caption: '左边是输入；右边三张是三个类别的模板。输入是一个“7”，但比模板向右偏了一格。' };
const FIXED = '**不是模型的可学习部分**，是固定的计算。';

console.log(JSON.stringify({
  title: 'TinyCNN 数据流',
  summary: '一个最小的卷积分类器，把 8×8 的图案分成 0、1、7 三类。**权重是手工写的，没有训练**：用来看清每一步在算什么。',
  example: `一张 8×8 的手写“7”，比模板向右偏了一格。模型判 **“${pred}”，把握 ${pct(probs[TARGET])}**。`,
  stats: [['模型的判断', `“${pred}” ✓`, '正确答案 7'], ['把握', pct(probs[TARGET]), ''], ['权重', '72 个', '卷积 18 + 全连接 54，手工设定'], ['损失', loss[0].toFixed(3), '']],
  background: [
    ['任务：**看一张 8×8 的黑白图案，说出它是 0、1 还是 7。**', '- 一个刻意做小的例子', '- 小到每个数都可以手算'],
    ['进去和出来：', '- 进去：64 个格子，1 是笔画，0 是空白', '- 出来：3 个概率，**最大的就是模型的答案**'],
    ['模型的想法：**先找边缘，再看这些边缘像哪个模板。**', '- 卷积：找横边和竖边', '- 池化：容忍一点位置偏移', '- 全连接：和三个模板比'],
    ['和真正的模型的区别：', '- 权重是手工写的，不是训练出来的', '- 这个示例用 JavaScript 写，没有用任何深度学习框架', '- 梯度是用“改一点点、看损失变多少”的办法算的'],
  ],
  concepts: [
    { name: '边缘特征', aliases: ['边缘', '特征'], short: '卷积把“像素”变成**“哪里有横边、哪里有竖边”**。',
      table: { head: ['第几张图', '卷积核找什么', '正值表示'], rows: [['第 1 张', '横向的边缘', '上亮下暗'], ['第 2 张', '竖向的边缘', '左亮右暗']] },
      text: ['- 一个卷积核是 3×3 的 9 个数，在图上滑动', '- ReLU 之后只留下正值', '- 池化之后每张图只剩 3×3：位置变粗了，但偏一格也认得出'] },
    { name: '模板', aliases: ['模板'], short: '全连接层的权重是**三个类别的“标准写法”的边缘特征**。', figure: TEMPLATES,
      text: ['- 先把三个模板各自过一遍卷积、ReLU、池化', '- 每个类别的权重 = 它的模板特征减去三者的平均', '- 所以输入的特征越像哪个模板，哪个分数越高'] },
  ],
  glossary: { '像素': '图上的一个小格。', '损失': '模型错了多少，用一个数表示，越小越好。' },
  levels: ['整体', '两个阶段', '每一层'],
  root: { name: 'root',
    children: [
      { name: '输入图像', type: 'Input', ...T(0, [8, 8]), desc: ['一张 8×8 的黑白图案。', '- 1：有笔画', '- 0：空白'],
        note: '下面的梯度图显示哪些像素一变，损失就跟着变：也就是模型判断时最在意的位置。' },
      { name: 'TinyCNN', type: 'Model', origin_kind: 'standard', figure: TEMPLATES, out: T(6, [3], { labels: CLASSES }).out,
        desc: ['整个模型。', '- 特征提取：从像素里找边缘', '- 分类头：根据边缘打分'],
        why: '直接比像素，图案偏一格就全对不上。**先变成边缘特征再比，就稳得多。**',
        origin: ['**最小的卷积网络**：卷积、激活、池化、全连接。', '- 结构和真正的图像分类网络一样，只是每样只有一层', '- 权重是手工写的，没有训练'],
        note: `模型认为这张图是 **“${pred}”**，把握 ${pct(probs[TARGET])}。`,
        children: [
          { name: '特征提取', type: 'Sequential', origin_kind: 'standard', mini: 'conv', desc: ['把像素变成一张粗略的“边缘地图”。', '- 卷积：找横边和竖边', '- ReLU：只留正的', '- 池化：缩小'],
            why: '**判断是哪个数字，靠的是笔画的形状，不是某个格子亮不亮。**', origin: '“卷积 → 激活 → 池化”是**卷积网络的标准开头**。',
            note: '64 个像素变成 18 个数：2 种边缘 × 3×3 个区域。',
            children: [
              { name: 'conv', type: 'Conv2d', params: 18, ...T(1, [2, 6, 6]), desc: ['两个 3×3 的卷积核，各自扫过整张图。', '- 第 1 个找横向的边缘', '- 第 2 个找竖向的边缘'],
                origin: '卷积是标准做法；这里的 18 个权重是**手工写的边缘检测核**，不是训练出来的。',
                terms: { '卷积核': '一小块权重（这里 3×3）。它和图上哪一块长得像，那里的输出就大。' },
                note: '第 1 张图：顶上那一横的上沿是负值，下沿是正值。第 2 张图：笔画左侧为负，右侧为正。不补边，所以 8×8 变成 6×6。' },
              { name: 'relu', type: 'ReLU', params: 0, ...T(2, [2, 6, 6]), note: '负值全部清零，只留下“上亮下暗”和“左亮右暗”这两种方向的边缘。' },
              { name: 'pool', type: 'MaxPool2d', params: 0, ...T(3, [2, 3, 3]), why: '图案偏一格，池化后的结果变化不大。**这就是它能容忍位置偏移的原因。**', note: '每 2×2 取最大值，6×6 变成 3×3。' },
            ] },
          { name: '分类头', type: 'Sequential', origin_kind: 'standard', mini: 'linear', desc: ['根据特征给每个类别打分，再变成概率。', '- 排成一排', '- 全连接：三个分数', '- softmax：三个概率'],
            why: '特征提取只回答“哪里有什么边”。**“所以这是几”由这一部分回答。**', origin: '“全连接 + softmax”是**分类网络的标准结尾**。', note: '18 个数变成 3 个概率。',
            children: [
              { name: 'flatten', type: 'Flatten', params: 0, ...T(4, [18]), note: '2×3×3 排成 18 个数，数值和上一步完全相同。' },
              { name: 'fc', type: 'Linear', params: 54, ...T(5, [3], { labels: CLASSES }), desc: ['全连接层：18 个数变成 3 个分数。', '- 每个类别一个分数'],
                origin: '全连接层是标准做法；这里的 54 个权重是**用三个模板算出来的**，不是训练出来的。',
                note: `三个分数：${CLASSES.map((c, i) => `“${c}” ${logits[i].toFixed(2)}`).join('，')}。特征越像哪个模板，哪个分数越高。` },
              { name: 'softmax', type: 'Softmax', params: 0, ...T(6, [3], { labels: CLASSES }),
                note: `分数变成概率：${CLASSES.map((c, i) => `“${c}” ${pct(probs[i])}`).join('，')}。` },
            ] },
        ] },
      { name: 'loss', type: 'CrossEntropyLoss', params: 0, out: { shape: [], data: r(loss, 4) },
        origin: FIXED + '交叉熵是分类任务的标准损失。',
        note: `正确类别“7”的概率是 ${pct(probs[TARGET])}，损失 = −ln(${probs[TARGET].toFixed(3)}) = **${loss[0].toFixed(3)}**。概率越接近 100%，损失越接近 0。` },
    ] },
}));
