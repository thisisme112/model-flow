"""Project glue for pytorch/examples mnist: train one epoch the way main.py does, keeping the weights at a few moments;
trace one test image through each of them; group the steps into levels; write the words for a beginner.
Run from the mnist/ directory (the data is expected in ../data, where main.py puts it):

    python <this file> spec.json

The first run trains (a few minutes on a CPU) and leaves the kept weights in mnist_stages.pt; later runs reuse them.
Needs matplotlib for the one figure (how the answer changes during training).
"""
import argparse
import base64
import copy
import io
import json
import os
import sys

import matplotlib
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path[:0] = [os.getcwd(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")]
from fxtrace import trace, variant  # noqa: E402
from main import Net, train  # noqa: E402

plt.rcParams.update({"font.sans-serif": ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "PingFang SC", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 9})
STAGES = [0, 20, 100, 400]  # weights are kept after this many training steps (batches of 64), and at the end of the epoch
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
test = datasets.MNIST("../data", train=False, transform=tf)


class Keep:
    """main.py's training loader, pausing to copy the weights before the steps in STAGES."""

    def __init__(self, loader, model):
        self.loader, self.model, self.dataset, self.kept = loader, model, loader.dataset, {}

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for step, batch in enumerate(self.loader):
            if step in STAGES:
                self.kept[step] = copy.deepcopy(self.model.state_dict())
            yield batch


def stages():
    if os.path.exists("mnist_stages.pt"):
        return torch.load("mnist_stages.pt")
    torch.manual_seed(1)  # main.py's defaults on a CPU: seed 1, batches of 64 in dataset order, Adadelta lr 1.0
    keep = Keep(torch.utils.data.DataLoader(datasets.MNIST("../data", train=True, transform=tf), batch_size=64), Net())
    train(argparse.Namespace(log_interval=10**9, dry_run=False), keep.model, "cpu", keep, optim.Adadelta(keep.model.parameters(), lr=1.0), 1)
    keep.kept[len(keep)] = copy.deepcopy(keep.model.state_dict())
    torch.save(keep.kept, "mnist_stages.pt")
    return keep.kept


snaps = stages()
last = max(snaps)
model = Net()
model.eval()
tx = ((test.data.float() / 255 - 0.1307) / 0.3081)[:, None]  # the whole test set, normalised like tf does
ty = test.targets
assert torch.allclose(tx[0], test[0][0], atol=1e-5)

# the example: among the first 500 test images, the one the trained model gets right with the least confidence
model.load_state_dict(snaps[last])
with torch.no_grad():
    conf, pred = model(tx[:500]).exp().max(1)
conf[pred != ty[:500]] = 2
i = int(conf.argmin())
y = int(ty[i])
labels = [str(d) for d in range(10)]
history = []  # per kept stage: (steps, test accuracy, probability given to the right answer, the answer)


def stage(step):
    """Trace the example through the weights kept after `step` steps; the notes quote that stage's own numbers."""
    model.load_state_dict(snaps[step])
    with torch.no_grad():
        acc = (torch.cat([model(b) for b in tx.split(2000)]).argmax(1) == ty).float().mean().item()
    leaves, raw = trace(model, tx[i : i + 1], F.nll_loss, ty[i : i + 1])
    L = {n["id"]: n for n in leaves}
    p = raw["log_softmax"][0].exp()
    a, b = (int(k) for k in p.topk(2).indices)
    z0, z1 = ((raw[k] == 0).float().mean().item() for k in ("relu", "relu_1"))
    history.append((step, acc, p[y].item(), a))
    notes = {
        "x": f"测试集第 {i} 张图，标注是“{y}”。蓝黑是笔画，淡红是背景。",
        "conv1": "1 张 28×28 变成 32 张 26×26。有正有负：蓝黑为正，红为负。",
        "relu": f"这张图在这一步有 {z0:.0%} 的数变成了 0。",
        "conv2": "32 张 26×26 变成 64 张 24×24。",
        "relu_1": f"{z1:.0%} 的数变成 0" + ("，比第一层更稀疏。" if z1 > z0 else "。"),
        "max_pool2d": "24×24 变成 12×12。数据量减到四分之一，最强的响应都还在。",
        "dropout1": "现在是推理，数据原样通过：进来和出去完全一样。",
        "flatten": "64×12×12 = 9216 个数排成一排，数值不变。",
        "fc1": "9216 个数压成 128 个。",
        "relu_2": f"128 个数里只有 {int((raw['relu_2'] > 0).sum())} 个不为 0。",
        "dropout2": "推理时原样通过。",
        "fc2": f"10 个分数。最高的是“{a}”，{raw['fc2'][0, a]:.2f} 分；其次是“{b}”，{raw['fc2'][0, b]:.2f} 分。",
        "log_softmax": f"换算回概率：**“{a}”{p[a]:.1%}**，“{b}”{p[b]:.1%}。",
        "target": "只在算损失时用到，不进入网络。",
        "loss": f"正确答案“{y}”的对数概率是 {raw['log_softmax'][0, y]:.3f}。去掉负号，损失就是 **{raw['loss'].item():.3f}**。",
    }
    for k, w in notes.items():
        L[k]["note"] = w
    L["fc2"]["out"]["labels"] = L["log_softmax"]["out"]["labels"] = L["fc2"]["grad"]["labels"] = L["log_softmax"]["grad"]["labels"] = labels
    groups = {  # the notes on modules that quote numbers
        "net": f"模型认为是 **“{a}”**，把握 {p[a]:.1%}；第二可能是“{b}”，{p[b]:.1%}。",
        "hidden": notes["relu_2"],
        "out": notes["log_softmax"],
    }
    if step == last:
        name = f"训练完一轮（{step} 步）· 准确率 {acc:.1%}"
        example = f"测试集第 {i} 张图，标注“{y}”。模型**答对了，但把握只有 {p[a]:.1%}**：前 500 张里它答对却最犹豫的一张。"
    else:
        name = f"第 {step} 步 · 准确率 {acc:.1%}" if step else f"还没训练 · 准确率 {acc:.1%}"
        example = f"测试集第 {i} 张图，标注“{y}”。" + (f"训练到第 {step} 步时，" if step else "还没训练时，") + f"模型认为是“{a}”，把握 {p[a]:.1%}，" + ("**答对了**。" if a == y else "**还没答对**。")
    item = variant(name, leaves, example=example)
    item["nodes"].update({k: {"note": v} for k, v in groups.items()})
    print(f"step {step:4d}  acc {acc:.4f}  pred {a} p={p[a]:.3f} loss={raw['loss'].item():.3f}")
    return item, L, groups, acc, example


items = []
for step in sorted(snaps):
    item, L, notes, acc, example = stage(step)
    items.append(item)


def G(name, kids, **kw):
    return {"name": name, "type": "模块", "children": kids, **kw}


def fig_progress():
    """The one picture: how accuracy on the test set and the answer to this image change as training goes on."""
    fig, ax = plt.subplots(figsize=(5.0, 2.4), constrained_layout=True)
    xs = range(len(history))
    ax.plot(xs, [h[1] for h in history], "o-", color="#2f6fb5", label="整个测试集的准确率")
    ax.plot(xs, [h[2] for h in history], "s--", color="#c8372d", label=f"这张图：给“{y}”的概率")
    for k, h in enumerate(history):
        ax.annotate(f"猜“{h[3]}”", (k, h[2]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color="#c8372d")
    ax.set_xticks(list(xs), [f"{h[0]} 步" for h in history])
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("训练了多少步（一步 = 64 张图）")
    ax.legend(loc="center right", fontsize=8, frameon=False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, facecolor="white")
    plt.close(fig)
    return {"image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), "caption": "横向是训练进度。蓝线：全部 1 万张测试图答对的比例；红线：模型给这张图正确答案的概率。"}


PROGRESS = fig_progress()
# ---- the words that hold for every stage: what each part does, why it is there, where it comes from ---------------------
L["x"].update(name="输入图像", desc=["一张 28×28 的灰度手写数字。", "- 784 个像素，每个是一个亮度", "- 已减去 0.1307、除以 0.3081（整个数据集的均值和标准差）"],
              why="减均值、除标准差：**让输入的数值大小适中**，训练更稳。",
              terms={"像素": "图片上的一个小格，这里用一个数表示它有多亮。", "归一化": "把数据平移、缩放到一个统一的范围。"})
L["conv1"].update(desc=["32 个 3×3 的卷积核，各自扫过整张图。", "- 每个位置算一次加权和", "- 一个卷积核出一张特征图"],
                  why="笔画的边缘、拐角这些小图案出现在哪里都一样。**同一个卷积核扫遍全图，哪里有就在哪里亮起来。**",
                  terms={"卷积核": "一小块权重（这里 3×3）。它和图上哪一块长得像，那里的输出就大。", "特征图": "卷积的输出：一张图，每个位置的数表示“这里有多像这个卷积核要找的图案”。"})
L["conv2"].update(desc=["64 个卷积核，每个同时看上一步的全部 32 张特征图。", "- 仍然是 3×3 的范围"],
                  why="第一层只认得出最简单的笔画片段。**第二层把片段组合起来**，能认出更大的形状。")
L["max_pool2d"].update(desc=["每 2×2 个格子只留最大的一个。", "- 高和宽各减半"], why="后面全连接层的参数量和输入大小成正比。**缩小四倍，参数就少四倍**；笔画挪动一两个像素，结果也不变。")
L["dropout1"].update(desc=["训练时随机把 25% 的数变成 0。", "- 推理时什么都不做"])
L["dropout2"].update(desc=["训练时随机把 50% 的数变成 0。", "- 推理时什么都不做"])
L["flatten"].update(why="卷积输出的是一叠图，全连接层只收一排数。**所以先排成一排。**")
L["fc1"].update(desc=["全连接层：9216 个数变成 128 个。", "- 每个输出都是全部输入的加权和"], why="卷积给出的是“哪里有什么笔画”。**这一层把它们综合起来**，变成对整张图的描述。")
L["fc2"].update(desc=["全连接层：128 个数变成 10 个分数。", "- 每个数字一个分数", "- 分数越高，模型越认为是这个数字"], why="分类任务最后要的就是**每个类别一个分数**。",
                terms={"分数 logit": "还没换算成概率的原始分数，可正可负，只有相互之间的高低有意义。"})
L["log_softmax"].update(desc=["把 10 个分数换成对数概率。", "- 先变成加起来等于 1 的概率", "- 再取对数：全是负数，越接近 0 概率越高"],
                        why="分数本身说不出“有多大把握”。**换成概率才能说。** 取对数是为了和后面的损失配合，计算更稳。",
                        terms={"对数概率": "概率取对数。概率 100% 对应 0，概率越低，数越负。"})
L["target"].update(name="标签", desc="人工标注的正确答案。", why="训练要知道正确答案，才能算出模型错了多少。**模型自己判断时看不到它。**")
L["loss"].update(name="损失", desc=["模型错了多少，用一个数表示。", "- 取出正确答案的对数概率", "- 去掉负号"],
                 why="训练就是调整参数，让这个数变小。**模型越有把握答对，它越接近 0。**",
                 origin="分类任务的**标准损失**：负对数似然（NLL），和 log_softmax 连用就是交叉熵。", origin_kind="standard")

n_par = sum(q.numel() for q in model.parameters())
spec = {  # the tree is the trained model's; the slider swaps in the numbers and notes of the other stages
    "title": "MNIST 卷积网络：一张图的旅程",
    "source": "pytorch/examples · mnist/main.py",
    "summary": "PyTorch 官方示例里的手写数字分类器。**两层卷积找笔画，两层全连接下判断。**",
    "example": example,
    "stats": [["这张图的标注", f"“{y}”", f"测试集第 {i} 张"], ["参数", f"{n_par:,}", "其中约 118 万在 fc1"], ["训练一轮后的准确率", f"{acc:.1%}", "1 万张测试图"],
              ["训练了", f"{last} 步", "每步 64 张图"]],
    "background": [
        ["任务：**看一张手写数字的图，说出它是 0 到 9 里的哪一个。**", "- MNIST：7 万张 28×28 的手写数字", "- 深度学习里最经典的入门数据集"],
        ["进去和出来：", "- 进去：一张 28×28 的灰度图，784 个数", "- 出来：10 个数，每个数字一个，**最大的那个就是模型的答案**"],
        ["模型的想法：**先找局部的笔画，再综合起来判断。**", "- 卷积层：在图上找笔画的片段", "- 全连接层：根据找到的片段判断是哪个数字"],
        ["怎么训练：", "- 给模型看一批图，算出它错了多少（损失）", "- 按梯度把每个参数调一点点，重复 938 次", "- 拖动“训练进度”可以看同一张图在训练前后的结果"],
    ],
    "glossary": {"MNIST": "一个手写数字图片数据集：6 万张用来训练，1 万张用来测试。", "推理": "模型训练好之后拿来用。和训练相对。", "过拟合": "模型把训练数据背了下来，遇到没见过的数据就不行。"},
    "concepts": [
        {"name": "特征图和通道", "aliases": ["特征图", "通道"], "short": "卷积的输出是一叠图：**每一张叫一个通道，是一种图案的“出现地图”。**",
         "table": {"head": ["在哪一步之后", "有几张", "每张多大"], "rows": [["输入", 1, "28×28"], ["conv1", 32, "26×26"], ["conv2", 64, "24×24"], ["max_pool2d", 64, "12×12"]]},
         "text": ["- 一个卷积核产生一张特征图", "- 图上某处的数越大，说明那里越像这个卷积核要找的图案", "- 越往后，张数越多、每张越小：位置信息变粗，图案种类变多"]},
        {"name": "分数、概率和损失", "aliases": ["对数概率", "概率", "分数"], "short": "最后三步把 128 个数变成**一个答案和一个“错了多少”**。",
         "table": {"head": ["名词", "怎么来的", "图上"], "rows": [["分数", "全连接层直接算出，可正可负", "`fc2`"], ["对数概率", "分数 → 概率 → 取对数", "`log_softmax`"], ["损失", "正确答案的对数概率，去掉负号", "损失"]]},
         "text": ["- 模型的答案：分数最高的那个数字", "- 把握：那个数字的概率", "- 损失只看正确答案那一项：给它的概率越高，损失越小"]},
        {"name": "训练进度", "aliases": ["训练进度", "训练"], "short": "第二个滑块：**同一张图，在训练的五个时刻分别走一遍。**", "figure": PROGRESS,
         "table": {"head": ["时刻", "测试准确率", f"给“{y}”的概率", "模型的答案"], "rows": [[f"{h[0]} 步" if h[0] else "还没训练", f"{h[1]:.1%}", f"{h[2]:.1%}", f"“{h[3]}”" + (" ✓" if h[3] == y else "")] for h in history]},
         "text": ["- 网络结构一直没变，变的只是参数的数值", "- 没训练时参数是随机的，10 个数字的概率都差不多", "- 拖动滑块，看每一层的输出怎样从杂乱变得有条理"]},
    ],
    "levels": ["整体", "两个阶段", "功能块", "每一层"],
    "variants": {"label": "训练进度", "items": items},
    "root": G("root", [
        L["x"],
        G("Net", [
            G("特征提取", [
                G("卷积块 1", [L["conv1"], L["relu"]], origin_kind="standard", mini="conv", desc=["卷积，再过 ReLU。", "- 找出最基本的笔画片段：横、竖、斜、拐角"],
                  why="识别数字要从笔画开始。**这一块负责在图上找笔画。**", origin="“卷积 + 激活”是**卷积网络最基本的单元**。", note="1 张 28×28 变成 32 张 26×26。"),
                G("卷积块 2", [L["conv2"], L["relu_1"]], origin_kind="standard", mini="conv", desc=["第二次卷积，再过 ReLU。", "- 把上一块找到的片段组合起来看"],
                  why="单个笔画片段分不出 4 和 9。**组合起来才看得出形状。**", origin="和卷积块 1 是同一种单元，通道更多。", note="32 张 26×26 变成 64 张 24×24。"),
                G("下采样", [L["max_pool2d"], L["dropout1"]], origin_kind="standard", mini="pool", desc=["把特征图缩小一半。", "- 池化：2×2 留最大的", "- Dropout：只在训练时起作用"],
                  why="**缩小数据，后面要算的就少。**", origin="最大池化和 Dropout 都是**标准做法**。", terms={"下采样": "把数据的尺寸缩小。", "Dropout": "训练时随机丢掉一部分数，防止过拟合。"},
                  note="64 张 24×24 变成 64 张 12×12。"),
            ], origin_kind="standard", desc=["把像素变成“哪里有什么样的笔画”。", "- 两层卷积", "- 一次池化"],
                why="直接拿 784 个像素去判断，字稍微挪一下就全变了。**先提取笔画这样的特征，判断才稳。**",
                origin="**最普通的小卷积网络**，没有专门的名字。结构和 1998 年的 LeNet 是一个思路。", terms={"特征": "模型从原始数据里算出来的、对判断有用的中间结果。"},
                note="784 个像素变成 64×12×12 = 9216 个特征。"),
            G("分类头", [
                L["flatten"],
                G("隐藏层", [L["fc1"], L["relu_2"], L["dropout2"]], id="hidden", origin_kind="standard", mini="linear", desc=["把 9216 个特征浓缩成 128 个。", "- 全连接", "- ReLU", "- Dropout（只在训练时）"],
                  why="9216 个特征是零散的。**这一层把它们综合成对整张图的描述。**", origin="“全连接 + 激活”是**多层感知机（MLP）的基本单元**。", note=notes["hidden"]),
                G("输出层", [L["fc2"], L["log_softmax"]], id="out", origin_kind="standard", desc=["给 10 个数字各打一个分，再换成对数概率。"], why="**这是模型给出答案的地方。**",
                  origin="“全连接 + softmax”是**分类网络的标准结尾**。", note=notes["out"]),
            ], origin_kind="standard", desc=["根据特征判断是哪个数字。", "- 先把特征图排成一排", "- 两层全连接"], why="特征提取只回答“哪里有什么”。**“所以这是几”由这一部分回答。**",
                origin="**标准的分类头**：一个小的多层感知机。", terms={"全连接": "输出的每个数都由全部输入加权求和得到。"}, note="9216 个特征变成 10 个对数概率。"),
        ], id="net", type="main.py 里的 Net 类", origin_kind="standard", figure=PROGRESS, desc=["整个网络。", "- 特征提取：从像素里找笔画", "- 分类头：根据笔画判断数字"],
            why="**这就是被训练的东西**：它的参数从随机开始，一步步调到能认出数字。", origin="PyTorch 官方示例里的**一个小卷积网络**，没有专门的名字。", note=notes["net"]),
        L["target"],
        L["loss"],
    ]),
}
json.dump(spec, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)
ref = "mnist_cnn.pt"
if os.path.exists(ref):
    same = all(torch.allclose(v, snaps[last][k], atol=1e-6) for k, v in torch.load(ref).items())
    print(f"end of epoch equals {ref} (main.py --epochs 1 --save-model): {same}")
print(f"sample {i} label {y}; leaves: {' '.join(L)}")
