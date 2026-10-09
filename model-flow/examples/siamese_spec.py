"""Project glue for pytorch/examples siamese_network: one pair of test images through the two towers.
Run from the siamese_network/ directory:

    python <this file> spec.json

The first run trains one epoch with main.py's own train() (a few minutes on a CPU) and leaves siamese_network.pt.
Needs matplotlib for the two figures (the pair, and where its score falls among the other test pairs).
"""
import argparse
import base64
import io
import json
import os
import random
import sys

import matplotlib
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path[:0] = [os.getcwd(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")]
from fxtrace import nest, trace  # noqa: E402
from main import APP_MATCHER, SiameseNetwork, train  # noqa: E402

plt.rcParams.update({"font.sans-serif": ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "PingFang SC", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 9})
if not os.path.exists("siamese_network.pt"):
    torch.manual_seed(1)  # main.py's defaults on a CPU: seed 1, batches of 64, Adadelta lr 1.0
    random.seed(1)  # main.py leaves this one unseeded; the training pairs are drawn with it
    loader = torch.utils.data.DataLoader(APP_MATCHER("../data", train=True), batch_size=64)
    model = SiameseNetwork()
    train(argparse.Namespace(log_interval=100, dry_run=False), model, "cpu", loader, optim.Adadelta(model.parameters(), lr=1.0), 1)
    torch.save(model.state_dict(), "siamese_network.pt")
model = SiameseNetwork()
model.load_state_dict(torch.load("siamese_network.pt"))
model.eval()

# pairs from the test set, built the way APP_MATCHER feeds them (raw 0..255 pixels): each of the first N images with
# the next image of the same digit, and with the next image of another digit
test = datasets.MNIST("../data", train=False)
data, digit = test.data[:, None].float(), test.targets
N = 300
same = [next(j for j in range(i + 1, len(digit)) if digit[j] == digit[i]) for i in range(N)]
other = [next(j for j in range(i + 1, len(digit)) if digit[j] != digit[i]) for i in range(N)]
with torch.no_grad():
    p_same, p_other = model(data[:N], data[same])[:, 0], model(data[:N], data[other])[:, 0]
acc = ((p_same > 0.5).sum() + (p_other <= 0.5).sum()).item() / (2 * N)
# the example: the same-digit pair the model gets right with the least confidence
masked = p_same.clone()
masked[masked <= 0.5] = 2
i = int(masked.argmin())
j, d = same[i], int(digit[i])

leaves, raw = trace(model, (data[i : i + 1], data[j : j + 1]), lambda out, t: F.binary_cross_entropy(out.squeeze(), t), torch.tensor(1.0))
for n in leaves:  # main.py wraps ResNet's layers in nn.Sequential, so many are just called "0", "1": say whose
    if n["name"].isdigit():
        n["name"] = n["path"][-1].split("@")[0].split(".")[-1] + "." + n["name"]
p, loss = raw["sigmoid"].item(), raw["loss"].item()
x1, x2, t1, v1, t2, v2, cat, fc, sig, target, lossn = nest(leaves, model)["children"]
assert (t1["name"], v1["type"], t2["name"], cat["type"], fc["name"]) == ("resnet", "view", "resnet", "cat", "fc"), [c["name"] for c in (t1, v1, t2, cat, fc)]
D = raw[v1["id"]].numel()  # the length of one tower's feature vector


def G(name, kids, **kw):
    return {"name": name, "type": "模块", "children": kids, **kw}


def first(n):
    return first(n["children"][0]) if "children" in n else n


def last(n):
    return last(n["children"][-1]) if "children" in n else n


def dims(t):  # "64 张 7×7" for a stack of maps, "512 个数" for a vector
    s = list(t.shape[1:])
    return f"{s[0]} 张 {s[1]}×{s[2]}" if len(s) == 3 and s[1] * s[2] > 1 else f"{t.numel()} 个数"


def change(n):  # what a step or a module turns its input into
    return f"{dims(raw[first(n)['from'][0]])} 变成 {dims(raw[last(n)['id']])}。"


def png(fig, caption):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, facecolor="white")
    plt.close(fig)
    return {"image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), "caption": caption}


def fig_pair():
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9), constrained_layout=True)
    for ax, k, tag in zip(axes, (i, j), "AB"):
        ax.imshow(test.data[k].numpy(), cmap="Greys")
        ax.set_title(f"图 {tag}：测试集第 {k} 张，“{int(digit[k])}”", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"模型：{p:.1%} 的把握是同一个数字", fontsize=10, color="#c8372d")
    return png(fig, "这次跟踪的一对图。两张都是同一个数字，但写法不一样。")


def fig_scores():
    """Where this pair's score falls among all the test pairs: the picture of what "above 0.5" means."""
    fig, ax = plt.subplots(figsize=(5.0, 2.3), constrained_layout=True)
    bins = [k / 20 for k in range(21)]
    ax.hist(p_same.numpy(), bins=bins, color="#2f6fb5", alpha=0.75, label=f"{N} 对相同的数字")
    ax.hist(p_other.numpy(), bins=bins, color="#b26a00", alpha=0.75, label=f"{N} 对不同的数字")
    ax.axvline(0.5, color="#1b2a41", lw=0.8, ls="--")
    ax.axvline(p, color="#c8372d", lw=1.5)
    ax.annotate("这一对", (p, ax.get_ylim()[1] * 0.8), textcoords="offset points", xytext=(5, 0), color="#c8372d", fontsize=9)
    ax.set_xlabel("模型给出的概率：两张图是同一个数字")
    ax.set_ylabel("多少对")
    ax.set_yscale("log")
    ax.legend(fontsize=8, frameon=False, loc="upper center")
    return png(fig, "蓝色应该靠右，橙色应该靠左。虚线是 0.5 的分界；红线是这一对，刚过线。")


PAIR, SCORES = fig_pair(), fig_scores()


def step(n, shortcut=False):
    """The words for one layer, quoting this pair's own numbers. Plain layers keep the page's built-in desc and why."""
    t, kind = raw[n["id"]], n["type"]
    a = raw[n["from"][0]] if n.get("from") else None
    if kind == "Conv2d":
        n["note"] = change(n) + ("尺寸减半。" if t.shape[-1] < a.shape[-1] else "")
    elif kind == "BatchNorm2d":
        n["note"] = f"数值范围从 {a.min():.1f}…{a.max():.1f} 变成 {t.min():.1f}…{t.max():.1f}。"
    elif kind == "ReLU":
        n["note"] = f"这一步有 {(t == 0).float().mean():.0%} 的数被清零。"
    elif kind == "MaxPool2d":
        n["note"] = "每个 3×3 的窗口只留最大值。" + change(n)
    elif kind == "add":
        n.update(note="主路的结果，加上块入口送来的数据（" + ("先经过捷径调整过" if shortcut else "原样，没有改动") + "）。",
                 why="这就是**残差连接**：这一块只需要学“在输入上改动多少”，层数多了也好训练。")
    elif kind == "AdaptiveAvgPool2d":
        n.update(note=f"每张特征图取一个平均值：{dims(a)} 变成 {t.shape[1]} 个数。", why="到这里，“在图上的哪个位置”已经不重要了。**每张特征图只留一个数：这种图案有多少。**")
    elif kind == "view":
        n.update(desc="改变形状：数值不变，只是换一种排法。", origin_kind="fixed", note=f"排成一排，{t.numel()} 个数。**这就是这张图的特征向量。**")
    elif kind == "Linear":
        n["note"] = change(n)[:-1] + (f"：**{t.item():.2f}**。这是“两张图有多像”的分数。" if t.numel() == 1 else "。")


def block(b, k):
    """A BasicBlock. With a projection shortcut, the main path and the shortcut become two lanes."""
    ks = b["children"]
    down = next((c for c in ks if c["name"] == "downsample"), None)
    for c in ks:
        for leaf in c.get("children", [c]):
            step(leaf, bool(down))
    b.update(name=f"残差块 {k}", origin_kind="named", note=change(b), origin="ResNet 的**基本残差块**（BasicBlock），He 等 2015。")
    if not down:
        b.update(desc=["两层卷积，结果再加上块的输入。", "- 卷积 → 归一化 → ReLU", "- 卷积 → 归一化", "- 加上输入，再过一次 ReLU"],
                 why="层数一多，普通网络反而训练不动。**加上输入这条直通的路，几十层也能训练。**")
        return b
    down.update(name="捷径", origin_kind="named", desc=["一次 1×1 卷积，加归一化。", "- 不提取新特征"], why="主路把尺寸减半、通道加倍了。**输入要先调成同样的形状，才能相加。**",
                origin="ResNet 里的**投影捷径**（projection shortcut）。", note=change(down))
    main = G("主路", ks[: ks.index(down)], origin_kind="named", mini="conv", desc=["两层 3×3 卷积。", "- 第一层把尺寸减半、通道加倍"], why="**真正提取特征的地方。**", origin="残差块里的主路。",
             note=change({"children": ks[: ks.index(down)]}))
    both = G("两条路", [main, down], parallel=True, origin_kind="named", desc=["块的输入同时走两条路。", "- 主路：两层卷积", "- 捷径：一次 1×1 卷积"],
             why="**两条路的输出形状相同，下一步才能相加。**", origin="带投影捷径的残差块。", note="两条路的输出形状相同，下一步把它们相加。")
    b.update(children=[both, *ks[ks.index(down) + 1:]], desc=["两层卷积，结果加上（调整过形状的）输入。", "- 这个块把尺寸减半、通道加倍", "- 所以输入要先走捷径"],
             why="尺寸变小，后面算得快；通道变多，能表示的图案更多。**残差连接保证这样叠很多层也能训练。**")
    return b


def tower(g, view, tag, x, idx):
    ks = g["children"]
    for leaf in [*ks[:4], ks[8], view]:
        step(leaf)
    stem = G("入口", ks[:4], origin_kind="named", mini="conv", desc=["先把图缩小。", "- 一层 7×7 的大卷积", "- 归一化、ReLU", "- 一次池化"], why="**大卷积核一次看得远**，先粗略地抓住笔画的大致形状。",
             origin="ResNet 的入口（stem）。", note=change({"children": ks[:4]}))
    stages = []
    for k, s in enumerate(ks[4:8], 1):
        s["children"] = [block(b, m) for m, b in enumerate(s["children"], 1)]
        s.update(name=f"阶段 {k}", origin_kind="named", desc=["两个残差块。", f"- 代码里的 `resnet.{k + 3}`，也就是 ResNet 的 layer{k}"] + (["- 第一个块把尺寸减半、通道加倍"] if k > 1 else []),
                 why="一个阶段处理一种尺寸。**越往后图越小、通道越多**：从“哪里有一笔”变成“整体像什么”。", origin="ResNet-18 的四个阶段之一。", note=change(s))
        stages.append(s)
    x.update(name=f"图 {tag}", desc=["一张 28×28 的手写数字。", "- 像素值是 0 到 255 的原始亮度", "- 这个示例没有做归一化"], note=f"测试集第 {idx} 张，是“{d}”。",
             terms={"像素": "图片上的一个小格，用一个数表示它有多亮。"})
    g.update(name=f"塔 {tag}", type="ResNet-18", origin_kind="named", children=[stem, *stages, ks[8], view],
             desc=["把一张图变成一个特征向量。", "- 入口：先缩小", "- 四个阶段：每个两个残差块", "- 最后每张特征图取平均，排成一排"],
             why="直接比两张图的像素没用：同一个数字，每个人写得都不一样。**先变成特征，再比特征。**",
             origin=["**ResNet-18**（He 等 2015），去掉了最后的分类层。", "- 一个很常用的图像网络，18 层", "- 这里从头训练，没有用预训练权重"],
             terms={"特征向量": "一排数，概括了这张图的内容。两张图越像，它们的特征向量越接近。"},
             note=f"图 {tag} 变成 {D} 个数的特征向量。")
    return g


towers = [tower(t1, v1, "A", x1, i), tower(t2, v2, "B", x2, j)]
head = [cat, *fc["children"], sig]
for leaf in fc["children"]:
    step(leaf)
cat.update(why="后面的全连接层要同时看到两张图的特征，**才能比较它们**。", note=f"两座塔各给出 {D} 个数，接成 {raw[cat['id']].numel()} 个。")
sig.update(why="分数可以是任何数。**压到 0 和 1 之间，才能读成概率。**", note=f"分数 {raw[sig['from'][0]].item():.2f} 变成 **{p:.1%}**：模型认为“两张图是同一个数字”的把握。")
target.update(name="标签", desc="人工标注的正确答案：1 = 同一个数字，0 = 不是。", why="训练要知道正确答案，才能算出模型错了多少。**模型自己判断时看不到它。**", note="这一对是 1。", out={"tokens": ["相同 (1)"]})
lossn.update(name="损失", type="BCELoss", origin_kind="standard", desc=["模型错了多少，用一个数表示。", "- 模型给正确答案的概率越低，损失越大"],
             why="训练就是调整参数，让这个数变小。", origin="二分类的**标准损失**：二元交叉熵（BCE）。", terms={"二元交叉熵 BCE": "模型给正确答案的概率是 p，损失就是 −ln(p)。"},
             note=f"正确答案是“相同”，模型给了 {p:.1%}。损失 = −ln({p:.3f}) = **{loss:.3f}**。")

n_par = sum(q.numel() for q in model.parameters())
spec = {
    "title": "MNIST 孪生网络",
    "source": "pytorch/examples · siamese_network/main.py",
    "summary": "PyTorch 官方示例：判断两张手写数字是不是同一个数字。**两张图各走一遍同一个 ResNet-18，再比较。**",
    "example": f"测试集第 {i} 张和第 {j} 张，都是“{d}”。模型**答对了，但把握只有 {p:.1%}**：{N} 对相同数字里它答对却最犹豫的一对。",
    "stats": [["模型的判断", "相同 ✓", f"两张都是“{d}”"], ["把握", f"{p:.1%}", "0.5 以上判相同"], ["参数", f"{n_par:,}", "两座塔共用一套"], [f"{2 * N} 对测试图判对", f"{acc:.1%}", "训练一轮后"]],
    "background": [
        ["任务：**给两张手写数字的图，判断它们是不是同一个数字。**", "- 不问“这是几”，只问“是不是同一个”", "- 这种问法也用于人脸比对、签名核验"],
        ["进去和出来：", "- 进去：两张 28×28 的灰度图", "- 出来：一个 0 到 1 之间的数，**大于 0.5 就判“相同”**"],
        ["模型的想法：**两张图用完全相同的方式变成特征向量，再比较两个向量。**", "- “孪生”指两座塔是同一个网络", "- 同一套权重，所以两张图得到的特征可以直接比"],
        ["怎么训练：", "- 给模型看一对图，告诉它是不是同一个数字", "- 用官方脚本训练了一轮，没有用预训练权重"],
    ],
    "glossary": {"MNIST": "一个手写数字图片数据集。", "ResNet": "残差网络：一类用“残差连接”把很多层叠起来的图像网络。18 是它的层数。", "通道": "同一步输出的多张图中的一张，每个通道在找一种不同的图案。"},
    "concepts": [
        {"name": "两座塔", "aliases": ["两座塔", "共享权重", "同一套权重", "塔 A", "塔 B", "塔"], "short": "**其实只有一个网络**，被调用了两次：图 A 走一遍，图 B 走一遍。", "figure": PAIR,
         "table": {"head": ["", "输入", "输出", "权重"], "rows": [["塔 A", f"测试集第 {i} 张", f"{D} 个数", "`self.resnet`"], ["塔 B", f"测试集第 {j} 张", f"{D} 个数", "同一个 `self.resnet`"]]},
         "text": ["- 为什么共用：两张图必须用同样的标准变成特征，比较才有意义", "- 图上画成上下两行，是为了看清两路数据；参数只算一份", "- 这种结构叫孪生网络（Siamese network）"]},
        {"name": "残差块", "aliases": ["残差连接", "残差块", "捷径", "主路"], "short": "两层卷积的结果**加上块的输入**。图上是跨过几个方框的弧线。",
         "table": {"head": ["块的种类", "输入怎么送过去", "出现在哪"], "rows": [["普通的", "原样相加", "每个阶段的第二个块，阶段 1 的两个块"], ["改变尺寸的", "先走捷径（1×1 卷积）调整形状", "阶段 2、3、4 的第一个块"]]},
         "text": ["- 主路：真正提取特征的两层卷积", "- 残差连接：给输入留一条直通的路", "- 好处：每一块只需要学“改动多少”，几十层也训练得动"]},
        {"name": "相似度和判断", "aliases": ["相似度", "分数"], "short": "两个特征向量接在一起，过两层全连接，得到**一个概率**。", "figure": SCORES,
         "table": {"head": ["步骤", "得到什么"], "rows": [["cat", f"两个向量接成 {2 * D} 个数"], ["fc", "一个分数，可正可负"], ["sigmoid", "压到 0 和 1 之间，读作概率"]]},
         "text": ["- 大于 0.5：判“同一个数字”", "- 这一对是模型答对的里面最接近 0.5 的"]},
    ],
    "levels": ["整体", "两座塔", "塔里的阶段", "残差块", "块的内部", "主路和捷径", "每一层"],
    "root": G("root", [
        x1, x2,
        G("两座塔", towers, parallel=True, type="同一套权重", origin_kind="named", figure=PAIR, desc=["两张图各走一遍同一个 ResNet-18。", "- 权重只有一套", "- 所以两张图是用完全相同的方式变成特征的"],
          why="如果两张图用不同的网络，得到的特征就不在同一把尺子上。**共用权重，特征才能直接比。**",
          origin=["**孪生网络**（Siamese network）的标准结构。", "- 塔本身是 ResNet-18"], note=f"两张 28×28 的图，各变成 {D} 个数。"),
        G("比较头", head, type="fc + sigmoid", origin_kind="standard", mini="linear", figure=SCORES, desc=["根据两个特征向量，判断是不是同一个数字。", "- 把两个向量接在一起", "- 两层全连接", "- sigmoid 压成概率"],
          why="两座塔只负责“看懂”各自的图。**“像不像”由这一部分回答。**", origin="一个小的**多层感知机**（MLP）。", note=f"{raw[cat['id']].numel()} 个数变成一个概率：**{p:.1%}**。"),
        target, lossn,
    ]),
}
json.dump(spec, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)
print(f"pair {i},{j} digit {d} p={p:.3f} loss={loss:.3f}; accuracy on {2 * N} pairs {acc:.4f}; {len(leaves)} leaves")
