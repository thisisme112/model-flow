"""Project glue for pytorch/examples siamese_network: one pair of test images through the two towers.
Run from the siamese_network/ directory:

    python <this file> spec.json

The first run trains one epoch with main.py's own train() (a few minutes on a CPU) and leaves siamese_network.pt.
"""
import argparse
import json
import os
import random
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets

sys.path[:0] = [os.getcwd(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")]
from fxtrace import nest, trace  # noqa: E402
from main import APP_MATCHER, SiameseNetwork, train  # noqa: E402

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
p_same[p_same <= 0.5] = 2
i = int(p_same.argmin())
j, d = same[i], int(digit[i])

leaves, raw = trace(model, (data[i : i + 1], data[j : j + 1]), lambda out, t: F.binary_cross_entropy(out.squeeze(), t), torch.tensor(1.0))
for n in leaves:  # main.py wraps ResNet's layers in nn.Sequential, so many are just called "0", "1": say whose
    if n["name"].isdigit():
        n["name"] = n["path"][-1].split("@")[0].split(".")[-1] + "." + n["name"]
p, loss = raw["sigmoid"].item(), raw["loss"].item()
x1, x2, t1, v1, t2, v2, cat, fc, sig, target, lossn = nest(leaves, model)["children"]
assert (t1["name"], v1["type"], t2["name"], cat["type"], fc["name"]) == ("resnet", "view", "resnet", "cat", "fc"), [c["name"] for c in (t1, v1, t2, cat, fc)]


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


def step(n, shortcut=False):
    """The words for one layer, quoting this pair's own numbers."""
    t, kind = raw[n["id"]], n["type"]
    a = raw[n["from"][0]] if n.get("from") else None
    if kind == "Conv2d":
        n["note"] = change(n) + ("尺寸减半。" if t.shape[-1] < a.shape[-1] else "")
    elif kind == "BatchNorm2d":
        n["note"] = f"每个通道重新调整了尺度：数值范围从 {a.min():.1f}…{a.max():.1f} 变成 {t.min():.1f}…{t.max():.1f}。"
    elif kind == "ReLU":
        n["note"] = f"负数变成 0：这一步有 {(t == 0).float().mean():.0%} 的数被清零。"
    elif kind == "MaxPool2d":
        n["note"] = "每个 3×3 的窗口只留最大值。" + change(n)
    elif kind == "add":
        n["note"] = "主路算出的结果，加上从块的入口送来的数据（" + ("先经过捷径调整过" if shortcut else "原样，没有改动") + "）。这就是残差连接。"
    elif kind == "AdaptiveAvgPool2d":
        n["note"] = f"每张特征图取一个平均值：{dims(a)} 变成 {t.shape[1]} 个数。"
    elif kind == "view":
        n.update(desc="改变形状：数值不变，只是换一种排法。", note=f"排成一排，{t.numel()} 个数。这就是这张图的特征向量。")
    elif kind == "Linear":
        n["note"] = change(n)[:-1] + (f"：{t.item():.2f}。这是“两张图有多像”的分数。" if t.numel() == 1 else "。")


def block(b, k):
    """A BasicBlock. With a projection shortcut, the main path and the shortcut become two lanes."""
    ks = b["children"]
    down = next((c for c in ks if c["name"] == "downsample"), None)
    for c in ks:
        for leaf in c.get("children", [c]):
            step(leaf, bool(down))
    b.update(name=f"残差块 {k}", note=change(b))
    if not down:
        b["desc"] = "两层卷积，结果再加上块的输入（残差连接），最后过一次 ReLU。"
        return b
    down.update(name="捷径", desc="一次 1×1 卷积加归一化：只为了把输入调整成和主路输出一样的形状。", note=change(down))
    main = G("主路", ks[: ks.index(down)], desc="两层 3×3 卷积：真正提取特征的地方。", note=change({"children": ks[: ks.index(down)]}))
    both = G("两条路", [main, down], parallel=True, desc="块的输入同时走两条路：主路做两层卷积，捷径只做一次 1×1 卷积。", note="两条路的输出形状相同，下一步把它们相加。")
    b.update(children=[both, *ks[ks.index(down) + 1:]], desc="两层卷积。这个块把尺寸减半、通道加倍，所以输入要先走捷径调整形状，才能和主路的结果相加。")
    return b


def tower(g, view, tag, x, idx):
    ks = g["children"]
    for leaf in [*ks[:4], ks[8], view]:
        step(leaf)
    stem = G("入口", ks[:4], desc="一层 7×7 的大卷积加一次池化，先把图缩小。", note=change({"children": ks[:4]}))
    stages = []
    for k, s in enumerate(ks[4:8], 1):
        s["children"] = [block(b, m) for m, b in enumerate(s["children"], 1)]
        s.update(name=f"阶段 {k}", desc=f"两个残差块（代码里的 resnet.{k + 3}，也就是 ResNet 的 layer{k}）。" + ("第一个块把尺寸减半、通道加倍。" if k > 1 else ""), note=change(s))
        stages.append(s)
    x.update(name=f"图 {tag}", note=f"测试集第 {idx} 张，是“{d}”。像素值是 0 到 255 的原始亮度，这个示例没有做归一化。")
    g.update(name=f"塔 {tag}", type="ResNet-18", children=[stem, *stages, ks[8], view],
             desc="ResNet-18 去掉最后的分类层，当作特征提取器。两座塔是同一个 self.resnet，被调用了两次。",
             note=f"图 {tag} 变成 {raw[view['id']].numel()} 个数的特征向量。")
    return g


towers = [tower(t1, v1, "A", x1, i), tower(t2, v2, "B", x2, j)]
head = [cat, *fc["children"], sig]
for leaf in fc["children"]:
    step(leaf)
cat["note"] = f"两座塔各给出 {raw[v1['id']].numel()} 个数，接成 {raw[cat['id']].numel()} 个。"
sig["note"] = f"分数 {raw[sig['from'][0]].item():.2f} 被压到 0 和 1 之间，得到 {p:.1%}：这是模型认为“两张图是同一个数字”的把握。"
target.update(name="标签", desc="人工标注的正确答案。", note="1 表示两张图是同一个数字，0 表示不是。这一对是 1。", out={"tokens": ["相同 (1)"]})
lossn.update(name="损失", type="BCELoss", desc="二元交叉熵：模型给正确答案的概率越低，损失越大。", note=f"正确答案是“相同”，模型给了 {p:.1%}。损失 = −ln({p:.3f}) = {loss:.3f}。")

spec = {
    "title": "MNIST 孪生网络",
    "source": "pytorch/examples · siamese_network/main.py",
    "summary": f"PyTorch 官方示例：两张手写数字各走一遍同一个 ResNet-18，再判断它们是不是同一个数字。共 {sum(q.numel() for q in model.parameters()):,} 个参数，"
               f"训练一轮后，在 {2 * N} 对测试图片（一半相同、一半不同）上答对 {acc:.1%}。",
    "example": f"测试集第 {i} 张和第 {j} 张，都是“{d}”。模型答对了，但把握只有 {p:.1%}，是 {N} 对相同数字里它答对却最犹豫的一对。",
    "levels": ["整体", "两座塔", "塔里的阶段", "残差块", "块的内部", "主路和捷径", "每一层"],
    "root": G("root", [
        x1, x2,
        G("两座塔", towers, parallel=True, type="同一套权重", desc="两张图各走一遍同一个 ResNet-18。权重只有一套，所以两张图是用完全相同的方式变成特征的。",
          note=f"两张 28×28 的图，各变成 {raw[v1['id']].numel()} 个数。"),
        G("比较头", head, type="fc + sigmoid", desc="把两座塔的特征接在一起，用两层全连接判断是不是同一个数字。", note=f"{raw[cat['id']].numel()} 个数变成一个概率：{p:.1%}。"),
        target, lossn,
    ]),
}
json.dump(spec, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)
print(f"pair {i},{j} digit {d} p={p:.3f} loss={loss:.3f}; accuracy on {2 * N} pairs {acc:.4f}; {len(leaves)} leaves")
