"""Project glue for pytorch/examples mnist: trace one test image, group the steps into levels, write the words.
Run from the mnist/ directory after `python main.py --save-model`:

    python <this file> spec.json
"""
import json
import os
import sys

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

sys.path[:0] = [os.getcwd(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]
from fxtrace import trace  # noqa: E402
from main import Net  # noqa: E402

model = Net()
model.load_state_dict(torch.load("mnist_cnn.pt"))
model.eval()
test = datasets.MNIST("../data", train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

# the example: among the first 500 test images, the one the model gets right with the least confidence
xs = torch.stack([test[i][0] for i in range(500)])
ys = test.targets[:500]
with torch.no_grad():
    conf, pred = model(xs).exp().max(1)
conf[pred != ys] = 2
i = int(conf.argmin())
y = int(ys[i])

leaves, raw = trace(model, xs[i : i + 1], F.nll_loss, ys[i : i + 1])
L = {n["id"]: n for n in leaves}
p = raw["log_softmax"][0].exp()
top = p.topk(2)
a, b = (int(k) for k in top.indices)
zero = lambda k: f"{(raw[k] == 0).float().mean().item():.0%}"  # noqa: E731

words = {
    "x": dict(name="输入图像", note=f"MNIST 测试集第 {i} 张图，人工标注是“{y}”。28×28 = 784 个像素，已经减去 0.1307 再除以 0.3081："
              "蓝黑是笔画（比平均亮），淡红是背景（比平均暗）。"),
    "conv1": dict(note="32 个 3×3 卷积核各自扫过整张图，每个位置算一次加权和，得到 32 张 26×26 的特征图。此时有正有负：蓝黑为正，红为负。"),
    "relu": dict(note=f"负数全部变成 0，正数不变。这张图在这一步有 {zero('relu')} 的数变成了 0。"),
    "conv2": dict(note="64 个卷积核，每个同时看上一步全部 32 张图里 3×3 的范围，得到 64 张 24×24 的特征图。"),
    "relu_1": dict(note=f"再清一次负数：{zero('relu_1')} 变成 0，留下的响应比第一层更稀疏。"),
    "max_pool2d": dict(note="每 2×2 个格子只留最大的一个，24×24 变成 12×12。数据量减到四分之一，最强的响应都还在。"),
    "dropout1": dict(note="训练时这里会随机丢掉 25% 的数。现在是推理，数据原样通过，所以进来和出去完全一样。"),
    "flatten": dict(note="64×12×12 = 9216 个数排成一排，数值不变。从这里起，数据不再有“图”的形状。"),
    "fc1": dict(note="9216 个数压成 128 个数。这一层有约 118 万个权重，占了整个网络的绝大部分。"),
    "relu_2": dict(note=f"清掉负数后，128 个数里只有 {int((raw['relu_2'] > 0).sum())} 个不为 0。"),
    "dropout2": dict(note="训练时随机丢掉 50%，推理时原样通过。"),
    "fc2": dict(note=f"128 个数变成 10 个分数，每个数字一个。最高的是“{a}”，{raw['fc2'][0, a]:.2f} 分；其次是“{b}”，{raw['fc2'][0, b]:.2f} 分。"),
    "log_softmax": dict(note=f"分数先变成概率，再取对数，所以全是负数，越接近 0 概率越高。换算回概率：“{a}”是 {p[a]:.1%}，“{b}”是 {p[b]:.1%}。"),
    "target": dict(name="标签", desc="人工标注的正确答案。", note="只在算损失时用到，不进入网络。"),
    "loss": dict(name="损失", note=f"取出正确答案“{y}”的对数概率 {raw['log_softmax'][0, y]:.3f}，去掉负号，损失就是 {raw['loss'].item():.3f}。模型越有把握答对，这个数越接近 0。"),
}
for k, w in words.items():
    L[k].update(w)
labels = [str(d) for d in range(10)]
L["fc2"]["out"]["labels"] = L["log_softmax"]["out"]["labels"] = L["fc2"]["grad"]["labels"] = L["log_softmax"]["grad"]["labels"] = labels


def G(name, kids, **kw):
    return {"name": name, "type": "模块", "children": kids, **kw}


spec = {
    "title": "MNIST 卷积网络：一张图的旅程",
    "source": "pytorch/examples · mnist/main.py",
    "summary": f"PyTorch 官方示例里的手写数字分类器，共 {sum(q.numel() for q in model.parameters()):,} 个参数，训练一轮后测试准确率约 98.6%。",
    "example": f"测试集第 {i} 张图，标注是“{y}”。模型答对了，但把握只有 {p[a]:.1%}，是前 500 张里它答对却最犹豫的一张。",
    "levels": ["整体", "两个阶段", "功能块", "每一层"],
    "root": G("root", [
        L["x"],
        G("Net", [
            G("特征提取", [
                G("卷积块 1", [L["conv1"], L["relu"]], desc="卷积之后接 ReLU：找出最基本的笔画片段。", note="1 张 28×28 的图变成 32 张 26×26 的特征图。"),
                G("卷积块 2", [L["conv2"], L["relu_1"]], desc="第二层卷积加 ReLU：把上一层找到的片段组合起来看。", note="32 张 26×26 变成 64 张 24×24。"),
                G("下采样", [L["max_pool2d"], L["dropout1"]], desc="池化缩小尺寸；Dropout 只在训练时起作用。", note="64 张 24×24 变成 64 张 12×12。"),
            ], desc="两层卷积加一次池化：把像素变成“哪里有什么样的笔画”。", note="784 个像素变成 64×12×12 = 9216 个特征。"),
            G("分类头", [
                L["flatten"],
                G("隐藏层", [L["fc1"], L["relu_2"], L["dropout2"]], desc="全连接加 ReLU：把 9216 个特征浓缩成 128 个。", note=words["relu_2"]["note"]),
                G("输出层", [L["fc2"], L["log_softmax"]], desc="给 10 个数字各打一个分，再换成对数概率。", note=words["log_softmax"]["note"]),
            ], desc="两层全连接：根据特征判断是哪个数字。", note="9216 个特征变成 10 个对数概率。"),
        ], type="main.py 里的 Net 类", desc="整个网络：先提取特征，再分类。", note=f"模型看完这张图，认为是“{a}”，把握 {p[a]:.1%}；第二可能是“{b}”，{p[b]:.1%}。"),
        L["target"],
        L["loss"],
    ]),
}
json.dump(spec, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)
print(f"sample {i} label {y} pred {a} p={p[a]:.3f} loss={raw['loss'].item():.3f}; leaves: {' '.join(L)}")
