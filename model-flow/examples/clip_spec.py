"""Project glue for OpenAI CLIP (Hugging Face openai/clip-vit-base-patch32): one photo against three captions.

    python <this file> spec.json [photo.jpg]

Needs `transformers`, `pillow` and matplotlib; the first run downloads the model (about 600 MB) into the Hugging Face cache.
The photo defaults to demo/clip/000000039769.jpg of the development repository (COCO val2017: two cats on a couch).
fx cannot trace this model, so the steps come from fxtrace's hook path; the three captions travel through the text
tower together, as a batch of three.
"""
import base64
import io
import json
import os
import sys

import matplotlib
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.sans-serif": ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "PingFang SC", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 9})
NAME = "openai/clip-vit-base-patch32"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from fxtrace import nest, tensor, trace  # noqa: E402

CAPTIONS = ["a photo of a cat", "a photo of a dog", "a photo of a couch"]  # each is 7 tokens, so nothing is padded and no mask is needed
SHORT, RIGHT = ["cat", "dog", "couch"], 0
proc = CLIPProcessor.from_pretrained(NAME)
model = CLIPModel.from_pretrained(NAME, attn_implementation="eager")  # eager: the attention weights are returned
PHOTO = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "..", "demo", "clip", "000000039769.jpg")
batch = proc(text=CAPTIONS, images=Image.open(PHOTO), return_tensors="pt", padding=True)
assert batch["attention_mask"].all(), "captions of different lengths would need the mask as an input"
ids, pixels = batch["input_ids"], batch["pixel_values"]
words = [proc.tokenizer.decode(t) for t in ids[RIGHT]]
T = ids.shape[1]

leaves, raw = trace(model, {"input_ids": ids, "pixel_values": pixels, "output_attentions": True},
                    lambda out, t: F.cross_entropy(out.logits_per_image, t), torch.tensor([RIGHT]), hooks=True, max_hw=24)
L = {n["id"]: n for n in leaves}
kids = nest(leaves, model)["children"]  # picked by name: the order of the two towers differs between transformers versions
by = {c["name"]: c for c in kids}
txt, img, vision, text, vproj, tproj, temp, target, loss = (by[k] for k in ("input_ids", "pixel_values", "vision_model", "text_model", "visual_projection", "text_projection", "logit_scale", "target", "loss"))
tnorm, vnorm = (next(c for c in kids if c["type"] == "div" and c["from"] == [q["id"]]) for q in (tproj, vproj))  # each tower's vector divided by its length
sim, flip = (next(c for c in kids if c["type"] == t) for t in ("mul", "t"))
loss["from"] = [sim["id"] if s == flip["id"] else s for s in loss["from"]]  # `t` only turns the 3x1 table of scores into 1x3: no step
scale = raw["logit_scale"].exp().item()
logits = raw[flip["id"]].flatten()
prob = logits.softmax(-1)
E_img, E_txt = raw[vnorm["id"]], raw[tnorm["id"]]  # the two towers' final unit vectors
eos = ids.argmax(-1)  # CLIP reads a caption off the position of its end-of-text token, the highest id


def G(name, kids, **kw):
    return {"name": name, "type": "模块", "children": kids, **kw}


def unit(v):
    return v / v.norm(dim=-1, keepdim=True)


def verdict(q):
    return "、".join(f"{SHORT[k]} {q[k]:.0%}" for k in q.argsort(descending=True))


with torch.no_grad():  # what the comparison would say if a tower stopped after some layer, the other one finished
    def after_vision(h):
        return (scale * unit(model.visual_projection(model.vision_model.post_layernorm(h[:, 0]))) @ E_txt.T)[0].softmax(-1)

    def after_text(h):
        return (scale * E_img @ unit(model.text_projection(model.text_model.final_layer_norm(h)[range(len(CAPTIONS)), eos])).T)[0].softmax(-1)

NORM = "每个位置的数各自调整到均值 0、方差 1 附近。"
grid_attn = {}  # per vision layer: where the class position looks, as a 7x7 map over the image patches
said_i, said_t = [], []  # the verdict if a tower stopped after each layer


def png(fig, caption, kind="png"):
    """A figure as a spec picture. A figure that contains a photo goes in as JPEG, or it weighs several hundred KB."""
    buf = io.BytesIO()
    fig.savefig(buf, format=kind, dpi=90, facecolor="white", **({"pil_kwargs": {"quality": 80}} if kind == "jpeg" else {}))
    plt.close(fig)
    return {"image": f"data:image/{kind};base64," + base64.b64encode(buf.getvalue()).decode(), "caption": caption}


def layer(b, k, vision_side):
    """The words for one transformer layer. q, k and v become three lanes: they read the same input side by side."""
    ln1, att, add1, ln2, mlp, add2 = b["children"]
    by = {c["name"]: c for c in att["children"]}
    q, kk, v, out = by["q_proj"], by["k_proj"], by["v_proj"], by["out_proj"]
    rest = [c for c in att["children"] if c not in (q, kk, v, out)]  # the weights, the weighted sum, and (in some versions) a dropout between
    w, mix = rest[0], rest[-1]
    # the weights arrive named after the last function that made them ("softmax" or "view", by version): check what they are
    assert out["from"] == [mix["id"]] and torch.allclose(raw[w["id"]].sum(-1), torch.tensor(1.0), atol=1e-4), [(c["name"], c["type"]) for c in att["children"]]
    fc1, act, fc2 = mlp["children"]
    d, a = raw[q["id"]].shape[-1], raw[w["id"]]
    ln1["note"] = ln2["note"] = NORM
    for n, what in ((q, "查询 Q：这个位置想找什么"), (kk, "键 K：这个位置有什么可被找到"), (v, "值 V：被看中时交出去的内容")):
        n.update(desc=f"一次线性变换，算出每个位置的{what}。", note=f"{d} 个数变成 {d} 个数。")
    for n in rest[1:-1]:
        n["note"] = "现在是推理，数据原样通过。"
    if vision_side:
        row = a[0].mean(0)[0]  # where the class position looks, averaged over the heads
        grid_attn[k] = row[1:].reshape(7, 7)
        r, c = divmod(int(row[1:].argmax()), 7)
        where = f"class 位置把 {row[0]:.0%} 的注意力留给自己。图块里最受关注的是第 {r + 1} 行第 {c + 1} 列那块（{row[1:].max():.0%}）。"
    else:
        row = a[RIGHT].mean(0)[-1]  # where the last position of the right caption looks
        where = f"以“{CAPTIONS[RIGHT]}”为例：句末位置最关注“{words[int(row.argmax())]}”（{row.max():.0%}）。"
    w.update(name="注意力权重", type="softmax(QKᵀ)", kind="attn", origin_kind="standard", mini="attention", out=tensor(a, max_c=4, max_hw=24), note=where,
             desc=["每个位置决定“看谁、看多少”。", "- 自己的 Q 和各个位置的 K 做点积", "- 过 softmax：一行加起来是 1", "- 每张小图是一个头"] + ([] if vision_side else ["- 文字这边只能看自己和前面，所以右上半边是空的"]),
             why="**哪些位置的信息值得拿过来，由这一步决定。**", origin="**缩放点积注意力**，Transformer 的核心。")
    w.pop("grad", None)
    mix.update(name="加权求和", type="权重 · V", kind="attn", origin_kind="standard", desc=["按注意力权重，把各个位置的 V 加起来。", "- 12 个头的结果拼回一排"], why="上一步只决定了“看谁”。**这一步才真的把信息取过来。**",
               note=f"每个位置重新得到 {d} 个数，里面混进了别的位置的信息。")
    out.update(desc="一次线性变换。", note=f"{d} 个数变成 {d} 个数。")
    add1.update(why="**残差连接**：注意力的结果是往主干上添一点，不是替换它。", note="把注意力算出的结果加回主干。")
    fc1["note"] = f"每个位置各自算：{d} 个数扩成 {4 * d} 个。"
    act.update(kind="act", origin_kind="standard", mini="gelu", desc="激活函数 QuickGELU：负数压到接近 0，正数基本不变。", note=f"这一步之后，{(raw[act['id']].abs() < 0.01).float().mean():.0%} 的数接近 0。")
    fc2["note"] = f"{4 * d} 个数压回 {d} 个。"
    add2.update(why="**残差连接**：前馈层的结果也是往主干上添。", note="把前馈层的结果加回主干。")
    qkv = G("Q K V", [q, kk, v], parallel=True, type="三个线性变换", origin_kind="standard", desc=["同一份输入分三路，各做一次线性变换。", "- Q：想找什么", "- K：有什么可被找到", "- V：被看中时交出去的内容"],
            why="注意力要拿 Q 去和 K 比，再按结果取 V。**所以先把数据变成这三份。**", origin="注意力的标准写法。", note="三路的输入完全相同，学到的变换不同。")
    att.update(name="注意力子层", origin_kind="standard", mini="attention", children=[qkv, *rest, out], desc=["各个位置互相看：每个位置从别的位置收集信息。", "- 算出 Q、K、V", "- 算注意力权重", "- 按权重把 V 加起来"],
               why="一个图块（或一个词）自己说明不了什么。**信息靠这一步在位置之间流动。**", origin="**多头自注意力**（multi-head self-attention），12 个头。",
               terms={"头": "一组独立的注意力。12 个头各看各的，再把结果拼起来。"}, note=where)
    mlp.update(name="前馈子层", origin_kind="standard", mini="linear", desc=["各个位置各自加工，互不相看。", f"- {d} → {4 * d}，过 QuickGELU，再 → {d}"], why="注意力只是把信息搬过来。**这一步对搬来的信息做加工。**",
               origin="Transformer 每层里**标准的前馈子层**。", note=f"{d} → {4 * d} → {d}。")
    said = (after_vision if vision_side else after_text)(raw[add2["id"]])
    (said_i if vision_side else said_t).append(said)
    b.update(name=f"第 {k + 1} 层", origin_kind="standard", desc=["一层 Transformer。", "- 注意力：各位置互相看", "- 前馈：各位置各自加工", "- 两步的结果都加回主干"],
             why="一层能做的加工有限。**十二层叠起来，才能从图块（或词）读出整体的意思。**", origin="**Transformer 编码器的一层**（pre-norm）。十二层结构相同，参数各不相同。",
             note=f"如果这座塔只算到这一层（另一座照常算完），结果会是：{verdict(said)}。")
    return b


# ---- image tower
emb, pre, enc, pick, post = vision["children"]
patch, pos, cls, add = emb["children"]
shown = (pixels[0] * torch.tensor(proc.image_processor.image_std)[:, None, None] + torch.tensor(proc.image_processor.image_mean)[:, None, None]).clamp(0, 1)
photo = (shown.permute(1, 2, 0) * 255).round().byte().numpy()  # the picture the model actually received: resized, cropped, un-normalised
buf = io.BytesIO()
Image.fromarray(photo).resize((112, 112)).save(buf, "JPEG", quality=85)
img.update(name="图片", out={"image": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), "shape": [3, 224, 224]},
           desc=["一张彩色照片。", "- 缩放并裁成 224×224", "- 红绿蓝三个通道，各减均值、除标准差"], note="COCO 数据集里的一张照片：两只猫躺在沙发上，旁边有两个遥控器。下面的梯度图显示哪些像素一变，损失就跟着变。",
           terms={"通道": "彩色图片有红、绿、蓝三张，每张叫一个通道。", "像素": "图片上的一个小格。"})
patch.update(desc=["把图切成 7×7 = 49 块，每块变成 768 个数。", "- 一层 32×32、步长 32 的卷积", "- 一块正好是一个卷积窗口"], why="Transformer 处理的是一串向量。**把图切成块，一张图就成了 49 个“词”。**",
             origin="**ViT 的切块嵌入**（patch embedding）。", terms={"图块 patch": "图片被切成的小方块。这里每块 32×32 像素。"}, note="3×224×224 变成 49 个 768 维向量。")
pos.update(desc="位置编码：每个位置一个学出来的向量。", why="**告诉模型这一块在图上的哪里。** 不加它，49 块打乱顺序结果也一样。", note="50 个位置（49 块 + 1 个 class 位置）各 768 个数。只和位置有关，所以没有箭头进来。")
cls.update(name="class 向量", desc=["一个学出来的向量，放在 49 块前面。", "- 不对应任何一块图"], why="需要一个地方汇总整张图的信息。**这个位置在 12 层里负责收集，最后用它代表整张图。**", note="768 个数，对每张图都一样。")
add.update(name="拼接并加位置", note="49 个图块前面接上 class 向量，成为 50 个位置；再各自加上位置向量。")
pre["note"] = NORM
enc["children"] = [layer(b, k, True) for k, b in enumerate(enc["children"])]
enc.update(name="编码器", origin_kind="standard", desc=["12 层结构相同的 Transformer。", "- 每层：注意力 + 前馈"], why="**图块之间要互相看，才能看出“这是一只猫”。**", origin="**Transformer 编码器**。",
           note="进去和出来都是 50×768：形状不变，内容一层层被改写。")
pick.update(name="取 class 位置", type="[:, 0]", origin_kind="fixed", desc="只留第 0 个位置（class）的结果。", why="**用它代表整张图。** 其余 49 个位置到这里就不用了。", note="50×768 里取出 768 个数。")
post["note"] = NORM
vproj.update(desc="一次线性变换：768 个数变成 512 个。", why="图片和文字的向量要能直接比较。**这一步把图片的向量投到两边共用的空间里。**", note="768 个数变成 512 个。")
vnorm.update(name="归一化", type="÷ 长度", origin_kind="fixed", desc=["除以向量自己的长度，使长度变成 1。"], why="长度都是 1 之后，**两个向量的点积就是夹角的余弦**：只比方向，不比大小。", note="512 个数，长度为 1。这就是这张图的“向量”。")
emb.update(name="切块和嵌入", origin_kind="named", desc=["把图变成一串向量。", "- 切成 49 块，每块一个向量", "- 前面加一个 class 向量", "- 各自加上位置向量"], why="**Transformer 只会处理一串向量**，所以先把图变成这种形式。",
           origin="**ViT**（Vision Transformer，Dosovitskiy 等 2020）的标准开头。", note="3×224×224 的图变成 50×768。")
tower_i = G("图像塔", [emb, pre, enc, pick, post, vproj, vnorm], type="ViT-B/32", origin_kind="named", desc=["把一张图变成 512 个数。", "- 切成 49 块", "- 过 12 层 Transformer", "- 取 class 位置，投影，归一化"],
            why="图片和文字本来没法比。**先把图变成一个向量。**", origin=["**ViT-B/32**：视觉 Transformer。", "- B：基础大小（12 层，宽 768）", "- /32：每块 32×32 像素"], note="一张图变成长度为 1 的 512 维向量。")

# ---- text tower
temb, tenc, tln, tpick = text["children"]
tokv, tpos, tadd = temb["children"]
txt.update(name="三句话", out={"tokens": CAPTIONS}, desc=["三句候选的描述，一起送进文字塔。", f"- 每句切成 {T} 个词元（含开头和结尾标记）"], note=f"比如第一句被切成“{'｜'.join(words)}”。",
           terms={"词元 token": "文字被切成的小片：一个词、半个词或一个标记。模型只认词元的编号。"})
tokv.update(why="编号本身没有含义。**换成向量后，意思相近的词可以有相近的向量。**", note=f"3 句话、每句 {T} 个词元，各换成 512 个数：3×{T}×512。三张叠在一起的小图就是这三句话。")
tpos.update(desc="位置编码：第 0、1、2… 个位置各有一个学出来的向量。", why="**告诉模型每个词排在第几个。**", note="只和位置有关，所以没有箭头进来。")
tadd["note"] = "词元的向量加上它所在位置的向量。"
tenc["children"] = [layer(b, k, False) for k, b in enumerate(tenc["children"])]
tenc.update(name="编码器", origin_kind="standard", desc=["12 层结构相同的 Transformer。", "- 每个位置只能看自己和前面的位置"], why="**词之间要互相看，才能读出整句的意思。**", origin="**Transformer**，带因果掩码（只能往前看）。", note=f"进去和出来都是 3×{T}×512。")
tln["note"] = NORM
tpick.update(name="取句末位置", type="[句末]", origin_kind="fixed", desc="每句话只留最后一个词元（结束标记）位置的结果。", why="因为只能往前看，**只有最后这个位置看过整句话**，所以用它代表整句。", note=f"3×{T}×512 里每句取出 512 个数。")
tproj.update(desc="一次线性变换：512 个数变成 512 个。", why="**把文字的向量投到和图片共用的空间里。**", note="512 个数变成 512 个。")
tnorm.update(name="归一化", type="÷ 长度", origin_kind="fixed", desc="除以向量自己的长度，使长度变成 1。", why="和图像塔一样：**只比方向，不比大小。**", note="三句话各得到一个长度为 1 的 512 维向量。")
temb.update(name="词元嵌入", origin_kind="standard", desc=["把词元变成向量，并标上位置。"], why="**文字要先变成数，才能进网络。**", origin="词嵌入 + 学出来的位置编码。", note=f"3 句话变成 3×{T}×512。")
tower_t = G("文字塔", [temb, tenc, tln, tpick, tproj, tnorm], type="Transformer", origin_kind="named", desc=["把一句话变成 512 个数。", "- 词元变成向量", "- 过 12 层 Transformer", "- 取句末位置，投影，归一化"],
            why="**把每句话也变成一个向量**，才能和图片的向量比。", origin="一个 12 层的**文字 Transformer**（结构同 GPT 一类）。", note="三句话各变成长度为 1 的 512 维向量。")

# ---- the comparison
cos = (logits / scale).tolist()
# The temperature (logit_scale) is a parameter this step multiplies by. As a card of its own it would stand in the
# main line between the towers and the comparison, so it is shown inside the step instead.
sim["from"] = [s for s in sim["from"] if s != temp["id"]]
sim["params"] = sim.get("params", 0) + temp.get("params", 0)


def fig_look():
    """The photo as the model received it, cut into its 49 patches, with where the class position looks in the last layer."""
    a = grid_attn[max(grid_attn)]
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.7), constrained_layout=True)
    for ax, title in zip(axes, ("模型看到的图：切成 7×7 块", f"第 {max(grid_attn) + 1} 层：class 位置在看哪些块")):
        ax.imshow(photo)
        for g in range(1, 7):
            ax.axhline(g * 32 - 0.5, color="white", lw=0.5)
            ax.axvline(g * 32 - 0.5, color="white", lw=0.5)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    axes[1].imshow(a.numpy().repeat(32, 0).repeat(32, 1), cmap="Reds", alpha=0.6)
    return png(fig, "左：每个小格是一个图块。右：颜色越红，class 位置从那一块取的信息越多（12 个头的平均）。", "jpeg")


def fig_sim():
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.0), constrained_layout=True)
    for ax, vals, title, fmt in ((axes[0], cos, "余弦相似度（乘温度之前）", "{:.3f}"), (axes[1], prob.tolist(), "换成概率之后", "{:.0%}")):
        ax.barh(range(3), vals, color=["#c8372d" if k == RIGHT else "#2f6fb5" for k in range(3)])
        for k, v in enumerate(vals):
            ax.text(v, k, " " + fmt.format(v), va="center", fontsize=8)
        ax.set_yticks(range(3), SHORT)
        ax.invert_yaxis()
        ax.set_xlim(0, max(vals) * 1.3)
        ax.set_title(title, fontsize=9)
    return png(fig, f"三句话和图片的相似度差得很少；乘上温度 {scale:.0f} 再过 softmax，差距才拉开。红色是正确答案。")


def fig_layers():
    fig, ax = plt.subplots(figsize=(5.0, 2.2), constrained_layout=True)
    xs = range(1, len(said_i) + 1)
    ax.plot(xs, [q[RIGHT].item() for q in said_i], "o-", color="#c8372d", label="图像塔只算到这一层")
    ax.plot(xs, [q[RIGHT].item() for q in said_t], "s--", color="#2f6fb5", label="文字塔只算到这一层")
    ax.set_xticks(list(xs))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("算到第几层")
    ax.set_ylabel(f"给“{SHORT[RIGHT]}”的概率")
    ax.legend(fontsize=8, frameon=False)
    return png(fig, "一座塔提前停下、另一座照常算完时，模型给正确答案的概率。可以看出答案是在哪几层成形的。")


LOOK, SIM, LAYERS = fig_look(), fig_sim(), fig_layers()
sim.update(name="相似度", type="点积 × 温度", kind="dense", origin_kind="named", figure=SIM,
           desc=["图片的向量和每句话的向量做点积，再乘上“温度”。", "- 长度都是 1，所以点积就是余弦相似度", "- 温度：一个学出来的数，把挤在一起的相似度拉开"],
           why="**两个向量方向越接近，说明这句话越像在描述这张图。** 不乘温度，三个相似度差得太少，softmax 分不出高下。", origin="**CLIP 的对比方式**：余弦相似度乘可学习的温度（Radford 等 2021）。",
           terms={"余弦相似度": "两个向量夹角的余弦：方向相同是 1，垂直是 0。", "温度": "相似度要乘的一个系数。越大，softmax 之后的概率越悬殊。"},
           note="余弦相似度：" + "、".join(f"{s} {c:.3f}" for s, c in zip(SHORT, cos)) + f"。乘上温度 {scale:.0f} 之后是 " + "、".join(f"{v:.1f}" for v in logits.tolist()) + "。",
           out={"shape": [3], "data": [round(v, 3) for v in logits.tolist()], "labels": SHORT},
           also={"乘温度之前的余弦相似度": {"shape": [3], "data": [round(c, 4) for c in cos], "labels": SHORT},
                 f"温度：模型里存的是 {raw['logit_scale'].item():.3f}，用的时候取指数": {"shape": [], "data": [round(scale, 2)]}})
sim["grad"] = {"shape": [3], "data": [round(v, 4) for v in (prob - F.one_hot(torch.tensor(RIGHT), 3)).tolist()], "labels": SHORT}  # d(loss)/d(score) = probability - one-hot answer
target.update(name="正确答案", desc="三句话里和图片相符的那一句。", why="**只用来算模型错了多少。** 模型自己判断时看不到它。", note="只在算损失时用到。", out={"tokens": [CAPTIONS[RIGHT]]})
loss.update(name="损失", type="cross_entropy", desc=["模型错了多少，用一个数表示。", "- 三个相似度过 softmax 变成概率", "- 给正确那句的概率越低，损失越大"], why="训练就是调整参数，让这个数变小：**让相符的图文更近，不相符的更远。**",
            note=f"概率：{verdict(prob)}。正确答案是“{CAPTIONS[RIGHT]}”，损失 = −ln({prob[RIGHT]:.3f}) = **{raw['loss'].item():.3f}**。")

second = int(prob.argsort(descending=True)[1])
choice = int(prob.argmax())
n_par = sum(q.numel() for q in model.parameters())
spec = {
    "title": "CLIP 给图片配文字",
    "source": "OpenAI CLIP · Hugging Face openai/clip-vit-base-patch32",
    "summary": "CLIP 有两座塔：一座看图，一座读文字。**哪句话的向量和图片的向量最接近，就认为哪句话在描述这张图。**",
    "example": f"一张两只猫躺在沙发上的照片，配三句话。模型选了 **“{CAPTIONS[choice]}”，把握 {prob.max():.0%}**；第二名是“{CAPTIONS[second]}”，{prob[second]:.0%}：沙发确实也在照片里。",
    "stats": [["模型选的", f"{SHORT[choice]} {'✓' if choice == RIGHT else '✗'}", CAPTIONS[choice]], ["把握", f"{prob.max():.0%}", f"第二名 {SHORT[second]} {prob[second]:.0%}"],
              ["向量长度", "512 个数", "图片和文字共用"], ["参数", f"{n_par:,}", "两座塔合计"], ["温度", f"{scale:.0f}", "学出来的"]],
    "background": [
        ["任务：**给一张图和几句话，找出哪句话在描述这张图。**", "- 换一组句子，就能拿来做别的分类，不用重新训练", "- 这叫“零样本”分类"],
        ["进去和出来：", "- 进去：一张 224×224 的照片，和三句英文", "- 出来：三个分数，**最高的那句就是模型的选择**"],
        ["模型的想法：**把图和文字都变成同一种向量，再看谁和谁方向最接近。**", "- 图像塔：图 → 512 个数", "- 文字塔：一句话 → 512 个数", "- 两座塔互相看不到对方"],
        ["怎么训练：", "- 4 亿对网上的“图片 + 配文”", "- 让配对的图文向量靠近，不配对的远离（对比学习）", "- 这里用的是 OpenAI 训练好的权重"],
    ],
    "glossary": {"CLIP": "OpenAI 2021 年的模型：Contrastive Language-Image Pre-training。", "Transformer": "一类以注意力为核心的网络结构。", "ViT": "Vision Transformer：把图切成块，当作一串“词”交给 Transformer。",
                 "对比学习": "训练方式：让应该相似的一对靠近，不该相似的远离。", "COCO": "一个常用的图片数据集，每张图配有人写的描述。", "推理": "模型训练好之后拿来用。"},
    "concepts": [
        {"name": "两座塔", "aliases": ["两座塔", "图像塔", "文字塔", "塔"], "short": "两个互不相通的网络：**一个只看图，一个只读文字**。", "figure": LAYERS,
         "table": {"head": ["", "输入", "切成", "宽度", "输出"], "rows": [["图像塔", "一张 224×224 的图", "49 个图块 + class", 768, "512 个数"], ["文字塔", "三句话", f"每句 {T} 个词元", 512, "每句 512 个数"]]},
         "text": ["- 结构相似：都是 12 层 Transformer；参数完全独立", "- 直到最后一步才相遇：比较两边的向量", "- 好处：图片的向量可以提前算好存起来，换句子不用重算"]},
        {"name": "向量和共同空间", "aliases": ["共用的空间", "向量"], "short": "图和文字最后都变成**长度为 1 的 512 个数**，可以直接比。", "figure": SIM,
         "text": ["- 两座塔各自的最后一步：投影到 512 维，再除以长度", "- 训练让“配对的图文”在这个空间里方向接近", "- 比较用点积：长度都是 1，所以点积就是余弦相似度"]},
        {"name": "class 位置和句末位置", "aliases": ["class 位置", "句末位置", "class 向量"], "short": "每座塔用**一个位置的结果**代表整张图或整句话。", "figure": LOOK,
         "table": {"head": ["塔", "用哪个位置", "为什么"], "rows": [["图像塔", "最前面的 class 位置", "它不属于任何图块，专门用来汇总"], ["文字塔", "句末的结束标记", "只能往前看，只有它看过整句"]]},
         "text": ["- 这个位置通过注意力，从其他位置收集信息", "- 12 层之后，取它的结果，其余位置不再使用"]},
        {"name": "注意力", "aliases": ["注意力权重", "注意力", "查询"], "short": "每个位置**按需要从别的位置取信息**。",
         "table": {"head": ["名字", "是什么"], "rows": [["查询 Q", "这个位置想找什么"], ["键 K", "这个位置有什么可被找到"], ["值 V", "被看中时交出去的内容"], ["权重", "Q 和 K 比出来的相似程度，一行加起来是 1"]]},
         "text": ["- 图像塔：50 个位置互相都能看", "- 文字塔：每个位置只能看自己和前面的", "- 12 个头同时做，各看各的"]},
    ],
    "levels": ["整体", "两座塔", "塔的内部", "十二层", "一层的内部", "每一步"],
    "root": G("root", [
        img, txt,
        G("两座塔", [tower_i, tower_t], parallel=True, type="图像塔 + 文字塔", origin_kind="named", figure=LOOK, desc=["图片和文字各走各的塔。", "- 图像塔：把图变成一个向量", "- 文字塔：把每句话变成一个向量", "- 两边互相看不到对方"],
          why="图片是像素，文字是词，没法直接比。**各自变成同一个空间里的向量，就能比了。**", origin=["**CLIP 的双塔结构**（Radford 等 2021）。", "- 也叫双编码器（dual encoder）"],
          note="一张图和三句话，各变成长度为 1 的 512 维向量。"),
        sim, target, loss,
    ]),
}
json.dump(spec, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)
print(f"{len(leaves)} leaves; {verdict(prob)}; loss {raw['loss'].item():.3f}; cos {[round(c, 3) for c in cos]}; scale {scale:.1f}")
