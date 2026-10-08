"""Project glue for OpenAI CLIP (Hugging Face openai/clip-vit-base-patch32): one photo against three captions.

    python <this file> spec.json [photo.jpg]

Needs `transformers` and `pillow`; the first run downloads the model (about 600 MB) into the Hugging Face cache.
The photo defaults to demo/clip/000000039769.jpg of the development repository (COCO val2017: two cats on a couch).
fx cannot trace this model, so the steps come from fxtrace's hook path; the three captions travel through the text
tower together, as a batch of three.
"""
import base64
import io
import json
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from fxtrace import nest, tensor, trace  # noqa: E402

CAPTIONS = ["a photo of a cat", "a photo of a dog", "a photo of a couch"]  # each is 7 tokens, so nothing is padded and no mask is needed
SHORT, RIGHT = ["cat", "dog", "couch"], 0
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", attn_implementation="eager")  # eager: the attention weights are returned
PHOTO = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "..", "demo", "clip", "000000039769.jpg")
batch = proc(text=CAPTIONS, images=Image.open(PHOTO), return_tensors="pt", padding=True)
assert batch["attention_mask"].all(), "captions of different lengths would need the mask as an input"
ids, pixels = batch["input_ids"], batch["pixel_values"]
words = [proc.tokenizer.decode(t) for t in ids[RIGHT]]
T = ids.shape[1]

leaves, raw = trace(model, {"input_ids": ids, "pixel_values": pixels, "output_attentions": True},
                    lambda out, t: F.cross_entropy(out.logits_per_image, t), torch.tensor([RIGHT]), hooks=True, max_hw=24)
L = {n["id"]: n for n in leaves}
txt, img, vision, vproj, text, tproj, temp, tnorm, vnorm, sim, flip, target, loss = nest(leaves, model)["children"]
assert (vision["name"], text["name"], temp["name"], sim["type"], flip["type"]) == ("vision_model", "text_model", "logit_scale", "mul", "t"), [c["name"] for c in (vision, text, temp, sim, flip)]
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

NORM = "每个位置的数各自被调整到均值 0、方差 1 附近，再乘上学到的缩放。"


def layer(b, k, vision_side):
    """The words for one transformer layer. q, k and v become three lanes: they read the same input side by side."""
    ln1, att, add1, ln2, mlp, add2 = b["children"]
    q, kk, v, w, mix, out = att["children"]
    fc1, act, fc2 = mlp["children"]
    d, a = raw[q["id"]].shape[-1], raw[w["id"]]
    ln1["note"] = ln2["note"] = NORM
    for n, what in ((q, "查询 Q：这个位置想找什么"), (kk, "键 K：这个位置有什么可被找到"), (v, "值 V：被看中时，交出去的内容")):
        n.update(desc=f"一次线性变换，算出每个位置的{what}。", note=f"{d} 个数变成 {d} 个数。")
    if vision_side:
        row = a[0].mean(0)[0]  # where the class position looks, averaged over the heads
        r, c = divmod(int(row[1:].argmax()), 7)
        where = f"class 位置把 {row[0]:.0%} 的注意力留给自己，其余分给 49 个图块；图块里最受关注的是第 {r + 1} 行第 {c + 1} 列那块（{row[1:].max():.0%}）。"
    else:
        row = a[RIGHT].mean(0)[-1]  # where the last position of the right caption looks
        where = f"以“{CAPTIONS[RIGHT]}”为例：句末位置最关注的是“{words[int(row.argmax())]}”（{row.max():.0%}）。"
    w.update(name="注意力权重", type="softmax(QKᵀ)", kind="attn", out=tensor(a, max_c=4, max_hw=24), note=where,
             desc="每个位置的 Q 和各个位置的 K 做点积，再过 softmax：得到它该看每个位置多少。每张小图是一个头，第 i 行是第 i 个位置在看谁。" + ("" if vision_side else "文字这边每个位置只能看自己和它前面的位置，所以右上半边是空的。"))
    w.pop("grad", None)
    mix.update(name="加权求和", type="权重 · V", kind="attn", desc="按注意力权重把各个位置的 V 加起来，再把 12 个头的结果拼回一排。", note=f"每个位置重新得到 {d} 个数：现在里面混进了别的位置的信息。")
    out.update(desc="一次线性变换。", note=f"{d} 个数变成 {d} 个数。")
    add1["note"] = "残差连接：把注意力算出的结果加回主干。"
    fc1["note"] = f"每个位置各自算：{d} 个数扩成 {4 * d} 个。"
    act.update(desc="激活函数 QuickGELU：和 ReLU 类似，负数基本变成 0，但在 0 附近是平滑过渡。", note=f"这一步之后，有 {(raw[act['id']].abs() < 0.01).float().mean():.0%} 的数接近 0。")
    fc2["note"] = f"{4 * d} 个数压回 {d} 个。"
    add2["note"] = "残差连接：把前馈层的结果加回主干。"
    qkv = G("Q K V", [q, kk, v], parallel=True, type="三个线性变换", desc="同一份输入分三路，各做一次线性变换。", note="三路的输入完全相同，学到的变换不同。")
    att.update(name="注意力子层", children=[qkv, w, mix, out], desc="各个位置互相看：每个位置从别的位置收集信息。", note=where)
    mlp.update(name="前馈子层", desc="各个位置各自加工：两层全连接，中间过一次 QuickGELU。", note=f"{d} → {4 * d} → {d}。")
    said = (after_vision if vision_side else after_text)(raw[add2["id"]])
    b.update(name=f"第 {k + 1} 层", desc="一层 Transformer：先做注意力（各位置互相看），再过前馈层（各位置各自加工），两次的结果都加回主干。",
             note=f"如果这座塔只算到这一层（另一座塔照常算完），比较的结果会是：{verdict(said)}。")
    return b


# ---- image tower
emb, pre, enc, pick, post = vision["children"]
patch, pos, cls, add = emb["children"]
shown = (pixels[0] * torch.tensor(proc.image_processor.image_std)[:, None, None] + torch.tensor(proc.image_processor.image_mean)[:, None, None]).clamp(0, 1)
buf = io.BytesIO()
Image.fromarray((shown.permute(1, 2, 0) * 255).round().byte().numpy()).resize((112, 112)).save(buf, "JPEG", quality=85)
img.update(name="图片", out={"image": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), "shape": [3, 224, 224]},
           note="COCO 数据集里的一张照片：两只猫躺在沙发上，旁边有两个遥控器。已经按 CLIP 的要求缩放并裁成 224×224，再减均值、除以标准差。下面的梯度图显示哪些像素一变，损失就跟着变。")
patch["note"] = "一层 32×32、步长也是 32 的卷积：等于把图切成 7×7 = 49 块，每块各变成 768 个数。"
pos.update(desc="位置编码：每个位置各有一个学出来的向量，告诉模型这一块在图上的哪里。", note="50 个位置（49 块 + 1 个 class 位置）各 768 个数。只和位置有关，所以没有箭头进来。")
cls.update(name="class 向量", desc="一个学出来的向量。它不对应任何一块图，只是在 49 块前面占一个位置；后面 12 层里，这个位置负责把整张图的信息收集起来。", note="768 个数，对每张图都一样。")
add.update(name="拼接并加位置", note="49 个图块前面接上 class 向量，成为 50 个位置；再各自加上位置向量。")
pre["note"] = NORM
enc["children"] = [layer(b, k, True) for k, b in enumerate(enc["children"])]
enc.update(name="编码器", desc="12 层结构相同的 Transformer。", note="进去和出来都是 50×768：形状不变，内容一层层被改写。")
pick.update(name="取 class 位置", type="[:, 0]", desc="只留第 0 个位置（class）的结果，用它代表整张图。", note="50×768 里取出 768 个数。")
post["note"] = NORM
vproj.update(desc="一次线性变换，把图片的向量投到和文字共用的空间里。", note="768 个数变成 512 个。")
vnorm.update(name="归一化", type="÷ 长度", desc="除以向量自己的长度，使长度变成 1。这样两个向量的点积就是它们夹角的余弦。", note="512 个数，长度为 1。这就是这张图的“向量”。")
emb.update(name="切块和嵌入", desc="把图切成小块，每块变成一个向量，再标上位置。", note="3×224×224 的图变成 50×768。")
tower_i = G("图像塔", [emb, pre, enc, pick, post, vproj, vnorm], type="ViT-B/32", desc="一个视觉 Transformer：把图切成 49 块，过 12 层，最后给出代表整张图的 512 个数。",
            note="一张图变成长度为 1 的 512 维向量。")

# ---- text tower
temb, tenc, tln, tpick = text["children"]
tokv, tpos, tadd = temb["children"]
txt.update(name="三句话", out={"tokens": CAPTIONS}, note=f"三句候选的描述，一起送进文字塔。每句被切成 {T} 个词元（含开头和结尾标记），比如“{'｜'.join(words)}”。")
tokv["note"] = f"3 句话、每句 {T} 个词元，各换成 512 个数：3×{T}×512。图里三张叠在一起的小图就是这三句话。"
tpos.update(desc="位置编码：第 0、1、2… 个位置各有一个学出来的向量。", note="只和位置有关，所以没有箭头进来。")
tadd["note"] = "词元的向量加上它所在位置的向量。"
tenc["children"] = [layer(b, k, False) for k, b in enumerate(tenc["children"])]
tenc.update(name="编码器", desc="12 层结构相同的 Transformer。每个位置只能看自己和它前面的位置。", note=f"进去和出来都是 3×{T}×512。")
tln["note"] = NORM
tpick.update(name="取句末位置", type="[句末]", desc="每句话只留最后一个词元（结束标记）位置的结果。因为只能往前看，只有这个位置看过整句话，所以用它代表整句。", note=f"3×{T}×512 里每句取出 512 个数。")
tproj.update(desc="一次线性变换，把文字的向量投到和图片共用的空间里。", note="512 个数变成 512 个。")
tnorm.update(name="归一化", type="÷ 长度", desc="除以向量自己的长度，使长度变成 1。", note="三句话各得到一个长度为 1 的 512 维向量。")
temb.update(name="词元嵌入", desc="把词元变成向量，并标上位置。", note=f"3 句话变成 3×{T}×512。")
tower_t = G("文字塔", [temb, tenc, tln, tpick, tproj, tnorm], type="Transformer", desc="一个文字 Transformer：12 层，最后给出代表整句话的 512 个数。", note="三句话各变成长度为 1 的 512 维向量。")

# ---- the comparison
cos = (logits / scale).tolist()
# The temperature (logit_scale) is a parameter this step multiplies by. As a card of its own it would stand in the
# main line between the towers and the comparison, so it is shown inside the step instead.
sim["from"] = [s for s in sim["from"] if s != temp["id"]]
sim.update(name="相似度", type="点积 × 温度", kind="dense",
           desc="图片的向量和每句话的向量做点积（因为长度都是 1，这就是余弦相似度），再乘上“温度”：一个学出来的数，把挤在一起的相似度拉开，softmax 才分得出高下。",
           note="余弦相似度：" + "、".join(f"{s} {c:.3f}" for s, c in zip(SHORT, cos)) + f"。差得很少；乘上温度 {scale:.0f} 之后是 " + "、".join(f"{v:.1f}" for v in logits.tolist()) + "，差距就拉开了。",
           out={"shape": [3], "data": [round(v, 3) for v in logits.tolist()], "labels": SHORT},
           also={"乘温度之前的余弦相似度": {"shape": [3], "data": [round(c, 4) for c in cos], "labels": SHORT},
                 f"温度：模型里存的是 {raw['logit_scale'].item():.3f}，用的时候取指数": {"shape": [], "data": [round(scale, 2)]}})
sim["grad"] = {"shape": [3], "data": [round(v, 4) for v in (prob - F.one_hot(torch.tensor(RIGHT), 3)).tolist()], "labels": SHORT}  # d(loss)/d(score) = probability - one-hot answer
target.update(name="正确答案", desc="三句话里和图片相符的那一句。", note="只在算损失时用到。", out={"tokens": [CAPTIONS[RIGHT]]})
loss.update(name="损失", type="cross_entropy", note=f"三个相似度过 softmax 变成概率：{verdict(prob)}。正确答案是“{CAPTIONS[RIGHT]}”，损失 = −ln({prob[RIGHT]:.3f}) = {raw['loss'].item():.3f}。")

second = int(prob.argsort(descending=True)[1])
spec = {
    "title": "CLIP 给图片配文字",
    "source": "OpenAI CLIP · Hugging Face openai/clip-vit-base-patch32",
    "summary": f"CLIP 有两座塔：一座看图，一座读文字，各自把输入变成 512 个数。哪句话的向量和图片的向量最接近，模型就认为哪句话在描述这张图。共 {sum(q.numel() for q in model.parameters()):,} 个参数。",
    "example": f"一张两只猫躺在沙发上的照片，配三句话。模型选了“{CAPTIONS[int(prob.argmax())]}”，把握 {prob.max():.0%}；第二名是“{CAPTIONS[second]}”，{prob[second]:.0%}——沙发确实也在照片里。",
    "levels": ["整体", "两座塔", "塔的内部", "十二层", "一层的内部", "每一步"],
    "root": G("root", [
        img, txt,
        G("两座塔", [tower_i, tower_t], parallel=True, type="图像塔 + 文字塔", desc="图片和文字各走各的塔，互相看不到对方，直到最后各自变成同一个空间里的一个向量。",
          note="一张图和三句话，各变成长度为 1 的 512 维向量。"),
        sim, target, loss,
    ]),
}
json.dump(spec, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)
print(f"{len(leaves)} leaves; {verdict(prob)}; loss {raw['loss'].item():.3f}; cos {[round(c, 3) for c in cos]}; scale {scale:.1f}")
