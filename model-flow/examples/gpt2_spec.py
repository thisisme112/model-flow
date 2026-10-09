"""Project glue for Hugging Face distilgpt2: one prompt, and the model's guess at its next token.

    python <this file> spec.json

Needs `transformers` and matplotlib; the first run downloads distilgpt2 (about 350 MB) into the Hugging Face cache.
fx cannot trace this model, so the steps come from fxtrace's hook path.
"""
import json
import os
import sys

import matplotlib
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from figure import picture  # noqa: E402
from fxtrace import nest, tensor, trace  # noqa: E402

plt.rcParams.update({"font.sans-serif": ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "PingFang SC", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 9})
NAME = "distilgpt2"
PROMPT, ANSWER = "Monday, Tuesday, Wednesday,", " Thursday"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, attn_implementation="eager")  # eager: the attention weights can be asked for
ids = tok(PROMPT, return_tensors="pt").input_ids
target = torch.tensor(tok.encode(ANSWER))
assert target.numel() == 1, "the answer has to be a single token"
words = [tok.decode(t).strip() for t in ids[0]]
T = len(words)

leaves, raw = trace(model, {"input_ids": ids, "use_cache": False, "output_attentions": True}, lambda out, t: F.cross_entropy(out.logits[:, -1], t), target, hooks=True)
with torch.no_grad():
    lens = lambda h: model.lm_head(model.transformer.ln_f(h))[0, -1].softmax(-1)  # noqa: E731  what the model would say if it stopped here
x, body, head, tgt, loss = nest(leaves, model)["children"]
wte, wpe, add, drop, *blocks, ln_f = body["children"]
assert (wte["name"], drop["name"], len(blocks), ln_f["name"], head["name"]) == ("wte", "drop", 6, "ln_f", "lm_head"), [c["name"] for c in body["children"]]
p = raw["lm_head"][0, -1].softmax(-1)
top = p.topk(8)
best, second = (tok.decode(t).strip() for t in top.indices[:2])
right = ANSWER.strip()
said = lambda q: f"“{tok.decode(q.argmax()).strip()}”（{q.max():.0%}）"  # noqa: E731
labels = [tok.decode(t).strip() for t in top.indices]
D = raw[wte["id"]].shape[-1]  # 768: the width of the trunk


def G(name, kids, **kw):
    return {"name": name, "type": "模块", "children": kids, **kw}


def fig_top():
    fig, ax = plt.subplots(figsize=(4.6, 2.2), constrained_layout=True)
    vals = top.values.tolist()
    ax.barh(range(8), vals, color=["#c8372d" if w == right else "#2f6fb5" for w in labels])
    for k, v in enumerate(vals):
        ax.text(v + 0.01, k, f"{v:.0%}", va="center", fontsize=8)
    ax.set_yticks(range(8), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.2)
    ax.set_xlabel("模型给的概率")
    ax.set_title(f"“{PROMPT}” 后面接什么？", fontsize=10)
    return picture(fig, f"词表里 {p.numel()} 个词中概率最高的 8 个。红色是正确答案。")


after = [lens(raw[drop["id"]])] + [lens(raw[b["children"][-1]["id"]]) for b in blocks]  # after the embedding, then after each block


def fig_layers():
    """The answer taking shape: what the model would say if it stopped after each layer."""
    fig, ax = plt.subplots(figsize=(5.0, 2.3), constrained_layout=True)
    ys = [q[target].item() for q in after]
    ax.plot(range(len(ys)), ys, "o-", color="#c8372d")
    for k, q in enumerate(after):
        ax.annotate(tok.decode(q.argmax()).strip() or "␣", (k, ys[k]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color="#1b2a41")
    ax.set_xticks(range(len(ys)), ["嵌入后"] + [f"第 {k} 层后" for k in range(1, len(ys))])
    ax.set_ylim(-0.03, max(ys) * 1.35 + 0.05)
    ax.set_ylabel(f"给“{right}”的概率")
    return picture(fig, "如果算到这一层就直接输出：红线是给正确答案的概率，点上面的字是那时模型会猜的词。")


def fig_attn():
    a = raw[blocks[-1]["children"][1]["children"][1]["id"]][0].mean(0)  # last layer, mean of the heads: [T, T]
    fig, ax = plt.subplots(figsize=(3.6, 3.0), constrained_layout=True)
    ax.imshow(a.numpy(), cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(T), words, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(T), words, fontsize=8)
    ax.set_xlabel("被看的位置")
    ax.set_ylabel("在看的位置")
    return picture(fig, f"第 {len(blocks)} 层的注意力（12 个头的平均）。每一行是一个位置在看谁；只能看自己和前面，所以右上角是空的。")


TOP, LAYERS, ATTN = fig_top(), fig_layers(), fig_attn()
PASS = "现在是推理，数据原样通过。"
x.update(name="句子的开头", out={"tokens": words}, desc=["一段文字，已经切成词元。", "- 模型要猜的是接下来的那个词元"], note=f"“{PROMPT}”被切成 {T} 个词元（逗号也算一个）。",
         terms={"词元 token": "文字被切成的小片：一个词、半个词或一个标点。模型只认词元的编号。"})
wte.update(desc=["查表：每个词元的编号换成一个向量。", f"- 词表里有 {p.numel()} 个词元，每个对应 {D} 个数"], why="编号本身没有含义。**换成向量后，意思相近的词可以有相近的向量。**", note=f"{T} 个词元各换成 {D} 个数，得到 {T}×{D}。")
wpe.update(desc="位置编码：第 0、1、2… 个位置各有一个学出来的向量。", why="注意力本身不分先后。**不加它，模型不知道哪个词在前、哪个在后。**", note="只和位置有关，和句子内容无关，所以没有箭头进来。")
add.update(why="**把“是什么词”和“在第几个”合成一份。** 从这里起，每个位置有一份数据，一层层往后传。", note=f"词元的向量加上位置的向量。结果是 {T}×{D}，这就是“主干”的起点。")
drop["note"] = PASS
for k, b in enumerate(blocks):
    ln1, att, add1, ln2, mlp, add2 = b["children"]
    c_attn, wts, *extra, mix, proj, rd = att["children"]  # some transformers versions add a dropout on the weights: extra
    assert (c_attn["name"], proj["name"], rd["type"]) == ("c_attn", "c_proj", "Dropout"), [c["name"] for c in att["children"]]
    for n in extra:
        n["note"] = PASS
    c_fc, act, c_proj, md = mlp["children"]
    w = raw[wts["id"]][0].mean(0)[-1]  # where the last position looks, averaged over the 12 heads
    ln1["note"] = ln2["note"] = f"每个位置的 {D} 个数调整到均值 0、方差 1 附近。"
    c_attn.update(kind="dense", origin_kind="standard", mini="linear", desc=["一次线性变换，同时算出三样东西：", "- 查询 Q：这个位置想找什么", "- 键 K：这个位置有什么可被找到", "- 值 V：被看中时交出去的内容"],
                  why="注意力要拿 Q 去和 K 比，再按结果取 V。**所以先把每个位置的数据变成这三份。**", origin="代码里叫 `Conv1D`，其实就是一个全连接层（GPT-2 的历史写法）。",
                  note=f"{D} 个数变成 {3 * D} = 3×{D} 个：Q、K、V 各一份。")
    wts.update(name="注意力权重", type="softmax(QKᵀ)", kind="attn", origin_kind="standard", mini="attention", out=tensor(raw[wts["id"]], max_c=12),
               desc=["每个位置决定“看谁、看多少”。", "- 拿自己的 Q 和各个位置的 K 比相似程度", "- 过 softmax：一行加起来是 1", "- 只能看自己和前面的位置"],
               why="猜下一个词要用到前文。**哪些前文重要，由这一步决定。**", origin="**缩放点积注意力**（scaled dot-product attention），Transformer 的核心。",
               note=f"最后一个位置（“{words[-1]}”）负责猜下一个词。这一层它最关注第 {int(w.argmax()) + 1} 个词元“{words[int(w.argmax())]}”，占 {w.max():.0%}。")
    mix.update(name="加权求和", type="权重 · V", kind="attn", origin_kind="standard", desc=["按注意力权重，把各个位置的 V 加起来。", f"- 12 个头的结果拼回 {D} 个数"], why="上一步只决定了“看谁”。**这一步才真的把信息取过来。**",
               note=f"每个位置重新得到 {D} 个数，里面混进了前面位置的信息。")
    proj.update(kind="dense", origin_kind="standard", desc="一次线性变换。", note=f"{D} 个数变成 {D} 个数。")
    rd["note"] = md["note"] = PASS
    add1.update(why="**残差连接**：注意力的结果是“往主干上添一点”，不是替换它。", note="把注意力算出的结果加回主干。")
    c_fc.update(kind="dense", origin_kind="standard", desc="全连接层：每个位置各自算。", note=f"{D} 个数扩成 {4 * D} 个。")
    act.update(kind="act", origin_kind="standard", mini="gelu", desc="激活函数 GELU：负数压到接近 0，正数基本不变。", note=f"这一步之后，{(raw[act['id']].abs() < 0.01).float().mean():.0%} 的数接近 0。")
    c_proj.update(kind="dense", origin_kind="standard", desc="全连接层。", note=f"{4 * D} 个数压回 {D} 个。")
    add2.update(why="**残差连接**：前馈层的结果也是往主干上添。", note="把前馈层的结果加回主干。")
    att.update(name="注意力子层", origin_kind="standard", mini="attention", desc=["各个位置互相看：每个位置从它前面的位置收集信息。", "- 算出 Q、K、V", "- 算注意力权重", "- 按权重把 V 加起来"],
               why="每个位置原本只知道自己这个词。**猜下一个词要用到整句话，信息靠这一步在位置之间流动。**", origin="**多头自注意力**（multi-head self-attention），这里 12 个头。",
               terms={"头": "一组独立的注意力。12 个头各看各的，再把结果拼起来。", "自注意力": "一组向量自己看自己：这里是句子里的各个位置互相看。"}, note=wts["note"])
    mlp.update(name="前馈子层", origin_kind="standard", mini="linear", desc=["各个位置各自加工，互不相看。", f"- {D} → {4 * D}，过 GELU，再 → {D}"], why="注意力只是把信息搬过来。**这一步对搬来的信息做加工。**",
               origin="Transformer 每层里**标准的前馈子层**（feed-forward / MLP）。", note=f"{D} → {4 * D} → {D}。")
    b.update(name=f"第 {k + 1} 层", origin_kind="standard", desc=["一层 Transformer。", "- 注意力：各位置互相看", "- 前馈：各位置各自加工", "- 两步的结果都加回主干"],
             why="一层能做的加工有限。**六层叠起来，一步步把“下一个词是什么”算出来。**", origin="**Transformer 解码器的一层**（pre-norm）。六层结构完全相同，参数各不相同。",
             note=f"如果在这一层之后就直接输出，模型会猜 {said(after[k + 1])}。")
ln_f.update(note="最后再归一化一次。")
scores = {"shape": [8], "data": [round(v, 3) for v in raw["lm_head"][0, -1][top.indices].tolist()], "labels": labels}
head.update(origin_kind="standard", mini="linear", figure=TOP, desc=["输出层：给词表里每个词元打一个分。", f"- 每个位置的 {D} 个数变成 {p.numel()} 个分数", "- 这里画的是最后一个位置上最高的 8 个，已换算成概率"],
            why="模型最终要回答“下一个词是哪个”。**给每个候选打分，分最高的就是答案。**", origin="一个全连接层。它的权重和 `wte` 是同一份（权重共享）。",
            note=f"最后一个位置：最高的是 **“{best}”，{top.values[0]:.0%}**；其次是“{second}”，{top.values[1]:.0%}。",
            out={"shape": [8], "data": [round(v, 4) for v in top.values.tolist()], "labels": labels, "full": [T, p.numel()]},
            also={"换算成概率之前的原始分数": scores})
# the traced gradient is pooled across the vocabulary like the output; for the 8 words shown, d(loss)/d(score) is
# exactly probability minus one-hot answer
head["grad"] = {"shape": [8], "data": [round(v, 4) for v in (p - F.one_hot(target, p.numel())[0])[top.indices].tolist()], "labels": labels}
tgt.update(name="正确答案", desc="这句话真正的下一个词元。", why="训练要知道正确答案，才能算出模型错了多少。**模型自己猜的时候看不到它。**", note="只在算损失时用到，不进入模型。", out={"tokens": [right]})
loss.update(name="损失", type="cross_entropy", desc=["模型错了多少，用一个数表示。", "- 只看模型给正确答案的概率", "- 概率越低，损失越大"], why="训练就是调整参数，让这个数变小。",
            note=f"正确答案是“{right}”，模型给它 {p[target].item():.1%}。损失 = −ln({p[target].item():.3f}) = **{raw['loss'].item():.3f}**。")

emb = G("嵌入", [wte, wpe, add, drop], origin_kind="standard", desc=["把词元变成向量，并标上位置。", "- `wte`：是什么词", "- `wpe`：在第几个位置", "- 两者相加"],
        why="网络只会算数。**文字要先变成数，才能进网络。**", origin="**词嵌入 + 学出来的位置编码**，GPT 的标准开头。", terms={"嵌入": "把一个编号换成一个向量。向量里的数是学出来的。"},
        note=f"{T} 个词元变成 {T}×{D} 个数。这时候如果直接输出，模型会猜 {said(after[0])}。")
n_par = sum(q.numel() for q in model.parameters())
spec = {
    "title": "distilgpt2 猜下一个词",
    "source": "Hugging Face · distilgpt2（transformers 的 GPT2LMHeadModel）",
    "summary": "GPT-2 的小号版本：6 层 Transformer。**给它一句话的开头，它猜下一个词。**",
    "example": f"“{PROMPT}”，正确的下一个词是“{right}”。模型猜 **“{best}”，把握 {top.values[0]:.0%}**；第二名是“{second}”，{top.values[1]:.0%}。",
    "stats": [["模型猜的下一个词", f"{best} {'✓' if best == right else '✗'}", f"正确答案 {right}"], ["把握", f"{top.values[0]:.0%}", f"第二名 {second} {top.values[1]:.0%}"], ["输入", f"{T} 个词元", PROMPT],
              ["参数", f"{n_par:,}", "6 层"], ["候选", f"{p.numel():,} 个", "词表大小"]],
    "background": [
        ["任务：**给一段文字的开头，猜下一个词元是什么。**", "- 这就是“语言模型”做的事", "- 把猜出来的词接上去再猜，就能一直写下去"],
        ["进去和出来：", f"- 进去：{T} 个词元（一句话切成的小片）", f"- 出来：词表里 {p.numel()} 个词元各一个分数，**分最高的就是模型的答案**"],
        ["模型的想法：**每个位置一份数据，沿着“主干”穿过六层。**", "- 每层先让各个位置互相看（注意力）", "- 再让每个位置各自加工（前馈）", "- 猜下一个词，只看最后一个位置"],
        ["怎么训练：", "- 拿大量文字，遮住下一个词让模型猜", "- 这里用的是别人训练好的权重，没有再训练", "- distilgpt2 是用“蒸馏”从 GPT-2 压缩来的"],
    ],
    "glossary": {"GPT-2": "OpenAI 2019 年发布的语言模型。distilgpt2 是它的蒸馏版：层数减半。", "Transformer": "一类以注意力为核心的网络结构，现在的语言模型大多是它。", "蒸馏": "让一个小模型学着模仿大模型的输出，得到一个更小更快的模型。",
                 "词表": "模型认识的全部词元的清单。", "推理": "模型训练好之后拿来用。和训练相对。"},
    "concepts": [
        {"name": "词元", "aliases": ["词元", "token"], "short": "文字被切成的小片。**模型看到的不是字，是词元的编号。**",
         "table": {"head": ["第几个", "词元"], "rows": [[k + 1, w] for k, w in enumerate(words)]},
         "text": ["- 一个词元可以是一个词、半个词或一个标点", "- 图上的“位置”就是指第几个词元", "- 模型的输出也是词元：一次猜一个"]},
        {"name": "主干", "aliases": ["主干", "残差连接"], "short": f"每个位置有一份 {D} 个数的数据，**从嵌入一直传到输出层**。", "figure": LAYERS,
         "text": ["- 每一层不替换主干，只往上面“加一点”（残差连接）", "- 图上跨过几个方框的弧线，就是主干绕过这一步的那条路", "- 所以可以在任何一层之后直接看“现在会猜什么”"]},
        {"name": "注意力", "aliases": ["注意力权重", "注意力", "查询", "Q", "K", "V"], "short": "每个位置**按需要从别的位置取信息**。", "figure": ATTN,
         "table": {"head": ["名字", "是什么"], "rows": [["查询 Q", "这个位置想找什么"], ["键 K", "这个位置有什么可被找到"], ["值 V", "被看中时交出去的内容"], ["权重", "Q 和 K 比出来的相似程度，一行加起来是 1"]]},
         "text": ["- 每个位置只能看自己和前面的位置（不能偷看答案）", "- 12 个头同时做，各看各的"]},
        {"name": "六层和输出", "aliases": ["六层"], "short": "六层结构相同；**最后一个位置的结果决定下一个词**。", "figure": TOP,
         "text": ["- 每层：注意力 + 前馈", "- 最后把最后一个位置的数据变成对每个词元的打分", "- 点开任意一层，看“如果在这里就输出会猜什么”"]},
    ],
    "levels": ["整体", "六层", "一层的内部", "每一步"],
    "root": G("root", [
        x,
        G("distilgpt2", [emb, *blocks, ln_f, head], type="GPT2LMHeadModel", origin_kind="named", figure=LAYERS, desc=["词元先变成向量，穿过 6 层，变成对下一个词的打分。", "- 嵌入", "- 6 层 Transformer", "- 输出层"],
          why="**这就是整个语言模型。**", origin=["**GPT-2 的蒸馏版**（Sanh 等 2019）。", "- GPT-2：OpenAI 2019，只有解码器的 Transformer", "- 这里是 6 层、12 个头、宽度 768"],
          note=f"模型最看好 **“{best}”，{top.values[0]:.0%}**；其次是“{second}”，{top.values[1]:.0%}。"),
        tgt, loss,
    ]),
}
json.dump(spec, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)
print(f"{T} tokens {words}; top: {[(w, round(v.item(), 3)) for w, v in zip(labels[:3], top.values)]}; loss {raw['loss'].item():.3f}; {len(leaves)} leaves")
