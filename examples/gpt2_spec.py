"""Project glue for Hugging Face distilgpt2: one prompt, and the model's guess at its next token.

    python <this file> spec.json

Needs `transformers`; the first run downloads distilgpt2 (about 350 MB) into the Hugging Face cache.
fx cannot trace this model, so the steps come from fxtrace's hook path.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fxtrace import nest, tensor, trace  # noqa: E402

PROMPT, ANSWER = "Monday, Tuesday, Wednesday,", " Thursday"
tok = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2", attn_implementation="eager")  # eager: the attention weights can be asked for
ids = tok(PROMPT, return_tensors="pt").input_ids
target = torch.tensor(tok.encode(ANSWER))
assert target.numel() == 1, "the answer has to be a single token"
words = [tok.decode(t).strip() for t in ids[0]]
T = len(words)

leaves, raw = trace(model, {"input_ids": ids, "use_cache": False}, lambda out, t: F.cross_entropy(out.logits[:, -1], t), target, hooks=True)
with torch.no_grad():
    attn = model(input_ids=ids, output_attentions=True).attentions  # per layer: 12 heads x T x T
    lens = lambda h: model.lm_head(model.transformer.ln_f(h))[0, -1].softmax(-1)  # noqa: E731  what the model would say if it stopped here
x, body, head, tgt, loss = nest(leaves, model)["children"]
wte, wpe, add, drop, *blocks, ln_f = body["children"]
assert (wte["name"], drop["name"], len(blocks), ln_f["name"], head["name"]) == ("wte", "drop", 6, "ln_f", "lm_head"), [c["name"] for c in body["children"]]
p = raw["lm_head"][0, -1].softmax(-1)
top = p.topk(8)
best, second = (tok.decode(t).strip() for t in top.indices[:2])
said = lambda q: f"“{tok.decode(q.argmax()).strip()}”（{q.max():.0%}）"  # noqa: E731


def G(name, kids, **kw):
    return {"name": name, "type": "模块", "children": kids, **kw}


PASS = "训练时随机丢掉一部分数，推理时原样通过。"
x.update(name="句子的开头", out={"tokens": words}, note=f"“{PROMPT}”被切成 {T} 个词元（逗号也算一个）。模型要猜的是接下来的那个词。")
wte["note"] = f"{T} 个词元各换成 768 个数，得到 {T}×768。"
wpe.update(desc="位置编码：第 0、1、2… 个位置各有一个学出来的向量，告诉模型每个词元排在第几个。", note="只和位置有关，和句子内容无关，所以这一步没有箭头进来。")
add["note"] = "词元的向量加上它所在位置的向量。从这里开始，每个位置的 768 个数沿着一条“主干”一层层往后传。"
drop["note"] = PASS
for k, b in enumerate(blocks):
    ln1, att, add1, ln2, mlp, add2 = b["children"]
    c_attn, mix, proj, rd = att["children"]
    c_fc, act, c_proj, md = mlp["children"]
    w = attn[k][0].mean(0)[-1]  # where the last position looks, averaged over the 12 heads
    ln1["note"] = ln2["note"] = "每个位置的 768 个数各自被调整到均值 0、方差 1 附近，再乘上学到的缩放。"
    c_attn.update(kind="dense", desc="一次线性变换，同时算出每个位置的查询（Q）、键（K）、值（V）。", note="768 个数变成 2304 = 3×768 个：Q、K、V 各占一份。")
    mix.update(name="注意力", type="softmax(QKᵀ)·V", kind="attn",
               desc="每个位置拿自己的 Q 去比它和它之前所有位置的 K，算出该看谁（注意力权重），再按权重把那些位置的 V 加起来。12 个头各算一遍，最后拼回 768 个数。",
               note=f"最后一个位置（“{words[-1]}”）负责猜下一个词。在这一层，它平均把 {w.max():.0%} 的注意力放在第 {int(w.argmax()) + 1} 个词元“{words[int(w.argmax())]}”上，是最多的。",
               also={"注意力权重：每张小图是一个头，第 i 行是第 i 个位置在看谁": tensor(attn[k], max_c=12)})
    proj.update(kind="dense", desc="一次线性变换。", note="768 个数变成 768 个数。")
    rd["note"] = md["note"] = PASS
    add1["note"] = "残差连接：把注意力算出的结果加回主干。"
    c_fc.update(kind="dense", desc="全连接层。", note="每个位置各自算：768 个数扩成 3072 个。")
    act.update(desc="激活函数 GELU：和 ReLU 类似，负数基本变成 0，但在 0 附近是平滑过渡。", note=f"这一步之后，有 {(raw[act['id']].abs() < 0.01).float().mean():.0%} 的数接近 0。")
    c_proj.update(kind="dense", desc="全连接层。", note="3072 个数压回 768 个。")
    add2["note"] = "残差连接：把前馈层的结果加回主干。"
    att.update(name="注意力子层", desc="各个位置互相看：每个位置从它之前的位置收集信息。", note=mix["note"])
    mlp.update(name="前馈子层", desc="各个位置各自加工：两层全连接，中间过一次 GELU。", note="768 → 3072 → 768。")
    b.update(name=f"第 {k + 1} 层", desc="一层 Transformer：先做注意力（各位置互相看），再过前馈层（各位置各自加工），两次的结果都加回主干。",
             note=f"如果在这一层之后就直接输出，模型会猜 {said(lens(raw[add2['id']]))}。")
ln_f["note"] = "最后再归一化一次。"
labels = [tok.decode(t).strip() for t in top.indices]
scores = {"shape": [8], "data": [round(v, 3) for v in raw["lm_head"][0, -1][top.indices].tolist()], "labels": labels}
head.update(desc="输出层：把每个位置的 768 个数变成 50257 个分数，词表里每个词一个。这里画的是最后一个位置上分数最高的 8 个词，已经用 softmax 换算成了概率。",
            note=f"只看最后一个位置：它后面该接什么词。概率最高的是“{best}”，{top.values[0]:.0%}；其次是“{second}”，{top.values[1]:.0%}。",
            out={"shape": [8], "data": [round(v, 4) for v in top.values.tolist()], "labels": labels, "full": [T, p.numel()]},
            also={"换算成概率之前的原始分数": scores})
# the traced gradient is pooled across the vocabulary like the output; for the 8 words shown, d(loss)/d(score) is
# exactly probability minus one-hot answer
head["grad"] = {"shape": [8], "data": [round(v, 4) for v in (p - F.one_hot(target, p.numel())[0])[top.indices].tolist()], "labels": labels}
tgt.update(name="正确答案", desc="这句话真正的下一个词。", note="只在算损失时用到，不进入模型。", out={"tokens": [ANSWER.strip()]})
loss.update(name="损失", type="cross_entropy", note=f"正确的下一个词是“{ANSWER.strip()}”，模型给它的概率是 {p[target].item():.1%}。损失 = −ln({p[target].item():.3f}) = {raw['loss'].item():.3f}。")

emb = G("嵌入", [wte, wpe, add, drop], desc="把词元变成向量，并标上位置。", note=f"{T} 个词元变成 {T}×768 个数。这时候如果直接输出，模型会猜 {said(lens(raw[drop['id']]))}。")
spec = {
    "title": "distilgpt2 猜下一个词",
    "source": "Hugging Face · distilgpt2（transformers 的 GPT2LMHeadModel）",
    "summary": f"GPT-2 的蒸馏版：6 层 Transformer，共 {sum(q.numel() for q in model.parameters()):,} 个参数。给它一句话的开头，它给词表里 {p.numel()} 个词各打一个分，分最高的就是它猜的下一个词。",
    "example": f"“{PROMPT}”，正确的下一个词是“{ANSWER.strip()}”。模型猜的是“{best}”，把握 {top.values[0]:.0%}；第二名是“{second}”，{top.values[1]:.0%}。",
    "levels": ["整体", "六层", "一层的内部", "每一步"],
    "root": G("root", [
        x,
        G("distilgpt2", [emb, *blocks, ln_f, head], type="GPT2LMHeadModel", desc="词元先变成向量，穿过 6 层结构相同的 Transformer，最后变成对下一个词的打分。",
          note=f"模型最看好的下一个词是“{best}”，{top.values[0]:.0%}；其次是“{second}”，{top.values[1]:.0%}。"),
        tgt, loss,
    ]),
}
json.dump(spec, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)
print(f"{T} tokens {words}; top: {[(w, round(v.item(), 3)) for w, v in zip(labels[:3], top.values)]}; loss {raw['loss'].item():.3f}; {len(leaves)} leaves")
