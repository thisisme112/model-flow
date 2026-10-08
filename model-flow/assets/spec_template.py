"""Glue for <project>: one <sample> through <model>, as a model-flow spec.

    <python> model-flow/spec.py model-flow/spec.json

Copy this file to model-flow/spec.py in the project and fill in the parts marked TODO. fxtrace.py sits next to it
(copied from the skill's scripts/). Complete scripts for four real projects are in the skill's examples/.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]  # fxtrace.py beside this file; the project one level up
from fxtrace import nest, tensor, trace  # noqa: E402,F401

# ---- 1. the model, loaded the way the project loads it ----------------------------------------------------------
# TODO: from the_project.model import Net
model = ...  # TODO: build it and load its weights; map_location="cpu"
model.eval()

# ---- 2. one real sample with a known right answer, chosen by a rule you can state ---------------------------------
# TODO: go through the project's own Dataset and transforms. Keep the batch dimension: x is [1, ...].
x = ...
y = ...  # the right answer, shaped the way the loss wants it
WHY = "…"  # TODO: one sentence for the page: why this sample (e.g. the one it gets right with the least confidence)

# ---- 3. trace ---------------------------------------------------------------------------------------------------
leaves, raw = trace(model, x, F.cross_entropy, y)  # TODO: the loss for this sample; hooks=True / max_hw=24 where needed
L = {n["id"]: n for n in leaves}
print(len(leaves), "steps:", " ".join(L))  # look at this once: these ids are what the rest of the script refers to
tree = nest(leaves, model)["children"]  # the same steps grouped by module call, as building blocks


def G(name, kids, **kw):
    """A group: a module of the figure. Give desc and note; add parallel=True when kids are branches."""
    return {"name": name, "type": "模块", "children": kids, **kw}


def first(n):
    return first(n["children"][0]) if "children" in n else n


def last(n):
    return last(n["children"][-1]) if "children" in n else n


def dims(t):
    """How to say a tensor's size in words: "64 张 7×7" for maps, "512 个数" for a vector."""
    s = list(t.shape[1:])
    return f"{s[0]} 张 {s[1]}×{s[2]}" if len(s) == 3 and s[1] * s[2] > 1 else f"{t.numel()} 个数"


def change(n):
    """What a step or a module turns its input into, from the real tensors."""
    return f"{dims(raw[first(n)['from'][0]])} 变成 {dims(raw[last(n)['id']])}。"


def drop(leaf, readers):
    """Take a step that only renames or reshapes out of the figure: whoever read it reads what it read."""
    for r in readers:
        r["from"] = [s for old in r.get("from", []) for s in (leaf["from"] if old == leaf["id"] else [old])]


# ---- 4. the words: a note for every step, from raw; a desc where the built-in one is missing or too general ---------
for n in leaves:
    t = raw.get(n["id"])
    if n["type"] == "Input" or t is None:
        continue
    if n["type"] in ("ReLU", "relu"):
        n["note"] = f"负数变成 0：这一步有 {(t == 0).float().mean():.0%} 的数被清零。"
    elif n.get("from"):
        n["note"] = change(n)
    # TODO: one branch per layer type that deserves a better sentence; see step() in examples/siamese_spec.py
# TODO: names and notes for the inputs, the target and the loss, e.g.
# L["x"].update(name="输入图像", note="…")
# L["target"].update(name="标签", desc="人工标注的正确答案。", out={"tokens": ["…"]})
# L["loss"].update(name="损失", type="cross_entropy", note=f"… {raw['loss'].item():.3f}。")

# ---- 5. the levels: regroup, most general first ---------------------------------------------------------------------
# TODO: replace this with the hierarchy you designed. Reuse the group nodes in `tree` (update their name / desc /
# note) and add groups of your own. Inputs and the label stay at the root so the overview shows them.
root = G("root", tree)

spec = {
    "title": "…",  # TODO: a short name
    "source": "…",  # TODO: repo · file
    "summary": f"…共 {sum(p.numel() for p in model.parameters()):,} 个参数。",  # TODO: what it is; one measured fact
    "example": f"… {WHY}",  # TODO: the sample, what the model said about it
    "levels": ["整体", "…", "每一层"],  # TODO: one name per slider position
    "root": root,
}
json.dump(spec, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)
print("wrote", sys.argv[1])
