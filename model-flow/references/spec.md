# The spec

One JSON object. `build.py` embeds it in the viewer; nothing else is needed to make a page.

```jsonc
{
  "title": "MNIST 孪生网络",                 // the page's name: short, no subtitle
  "source": "pytorch/examples · siamese_network/main.py",
  "summary": "what the model is, how big, one measured fact",
  "example": "the one sample being traced, what the model said, why this sample",
  "background": ["任务是什么…", "进去什么、出来什么…", "这个模型的想法…"],   // paragraphs under "读图之前", for a newcomer
  "glossary": { "dev 集": "…" },             // terms of the whole page; merged with every node's "terms" into one list
  "concepts": [ { "name": "五个问题",        // things many boxes refer to: one card each below the figure, linked from every text
                  "aliases": ["问题向量", "问题"],      // other spellings the texts use (two characters or more)
                  "short": "…",              // the hover line; default: the first sentence of text
                  "text": ["…", "…"],        // what it is, what it does; a string or a list of paragraphs
                  "table": { "head": ["问题", "能看的区域"], "rows": [["高频 hf", "…"]] },   // optional: one row per member
                  "after": "…" } ],          // optional small print under the table
  "stats": [["模型的判断", "real ✓", "标注是 real"], ["参数", "2,221,931"]],   // tiles under the title: label, value, small print
  "levels": ["整体", "两座塔", "每一层"],     // optional names for the slider positions, most general first
  "root": { "name": "root", "children": [NODE] },      // root itself is never drawn

  "variants": {                              // optional: the same steps traced again; adds a slider
    "label": "训练进度",
    "items": [ { "name": "第 100 步 · 准确率 95%",
                 "example": "…",             // replaces the top-level example while this item is shown
                 "nodes": { "conv1": { "out": TENSOR, "grad": TENSOR, "note": "…" } } } ] }
                                             // keyed by node id; the last item is the one shown first
}
```

## NODE

A group if it has `children`, otherwise one operation. Children are in an order where every step comes after the
steps it reads.

```jsonc
{ "id": "conv1",            // optional. Default: the name at the top level, otherwise "parent id/name". Must be unique.
  "name": "conv1",          // on the box. Keep the code's name for leaves.
  "type": "Conv2d",         // under the name; also picks the built-in description and the colour
  "desc": "…",              // 做什么: what this step does in general
  "why": "…",               // 为什么需要: what it is for in this model. Built in for common layers; write it for groups
  "origin": "…",            // 来历: a fixed formula / a standard layer (its name) / a named architecture / the project's own
  "params": 18816,          // learned numbers in this step; 0 = a fixed computation. The page adds groups up itself
  "module": "cnn.lf.b1.0",  // which module or parameter "params" counts, so one that is called twice is added once
  "origin_kind": "fixed",   // a badge: fixed 固定计算 · standard 标准做法 · named 知名结构 · own 项目自创 (standard layers get theirs)
  "mini": "pool",           // a small drawing of what the step does, for a step of the project's own that works like a known
                            // layer: conv pool relu gelu sigmoid linear norm attention add cat softmax dropout
  "figure": { "image": "data:image/png;base64,…", "caption": "…" },   // a picture drawn from this example's tensors; or a list
  "terms": { "帧": "…" },   // words the texts of this node use that a beginner has not met
  "note": "…",              // what happened to this example here
  "from": ["relu", "stem"], // ids this step reads. Default: the previous leaf. A group's id means the group's last leaf.
                            // On a group it applies to the group's first leaf.
  "out": TENSOR,            // this step's output for the example
  "grad": TENSOR,           // d(loss)/d(out); a bare number is taken as the norm
  "also": { "注意力权重": TENSOR },   // further results of this step, shown in the panel under these labels
  "src": "main.py:21",      // where in the code this step runs
  "kind": "conv",           // colour: conv pool dense act norm attn shape loss. Guessed from type when absent.
  "parallel": true,         // on a group: its children are branches, drawn as lanes one under the other
  "children": [NODE] }
```

A leaf with `"type": "Input"` is data entering from outside: the sample, the label, a learned table. It has a plain
card, takes no playback step, and may sit inside the group that uses it.

`desc`, `why`, `origin`, each `background` item and a concept's `text` may be a list of strings instead of one
string: every string is a paragraph, and one that starts with `"- "` is a point of a list; a point that opens with
a short label and a colon gets the label in bold. A concept may also have a `figure`.

Texts (`summary`, `example`, `background`, `desc`, `why`, `origin`, `note`) may use `**重点**` for the phrase to
remember and `` `name` `` for a name from the code; nothing else is interpreted. A word that is a key (or a
space- or slash-separated part of a key, two characters or more) of any `terms` or of `glossary` is underlined on
its first occurrence in a text and shows its meaning on hover. A `background` paragraph that opens with a label of
up to 12 characters and a colon gets the label as a coloured heading.

`build.py` rejects a spec in which two nodes share an id or a `from` names no node.

## TENSOR

One sample, no batch dimension.

```jsonc
{ "shape": [8, 26, 26],
  "data": [ /* flat, row-major */ ],
  "full": [32, 26, 26],        // the true shape, when channels were dropped or a map was shrunk
  "labels": ["cat", "dog"],    // names for the entries of a short vector
  "tokens": ["The", "cat"],    // text; instead of data
  "image": "data:image/png;base64,…" }   // a picture; instead of data
```

How it is drawn:

| Tensor | On the box | In the panel |
|---|---|---|
| one number | the number | the number, large |
| a vector of up to 24 | bars (up to 16) or a strip | labelled bars, signed around a centre line |
| a longer vector | a strip | a grid of cells |
| 2-D and more | the first three channels as stacked maps | up to 16 channels as heat maps; up to 10×10 with at most 2 channels as tables of numbers |
| `tokens` | the first few | every token as a chip |
| `image` | the picture | the picture |

The last two dimensions are height and width; everything before them is channels. Ink is positive, red is negative,
depth of colour is size relative to the largest value in that tensor. A sequence (`T×D` with few rows) is drawn
taller than its true proportions so the rows can be told apart.

`data` omitted: the panel says that only the shape was recorded. That is how a model that could not be run is
shown.

## Size

`fxtrace.tensor` keeps at most 8 channels of 32×32 and vectors up to 1024 by default. Keep a page under about 3 MB;
`build.py` warns above that. See "Page weight" in `tracing.md`.

## The page itself

- Follows the viewer's light or dark setting; the button at the top right switches and remembers.
- `page.html#level-3` opens at that slider position.
- "导出 SVG" saves the figure as it stands (level, opened modules) as plain vector shapes, always ink on white.
- ← and → step, space plays and pauses.
- `flowCheck()` and `flowCheckAll()` in the page's console list arrows that run through a box, hug one, lie on an
  unrelated arrow or miss their target. `scripts/check.py` calls them for you.
