# The glue script: levels, branches, words

Contents: what the script does · designing levels (four worked cases) · branches · inputs and parameters · repeats
· steps to drop or rename · writing the words · drawing awkward data · before you build.

## What the script does

`trace` gives a flat list of steps with edges. A reader needs a figure with a few named parts that open into
smaller parts. The glue script is where a person's understanding of the model goes in: it **regroups** the steps
and **writes the words**. Start from `assets/spec_template.py`.

The usual shape:

```python
leaves, raw = trace(model, x, loss_fn, target)
a, b, *rest = nest(leaves, model)["children"]     # take the module tree apart…
G = lambda name, kids, **kw: {"name": name, "type": "模块", "children": kids, **kw}
root = G("root", [a, G("特征提取", [...], desc="…", note="…"), ...])   # …and put the levels together
```

`nest` already groups by module, which is often two thirds of the work: reuse its group nodes (`update()` their
name, desc and note) and add groups of your own where the code has none (a "stem", an "embedding", "main path").
Edges are stored on the leaves as ids, so moving a leaf into another group never breaks an arrow.

Two rules keep the tree valid:

- **Every step comes after the steps it reads.** Tree order is playback order. You may move a step (the
  normalisation that ends each CLIP tower runs after both towers, but belongs at the end of its own lane) as long as
  this holds.
- **Ids are unique.** Leaves have them from the tracer. A group you make gets the id `parent id/name` unless you give
  one; give one when you need to refer to it (`variants`) or when two siblings share a name.

## Designing levels

A level is one position of the slider: everything down to that depth is open. Each should be a picture worth
looking at by itself, and each step from one level to the next should add one idea.

| Project | Levels | Why |
|---|---|---|
| MNIST CNN, 13 layers in a row (`examples/mnist_spec.py`) | 整体 → 两个阶段 (特征提取, 分类头) → 功能块 (卷积块 1/2, 下采样, 隐藏层, 输出层) → 每一层 | the code has no modules at all, so every group was invented from what the layers do together |
| Siamese ResNet-18 (`siamese_spec.py`) | 整体 → 两座塔 → 塔里的阶段 → 残差块 → 块的内部 → 主路和捷径 → 每一层 | deep nesting in the code; each code level became a slider level; the two towers and each projection shortcut are branches |
| distilgpt2 (`gpt2_spec.py`) | 整体 → 六层 → 一层的内部 → 每一步 | six identical blocks: the block level shows six boxes, the next opens one |
| CLIP (`clip_spec.py`) | 整体 → 两座塔 → 塔的内部 → 十二层 → 一层的内部 → 每一步 | two different towers side by side; inside each layer Q, K, V are three single-step lanes |

Guidelines that came out of those:

- **Level 1 is input → model → loss**, with both ends visible. Put the sample and the label at the root, outside every
  group, so the overview shows what goes in.
- **Aim for 4 to 8 boxes per newly opened group.** A group of 20 leaves needs a level in between; a group of one
  is a level that adds nothing.
- **Name `levels`** after what appears ("两座塔", "十二层", "每一步"), not "level 3".
- **Follow the code where the code has structure**, invent groups where it does not. A reader who opens the source
  should recognise the page.
- The slider opens groups by depth. A group nested one level deeper than its siblings opens one stop later; if that
  looks odd, flatten it or wrap the siblings.

## Branches

Mark a group `"parallel": True` when its children are paths that run beside each other and (usually) meet again.
They are drawn as lanes, one under the other, in a box with a gutter on each side for the arrows.

Use it for:

- **towers**: two encoders over two inputs (siamese, CLIP, any dual-encoder);
- **a main path beside a shortcut that does work**: the 1×1 convolution on a downsampling residual block. The group is
  `[main path group, shortcut group]`, followed outside by the `add`;
- **fan-out to single steps**: the Q, K and V projections of an attention layer read the same input. Three lanes of
  one step each. Lanes that are single steps open together with the module around them, with no slider stop of
  their own;
- **several heads** over one feature map (classification and box regression).

Do not use it for an identity skip connection: there is no second path to draw, only an edge, and the viewer draws
it as an arc over the steps it skips.

Lanes are played one after the other, top to bottom, so their order must respect the edges like everything else.

When branches are missing the symptom is three or four arcs fanning out of one box over each other, and
`check.py` reporting overlaps.

## Inputs and parameters

A leaf with `"type": "Input"` is data from outside: the sample, the label, a learned table. It is drawn as a card
without a coloured head, and it is not a playback step.

- **The sample and the label** go at the root.
- **A learned parameter used directly** (a class token, a position table added by hand) is traced as an input.
  Leave it inside the group that uses it, right before the step that reads it.
- **A single number** that would stand in the main line between two steps (CLIP's temperature between the towers
  and the similarity) is better folded into the step: remove the leaf and its id from the reader's `from`, and put
  the value under `also` with a label that says what it is.
- **Two inputs feeding two towers**: both at the root, before the parallel group, in the same order as the lanes.

## Repeats

Leave repeated blocks as siblings and let the viewer fold them: of three or more siblings whose subtrees have the
same types, the slider opens the first and labels it ×N. Give every block its notes anyway (a reader can open any
of them), and make the per-block note worth opening for: "if the model stopped after this layer it would say …"
(run the remaining head on that block's output; both transformer examples do).

Blocks differ in type signature when one has a shortcut and the others do not; then the fold starts at the second.
That is correct, not a problem to fix.

## Steps to drop or rename

The tracer reports what ran. Not everything that ran deserves a box:

- **A step that only re-arranges** (a transpose before the loss, a reshape to the same numbers): drop the leaf and
  point its readers at what it read:

  ```python
  loss["from"] = [keep["id"] if s == gone["id"] else s for s in loss["from"]]
  ```

- **A box named after the last function of several** (hook path: an attention layer's weighted sum arrives as
  `view` or `unsafeview`): keep it, rename it (`name="加权求和"`, `type="权重 · V"`), set `kind`, and say in the
  `desc` what the functions did together.
- **Framework-internal names** (`Conv1D` in GPT-2 is a linear layer; `NewGELUActivation`): keep the `type` so the
  page matches the code, set `kind` so the colour is right (`"dense"`, `"act"`), and say what it is in `desc`.
- **Children called "0", "1", "2"** (`nn.Sequential`): prefix the parent (`fc.0`), or the box says nothing.

`kind` values: `conv pool dense act norm attn shape loss`. It is guessed from `type` when absent.

## Writing the words

Two fields, two jobs:

- `desc` is true of the step in any run: what it does and why it is there. One or two sentences. The first time a
  term appears, say what it means ("激活函数：负数变成 0，正数不变"). The viewer has descriptions for common layer
  types; yours override them.
- `note` is true of this run only: what happened to this example here, with numbers the reader can check against
  the panel beside it.

Good notes:

- quote what changed: "64 张 24×24 变成 64 张 12×12";
- quote a consequence: "这一步有 61% 的数变成了 0";
- name names: "最高的是“9”，6.31 分；其次是“4”，5.87 分";
- at the end, close the loop: "正确答案“9”的对数概率是 −0.623，去掉负号，损失就是 0.623".

Compute every number in the script (`f"{(raw['relu'] == 0).float().mean():.0%}"`), never by reading a picture.
Avoid claims that only hold for some weights ("the second layer is sparser") unless the script checks them and
picks the sentence.

A helper keyed by layer type writes most leaf notes in one place; see `step()` in `examples/siamese_spec.py`.
Groups need notes too, since a reader meets them first: say what the group turns into what ("784 个像素变成 9216
个特征"), and at the top group, what the model concluded.

Header fields:

- `title`: a short name, like "MNIST 孪生网络". No subtitle.
- `summary`: what the model is and how big, plus one measured fact ("训练一轮后测试准确率 98.6%"). Measure it.
- `example`: which sample, what the model said, and why this sample was chosen.

## Drawing awkward data

`out` and `grad` are spec TENSORs (see `spec.md`). The tracer fills them; replace them where the default drawing
says nothing:

| Data | Default drawing | Better |
|---|---|---|
| an RGB photo (`3×224×224`) | three grey maps | `{"image": "data:image/jpeg;base64,…", "shape": [3, 224, 224]}` built from the tensor the model actually received (undo the normalisation, resize to about 112px) |
| token ids | a row of numbers | `{"tokens": ["Monday", ",", "Tuesday"]}` |
| scores over a vocabulary or 1000 classes | a pooled blur | the top 8 with `labels`, and `"full": [T, 50257]` so the true shape still shows; say in the note that these are the top 8 |
| class scores for a few classes | unlabelled bars | add `"labels"` |
| logits where only relative size matters | bars that all look full | show probabilities and keep the raw scores under `also`, saying so in `desc` |
| a batch of candidates (3 captions) | a stack of 3 maps | fine as it is; say in the note that each map is one caption |
| attention weights (`heads×T×T`) | 8 maps | `tensor(raw[id], max_c=12)` to keep all heads, and drop the gradient |

Whenever you replace `out`, the note must agree with what is now drawn.

## Before you build

- [ ] The script states how the sample was chosen and measures what the summary claims.
- [ ] Level 1 has the inputs, at most a handful of boxes, and the loss.
- [ ] Every group and every leaf type has a `desc`; every node has a `note` with a number from `raw`.
- [ ] Towers, shortcuts with work on them, and fan-outs are `parallel` groups.
- [ ] No box is called `0`, `view` or `unsafeview`.
- [ ] Photos are images, text is tokens, short vectors have labels.
- [ ] Random weights, a skipped part of the model, anything else a reader would assume otherwise: said in `summary`.
