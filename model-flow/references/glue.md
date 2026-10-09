# The glue script: levels, branches, words

Contents: what the script does · designing levels (four worked cases) · branches · inputs and parameters · repeats
· steps to drop or rename · writing for a beginner (the fields, what must be explained, what helps, weak and good
texts) · drawing awkward data · before you build.

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

## Writing for a beginner

The page is read by someone starting research: some Python, basic linear algebra, no deep-learning vocabulary. They
will click a box and read the panel. What they need there is not a definition of the layer but an answer to four
questions, and the spec has one field for each:

| Field | Question | True of |
|---|---|---|
| `desc` | 做什么 | the step in any run |
| `why` | 为什么需要 | this model: what the step is for here, what would break without it |
| `origin` | 来历 | the field: a fixed formula, a standard layer, a named architecture, or the project's own design |
| `params` | 有没有可学习的参数 | the model (the tracer counts; you only set it on nodes you make by hand) |
| `terms` | 这些词是什么意思 | the words the three texts above and the note use |
| `note` | 这条样本在这里发生了什么 | this run only |

The viewer has a built-in `desc` and `why` for common layer types (convolution, activation, pooling, linear,
normalisation, dropout, embedding, attention, add, cat, the usual losses). They are general; override them wherever
you can say something about *this* model ("卷积在这里只沿着相邻的几帧和几个频带看，所以一个 token 只知道前后约 0.3
秒"). Groups have no built-in text at all: every group needs `desc`, `why` and `origin` from you.

### What must be explained, always

A page is not finished while any of these is missing:

1. **The task.** What is being predicted, from what, and what counts as right. In `background`. A reader who does
   not know what "fake speech detection" or "next-token prediction" is cannot make sense of anything after it.
2. **The input, in human units.** "1.53 秒的录音，每秒 16000 个数" before "24480". Likewise the output: "一个数，大于
   0 判伪造".
3. **For every group: is it a model?** `origin` says so in its first words: "不是模型…" / "一个小的卷积网络…" /
   "ResNet-18…" / "这个项目自己设计的…". This is the question beginners ask first and figures answer least.
4. **For every group and non-trivial step: why it exists.** State the problem it solves. If you cannot say what
   would go wrong without it, read the code and the project's notes until you can; do not write "提取特征".
5. **Learned or fixed.** Comes from `params`. Check that hand-made nodes have it and that the totals on the level-1
   boxes look right (a front end of fixed transforms must say "没有").
6. **Every term on first use.** Shape notation ("32×27×154 是 32 张 27×154 的图"), and the field's words: 通道, 帧,
   token, 特征图, 感受野, logit, 概率, 残差, 归一化, 注意力, 查询, 掩码, 嵌入, 池化, 步长, 批, mel, 分贝… Put the
   explanation in `terms` on the node where the word first appears.
7. **What the numbers mean.** Not only "fake_logit = −1.09" but "小于 0，所以判真实；换成概率是 25%，说明它并不太
   确定". A sign convention (正 = 像伪造) is stated wherever a signed number is shown.
8. **How to read each picture.** What the rows and columns of a map are, which end is low frequency, what dark and
   light mean, when a thumbnail is nearly blank and why.
9. **The loss and the label.** What is compared with what, why a smaller number is better, and that the label is
   used only for training, never by the model itself.
10. **What is not in the figure.** Parts of the system left out, parameters that exist but are not run, a single seed
    of an ensemble. In `summary` and `background`.

### What helps, when there is something true to say

- **An everyday comparison**, one per idea at most: "像把一张照片切成上、中、下三条，各请一个人看". Only when it
  does not mislead.
- **The alternative and why it was not taken**: "也可以让一个网络看整张谱图，但那样就说不清判决靠的是哪一部分".
  Take it from the project's notes or comments, not from your own guess; if you guess, say "可能是因为".
- **What it would be called in a paper**, with the English term, so the reader can search: "交叉注意力
  cross-attention".
- **Where the same idea appears elsewhere**: "和 DETR 里的查询向量是同一种做法".
- **What to look at**: "点开这个框，看第 5 行是不是全空".
- **What a design choice costs**: "时间减半，省了计算，但小于 20 毫秒的细节从这里开始就分不开了".
- **How a group's parts relate**, on the group's `desc`: the order of its children and what each hands to the next.
- **Common confusions, named**: "叫 queries，但不是文字问题，是一组学出来的数".

### Concepts that run through the figure

Some nouns are used by many boxes and belong to none: this model's "五个问题" are created in one box, restricted in
another, read from memory in a third, scored in a fourth and summed in a fifth. Explaining them in the box where
they first appear does not work, because a reader does not go through the boxes in order; they click the one that
looks interesting and meet "每个问题只读自己的区域" cold. So such a noun gets a **concept card** (the cards sit below the figure; a row of chips under the title leads to them),
and every text that uses the word links to it.

How to find them: when the texts are written, go through them once and list every noun that (a) is this model's
own, or is used here in a particular sense, and (b) occurs in three or more boxes. Typical ones:

- the model's actors: 问题 / 查询, 塔, 分支, 头, 主干, 专家;
- its units of data: token, 帧, patch, 记忆, 特征图, 类别 token;
- its partitions: 频带, 区域, 阶段, 层 (when "第 3 层" alone would not tell a reader what a layer is here);
- its quantities: 分数, 贡献, logit, 相似度, 温度.

What a card must say, in this order: **what it is** (and what it is not: "不是文字，是一组学出来的数"), **how many
there are and what distinguishes them** (a table, one row per member, whenever there are members), **what it does
in the model** (what it takes, what it gives), and **where to see it in the figure** (which boxes). If a column of
the table is the designers' intention rather than something measured, say so under the table (`after`).

```python
spec["concepts"] = [{
    "name": "五个问题", "aliases": ["分面问题", "问题向量", "问题"],     # every spelling the texts use
    "short": "五组学出来的数，各自只看一块证据，回答“像不像合成的”。",      # the hover line; default: the first sentence
    "text": ["它们问的是同一句话：**只看这一部分，像不像合成的？** 区别只在各自被允许看哪里。…",
             "每个问题最后给出一个分数…五个分数加权相加就是判决。"],
    "table": {"head": ["问题", "能看的区域", "想抓的线索"], "rows": [["高频 hf", "≥4 kHz 的有声帧", "擦音的细节"], …]},
    "after": "“想抓的线索”是代码注释里写的设计意图，没有验证过模型实际依据什么。",
}]
```

Three to six cards. A term that only needs one line stays in `terms`; a concept needs a paragraph or a table. A
word that is both (a term on some node and a concept) links to the card.

### Easy to read

A correct explanation that nobody reads has explained nothing. The answer is not fewer facts and not folding text
away (a folded text is an unread text): it is wording, layout, emphasis and pictures. All four, on every page.

**1. Wording.** Write the way you would say it to someone across a table.

- One idea per sentence. About 25 characters; `build.py` reports any sentence over 55.
- Start with the thing itself: "把波形变成一张图。" not "这一部分的作用是将输入的波形信号转换为…".
- Cut the scaffolding: 的作用是, 也就是说, 需要注意的是, 通过…来实现, 进行…操作.
- A verb, not a noun phrase: "减去平均值" not "进行去均值处理".
- A number, not an adjective: "少了三分之二" not "大幅减少".
- Say a thing once. If `why` repeats `desc`, one of them is wrong.

**2. Layout.** Prose is for one line of thought. Everything else has a shape, and the page can draw it:

| The content is | Write it as | Shows as |
|---|---|---|
| a statement plus its parts | `["一句话。", "- 标签：内容", "- 标签：内容"]` | a sentence and a list with bold labels |
| steps in order | points, in order | a list |
| several members with the same attributes | a concept's `table` | a table |
| the page's key numbers | `"stats": [["标签", "值", "小字"], …]` | large tiles under the title |
| what kind of thing a box is | `"origin_kind": "fixed" / "standard" / "named" / "own"` | a badge: 固定计算 / 标准做法 / 知名结构 / 项目自创 |
| the five background topics | each a list: `["任务：**一句话**", "- …", "- …"]` | five cards side by side |

A paragraph of more than three sentences is reported by `build.py`: break it, or turn its parallel parts into
points. A `desc` is typically one sentence and two to four points; a `why` is two or three short sentences (it is
an argument, so it stays prose); an `origin` is a verdict and one to three points.

**3. Emphasis.** See the next section.

**4. Pictures.** Words next to the picture they describe are read; words alone are skimmed.

- *Drawings of layers.* The page draws what convolution, pooling, an activation, a linear layer, normalisation,
  attention, add, cat, softmax and dropout do, beside their "做什么". Give `"mini": "pool"` (or `conv`, `relu`,
  `gelu`, `sigmoid`, `linear`, `norm`, `attention`, `add`, `cat`, `softmax`, `dropout`) to a step or group of the
  project's own that works the same way.
- *Figures from this example.* `"figure": {"image": "data:image/png;base64,…", "caption": "…"}` on a node or a
  concept. Draw them in the glue script with matplotlib from `raw`, with what a thumbnail cannot carry: axes with
  units, labelled rows, boundaries, a legend. Ask of every group and every concept: *is there something here one
  could point at?* Usual answers:
  - the input in human terms (a spectrogram with seconds and Hz; an image with the crop drawn on it);
  - a partition (where the bands are cut; which tokens each query may read, as coloured overlays on the input);
  - attention, with row and column labels and the groups of columns marked off;
  - how the final numbers combine (signed bars with their labels and the total);
  - what changes across repeated blocks (a small multiple, one panel per block).
- A figure shows this run, so it obeys "never invent numbers": computed from `raw`, in the script. Use one figure in
  both places it explains (the node and the concept card). White background, a font that has the page's language
  (`plt.rcParams["font.sans-serif"]`), about 5 inches wide at 90 dpi so that four or five figures stay under 1 MB.
- **Save it with `picture(fig, caption)`** (`from figure import picture`; `kind="jpeg"` when the figure contains a
  photo). It returns the `{"image", "caption"}` object, and before saving it reads the figure back and corrects
  what is on top of something else. Where a label lands depends on this example's numbers, so a place that was
  free for one sample is taken for the next; do not try to get it right by hand, draw and let it correct:
  - x tick labels that run into each other are turned, 40° or upright;
  - a legend lying on text or on bars, lines or points goes to the place that covers no text and the least data
    (so leave `loc=` out unless you have a reason);
  - a label or annotation cut off by the edge of the figure comes back onto it; one on other text slides to the
    nearest free spot; one that a plotted line runs through, or that touches the frame, does so when a clear spot
    is near.

  It prints one line per figure it changed (`figure …: moved the legend to a free place; moved “这一对”`). A line
  with `TEXT STILL OVERLAPS` means there was no room: the figure has too much text for its size, and that is
  yours to fix. Labels set inside bars on purpose stay where they are; text on a slant is judged by its upright
  box and may be reported when it is fine. Look at the figures in the built page all the same: the pass sees
  text, lines and frames, not whether a label still reads as belonging to its point.
- No decoration: an icon or a picture that carries no information is noise.

### Emphasis

The page styles the structure itself: a coloured icon and label per field, a chip for learned / fixed, the note as
a red-edged callout, glossary words underlined with their meaning on hover. Inside a text you have three marks:

| Write | Shows as | Use for |
|---|---|---|
| `**不是模型**` | bold with a marker stroke | the one thing to remember from this text: the verdict of an `origin`, the reason in a `why`, the result or the surprising number in a `note` |
| `` `fake_logit` `` | code | names from the source: variables, classes, functions |
| `任务：…` at the start of a `background` paragraph | a coloured label | the five background paragraphs |

One marked phrase per text, two when a note has a result and a number. Mark the conclusion, not the topic: in
"波形里看不出音高，**换成这张图之后音高变成了条纹**", not "**波形**里看不出音高…". Keep a marked phrase short
(under about fifteen characters); a marked sentence is a paragraph again. A glossary word is underlined wherever it
first occurs in a text, so name your `terms` keys the way the texts spell the word ("感受野", "分贝 dB": either part
matches); a key of one character is never matched.

Do not pad. A `why` that repeats the `desc` in other words, or an analogy for something already plain, makes the
panel longer and the page worse.

### Weak and good

A group that is a fixed transform:

- weak: `desc`: "声学前端：提取对数 mel 谱。" (three unexplained terms; no reason; a reader cannot tell it is not a
  network)
- good: `desc`: "把波形变成一张“时间 × 频率”的图：横向是时间，每 10 毫秒一列；纵向是 80 个频带；每一格是那个时刻、
  那个频带的声音有多强。" `why`: "波形只是一长串气压读数，音高、音色这些东西在里面是看不出来的。换成这张图之后，
  它们变成了图上的条纹和纹理，后面的卷积网络才有图案可找。" `origin`: "不是模型，没有任何要训练的数，是三步固定的
  信号处理。合起来的标准叫法是对数 mel 谱（log-mel spectrogram），语音任务里最常用的输入。" `terms`: {"帧": "…",
  "mel 频带": "…", "分贝 dB": "…"}

A step of the project's own:

- weak: "交叉注意力，带掩码。"
- good: `desc` says who looks at whom and what the mask removes; `why` says what the mask buys the project (each
  question's answer can be attributed to one region); `origin` says "交叉注意力是标准做法（cross-attention）；去掉
  问题之间的自注意力、按区域加掩码是这个项目自己的设计"; `terms` explains 查询, 掩码, 空 token.

A plain layer, when the built-in text is enough: leave `desc` and `why` alone and spend the words on the `note`.

### Notes

`note` is true of this run only: what happened to this example here, with numbers the reader can check against the
panel beside it, and what those numbers mean.

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
- `background`: a list of short paragraphs shown open under the title as "读图之前：这个模型在做什么". In order:
  the task and why it is hard; what goes in and what comes out; the model's idea in plain words; how it is
  trained; what the figure leaves out. Three to five paragraphs, two or three sentences each. No term that is not
  in `terms` / `glossary` or explained on the spot.
- `glossary`: `{"词": "解释"}` for words of the task and the project that belong to no single box. The page merges
  it with every node's `terms` into one list.

### Where the facts come from

`desc`, `why` and `origin` are claims about the project. Take them from the code, its comments and docstrings, the
README and the project's notes, in that order. A design reason you inferred rather than read is written as an
inference ("看起来是为了…"). A claim about what a part has *learned* to detect is a claim about weights: either the
script measures it, or the text says it is the design intent, not a finding.

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
- [ ] Every group, and every step that is not a plain layer, has `why` and `origin`; `origin` starts by saying
      whether it is a model.
- [ ] Hand-made nodes have `params`; the level-1 totals look right.
- [ ] `background` covers task, input and output, idea, training, what is left out.
- [ ] Every term a first-year student would have to look up is in some node's `terms` or in `glossary`.
- [ ] Every noun of the model that three or more boxes use has a card in `concepts`, with a table when it has
      members, and its `aliases` cover the spellings the texts use (click one in the built page to check).
- [ ] `build.py` reports no heavy wording; parallel content is in points, tables or tiles, not in sentences.
- [ ] Every group and every concept that describes something visible has a `figure` drawn from `raw`; the project's
      own pooling-, attention- or add-like steps have a `mini`.
- [ ] Every figure is saved with `picture()`, and the script prints no `TEXT STILL OVERLAPS`.
- [ ] Every group has an `origin_kind`; the page has `stats`.
- [ ] Each `origin`, each group's `why` and each note that states a result has one `**marked**` phrase; code names
      are in backticks; background paragraphs start with a label.
- [ ] Towers, shortcuts with work on them, and fan-outs are `parallel` groups.
- [ ] No box is called `0`, `view` or `unsafeview`.
- [ ] Photos are images, text is tokens, short vectors have labels.
- [ ] Random weights, a skipped part of the model, anything else a reader would assume otherwise: said in `summary`.
