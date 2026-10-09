---
name: model-flow
description: Turn an ML/DL project into an interactive, paper-style framework diagram - one self-contained HTML page that follows one real example from input to loss, with a simple-to-detailed slider, the actual data drawn at every step, step-by-step playback, light and dark looks, and SVG export for papers and slides. Use this whenever the user wants to see, draw, explain, present or demo how a model, network or training pipeline is built or how data moves through it, in any codebase (PyTorch, Hugging Face, or other) - 画框架图, 模型结构图, 架构图, 可视化模型, 数据怎么流动, 讲讲这个模型怎么工作, 给论文画模型图, model architecture diagram, visualize this network, walk me through this model, figure for my paper. Use it even when they only say 画个图 or "explain this model" about a repository that contains a model, and prefer it to a hand-drawn SVG, a Mermaid chart or prose whenever the subject is a neural network.
---

# Model Flow

You produce **one HTML page** for the user's project: a framework figure of the kind a paper prints, except that it
is alive. A slider moves between the most general view (input → model → loss) and every single layer. Each box
shows the real data it turned its input into, for one real example. Playback steps through the computation; a panel
spells out what went in, what the step does, what came out, and the gradient. It has a light and a dark look and
exports the figure as SVG.

One method covers every project: **a tree of steps, edges between them, and one example traced through.**

- The tree gives the detail levels. Closed, a module is one box; open, it is a labelled frame around its parts.
- The edges give the arrows, including branches and skip connections.
- The example gives every arrow a real piece of data to draw.

**The reader is a beginner who is starting research.** They can read some Python and know what a matrix is. They
have not met a mel spectrogram, attention, a residual connection or a logit, and they cannot tell from a box's name
whether it is a trained network, a fixed formula, a famous architecture or something this project made up. The
page is how they get into the project, so it has to answer, for every box they can land on: what does it do, why
is it there, is it learned or fixed, and is it a known thing with a name. A figure that is correct but leaves those
unanswered has failed at its job.

The viewer (`assets/viewer.html`) is generic and finished. Your work is producing the **spec** (a JSON file) for this
project, by running the project's own code on one sample. Three things are not negotiable, because they are what
makes the page worth more than a sketch:

- **Never invent numbers.** Every tensor on the page comes from running the real model; every number quoted in a
  note is computed in the script that wrote it. If the model cannot be run, show shapes only and say so on the page.
- **Explain every box for a beginner.** Every group and every step says what it does, why it is needed, where it
  comes from, and whether it has learned parameters; every term it uses is explained on the page. Step 5 says how.
- **Read your own page back before you hand it over.** A diagram with an arrow through a box, or an overview level
  with thirty boxes, is not done. Step 6 has a script for this.

## What is in this skill

| Path | What it is | When |
|---|---|---|
| `scripts/fxtrace.py` | Traces one example through a PyTorch model: steps, edges, activations, gradients, source lines. `python fxtrace.py` self-checks. | Step 4 |
| `scripts/figure.py` | `picture(fig, caption)` turns a matplotlib figure into a spec picture. First it reads the figure back and corrects text that lies on other text, on a plotted line or on the frame, a legend that hides data, and crowded tick labels; it prints what it moved and what it could not fix. `python figure.py` self-checks. | Step 5 |
| `scripts/deps.py` | Lists what a project imports and what an interpreter is missing; prints the install command. Installs nothing. | Step 2 |
| `scripts/build.py` | `python build.py spec.json page.html` puts a spec into the viewer; checks ids and edges first. | Step 6 |
| `scripts/check.py` | Opens a built page in a headless browser at desktop and phone width, reports arrow problems and script errors, saves screenshots. | Step 6 |
| `assets/viewer.html` | The page. Do not edit it per project. | — |
| `assets/spec_template.py` | A glue script to copy and fill in. | Step 5 |
| `references/environment.md` | Finding the project's interpreter, installing what is missing, weights and data, what to tell the user. | Step 2 |
| `references/tracing.md` | `trace()` in detail: fx and hook paths, inputs, losses, sizes, several checkpoints, non-PyTorch projects, errors. | Step 4 |
| `references/glue.md` | Grouping steps into levels, branches, repeats, writing for a beginner (what must be explained, with examples), drawing awkward data. | Step 5 |
| `references/spec.md` | The spec format, field by field, and how tensors are drawn. | Steps 5–6 |
| `references/presenting.md` | Opening the page, walking someone through it, exporting, publishing, the final report. | Step 7 |
| `examples/*.py` | Four complete glue scripts: a plain CNN, a two-tower ResNet, a GPT, CLIP. To read, not to run: each belongs to the project it was written for. Read the one nearest yours: for the regrouping, and for how the beginner's texts, the concept cards, the tiles and the figures drawn from `raw` are written in practice. They pick the parts of a model by name rather than by position where library versions differ; do the same. | Step 5 |

`<skill>` below means this skill's folder. Python 3.9+ with `torch` is needed to trace, and `matplotlib` for the
figures of Step 5 (install it like any other missing package, see `references/environment.md`); `check.py` needs
any Chromium-based browser (Chrome, Edge, Chromium, Brave) but nothing else.

## Before you start

Tell the user in two or three lines what you are about to do: which model you will diagram, that you will run it on
one real sample, where the files will go, and that you may need to install packages or download weights (you will
say what and how big before doing it). Then work without stopping for every small choice; ask only the questions
`references/environment.md` says to ask.

Work in a folder of your own inside the project, `model-flow/` by default:

```
<project>/model-flow/
  fxtrace.py        copied from <skill>/scripts, so the glue script runs on its own later
  figure.py         copied likewise: saves the figures you draw, after untangling their text
  spec.py           the glue script you write (Step 5)
  spec.json         what it writes
  <name>.html       the page
  shots/            screenshots from check.py
```

Do not commit it and do not edit the project's `.gitignore` or any of its source files: the project stays as you
found it. If the user wants the folder elsewhere, put it there.

## Step 1 — Survey the project

Read before running anything. Find, with file and line:

| What | Where it usually is |
|---|---|
| Framework | imports: `torch`, `transformers`, `tensorflow` / `keras`, `jax` / `flax`, `sklearn` |
| Entry points | `train.py`, `main.py`, `if __name__ == "__main__"`, configs (`argparse`, `hydra`, yaml) |
| The model | `nn.Module` subclasses and their `forward`; which one is the top; how it is built (arguments, config) |
| The data path | `Dataset`, transforms, tokenizer, collate function: what one sample looks like when it reaches `forward` |
| Loss and target | the call that produces the scalar being minimised, and what it is compared with |
| Weights | a checkpoint in the repo, a documented pretrained download, or the command that trains one |

Then write down, for yourself, the forward path from a raw sample to the loss in one paragraph. If you cannot, keep
reading. If the project has several models, pick the one the user means, or the one its README leads with, and say
which.

## Step 2 — Make it run

The page needs one forward and one backward pass, on one sample. That is small: a CPU is enough, and so is the
project's existing environment. Read `references/environment.md` now; in short:

1. Find the interpreter the project already uses (a `.venv`, a conda env named in `environment.yml`, what the README
   says). Only if there is none, make a virtual environment inside `model-flow/`.
2. `"<that python>" <skill>/scripts/deps.py <project> <model file> <training script>` shows what it lacks.
3. Tell the user what you will install, why, into which environment and roughly how big; then install it. Ask first
   only in the cases the reference lists (pinned environments, system-wide installs, GPU builds, anything over about
   1 GB, anything behind a login).
4. Get weights, in this order of preference: a checkpoint already there; the project's documented pretrained
   weights; a short training run with the project's own training code when that takes minutes; random
   initialisation, in which case the page must say that its numbers mean nothing.
5. Get one sample the same way: the project's dataset if it is present, otherwise ask or fetch the standard one.

Keep a list of everything you installed or downloaded. It goes into your final message.

## Step 3 — Design the levels

Write the hierarchy down before tracing. Aim for three to six levels, each a complete picture by itself:

1. **Whole**: inputs → model → loss. At most six or seven boxes.
2. **Stages**: the two to five parts a paper's figure 1 would name (backbone and head; encoder and decoder; the two
   towers).
3. **Blocks**: the repeated units (a residual block, a transformer layer).
4. **Layers**: every operation.

Three decisions shape the figure more than anything else:

- **Branches.** Where the code runs two or more paths that meet again (two towers, a projection shortcut beside the
  main path, the Q, K and V projections of attention), each path is a group or a step of its own, and they are
  wrapped in a group marked `"parallel": true`. They are drawn as lanes, one under the other. An identity skip
  needs nothing: it is only an edge and is drawn as an arc.
- **Repeats.** Leave repeated blocks as siblings. Of three or more built the same way, the slider opens only the
  first and marks it ×N; the reader can open any other.
- **Names.** Leaves keep the code's own names, so the page maps back to the source. Groups get names in the
  reader's language.

`references/glue.md` has the patterns and the reasoning.

## Step 4 — Trace one example

Pick **one real sample with a known right answer**, and prefer an interesting one to the first one: a case the model
gets right but barely, a near miss, a sentence with an obvious continuation. The four examples all choose theirs by
a rule written in the script ("of the first 500 test images, the one it gets right with the least confidence").

For PyTorch, copy `<skill>/scripts/fxtrace.py` into your folder and call:

```python
from fxtrace import trace, nest
leaves, raw = trace(model, x, loss_fn, target)   # x: a tensor, a tuple of inputs, or a dict of keyword inputs
tree = nest(leaves, model)                       # the leaves grouped by module call
```

`leaves` is one spec node per executed step, in order, with `from` (real edges), `out`, `grad`, `src` (file:line),
and `also` for a step's further results. `raw` maps each step's id to its full tensor, for the numbers you will
quote. `trace` tries torch.fx first and falls back to forward hooks by itself when fx cannot follow the model (it
prints which); both give the same kind of leaves.

Things that matter in practice (details and more cases in `references/tracing.md`):

- Batch of one. A second input that is naturally several (three candidate captions) may stay a batch; it is drawn as
  a stack.
- The loss is whatever compares the output with the right answer for this sample: pass a small function.
- Hugging Face models: pass keyword inputs, `hooks=True`, `use_cache=False`, and `output_attentions=True` with
  `attn_implementation="eager"` at load time, so attention weights become a step.
- Hundreds of steps: add `max_hw=24` so the page stays under about 3 MB.
- Not PyTorch: a few lines in the project's own language that run the sample and dump each step's output into the
  same spec. `examples/tiny-cnn.js` is the pattern.

## Step 5 — Write the glue script

Copy `<skill>/assets/spec_template.py` to `model-flow/spec.py` and fill it in. It does four things: load the model
and the sample, trace, **regroup the leaves into the levels of Step 3**, and **write the words**. Read the example
nearest your project first (`examples/mnist_spec.py` plain CNN, `siamese_spec.py` branches and towers,
`gpt2_spec.py` transformer on the hook path, `clip_spec.py` two different towers), then `references/glue.md`.

Every node a reader can land on, groups included, gets these. The panel shows them under the box's name, in this
order:

| Field | The question it answers | Rule |
|---|---|---|
| `desc` | 做什么: what goes in, what is done to it, what comes out | One to three sentences in everyday words. Common layer types have a built-in one; write your own for groups, for the project's own types, and wherever the built-in one does not say what the step is doing *here* |
| `why` | 为什么需要: what this step is for in this model, and what would go wrong without it | Required on every group and on every step that is not a plain conv / norm / activation (those have a built-in one). Tie it to the task: not "extracts features" but "a waveform is 24,000 numbers that say nothing by themselves about pitch; this turns it into a picture where pitch and timbre are visible" |
| `origin` | 来历: is this a model? does it have a name? | Required on every group and on every step whose type is not a standard layer. One of: a fixed signal-processing or maths step ("不是模型，是固定的信号处理；标准叫法是对数 mel 谱 log-mel spectrogram"); a standard layer or pattern, with the name to search for in Chinese and English; a named architecture, with the name and where it comes from ("ResNet-18，He 等 2015"); the project's own design, saying what known thing it resembles and how it differs |
| `params` | 有没有可学习的参数 | `trace` fills it on every step and the page adds it up for groups. Set `"params": 0` on a step you add by hand; when you fold a parameter into a step, add its size to that step (`fold()` in the template does) |
| `terms` | 这些词是什么意思 | `{"词": "一句话解释"}` for every term the texts of this node use that a beginner has not met: mel, 帧, token, logit, 通道, 残差, 感受野… Define a term on the first node that uses it. The page also collects all of them into one list under the title |
| `note` | 这条样本在这里发生了什么 | Numbers the panel shows, computed from `raw`: "24×24 becomes 12×12", "the top score is 9 with 6.31", "61% of the values became 0". Say what the number means, not only what it is |

And the page gets a header:

- `title` (a short name), `source` (repo and file), `levels` (a name for each slider position);
- `summary`: what the model is, its real parameter count, a real accuracy if you measured one;
- `example`: the sample, what the model said, why this sample;
- `background`: a list of three to five short paragraphs under "读图之前", for someone who has never seen the project:
  **the task** (what question is being answered, about what data, and why it is hard); **what goes in and what comes
  out**, in units a person understands (seconds of audio, a 28×28 picture; one number above or below zero); **the
  idea of this model in plain words**, the way you would say it across a table; **how it is trained** (what the
  loss compares) and **what the figure leaves out** of the larger system;
- `glossary`: terms that belong to the whole page rather than to one box (the task's own words: 伪造语音, dev 集);
- `concepts`: **the things several boxes talk about.** Every model has a few nouns of its own that the texts keep
  using: "五个问题", "token", "记忆", "两座塔", "主干", "类别 token". A hover line is not enough for these: a reader
  who meets "每个问题只读自己的区域" in the sixth box must be able to find out what the questions are, how many,
  what each one is for and what comes out of them. Each gets a card below the figure, reached from a row of chips under the title (name, a few sentences on what
  it is and what it does, a table when it has members: one row per question, per band, per tower), and every text
  that names it links to the card. Before building, list the nouns that appear in three or more boxes; each one is
  a concept or you have not finished.

Write all of it in the user's language. The page's own labels are in Chinese.

**Make it something a person wants to read.** A beginner who sees a block of even prose does not start reading it.
The cure is not to say less and not to hide text behind a fold: everything stays on the page, and four things make
it readable. `references/glue.md` ("Easy to read") has the rules and examples for each; `build.py` lists the
sentences and paragraphs that break the first two.

1. **Plain, short wording.** One idea per sentence, about 25 characters, never over 55. Say the thing, then stop.
2. **Layout instead of paragraphs.** Any text may be a list of strings; a string starting with `"- "` is a point.
   Parallel things (the parts of a step, the rows of a picture, the steps of a recipe) are points with a label
   ("- 横向：时间，每 10 毫秒一列"), never a sentence with three commas. Numbers that matter go into `stats` tiles,
   members of a concept into its `table`, the kind of thing a box is into `origin_kind` (a badge).
3. **Emphasis.** One marked phrase per text (below).
4. **Pictures with the words.** Common layers get a small drawing of what they do automatically; give `mini` to
   your own steps that work like one. Wherever a sentence describes something that can be seen (which region a
   question reads, where the bands are cut, how five numbers add up), draw it from this example's own tensors with
   matplotlib and attach it as `figure`: on the node, and on the concept card that explains the idea. Save every
   figure with `picture(fig, caption)` from `figure.py`, never with `savefig` directly: it corrects overlapping
   text before saving, the same "draw first, read back, correct" the page does for its arrows.

The marks, which work in `summary`, `example`, `background`, `desc`, `why`, `origin` and `note`:

- `**…**` marks the one phrase a reader should take away from that text: the verdict in `origin` ("**不是模型**"), the
  reason in `why`, the result in a `note` ("判 **real**，把握只有 **74.8%**"). One per text, two at most; when
  everything is marked nothing is.
- `` `…` `` sets a name from the code (`fake_logit`, `nn.MultiheadAttention`) apart from the prose.
- A `background` paragraph opens with a short label and a colon ("任务：…", "进去和出来：…"); the page turns the
  label into a coloured heading.

**The test before you build:** read the level-1 and level-2 boxes in order, as someone who took one programming
course. At every sentence ask "which word here would they have to look up?" and "would they know why this box
exists?". Each such word goes into `terms`; each missing reason into `why`. Then ask of every group: "could they
tell whether this is a trained network?" If not, `origin` is missing. And of every noun that keeps coming back:
"if they clicked only this one box, would they know what that is?" If not, it belongs in `concepts`. `references/glue.md` has the list of what
must always be explained and worked examples of weak and good texts.

## Step 6 — Build, read back, correct

```
python <skill>/scripts/build.py model-flow/spec.json model-flow/<name>.html
python <skill>/scripts/check.py model-flow/<name>.html --shots model-flow/shots
```

`build.py` refuses a spec with duplicate ids or a `from` that names nothing, and says which. `check.py` opens the
page at desktop and phone width and prints, per level, what the page found wrong with its own arrows. The viewer
has already corrected what it can (it routes each arrow, then slides pieces clear of boxes and moves unrelated
arrows onto separate tracks); what is listed is what it could not fix, and **you fix it in the spec**:

| What the check says | What it means | What to change |
|---|---|---|
| the page did not draw: … | the spec is wrong in a way the viewer could not survive | read the message; usually an id or a `from` |
| 穿过方框 / 紧贴着方框的边 (through a box / hugging one) | something stands in the main line between two connected steps | an input card: move it into the lane or group that reads it, or fold it into the step as `also` when it is one number. A step: check that tree order follows the edges |
| 有一段重叠 (two arrows overlap) | several arrows fan out of one step, or cross a long way | the consumers are branches: wrap them in a `parallel` group. Or a step in between only renames or reshapes: drop it and point its readers at what it read |
| 箭头没有落在目标方框上 (misses its target) | a layout case the viewer does not handle | regroup so that the two steps are in the same row or lane; report it if it stays |

Then **look at the screenshots** in `shots/` (read them as images). The script cannot judge these:

- Level 1 reads as an overview: a handful of boxes, the input and the loss both visible.
- Each level adds one idea. If a level adds forty boxes at once, an intermediate group is missing.
- Names fit their boxes; thumbnails show something (not a blank or a solid block); the deepest level has every layer.
- Branches sit side by side as lanes and join where the code joins them.

Rebuild and re-check after every change. Stop when the check prints `clean` at both widths and the screenshots tell
the story in order; if one problem will not go away after two or three attempts, leave it and tell the user.

No headless browser, or the shell cannot start one? Open the page with whatever browser tool you have (serve the
folder with `python -m http.server` if `file://` is refused) and evaluate `flowCheckAll()` in it: it returns the same
list, `{}` when clean.

## Step 7 — Present

Read `references/presenting.md`. In short: open the page for the user (or give them the path), and give them a
short guided tour in your message rather than a list of features: what to look at first, which box is the
interesting one for this example, what the surprising number is. Then report plainly:

- what the page shows (model, the example, what the model said about it), in words the same beginner follows;
- where the files are, and the one command that regenerates the page;
- everything you installed or downloaded, where it went and how big it is;
- what the page does **not** show or could not get right (random weights, a part of the model you left out, an arrow
  problem that stayed).

## When something goes wrong

| Symptom | Likely cause | Do |
|---|---|---|
| `trace` prints that fx cannot trace the model | normal for models with data-dependent control flow and most Hugging Face models | nothing: the hook path took over |
| `AssertionError: fx graph disagrees with model.forward` | the model is not deterministic in eval mode | call `trace(..., hooks=True)`; find what is random (dropout left on, sampling) |
| a step you expected is missing on the hook path | functions computed it and no module read or returned it | ask the model to return it (`output_attentions=True`), or compute it in the glue and attach it as `also` |
| one step reads from ten earlier steps | a hand-written residual stream on an old `fxtrace.py` | use the bundled `fxtrace.py`; residual sums are steps of their own |
| the page is blank | the spec could not be drawn | `check.py` prints the page's own error |
| the page is many MB | hundreds of steps at full size | `trace(..., max_hw=24)`, fewer channels for attention maps (`tensor(t, max_c=4)`) |
| thumbnails are solid or empty | the tensor is all one value, or a huge range hides the rest | say so in the note; it may be the truth (a dead layer, a mask) |
| the glue script prints `figure …: TEXT STILL OVERLAPS` | `picture()` found text on text in a figure and had no free place to move it to | fix the figure in the script: shorter labels, fewer of them, a larger figure, or a second panel; rerun until the line is gone |
| a label in a figure ended up away from the thing it names | `picture()` moved it off other text or off a line | place it yourself where there is room (`xytext=`), or leave the legend without `loc=`; what it moved is printed as `figure …: moved …` |
| numbers look wrong for a layer followed by an in-place op | an old tracer | the bundled one copies values when it records them |

## Limits, so you can say them

- Branches are drawn where the spec says `parallel`; nothing finds them from the edges.
- Arrows are routed one at a time and corrected in one pass. Where boxes leave less than 16px, an arrow can only be
  centred in the gap.
- One example per page. `variants` (see `references/spec.md`) puts several checkpoints or samples of the *same*
  steps behind one more slider; models with different steps need separate pages.
- On the hook path, several functions in a row share one box, named after the last of them: rename it and say in
  its note what they did together.
- Playback follows tree order, lane after lane, although the lanes are conceptually parallel.
