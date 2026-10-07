---
name: model-flow
description: Turn any ML/DL project into an interactive framework diagram, the kind a paper would print, that follows one real example from input to loss. A slider moves between the most general view and every layer; the data is drawn on the diagram; playback steps forward and back. Use when the user asks to visualize, diagram, explain or demo how a model or training pipeline processes data ("画框架图", "可视化模型", "数据怎么流动", "架构图", "model walkthrough").
---

# Model Flow

One method for every project: **a tree of steps, edges between them, and one example traced through**.

- The tree gives the detail levels. Closed, a module is one box; open, it is a dashed frame around its parts.
- The edges give the arrows, including branches and skip connections.
- The example gives every arrow a real piece of data to draw.

The page is `viewer.html` plus one spec JSON. The viewer is generic; the work is producing the spec.

## 1. Survey the project

Read before tracing. Find and write down, with file and line:

| What | Where it usually is |
|---|---|
| Framework | imports: `torch`, `tensorflow`/`keras`, `jax`/`flax`, `sklearn`, `xgboost` |
| Entry point | `train.py`, `main.py`, `if __name__ == "__main__"`, CLI/config (`argparse`, `hydra`, yaml) |
| Model | `nn.Module` / `keras.Model` subclasses, `forward` / `call`, or the `Pipeline([...])` for sklearn |
| Data path | `Dataset`, `DataLoader`, transforms, tokenizer, augmentation: what one sample looks like on entry |
| Loss and target | the call that produces the scalar being minimised, and what it compares |
| Weights | a checkpoint to load, or the command that trains one |

Then state the forward path from raw sample to loss in one paragraph. If you cannot, keep reading.

## 2. Design the levels

Write the hierarchy before running anything. Aim for 3–4 levels, each a complete picture on its own:

1. **Whole**: input → model → loss. Three boxes.
2. **Stages**: the 2–5 parts a paper's figure 1 would name (backbone, neck, head; encoder, decoder).
3. **Blocks**: repeated units (a residual block, a transformer layer).
4. **Layers**: every operation.

Follow the code's own module names at the leaves so the page maps back to the source. Name groups in the
reader's language. Give the level names in `levels`.

## 3. Trace one example

Pick one real sample with a known right answer; an interesting one beats the first one (a hard case, a
near miss). Never invent numbers.

- **PyTorch**: `fxtrace.py` — `leaves, raw = trace(model, x, loss_fn, target)`. It uses torch.fx, so it sees
  functional ops and real edges, and returns execution-ordered leaves with activations and gradients.
  `nest(leaves, model)` groups them by module hierarchy; usually you regroup by hand into the levels from
  step 2. `examples/mnist_spec.py` is the whole pattern.
- **fx cannot trace it** (data-dependent control flow, many HF models): restate `forward` step by step and
  record each intermediate, asserting the result equals `model(x)`. `examples/mnist_trace.py` does this.
- **Anything else** (sklearn, JAX, TF, plain code): a few lines in the project's language that run the sample
  and dump each step's output. `examples/tiny-cnn.js` shows it, with finite-difference gradients.
- **Cannot run it at all**: shapes only (omit `data`), and say so in `summary`.

## 4. Write the words

For every node a reader can land on, groups included:

- `desc` — what this step does in general. One or two sentences, no term the reader has not met.
  Common layer types have built-in defaults; override when the project uses one unusually.
- `note` — what happened to *this example* here, quoting numbers visible in the panel ("24×24 becomes 12×12",
  "the top score is 9 with 6.31"). Compute them from `raw`; do not eyeball them.

## 5. Build and look

`node build.js spec.json out.html`, then open it once: the lowest level should read as an overview, the
highest should show every layer, and stepping through should tell the story in order.

## Spec

```jsonc
{
  "title": "…", "source": "repo · file", "summary": "what the model is, 1–2 sentences",
  "example": "the one sample being traced",
  "levels": ["整体", "两个阶段", "功能块", "每一层"],   // optional names for the slider positions
  "root": { "name": "root", "children": [NODE] }       // root itself is never drawn
}
// NODE — a group if it has children, otherwise one operation. Children are in execution order.
{ "id": "conv1",            // optional; default is the path of names. Must be unique.
  "name": "conv1", "type": "Conv2d", "desc": "…", "note": "…",
  "from": ["relu", "stem"], // ids this step reads. Default: the previous leaf. A group id means its output.
                            // On a group, applies to its first leaf.
  "out": TENSOR,            // this step's output for the example
  "grad": TENSOR | number,  // d(loss)/d(out); a number is taken as the norm
  "kind": "conv",           // optional colour: conv pool dense act norm attn shape loss. Guessed from type.
  "children": [NODE] }
// A leaf with "type": "Input" is data entering from outside (the sample, the label). It has no box.
// TENSOR — one sample, no batch dimension
{ "shape": [8, 26, 26], "data": [/* flat, row-major */],
  "full": [32, 26, 26],        // true shape, when channels were dropped or the map was shrunk
  "labels": ["cat", "dog"],    // names for a short vector
  "tokens": ["The", "cat"],    // text; use instead of data
  "image": "data:image/png;base64,…" }  // any picture; use instead of data (RGB photos, spectrograms)
```

How tensors are drawn: one number → the number; a vector → bars (≤24) or a strip; 2-D and up → heat maps,
last two dimensions are H×W and everything before is channels. Blue-black is positive, red is negative.

Keep it light: `fxtrace.tensor` keeps at most 8 channels of 32×32 and vectors up to 1024. Stay under ~3 MB.

## Limits

- Layout is the browser's own text flow: steps run left to right and wrap at the page edge, and a module that
  does not fit on one row carries on in the next with its frame left open. Parallel branches are therefore drawn
  one after the other and joined by arcs, not stacked side by side.
- Arrows are not routed around obstacles. A skip connection that lands in the middle of a row two or more rows
  below its source can cross other boxes.
- One example per page, one set of weights. For a comparison, build two pages.
- Tree order is playback order, so it must be execution order.
- `fxtrace` drops `get_attr` nodes (parameters used directly in `forward`), so their edges do not appear.
