# Tracing one example

Contents: the call · choosing the sample and the loss · the two tracers · what a leaf holds · Hugging Face and
other keyword-input models · page weight · several checkpoints on one page · projects that are not PyTorch ·
models that cannot be run · errors.

## The call

```python
import sys; sys.path.insert(0, "model-flow")      # where you copied fxtrace.py
from fxtrace import trace, nest, tensor, variant

leaves, raw = trace(model, x, loss_fn=None, target=None, hooks=None, **size)
tree = nest(leaves, model)
```

- `model`: the top `nn.Module`. `trace` puts it in eval mode and switches in-place modules (`ReLU(inplace=True)`) to
  out-of-place; the numbers are the same, and a step is no longer overwritten by the one after it.
- `x`: one tensor, a tuple of positional inputs, or a dict of keyword inputs. Float inputs get a gradient; integer
  inputs (token ids) do not. Non-tensor values in a dict (`use_cache=False`) are passed through.
- `loss_fn(output, target)`: returns the scalar. `output` is whatever the model returns, so index into it as
  needed: `lambda out, t: F.cross_entropy(out.logits[:, -1], t)`.
- `target`: a tensor. One number is shown as that number; change its `out` to `{"tokens": ["cat"]}` in the glue to
  show a word instead.
- `hooks`: `None` tries torch.fx and falls back to forward hooks; `True` goes straight to hooks; `False` is fx or an
  exception.
- `**size`: `max_c` (channels kept, default 8), `max_hw` (largest side of a map, default 32), `max_len` (longest
  vector, default 1024).

Returns `leaves`, a list of spec nodes in execution order, and `raw`, a dict from each leaf's id to the tensor it
shows (with the batch dimension; a step's further results are under `id.1`, `id.2`, and `raw["loss"]`,
`raw["target"]` are there too). Use `raw` for every number you quote.

`python fxtrace.py` runs a self-check on small models covering both tracers.

## Choosing the sample and the loss

One sample, batch dimension of 1, with a right answer you know. Choose it by a rule the script states, so the page
can say why this one:

- classification: among the first few hundred test items, the one the model gets right with the least confidence;
- language models: a prompt with one obvious continuation that the model finds, with a believable runner-up
  ("Monday, Tuesday, Wednesday," → " Thursday" at 49%, " Friday" at 17%);
- matching / retrieval: one query against a few candidates where the runner-up is plausible (a photo of cats on a
  couch against "a cat", "a dog", "a couch").

Try a handful and print the model's answers before settling; a model that is 99.9% sure or plainly wrong on the
first thing you try makes a dull page.

The loss is the project's training loss applied to this one sample when that makes sense, otherwise the plain
comparison of the output with the right answer (cross-entropy over the candidates). It exists so that every step
has a gradient to show.

## The two tracers

**torch.fx** (tried first) records the graph of `forward`. Steps are the graph's nodes: module calls *and* functions
(`F.relu`, `torch.flatten`, `+`), each with its own activation. It fails on data-dependent control flow (`if
x.sum() > 0`), on most Hugging Face models, and sometimes with an internal error; any failure triggers the fallback
and a line on stderr saying so.

Things to know about the fx path:
- `torch.nn` modules are single steps, even containers such as `nn.TransformerEncoder`. To see inside one, use
  `hooks=True`.
- A parameter that `forward` uses directly (a position table, a class token) becomes an `Input` leaf.
- A step returning several tensors (`nn.MultiheadAttention`, `nn.LSTM`) shows the first; the others are in `also`.
- A module called twice (a shared tower) appears twice; `nest` gives each call its own group (`g.enc`, `g.enc@1`).

**Forward hooks + the autograd graph** (the fallback) runs the real `forward`. Steps are:
- every module call that has no module call inside it;
- every tensor that plain functions computed and that a module then reads, a module returns, or the model returns:
  a residual `+`, a softmax, a similarity. One step per such tensor, named after the last function that made it
  (`add`, `softmax`, `view`, `mul`); several functions in a row therefore share one box;
- every parameter used directly, as an `Input` leaf.

Edges are found by walking the autograd graph back from a step's inputs to the nearest earlier steps. This path
handles anything PyTorch can run. What it cannot show: a tensor that functions compute and that nothing returns or
passes to a module. Make the model return it (see below) or compute it yourself from `raw` and attach it as `also`.

A view to the same shape, or anything else that leaves the numbers untouched, is not a step on either path.

## What a leaf holds

```python
{"id": "layer1_0_conv1", "name": "conv1", "type": "Conv2d",
 "path": ["layer1", "layer1.0"],          # the module calls it sits in, outermost first; "name@1" is a second call
 "from": ["layer1_0_relu"],               # ids of the steps it reads; absent on inputs
 "out": {...}, "grad": {...},             # spec TENSORs, already shrunk
 "also": {"1": {...}},                    # further results, keyed "1", "2"…: rename the keys to say what they are
 "src": "resnet.py:92"}                   # where in the code this step ran
```

`nest(leaves, model)` turns `path` into a tree of groups (`{"id": "g.layer1", "name": "layer1", "type":
"Sequential", "children": [...]}`) under a root named after the model's class. You will usually take that tree apart
and rebuild the levels you designed; the group nodes are still handy as building blocks.

`tensor(t, max_c=8, max_hw=32, max_len=1024)` makes a spec TENSOR from any tensor: drops a batch dimension of 1,
keeps the first `max_c` channels, average-pools maps to at most `max_hw` a side and vectors to `max_len`, and
records the true shape in `full` when it left something out. Use it for anything you add by hand.

## Hugging Face and other keyword-input models

```python
model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager")
leaves, raw = trace(model, {"input_ids": ids, "use_cache": False, "output_attentions": True},
                    lambda out, t: F.cross_entropy(out.logits[:, -1], t), target, hooks=True)
```

- `hooks=True` skips the fx attempt, which only costs time here.
- `use_cache=False`: otherwise the returned key/value cache adds steps nothing reads (they are pruned, but why).
- `output_attentions=True` with eager attention: the attention module then returns its weights, and they become a
  step between the Q/K projections and the weighted sum. Without it the weights never leave the function that
  computes them.
- Pass only the inputs the forward pass needs. An `attention_mask` of all ones can be left out (choose samples of
  equal length for a batch of candidates); passed in, it becomes an input card with nothing reading it.
- Token ids arrive as numbers: in the glue, set that leaf's `out` to `{"tokens": [...decoded words...]}`.

## Page weight

Each step stores its output and its gradient. With a few hundred steps of sequence-shaped activations this reaches
several MB. `build.py` warns above 3 MB. To bring it down:

- `trace(..., max_hw=24)` (or 16) for everything;
- repack the heavy leaves in the glue: `leaf["out"] = tensor(raw[leaf["id"]], max_c=4, max_hw=24)` and
  `leaf.pop("grad", None)` for attention maps, whose gradient nobody reads.

Do not drop steps to save space; the folded repeats cost little and are what makes "every layer" true.

## Several checkpoints on one page

To show the same sample at several moments of training, or before and after fine-tuning:

```python
items = []
for name, state in stages:                      # e.g. weights after 0, 20, 100, 400 steps and at the end
    model.load_state_dict(state)
    leaves, raw = trace(model, x, loss_fn, target)
    ...write the notes for this stage from this raw...
    items.append(variant(name, leaves, example="…what the model says at this stage…"))
spec["variants"] = {"label": "训练进度", "items": items}
```

Build the tree from the last trace; each item overrides `out`, `grad`, `also` and `note` per node id, and the page
gets one more slider. Notes on groups that quote numbers go into `item["nodes"][group_id] = {"note": ...}`.
`examples/mnist_spec.py` does this, getting its stages by wrapping the project's own data loader so that the
project's own `train()` produces them.

## Projects that are not PyTorch

The spec does not care where the numbers come from. Write a few lines in the project's own language that run the
sample and record each step's output, then emit the same JSON. Only the last of these has been run as part of this
skill; the others are the usual way to get intermediate values in those frameworks, so check them as you go:

- **Keras / TensorFlow**: build a second model whose outputs are every layer's output
  (`keras.Model(model.inputs, [l.output for l in model.layers])`), run it once; gradients from `tf.GradientTape`
  watching those outputs. Edges from `layer._inbound_nodes`, or leave `from` out where the model is a plain chain.
- **JAX / Flax**: `flax` modules can return intermediates (`capture_intermediates=True`); gradients by `jax.grad`
  with respect to an added zero perturbation at each step.
- **scikit-learn**: a `Pipeline` is already a chain: call each step's `transform` in turn.
- **Anything else**: restate the forward pass step by step and check the result against the model's own output.
  `examples/tiny-cnn.js` does this in forty lines of JavaScript, with finite-difference gradients.

Omit `from` for a simple chain (the default is "the previous leaf"); give it wherever something branches or skips.

## Models that cannot be run at all

Missing weights you may not download, hardware you do not have, a dependency that will not install: build the tree
from reading the code, give each node an `out` with only `"shape"` (no `data`), leave out `grad`, and say in the
`summary` that the page shows the structure and shapes, not a real run. Shapes come from the code and config, or
from a run with random weights on the CPU when only the checkpoint is the obstacle (then say "random weights").

## Errors

| Message | Cause | Fix |
|---|---|---|
| `fxtrace: torch.fx cannot trace this model (…); using forward hooks` | expected for many models | none |
| `AssertionError: fx graph disagrees with model.forward` | the model gives different outputs on two eval-mode calls | `hooks=True`; then find the randomness |
| `RuntimeError: … does not require grad` at `loss.backward()` | the loss does not depend on any float input or parameter (everything detached, `torch.no_grad()` inside forward) | trace without `loss_fn`; the page then has no gradients |
| `RuntimeError: Expected all tensors to be on the same device` | model on GPU, sample on CPU or the reverse | move both to the CPU |
| a `KeyError` on `raw[...]` | ids differ between paths (`relu_1` on fx, `layer_relu` on hooks) | print `[n["id"] for n in leaves]` once and use what is there |
| an input card with no arrow leaving it | you passed an input the forward pass does not use differentiably (a mask) | leave it out of `x`, or drop the leaf in the glue |
