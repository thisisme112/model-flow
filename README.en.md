[中文](README.md) | **English**

# model-flow

model-flow is a [Claude Code](https://claude.com/claude-code) skill that turns a machine-learning or deep-learning project into an interactive framework diagram. The output is a single self-contained HTML page that follows one real example through the whole computation, from the input to the loss.

![The siamese-network example page: background and key figures at the top, the framework diagram with two ResNet-18 towers side by side in the middle, the explanation of the current step at the bottom](docs/siamese.png)

The interface text of the generated pages is in Chinese; see [Known limitations](#known-limitations).

## How it works

Once the skill is installed, ask Claude to draw a framework diagram in any project that contains a model. Claude then:

1. Reads the project's code and locates the model, the data pipeline and the loss function.
2. Runs one forward and one backward pass on a single real example, using the project's own code and weights.
3. Records the output, tensor shape, gradient and source location of every computation step.
4. Generates the HTML page, reads it back in a headless browser to check it, and hands it over.

## Features

| Feature | Description |
|---|---|
| Levels of detail | A slider moves between the overview (input → model → loss) and the layer-by-layer view. |
| Real data | Every box shows what the example actually became at that step, not a schematic. |
| Step-by-step playback | Play, pause, step forward and step back; a panel shows the input, output and gradient of the current step. |
| Explanations for beginners | Every box states what it does, why it is needed, where it comes from (a fixed computation, a standard technique, a named architecture or the project's own design) and whether it has learnable parameters; the terms it uses are explained. |
| Background | The top of the page summarises the task, the input and output, the idea behind the model and how it is trained. |
| Concept cards | Concepts that several steps refer to (for example "the two towers" or "attention") are explained on cards of their own, with tables and figures drawn from the traced example; the words that name them link to the cards. |
| Branches and repeats | Parallel branches (two towers, residual shortcuts, the Q / K / V projections of attention) are drawn side by side; of repeated modules only the first is expanded by default, marked with the count. |
| Automatic correction | After the arrows are drawn they are read back, and arrows that cross a box or lie on one another are corrected. Before a figure is saved, overlapping text, a legend that hides data and crowded tick labels are detected and corrected; what cannot be corrected is reported. |
| Appearance and export | Light and dark appearances; the current view can be exported as SVG for papers and slides. |

The page uses no external scripts and can be distributed as a single file.

## Installation

Prerequisite: Claude Code is installed.

### Option 1: copy the skill directory (recommended)

Clone this repository:

```bash
git clone https://github.com/thisisme112/model-flow.git
```

Copy the `model-flow/` directory inside the repository (the one that contains `SKILL.md`) to the skills directory.

macOS / Linux:

```bash
mkdir -p ~/.claude/skills && cp -r model-flow/model-flow ~/.claude/skills/
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.claude\skills" | Out-Null; Copy-Item -Recurse model-flow\model-flow "$env:USERPROFILE\.claude\skills\"
```

To use the skill in one project only, copy the directory to that project's `.claude/skills/` instead.

### Option 2: package it as a single file

```bash
python pack.py
```

This writes `dist/model-flow.skill` (a zip archive), which can be uploaded as a skill in the settings of the Claude apps or passed on directly.

## Requirements

| Dependency | Purpose | Notes |
|---|---|---|
| Python 3.9 or later, `torch` | Running the model and recording each step | The project's existing environment is used; missing packages are installed after the user has been told |
| `matplotlib` | Drawing the explanatory figures | As above; about 35 MB installed |
| A Chromium-based browser (Chrome, Edge, Chromium, Brave) | Automatic checking and screenshots after the page is built | Optional; without it the automatic check cannot run |
| Model weights and one example | The source of every number on the page | Files already in the project are preferred; before anything is downloaded the user is asked, with its name, source and size |

No GPU is required. Only one example is processed, which a CPU handles.

## Usage

### Invoking the skill

Start Claude Code in the project directory and ask in natural language, for example:

- "Draw a framework diagram of the model in this project."
- "Explain how this model works, with a diagram."
- "Draw a model architecture figure for my paper."

Alternatively, enter `/model-flow`.

Before starting, Claude states briefly which model it will draw, which example it will use and where the files will be placed.

### Output files

Everything is written to the `model-flow/` directory inside the project:

```
<project>/model-flow/
  spec.py        the script that produces the page data; edit it to change the example, the grouping or the texts
  spec.json      the page data
  <name>.html    the page; open it directly in a browser
  shots/         screenshots of every level of detail
```

The skill does not modify the project's source code or its `.gitignore`.

### Controls

| Control | Effect |
|---|---|
| Detail slider | The leftmost position is the overview; each position to the right expands one more level of structure. |
| Play, previous, next | Advance in the order of computation. The red box is the step the data is at; faded boxes have not been reached yet. |
| Click on a box | The panel shows the input, output and gradient of that step, together with what it does, why it is needed and where it comes from. |
| "＋" on a box, "−" beside a module's name | Expand or collapse a single module. |
| Underlined words | Dotted underline: hover for the definition. Solid underline: click to jump to the concept card. |
| Keyboard | ← and → step; the space bar plays and pauses. |
| Address suffix `#level-3` | Opens the page at level 3. |

### Exporting a figure

Set the slider and expand or collapse modules as required, resize the browser window to the intended width of the figure, then click "导出 SVG" (Export SVG). The result is a vector image on a white background that opens in Inkscape, Illustrator and PowerPoint; save it as PDF from one of these if a PDF is needed.

## Examples

The `examples/` directory contains six generated pages. Download one and open it in a browser.

| Page | Project | What it shows |
|---|---|---|
| `mnist.html` | MNIST from pytorch/examples | A plain convolutional network; an additional slider shows the same example at five moments of one training epoch |
| `siamese.html` | The siamese network from pytorch/examples | Two ResNet-18 towers with shared weights drawn side by side; downsampling shortcuts drawn side by side |
| `gpt2.html` | Hugging Face `distilgpt2` | A Transformer, attention weights, and the prediction after each layer |
| `clip.html` | Hugging Face `openai/clip-vit-base-patch32` | An image tower and a text tower; the Q / K / V projections of attention side by side |
| `resblock.html` | A single residual block | A minimal example of a skip connection |
| `tiny-cnn.html` | A small convolutional network written in JavaScript | How a project that does not use PyTorch is connected |

![The CLIP example page in the dark appearance: an overview of the two towers, with the concept cards below](docs/clip.png)

The scripts that produced these pages are in `model-flow/examples/` and serve as references for writing `spec.py`.

## Using the scripts on their own

The scripts in `model-flow/scripts/` run independently of Claude.

| Command | Purpose |
|---|---|
| `python model-flow/scripts/fxtrace.py` | Runs the tracer's self-check |
| `python model-flow/scripts/deps.py <project directory> <model file>` | Lists the packages the project uses and those missing from the current environment |
| `python model-flow/scripts/figure.py` | Runs the self-check of the text-overlap correction for figures |
| `python model-flow/scripts/build.py spec.json page.html` | Embeds the page data in the page |
| `python model-flow/scripts/check.py page.html --shots shots` | Opens the page in a headless browser, checks the arrows and saves screenshots |

Tracing a PyTorch model from a script of your own:

```python
from fxtrace import trace, nest
leaves, raw = trace(model, x, loss_fn, target)   # x: a tensor, a tuple of inputs, or a dict of keyword inputs
tree = nest(leaves, model)                       # the steps grouped by module call
```

The format of the page data is described in `model-flow/references/spec.md`, the complete procedure in `model-flow/SKILL.md`.

## Repository layout

| Path | Contents |
|---|---|
| `model-flow/` | The skill itself; this is the directory that is installed |
| `model-flow/SKILL.md` | The seven steps Claude follows |
| `model-flow/references/` | Reference documents: environment and installation, tracing, grouping and writing the texts, the data format, presenting |
| `model-flow/scripts/` | The tracer, the dependency check, figure saving with text-overlap correction, the page builder, the read-back check |
| `model-flow/assets/viewer.html` | The generic page template |
| `examples/` | Six generated pages and their data |
| `demo/` | The project sources and the image used by the examples |
| `docs/` | The screenshots used in this document |
| `pack.py` | Packages the skill as a `.skill` file |

## Known limitations

- **Scope of validation**: validated end to end on four PyTorch projects (MNIST, a siamese ResNet-18, distilgpt2, CLIP). For Keras, JAX and scikit-learn only the approach is described; it has not been validated.
- **Model types not yet validated**: detection and segmentation models with several output heads, U-Net-style long skip connections, diffusion models (a loop over time steps), recurrent networks unrolled over a sequence.
- **Platforms**: the automatic check has only been run on Windows with Edge; its behaviour on macOS, on Linux and with Chrome has not been verified.
- **Branch detection**: parallel branches must be marked in the page data; they are not inferred from the edges.
- **Arrow routing**: arrows are drawn one at a time and corrected in a single pass; there is no global router. Where boxes are very close together, an arrow can only pass through the middle of the gap.
- **Wrapping**: a module that does not fit on one row continues on the next; with many wrapped modules (for example CLIP at the finest level) the layout becomes fragmented.
- **Number of examples**: each page follows a single example.
- **Interface language**: the interface text of the page is in Chinese. Names, descriptions and notes are written by the skill in the user's language.

## Licence

The code and documentation of this repository are released under the [MIT licence](LICENSE). Third-party content in `demo/` remains under its own licence:

- `demo/pytorch-examples/` comes from [pytorch/examples](https://github.com/pytorch/examples) and is under the BSD 3-Clause licence; the licence file is in that directory.
- `demo/clip/000000039769.jpg` is an image from the COCO val2017 dataset; copyright remains with its author.
