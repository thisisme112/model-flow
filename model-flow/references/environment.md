# Making the project run: environment, dependencies, weights, data

The page needs one forward pass and one backward pass on one sample. Everything here serves that and nothing more:
no full training setup, no GPU, no dataset download beyond one sample unless the project's own loader insists.

The user's rule for this skill is **install what is needed and tell them**, not "ask about every package". So the
default is to act and report. The exceptions are listed under "Ask first" and they are there because those actions
are slow, large, or change something the user relies on.

## 1. Find the interpreter

Use the environment the project already has: it holds the project's own dependencies at the versions it was written
for. Look, in this order:

| Sign | Interpreter |
|---|---|
| `.venv/`, `venv/`, `env/` with a `pyvenv.cfg` in the project | `<that>/Scripts/python.exe` on Windows, `<that>/bin/python` elsewhere |
| `uv.lock` or `[tool.uv]` in `pyproject.toml` | `uv run python …` from the project root |
| `poetry.lock` | `poetry run python …` |
| `environment.yml` naming a conda env that exists (`conda env list`) | `conda run -n <name> python …` |
| a README or Makefile that says how to run it | what it says |
| an activated environment in the shell you were given (`VIRTUAL_ENV`, `CONDA_PREFIX`) | that one |

If none exists, make one **inside your work folder**, so nothing of the user's changes:

```
uv venv model-flow/.venv            # or: python -m venv model-flow/.venv
```

and install the project's declared requirements into it (see below). Never install into a system-wide Python: other
programs depend on it, and `deps.py` warns when you are about to.

## 2. See what is missing

```
"<python>" <skill>/scripts/deps.py <project> <model file> <training or inference script>
```

Give it the entry files: it then follows only the project's own modules from there, which is the set the forward
pass actually needs. Without them it scans every `.py` in the project and will list tools for evaluation, logging and
deployment that you do not need.

It prints the interpreter, whether torch is there and whether CUDA is, which requirement files the project has, each
third-party import as present or MISSING, and the install command for this environment. Package names for missing
imports are guessed (`cv2` → `opencv-python`): check them against the project's requirement files, which also carry
the version pins.

## 3. Install

**Install without asking, after saying what you are doing**, when all of these hold:

- the package is one the code imports on the way to the forward pass, or is named in the project's requirement files;
- it goes into the project's own environment or the one you made;
- nothing already installed gets replaced by another version;
- the download is modest (packages under about 1 GB in total).

Say it in one message before you start, like this:

> 这个项目的前向计算还缺 3 个包：`timm`、`einops`、`opencv-python`（约 80 MB）。我把它们装进项目自己的环境
> `.venv`，不动其他已安装的包。

Use the command `deps.py` printed. If torch itself is missing and you are making the environment, install the CPU
build: it is a fraction of the size and one sample does not need more
(`--index-url https://download.pytorch.org/whl/cpu` with pip or uv).

**Ask first** when:

- installing would upgrade or downgrade something the project pins, or would rewrite a lock file;
- the only place to install is a system-wide or shared environment;
- it needs a GPU build, a compiler, system packages (`apt`, `brew`, Visual Studio build tools) or administrator rights;
- the total is over about 1 GB, or the connection is clearly slow;
- the package comes from somewhere other than the standard index (a git URL, a private index, a wheel in an issue);
- the project's licence or a login stands between you and a download.

When you ask, give the name, the source and the size, and say what you will do instead if the answer is no (usually:
a separate environment, or shapes only).

If an install fails, read the error; the usual causes are a Python version the package does not support and a
missing build tool. Try the project's pinned version before anything clever. After two failed attempts stop and tell
the user what failed and why.

## 4. Weights

In this order:

1. **A checkpoint that is already there** (`*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, a `checkpoints/` folder, a path
   in a config). Load it the way the project's own code does.
2. **The project's documented pretrained weights** (a `from_pretrained` name, a URL in the README, a hub entry).
   This is a download: tell the user the name, the host and the size first; over about 1 GB, ask. Large downloads
   can sit for minutes without showing progress: run them in the background, do not restart them, and carry on with
   Step 3 meanwhile.
3. **A short training run with the project's own training code**, when that is minutes on a CPU (the MNIST and siamese
   examples do this: one epoch, using the project's `train()` and its default seed). Say how long it took and what
   accuracy came out, and put that number in the page's `summary`.
4. **Random initialisation.** The structure, shapes and edges are still right; the numbers are noise. Then the page's
   `summary` must say so in its first sentence, notes must not interpret the values, and your final message repeats it.

Keep whatever you downloaded or trained out of version control (do not add it), and tell the user where it is.

## 5. One sample

1. The project's own dataset, if it is on disk: go through the project's `Dataset` and transforms, so that the
   sample reaches the model exactly as in training.
2. A sample file in the repository (`assets/`, `examples/`, `tests/`).
3. A standard public sample for that kind of model (an image, a sentence). Small downloads (a photo, a text file)
   need only the one-line notice; a whole dataset is a question for the user.
4. Ask the user for one.

Apply the project's own preprocessing. A sample the model was never meant to see (wrong normalisation, wrong
tokenizer) produces a page that looks fine and means nothing.

## 6. What to tell the user at the end

List, in your final message, each thing you installed or downloaded: its name, where it went, and its size. For
example:

> 安装和下载：
> - `transformers`（约 60 MB）装进了 `D:\proj\.venv`
> - 权重 `openai/clip-vit-base-patch32`（605 MB）在 Hugging Face 的默认缓存目录 `~/.cache/huggingface`
> - 示例图片 `000000039769.jpg`（173 KB）在 `model-flow/`

If you made an environment, say that deleting `model-flow/.venv` removes it.

## Notes for particular setups

- **Windows.** Paths with spaces need quotes. `uv`-made environments have no `pip`: use
  `uv pip install --python "<python>" …`. PowerShell prints Chinese correctly only when the script writes UTF-8;
  the bundled scripts do.
- **Hugging Face.** The cache is shared between projects (`~/.cache/huggingface`); a model the user already has is
  not downloaded again. `HF_ENDPOINT` set in the user's shell means they use a mirror: leave it as it is.
- **GPU-only code.** Code that calls `.cuda()` unconditionally fails on a CPU. Load the model on the CPU yourself
  (`map_location="cpu"`) and call it directly instead of going through the project's training script; do not edit the
  project's files.
- **Config systems.** With Hydra or argparse-driven builders, construct the model by calling the same builder the
  entry point calls, with the default config. Printing the resolved config first saves guessing.
