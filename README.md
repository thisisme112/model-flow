# model-flow

model-flow 是一个 [Claude Code](https://claude.com/claude-code) 技能（skill），用于将机器学习或深度学习项目转换为可交互的模型框架图。输出为单个自包含的 HTML 页面，以一条真实样本为线索，展示数据从输入到损失的完整计算过程。

*A Claude skill that turns an ML/DL project into an interactive, paper-style framework diagram: one self-contained
HTML page that follows one real example from input to loss.*

![孪生网络示例页面：顶部为背景介绍与关键数值，中部为两座 ResNet-18 塔并排的框架图，底部为当前步骤的说明](docs/siamese.png)

## 工作流程

安装后，在包含模型的项目中向 Claude 提出绘制框架图的请求，Claude 将依次完成以下步骤：

1. 阅读项目代码，定位模型、数据处理流程和损失函数。
2. 使用项目自身的代码与权重，对一条真实样本执行一次前向计算和反向传播。
3. 记录每个计算步骤的输出、张量形状、梯度及对应的源码位置。
4. 生成 HTML 页面，在无界面浏览器中回读检查后交付。

## 功能

| 功能 | 说明 |
|---|---|
| 详略分级 | 通过滑块在“输入 → 模型 → 损失”的总览与逐层视图之间切换。 |
| 真实数据 | 每个方框显示该样本在对应步骤的实际输出，而非示意图。 |
| 逐步播放 | 支持播放、暂停、单步前进与后退；面板显示当前步骤的输入、输出和梯度。 |
| 面向初学者的说明 | 每个方框说明其作用、必要性、来历（固定计算、标准做法、知名结构或项目自创）以及是否含可学习参数，并解释所用术语。 |
| 背景介绍 | 页面顶部概述任务、输入与输出、模型思路和训练方式。 |
| 概念卡片 | 对多个步骤共同涉及的概念（如“两座塔”“注意力”）单独说明，配有表格和基于该样本绘制的说明图；正文中的相关词语链接至对应卡片。 |
| 分支与重复结构 | 并行分支（双塔、残差支路、注意力的 Q / K / V）并排绘制；重复的模块默认仅展开第一个，并标注数量。 |
| 自动校正 | 连线绘制完成后自动检查，修正穿过方框或相互重叠的连线；说明图保存前自动检测并修正文字重叠、图例遮挡数据、刻度标签拥挤等问题，无法修正的予以报告。 |
| 外观与导出 | 提供浅色和深色两种外观；可将当前视图导出为 SVG，用于论文或幻灯片。 |

页面不依赖任何外部脚本，可作为单个文件分发。

## 安装

前提：已安装 Claude Code。

### 方式一：复制技能目录（推荐）

克隆本仓库：

```bash
git clone https://github.com/thisisme112/model-flow.git
```

将仓库中的 `model-flow/` 目录（即包含 `SKILL.md` 的目录）复制到技能目录。

macOS / Linux：

```bash
mkdir -p ~/.claude/skills && cp -r model-flow/model-flow ~/.claude/skills/
```

Windows PowerShell：

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.claude\skills" | Out-Null; Copy-Item -Recurse model-flow\model-flow "$env:USERPROFILE\.claude\skills\"
```

如仅需在单个项目中使用，可将该目录复制到项目的 `.claude/skills/` 下。

### 方式二：打包为单个文件

```bash
python pack.py
```

该命令生成 `dist/model-flow.skill`（zip 格式），可在 Claude 应用的设置中作为技能上传，也可直接分发。

## 环境要求

| 依赖 | 用途 | 说明 |
|---|---|---|
| Python 3.9 及以上、`torch` | 运行模型并记录各计算步骤 | 使用项目已有的环境；缺少的包在告知用户后安装 |
| `matplotlib` | 绘制说明图 | 同上；安装体积约 35 MB |
| Chromium 内核浏览器（Chrome、Edge、Chromium、Brave） | 页面生成后的自动检查与截图 | 可选；缺少时无法执行自动检查 |
| 模型权重与一条样本 | 页面中全部数值的来源 | 优先使用项目内已有的文件；需要下载时先征得用户同意，并说明名称、来源和大小 |

无需 GPU。整个过程仅处理一条样本，可在 CPU 上完成。

## 使用方法

### 调用

在项目目录中启动 Claude Code，以自然语言提出请求，例如：

- “为这个项目绘制模型框架图”
- “讲解这个模型的工作方式，并画图说明”
- “为论文绘制模型结构图”

也可直接输入 `/model-flow`。

开始之前，Claude 会简要说明将要绘制的模型、所用样本以及文件的存放位置。

### 输出文件

全部产物位于项目内的 `model-flow/` 目录：

```
<项目>/model-flow/
  spec.py        生成页面数据的脚本；更换样本、调整分组或修改说明文字时编辑此文件
  spec.json      页面数据
  <名称>.html    页面，可直接在浏览器中打开
  shots/         各详略级别的截图
```

该技能不修改项目源码，也不修改项目的 `.gitignore`。

### 页面操作

| 操作 | 效果 |
|---|---|
| 详略滑块 | 最左为总览，向右逐级展开结构。 |
| 播放、上一步、下一步 | 按计算顺序推进。红色方框表示数据当前所在的步骤，浅色方框表示尚未执行的步骤。 |
| 点击方框 | 面板显示该步骤的输入、输出和梯度，以及其作用、必要性和来历。 |
| 方框上的“＋”、模块名称旁的“−” | 展开或收起单个模块。 |
| 带下划线的词语 | 虚线下划线：悬停显示释义；实线下划线：点击跳转至对应的概念卡片。 |
| 键盘 | ← 和 → 单步移动，空格键播放或暂停。 |
| 地址后缀 `#level-3` | 直接打开第 3 级视图。 |

### 导出图片

调整滑块位置及各模块的展开状态，并将浏览器窗口调整至所需宽度，然后点击“导出 SVG”。导出结果为白底矢量图，可在 Inkscape、Illustrator 或 PowerPoint 中打开；如需 PDF，可在上述软件中另存。

## 示例

`examples/` 目录包含六个已生成的页面，下载后可直接在浏览器中打开。

| 页面 | 项目 | 展示内容 |
|---|---|---|
| `mnist.html` | pytorch/examples 的 MNIST | 普通卷积网络；附带训练进度滑块，可查看同一样本在一轮训练中五个时刻的结果 |
| `siamese.html` | pytorch/examples 的孪生网络 | 两座共享权重的 ResNet-18 塔并排绘制，下采样支路并排绘制 |
| `gpt2.html` | Hugging Face `distilgpt2` | Transformer 结构、注意力权重，以及各层之后的预测结果 |
| `clip.html` | Hugging Face `openai/clip-vit-base-patch32` | 图像塔与文本塔，注意力的 Q / K / V 并排绘制 |
| `resblock.html` | 单个残差块 | 跳跃连接的最小示例 |
| `tiny-cnn.html` | 以 JavaScript 实现的小型卷积网络 | 非 PyTorch 项目的接入方式 |

![CLIP 示例页面（深色外观）：两座塔的总览，下方为贯穿全图的概念卡片](docs/clip.png)

生成上述页面的脚本位于 `model-flow/examples/`，可作为编写 `spec.py` 的参考。

## 独立使用脚本

`model-flow/scripts/` 中的脚本均可脱离 Claude 单独运行。

| 命令 | 作用 |
|---|---|
| `python model-flow/scripts/fxtrace.py` | 运行跟踪器的自检 |
| `python model-flow/scripts/deps.py <项目目录> <模型文件>` | 列出项目使用的包，以及当前环境中缺少的包 |
| `python model-flow/scripts/figure.py` | 运行说明图文字重叠校正的自检 |
| `python model-flow/scripts/build.py spec.json page.html` | 将页面数据嵌入页面 |
| `python model-flow/scripts/check.py page.html --shots shots` | 在无界面浏览器中打开页面，检查连线并保存截图 |

在自定义脚本中跟踪 PyTorch 模型：

```python
from fxtrace import trace, nest
leaves, raw = trace(model, x, loss_fn, target)   # x：张量、输入元组或关键字参数字典
tree = nest(leaves, model)                       # 按模块调用分组后的步骤
```

页面数据的格式见 `model-flow/references/spec.md`，完整流程见 `model-flow/SKILL.md`。

## 目录结构

| 路径 | 内容 |
|---|---|
| `model-flow/` | 技能本体，即安装时复制的目录 |
| `model-flow/SKILL.md` | Claude 执行的七个步骤 |
| `model-flow/references/` | 参考文档：环境与安装、跟踪、分组与说明文字的撰写、数据格式、演示方式 |
| `model-flow/scripts/` | 跟踪器、依赖检查、说明图保存与文字重叠校正、页面生成、回读检查 |
| `model-flow/assets/viewer.html` | 通用页面模板 |
| `examples/` | 六个已生成的页面及其数据 |
| `demo/` | 示例所用的项目源码与图片 |
| `docs/` | 本文档使用的截图 |
| `pack.py` | 将技能打包为 `.skill` 文件 |

## 已知限制

- **验证范围**：已在四个 PyTorch 项目（MNIST、孪生 ResNet-18、distilgpt2、CLIP）上完整验证。对 Keras、JAX 和 scikit-learn 仅提供了做法说明，尚未实际验证。
- **未验证的模型类型**：具有多个输出头的检测与分割模型、U-Net 式的长跳跃连接、扩散模型（按时间步循环）、按序列展开的循环网络。
- **运行平台**：自动检查脚本仅在 Windows 与 Edge 上运行过，在 macOS、Linux 及 Chrome 下的行为尚未验证。
- **分支识别**：并行分支需在页面数据中显式标注，不会根据连线自动识别。
- **连线布局**：连线逐条绘制后统一校正一次，未实现全局布线；方框间距过小时，连线只能从间隙中央通过。
- **折行显示**：单行容纳不下的模块会折行显示；折行较多时（例如 CLIP 的最细级别），版面较为零散。
- **样本数量**：每个页面仅跟踪一条样本。
- **界面语言**：页面的界面文字为中文。

## 许可

本仓库的代码与文档采用 [MIT 许可](LICENSE)。`demo/` 目录中的第三方内容适用其各自的许可：

- `demo/pytorch-examples/` 来自 [pytorch/examples](https://github.com/pytorch/examples)，采用 BSD 3-Clause 许可，许可文件位于该目录内。
- `demo/clip/000000039769.jpg` 为 COCO val2017 数据集中的图片，版权归原作者所有。
