# Presenting the page

The page is one HTML file with everything inside it (the only thing it fetches is a web font, and it reads fine
without). It can be opened from disk, sent as a file, or published.

## Opening it

- **The user is at this machine**: give the path to `model-flow/<name>.html`; double-clicking opens it in their
  browser. If you have a tool that opens files or URLs for the user, use it.
- **Your browser tool refuses `file://`**: serve the folder and open the local address:
  `python -m http.server 8765 --bind 127.0.0.1 --directory model-flow`, then `http://127.0.0.1:8765/<name>.html`.
  Stop the server when you are done.
- **You can publish pages** (an artifact or page-hosting tool): publish the HTML as it is. On claude.ai artifacts,
  declare the `downloads` capability (`capabilities: {"downloads": true}`): a published page may not start a download
  itself, and "导出 SVG" then saves through that capability. Publishing puts the user's model, and possibly their
  data sample, on a server: do it when they ask for a link or have already been getting links from you, not as a
  surprise. Published pages start private; say so.

## The tour

Do not list features. Tell the user where to look, in the order that makes this model make sense. Five or six lines
in your message, adapted to the project:

1. **Start at the left end of the slider.** "最左边一档是整体：一张图进去，模型给出判断，和正确答案比出损失。"
2. **Press play once.** The red box is where the data is; faded boxes have not happened yet. The panel below shows
   what went in, what the step does, what came out.
3. **Move the slider one stop at a time** and say what each stop adds for this model ("第二档两座塔并排出现；第四档
   能看到十二层，只展开了第一层，其余点“＋”").
4. **Point at the one box worth clicking.** Every example has one: the layer where the prediction flips, the
   attention map that looks at the right word, the similarity that barely wins. Name it and say what to see there.
5. **Mention the other controls only if they apply**: the second slider when the page has `variants` ("拖动训练进度，
   看同一张图在训练前后的结果"), "导出 SVG" when they want a figure, the light/dark button.

Keyboard: ← and → step, space plays and pauses. A link ending in `#level-3` opens at that stop, which is handy in a
talk.

## A figure for a paper or a slide

"导出 SVG" writes what is on screen at that moment: the chosen level, with whatever modules are opened or closed.
So: set the slider, open or close individual modules with ＋ and −, then export. The file is plain vector shapes and
text, ink on white whatever the page's theme, heat maps as embedded images. It opens in Inkscape, Illustrator and
PowerPoint; convert to PDF there for LaTeX. The page width decides where rows wrap: make the browser window as wide
as the figure should be before exporting.

On a published page, the browser asks before saving; if no save is offered there, open the local file instead.

## The final message

Lead with what the page shows, then the facts the user needs. A template, in the user's language:

> 页面做好了：`model-flow/clip.html`
>
> 它跟着一张照片和三句话走完 CLIP：图片和文字各过一座塔，变成两个向量，再比相似度。这个例子里模型选了
> “a photo of a cat”（74%），第二名是 “a photo of a couch”（25%）。
>
> **怎么看**：（the tour, five or six lines）
>
> **文件**：都在 `model-flow/`。重新生成：`python model-flow/spec.py model-flow/spec.json`，再
> `python <skill>/scripts/build.py model-flow/spec.json model-flow/clip.html`。
>
> **安装和下载**：（each item: name, where, size）
>
> **没做到的**：（random weights; a part of the model left out; an arrow problem that stayed; anything the reader would
> otherwise assume）

Keep it honest and short. If the check was clean at both widths, say so in one line; if you could not run it, say
that instead. If the weights are random, that sentence comes first, not last.

## If the user wants changes

- **Different sample**: change the rule or the index in `spec.py`, rerun, rebuild. Everything else follows.
- **Different grouping or wording**: edit `spec.py`, not the JSON or the HTML: the next rerun would lose it.
- **Different look**: the tokens at the top of `assets/viewer.html` (colours, fonts) are the place, in a copy of the
  viewer passed to `build.build(spec, out, viewer=...)`. Do not restyle the bundled viewer for one project.
- **Something the viewer cannot do** (a different kind of diagram, another layout): say so rather than bending the
  spec; this skill draws one kind of figure.
