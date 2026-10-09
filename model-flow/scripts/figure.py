"""Pictures for the page, drawn with matplotlib: picture(fig, caption) gives a spec's {"image": ..., "caption": ...}.

Before it saves a figure, picture() reads it back the way check.py reads a page. Every piece of text has a box, and
no two boxes may share space. untangle(fig) corrects what it finds, in this order:

  1. tick labels along x that run into each other are turned;
  2. a legend lying on text, or on bars, lines or points, goes to the place where it covers no text and the least
     of the data (it stays where it is when no place is better);
  3. a label or an annotation cut off by the edge of the figure comes back onto it; one lying on other text slides
     to the nearest free spot; one that a plotted line runs through, or that the frame of its axes cuts, does the
     same when a spot clear of lines and frame is within reach.

What it moved, and what it could not get apart, picture() prints. The second kind is yours to fix in the script:
shorter labels, a larger figure, fewer of them. Boxes are upright rectangles, so text set at a slant is judged
too wide; only texts at the same slant (turned tick labels) are compared properly.

Needs matplotlib only. `python figure.py` runs the self-check.
"""
import base64
import io
import math
import sys

import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.text import Text

LOCS = ("upper right", "upper left", "lower right", "lower left", "center right", "center left", "upper center", "lower center", "center")


def _hit(a, b, pad=1.0):
    """Two boxes share more than a hairline."""
    return min(a.x1, b.x1) - max(a.x0, b.x0) > pad and min(a.y1, b.y1) - max(a.y0, b.y0) > pad


def _box(t, r):
    if hasattr(t, "update_positions"):  # an annotation: its own extent would take in the arrow
        t.update_positions(r)
        return Text.get_window_extent(t, r)
    return t.get_window_extent(r)


def _shown(ts):
    return [t for t in ts if t is not None and t.get_visible() and t.get_text().strip()]


def _ticks(axis):
    """The tick labels an axis really draws."""
    if not (axis.axes.axison and axis.get_visible()):
        return []
    lo, hi = sorted(axis.get_view_interval())
    return _shown(lab for t in axis.get_major_ticks() if lo <= t.get_loc() <= hi for lab in (t.label1, t.label2))


def _texts(fig):
    """(free, fixed): labels and annotations, which may be moved, and the text that belongs where it is."""
    free, fixed = [], _shown(fig.texts)
    for ax in fig.axes:
        free += _shown(ax.texts)
        fixed += _shown([ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None)]) + _ticks(ax.xaxis) + _ticks(ax.yaxis)
        fixed += _shown(a.label for a in (ax.xaxis, ax.yaxis) if ax.axison and a.get_visible())
    return free, fixed


def _cover(fig, b, r):
    """How much of what the figure plots a box at b would hide: bars, lines and scatter points under it."""
    n = 0
    for ax in fig.axes:
        n += sum(_hit(b, p.get_window_extent(r)) for p in ax.patches if p.get_visible())
        n += sum(bool(l.get_transform().transform_path(l.get_path()).intersects_bbox(b, filled=False)) for l in ax.lines if l.get_visible())
        for c in ax.collections:
            if isinstance(c, PathCollection) and len(c.get_offsets()):
                xy = np.asarray(c.get_offset_transform().transform(c.get_offsets()), float).reshape(-1, 2)
                n += int(((xy[:, 0] > b.x0) & (xy[:, 0] < b.x1) & (xy[:, 1] > b.y0) & (xy[:, 1] < b.y1)).sum())
    return n


def _slide(t, dx, dy):
    """Move a text by pixels, whatever coordinates it was placed in (data, axes, offset points)."""
    try:
        tr, (x, y) = t.get_transform(), t.get_position()
        t.set_position(tr.inverted().transform(tr.transform((t.convert_xunits(x), t.convert_yunits(y))) + np.array([dx, dy])))
        return True
    except Exception:  # coordinates that cannot be turned back (a category axis): leave it, it is reported
        return False


def _apart(t, u, dpi):
    """Two texts set at the same slant (turned tick labels) are clear of each other when their lines are a line
    height apart, although their upright boxes collide."""
    a = t.get_rotation() % 180 if hasattr(t, "get_rotation") else 0
    if not a % 90 or not hasattr(u, "get_rotation") or a != u.get_rotation() % 180:
        return False
    try:
        (x0, y0), (x1, y1) = (v.get_transform().transform((v.convert_xunits(v.get_position()[0]), v.convert_yunits(v.get_position()[1]))) for v in (t, u))
    except Exception:
        return False
    return abs((x1 - x0) * math.sin(math.radians(a)) - (y1 - y0) * math.cos(math.radians(a))) >= max(t.get_fontsize(), u.get_fontsize()) * dpi / 72


def _name(t):
    return "“" + " ".join(t.get_text().split())[:14] + "”"


def untangle(fig):
    """Find text that lies on other text in a matplotlib figure and move it clear.
    Returns (what was moved, what still overlaps): two lists of sentences, both empty for a figure that was fine."""
    fig.canvas.draw()
    r, did = fig.canvas.get_renderer(), []

    for ax in fig.axes:  # 1. tick labels along x that run into each other: turn them, upright if a slant is not enough
        labs = _ticks(ax.xaxis)
        bs = [_box(t, r) for t in labs]
        if not any(t.get_rotation() for t in labs) and any(_hit(a, b) for a, b in zip(bs, bs[1:])):
            gap = min(abs(a.x0 + a.x1 - b.x0 - b.x1) / 2 for a, b in zip(bs, bs[1:]))
            turn = 40 if gap * math.sin(math.radians(40)) > max(b.height for b in bs) + 1 else 90
            ax.tick_params(axis="x", labelrotation=turn)
            for t in labs if turn == 40 else []:
                t.set(ha="left" if t.get_va() == "bottom" else "right", rotation_mode="anchor")
            did.append(f"turned the x tick labels by {turn} degrees")
    if did:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()

    legends = [l for l in [ax.get_legend() for ax in fig.axes] + list(fig.legends) if l is not None and l.get_visible()]
    for leg in legends:  # 2. a legend on text, or on what is plotted: the place that covers no text and the least of the data
        others = [_box(t, r) for t in sum(_texts(fig), [])] + [l.get_window_extent(r) for l in legends if l is not leg]

        def score():
            b = leg.get_window_extent(r)
            return sum(_hit(b, o) for o in others), _cover(fig, b, r)

        was, best = getattr(leg, "_loc", None), (score(), None)
        if best[0] == (0, 0) or leg in fig.legends or not hasattr(leg, "set_loc"):
            continue
        # the nine named places first; where each of them hides something, any place in the axes, from the top down
        b, ax = leg.get_window_extent(r), leg.axes.bbox
        w, h = b.width / ax.width, b.height / ax.height
        grid = [(float(x), float(y)) for y in np.linspace(0.98 - h, 0.02, 5) for x in np.linspace(0.02, 0.98 - w, 7)] if max(w, h) < 0.9 else []
        for loc in (*LOCS, *grid):
            leg.set_loc(loc)
            if score() < best[0]:
                best = (score(), loc)
        if best[1]:
            leg.set_loc(best[1])
            did.append("moved the legend to " + (f"the {best[1]}" if isinstance(best[1], str) else "a free place"))
        else:
            leg._loc = was

    free, fixed = _texts(fig)  # 3. a label on other text: the nearest spot that is free, off the data if there is one
    taken = [_box(t, r) for t in fixed] + [l.get_window_extent(r) for l in legends]
    page = fig.bbox

    def struck(c, t):
        """A plotted line runs through a label at c, or the frame of its axes cuts it or comes within 3px of it."""
        f = t.axes.bbox if getattr(t, "axes", None) is not None and t.axes.axison else None
        if f is not None and _hit(c, f, -3) and not (f.x0 + 3 <= c.x0 and c.x1 <= f.x1 - 3 and f.y0 + 3 <= c.y0 and c.y1 <= f.y1 - 3):
            return True
        return any(l.get_transform().transform_path(l.get_path()).intersects_bbox(c, filled=False) for a in fig.axes for l in a.lines if l.get_visible())

    for k, t in enumerate(free):
        b, how = _box(t, r), None
        off = (max(page.x0 - b.x0, 0) + min(page.x1 - b.x1, 0), max(page.y0 - b.y0, 0) + min(page.y1 - b.y1, 0))
        if max(map(abs, off)) > 1 and _slide(t, *off):  # cut off by the edge of the figure: back onto it
            b, how = _box(t, r), " back onto the figure"
        on_text = any(_hit(b, o) for o in taken)
        if on_text or struck(b, t):
            rest, step = taken + [_box(u, r) for u in free[k + 1:]], b.height + 2
            spots = [b.translated(sx * n * step, sy * n * step) for n in range(1, 9) for sx in (-1, 0, 1) for sy in (-1, 0, 1) if sx or sy]
            spots = [c for c in spots if page.x0 <= c.x0 and c.x1 <= page.x1 and page.y0 <= c.y0 and c.y1 <= page.y1 and not any(_hit(c, o) for o in rest)]
            if spots:
                c = min(spots, key=lambda c: (struck(c, t), _cover(fig, c, r) > 0, math.hypot(c.x0 - b.x0, c.y0 - b.y0)))
                # off other text it goes in any case; off a line or the frame only to a spot that is clear of both
                if (on_text or not struck(c, t)) and _slide(t, c.x0 - b.x0, c.y0 - b.y0):
                    b, how = _box(t, r), how or ""
        if how is not None:
            did.append(f"moved {_name(t)}{how}")
        taken.append(b)

    fig.canvas.draw()  # 4. read it back once more: what is still on top of what
    r = fig.canvas.get_renderer()
    items = [(t, _name(t), _box(t, r)) for t in sum(_texts(fig), [])] + [(l, "the legend", l.get_window_extent(r)) for l in legends]
    left = [f"{a} on {b}" for i, (t, a, p) in enumerate(items) for u, b, q in items[i + 1:] if _hit(p, q) and not _apart(t, u, fig.dpi)]
    return did, left


def picture(fig, caption=None, kind="png"):
    """A matplotlib figure as a spec picture, for "figure" on a node or a concept: untangled, saved, closed.
    kind="jpeg" for a figure that contains a photo, or it weighs several hundred KB."""
    import matplotlib.pyplot as plt

    did, left = untangle(fig)
    for line in ([f"figure {caption or ''!r:.24}: " + "; ".join(did)] if did else []) + \
                ([f"figure {caption or ''!r:.24}: TEXT STILL OVERLAPS, fix it in the script: " + "; ".join(left[:6])] if left else []):
        print(line.encode(sys.stdout.encoding or "utf-8", "replace").decode(sys.stdout.encoding or "utf-8"))
    buf = io.BytesIO()
    fig.savefig(buf, format=kind, dpi=90, facecolor="white", **({"pil_kwargs": {"quality": 80}} if kind == "jpeg" else {}))
    plt.close(fig)
    return {"image": f"data:image/{kind};base64," + base64.b64encode(buf.getvalue()).decode(), **({"caption": caption} if caption else {})}


def _demo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 2.2))  # a legend on an annotation
    ax.bar([0, 1, 2], [3, 1, 2], label="bars")
    ax.annotate("this one", (1, 2.5), ha="center")
    ax.legend(loc="upper center")
    did, left = untangle(fig)
    assert did == ["moved the legend to the upper right"] and not left, (did, left)

    fig, ax = plt.subplots(figsize=(5, 2.4))  # every named place hides a bar, a line or the label: any free place
    ax.bar(range(10), [10, 1, 1, 1, 1, 1, 1, 1, 1, 10], label="bars")
    ax.axvline(4.5)
    ax.annotate("this one", (4.7, 9.6), ha="left")
    ax.legend(loc="upper center")
    did, left = untangle(fig)
    assert did == ["moved the legend to a free place"] and not left, (did, left)

    fig, ax = plt.subplots(figsize=(4, 2.2))  # a legend on a bar, with room beside it
    ax.bar(range(4), [1, 1, 1, 10], label="bars")
    ax.legend(loc="upper right")
    did, left = untangle(fig)
    assert did == ["moved the legend to the upper left"] and not left, (did, left)

    fig, ax = plt.subplots(figsize=(4, 2.2))  # a label cut off by the edge comes back
    ax.plot([0, 1], [0, 1])
    ax.annotate("a label that runs off the page", (1, 0.5), ha="left")
    did, left = untangle(fig)
    assert did == ["moved “a label that r” back onto the figure"] and not left, (did, left)

    fig, ax = plt.subplots(figsize=(4, 2.2))  # a line through a label, and a label cut by the frame
    ax.plot([0, 1], [0, 1])
    ax.annotate("on the line", (0.5, 0.5), ha="center", va="center")
    ax.annotate("on the frame", (0, 0.8), ha="center")
    did, left = untangle(fig)
    assert did == ["moved “on the line”", "moved “on the frame”"] and untangle(fig) == ([], []), (did, left)

    fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)  # labels the author slanted: nothing to do
    ax.plot(range(6))
    ax.set_xticks(range(6), [f"label {k}" for k in range(6)], rotation=45, ha="right")
    assert untangle(fig) == ([], []), untangle(fig)

    fig, ax = plt.subplots(figsize=(4, 2.2))  # two labels on one spot: one of them moves, and stays on the page
    ax.plot([0, 1], [0, 1])
    first, second = ax.text(0.2, 0.7, "first"), ax.annotate("second", (0.2, 0.7), textcoords="offset points", xytext=(2, 0))
    at = first.get_position()
    did, left = untangle(fig)
    assert did == ["moved “second”"] and not left and first.get_position() == at, (did, left)

    fig, ax = plt.subplots(figsize=(3, 2), constrained_layout=True)  # tick labels that run into each other
    ax.plot(range(6))
    ax.set_xticks(range(6), [f"a long label {k}" for k in range(6)])
    did, left = untangle(fig)
    assert did == ["turned the x tick labels by 40 degrees"] and not left, (did, left)
    fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)  # too close together for a slant: upright
    ax.plot(range(12))
    ax.set_xticks(range(12), [f"label {k}" for k in range(12)], fontsize=12)
    did, left = untangle(fig)
    assert did == ["turned the x tick labels by 90 degrees"] and not left, (did, left)

    fig, ax = plt.subplots(figsize=(4, 2.2))  # nothing wrong: nothing moves
    ax.plot([0, 1], [0, 1], label="line")
    ax.set_xlabel("x")
    ax.legend()
    assert untangle(fig) == ([], [])
    assert picture(fig, "ok")["image"].startswith("data:image/png;base64,")
    print("figure ok: a legend on a label or on a bar; a label on a label, off the edge, under a line, on the frame; crowded tick labels; slanted ones and a clean figure left alone")


if __name__ == "__main__":
    _demo()
