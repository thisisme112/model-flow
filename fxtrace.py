"""Trace one example through a PyTorch model.

    from fxtrace import trace, nest
    leaves, raw = trace(model, x, loss_fn, target)     # x and target carry a batch dimension of 1
    root = nest(leaves, model)                          # or group the leaves into levels yourself

x:      one tensor, a tuple of positional inputs, or a dict of keyword inputs.
leaves: one spec node per executed step, in execution order, with real edges ("from"), this example's
        activations ("out") and gradients ("grad"), the further results of a multi-output step ("also") and the
        source line that ran it ("src").
raw:    id -> the tensor shown for that step (further results under "id.1", "id.2"), for the numbers quoted in notes.

Two tracers. torch.fx goes first: it sees functional ops (F.relu, torch.flatten, +) as steps of their own. Where fx
cannot go (data-dependent control flow, most Hugging Face models) forward hooks take over: the steps are then the
innermost module calls, and the edges are read off the autograd graph, so an op between two modules (the + of a
residual, a softmax) has no box of its own. hooks=True forces that path; use it to open torch.nn containers that fx
treats as one step (nn.TransformerEncoder).

Self-check: python fxtrace.py [demo.json]
"""
import inspect
import json
import math
import operator
import os
import re
import sys

import torch
import torch.fx as fx
import torch.nn.functional as F


def tensor(t, max_c=8, max_hw=32, max_len=1024):
    """Tensor -> spec TENSOR: a batch dimension of 1 dropped, shrunk to something a page can carry."""
    t = t.detach().float().cpu()
    if t.dim() > 1 and t.shape[0] == 1:
        t = t[0]
    full = list(t.shape)
    if t.dim() == 1 and t.numel() > max_len:
        t = F.adaptive_avg_pool1d(t[None, None], max_len)[0, 0]
    if t.dim() >= 2:
        t = t.reshape(-1, *t.shape[-2:])[:max_c]
        t = F.adaptive_avg_pool2d(t, (min(t.shape[-2], max_hw), min(t.shape[-1], max_hw)))
        if t.shape[0] == 1:
            t = t[0]
    d = {"shape": list(t.shape), "data": [round(v, 4) for v in t.flatten().tolist()]}
    if len(d["data"]) != math.prod(full):  # only when something was actually left out
        d["full"] = full
    return d


def _tensors(o):
    """Every tensor in a step's result, in order: a tensor, or tuples / lists / dicts of them."""
    if torch.is_tensor(o):
        return [o]
    if isinstance(o, dict):
        o = list(o.values())
    return [t for v in o for t in _tensors(v)] if isinstance(o, (tuple, list)) else []


def _keep(ts):
    """Ask for gradients, and copy the values now: a later in-place op must not change what this step shows."""
    for t in ts:
        if t.requires_grad and not t.is_leaf:
            t.retain_grad()
    return [t.detach().clone() for t in ts]


def _leaf(id, name, type, path, ts, vals, frm, src):
    leaf = {"id": id, "name": name, "type": type, "path": path, "out": tensor(vals[0])}
    if frm is not None:
        leaf["from"] = frm
    if ts[0].grad is not None:
        leaf["grad"] = tensor(ts[0].grad)
    if len(vals) > 1:
        leaf["also"] = {str(k): tensor(v) for k, v in enumerate(vals[1:4], 1)}
    if src:
        leaf["src"] = src
    return leaf


class _Recorder(fx.Interpreter):
    def run_node(self, n):
        out = super().run_node(n)
        ts = _tensors(out)
        if ts:
            self.seen[n.name] = (out, ts, _keep(ts))
        return out


_TORCH = os.path.dirname(torch.__file__) + os.sep


def _where(stack):
    """The innermost frame of an fx stack trace that is not torch's own, as file.py:line."""
    hits = [h for h in re.findall(r'File "(.+?)", line (\d+)', stack or "") if not h[0].startswith(_TORCH)]
    return f"{os.path.basename(hits[-1][0])}:{hits[-1][1]}" if hits else None


def _fx(model, args, kw):
    """Steps are the fx graph nodes that yield a tensor; edges are the graph's own."""
    tracer = fx.Tracer()
    tracer.record_stack_traces = True
    gm = fx.GraphModule(model, tracer.trace(model))
    for n in gm.graph.nodes:
        if n.kwargs.get("inplace"):
            n.kwargs = {**n.kwargs, "inplace": False}
    rec = _Recorder(gm)
    rec.seen = {}
    out = rec.run(*(args or [kw.get(n.target, n.args[0] if n.args else None) for n in gm.graph.nodes if n.op == "placeholder"]))
    with torch.no_grad():
        ref = model(*args, **kw)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(_tensors(out), _tensors(ref))), "fx graph disagrees with model.forward"

    mods, recs, kept, alias = dict(model.named_modules()), [], {}, {}

    def sources(n):  # the shown steps a node reads, looking through the ones that are not shown
        return list(dict.fromkeys(s for a in n.all_input_nodes for s in ([a.name] if a.name in kept else alias.get(a.name, []))))

    for n in gm.graph.nodes:
        frm, got = sources(n), rec.seen.get(n.name)
        if n.op == "output":
            last = frm
            continue
        if got is None:  # no tensor here (a size, a None): reading it is not data flow, so it passes no edge on
            continue
        out_n, ts, vals = got
        if n.target is operator.getitem and not torch.is_tensor(rec.seen.get(getattr(n.args[0], "name", None), (out_n,))[0]):
            # picking one result out of a multi-output step: a step of its own only when it is a tensor somebody uses
            # and the multi-output step is not already showing it
            if not n.users or not torch.is_tensor(out_n) or (len(frm) == 1 and out_n is kept[frm[0]]):
                alias[n.name] = frm
                continue
        path = list(n.meta.get("nn_module_stack", {}))  # the module calls this node sits in; fx marks a repeat call "name@1"
        if n.op in ("placeholder", "get_attr"):  # data from outside: the sample, or a parameter forward uses directly
            typ, name, frm = "Input", n.target.split(".")[-1], None
        elif n.op == "call_module":
            typ, name, path = type(mods[n.target]).__name__, n.target.split(".")[-1], path[:-1]
        else:
            typ, name = getattr(n.target, "__name__", str(n.target)), n.name
        kept[n.name] = ts[0]
        recs.append(dict(id=n.name, name=name, type=typ, path=path, ts=ts, vals=vals, frm=frm, src=_where(n.stack_trace)))
    return recs, last, out


def _caller():
    """file.py:line of the code that made the current module call: the first frame outside torch."""
    f = sys._getframe(2)
    while f and f.f_globals.get("__name__", "").split(".")[0] == "torch":
        f = f.f_back
    return f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}" if f and f.f_code is not _hooks.__code__ else None


def _hooks(model, args, kw):
    """Steps are the module calls with no module call inside them; edges are read off the autograd graph."""
    ids = {m: n for n, m in model.named_modules() if n}
    recs, stack, made, count = [], [], {}, {}  # made: what produced a tensor -> index of the step that shows it

    def key(t):  # a tensor autograd did not make (an input, or a view of an integer one) is known by its storage
        return t.grad_fn if t.grad_fn is not None else t.untyped_storage().data_ptr()

    def producers(ts):
        found, todo, seen = set(), [t.grad_fn if t.grad_fn is not None else t for t in ts], set()
        while todo:
            f = todo.pop()
            if torch.is_tensor(f) or hasattr(f, "variable"):  # AccumulateGrad: a leaf tensor, i.e. an input or a weight
                f = (f if torch.is_tensor(f) else f.variable).untyped_storage().data_ptr()
            if f in seen:
                continue
            seen.add(f)
            if f in made:
                found.add(made[f])
            elif not isinstance(f, int):
                todo += [g for g, _ in f.next_functions if g is not None]
        return [recs[i]["id"] for i in sorted(found)]

    def add(id, name, typ, path, ts, frm, src):
        recs.append(dict(id=id, name=name, type=typ, path=path, ts=ts, vals=_keep(ts), frm=frm, src=src))
        for t in ts:
            made[key(t)] = len(recs) - 1

    def pre(m, a, k):
        if stack:
            stack[-1][0] = False  # the caller has a module call inside it, so it is not a step itself
        c = count[m] = count.get(m, -1) + 1
        stack.append([True, c, producers(_tensors([a, k])), _caller(), ids[m] + (f"@{c}" if c else "")])

    def post(m, a, k, out):
        innermost, c, frm, src, _ = stack.pop()
        ts = _tensors(out)
        if innermost and ts:
            add(ids[m].replace(".", "_") + (f"_{c}" if c else ""), ids[m].split(".")[-1], type(m).__name__, [f[4] for f in stack], ts, frm, src)

    names = list(kw) or list(inspect.signature(model.forward).parameters)
    for name, t in zip(names, kw.values() if kw else args):
        if torch.is_tensor(t):
            add(name, name, "Input", [], [t], None, None)
    # ponytail: a parameter that forward uses directly (a position table added by hand) gets no box on this path;
    # the fx path shows it. Add it here from AccumulateGrad.variable when a hook-traced model needs it.
    hs = [h for m in ids for h in (m.register_forward_pre_hook(pre, with_kwargs=True), m.register_forward_hook(post, with_kwargs=True))]
    try:
        out = model(*args, **kw)
    finally:
        for h in hs:
            h.remove()
    return recs, producers, out


def trace(model, x, loss_fn=None, target=None, hooks=None):
    """hooks: None tries torch.fx and falls back to forward hooks, True goes straight to hooks, False is fx or an error."""
    model.eval()
    for m in model.modules():
        if getattr(m, "inplace", False):
            m.inplace = False  # same numbers; in place, a ReLU would overwrite the output of the step before it
    kw = dict(x) if isinstance(x, dict) else {}
    args = () if kw else tuple(x) if isinstance(x, (tuple, list)) else (x,)

    def live(t):
        return t.clone().requires_grad_(True) if torch.is_tensor(t) and t.is_floating_point() else t

    args, kw = tuple(map(live, args)), {k: live(v) for k, v in kw.items()}
    if not hooks:
        try:
            recs, last, out = _fx(model, args, kw)
        except Exception as e:
            if hooks is False:
                raise
            print(f"fxtrace: torch.fx cannot trace this model ({type(e).__name__}: {str(e)[:100]}); using forward hooks", file=sys.stderr)
            hooks = True
    if hooks:
        recs, producers, out = _hooks(model, args, kw)
    loss = None
    if loss_fn is not None:
        loss = loss_fn(out, target)
        loss.backward()
        if hooks:
            last = producers([loss])
    leaves = [_leaf(**r) for r in recs]
    raw = {f"{r['id']}.{k}" if k else r["id"]: v for r in recs for k, v in enumerate(r["vals"])}
    model.zero_grad()
    if loss is not None:
        if target is not None:
            shown = {"tokens": [str(target.item())]} if target.numel() == 1 else tensor(target)
            leaves.append({"id": "target", "name": "target", "type": "Input", "path": [], "out": shown})
            last = [*last, "target"]
        leaves.append({"id": "loss", "name": "loss", "type": getattr(loss_fn, "__name__", type(loss_fn).__name__),
                       "path": [], "from": last, "out": tensor(loss)})
        raw.update(loss=loss.detach(), target=target)
    return leaves, raw


def nest(leaves, model):
    """Group leaves into a tree of module calls, following each leaf's "path" (the calls it sits in, outermost first;
    "name@1" is the second call of a module). Tree order is execution order, so a call whose steps are interrupted
    by other work gets a second group."""
    mods = dict(model.named_modules())
    root = {"name": type(model).__name__, "type": type(model).__name__, "children": []}
    chain, prev, used = [root], (), {}  # chain: the groups that are open right now, outermost first
    for leaf in leaves:
        path = tuple(leaf["path"])
        same = next((i for i, (a, b) in enumerate(zip(path, prev)) if a != b), min(len(path), len(prev)))
        del chain[same + 1:]
        for i in range(same, len(path)):
            qual, outer = path[i].split("@")[0], path[i - 1].split("@")[0] + "." if i else ""
            k = used[path[i]] = used.get(path[i], 0) + 1
            g = {"id": "g." + path[i] + (f"#{k}" if k > 1 else ""), "name": qual.removeprefix(outer), "type": type(mods[qual]).__name__, "children": []}
            chain[-1]["children"].append(g)
            chain.append(g)
        chain[-1]["children"].append(leaf)
        prev = path
    return root


def _demo(out=None):
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.conv1, self.bn1 = nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c)
            self.conv2, self.bn2 = nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c)

        def forward(self, x):
            y = F.relu(self.bn1(self.conv1(x)))
            return F.relu(self.bn2(self.conv2(y)) + x)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem, self.block, self.fc = nn.Conv2d(1, 4, 3, padding=1), Block(4), nn.Linear(4, 3)

        def forward(self, x):
            x = self.block(self.stem(x))
            return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))

    torch.manual_seed(0)
    model, y = Tiny(), torch.tensor([1])
    leaves, raw = trace(model, torch.randn(1, 1, 8, 8), F.cross_entropy, y)
    by = {n["id"]: n for n in leaves}
    assert by["add"]["from"] == ["block_bn2", "stem"], by["add"]["from"]  # the skip connection is a real edge
    assert by["relu"]["type"] == "relu" and by["relu"]["path"] == ["block"]  # functional op, placed in its module
    assert all("grad" in n for n in leaves if n["id"] not in ("target", "loss"))
    assert by["loss"]["from"] == ["fc", "target"] and by["x"]["out"]["shape"] == [8, 8]
    assert by["stem"]["src"].startswith("fxtrace.py:"), by["stem"]["src"]
    root = nest(leaves, model)
    assert [c["name"] for c in root["children"]] == ["x", "stem", "block", "adaptive_avg_pool2d", "flatten", "fc", "target", "loss"]
    print("fxtrace ok:", " -> ".join(n["id"] for n in leaves))
    if out:
        spec = {"title": "残差块示例", "summary": "一个随机初始化的小网络，用来检查分支和残差连接能不能画对。", "example": "一张 8×8 的随机噪声图。", "root": root}
        json.dump(spec, open(out, "w", encoding="utf-8"), ensure_ascii=False)

    def steps(m, x, **kw):
        ls, raw = trace(m, x, F.cross_entropy, y, **kw)
        return {n["id"]: n for n in ls}, [n["type"] for n in ls], raw

    # an in-place ReLU must not overwrite what the step before it shows
    by, _, _ = steps(nn.Sequential(nn.Conv2d(1, 2, 3), nn.BatchNorm2d(2), nn.ReLU(inplace=True), nn.Flatten(), nn.Linear(32, 3)), torch.randn(1, 1, 6, 6))
    assert min(by["_1"]["out"]["data"]) < 0 <= min(by["_2"]["out"]["data"])

    class Attn(nn.Module):  # a step with two results, and a parameter used directly
        def __init__(self):
            super().__init__()
            self.pos, self.ln, self.att, self.fc = nn.Parameter(torch.randn(1, 5, 8)), nn.LayerNorm(8), nn.MultiheadAttention(8, 2, batch_first=True), nn.Linear(8, 3)

        def forward(self, x):
            h = self.ln(x + self.pos)
            a, _ = self.att(h, h, h)
            return self.fc((h + a).mean(1))

    seq = torch.randn(1, 5, 8)
    by, types, _ = steps(Attn(), seq)
    assert types == ["Input", "Input", "add", "LayerNorm", "MultiheadAttention", "add", "mean", "Linear", "Input", "cross_entropy"], types
    assert by["add"]["from"] == ["x", "pos"] and by["att"]["from"] == ["ln"] and by["add_1"]["from"] == ["ln", "att"]
    assert by["att"]["also"]["1"]["shape"] == [5, 5] and "grad" in by["pos"]  # the attention weights; d(loss)/d(parameter)

    class Rnn(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn, self.fc = nn.LSTM(8, 6, batch_first=True), nn.Linear(6, 3)

        def forward(self, x):
            o, (h, c) = self.rnn(x)
            return self.fc(o[:, -1])

    by, types, _ = steps(Rnn(), seq)
    assert types == ["Input", "LSTM", "getitem", "Linear", "Input", "cross_entropy"] and by["rnn"]["from"] == ["x"], types

    class Twin(nn.Module):  # two inputs through one shared tower
        def __init__(self):
            super().__init__()
            self.enc, self.head = nn.Sequential(nn.Linear(4, 4), nn.ReLU()), nn.Linear(8, 3)

        def forward(self, a, b):
            return self.head(torch.cat([self.enc(a), self.enc(b)], 1))

    twin = Twin()
    ls, _ = trace(twin, (torch.randn(1, 4), torch.randn(1, 4)), F.cross_entropy, y)
    kids = nest(ls, twin)["children"]
    assert [c.get("id") for c in kids] == ["a", "b", "g.enc", "g.enc@1", "cat", "head", "target", "loss"], [c.get("id") for c in kids]
    assert kids[3]["children"][0]["from"] == ["b"]

    class Branchy(nn.Module):  # fx cannot trace the `if`; hooks can. `h += x` is in place.
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(4, 4), nn.Linear(4, 3)

        def forward(self, x):
            h = self.a(x)
            if h.sum() > -1e9:
                h += x
            return self.b(F.relu(h))

    m, x = Branchy(), torch.randn(1, 4)
    by, types, raw = steps(m, x)
    assert types == ["Input", "Linear", "Linear", "Input", "cross_entropy"] and by["b"]["from"] == ["x", "a"], (types, by["b"])
    assert torch.equal(raw["a"], m.a(x)) and torch.equal(raw["b"], m(x)) and "grad" in by["a"] and "grad" in by["x"]
    assert by["b"]["src"].startswith("fxtrace.py:") and by["loss"]["from"] == ["b", "target"]
    by, types, _ = steps(Attn(), seq, hooks=True)  # the same edges from the autograd graph, without the functional steps
    assert types == ["Input", "LayerNorm", "MultiheadAttention", "Linear", "Input", "cross_entropy"] and by["fc"]["from"] == ["ln", "att"], (types, by["fc"])
    print("fxtrace ok: in-place, multi-output, parameters, shared modules, hook fallback")


if __name__ == "__main__":
    _demo(*sys.argv[1:2])
