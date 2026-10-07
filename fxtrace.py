"""Trace one example through a PyTorch model with torch.fx.

    from fxtrace import trace, nest
    leaves, raw = trace(model, x, loss_fn, target)     # x and target carry a batch dimension of 1
    root = nest(leaves, model)                          # or group the leaves into levels yourself

leaves: one spec node per executed operation, in execution order, with real edges ("from"), this example's
        activations ("out") and gradients ("grad"). Functional ops (F.relu, torch.flatten, +) are included.
raw:    name -> full tensor, for computing the numbers quoted in notes.

Self-check: python fxtrace.py [demo.json]
"""
import json
import math
import sys

import torch
import torch.fx as fx
import torch.nn.functional as F


def tensor(t, max_c=8, max_hw=32, max_len=1024):
    """Tensor -> spec TENSOR: batch dimension dropped, shrunk to something a page can carry."""
    t = t.detach().float().cpu()
    if t.dim() > 1:
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


class _Recorder(fx.Interpreter):
    def run_node(self, n):
        out = super().run_node(n)
        if torch.is_tensor(out):
            if out.requires_grad and not out.is_leaf:
                out.retain_grad()  # breaks on in-place ops (ReLU(inplace=True)): switch them off first
            self.seen[n.name] = out
        return out


def trace(model, x, loss_fn=None, target=None):
    model.eval()
    # ponytail: plain symbolic_trace. Models with data-dependent control flow (and most HF transformers) fail
    # here; restate their forward step by step instead (examples/mnist_trace.py does) or use their own fx tracer.
    gm = fx.symbolic_trace(model)
    rec = _Recorder(gm)
    rec.seen = {}
    if x.is_floating_point():
        x = x.clone().requires_grad_(True)
    out = rec.run(x)
    assert torch.allclose(out, model(x), atol=1e-5), "fx graph disagrees with model.forward"
    loss = None
    if loss_fn is not None:
        loss = loss_fn(out, target)
        loss.backward()
        model.zero_grad()

    mods = dict(model.named_modules())
    leaves, kept = [], set()
    for n in gm.graph.nodes:
        t = rec.seen.get(n.name)
        if t is None or n.op in ("output", "get_attr"):
            continue
        stack = [v[0] for v in n.meta.get("nn_module_stack", {}).values() if v[0]]
        if n.op == "call_module":
            typ, path = type(mods[n.target]).__name__, n.target.split(".")[:-1]
        else:
            typ = "Input" if n.op == "placeholder" else getattr(n.target, "__name__", str(n.target))
            path = stack[-1].split(".") if stack else []
        leaf = {"id": n.name, "name": n.target.split(".")[-1] if n.op == "call_module" else n.name, "type": typ, "path": path, "out": tensor(t)}
        if n.op != "placeholder":
            leaf["from"] = [a.name for a in n.all_input_nodes if a.name in kept]
        if t.grad is not None:
            leaf["grad"] = tensor(t.grad)
        leaves.append(leaf)
        kept.add(n.name)
    raw = dict(rec.seen)
    if loss is not None:
        last = next(n for n in gm.graph.nodes if n.op == "output").all_input_nodes[0].name
        shown = {"tokens": [str(target.item())]} if target.numel() == 1 else tensor(target)
        leaves.append({"id": "target", "name": "target", "type": "Input", "path": [], "out": shown})
        leaves.append({"id": "loss", "name": "loss", "type": getattr(loss_fn, "__name__", type(loss_fn).__name__),
                       "path": [], "from": [last, "target"], "out": tensor(loss)})
        raw.update(loss=loss, target=target)
    return leaves, raw


def nest(leaves, model):
    """Group leaves into a tree that follows the module hierarchy. Tree order must stay execution order,
    so a module that is called twice with other work in between should be grouped by hand instead."""
    mods = dict(model.named_modules())
    root = {"name": type(model).__name__, "type": type(model).__name__, "children": []}
    index = {(): root}
    for leaf in leaves:
        path = tuple(leaf["path"])
        for i in range(1, len(path) + 1):
            if path[:i] not in index:
                name = ".".join(path[:i])
                index[path[:i]] = {"id": "g." + name, "name": path[i - 1], "type": type(mods[name]).__name__, "children": []}
                index[path[: i - 1]]["children"].append(index[path[:i]])
        index[path]["children"].append(leaf)
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
    model = Tiny()
    leaves, raw = trace(model, torch.randn(1, 1, 8, 8), F.cross_entropy, torch.tensor([1]))
    by = {n["id"]: n for n in leaves}
    assert by["add"]["from"] == ["block_bn2", "stem"], by["add"]["from"]  # the skip connection is a real edge
    assert by["relu"]["type"] == "relu" and by["relu"]["path"] == ["block"]  # functional op, placed in its module
    assert all("grad" in n for n in leaves if n["id"] not in ("target", "loss"))
    assert by["loss"]["from"] == ["fc", "target"] and by["x"]["out"]["shape"] == [8, 8]
    root = nest(leaves, model)
    assert [c["name"] for c in root["children"]] == ["x", "stem", "block", "adaptive_avg_pool2d", "flatten", "fc", "target", "loss"]
    print("fxtrace ok:", " -> ".join(n["id"] for n in leaves))
    if out:
        spec = {"title": "残差块示例", "summary": "一个随机初始化的小网络，用来检查分支和残差连接能不能画对。", "example": "一张 8×8 的随机噪声图。", "root": root}
        json.dump(spec, open(out, "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    _demo(*sys.argv[1:2])
