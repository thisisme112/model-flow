"""Train pytorch/examples mnist for one epoch and record what the page shows. Run from the mnist/ directory:

    python <this file> out.html

Same model, optimizer, batch size and seed as main.py. Snapshots of the weights are kept at a few training
steps; four test images are then traced through every snapshot (activations, gradients, loss).
"""
import base64
import copy
import json
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

sys.path.insert(0, os.getcwd())
from main import Net  # noqa: E402

CKPTS = [0, 5, 20, 60, 150, 400]  # plus the end of the epoch
SHOW = 8  # feature maps drawn per conv layer


def forward(model, x):
    """Net.forward restated step by step: the original uses functional ops, so hooks cannot see the middle."""
    c1 = F.relu(model.conv1(x))
    c2 = F.relu(model.conv2(c1))
    p = F.max_pool2d(c2, 2)
    f1 = F.relu(model.fc1(torch.flatten(p, 1)))
    z = model.fc2(f1)
    return [x, c1, c2, p, f1, z], F.log_softmax(z, dim=1)


def q(t):
    """Non-negative tensor -> bytes, scaled so its own max is 255."""
    return bytes((t / (t.max().item() or 1) * 255).round().clamp(0, 255).byte().flatten().tolist())


def trace(model, x, y, ch1, ch2):
    x = x[None].clone().requires_grad_(True)
    acts, logp = forward(model, x)
    assert torch.allclose(logp, model(x), atol=1e-5), "restated forward disagrees with Net.forward"
    for a in acts[1:]:
        a.retain_grad()
    loss = F.nll_loss(logp, torch.tensor([y]))
    loss.backward()
    _, c1, c2, p, f1, z = acts
    blob = q(c1[0, ch1]) + q(c2[0, ch2]) + q(p[0, ch2]) + q(f1[0]) + q(x.grad.abs()[0, 0])
    return {
        "b": base64.b64encode(blob).decode(),
        "logits": [round(v, 3) for v in z[0].tolist()],
        "probs": [round(v, 5) for v in logp[0].exp().tolist()],
        "loss": round(loss.item(), 4),
        "zero": [round((c1 == 0).float().mean().item(), 3), round((c2 == 0).float().mean().item(), 3)],
        "nz": int((f1 > 0).sum()),
        "gn": [a.grad.norm().item() for a in acts],
    }


def main(out):
    torch.manual_seed(1)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST("../data", train=True, download=True, transform=tf)
    test = datasets.MNIST("../data", train=False, transform=tf)
    tx = ((test.data.float() / 255 - 0.1307) / 0.3081)[:, None]
    ty = test.targets
    assert torch.allclose(tx[0], test[0][0], atol=1e-5)

    model = Net()
    opt = optim.Adadelta(model.parameters(), lr=1.0)
    losses, snaps = [], {}
    for step, (x, y) in enumerate(torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)):
        if step in CKPTS:
            snaps[step] = copy.deepcopy(model.state_dict())
        model.train()
        opt.zero_grad()
        loss = F.nll_loss(model(x), y)
        loss.backward()
        opt.step()
        losses.append(round(loss.item(), 3))
    snaps[len(losses)] = copy.deepcopy(model.state_dict())

    model.eval()

    def evaluate():
        with torch.no_grad():
            logp = torch.cat([model(b) for b in tx.split(1000)])
        return (logp.argmax(1) == ty).float().mean().item(), F.nll_loss(logp, ty, reduction="none")

    # final model picks the hardest test image and the most active channels
    _, per = evaluate()
    idx = [0, 1, 3]
    per[idx] = -1
    idx.append(int(per.argmax()))
    with torch.no_grad():
        acts, _ = forward(model, tx[idx])
    ch1 = sorted(acts[1].mean((0, 2, 3)).topk(SHOW).indices.tolist())
    ch2 = sorted(acts[2].mean((0, 2, 3)).topk(SHOW).indices.tolist())

    ckpts, traces = [], []
    for step, state in snaps.items():
        model.load_state_dict(state)
        acc, _ = evaluate()
        ckpts.append({"step": step, "acc": round(acc, 4), "k1": [round(v, 3) for v in model.conv1.weight[ch1, 0].flatten().tolist()]})
        traces.append([trace(model, tx[i], int(ty[i]), ch1, ch2) for i in idx])
        print(f"step {step:4d}  acc {acc:.4f}  losses {[t['loss'] for t in traces[-1]]}")

    data = {
        "params": sum(p.numel() for p in model.parameters()), "tests": len(ty), "losses": losses, "ckpts": ckpts, "traces": traces,
        "samples": [{"idx": i, "label": int(ty[i]), "pix": base64.b64encode(bytes(test.data[i].flatten().tolist())).decode()} for i in idx],
    }
    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist-page.html"), encoding="utf-8").read()
    with open(out, "w", encoding="utf-8") as fp:
        fp.write(tpl.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print("wrote", out, os.path.getsize(out) // 1024, "KB")


if __name__ == "__main__":
    main(sys.argv[1])
