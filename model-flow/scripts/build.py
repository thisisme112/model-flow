"""python build.py spec.json out.html — put a spec into the viewer page (../assets/viewer.html).

The result is one self-contained HTML file. Before writing it, the spec is checked for the two mistakes that leave
the page blank: two nodes with the same id, and a "from" that names no node.
"""
import html
import json
import os
import re
import sys

VIEWER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "viewer.html")


def problems(spec):
    """Ids are worked out the way the viewer does: a node's own "id", else its parent's id + "/" + its name."""
    seen, froms, bad = {}, [], []

    def walk(n, parent, top):
        n_id = n.get("id") or (n["name"] if top else parent + "/" + n["name"])
        if n_id in seen:
            bad.append(f'two nodes have the id "{n_id}"; give one of them an "id" of its own')
        seen[n_id] = n
        froms.extend((n_id, f) for f in n.get("from") or [])
        for c in n.get("children") or []:
            walk(c, n_id, False)

    for c in spec["root"]["children"]:
        walk(c, "", True)
    bad += [f'"{n}" reads from "{f}", which is not the id of any node' for n, f in froms if f not in seen]
    return bad


def wordy(spec):
    """Wording a newcomer gives up on: a sentence that runs on, or a paragraph that lists things in prose. (The cure
    is shorter sentences and a list, not less content.)"""
    out = []

    def check(where, text, limit=55):
        for para in (text if isinstance(text, list) else [text or ""]):
            plain = re.sub(r"\*\*|`", "", para)
            for sent in re.split(r"(?<=[。！？；])", plain):
                if len(sent) > limit:
                    out.append(f"{where}: a sentence of {len(sent)} characters (split it, or make it a list): {sent[:22]}…")
            if not para.startswith("- ") and len(re.findall(r"[。！？]", plain)) > 3:
                out.append(f"{where}: a paragraph of {len(re.findall(r'[。！？]', plain))} sentences (break it up, or make its parallel parts a list): {plain[:22]}…")

    def walk(n):
        for k in ("desc", "why", "origin"):
            if n.get(k):
                check(f'{n["name"]} · {k}', n[k])
        if n.get("children") and n.get("origin") and not n.get("origin_kind"):
            out.append(f'{n["name"]}: has no "origin_kind" (fixed / standard / named / own), so its panel has no badge')
        for c in n.get("children") or []:
            walk(c)

    for c in spec["root"]["children"]:
        walk(c)
    check("summary", spec.get("summary"), 70)
    check("example", spec.get("example"), 70)
    for i, t in enumerate(spec.get("background") or []):
        check(f"background {i + 1}", t)
    for c in spec.get("concepts") or []:
        check(f'concept {c["name"]}', c.get("text"))
    return list(dict.fromkeys(out))


def build(spec_path, out_path, viewer=VIEWER):
    spec = json.load(open(spec_path, encoding="utf-8"))
    if "children" not in spec.get("root", {}):
        raise SystemExit('spec needs "root" with "children"')
    bad = problems(spec)
    if bad:
        raise SystemExit("spec cannot be drawn:\n  " + "\n  ".join(bad[:20]))
    data = json.dumps(spec, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    page = open(viewer, encoding="utf-8").read()
    page = re.sub(r'(<script type="application/json" id="spec">).*?(</script>)', lambda m: m.group(1) + data + m.group(2), page, count=1, flags=re.S)
    page = re.sub(r"<title>.*?</title>", lambda m: "<title>" + html.escape(spec.get("title") or "Model Flow") + "</title>", page, count=1)
    open(out_path, "w", encoding="utf-8", newline="\n").write(page)
    size = len(page.encode("utf-8"))
    long = wordy(spec)
    if long:
        print(f"{len(long)} place(s) where the wording is heavy (see \"Easy to read\" in references/glue.md):\n  " + "\n  ".join(long[:60]))
    print(f"wrote {out_path} {size // 1024}KB" + ("  (over 3 MB: trace with max_hw=24 or max_c=4 to keep the page light)" if size > 3 << 20 else ""))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python build.py spec.json out.html")
    build(*sys.argv[1:])
