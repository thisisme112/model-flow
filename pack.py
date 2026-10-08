"""python pack.py — check the skill folder model-flow/ and zip it as dist/model-flow.skill.

Same rules and same archive layout as the skill-creator's package_skill.py, without needing that skill installed.
"""
import os
import re
import sys
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.join(ROOT, "model-flow")


def check():
    text = open(os.path.join(SKILL, "SKILL.md"), encoding="utf-8").read().replace("\r\n", "\n")
    head = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert head, "SKILL.md must start with --- frontmatter ---"
    fields = dict(re.findall(r"^([a-z-]+):\s*(.*)$", head.group(1), re.M))
    assert set(fields) <= {"name", "description", "license", "allowed-tools", "metadata", "compatibility"}, set(fields)
    name, desc = fields.get("name", ""), fields.get("description", "")
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name) and name == os.path.basename(SKILL), f"name {name!r} must be kebab-case and match the folder"
    assert 0 < len(desc) <= 1024 and "<" not in desc and ">" not in desc, f"description: {len(desc)} characters (max 1024, no angle brackets)"
    extra = [os.path.join(d, f) for d, _, fs in os.walk(SKILL) for f in fs if f == "SKILL.md" and d != SKILL]
    assert not extra, f"only one SKILL.md allowed, also found {extra}"
    lines = text.count("\n")
    return name, len(desc), lines


def pack():
    name, n, lines = check()
    os.makedirs(os.path.join(ROOT, "dist"), exist_ok=True)
    out = os.path.join(ROOT, "dist", name + ".skill")
    files = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for d, subs, fs in os.walk(SKILL):
            subs[:] = [s for s in subs if s not in ("__pycache__", "node_modules") and not (d == SKILL and s == "evals")]
            for f in sorted(fs):
                if not f.endswith(".pyc") and f != ".DS_Store":
                    z.write(os.path.join(d, f), os.path.relpath(os.path.join(d, f), ROOT).replace(os.sep, "/"))
                    files += 1
    print(f"{out}: {files} files, {os.path.getsize(out) // 1024} KB; description {n} characters, SKILL.md {lines} lines")


if __name__ == "__main__":
    sys.exit(pack())
