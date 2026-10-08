"""python deps.py PROJECT_DIR [entry.py ...] — what a project imports, and what this interpreter lacks.

Run it WITH THE INTERPRETER THE PROJECT SHOULD RUN IN (its venv's python): a package counts as present when that
interpreter can import it. With entry files (the model file, the training script) only those and the project's own
modules they import are read; without, every .py under the project.

It installs nothing. It prints the interpreter, the packages found and missing, and the command that would install
the missing ones into this interpreter's environment. `python deps.py --selfcheck` tests the scanner.
"""
import ast
import importlib.metadata
import importlib.util
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]  # this folder's own scripts are not the project's packages
SKIP = {".git", ".hg", "node_modules", "__pycache__", "site-packages", "build", "dist", ".tox", ".eggs", ".mypy_cache", ".pytest_cache"}
# import name -> name to install, where they differ
PIP = {"PIL": "pillow", "cv2": "opencv-python", "sklearn": "scikit-learn", "skimage": "scikit-image", "yaml": "pyyaml", "bs4": "beautifulsoup4",
       "attr": "attrs", "dateutil": "python-dateutil", "dotenv": "python-dotenv", "hydra": "hydra-core", "faiss": "faiss-cpu", "Bio": "biopython",
       "pytorch_lightning": "pytorch-lightning", "torch_geometric": "torch-geometric", "jwt": "pyjwt", "serial": "pyserial", "zmq": "pyzmq",
       "docx": "python-docx", "fitz": "pymupdf", "google.protobuf": "protobuf", "OpenSSL": "pyopenssl", "magic": "python-magic", "lightning_fabric": "lightning"}
DECLARED = ("requirements.txt", "requirements-dev.txt", "requirements", "pyproject.toml", "setup.py", "setup.cfg", "environment.yml", "environment.yaml",
            "Pipfile", "poetry.lock", "uv.lock", "conda-lock.yml")


def imports(path):
    """Top-level names a file imports absolutely -> first line; and the modules it names in full, for following local ones."""
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except SyntaxError:
        return {}, []
    tops, full = {}, []
    for n in ast.walk(tree):
        names = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module] if isinstance(n, ast.ImportFrom) and n.module and not n.level else []
        for name in names:
            tops.setdefault(name.split(".")[0], n.lineno)
            full.append(name)
    return tops, full


def local(name, roots):
    """The file of a module that belongs to the project itself, if it is one."""
    for r in roots:
        base = os.path.join(r, *name.split("."))
        for p in (base + ".py", os.path.join(base, "__init__.py")):
            if os.path.exists(p):
                return p
        if os.path.isdir(base):
            return base
    return None


def scan(project, entries):
    project = os.path.abspath(project)
    if entries:
        todo, files = [os.path.abspath(e) for e in entries], []
        while todo:  # follow the project's own modules from the entry files
            f = todo.pop()
            if f in files or not f.endswith(".py"):
                continue
            files.append(f)
            for name in imports(f)[1]:
                hit = local(name, [project, os.path.dirname(f)])
                if hit:
                    todo.append(hit if hit.endswith(".py") else os.path.join(hit, "__init__.py"))
    else:
        files = []
        for d, subs, names in os.walk(project):
            subs[:] = [s for s in subs if s not in SKIP and not os.path.exists(os.path.join(d, s, "pyvenv.cfg")) and not os.path.isdir(os.path.join(d, s, "conda-meta"))]
            files += [os.path.join(d, n) for n in names if n.endswith(".py")]
    used = {}
    for f in files[:3000]:
        for top, line in imports(f)[0].items():
            used.setdefault(top, f"{os.path.relpath(f, project)}:{line}")
    roots = {project} | {os.path.dirname(f) for f in files}
    third = {t: w for t, w in used.items() if t not in sys.stdlib_module_names and not local(t, roots)}
    return files, third


def present(top):
    try:
        if importlib.util.find_spec(top) is None:
            return None
    except (ImportError, ValueError):
        return None
    dist = (importlib.metadata.packages_distributions().get(top) or [None])[0]
    try:
        return f"{dist} {importlib.metadata.version(dist)}" if dist else "(no version)"
    except importlib.metadata.PackageNotFoundError:
        return "(no version)"


def environment():
    cfg = os.path.join(sys.prefix, "pyvenv.cfg")
    kind = "conda env" if os.path.isdir(os.path.join(sys.prefix, "conda-meta")) else "venv" if sys.prefix != sys.base_prefix else "NOT a virtual environment"
    by_uv = os.path.exists(cfg) and "uv = " in open(cfg, encoding="utf-8", errors="replace").read()
    pip, uv = importlib.util.find_spec("pip") is not None, shutil.which("uv")
    if by_uv and uv or not pip and uv:
        how = f'uv pip install --python "{sys.executable}"'
    elif pip:
        how = f'"{sys.executable}" -m pip install'
    else:
        how = None
    return kind + (" made by uv" if by_uv else ""), how


def report(project, entries):
    files, third = scan(project, entries)
    kind, how = environment()
    print(f"interpreter  {sys.executable}\n             Python {sys.version.split()[0]}, {kind}")
    try:
        import torch
        print(f"torch        {torch.__version__}, " + ("CUDA available" if torch.cuda.is_available() else "CPU only"))
    except ImportError:
        print("torch        not installed in this interpreter")
    found = [n for n in DECLARED if os.path.exists(os.path.join(project, n))]
    print("declared in  " + (", ".join(found) or "no requirements / pyproject / environment file at the project root"))
    print(f"scanned      {len(files)} file(s)" + (" from the entry files" if entries else " under the project"))
    have = {t: present(t) for t in sorted(third, key=str.lower)}
    for t, v in have.items():
        print(f"  {'ok     ' if v else 'MISSING'} {t:24} {v or 'first used at ' + third[t]}")
    missing = [PIP.get(t, t) for t, v in have.items() if not v]
    if not missing:
        print("nothing is missing.")
    elif how:
        print(f"to install the missing ones here:\n  {how} {' '.join(missing)}")
        print("(names are guessed from the imports: check them against the project's own requirement files first)")
    else:
        print("missing: " + " ".join(missing) + "\nthis interpreter has no pip and uv is not on PATH: install one of them first.")
    if kind.startswith("NOT"):
        print("warning: this is a system-wide Python. Do not install into it; use the project's environment or make one.")
    return 1 if missing else 0


def selfcheck():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        os.mkdir(os.path.join(d, "pkg"))
        open(os.path.join(d, "pkg", "__init__.py"), "w").close()
        open(os.path.join(d, "helper.py"), "w").write("import json\nimport surely_not_installed_xyz\n")
        open(os.path.join(d, "unused.py"), "w").write("import another_missing_one\n")
        open(os.path.join(d, "main.py"), "w").write("import os, helper\nfrom pkg import thing\nimport PIL.Image\nfrom . import sibling\n")
        files, third = scan(d, [os.path.join(d, "main.py")])
        assert set(third) == {"PIL", "surely_not_installed_xyz"}, third  # stdlib and local modules left out, unused.py never read
        assert set(scan(d, [])[1]) == {"PIL", "surely_not_installed_xyz", "another_missing_one"}
        assert present("surely_not_installed_xyz") is None and present("json")
    print("deps ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selfcheck"]:
        selfcheck()
    elif len(sys.argv) < 2:
        raise SystemExit(__doc__)
    else:
        sys.exit(report(sys.argv[1], sys.argv[2:]))
