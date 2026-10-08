"""python check.py page.html [--shots DIR] [--levels 1,3] — read a built page back in a headless browser.

For desktop width (1280) and phone width (375) it prints what the page's own flowCheckAll() found at every level:
an arrow through a box, hugging one, lying on an unrelated arrow, or missing its target; and any script error.
With --shots DIR it also saves one screenshot per level (desktop width) to look at: DIR/<page>-L<level>.png.

Needs a Chromium-based browser (Chrome, Edge, Chromium, Brave). Set MODEL_FLOW_BROWSER to its path if none is found.
Exit code: 0 clean, 1 the page has problems, 2 the check could not run.
"""
import argparse
import base64
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time


def browser():
    found = [os.environ.get("MODEL_FLOW_BROWSER")]
    for root in (os.environ.get(k) for k in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")):
        if root:
            found += [os.path.join(root, *p) for p in (("Google", "Chrome", "Application", "chrome.exe"), ("Microsoft", "Edge", "Application", "msedge.exe"),
                                                       ("Chromium", "Application", "chrome.exe"), ("BraveSoftware", "Brave-Browser", "Application", "brave.exe"))]
    found += [f"/Applications/{a}.app/Contents/MacOS/{a}" for a in ("Google Chrome", "Microsoft Edge", "Chromium", "Brave Browser")]
    found += [shutil.which(n) for n in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge", "msedge", "chrome", "brave-browser")]
    return next((p for p in found if p and os.path.exists(p)), None)


def visit(exe, url, width, height, shot):
    """Open the page headless, take a screenshot, return what the page wrote to the console.
    The screenshot doubles as the signal that the browser is done: on Windows the program that was started can return
    at once and leave the work to a browser it hands off to, so its exit says nothing and its stdout is lost."""
    if os.path.exists(shot):
        os.remove(shot)
    text = ""
    for attempt in range(3):  # a start now and then comes to nothing, mostly right after another browser has closed
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:  # its own profile: never the browser the user has open
            log = os.path.join(profile, "page.log")
            subprocess.run([exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run", "--no-default-browser-check", f"--user-data-dir={profile}",
                            f"--window-size={width},{height}", "--virtual-time-budget=15000", "--enable-logging", f"--log-file={log}", "--log-level=0",
                            f"--screenshot={shot}", url], capture_output=True, timeout=180)
            for _ in range(50):
                if os.path.exists(shot) and os.path.getsize(shot):
                    break
                time.sleep(0.5)
            time.sleep(0.3)
            text = "".join(open(p, encoding="utf-8", errors="replace").read() for p in (log, os.path.join(profile, "chrome_debug.log")) if os.path.exists(p))
        if os.path.exists(shot):
            break
        time.sleep(2)
    return text


def said(text, tag):
    got = re.search(tag + r" ([A-Za-z0-9+/=]+)", text)
    return base64.b64decode(got.group(1)).decode("utf-8") if got else None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("page")
    ap.add_argument("--shots", metavar="DIR", help="save a screenshot of every level here")
    ap.add_argument("--levels", help="only these levels for --shots, e.g. 1,3")
    a = ap.parse_args()
    exe, page = browser(), pathlib.Path(a.page).resolve()
    if not exe:
        print("no Chromium-based browser found (Chrome, Edge, Chromium, Brave). Install one, or set MODEL_FLOW_BROWSER to its path.")
        return 2
    if not page.exists():
        print(f"no such page: {page}")
        return 2
    url, bad, desktop = page.as_uri(), 0, None
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for width, name in ((1280, "desktop"), (375, "phone")):
            text = visit(exe, url + "#check", width, 900, os.path.join(tmp, "probe.png"))
            err, got = said(text, "MODEL_FLOW_ERROR"), said(text, "MODEL_FLOW_CHECK")
            if err:
                print(f"{name} {width}px: the page did not draw: {err}")
                return 1
            if not got:
                print(f"{name} {width}px: nothing came back from {os.path.basename(exe)}. If this shell is sandboxed the browser may not be allowed to start: "
                      "run this outside the sandbox, or open the page with a browser tool and evaluate flowCheckAll() there.")
                return 2
            got = json.loads(got)
            desktop = desktop or got
            n = sum(map(len, got["problems"].values()))
            bad += n
            print(f"{name} {width}px: " + ("clean" if not n else f"{n} problem(s)") + "  |  " + ", ".join(f"L{v['level']} {v['name']}: {v['steps']} steps, {v['height']}px" for v in got["levels"]))
            for level, items in got["problems"].items():
                for item in items:
                    print(f"  [{level}] {item}")
    if a.shots:
        os.makedirs(a.shots, exist_ok=True)
        want = {int(v) for v in a.levels.split(",")} if a.levels else None
        for v in desktop["levels"]:
            if want is None or v["level"] in want:
                out = os.path.join(os.path.abspath(a.shots), f"{page.stem}-L{v['level']}.png")
                # ponytail: one tall window, capped at 6000px; a taller level is cut off at the bottom
                visit(exe, f"{url}#level-{v['level']}", 1280, min(v["height"] + 20, 6000), out)
                print(f"shot {out}" if os.path.exists(out) else f"no screenshot for level {v['level']}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
