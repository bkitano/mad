# /// script
# requires-python = ">=3.10"
# dependencies = ["modal>=0.64.0"]
# ///
"""
Run a local Python script on a cloud GPU via Modal.

Examples:
    uv run gpu.py task.py
    uv run gpu.py train.py --gpu A10G --gpus 1 --cpu 4 --memory 8192 --timeout 1800
    uv run gpu.py train.py --pip torch --pip transformers -- --batch-size 32

Anything after `--` is forwarded to the script as argv.

If the target script has PEP 723 inline metadata, dependencies are read
from it automatically:

    # /// script
    # requires-python = ">=3.11"
    # dependencies = ["torch", "numpy"]
    # ///

Prereq: `modal setup` once locally to auth.
"""

import argparse
import re
import sys
from pathlib import Path

import modal

PEP723_RE = re.compile(
    r"(?m)^# /// script\s*$\n(?P<body>(?:^#(?:[^\n]*)\n)+)^# ///\s*$"
)


def parse_pep723(source: str) -> dict:
    m = PEP723_RE.search(source)
    if not m:
        return {}
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore
    body = "\n".join(
        line[2:] if line.startswith("# ") else line[1:]
        for line in m.group("body").splitlines()
    )
    try:
        return tomllib.loads(body)
    except Exception:
        return {}


def normalize_python(spec: str | None) -> str:
    if not spec:
        return "3.11"
    cleaned = spec.strip().lstrip("><=~ ").strip()
    return cleaned or "3.11"


def main() -> int:
    p = argparse.ArgumentParser(
        prog="gpu.py",
        description="Run a local Python script on a cloud GPU via Modal.",
    )
    p.add_argument("script", help="Path to the Python script to run")
    p.add_argument("--gpu", default="T4", help="GPU type: T4, L4, A10G, A100, A100-80GB, H100 (default: T4)")
    p.add_argument("--gpus", type=int, default=1, help="Number of GPUs (default: 1)")
    p.add_argument("--cpu", type=float, default=2.0, help="CPU cores (default: 2)")
    p.add_argument("--memory", type=int, default=4096, help="Memory in MiB (default: 4096)")
    p.add_argument("--timeout", type=int, default=600, help="Timeout in seconds (default: 600)")
    p.add_argument("--pip", action="append", default=[], help="Extra pip packages (repeatable)")
    p.add_argument("--mount", action="append", default=[], help="Extra local dirs/files to include (repeatable)")
    p.add_argument("--no-mount-cwd", action="store_true", help="Don't mount the script's parent directory")
    p.add_argument("--python", default=None, help="Python version, e.g. 3.11")
    p.add_argument("--name", default="gpu-run", help="Modal app name")
    p.add_argument("script_args", nargs=argparse.REMAINDER, help="Args forwarded to the script (after --)")
    args = p.parse_args()

    script_path = Path(args.script).resolve()
    if not script_path.exists():
        print(f"error: script not found: {script_path}", file=sys.stderr)
        return 2

    source = script_path.read_text()
    meta = parse_pep723(source)
    deps = list(meta.get("dependencies", [])) + list(args.pip)
    python_version = normalize_python(args.python or meta.get("requires-python"))

    image = modal.Image.debian_slim(python_version=python_version)
    if deps:
        image = image.pip_install(*deps)

    mount_targets: list[Path] = []
    if not args.no_mount_cwd:
        mount_targets.append(script_path.parent)
    for m in args.mount:
        mount_targets.append(Path(m).resolve())

    workdir = "/work"
    for mp in mount_targets:
        if not mp.exists():
            print(f"error: mount path not found: {mp}", file=sys.stderr)
            return 2
        if mp.is_dir():
            image = image.add_local_dir(str(mp), f"{workdir}/{mp.name}", copy=False)
        else:
            image = image.add_local_file(str(mp), f"{workdir}/{mp.name}", copy=False)

    primary_workdir = (
        f"{workdir}/{script_path.parent.name}" if not args.no_mount_cwd else workdir
    )

    gpu_spec = f"{args.gpu}:{args.gpus}" if args.gpus > 1 else args.gpu
    app = modal.App(args.name)

    @app.function(
        image=image,
        gpu=gpu_spec,
        cpu=args.cpu,
        memory=args.memory,
        timeout=args.timeout,
    )
    def _run(src: str, script_name: str, argv: list[str], cwd: str) -> None:
        import os
        import sys as _sys

        os.chdir(cwd)
        _sys.argv = [script_name, *argv]
        if cwd not in _sys.path:
            _sys.path.insert(0, cwd)
        ns = {"__name__": "__main__", "__file__": os.path.join(cwd, script_name)}
        exec(compile(src, script_name, "exec"), ns)

    extra = args.script_args
    if extra[:1] == ["--"]:
        extra = extra[1:]

    with app.run():
        _run.remote(source, script_path.name, extra, primary_workdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
