"""Content identities, immutable outputs and bounded external processes."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import tempfile

from . import PROCESSING_VERSION


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def rows(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(s, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))) for s in f if s.strip()]


def save(path, value, jsonl=False):
    """Idempotent, exclusive publication: never replace an older different result."""
    path = Path(path)
    data = ("".join(canonical(x) + "\n" for x in value) if jsonl else canonical(value) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == data:
            return
        raise FileExistsError(f"Refusing to overwrite {path}; use a new output directory")
    fd,temporary=tempfile.mkstemp(prefix='.publish-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(data);f.flush();os.fsync(f.fileno())
        try:
            os.link(temporary,path)
        except FileExistsError:
            if path.read_bytes()!=data:raise
    finally:
        Path(temporary).unlink(missing_ok=True)


def versions():
    result = {"processing_version": PROCESSING_VERSION,
              "implementation_sha256":digest({p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))})}
    for name in ("numpy", "scipy", "ase", "pymatgen", "matplotlib", "fairchem-core", "torch"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def run_process(command, cwd, timeout, stem="process", env=None):
    """Kill the entire POSIX process group on timeout; retain raw logs and command."""
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    log = {"command": list(map(str, command)), "timeout_s": timeout, "returncode": None}
    stdout_path, stderr_path = cwd / (stem + ".stdout"), cwd / (stem + ".stderr")
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            proc = subprocess.Popen(log["command"], cwd=cwd, stdout=stdout, stderr=stderr,
                                    env=env, start_new_session=(os.name == "posix"))
            try:
                proc.wait(timeout=timeout)
                log["status"] = "ok" if proc.returncode == 0 else "nonzero_exit"
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
                proc.wait()
                log["status"] = "timeout"
            log["returncode"] = proc.returncode
    except OSError as exc:
        log.update(status="unavailable", error=str(exc))
    log["elapsed_s"] = time.monotonic() - start
    log.update(stdout=str(stdout_path), stderr=str(stderr_path))
    save(cwd / (stem + ".json"), log)
    return log


def finite(value):
    return value is not None and math.isfinite(float(value))


def unique_index(records, key):
    result = {}
    for row in records:
        identity = row[key]
        if identity in result:
            raise ValueError(f"Duplicate {key}: {identity}; do not silently merge distributed outputs")
        result[identity] = row
    return result


def condition_groups(candidates):
    """Disallow mixed runs, datasets, processing and chemical composition."""
    unique_index(candidates, "candidate_id")
    groups = {}
    for c in candidates:
        groups.setdefault(c["condition_id"], []).append(c)
    for group in groups.values():
        for key in ("dataset_version", "split", "lmdb_key", "run_id", "model_version", "checkpoint_id",
                    "sampling_config_hash", "processing_version", "expected_elements", "representation", "state", "expected_k", "chemistry"):
            if len({canonical(c[key]) for c in group}) != 1:
                raise ValueError(f"Incompatible {key} within condition")
        indices = sorted(c["candidate_index"] for c in group)
        if group[0]['expected_k']!=len(group) or indices != list(range(len(group))):
            raise ValueError("Candidate plan must include every index 0..Kmax-1, including failures")
    return groups
