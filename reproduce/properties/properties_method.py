"""CPU adapter for the pore-property calculation."""
from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

ENGINE_SHA256 = "44bac9522ba8b6abb6cb215432b1003211f5b6677c52048aeaf783dee9e317be"
PROPERTY_CONFIG_SHA256 = "3525c53bfe0565875b6272fc88d181deec982d8c626cf69c6731eef9a8a3a1a2"
BINARY_SHA256 = "05aeab4885d3c3e38154aa3e148a342bc20672b0a5b8eb493378e5b9d439aeca"
REQUIRED_VERSIONS = {"ase": "3.25.0", "numpy": "1.26.4", "scipy": "1.17.1", "pymatgen": "2026.5.4"}
PROPERTIES = ("VSA", "GSA", "VF", "PLD", "LCD", "density")
UNITS = dict(VSA="m2/cm3", GSA="m2/g", VF="1", PLD="angstrom", LCD="angstrom", density="g/cm3", AV="angstrom3", UCV="angstrom3")
_IMPORT_LOCK = threading.RLock()
_PROCESS_LOCK = threading.RLock()
_CANCELLED = threading.Event()
_ACTIVE: set[subprocess.Popen] = set()
_ENGINES: dict[str, SimpleNamespace] = {}


class CalculationCancelled(BaseException):
    """An interrupted computation is not a cacheable scientific failure."""


def _check_cancelled():
    if _CANCELLED.is_set():
        raise CalculationCancelled("Figure 5 property calculation cancelled")


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def _publish(path, value):
    """Standalone CLI publication; never overwrite a different result."""
    path = Path(path)
    payload = (_canonical(value) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Refusing to replace {path}")
        return
    with path.open("xb") as stream:
        stream.write(payload)


def load_method(config_path: Path) -> dict:
    config_path = Path(config_path).resolve(strict=True)
    result = _json(config_path)
    if result.get("schema_version") != 1:
        raise ValueError("Unsupported Figure 5 method configuration")
    for key in ("engine_root", "binary"):
        value = Path(result[key]).expanduser()
        result[key] = str((config_path.parent / value).resolve() if not value.is_absolute() else value.resolve())
    result["configuration_file_sha256"] = _sha(config_path)
    return result


def _source_config(config):
    source = Path(config["engine_root"]) / "configs/downstream/paper_compatible_full.json"
    if _sha(source) != PROPERTY_CONFIG_SHA256:
        raise ValueError("Property configuration changed")
    result = _json(source)
    if result["binary_sha256"] != BINARY_SHA256:
        raise ValueError("Zeo++ identity mismatch")
    result["binary"] = str(Path(config["binary"]).resolve())
    return result


@functools.lru_cache(maxsize=1)
def _versions():
    values = {}
    for name in ("ase", "numpy", "scipy", "pymatgen", "torch", "matplotlib", "fairchem-core"):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def preflight(config: dict) -> dict:
    """Reuse successful checks only while immutable input stat identities agree."""
    root = Path(config["engine_root"])
    paths = sorted((root / "src/flexmof/downstream").glob("*.py"))
    paths += [root / "configs/downstream/paper_compatible_full.json", Path(config["binary"]), Path(__file__)]
    signature = []
    for path in paths:
        stat = path.stat()
        signature.append((str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    # Round-trip to prevent a caller mutating the cached identity dictionary.
    return json.loads(_preflight_cached(_canonical(config), tuple(signature)))


@functools.lru_cache(maxsize=8)
def _preflight_cached(config_json, stat_signature):
    """Hash frozen files after any path/inode/size/mtime/ctime change."""
    config = json.loads(config_json)
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported Figure 5 method configuration")
    directory = Path(config["engine_root"]) / "src/flexmof/downstream"
    sources = {p.name: _sha(p) for p in sorted(directory.glob("*.py"))}
    if _digest(sources) != ENGINE_SHA256:
        raise ValueError("Property engine source hash mismatch")
    effective = _source_config(config)
    binary = Path(config["binary"])
    if not binary.is_file() or _sha(binary) != BINARY_SHA256:
        raise ValueError("Zeo++ binary missing or SHA256 mismatch")
    if not os.access(binary, os.X_OK):
        raise ValueError("Zeo++ binary is not executable")
    versions = dict(_versions())
    for name, required in REQUIRED_VERSIONS.items():
        if versions[name] != required:
            raise ValueError(f"Figure 5 requires {name} {required}, found {versions[name]}")
    identity = dict(schema_version=1, method="ICLR Figure 5 fixed-candidate properties",
                    adapter_sha256=_sha(__file__), engine_sha256=ENGINE_SHA256,
                    method_config_sha256=PROPERTY_CONFIG_SHA256,
                    effective_config_sha256=_digest(effective), binary_sha256=BINARY_SHA256,
                    versions=versions, configuration=_digest(config),
                    selection="fixed candidate 0; no matching, quality, or success selection",
                    failure_policy="per-property null and explicit status; genuine zeros retained")
    identity["identity_sha256"] = _digest(identity)
    return _canonical(identity)


def cancel_active() -> None:
    """Cancel only Zeo++ process groups created and tracked by this adapter."""
    _CANCELLED.set()
    with _PROCESS_LOCK:
        for proc in list(_ACTIVE):
            if proc.poll() is not None:
                continue
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            except ProcessLookupError:
                pass


def _run_process(command, cwd, timeout, stem="process", env=None):
    """Run the property tool with an owned-process cancellation hook."""
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    record = {"command": list(map(str, command)), "timeout_s": timeout, "returncode": None}
    stdout_path, stderr_path = cwd / (stem + ".stdout"), cwd / (stem + ".stderr")
    proc = None
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            with _PROCESS_LOCK:
                if _CANCELLED.is_set():
                    record["status"] = "cancelled"
                else:
                    child_env = dict(os.environ if env is None else env)
                    child_env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
                    proc = subprocess.Popen(record["command"], cwd=cwd, stdout=stdout, stderr=stderr,
                                            env=child_env, start_new_session=(os.name == "posix"))
                    _ACTIVE.add(proc)
            if proc is not None:
                try:
                    proc.wait(timeout=timeout)
                    record["status"] = "cancelled" if _CANCELLED.is_set() else "ok" if proc.returncode == 0 else "nonzero_exit"
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                    proc.wait()
                    record["status"] = "timeout"
                record["returncode"] = proc.returncode
    except OSError as exc:
        record.update(status="unavailable", error=str(exc))
    finally:
        if proc is not None:
            with _PROCESS_LOCK:
                _ACTIVE.discard(proc)
    record.update(elapsed_s=time.monotonic() - started, stdout=str(stdout_path), stderr=str(stderr_path))
    _publish(cwd / (stem + ".json"), record)
    # The archived engine catches Exception and caches scientific failures.
    # Bypass those handlers on cancellation, retaining raw logs but no result.
    if record["status"] == "cancelled":
        raise CalculationCancelled("Figure 5 property subprocess cancelled")
    _check_cancelled()
    return record


def _engine(config):
    """Import only the archived CPU downstream subpackage, never model code."""
    directory = (Path(config["engine_root"]) / "src/flexmof/downstream").resolve()
    key = str(directory)
    with _IMPORT_LOCK:
        if key not in _ENGINES:
            name = "_flexmof_figure5_" + hashlib.sha256(key.encode()).hexdigest()[:16]
            spec = importlib.util.spec_from_file_location(name, directory / "__init__.py", submodule_search_locations=[str(directory)])
            package = importlib.util.module_from_spec(spec)
            sys.modules[name] = package
            spec.loader.exec_module(package)
            props = importlib.import_module(name + ".properties")
            geometry = importlib.import_module(name + ".geometry")
            common = importlib.import_module(name + ".common")
            props.versions = functools.lru_cache(maxsize=1)(props.versions)
            props.run_process = _run_process
            _ENGINES[key] = SimpleNamespace(properties=props, geometry=geometry, common=common)
        return _ENGINES[key]


def _audit_normalized(source_cif, result, expected, engine):
    raw = result.get("work_dir")
    normalized_path = Path(raw) / "normalized.cif" if raw else None
    if normalized_path is None or not normalized_path.is_file():
        return {"status": "not_available", "reason": "No normalized Zeo++ input was produced"}
    normalized_sha256 = _sha(normalized_path)
    if result.get("normalized_sha256") != normalized_sha256:
        raise ValueError("normalized_cif_hash_changed")
    import numpy as np
    original, _ = engine.geometry.read_cif(source_cif, expected)
    restored, _ = engine.geometry.read_cif(normalized_path, expected)
    original = engine.geometry.right_handed(original)
    restored = engine.geometry.right_handed(restored)
    if not np.array_equal(original.numbers, restored.numbers):
        raise ValueError("normalized_atom_identity_or_order_changed")
    ga = original.cell.array @ original.cell.array.T
    gb = restored.cell.array @ restored.cell.array.T
    delta = original.get_scaled_positions(wrap=False) - restored.get_scaled_positions(wrap=False)
    delta -= np.rint(delta)
    residual = float(np.max(np.linalg.norm(delta @ original.cell.array, axis=-1)))
    if not np.allclose(ga, gb, atol=1e-7, rtol=1e-9):
        raise ValueError("normalized_lattice_metric_changed")
    if residual > 1e-6:
        raise ValueError("normalized_fractional_geometry_changed")
    return dict(status="ok", metric_max_abs_A2=float(np.max(np.abs(ga - gb))),
                max_periodic_coordinate_residual_A=residual, normalized_sha256=normalized_sha256)


def _failure(reason, input_sha256):
    return dict(status="failed", error=reason, input_sha256=input_sha256,
                values={p: None for p in UNITS}, property_status={p: reason for p in UNITS},
                units=UNITS, commands=[])


def calculate(sampled_cif: Path, original_cif: Path, output_dir: Path, config: dict) -> dict:
    """Calculate both fixed CIFs; scientific failures are returned, never resampled."""
    _check_cancelled()
    identity = preflight(config)
    sampled_cif, original_cif = Path(sampled_cif).resolve(strict=True), Path(original_cif).resolve(strict=True)
    paths = {"candidate": sampled_cif, "reference": original_cif}
    before = {name: _sha(path) for name, path in paths.items()}
    engine = _engine(config)
    method = _source_config(config)
    output_dir = Path(output_dir).resolve()
    results, audits = {}, {}
    try:
        reference, _ = engine.geometry.read_cif(original_cif)
        expected = engine.geometry.elements(reference)
    except Exception as exc:
        reason = f"reference_geometry_invalid: {type(exc).__name__}: {exc}"
        results = {name: _failure(reason, before[name]) for name in paths}
        audits = {name: {"status": "not_available", "reason": reason} for name in paths}
    else:
        for name, path in paths.items():
            _check_cancelled()
            row = dict(candidate_id=_digest([name, before[name]]), condition_id=before["reference"],
                       representation=method["representation"], chemistry={"status": "unverified"},
                       cif_path=str(path), cif_sha256=before[name], expected_elements=expected, status="present")
            result = engine.properties.properties_one(row, method, output_dir / name)
            try:
                audit = _audit_normalized(path, result, expected, engine)
                if result["status"] == "ok" and audit["status"] != "ok":
                    raise ValueError("successful_properties_without_normalized_geometry_audit")
            except Exception as exc:
                reason = f"normalization_audit_failed: {type(exc).__name__}: {exc}"
                result = dict(result, status="failed", error=reason,
                              values={p: None for p in UNITS}, property_status={p: reason for p in UNITS})
                audit = {"status": "failed", "reason": reason}
            results[name], audits[name] = result, audit
    if any(_sha(path) != before[name] for name, path in paths.items()):
        raise ValueError("input_cif_changed_during_property_calculation")
    absolute_errors = {}
    for prop in PROPERTIES:
        a, b = results["candidate"], results["reference"]
        av, bv = a["values"].get(prop), b["values"].get(prop)
        valid = (a["property_status"].get(prop) == b["property_status"].get(prop) == "ok"
                 and av is not None and bv is not None and math.isfinite(av) and math.isfinite(bv))
        absolute_errors[prop] = dict(value=abs(av - bv) if valid else None,
                                     status="ok" if valid else "unavailable",
                                     candidate_status=a["property_status"].get(prop),
                                     reference_status=b["property_status"].get(prop), unit=UNITS[prop])
    good = sum(row["status"] == "ok" for row in absolute_errors.values())
    # Also cover cancellation after the last Zeo command / cached result.
    _check_cancelled()
    return dict(schema_version=1, status="ok" if good == len(PROPERTIES) else "partial_failure" if good else "failed",
                **results, absolute_errors=absolute_errors, geometry_audits=audits,
                method_identity=identity, input_sha256={"sampled": before["candidate"], "original": before["reference"]},
                selection="fixed candidate 0; no structure-match filtering or candidate replacement")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init-config")
    init.add_argument("--engine-root", type=Path, required=True)
    init.add_argument("--binary", type=Path, required=True)
    init.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("preflight")
    check.add_argument("--config", type=Path, required=True)
    run = commands.add_parser("calculate")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--sampled-cif", type=Path, required=True)
    run.add_argument("--original-cif", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "init-config":
        value = dict(schema_version=1, engine_root=str(args.engine_root.resolve()), binary=str(args.binary.resolve()))
        preflight(value)
        _publish(args.output, value)
        print(_canonical({"status": "configured", "path": str(args.output.resolve())}))
    elif args.command == "preflight":
        print(_canonical(preflight(load_method(args.config))))
    else:
        result = calculate(args.sampled_cif, args.original_cif, args.output_dir, load_method(args.config))
        _publish(args.output_dir / "properties.json", result)
        print(_canonical({"status": result["status"], "output": str((args.output_dir / "properties.json").resolve())}))


if __name__ == "__main__":
    main()
