"""Zeo++ subprocess interface. Settings inspired by AtomMOF; independently implemented.

Attribution and locked source versions: docs/downstream/protocol.md.
No parse failure is ever replaced with zero.
"""
from __future__ import annotations

import math
from pathlib import Path
import re

from ase.io import write

from .common import digest, file_hash, load, run_process, save, versions
from .geometry import read_cif, right_handed, geometry_hash

PROPERTIES = ("VSA", "GSA", "VF", "PLD", "LCD", "density")
UNITS = dict(VSA="m2/cm3", GSA="m2/g", VF="1", PLD="angstrom", LCD="angstrom",
             density="g/cm3", AV="angstrom3", UCV="angstrom3")
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?"


def number(text):
    value = float(text.replace("D", "E").replace("d", "e"))
    if not math.isfinite(value):
        raise ValueError("nonfinite_output")
    return value


def field(text, label):
    match = re.search(r"(?:^|\s)" + re.escape(label) + r"\s*:\s*(" + NUMBER + r")(?=\s|$)", text)
    if not match:
        raise ValueError(f"missing_field:{label}")
    return number(match.group(1))


def parse_res(text):
    line = next((x for x in text.splitlines() if x.strip()), "")
    parts = line.split()
    if len(parts) != 4:
        raise ValueError("res_requires_filename_and_three_diameters")
    lcd, pld, path_lcd = map(number, parts[1:])
    if min(lcd, pld, path_lcd) < 0 or pld > lcd + 1e-4:
        raise ValueError("inconsistent_pore_diameters")
    return dict(LCD=lcd, PLD=pld, path_LCD=path_lcd)


def parse_sa(text):
    return dict(VSA=field(text, "ASA_m^2/cm^3"), GSA=field(text, "ASA_m^2/g"),
                zeo_density_sa=field(text, "Density"), zeo_volume_sa=field(text, "Unitcell_volume"))


def parse_vol(text):
    return dict(AV=field(text, "AV_A^3"), VF=field(text, "AV_Volume_fraction"),
                zeo_density_vol=field(text, "Density"), zeo_volume_vol=field(text, "Unitcell_volume"))


def check_units(values, tolerance=0.005):
    for key in PROPERTIES + ("AV", "UCV"):
        if values.get(key) is not None and (not math.isfinite(values[key]) or values[key] < 0):
            raise ValueError(f"invalid_range:{key}")
    if values.get("VF") is not None and not 0 <= values["VF"] <= 1:
        raise ValueError("VF_out_of_range")
    pairs = [("VSA", values.get("GSA", 0) * values["density"]) ] if values.get("GSA") is not None else []
    pairs += [("AV", values["VF"] * values["UCV"])] if values.get("VF") is not None else []
    for key in ("sa", "vol"):
        pairs += [("zeo_density_" + key, values["density"]), ("zeo_volume_" + key, values["UCV"])]
    for key, expected in pairs:
        observed = values.get(key)
        if observed is not None and not math.isclose(observed, expected, rel_tol=tolerance, abs_tol=1e-5):
            raise ValueError(f"unit_consistency_failed:{key}:{observed}:{expected}")


def properties_one(row, config, work, repeat=0):
    identity = row.get("candidate_id", row.get("reference_id"))
    binary = Path(config["binary"])
    config_id = digest([config,versions()])
    binary_actual_hash=file_hash(binary) if binary.is_file() else None
    key = digest([row, config, versions(), binary_actual_hash, repeat])
    job = Path(work) / key
    if (job / "result.json").exists():
        cached = load(job / "result.json")
        # Do not return stale results if external binaries or source files changed.
        if row["cif_sha256"] and file_hash(row["cif_path"]) != row["cif_sha256"]:
            raise ValueError("cif_hash_changed")
        if binary.exists() and file_hash(binary) != config["binary_sha256"]:
            raise ValueError("zeo_binary_changed")
        return cached
    # An interrupted work directory is retained; the next attempt gets its own directory.
    job.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while (job / f"attempt_{attempt}").exists():
        attempt += 1
    raw = job / f"attempt_{attempt}"
    raw.mkdir()
    values = {p: None for p in UNITS}
    states = {p: "pending" for p in UNITS}
    result = dict(record_id=identity, condition_id=row["condition_id"], config_id=config_id,
                  representation=row["representation"], chemistry_status=row["chemistry"]["status"],
                  input_sha256=row["cif_sha256"], repeat=repeat, units=UNITS, values=values,
                  property_status=states, status="pending", work_dir=str(raw.resolve()), runtime=versions(), commands=[])
    try:
        if row["status"] != "present":
            raise ValueError(row["status"])
        if row['representation']!=config['representation']:
            raise ValueError('property_representation_mismatch')
        if file_hash(row["cif_path"]) != row["cif_sha256"]:
            raise ValueError("cif_hash_changed")
        atoms, notes = read_cif(row["cif_path"], row["expected_elements"])
        result.update(warnings=notes, geometry_hash=geometry_hash(atoms), n_atoms=len(atoms))
        values.update(UCV=float(atoms.get_volume()), density=float(atoms.get_masses().sum() * 1.66053906660 / atoms.get_volume()))
        states.update(UCV="ok", density="ok")
        if not binary.is_file():
            raise ValueError("zeo_unavailable")
        if file_hash(binary) != config["binary_sha256"]:
            raise ValueError("zeo_binary_changed")
        if config["radii"]["mode"] != "compiled_default":
            raise ValueError("Only locked compiled_default radii supported in this profile")
        if set(row["expected_elements"]) - set(config["radii"]["supported_elements"]):
            raise ValueError("unverified_radius_element")
        normalized = right_handed(atoms)
        normalized.wrap()
        write(raw / "normalized.cif", normalized, format="cif")
        result["normalized_sha256"] = file_hash(raw / "normalized.cif")
        restored, _ = read_cif(raw / "normalized.cif", row["expected_elements"])
        if len(restored) != len(atoms):
            raise ValueError("normalization_atom_count_changed")
        commands = [
            ("res", [*(['-ha'] if config['res_high_accuracy'] else []), "-res", "result.res"], parse_res, ["LCD", "PLD"]),
            ("sa", [*(['-ha'] if config['surface_high_accuracy'] else []), "-sa", str(config["channel_radius_A"]), str(config["probe_radius_A"]), str(config["sa_samples_per_atom"]), "result.sa"], parse_sa, ["VSA", "GSA"]),
            ("vol", [*(["-ha"] if config["volume_high_accuracy"] else []), "-vol", str(config["channel_radius_A"]), str(config["probe_radius_A"]), str(config["volume_samples"]), "result.vol"], parse_vol, ["AV", "VF"]),
        ]
        for name, flags, parser, fields in commands:
            log = run_process([str(binary), *flags, "normalized.cif"], raw, config["timeout_s"], name)
            result["commands"].append(log)
            try:
                if log["status"] != "ok":
                    raise ValueError(log["status"])
                combined = Path(log["stdout"]).read_text(errors="replace") + Path(log["stderr"]).read_text(errors="replace")
                if re.search(r"unknown.*(?:radius|radii|atom)|(?:radius|radii).*not found|default.*radius.*used|error:|failed|not found in (?:rad|mass)", combined, re.I):
                    raise ValueError("zeo_reported_error_or_unknown_radius")
                parsed = parser((raw / ("result." + name)).read_text())
                trial = {**values, **parsed}
                check_units(trial, config["unit_relative_tolerance"])
                values.update(parsed)
                states.update({p: "ok" for p in fields})
            except Exception as exc:
                states.update({p: str(exc) for p in fields})
        result["status"] = "ok" if all(states[p] == "ok" for p in UNITS) else "partial_failure"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        for p in states:
            if states[p] == "pending":
                states[p] = str(exc)
    result["elapsed_s"] = sum(x["elapsed_s"] for x in result["commands"])
    save(job / "result.json", result)
    return result


def batch_properties(manifest, config, work, out, repeats=1):
    results = [properties_one(r, config, work, i) for r in manifest for i in range(repeats)]
    save(out, results, jsonl=True)
    return results
